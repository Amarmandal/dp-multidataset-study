"""
Driver: run the LiRA (Carlini et al. 2022) per-example membership inference
attack against the exported Standard vs DP targets. Its headline metric is
**TPR at a low fixed FPR** — the worst-case leakage number that the average-case
Shokri / Yeom attacks already in the study cannot see.

It reuses the Shokri pipeline's target loaders and shadow factory
(``Attack/MIA_Shokri/exported_models.py``), but trains shadows on **real data
splits** rather than synthetic records.

The target is always the **exported** artifact (``std_<fam>_model.pkl`` /
``dp_<fam>_model_eps_<eps>.pkl``) — the same file whose accuracy produced the ACL
number in the utility pipeline. It is never retrained and never overwritten. The
only randomness in a run is the reseeding of the 32 shadow models, so the run-to-run
spread reported in ``*_std`` is purely shadow-calibration noise around one fixed
target.

Outputs (under Attack/LiRA/results/):
  * results/<dataset>/<dataset>_lira_results.csv   configuration summaries
  * results/<dataset>/<dataset>_lira_runs.csv      one row per attack run
  * results/<dataset>/<dataset>_lira_roc.csv.gz    reconstructible ROC curves
  No JSON is written.

Usage:
    python run_lira.py --datasets KIDNEY_STONE
    python run_lira.py --datasets KIDNEY_STONE --n-shadow 64
    python run_lira.py --datasets KIDNEY_STONE --balance   # 1:1 members:non-members
    python run_lira.py --runs 30 --dnn-runs 5         # per-family run counts
    python run_lira.py --runs 30 --skip-dnn            # SVM stays capped at its default 5 runs
    python run_lira.py --datasets DIABETES --models SVM --workers 3
    python run_lira.py --quick                       # fast smoke test
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Reuse the Shokri folder's loaders / shadow factory. The repo root goes on the
# path too: the SVM notebooks import their DP estimator from ``common_svm``, so
# the exported targets pickle as ``common_svm.DifferentiallyPrivateSVM`` and
# pickle.load has to be able to import that module.
SHOKRI_DIR = os.path.join(os.path.dirname(__file__), "..", "MIA_Shokri")
sys.path.insert(0, os.path.abspath(SHOKRI_DIR))
sys.path.insert(0, REPO)
import exported_models as em  # noqa: E402
from lira import run_lira  # noqa: E402

warnings.filterwarnings("ignore")


DATASETS = [
    "GALLSTONE",
    "KIDNEY_STONE",
    "LUNG_CANCER",
    "BCP",
    "CANCER_RISK",
    "DIABETES",
]
DEFAULT_EPSILONS = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0, 10.0]
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_dataset(name):
    path = os.path.join(REPO, name, "data", "processed_data.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    data = {
        "X_train": np.asarray(d["X_train"], dtype=float),
        "X_test": np.asarray(d["X_test"], dtype=float),
        "y_train": np.asarray(d["y_train"]).ravel().astype(int),
        "y_test": np.asarray(d["y_test"]).ravel().astype(int),
    }
    n_classes = int(d.get("n_classes", len(np.unique(data["y_train"]))))
    low, high = d["bounds"]
    return data, n_classes, np.asarray(low, float), np.asarray(high, float)


def target_test_acc(target, view):
    return float((target.predict(view["X_test"]) == view["y_test"]).mean())


# --------------------------------------------------------------------------- #
# Attack one exported target
# --------------------------------------------------------------------------- #
def family_runs(fam, args):
    """Return the configured run count for one model family.

    A LiRA run trains ``--n-shadow`` (32) shadows, so the DNN is orders of
    magnitude more expensive than LR/RF/GNB. SVM is also capped separately
    because RBF scoring is costly on non-separable data.
    """
    if fam == "DNN":
        return args.dnn_runs
    if fam == "SVM" and args.svm_runs is not None:
        return args.svm_runs
    return args.runs


# Aggregated across seeds: bare name carries the mean, ``<name>_std`` the spread.
# The bare names are what prepare_mia_data.py already aliases to ``*_mean``.
AGG_KEYS = [
    "attack_auc",
    "advantage",
    "tpr_at_10pct",
    "tpr_at_1pct",
    "tpr_at_0p1pct",
    "sd_in",
    "sd_out",
    "test_acc",
]
# Constant across seeds; taken from the first successful run.
PASS_KEYS = ["n_members", "n_nonmembers", "n_shadow_trained", "online", "balanced"]


def aggregate(runs, family, variant, epsilon):
    """Mean/std across seeds. Mirrors Attack/MIA_Shokri/run_mia.py:133-154."""
    ok = [r for r in runs if "error" not in r]
    row = {"model": family, "variant": variant, "epsilon": epsilon, "n_runs": len(runs)}
    if not ok:
        row["error"] = runs[0].get("error", "unknown")
        return row
    for k in AGG_KEYS:
        vals = [r[k] for r in ok if k in r]
        row[k] = float(np.mean(vals)) if vals else None
        row[f"{k}_std"] = float(np.std(vals)) if vals else None
    for k in PASS_KEYS:
        row[k] = ok[0].get(k)
    # Carry the first run's curve for callers that inspect aggregate results.
    # Authoritative persisted ROC coordinates are written from the raw runs.
    for k in ("_roc_fpr", "_roc_tpr", "_roc_threshold"):
        if k in ok[0]:
            row[k] = ok[0][k]
    return row


def attack_target(
    fam,
    variant,
    data,
    n_classes,
    low,
    high,
    args,
    ds,
    epsilon=None,
    seeds=(0,),
    include_runs=False,
):
    view, lo, hi = em.family_dataview(fam, data, low, high)
    n_features = view["X_train"].shape[1]

    # The exported artifact is the fixed target: loaded once, before the seed
    # loop, so every run attacks the identical object. It is never retrained and
    # never written back. Only the shadow models are reseeded per run.
    try:
        target = em.load_target(REPO, ds, fam, variant, epsilon, n_classes)
    except Exception as exc:  # unloadable target -> one error row, run continues
        failed = [{
            "dataset": ds,
            "model": fam,
            "variant": variant,
            "epsilon": epsilon,
            "run_seed": None,
            "error": str(exc),
        }]
        row = aggregate(failed, fam, variant, epsilon)
        row["dataset"] = ds
        return (row, failed) if include_runs else row

    runs = []
    for seed in tuple(seeds):
        try:
            make_shadow = em.make_shadow_factory(
                fam,
                variant,
                epsilon=epsilon,
                bounds=(lo, hi),
                n_features=n_features,
                n_classes=n_classes,
                base_seed=seed,
                dataset=ds,
            )
            m = run_lira(
                target,
                make_shadow,
                view,
                n_classes,
                n_shadow=args.n_shadow,
                online=not args.offline,
                seed=seed,
                balance=args.balance,
            )
            if "error" not in m:
                m["test_acc"] = target_test_acc(target, view)
        except Exception as exc:  # a single bad seed shouldn't sink the run
            m = {"error": str(exc)}
        m.update({
            "dataset": ds,
            "model": fam,
            "variant": variant,
            "epsilon": epsilon,
            "run_seed": int(seed),
        })
        runs.append(m)

    row = aggregate(runs, fam, variant, epsilon)
    row["dataset"] = ds
    return (row, runs) if include_runs else row


def _attack_target_worker(ds, fam, variant, epsilon, args, seeds, include_runs=False):
    """Run one independent target configuration in a child process."""
    label = "standard" if variant == "standard" else f"epsilon={epsilon}"
    print(f"  [{fam}] worker started {label}", flush=True)
    data, n_classes, low, high = load_dataset(ds)
    return attack_target(
        fam,
        variant,
        data,
        n_classes,
        low,
        high,
        args,
        ds,
        epsilon=epsilon,
        seeds=seeds,
        include_runs=include_runs,
    )


def _parallel_targets(ds, fam, args, seeds, include_runs=False):
    """Evaluate DP targets and individual standard seeds concurrently."""
    dp_epsilons = [
        eps
        for eps in args.epsilons
        if em.model_exists(REPO, ds, fam, "dp", eps)
    ]
    dp_results = {}
    standard_runs = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {}

        # Submit the cheap DP targets first so their progress is visible while
        # the much slower standard RBF-SVM seeds occupy the pool afterward.
        for epsilon in dp_epsilons:
            future = pool.submit(
                _attack_target_worker,
                ds,
                fam,
                "dp",
                epsilon,
                args,
                seeds,
                include_runs,
            )
            pending[future] = ("dp", epsilon)

        # Each standard LiRA seed is independent. Splitting them here preserves
        # the exact seed-level metrics and aggregates them in seed order below.
        for seed in seeds:
            future = pool.submit(
                _attack_target_worker,
                ds,
                fam,
                "standard",
                None,
                args,
                (seed,),
                include_runs,
            )
            pending[future] = ("standard_seed", seed)

        for future in as_completed(pending):
            kind, value = pending[future]
            result = future.result()
            if include_runs:
                result, raw_runs = result
            else:
                raw_runs = []
            if kind == "dp":
                epsilon = value
                dp_results[epsilon] = result
                if include_runs:
                    dp_results[(epsilon, "runs")] = raw_runs
                _print_row(fam, f"eps={epsilon:<4}", result)
                print(f"  [{fam}] completed epsilon={epsilon}", flush=True)
            else:
                seed = value
                standard_runs[seed] = result
                if include_runs:
                    standard_runs[(seed, "runs")] = raw_runs
                print(
                    f"  [{fam}] completed standard seed={seed} "
                    f"({sum(isinstance(k, int) for k in standard_runs)}/{len(seeds)})",
                    flush=True,
                )

    if include_runs:
        standard_raw = [
            run
            for seed in seeds
            for run in standard_runs[(seed, "runs")]
        ]
        standard = aggregate(standard_raw, fam, "standard", None)
    else:
        standard_raw = []
        standard = aggregate(
            [standard_runs[seed] for seed in seeds], fam, "standard", None
        )
    standard["dataset"] = ds
    _print_row(fam, "STD  ", standard)
    print(f"  [{fam}] completed standard aggregate", flush=True)
    summaries = [standard] + [dp_results[epsilon] for epsilon in dp_epsilons]
    if not include_runs:
        return summaries
    raw = standard_raw + [
        run
        for epsilon in dp_epsilons
        for run in dp_results[(epsilon, "runs")]
    ]
    return summaries, raw


def run_dataset(ds, args):
    data, n_classes, low, high = load_dataset(ds)
    print(
        f"\n{'=' * 72}\n{ds}  (classes={n_classes}, n_train={len(data['y_train'])}, "
        f"n_features={data['X_train'].shape[1]})  [LiRA, {args.n_shadow} shadows]\n{'=' * 72}"
    , flush=True)
    results = []
    run_records = []

    for fam in args.models:
        if fam not in em.MODEL_FAMILIES or not em.model_exists(
            REPO, ds, fam, "standard"
        ):
            print(f"  [{fam}] skipped (no exported model)", flush=True)
            continue

        t0 = time.time()
        # Distinct shadow seeds, starting at --seed for reproducible resumptions.
        seeds = tuple(args.seed + i for i in range(family_runs(fam, args)))
        print(
            f"  [{fam}] {len(seeds)} run(s), shadow seeds {seeds[0]}..{seeds[-1]}",
            flush=True,
        )
        if ds == "DIABETES" and fam == "SVM" and args.workers > 1:
            print(
                f"  [{fam}] processing standard + {len(args.epsilons)} epsilon "
                f"targets with {args.workers} workers",
                flush=True,
            )
            summaries, raw_runs = _parallel_targets(
                ds, fam, args, seeds, include_runs=True
            )
            results.extend(summaries)
            run_records.extend(raw_runs)
        else:
            print(f"  [{fam}] processing standard", flush=True)
            m, raw_runs = attack_target(
                fam,
                "standard",
                data,
                n_classes,
                low,
                high,
                args,
                ds,
                seeds=seeds,
                include_runs=True,
            )
            results.append(m)
            run_records.extend(raw_runs)
            _print_row(fam, "STD  ", m)

            for eps in args.epsilons:
                if not em.model_exists(REPO, ds, fam, "dp", eps):
                    continue
                print(f"  [{fam}] processing epsilon={eps}", flush=True)
                m, raw_runs = attack_target(
                    fam,
                    "dp",
                    data,
                    n_classes,
                    low,
                    high,
                    args,
                    ds,
                    epsilon=eps,
                    seeds=seeds,
                    include_runs=True,
                )
                results.append(m)
                run_records.extend(raw_runs)
                _print_row(fam, f"eps={eps:<4}", m)
        print(f"  [{fam}] done in {time.time() - t0:.1f}s", flush=True)

    write_results(ds, results, run_records)


def _print_row(fam, tag, m):
    if "error" in m:
        print(f"  [{fam}] {tag} ERROR: {m['error']}", flush=True)
    else:
        print(
            f"  [{fam}] {tag} AUC={m['attack_auc']:.3f}  "
            f"TPR@1%={m['tpr_at_1pct']:.3f}  TPR@.1%={m['tpr_at_0p1pct']:.3f}  "
            f"adv={m['advantage']:.3f}",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# Authoritative CSV outputs
# --------------------------------------------------------------------------- #
def _without_roc_arrays(record):
    return {key: value for key, value in record.items() if not key.startswith("_roc")}


def _representative_roc_rows(run_records):
    """Long-form ROC coordinates for the first successful standard run per model.

    This is the exact curve the old attack-side plot used.  Run-level operating
    thresholds and integer outcomes for every repetition live in
    ``*_lira_runs.csv``; only the representative full curves are persisted here
    to keep the reconstructible figure input compact.
    """
    rows = []
    successful = [
        record
        for record in run_records
        if record.get("variant") == "standard"
        and "error" not in record
        and "_roc_fpr" in record
    ]
    models = sorted({record["model"] for record in successful})
    for model in models:
        candidates = sorted(
            (record for record in successful if record["model"] == model),
            key=lambda record: record["run_seed"],
        )
        record = candidates[0]
        fpr = record["_roc_fpr"]
        tpr = record["_roc_tpr"]
        thresholds = record["_roc_threshold"]
        if not (len(fpr) == len(tpr) == len(thresholds)):
            raise ValueError(
                f"ROC array length mismatch for {record['dataset']}/{model}"
            )
        for index, (x, y, threshold) in enumerate(zip(fpr, tpr, thresholds)):
            rows.append({
                "dataset": record["dataset"],
                "model": model,
                "variant": "standard",
                "epsilon": None,
                "run_seed": record["run_seed"],
                "point_index": index,
                "threshold": threshold,
                "fpr": x,
                "tpr": y,
                "curve_selection": "first_successful_standard_run",
            })
    return rows


def write_results(ds, summaries, run_records):
    out = os.path.join(OUT_DIR, ds)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame([_without_roc_arrays(row) for row in summaries]).to_csv(
        os.path.join(out, f"{ds}_lira_results.csv"), index=False
    )
    pd.DataFrame([_without_roc_arrays(row) for row in run_records]).to_csv(
        os.path.join(out, f"{ds}_lira_runs.csv"), index=False
    )
    pd.DataFrame(_representative_roc_rows(run_records)).to_csv(
        os.path.join(out, f"{ds}_lira_roc.csv.gz"),
        index=False,
        compression="gzip",
    )
    print(f"  -> summary, per-run, and ROC CSVs written to {out}/", flush=True)

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="LiRA: Standard vs DP, TPR@low-FPR")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=em.MODEL_FAMILIES)
    ap.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    ap.add_argument("--n-shadow", type=int, default=32, help="shadow models per target")
    ap.add_argument(
        "--offline", action="store_true", help="offline LiRA (OUT-only, no IN models)"
    )
    ap.add_argument(
        "--balance",
        action="store_true",
        help="subsample members to the non-member count (1:1 evaluation set)",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="first shadow-calibration run seed"
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=30,
        help="shadow-model reseeds per target for LR/RF/GNB/SVM (target is fixed)",
    )
    ap.add_argument(
        "--dnn-runs",
        type=int,
        default=5,
        help="shadow-model reseeds per DNN target (32 shadows/run makes it costly)",
    )
    ap.add_argument(
        "--svm-runs",
        type=int,
        default=5,
        help="shadow-model reseeds per SVM target (default: 5; overrides --runs)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel target configurations for DIABETES SVM only (default: 1)",
    )
    ap.add_argument(
        "--skip-dnn",
        action="store_true",
        help="drop the DNN family entirely (5 seeds x 32 shadows is ~8.5h)",
    )
    ap.add_argument("--quick", action="store_true", help="tiny config for smoke test")
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be at least 1")
    if args.runs < 1 or args.dnn_runs < 1 or args.svm_runs < 1:
        ap.error("--runs, --dnn-runs, and --svm-runs must each be at least 1")

    if args.skip_dnn:
        args.models = [m for m in args.models if m != "DNN"]

    if args.quick:
        args.epsilons = [0.1, 1.0]
        args.n_shadow = 8
        args.runs = min(args.runs, 3)
        args.dnn_runs = min(args.dnn_runs, 2)
        args.svm_runs = min(args.svm_runs, 3)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ds in args.datasets:
        run_dataset(ds, args)


if __name__ == "__main__":
    main()
