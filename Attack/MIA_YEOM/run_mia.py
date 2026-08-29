"""
Driver: run Yeom's loss-threshold membership inference attack (Yeom et al. 2018)
against the exported Standard vs DP targets for **every** model family. Its
headline metric is the **membership advantage**
(TPR − FPR) — the average-case leakage number, the companion to the per-example
TPR@low-FPR that ``Attack/LiRA`` reports.

It reuses the Shokri pipeline's target loaders and feature-space views
(``Attack/MIA_Shokri/exported_models.py``), exactly like ``Attack/LiRA/run_lira.py``,
so a single command attacks LR / RF / GNB / SVM / DNN, Standard and DP, across
all six study datasets. The attack itself lives in ``mia.py`` and is unchanged from
the original per-model ``*_mia.py`` scripts — only the way it is *run* is now
modular.

The target is always the **exported** artifact (``std_<fam>_model.pkl`` /
``dp_<fam>_model_eps_<eps>.pkl``) — the very same file whose accuracy produced the
ACL number in the utility pipeline. It is never retrained here. Together with the
fixed 80/20 member / non-member split (``random_state=42``, baked into
``processed_data.pkl``) that makes Yeom **fully deterministic**: one run per
target, every ``*_std`` column 0.0, and an explicit ``deterministic=True`` column
recording that the zero dispersion is intended, not a seeding bug.

Outputs (under Attack/MIA_YEOM/results/):
  * results/<dataset>/<dataset>_mia_results.csv   authoritative per-dataset data
  No JSON is written.

Usage:
    python run_mia.py --datasets KIDNEY_STONE
    python run_mia.py                                 # all datasets, all models
    python run_mia.py --verify-determinism            # run twice, assert identical
    python run_mia.py --quick                         # fast smoke test
"""

import argparse
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Reuse the Shokri folder's loaders / feature-space views (same as LiRA). The
# repo root goes on the path too: the SVM notebooks import their DP estimator
# from ``common_svm``, so exported targets pickle as
# ``common_svm.DifferentiallyPrivateSVM`` and pickle.load must import it.
SHOKRI_DIR = os.path.join(os.path.dirname(__file__), "..", "MIA_Shokri")
sys.path.insert(0, os.path.abspath(SHOKRI_DIR))
sys.path.insert(0, REPO)
import exported_models as em  # noqa: E402
from mia import yeom_mia  # noqa: E402

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
# Aggregated across runs: bare name carries the mean, ``<name>_std`` the spread.
# The bare names are what prepare_mia_data.py already aliases to ``*_mean``.
AGG_KEYS = [
    "advantage",
    "attack_auc",
    "tpr",
    "fpr",
    "test_acc",
    "threshold_tau",
    "loss_gap",
]
# Sample-level statistics from a single attack; averaged across seeds, no std.
PASS_KEYS = ["train_loss_mean", "train_loss_std", "test_loss_mean", "test_loss_std"]
COUNT_KEYS = ["n_members", "n_nonmembers"]


def aggregate(runs, family, variant, epsilon):
    """Mean/std across runs. Mirrors Attack/MIA_Shokri/run_mia.py.

    Yeom is deterministic (see ``attack_target``), so ``runs`` holds exactly one
    entry and every ``<key>_std`` comes out 0.0. The schema is kept identical to
    the stochastic attacks so downstream consumers do not need a special case.
    """
    ok = [r for r in runs if "error" not in r]
    row = {"model": family, "variant": variant, "epsilon": epsilon, "n_runs": len(runs)}
    if not ok:
        row["error"] = runs[0].get("error", "unknown")
        row["deterministic"] = True
        return row
    for k in AGG_KEYS:
        vals = [r[k] for r in ok if k in r]
        row[k] = float(np.mean(vals)) if vals else None
        row[f"{k}_std"] = float(np.std(vals)) if vals else None
    for k in PASS_KEYS:
        vals = [r[k] for r in ok if k in r]
        row[k] = float(np.mean(vals)) if vals else None
    for k in COUNT_KEYS:
        row[k] = ok[0].get(k)
    # Explicit methodological marker: the target is the fixed exported artifact
    # and the member/non-member partition is the fixed 80/20 split, so a repeat
    # call reproduces these numbers exactly. std=0.0 is expected, not a bug.
    row["deterministic"] = True
    return row


def attack_target(fam, variant, data, n_classes, low, high, ds, epsilon=None):
    """Attack the **exported** target once. No retraining, no seeds.

    The target is loaded from ``<DATASET>/<FAM>/output/model/`` — the same
    artifact the utility pipeline measured for ACL. Members are ``X_train`` and
    non-members ``X_test`` from ``processed_data.pkl`` (stratified 80/20,
    random_state=42); neither is resampled here. Yeom's rule (loss <= mean
    training loss) has no randomness of its own, so the whole procedure is
    deterministic and one run is the complete result.
    """
    view, lo, hi = em.family_dataview(fam, data, low, high)

    try:
        target = em.load_target(REPO, ds, fam, variant, epsilon, n_classes)
        m = yeom_mia(target, view)
        m["test_acc"] = target_test_acc(target, view)
    except Exception as exc:
        m = {"error": str(exc)}

    row = aggregate([m], fam, variant, epsilon)
    row["dataset"] = ds
    return row


def run_dataset(ds, args):
    data, n_classes, low, high = load_dataset(ds)
    print(
        f"\n{'=' * 72}\n{ds}  (classes={n_classes}, n_train={len(data['y_train'])}, "
        f"n_features={data['X_train'].shape[1]})  [Yeom MIA]\n{'=' * 72}"
    )
    results = []

    for fam in args.models:
        if fam not in em.MODEL_FAMILIES or not em.model_exists(
            REPO, ds, fam, "standard"
        ):
            print(f"  [{fam}] skipped (no exported model)")
            continue

        t0 = time.time()
        m = attack_target(fam, "standard", data, n_classes, low, high, ds)
        _verify_determinism(args, fam, "standard", data, n_classes, low, high, ds, m)
        results.append(m)
        _print_row(fam, "STD  ", m)

        for eps in args.epsilons:
            if not em.model_exists(REPO, ds, fam, "dp", eps):
                continue
            m = attack_target(
                fam, "dp", data, n_classes, low, high, ds, epsilon=eps
            )
            _verify_determinism(
                args, fam, "dp", data, n_classes, low, high, ds, m, epsilon=eps
            )
            results.append(m)
            _print_row(fam, f"eps={eps:<4}", m)
        print(f"  [{fam}] done in {time.time() - t0:.1f}s")

    df = pd.DataFrame(results)
    write_results(ds, df)


def _verify_determinism(
    args, fam, variant, data, n_classes, low, high, ds, first, epsilon=None
):
    """--verify-determinism: attack the same target twice, assert identical rows.

    This is the repeat check that justifies ``n_runs=1``: if a second, entirely
    independent call reproduces every metric bit-for-bit, the zero std is a
    property of the method, not of a missing seed.
    """
    if not args.verify_determinism:
        return
    second = attack_target(
        fam, variant, data, n_classes, low, high, ds, epsilon=epsilon
    )
    keys = [k for k in AGG_KEYS + PASS_KEYS if k in first]
    deltas = {
        k: abs(float(first[k]) - float(second[k]))
        for k in keys
        if first.get(k) is not None and second.get(k) is not None
    }
    # Tolerance, not exact equality. sklearn's RandomForest averages tree
    # posteriors across threads (n_jobs=-1), so the summation order varies and
    # predict_proba differs at ~1e-16 between two calls on the *same* object.
    # That floating-point noise reaches the mean-loss keys (threshold_tau,
    # loss_gap, *_loss_mean) while every rate metric stays bit-identical. A real
    # seeding leak -- a target actually being retrained -- moves these by 1e-3
    # or more, so 1e-9 separates the two cases with many orders of magnitude to
    # spare.
    tol = 1e-9
    bad = {k: d for k, d in deltas.items() if d > tol}
    tag = "STD" if variant == "standard" else f"eps={epsilon:g}"
    if bad:
        detail = ", ".join(f"{k} Δ={d:.3e}" for k, d in sorted(bad.items()))
        raise SystemExit(
            f"DETERMINISM CHECK FAILED [{ds}/{fam}/{tag}]: {detail}"
        )
    worst = max(deltas.values()) if deltas else 0.0
    print(
        f"  [{fam}] {tag} determinism check OK "
        f"(repeat call identical, max Δ={worst:.1e})"
    )


def _print_row(fam, tag, m):
    if "error" in m:
        print(f"  [{fam}] {tag} ERROR: {m['error']}")
    else:
        print(
            f"  [{fam}] {tag} adv={m['advantage']:.3f}  AUC={m['attack_auc']:.3f}  "
            f"TPR={m['tpr']:.3f}  FPR={m['fpr']:.3f}  acc={m.get('test_acc', float('nan')):.3f}"
        )


# --------------------------------------------------------------------------- #
# Authoritative CSV output. Figures and interpretation belong in analysis/.
# --------------------------------------------------------------------------- #
def write_results(ds, df):
    out = os.path.join(OUT_DIR, ds)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{ds}_mia_results.csv")
    df.to_csv(path, index=False)
    print(f"  -> CSV written to {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Yeom MIA: Standard vs DP, membership advantage"
    )
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=em.MODEL_FAMILIES)
    ap.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    ap.add_argument(
        "--verify-determinism",
        action="store_true",
        help="attack every target twice and assert the two runs are identical",
    )
    ap.add_argument("--quick", action="store_true", help="tiny config for smoke test")
    args = ap.parse_args()

    if args.quick:
        args.epsilons = [0.1, 1.0]

    os.makedirs(OUT_DIR, exist_ok=True)
    for ds in args.datasets:
        run_dataset(ds, args)


if __name__ == "__main__":
    main()
