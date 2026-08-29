"""
Driver: run the Shokri shadow-model MIA against Standard vs DP targets across
the six study datasets (GALLSTONE, KIDNEY_STONE, LUNG_CANCER, BCP, CANCER_RISK
and DIABETES).

For every (dataset, model family):
  * one Standard (non-private) target
  * one DP target per epsilon

Every target is the **exported** artifact (``std_<fam>_model.pkl`` /
``dp_<fam>_model_eps_<eps>.pkl``) — the same file whose accuracy produced the ACL
number in the utility pipeline. It is loaded once per target and never retrained
or overwritten. The only randomness in a run is the reseeding of the shadow
models and the Algorithm-1 synthesis RNG, so ``*_std`` reports shadow-calibration
noise around one fixed target.

Outputs (under Attack/MIA_Shokri/results/):
  * results/<dataset>/<dataset>_results.csv   authoritative per-dataset data
  No JSON is written.

Usage:
    python run_mia.py                         # all datasets, default config
    python run_mia.py --datasets GALLSTONE    # subset
    python run_mia.py --epsilons 0.1 1.0 10.0 --dp-runs 30 --n-per-class 200
    python run_mia.py --dp-runs 30 --dnn-runs 5   # per-family run counts
    python run_mia.py --quick                 # fast smoke test
"""

import argparse
import os
import pickle
import sys
import time
import warnings

import exported_models as em
import numpy as np
import pandas as pd
from shokri_mia import run_shokri_attack

warnings.filterwarnings("ignore")  # diffprivlib privacy/convergence chatter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# The SVM notebooks import their DP estimator from ``common_svm`` at the repo
# root, so exported targets pickle as ``common_svm.DifferentiallyPrivateSVM``
# and pickle.load has to be able to import that module.
sys.path.insert(0, REPO)

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


def family_runs(fam, args):
    """Run count for a family: classical families and the DNN differ by cost.

    Each run retrains ``--n-shadow`` shadows from scratch, so the DNN is far more
    expensive than LR/RF/GNB/SVM. Both counts are CLI flags, so classical and DNN
    launches need no source edit between them.
    """
    return args.dnn_runs if fam == "DNN" else args.dp_runs


def target_utility(model, data):
    def acc(X, y):
        return float((model.predict(X) == y).mean())

    tr = acc(data["X_train"], data["y_train"])
    te = acc(data["X_test"], data["y_test"])
    return {"train_acc": tr, "test_acc": te, "gen_gap": tr - te}


def attack_disk(
    family,
    variant,
    data,
    n_classes,
    low,
    high,
    args,
    repo,
    dataset,
    epsilon=None,
    seeds=(0,),
):
    """Attack an *exported* target loaded from <dataset>/.../output/model/.

    The target is loaded once, before the seed loop, so every run attacks the
    identical object; it is never retrained and never written back. Shadows of
    the same kind are trained fresh per seed (inherent to the shadow-model
    attack) and are the only source of run-to-run variation. LR and SVM are
    attacked in their ||x||<=1 row-clipped feature space (``em.family_dataview``).
    """
    view, lo, hi = em.family_dataview(family, data, low, high)
    n_features = view["X_train"].shape[1]
    target = em.load_target(repo, dataset, family, variant, epsilon, n_classes)
    util = target_utility(target, view)

    runs = []
    for seed in seeds:
        make_shadow = em.make_shadow_factory(
            family,
            variant,
            epsilon=epsilon,
            bounds=(lo, hi),
            n_features=n_features,
            n_classes=n_classes,
            base_seed=seed,
            dataset=dataset,
        )
        metrics = run_shokri_attack(
            target,
            make_shadow,
            view,
            n_classes,
            lo,
            hi,
            n_per_class=args.n_per_class,
            n_shadow=args.n_shadow,
            conf_min=args.conf_min,
            seed=seed * 7 + 1,
        )
        metrics.update(util)
        runs.append(metrics)

    return aggregate(runs, family, variant, epsilon)


def aggregate(runs, family, variant, epsilon):
    keys = [
        "advantage",
        "attack_accuracy",
        "attack_auc",
        "attack_precision",
        "tpr",
        "fpr",
        "train_acc",
        "test_acc",
        "gen_gap",
    ]
    ok = [r for r in runs if "error" not in r]
    row = {"model": family, "variant": variant, "epsilon": epsilon, "n_runs": len(runs)}
    if not ok:
        row["error"] = runs[0].get("error", "unknown")
        return row
    for k in keys:
        vals = [r[k] for r in ok if k in r]
        row[f"{k}_mean"] = float(np.mean(vals)) if vals else None
        row[f"{k}_std"] = float(np.std(vals)) if vals else None
    return row


def run_dataset_disk(ds, args):
    """Attack every exported target for one dataset."""
    repo = REPO
    data, n_classes, low, high = load_dataset(ds)
    print(
        f"\n{'=' * 72}\n{ds}  (classes={n_classes}, n_train={len(data['y_train'])}, "
        f"n_features={data['X_train'].shape[1]})  [exported targets]\n{'=' * 72}"
    )
    ds_rows = []

    for fam in args.models:
        if fam not in em.MODEL_FAMILIES:
            print(f"  [{fam}] skipped (unknown family)")
            continue
        if not em.model_exists(repo, ds, fam, "standard"):
            print(f"  [{fam}] skipped (no exported model for this dataset)")
            continue

        t0 = time.time()
        fam_seeds = tuple(range(family_runs(fam, args)))
        print(
            f"  [{fam}] {len(fam_seeds)} run(s), shadow seeds "
            f"{fam_seeds[0]}..{fam_seeds[-1]}"
        )
        # Standard target
        row = attack_disk(
            fam,
            "standard",
            data,
            n_classes,
            low,
            high,
            args,
            repo,
            ds,
            epsilon=None,
            # Same seed count as the DP rows: the standard target is loaded once
            # from disk, but its shadows are reseeded per run, so the baseline
            # gets a real error bar instead of a lone n_runs=1.
            seeds=fam_seeds,
        )
        ds_rows.append({"dataset": ds, **row})
        print(
            f"  [{fam}] STD       adv={row.get('advantage_mean'):.3f} "
            f"auc={row.get('attack_auc_mean'):.3f} "
            f"gap={row.get('gen_gap_mean'):.3f}"
        )

        # DP targets (only epsilons actually exported for this family)
        for eps in args.epsilons:
            if not em.model_exists(repo, ds, fam, "dp", eps):
                print(f"  [{fam}] DP eps={eps:<5} skipped (not exported)")
                continue
            seeds = fam_seeds
            row = attack_disk(
                fam,
                "dp",
                data,
                n_classes,
                low,
                high,
                args,
                repo,
                ds,
                epsilon=eps,
                seeds=seeds,
            )
            ds_rows.append({"dataset": ds, **row})
            if "error" in row:
                print(f"  [{fam}] DP eps={eps:<5} ERROR: {row['error']}")
            else:
                print(
                    f"  [{fam}] DP eps={eps:<5} adv={row['advantage_mean']:.3f} "
                    f"auc={row['attack_auc_mean']:.3f} "
                    f"gap={row['gen_gap_mean']:.3f}"
                )
        print(f"  [{fam}] done in {time.time() - t0:.1f}s")

    out_dir = os.path.join(OUT_DIR, ds)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{ds}_results.csv")
    pd.DataFrame(ds_rows).to_csv(csv_path, index=False)
    print(f"  -> CSV written to {csv_path}")


def main():
    ap = argparse.ArgumentParser(description="Shokri MIA: Standard vs DP")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="default: all exported families",
    )
    ap.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    ap.add_argument(
        "--dp-runs",
        type=int,
        default=30,
        help="shadow-model reseeds per target for LR/RF/GNB/SVM (target is fixed)",
    )
    ap.add_argument(
        "--dnn-runs",
        type=int,
        default=5,
        help="shadow-model reseeds per DNN target (retraining shadows is costly)",
    )
    ap.add_argument(
        "--n-per-class", type=int, default=200, help="synthetic records/class"
    )
    ap.add_argument("--n-shadow", type=int, default=5)
    ap.add_argument("--conf-min", type=float, default=0.4)
    ap.add_argument("--quick", action="store_true", help="tiny config for smoke test")
    args = ap.parse_args()

    if args.dp_runs < 1 or args.dnn_runs < 1:
        ap.error("--dp-runs and --dnn-runs must each be at least 1")

    if args.models is None:
        args.models = em.MODEL_FAMILIES

    if args.quick:
        args.epsilons = [0.1, 1.0]
        args.dp_runs = 2
        args.dnn_runs = 2
        args.n_per_class = 60
        args.n_shadow = 3

    os.makedirs(OUT_DIR, exist_ok=True)
    for ds in args.datasets:
        run_dataset_disk(ds, args)


if __name__ == "__main__":
    main()
