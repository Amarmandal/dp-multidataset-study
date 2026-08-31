"""Artifact-matched correlation statistics and labeled sensitivity analyses.

(a) all_correlations.csv  — every correlation computed anywhere in the RQ1 correlation analysis,
    each with rho, n, p and a 95% CI. No bare rho is emitted.
(b) rho_by_epsilon.csv    — at each budget, ACL_exported vs leakage
    across the 30 dataset x model pairs (n=30 per row), one row per metric.
(c) robustness_excluded.csv — (b) recomputed after excluding pairs whose
    NON-PRIVATE baseline accuracy is within 0.05 of majority-class accuracy.
    Majority-class accuracy is read from the label distribution in each
    <DATASET>/data/processed_data.pkl.
(d) balanced_accuracy.csv — AL rebuilt from balanced accuracy and the
    headline RQ1 correlation repeated.

Nothing is winsorised, floored or tuned.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from common import (EPSILONS, LEAKAGE_METRICS, MODEL_MAP, PRIMARY_METRIC, REPO,
                    build_paired, load_mia, load_utility, spearman_with_ci,
                    summarise)

OUT = Path(__file__).resolve().parents[1] / "stats" / "correlation_stats"

MAJORITY_TOL = 0.05


# --------------------------------------------------------------------------- #
# (c) majority-class accuracy from the actual pickled labels
# --------------------------------------------------------------------------- #
def majority_class_accuracy(dataset_dirs) -> pd.DataFrame:
    """Majority-class rate from y_test / y_train / combined in processed_data.pkl.

    Baseline accuracy in consolidated_data.csv is a held-out test-set number, so
    the test-split majority rate is the comparable quantity and is used for the
    exclusion rule; the others are reported for transparency.
    """
    rows = []
    for d in sorted(set(dataset_dirs)):
        p = REPO / d / "data" / "processed_data.pkl"
        if not p.exists():
            rows.append({"dataset_dir": d, "majority_acc_test": np.nan,
                         "majority_acc_train": np.nan, "majority_acc_all": np.nan,
                         "n_train": np.nan, "n_test": np.nan,
                         "majority_class": "n/a", "class_counts_test": "n/a",
                         "source": f"MISSING: {p}"})
            continue
        with open(p, "rb") as fh:
            blob = pickle.load(fh)
        ytr = np.asarray(blob["y_train"]).ravel()
        yte = np.asarray(blob["y_test"]).ravel()
        yall = np.concatenate([ytr, yte])

        def maj(y):
            _, c = np.unique(y, return_counts=True)
            return float(c.max() / len(y))

        lab, cnt = np.unique(yte, return_counts=True)
        rows.append({
            "dataset_dir": d,
            "majority_acc_test": maj(yte),
            "majority_acc_train": maj(ytr),
            "majority_acc_all": maj(yall),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "majority_class": str(lab[int(cnt.argmax())]),
            "class_counts_test": ";".join(f"{a}:{b}" for a, b in zip(lab, cnt)),
            "source": str(p.relative_to(REPO)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (b) rho vs epsilon
# --------------------------------------------------------------------------- #
def rho_by_epsilon(
    paired: pd.DataFrame,
    al_col: str = "AL_exported",
    tag: str = "all_pairs",
):
    rows = []
    for _, _, label in LEAKAGE_METRICS:
        for eps in EPSILONS:
            g = paired[np.isclose(paired["epsilon"], eps)]
            res = spearman_with_ci(g[al_col], g[label])
            rows.append({
                "subset": tag, "al_definition": al_col, "metric": label,
                "epsilon": eps, "rho": res["rho"], "n": res["n"], "p": res["p"],
                "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                "n_pairs_available": int(len(g)),
                "note": res["note"],
            })
    return pd.DataFrame(rows)


def within_pair(
    paired: pd.DataFrame,
    al_col: str = "AL_exported",
    tag: str = "all_pairs",
):
    rows = []
    for (dataset, model), g in paired.groupby(["dataset", "model"], sort=True):
        g = g.sort_values("epsilon")
        for _, _, label in LEAKAGE_METRICS:
            res = spearman_with_ci(g[al_col], g[label])
            rows.append({
                "subset": tag, "al_definition": al_col, "dataset": dataset,
                "model": model, "metric": label, "rho": res["rho"], "n": res["n"],
                "p": res["p"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                "note": res["note"],
            })
    return pd.DataFrame(rows)


def pooled(
    paired: pd.DataFrame,
    al_col: str = "AL_exported",
    tag: str = "all_pairs",
):
    rows = []
    for _, _, label in LEAKAGE_METRICS:
        res = spearman_with_ci(paired[al_col], paired[label])
        rows.append({
            "subset": tag, "al_definition": al_col, "metric": label,
            "rho": res["rho"], "n": res["n"], "p": res["p"],
            "ci_low": res["ci_low"], "ci_high": res["ci_high"],
            "note": (res["note"] + "; points are not independent "
                     "(9 epsilons nested in 30 pairs)").strip("; "),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    utility = load_utility()
    paired = build_paired(utility, load_mia())

    # ---------------- (d) balanced-accuracy AL ---------------------------- #
    base = utility[utility["variant"] == "standard"][
        ["dataset", "model", "accuracy_mean", "balanced_accuracy_mean"]
    ].rename(columns={"accuracy_mean": "baseline_accuracy",
                      "balanced_accuracy_mean": "baseline_balanced_accuracy"})
    paired = paired.merge(base, on=["dataset", "model"], how="left")
    paired["AL_balanced"] = (
        (paired["baseline_balanced_accuracy"] - paired["balanced_accuracy_mean"])
        / paired["baseline_balanced_accuracy"]
    )
    paired.to_csv(OUT / "paired_input.csv", index=False)

    # ---------------- (c) majority-class exclusion ------------------------ #
    maj = majority_class_accuracy(paired["dataset_dir"].unique())
    maj.to_csv(OUT / "majority_class_accuracy.csv", index=False)
    print("=== (c) majority-class accuracy from processed_data.pkl ===")
    print(maj.to_string(index=False))

    pair_tbl = (paired[["dataset", "dataset_dir", "model", "baseline_accuracy"]]
                .drop_duplicates().merge(maj, on="dataset_dir", how="left"))
    pair_tbl["margin_over_majority"] = (
        pair_tbl["baseline_accuracy"] - pair_tbl["majority_acc_test"])
    pair_tbl["excluded"] = pair_tbl["margin_over_majority"].abs() < MAJORITY_TOL
    pair_tbl["reason"] = np.where(
        pair_tbl["excluded"],
        ("non-private baseline accuracy within " + str(MAJORITY_TOL)
         + " of test-split majority-class accuracy"),
        "retained",
    )
    pair_tbl = pair_tbl.sort_values(["dataset", "model"])
    pair_tbl.to_csv(OUT / "robustness_excluded.csv", index=False)

    excluded = pair_tbl[pair_tbl["excluded"]][["dataset", "model"]]
    print(f"\n=== (c) exclusion rule: |baseline_acc - majority_acc_test| < {MAJORITY_TOL} ===")
    print(pair_tbl[["dataset", "model", "baseline_accuracy", "majority_acc_test",
                    "margin_over_majority", "excluded"]].to_string(index=False))
    print(f"\nexcluded pairs: {len(excluded)} of {len(pair_tbl)}")

    keys = set(map(tuple, excluded.values))
    kept = paired[~paired.apply(lambda r: (r["dataset"], r["model"]) in keys, axis=1)]

    # ---------------- (b) + (c) rho vs epsilon ---------------------------- #
    rbe_all = rho_by_epsilon(paired, "AL_exported", "all_pairs")
    rbe_rob = rho_by_epsilon(
        kept, "AL_exported", "robustness_excluded_near_majority"
    )
    rbe_bal = rho_by_epsilon(paired, "AL_balanced", "all_pairs")
    rho_eps = pd.concat([rbe_all, rbe_rob, rbe_bal], ignore_index=True)
    rho_eps.to_csv(OUT / "rho_by_epsilon.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== (b) rho vs epsilon, artifact-matched ACL_exported, all 30 pairs ===")
    print(rbe_all[["metric", "epsilon", "rho", "n", "p", "ci_low", "ci_high"]]
          .to_string(index=False))
    print("\n=== (c) rho vs epsilon, robustness subset ===")
    print(rbe_rob[["metric", "epsilon", "rho", "n", "p", "ci_low", "ci_high"]]
          .to_string(index=False))

    # ---------------- (d) balanced-accuracy headline ---------------------- #
    wp_acc = within_pair(paired, "AL_exported", "all_pairs")
    wp_bal = within_pair(paired, "AL_balanced", "all_pairs")
    wp_rob = within_pair(
        kept, "AL_exported", "robustness_excluded_near_majority"
    )

    bal_rows = []
    for al_col, wp, tag in (("AL_exported", wp_acc, "artifact-matched ACL"),
                            ("AL_balanced", wp_bal, "balanced-accuracy AL")):
        for _, _, label in LEAKAGE_METRICS:
            sub = wp[wp["metric"] == label]
            s = {"al_definition": al_col, "description": tag, "metric": label,
                 "analysis": "within-pair Spearman, n=9 per pair"}
            s.update(summarise(sub["rho"], sub["p"]))
            bal_rows.append(s)
        p = pooled(paired, al_col, "all_pairs")
        for _, r in p.iterrows():
            bal_rows.append({
                "al_definition": al_col, "description": tag, "metric": r["metric"],
                "analysis": "pooled Spearman over all paired points",
                "n_pairs_total": np.nan, "n_computable": np.nan, "n_undefined": np.nan,
                "median_rho": np.nan, "iqr_low": np.nan, "iqr_high": np.nan,
                "min_rho": np.nan, "max_rho": np.nan,
                "n_significant_p05": np.nan, "n_significant_negative": np.nan,
                "n_significant_positive": np.nan,
                "pooled_rho": r["rho"], "pooled_n": r["n"], "pooled_p": r["p"],
                "pooled_ci_low": r["ci_low"], "pooled_ci_high": r["ci_high"],
            })
    bal = pd.DataFrame(bal_rows)
    bal.to_csv(OUT / "balanced_accuracy.csv", index=False)

    print("\n=== (d) headline RQ1: accuracy-based vs balanced-accuracy AL ===")
    print(bal[bal["analysis"].str.startswith("within-pair")][
        ["al_definition", "metric", "n_computable", "median_rho", "iqr_low",
         "iqr_high", "n_significant_p05", "n_significant_negative",
         "n_significant_positive"]].to_string(index=False))
    print("\n--- pooled ---")
    print(bal[bal["analysis"].str.startswith("pooled")][
        ["al_definition", "metric", "pooled_rho", "pooled_n", "pooled_p",
         "pooled_ci_low", "pooled_ci_high"]].to_string(index=False))

    # ------- (b) trend: does rho move systematically with the budget? ----- #
    trend_rows = []
    for tag, tbl in (("all_pairs", rbe_all),
                     ("robustness_excluded_near_majority", rbe_rob),
                     ("all_pairs_balanced_AL", rbe_bal)):
        for _, _, label in LEAKAGE_METRICS:
            s = tbl[tbl["metric"] == label].sort_values("epsilon")
            for stat, series in (("rho", s["rho"]), ("abs_rho", s["rho"].abs())):
                res = spearman_with_ci(s["epsilon"], series)
                trend_rows.append({
                    "subset": tag, "metric": label, "trend_statistic": stat,
                    "rho": res["rho"], "n": res["n"], "p": res["p"],
                    "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                    "note": (res["note"] + "; Spearman of the per-epsilon "
                             f"{stat} against epsilon, n=9 budgets").strip("; "),
                })
    trend = pd.DataFrame(trend_rows)
    trend.to_csv(OUT / "rho_epsilon_trend.csv", index=False)
    print("\n=== (b) trend of rho with epsilon (n=9 budgets) ===")
    print(trend[["subset", "metric", "trend_statistic", "rho", "n", "p",
                 "ci_low", "ci_high"]].to_string(index=False))

    # ---------------- (a) every correlation, full stats ------------------- #
    def norm(df, analysis):
        d = df.copy()
        d["analysis"] = analysis
        for c in ("dataset", "model", "epsilon"):
            if c not in d:
                d[c] = "n/a" if c != "epsilon" else np.nan
        return d[["analysis", "subset", "al_definition", "dataset", "model",
                  "metric", "epsilon", "rho", "n", "p", "ci_low", "ci_high", "note"]]

    all_corr = pd.concat([
        norm(wp_acc, "within-pair (n=9 across epsilons)"),
        norm(wp_rob, "within-pair, robustness subset"),
        norm(wp_bal, "within-pair, balanced-accuracy AL"),
        norm(rbe_all, "across-pair at fixed epsilon (n=30)"),
        norm(rbe_rob, "across-pair at fixed epsilon, robustness subset"),
        norm(rbe_bal, "across-pair at fixed epsilon, balanced-accuracy AL"),
        norm(pooled(paired, "AL_exported", "all_pairs"),
             "pooled over all paired points, artifact-matched ACL"),
        norm(pooled(paired, "AL_balanced", "all_pairs"),
             "pooled, balanced-accuracy AL"),
        norm(pooled(kept, "AL_exported", "robustness_excluded_near_majority"),
             "pooled, robustness subset, artifact-matched ACL"),
        norm(trend.assign(al_definition="see subset"),
             "trend of per-epsilon rho against epsilon (n=9)"),
    ], ignore_index=True)
    all_corr.to_csv(OUT / "all_correlations.csv", index=False)
    print(f"\nall_correlations.csv: {len(all_corr)} correlations, "
          f"{int(all_corr['rho'].isna().sum())} undefined")


if __name__ == "__main__":
    main()
