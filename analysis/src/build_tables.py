"""Build the nine analysis tables as CSV files.

Every value is copied or arithmetically derived from a file listed in
``analysis/README.md``.  Nothing is estimated: a value that is not on disk is
written as the literal ``[MISSING: ...]`` string.

Run:  python build_tables.py
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

import config as C
import loaders as L
from extract_provenance import build_provenance

def write_table(name: str, df: pd.DataFrame) -> None:
    """Write one authoritative table to ``analysis/tables/csv``."""
    C.TABLES_CSV.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLES_CSV / f"{name}.csv", index=False)
    print(f"  wrote tables/csv/{name}.csv  ({len(df)} rows)")


# ==========================================================================
# Shared helpers
# ==========================================================================

def bootstrap_run_mean_ci(
    values: np.ndarray, *, seed: int, n_resamples: int = 20_000
) -> tuple[float, float]:
    """Percentile 95% CI for the mean across independent attack repetitions."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("run-level bootstrap needs at least two finite outcomes")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _low_fpr_run_summary() -> pd.DataFrame:
    """Summarise exact 1%-FPR outcomes across LiRA shadow-calibration runs."""
    runs = L.load_all_lira_runs()
    runs = runs[
        (runs["variant"] == "dp")
        & np.isclose(pd.to_numeric(runs["epsilon"], errors="coerce"), C.EPS_TARGET)
    ].copy()
    if runs.empty:
        raise ValueError(f"no LiRA per-run rows found at epsilon={C.EPS_TARGET}")
    if "error" in runs and runs["error"].notna().any():
        failed = runs.loc[runs["error"].notna(), ["dataset_dir", "model", "run_seed"]]
        raise ValueError(f"LiRA has failed epsilon=1 runs; rerun them:\n{failed}")

    count_cols = ["op_1pct_tp", "op_1pct_fp", "op_1pct_tn", "op_1pct_fn"]
    for column in count_cols:
        values = pd.to_numeric(runs[column], errors="raise").to_numpy(float)
        if not np.equal(values, np.floor(values)).all() or (values < 0).any():
            raise ValueError(f"{column} must contain non-negative integer outcomes")
        runs[column] = values.astype(int)

    rows = []
    grouped = runs.groupby(["dataset_dir", "model"], sort=True)
    for group_index, ((dataset_dir, model), group) in enumerate(grouped):
        if group["run_seed"].duplicated().any():
            raise ValueError(f"duplicate LiRA run_seed for {dataset_dir}/{model}")
        target = pd.to_numeric(group["op_1pct_target_fpr"], errors="raise").to_numpy(float)
        if not np.allclose(target, C.RANDOM_FPR):
            raise ValueError(f"unexpected target FPR for {dataset_dir}/{model}")
        if set(group["op_1pct_selection_rule"]) != {
            "max_tpr_with_empirical_fpr_le_target"
        }:
            raise ValueError(f"unexpected ROC-point rule for {dataset_dir}/{model}")

        n_members = group["op_1pct_tp"] + group["op_1pct_fn"]
        n_nonmembers = group["op_1pct_fp"] + group["op_1pct_tn"]
        if n_members.nunique() != 1 or n_nonmembers.nunique() != 1:
            raise ValueError(f"evaluation-set sizes vary across {dataset_dir}/{model} runs")
        tpr = group["op_1pct_tp"].to_numpy(float) / n_members.to_numpy(float)
        fpr = group["op_1pct_fp"].to_numpy(float) / n_nonmembers.to_numpy(float)
        recorded_tpr = pd.to_numeric(group["op_1pct_tpr"], errors="raise").to_numpy(float)
        recorded_fpr = pd.to_numeric(group["op_1pct_fpr"], errors="raise").to_numpy(float)
        if not np.allclose(tpr, recorded_tpr) or not np.allclose(fpr, recorded_fpr):
            raise ValueError(f"stored rates disagree with integer outcomes for {dataset_dir}/{model}")
        if (fpr > C.RANDOM_FPR + 1e-15).any():
            raise ValueError(f"selected ROC point exceeds 1% FPR for {dataset_dir}/{model}")

        low, high = bootstrap_run_mean_ci(tpr, seed=20_260_829 + group_index)
        resolvable = group["op_1pct_fpr_resolvable"].astype(str).str.lower().eq("true")
        rows.append({
            "dataset": L.DIR_TO_DISPLAY[dataset_dir],
            "model": model,
            "lira_tpr_at_1pct": float(np.mean(tpr)),
            "lira_tpr_at_1pct_std": float(np.std(tpr)),
            "lira_tpr_at_1pct_ci_low": low,
            "lira_tpr_at_1pct_ci_high": high,
            "n_attack_runs": int(len(group)),
            "n_members_per_run": int(n_members.iloc[0]),
            "n_nonmembers_per_run": int(n_nonmembers.iloc[0]),
            "all_runs_fpr_resolvable": "yes" if resolvable.all() else "no",
            "low_fpr_ci_method": "run-level percentile bootstrap (20000 resamples)",
        })
    return pd.DataFrame(rows)


def _std_leakage() -> pd.DataFrame:
    """One row per (dataset, family) with every baseline leakage metric."""
    m = L.load_mia()
    std = m[m["variant"] == "standard"]

    yeom = (std[std["attack"] == "yeom"]
            [["dataset", "model", "advantage_mean", "advantage_std",
              "tpr_mean", "n_members", "n_nonmembers"]]
            .rename(columns={"advantage_mean": "yeom_advantage",
                             "advantage_std": "yeom_advantage_std",
                             "tpr_mean": "yeom_tpr"}))
    shokri = (std[std["attack"] == "shokri"]
              [["dataset", "model", "attack_auc_mean", "attack_auc_std"]]
              .rename(columns={"attack_auc_mean": "shokri_auc",
                               "attack_auc_std": "shokri_auc_std"}))
    lira = (std[std["attack"] == "lira"]
            [["dataset", "model", "attack_auc_mean", "attack_auc_std",
              "tpr_at_1pct", "tpr_at_1pct_std"]]
            .rename(columns={"attack_auc_mean": "lira_auc",
                             "attack_auc_std": "lira_auc_std",
                             "tpr_at_1pct": "lira_tpr_at_1pct",
                             "tpr_at_1pct_std": "lira_tpr_at_1pct_std"}))
    out = yeom.merge(shokri, on=["dataset", "model"]).merge(lira, on=["dataset", "model"])
    return out


def _order(df: pd.DataFrame) -> pd.DataFrame:
    """Stable dataset-by-N then family ordering."""
    ds_rank = {L.DIR_TO_DISPLAY[d]: i for i, d in enumerate(L.DATASET_DIRS)}
    fam_rank = {f: i for i, f in enumerate(L.FAMILIES)}
    d = df.copy()
    d["_ds"] = d["dataset"].map(ds_rank)
    key = "family" if "family" in d.columns else "model"
    d["_fm"] = d[key].map(lambda v: fam_rank.get(L.DP_TO_SHORT.get(v, v), 99))
    d = d.sort_values(["_ds", "_fm"]).drop(columns=["_ds", "_fm"])
    return d.reset_index(drop=True)


# ==========================================================================
# 1. dataset_characteristics                                   [48][49][50]
# ==========================================================================

def t_dataset_characteristics(conflicts: list) -> pd.DataFrame:
    u = L.load_utility()
    rows = []
    for ds in L.active_dataset_dirs():
        disp = L.DIR_TO_DISPLAY[ds]
        info = L.load_split_info(ds)
        ucsv = u[u["dataset"] == disp].iloc[0]

        # Cross-check N and d between the pickle and the consolidated CSV.
        n_samples, n_features = info["n_samples"], info["n_features"]
        if int(ucsv["n_samples"]) != n_samples:
            conflicts.append(
                f"n_samples for {disp}: {info['source']} = {n_samples} vs "
                f"{L.rel(C.UTILITY_CSV)} = {int(ucsv['n_samples'])}")
            n_samples = C.CONFLICT
        if int(ucsv["n_features"]) != n_features:
            conflicts.append(
                f"n_features for {disp}: {info['source']} = {n_features} vs "
                f"{L.rel(C.UTILITY_CSV)} = {int(ucsv['n_features'])}")
            n_features = C.CONFLICT

        counts = info["class_counts_all"]
        total = sum(counts.values())
        balance = "; ".join(f"{k}: {v} ({100 * v / total:.1f}%)"
                            for k, v in counts.items())
        k = info["n_classes"]
        task = ("Binary classification" if k == 2
                else f"{k}-class classification")

        synthetic, provenance = L.DATASET_PROVENANCE[ds]
        rows.append({
            "dataset": L.DIR_TO_LONG[ds],
            "dataset_dir": ds,
            "n_samples": n_samples,
            "n_features": n_features,
            "task": task,
            "n_classes": k,
            "class_balance": balance,
            "n_train": info["n_train"],
            "n_test": info["n_test"],
            "source_url": L.DATASET_URL[ds],
            "real_or_synthetic": synthetic,
            "provenance": provenance,
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 2. baseline_accuracy_matrix                                      [11][51]
# ==========================================================================

def t_baseline_accuracy_matrix(conflicts: list) -> pd.DataFrame:
    u = L.load_utility()
    std = L.filter_datasets(u[u["variant"] == "standard"])
    m = L.load_mia()
    shokri = m[(m["attack"] == "shokri") & (m["variant"] == "standard")]

    rows = []
    for r in std.itertuples():
        fam = L.DP_TO_SHORT[r.model]
        sk = shokri[(shokri["dataset"] == r.dataset) & (shokri["model"] == fam)]
        exp_test = float(sk.iloc[0]["test_acc_mean"]) if len(sk) == 1 else np.nan
        exp_train = float(sk.iloc[0]["train_acc_mean"]) if len(sk) == 1 else np.nan
        rows.append({
            "dataset": r.dataset,
            "model": fam,
            "family": fam,
            "test_accuracy": float(r.accuracy_mean),
            "train_accuracy": float(r.train_accuracy_mean),
            "train_test_gap": float(r.train_accuracy_mean) - float(r.accuracy_mean),
            "test_accuracy_exported": exp_test,
            "train_accuracy_exported": exp_train,
            "train_test_gap_exported": exp_train - exp_test,
        })
    return _order(pd.DataFrame(rows))


# ==========================================================================
# 3. baseline_leakage_all_pairs                                        [10]
# ==========================================================================

def t_baseline_leakage_all_pairs(conflicts: list) -> pd.DataFrame:
    d = L.filter_datasets(_std_leakage())
    cols = ["dataset", "model",
            "yeom_advantage", "yeom_advantage_std",
            "shokri_auc", "shokri_auc_std",
            "lira_auc", "lira_auc_std",
            "lira_tpr_at_1pct", "lira_tpr_at_1pct_std"]
    return _order(d[cols])


# ==========================================================================
# 4. evaluation_set_sizes                                               [8]
# ==========================================================================

def t_evaluation_set_sizes(conflicts: list) -> pd.DataFrame:
    m = L.load_mia()
    rows = []
    for ds in L.active_dataset_dirs():
        disp = L.DIR_TO_DISPLAY[ds]
        sub = m[m["dataset"] == disp]

        # LiRA and Yeom both record the evaluation-set sizes; Shokri does not.
        sizes = {}
        for attack in ("lira", "yeom", "shokri"):
            a = sub[sub["attack"] == attack]
            vals = a[["n_members", "n_nonmembers"]].dropna().drop_duplicates()
            if len(vals) == 1:
                sizes[attack] = (int(vals.iloc[0]["n_members"]),
                                 int(vals.iloc[0]["n_nonmembers"]))
            elif len(vals) > 1:
                sizes[attack] = "MULTI"
        present = {k: v for k, v in sizes.items() if v != "MULTI"}
        distinct = set(present.values())
        if len(distinct) > 1:
            conflicts.append(
                f"evaluation-set sizes for {disp} differ across attacks: {present}")
            n_mem = n_non = C.CONFLICT
        else:
            n_mem, n_non = next(iter(distinct))

        # Cross-check against the split recorded in processed_data.pkl.
        info = L.load_split_info(ds)
        if not isinstance(n_mem, str) and (n_mem, n_non) != (info["n_train"], info["n_test"]):
            conflicts.append(
                f"evaluation-set sizes for {disp}: attacks report "
                f"({n_mem}, {n_non}) but {info['source']} has "
                f"(n_train={info['n_train']}, n_test={info['n_test']})")

        if isinstance(n_mem, str):
            ratio = min_fpr = exceeds = C.CONFLICT
        else:
            ratio = n_mem / n_non
            min_fpr = 1.0 / n_non
            exceeds = "yes" if min_fpr > C.RANDOM_FPR else "no"

        rows.append({
            "dataset": disp,
            "dataset_dir": ds,
            "n_members": n_mem,
            "n_nonmembers": n_non,
            "member_to_nonmember_ratio": ratio,
            "min_resolvable_fpr": min_fpr,
            "min_resolvable_fpr_exceeds_1pct": exceeds,
            "sizes_recorded_by": ", ".join(sorted(present)) or
                                 C.MISSING.format(what="attack-reported set sizes"),
            "balanced_1to1_evaluation": "no (balanced == False on all 300 LiRA rows)",
        })
    return pd.DataFrame(rows)


# ==========================================================================
# 5. residual_leakage_eps1                                         [14][15]
# ==========================================================================

def t_residual_leakage_eps1(conflicts: list) -> pd.DataFrame:
    m = L.filter_datasets(L.load_mia())
    dp = m[(m["variant"] == "dp") & (m["epsilon"] == C.EPS_TARGET)]

    yeom = (dp[dp["attack"] == "yeom"]
            [["dataset", "model", "advantage_mean", "advantage_std", "n_members"]]
            .rename(columns={"advantage_mean": "yeom_advantage",
                             "advantage_std": "yeom_advantage_std"}))
    shokri = (dp[dp["attack"] == "shokri"]
              [["dataset", "model", "attack_auc_mean", "attack_auc_std"]]
              .rename(columns={"attack_auc_mean": "shokri_auc",
                               "attack_auc_std": "shokri_auc_std"}))
    lira = (dp[dp["attack"] == "lira"]
            [["dataset", "model", "attack_auc_mean", "attack_auc_std"]]
            .rename(columns={"attack_auc_mean": "lira_auc",
                             "attack_auc_std": "lira_auc_std"}))
    d = (yeom.drop(columns=["n_members"])
         .merge(shokri, on=["dataset", "model"])
         .merge(lira, on=["dataset", "model"])
         .merge(_low_fpr_run_summary(), on=["dataset", "model"]))

    d["epsilon"] = C.EPS_TARGET
    d["family"] = d["model"]
    d["model"] = d["model"].map(L.SHORT_TO_DP)
    d["random_guess_tpr"] = C.RANDOM_FPR
    d["ci_excludes_random"] = [
        ("yes" if (not isinstance(lo, str) and lo > C.RANDOM_FPR) else "no")
        for lo in d["lira_tpr_at_1pct_ci_low"]]

    cols = ["dataset", "model", "family", "epsilon",
            "yeom_advantage", "yeom_advantage_std",
            "shokri_auc", "shokri_auc_std",
            "lira_auc", "lira_auc_std",
            "lira_tpr_at_1pct", "lira_tpr_at_1pct_std",
            "lira_tpr_at_1pct_ci_low", "lira_tpr_at_1pct_ci_high",
            "n_attack_runs", "n_members_per_run", "n_nonmembers_per_run",
            "all_runs_fpr_resolvable", "low_fpr_ci_method",
            "random_guess_tpr", "ci_excludes_random"]
    return _order(d[cols])


# ==========================================================================
# 6. rho_by_epsilon                                                    [19]
# ==========================================================================

def t_rho_by_epsilon(conflicts: list) -> pd.DataFrame:
    src = L.load_rho_by_epsilon()
    eps = sorted(src["epsilon"].unique())
    rows = []
    for (subset, al, metric), grp in src.groupby(
            ["subset", "al_definition", "metric"], sort=False):
        g = grp.set_index("epsilon")
        row = {"subset": subset, "al_definition": al, "metric": metric,
               "n": int(g["n"].iloc[0])}
        for e in eps:
            if e in g.index:
                row[f"rho_eps_{e}"] = float(g.loc[e, "rho"])
                row[f"p_eps_{e}"] = float(g.loc[e, "p"])
            else:
                row[f"rho_eps_{e}"] = C.MISSING.format(what=f"rho at eps={e}")
                row[f"p_eps_{e}"] = C.MISSING.format(what=f"p at eps={e}")
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
# 7. within_pair_correlations                                          [18]
# ==========================================================================

def t_within_pair_correlations(conflicts: list) -> pd.DataFrame:
    s = L.load_within_pair_summary()
    out = s.rename(columns={
        "median_rho": "median_rho", "iqr_low": "iqr_q1", "iqr_high": "iqr_q3"})
    out.insert(1, "n_epsilons_per_pair", 9)
    return out


# ==========================================================================
# 8. bound_violations                                              [3][33]
# ==========================================================================

def t_bound_violations(conflicts: list) -> pd.DataFrame:
    m = L.filter_datasets(L.load_mia())
    lira = m[(m["attack"] == "lira") & (m["variant"] == "dp")].copy()

    rows = []
    for r in lira.itertuples():
        bound = math.exp(float(r.epsilon)) * C.RANDOM_FPR
        tpr = float(r.tpr_at_1pct)
        if not (tpr > bound):
            continue
        rows.append({
            "dataset": r.dataset,
            "model": L.SHORT_TO_DP[r.model],
            "family": r.model,
            "epsilon": float(r.epsilon),
            "lira_tpr_at_1pct": tpr,
            "bound_exp_eps_times_fpr": bound,
            "ratio_tpr_over_bound": tpr / bound,
            "n_members": int(r.n_members),
            "n_nonmembers": int(r.n_nonmembers),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "dataset", "model", "family", "epsilon", "lira_tpr_at_1pct",
            "bound_exp_eps_times_fpr", "ratio_tpr_over_bound", "n_members",
            "n_nonmembers"])
    return _order(pd.DataFrame(rows)).sort_values(
        "ratio_tpr_over_bound", ascending=False).reset_index(drop=True)


# ==========================================================================
# 9. config_inventory                                          [9][59][61]
# ==========================================================================

def t_config_inventory(conflicts: list) -> pd.DataFrame:
    d = L.filter_datasets(build_provenance())
    return _order(d)


# ==========================================================================
# Driver
# ==========================================================================

def main() -> list[str]:
    conflicts: list[str] = []
    print("Building tables...")

    tables = [
        ("dataset_characteristics", t_dataset_characteristics),
        ("baseline_accuracy_matrix", t_baseline_accuracy_matrix),
        ("baseline_leakage_all_pairs", t_baseline_leakage_all_pairs),
        ("evaluation_set_sizes", t_evaluation_set_sizes),
        ("residual_leakage_eps1", t_residual_leakage_eps1),
        ("rho_by_epsilon", t_rho_by_epsilon),
        ("within_pair_correlations", t_within_pair_correlations),
        ("bound_violations", t_bound_violations),
        ("config_inventory", t_config_inventory),
    ]
    for name, builder in tables:
        write_table(name, builder(conflicts))

    # Persist for gaps.md; verify.py appends its own findings to the same file.
    C.LOGS.mkdir(parents=True, exist_ok=True)
    (C.LOGS / "_conflicts.json").write_text(json.dumps(conflicts, indent=2) + "\n")
    print(f"\n{len(conflicts)} conflict(s) recorded -> logs/_conflicts.json")
    return conflicts


if __name__ == "__main__":
    conf = main()
    if conf:
        print("\nCONFLICTS:")
        for c in conf:
            print("  -", c)
