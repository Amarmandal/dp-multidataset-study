"""Independent re-derivation of every numeric cell in every generated table.

The tables are built from the two consolidated CSVs plus the reused ``analysis/``
outputs.  This module deliberately does NOT re-read those same aggregates where
a more primary source exists: leakage cells are checked against the per-dataset
attack CSVs under ``Attack/*/results/``, baseline accuracies against the
per-model ``std_*_report.json`` files, and split sizes against
``processed_data.pkl``.  Reformat-only tables (``rho_by_epsilon``,
``within_pair_correlations``) are checked cell-for-cell against the
``analysis/`` CSVs they reproduce, which is the correct check for a reformat.

Writes ``logs/verification.txt``:
    <table> | <row> | <column> | <value> | <source file> | <source row/filter>
Failures are listed at the top.  Exits non-zero if anything fails.

Run:  python verify.py
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C
import loaders as L
from build_tables import clopper_pearson

TOL = 1e-6

_checks: list[str] = []
_failures: list[str] = []
_notes: list[str] = []


def log(table: str, row: str, column: str, value, source: str, filt: str) -> None:
    v = value if isinstance(value, str) else (
        f"{value:.12g}" if isinstance(value, float) else str(value))
    _checks.append(f"{table} | {row} | {column} | {v} | {source} | {filt}")


def fail(msg: str) -> None:
    _failures.append(msg)


def close(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    if a is None or b is None:
        return a is b
    if (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return (isinstance(a, float) and math.isnan(a)) and \
               (isinstance(b, float) and math.isnan(b))
    return abs(float(a) - float(b)) <= TOL


def check(table: str, row: str, column: str, got, expected, source: str,
          filt: str) -> None:
    """Assert one cell against an independently obtained value."""
    if close(got, expected):
        log(table, row, column, got, source, filt)
    else:
        fail(f"{table} | {row} | {column} | table={got!r} != source={expected!r} "
             f"| {source} | {filt}")


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(C.TABLES_CSV / f"{name}.csv")


# ==========================================================================
# Per-dataset attack ground truth, indexed for lookup
# ==========================================================================

def _attack_index() -> dict:
    """(attack, dataset_dir, family, variant, epsilon) -> row of the per-dataset CSV."""
    idx = {}
    for attack in ("lira", "shokri", "yeom"):
        for ds in L.DATASET_DIRS:
            df = L.load_attack_csv(attack, ds)
            for r in df.itertuples():
                eps = float(r.epsilon) if pd.notna(r.epsilon) else None
                idx[(attack, ds, r.model, r.variant, eps)] = r
    return idx


ATT = _attack_index()

ATT_SRC = {
    "lira": "Attack/LiRA/results/{ds}/{ds}_lira_results.csv",
    "shokri": "Attack/MIA_Shokri/results/{ds}/{ds}_results.csv",
    "yeom": "Attack/MIA_YEOM/results/{ds}/{ds}_mia_results.csv",
}


def att(attack: str, ds_dir: str, fam: str, variant: str, eps):
    return ATT.get((attack, ds_dir, fam, variant, eps))


def att_src(attack: str, ds_dir: str, fam: str, variant: str, eps) -> tuple[str, str]:
    src = ATT_SRC[attack].format(ds=ds_dir)
    filt = f"model={fam}, variant={variant}, epsilon={eps}"
    return src, filt


# ==========================================================================
# 1. dataset_characteristics
# ==========================================================================

def v_dataset_characteristics() -> None:
    t = read_table("dataset_characteristics")
    maj = L.load_majority_class_accuracy().set_index("dataset_dir")
    for r in t.itertuples():
        ds = r.dataset_dir
        info = L.load_split_info(ds)          # independent re-read of the pickle
        src = info["source"]
        for col, exp in (("n_samples", info["n_samples"]),
                         ("n_features", info["n_features"]),
                         ("n_train", info["n_train"]),
                         ("n_test", info["n_test"]),
                         ("n_classes", info["n_classes"])):
            check("dataset_characteristics", ds, col, getattr(r, col), exp,
                  src, "y_train/y_test/X_train")

        # Second, independent source for the split sizes.
        if ds in maj.index:
            for col, key in (("n_train", "n_train"), ("n_test", "n_test")):
                check("dataset_characteristics", ds, f"{col} (cross-check)",
                      getattr(r, col), int(maj.loc[ds, key]),
                      "analysis/correlation_stats/majority_class_accuracy.csv",
                      f"dataset_dir={ds}")

        # Class-balance percentages must sum to 100 and match the raw counts.
        total = sum(info["class_counts_all"].values())
        parts = [p.strip() for p in str(r.class_balance).split(";")]
        pct_sum = 0.0
        for part in parts:
            cls = part.split(":")[0].strip()
            pct = float(part.split("(")[1].rstrip("%)"))
            pct_sum += pct
            check("dataset_characteristics", f"{ds}/class {cls}", "class_balance_pct",
                  pct, round(100 * info["class_counts_all"][cls] / total, 1),
                  src, f"class {cls} count / N")
        if abs(pct_sum - 100.0) > 0.15:
            fail(f"dataset_characteristics | {ds} | class_balance | "
                 f"percentages sum to {pct_sum}, not 100")

        # The URL must be byte-identical to the supplied constant.
        check("dataset_characteristics", ds, "source_url", r.source_url,
              L.DATASET_URL[ds], "analysis/src/loaders.py:DATASET_URL",
              "verbatim as supplied")
        check("dataset_characteristics", ds, "real_or_synthetic",
              r.real_or_synthetic, C.MISSING.format(what="real/synthetic"),
              "n/a", "not recorded anywhere in the repository")


# ==========================================================================
# 2. baseline_accuracy_matrix
# ==========================================================================

def _report_accuracy(report: dict) -> tuple:
    """(test_accuracy, train_accuracy) from a std report, wherever they live."""
    if report is None:
        return None, None
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    test = report.get("accuracy", metrics.get("accuracy"))
    train = report.get("train_accuracy", metrics.get("train_accuracy"))
    return test, train


def v_baseline_accuracy_matrix() -> None:
    t = read_table("baseline_accuracy_matrix")
    for r in t.itertuples():
        ds = L.DISPLAY_TO_DIR[r.dataset]
        fam = r.family
        rep = L.read_report(ds, fam, "std")
        src = (rep or {}).get("__path__", "n/a")
        exp_test, exp_train = _report_accuracy(rep)

        if exp_test is None:
            _notes.append(f"baseline_accuracy_matrix | {r.dataset}/{fam} | "
                          f"std report {src} carries no accuracy field; "
                          f"cell not independently re-derivable")
        else:
            check("baseline_accuracy_matrix", f"{r.dataset}/{fam}", "test_accuracy",
                  r.test_accuracy, float(exp_test), src, "accuracy")
        if exp_train is None:
            _notes.append(f"baseline_accuracy_matrix | {r.dataset}/{fam} | "
                          f"std report {src} carries no train_accuracy field")
        else:
            check("baseline_accuracy_matrix", f"{r.dataset}/{fam}", "train_accuracy",
                  r.train_accuracy, float(exp_train), src, "train_accuracy")

        # The gap is arithmetic on the two columns in the same row.
        check("baseline_accuracy_matrix", f"{r.dataset}/{fam}", "train_test_gap",
              r.train_test_gap, r.train_accuracy - r.test_accuracy,
              "derived", "train_accuracy - test_accuracy")

        # The exported-artefact columns come from the Shokri per-dataset CSV.
        a = att("shokri", ds, fam, "standard", None)
        if a is not None:
            src2, filt2 = att_src("shokri", ds, fam, "standard", None)
            check("baseline_accuracy_matrix", f"{r.dataset}/{fam}",
                  "test_accuracy_exported", r.test_accuracy_exported,
                  float(a.test_acc_mean), src2, filt2)
            check("baseline_accuracy_matrix", f"{r.dataset}/{fam}",
                  "train_accuracy_exported", r.train_accuracy_exported,
                  float(a.train_acc_mean), src2, filt2)
            check("baseline_accuracy_matrix", f"{r.dataset}/{fam}",
                  "train_test_gap_exported", r.train_test_gap_exported,
                  r.train_accuracy_exported - r.test_accuracy_exported,
                  "derived", "train_accuracy_exported - test_accuracy_exported")


# ==========================================================================
# 3. baseline_leakage_all_pairs
# ==========================================================================

_BASE_COLS = [
    ("yeom", "yeom_advantage", "advantage"),
    ("yeom", "yeom_advantage_std", "advantage_std"),
    ("shokri", "shokri_auc", "attack_auc_mean"),
    ("shokri", "shokri_auc_std", "attack_auc_std"),
    ("lira", "lira_auc", "attack_auc"),
    ("lira", "lira_auc_std", "attack_auc_std"),
    ("lira", "lira_tpr_at_1pct", "tpr_at_1pct"),
    ("lira", "lira_tpr_at_1pct_std", "tpr_at_1pct_std"),
]


def v_baseline_leakage_all_pairs() -> None:
    t = read_table("baseline_leakage_all_pairs")
    for r in t.itertuples():
        ds = L.DISPLAY_TO_DIR[r.dataset]
        fam = r.model
        for attack, tcol, scol in _BASE_COLS:
            a = att(attack, ds, fam, "standard", None)
            src, filt = att_src(attack, ds, fam, "standard", None)
            if a is None:
                fail(f"baseline_leakage_all_pairs | {r.dataset}/{fam} | {tcol} | "
                     f"no matching row in {src} ({filt})")
                continue
            check("baseline_leakage_all_pairs", f"{r.dataset}/{fam}", tcol,
                  getattr(r, tcol), float(getattr(a, scol)), src, filt)


# ==========================================================================
# 4. evaluation_set_sizes
# ==========================================================================

def v_evaluation_set_sizes() -> None:
    t = read_table("evaluation_set_sizes")
    for r in t.itertuples():
        ds = r.dataset_dir
        a = att("lira", ds, "RF", "standard", None)
        src, filt = att_src("lira", ds, "RF", "standard", None)
        check("evaluation_set_sizes", ds, "n_members", r.n_members,
              int(a.n_members), src, filt)
        check("evaluation_set_sizes", ds, "n_nonmembers", r.n_nonmembers,
              int(a.n_nonmembers), src, filt)

        info = L.load_split_info(ds)
        check("evaluation_set_sizes", ds, "n_members (cross-check)", r.n_members,
              info["n_train"], info["source"], "len(y_train)")
        check("evaluation_set_sizes", ds, "n_nonmembers (cross-check)",
              r.n_nonmembers, info["n_test"], info["source"], "len(y_test)")

        check("evaluation_set_sizes", ds, "member_to_nonmember_ratio",
              r.member_to_nonmember_ratio, r.n_members / r.n_nonmembers,
              "derived", "n_members / n_nonmembers")
        check("evaluation_set_sizes", ds, "min_resolvable_fpr",
              r.min_resolvable_fpr, 1.0 / r.n_nonmembers,
              "derived", "1 / n_nonmembers")
        check("evaluation_set_sizes", ds, "min_resolvable_fpr_exceeds_1pct",
              r.min_resolvable_fpr_exceeds_1pct,
              "yes" if (1.0 / r.n_nonmembers) > C.RANDOM_FPR else "no",
              "derived", f"1/n_nonmembers > RANDOM_FPR={C.RANDOM_FPR}")

    # The 'balanced' claim underpinning the [8] gap must hold on all LiRA rows.
    lira_all = L.load_all_attack_csvs("lira")
    n_bal = int(lira_all["balanced"].astype(str).str.lower().eq("true").sum())
    if n_bal != 0:
        fail(f"evaluation_set_sizes | all | balanced | {n_bal} LiRA rows have "
             f"balanced==True, contradicting the table note")
    else:
        log("evaluation_set_sizes", "all", "balanced==False on all rows",
            f"{len(lira_all)} rows", "Attack/LiRA/results/*/*_lira_results.csv",
            "column 'balanced'")


# ==========================================================================
# 5. residual_leakage_eps1
# ==========================================================================

_DP_COLS = [
    ("yeom", "yeom_advantage", "advantage"),
    ("yeom", "yeom_advantage_std", "advantage_std"),
    ("shokri", "shokri_auc", "attack_auc_mean"),
    ("shokri", "shokri_auc_std", "attack_auc_std"),
    ("lira", "lira_auc", "attack_auc"),
    ("lira", "lira_auc_std", "attack_auc_std"),
    ("lira", "lira_tpr_at_1pct", "tpr_at_1pct"),
    ("lira", "lira_tpr_at_1pct_std", "tpr_at_1pct_std"),
]


def v_residual_leakage_eps1() -> None:
    t = read_table("residual_leakage_eps1")
    for r in t.itertuples():
        ds = L.DISPLAY_TO_DIR[r.dataset]
        fam = r.family
        tag = f"{r.dataset}/{fam}"
        for attack, tcol, scol in _DP_COLS:
            a = att(attack, ds, fam, "dp", C.EPS_TARGET)
            src, filt = att_src(attack, ds, fam, "dp", C.EPS_TARGET)
            if a is None:
                fail(f"residual_leakage_eps1 | {tag} | {tcol} | no row in {src} ({filt})")
                continue
            check("residual_leakage_eps1", tag, tcol, getattr(r, tcol),
                  float(getattr(a, scol)), src, filt)

        # Clopper-Pearson recomputed from scratch.
        a = att("lira", ds, fam, "dp", C.EPS_TARGET)
        src, filt = att_src("lira", ds, fam, "dp", C.EPS_TARGET)
        k, n, lo, hi = clopper_pearson(float(a.tpr_at_1pct), float(a.n_members))
        check("residual_leakage_eps1", tag, "cp_k", r.cp_k, k,
              "derived", "round(TPR * n_members)")
        check("residual_leakage_eps1", tag, "cp_n_members", r.cp_n_members, n,
              src, filt + " -> n_members")
        check("residual_leakage_eps1", tag, "lira_tpr_at_1pct_ci_low",
              r.lira_tpr_at_1pct_ci_low, lo,
              "derived", f"binomtest({k},{n}).proportion_ci(method='exact')")
        check("residual_leakage_eps1", tag, "lira_tpr_at_1pct_ci_high",
              r.lira_tpr_at_1pct_ci_high, hi,
              "derived", f"binomtest({k},{n}).proportion_ci(method='exact')")
        check("residual_leakage_eps1", tag, "ci_excludes_random",
              r.ci_excludes_random, "yes" if lo > C.RANDOM_FPR else "no",
              "derived", f"ci_low > RANDOM_FPR={C.RANDOM_FPR}")
        check("residual_leakage_eps1", tag, "epsilon", r.epsilon, C.EPS_TARGET,
              "analysis/src/config.py", "EPS_TARGET")


# ==========================================================================
# 6 + 7. reformat-only tables
# ==========================================================================

def v_rho_by_epsilon() -> None:
    t = read_table("rho_by_epsilon")
    src_df = L.load_rho_by_epsilon()
    src = "analysis/correlation_stats/rho_by_epsilon.csv"
    eps_cols = [c for c in t.columns if c.startswith("rho_eps_")]
    # Column names like "rho_eps_0.1" are not valid identifiers, so index by
    # label rather than using itertuples().
    for _i, r in t.iterrows():
        tag = f"{r['subset']}/{r['al_definition']}/{r['metric']}"
        for col in eps_cols:
            e = float(col.replace("rho_eps_", ""))
            m = src_df[(src_df["subset"] == r["subset"])
                       & (src_df["al_definition"] == r["al_definition"])
                       & (src_df["metric"] == r["metric"])
                       & (src_df["epsilon"] == e)]
            filt = (f"subset={r['subset']}, al_definition={r['al_definition']}, "
                    f"metric={r['metric']}, epsilon={e}")
            if len(m) != 1:
                fail(f"rho_by_epsilon | {tag} | {col} | {len(m)} source rows for {filt}")
                continue
            check("rho_by_epsilon", tag, col, r[col],
                  float(m.iloc[0]["rho"]), src, filt)
            check("rho_by_epsilon", tag, f"p_eps_{e}", r[f"p_eps_{e}"],
                  float(m.iloc[0]["p"]), src, filt)
            check("rho_by_epsilon", tag, "n", r["n"], int(m.iloc[0]["n"]), src, filt)


def v_within_pair_correlations() -> None:
    t = read_table("within_pair_correlations")
    src_df = L.load_within_pair_summary()
    src = "analysis/within_pair/summary.csv"
    ren = {"iqr_q1": "iqr_low", "iqr_q3": "iqr_high"}
    numeric = [c for c in t.columns
               if c not in ("metric", "n_epsilons_per_pair")]
    for r in t.itertuples():
        m = src_df[src_df["metric"] == r.metric]
        filt = f"metric={r.metric}"
        if len(m) != 1:
            fail(f"within_pair_correlations | {r.metric} | - | {len(m)} source rows")
            continue
        for col in numeric:
            check("within_pair_correlations", r.metric, col, getattr(r, col),
                  float(m.iloc[0][ren.get(col, col)]), src, filt)

    # The nine-point-per-pair claim must match the per-pair file.
    pairs = L.load_within_pair_correlations()
    ns = sorted(pairs["n"].dropna().unique().tolist())
    if ns != [9]:
        fail(f"within_pair_correlations | all | n_epsilons_per_pair | "
             f"pair_correlations.csv has n in {ns}, not [9]")
    else:
        log("within_pair_correlations", "all", "n_epsilons_per_pair", 9,
            "analysis/within_pair/pair_correlations.csv", "column 'n'")


# ==========================================================================
# 8. bound_violations
# ==========================================================================

def v_bound_violations() -> None:
    t = read_table("bound_violations")
    src_rows = 0
    # Recompute the whole violation set from the per-dataset LiRA CSVs.
    expected = {}
    for ds in L.active_dataset_dirs():
        df = L.load_attack_csv("lira", ds)
        for r in df[df["variant"] == "dp"].itertuples():
            src_rows += 1
            bound = math.exp(float(r.epsilon)) * C.RANDOM_FPR
            if float(r.tpr_at_1pct) > bound:
                expected[(ds, r.model, float(r.epsilon))] = (r, bound)

    got = {(L.DISPLAY_TO_DIR[r.dataset], r.family, float(r.epsilon))
           for r in t.itertuples()}
    if got != set(expected):
        missing = set(expected) - got
        extra = got - set(expected)
        if missing:
            fail(f"bound_violations | - | membership | {len(missing)} violating "
                 f"configs absent from the table: {sorted(missing)[:5]}")
        if extra:
            fail(f"bound_violations | - | membership | {len(extra)} non-violating "
                 f"configs present in the table: {sorted(extra)[:5]}")
    else:
        log("bound_violations", "all", "row set", f"{len(got)} of {src_rows} DP configs",
            "Attack/LiRA/results/*/*_lira_results.csv",
            f"variant=dp and tpr_at_1pct > exp(eps)*{C.RANDOM_FPR}")

    for r in t.itertuples():
        ds = L.DISPLAY_TO_DIR[r.dataset]
        key = (ds, r.family, float(r.epsilon))
        if key not in expected:
            continue
        a, bound = expected[key]
        src, filt = att_src("lira", ds, r.family, "dp", float(r.epsilon))
        tag = f"{r.dataset}/{r.family}/eps={r.epsilon}"
        check("bound_violations", tag, "lira_tpr_at_1pct", r.lira_tpr_at_1pct,
              float(a.tpr_at_1pct), src, filt)
        check("bound_violations", tag, "bound_exp_eps_times_fpr",
              r.bound_exp_eps_times_fpr, bound,
              "derived", f"exp({r.epsilon}) * {C.RANDOM_FPR}")
        check("bound_violations", tag, "ratio_tpr_over_bound",
              r.ratio_tpr_over_bound, float(a.tpr_at_1pct) / bound,
              "derived", "tpr_at_1pct / bound")
        check("bound_violations", tag, "n_members", r.n_members,
              int(a.n_members), src, filt)
        check("bound_violations", tag, "n_nonmembers", r.n_nonmembers,
              int(a.n_nonmembers), src, filt)
        k, n, lo, hi = clopper_pearson(float(a.tpr_at_1pct), float(a.n_members))
        check("bound_violations", tag, "cp_ci_low", r.cp_ci_low, lo,
              "derived", f"binomtest({k},{n}).proportion_ci(method='exact')")
        check("bound_violations", tag, "cp_ci_high", r.cp_ci_high, hi,
              "derived", f"binomtest({k},{n}).proportion_ci(method='exact')")
        check("bound_violations", tag, "ci_low_exceeds_bound",
              r.ci_low_exceeds_bound, "yes" if lo > bound else "no",
              "derived", "cp_ci_low > bound")


# ==========================================================================
# 9. config_inventory
# ==========================================================================

def v_config_inventory() -> None:
    t = read_table("config_inventory")
    u = L.load_utility()
    for r in t.itertuples():
        ds = r.dataset_dir
        fam = r.family
        tag = f"{r.dataset}/{fam}"

        std_path = L.find_report(ds, fam, "std")
        dp_path = L.find_report(ds, fam, "dp")
        check("config_inventory", tag, "source_std_report", r.source_std_report,
              L.rel(std_path) if std_path else C.MISSING.format(what="std report JSON"),
              "filesystem glob", f"{ds}/{L.SHORT_TO_DIR[fam]}/**/std_*_report.json")
        check("config_inventory", tag, "source_dp_report", r.source_dp_report,
              L.rel(dp_path) if dp_path else C.MISSING.format(what="dp report JSON"),
              "filesystem glob", f"{ds}/{L.SHORT_TO_DIR[fam]}/**/dp_*_report.json")

        su = u[(u["variant"] == "standard") & (u["dataset"] == r.dataset)
               & (u["model"] == L.SHORT_TO_DP[fam])]
        du = u[(u["variant"] == "dp") & (u["dataset"] == r.dataset)
               & (u["model"] == L.SHORT_TO_DP[fam])
               & (u["epsilon"] == C.EPS_TARGET)]
        check("config_inventory", tag, "std_train_time_s", r.std_train_time_s,
              float(su.iloc[0]["train_time_mean"]),
              "Results/dataset_results/consolidated_data.csv",
              f"variant=standard, dataset={r.dataset}, model={L.SHORT_TO_DP[fam]}")
        check("config_inventory", tag, "train_time_mean", r.train_time_mean,
              float(du.iloc[0]["train_time_mean"]),
              "Results/dataset_results/consolidated_data.csv",
              f"variant=dp, epsilon={C.EPS_TARGET}, model={L.SHORT_TO_DP[fam]}")
        check("config_inventory", tag, "train_time_std", r.train_time_std,
              float(du.iloc[0]["train_time_std"]),
              "Results/dataset_results/consolidated_data.csv",
              f"variant=dp, epsilon={C.EPS_TARGET}, model={L.SHORT_TO_DP[fam]}")

        for col in [c for c in t.columns if c.startswith("version_")]:
            check("config_inventory", tag, col, getattr(r, col),
                  C.MISSING.format(what="software version"),
                  "n/a", "not recorded in any report JSON; not read from this machine")


# ==========================================================================
# Global assertions
# ==========================================================================

def v_n_runs() -> None:
    u = L.load_utility()
    dp = u[u["variant"] == "dp"]
    ok = dp[dp["n_runs"] == 30]
    if len(ok) != 378:
        fail(f"global | n_runs | expected 378 DP rows with n_runs==30, found {len(ok)} "
             f"of {len(dp)} DP rows | Results/dataset_results/consolidated_data.csv")
    else:
        log("global", "all DP utility rows", "n_runs == 30", 378,
            "Results/dataset_results/consolidated_data.csv", "variant=dp")

    blank = dp[dp["n_runs"].isna()]
    if len(blank) != len(dp) - 378:
        fail(f"global | n_runs | {len(dp) - 378 - len(blank)} DP rows have an "
             f"n_runs that is neither 30 nor blank")
    if len(blank):
        where = sorted({f"{a}/{b}" for a, b in
                        zip(blank["dataset"], blank["model"])})
        _notes.append(f"global | n_runs | {len(blank)} DP rows carry no n_runs at all "
                      f"({', '.join(where)}); their dp_dnn_report.json predates the "
                      f"per-run bookkeeping. Not a value mismatch -- an absent field.")
        log("global", "DP utility rows without n_runs", "count", len(blank),
            "Results/dataset_results/consolidated_data.csv",
            "variant=dp and n_runs is NaN")


def v_rq_correlations() -> None:
    """The six RQ-figure correlations must reproduce from figure_input.csv."""
    d = L.load_figure_input()
    corr = L.load_correlations()
    src = "analysis/figures/figure_input.csv"

    rf = d[d["model"] == "RF"]
    spec = {
        "L4  yeom_adv_base vs lira_auc_base": (d, "yeom_adv_base", "lira_auc_base"),
        "RQ1c ACL vs tpr1_dp": (d.dropna(subset=["ACL"]), "ACL", "tpr1_dp"),
        "RQ1c ACL_exported vs tpr1_dp":
            (d.dropna(subset=["ACL_exported"]), "ACL_exported", "tpr1_dp"),
        "RQ2a tpr1_base vs reduction": (d, "tpr1_base", "reduction"),
        "RQ2b tpr1_base vs tpr1_dp": (d, "tpr1_base", "tpr1_dp"),
        "L6  n_samples vs std-RF tpr1_base": (rf, "n_samples", "tpr1_base"),
    }
    seen = set()
    for r in corr.itertuples():
        fig = r.figure
        seen.add(fig)
        if fig not in spec:
            fail(f"rq_figures | {fig} | figure | present in correlations.csv but this "
                 f"module has no re-derivation for it")
            continue
        frame, xc, yc = spec[fig]
        res = spearmanr(frame[xc], frame[yc])
        check("d16_correlations", fig, "rho", float(r.rho), float(res.statistic),
              src, f"spearmanr({xc}, {yc}), n={len(frame)}")
        check("d16_correlations", fig, "p", float(r.p), float(res.pvalue),
              src, f"spearmanr({xc}, {yc})")
        check("d16_correlations", fig, "n", int(r.n), len(frame),
              src, f"len of non-null {xc}/{yc}")
    for fig in spec:
        if fig not in seen:
            fail(f"rq_figures | {fig} | figure | expected in correlations.csv, absent")


def v_consolidated_vs_per_dataset() -> None:
    """Every consolidated MIA value must match its per-dataset attack CSV."""
    m = L.load_mia()
    cols = {
        "lira": [("attack_auc_mean", "attack_auc"), ("attack_auc_std", "attack_auc_std"),
                 ("advantage_mean", "advantage"), ("advantage_std", "advantage_std"),
                 ("tpr_at_1pct", "tpr_at_1pct"), ("tpr_at_1pct_std", "tpr_at_1pct_std"),
                 ("tpr_at_10pct", "tpr_at_10pct"), ("tpr_at_0p1pct", "tpr_at_0p1pct"),
                 ("n_members", "n_members"), ("n_nonmembers", "n_nonmembers")],
        "yeom": [("advantage_mean", "advantage"), ("advantage_std", "advantage_std"),
                 ("attack_auc_mean", "attack_auc"), ("attack_auc_std", "attack_auc_std"),
                 ("tpr_mean", "tpr"), ("fpr_mean", "fpr"),
                 ("n_members", "n_members"), ("n_nonmembers", "n_nonmembers")],
        "shokri": [("attack_auc_mean", "attack_auc_mean"),
                   ("attack_auc_std", "attack_auc_std"),
                   ("advantage_mean", "advantage_mean"),
                   ("advantage_std", "advantage_std"),
                   ("tpr_mean", "tpr_mean"), ("fpr_mean", "fpr_mean"),
                   ("train_acc_mean", "train_acc_mean"),
                   ("test_acc_mean", "test_acc_mean")],
    }
    n_ok = 0
    for r in m.itertuples():
        ds = r.dataset_dir
        eps = float(r.epsilon) if pd.notna(r.epsilon) else None
        a = att(r.attack, ds, r.model, r.variant, eps)
        src, filt = att_src(r.attack, ds, r.model, r.variant, eps)
        if a is None:
            fail(f"consolidated | {ds}/{r.model}/{r.variant}/eps={eps} | - | "
                 f"no matching row in {src}")
            continue
        for ccol, acol in cols[r.attack]:
            cv, av = getattr(r, ccol), getattr(a, acol, None)
            if av is None:
                continue
            if not close(cv, av):
                fail(f"consolidated vs per-dataset | {ds}/{r.model}/{r.variant}/"
                     f"eps={eps} | {ccol} | consolidated={cv!r} != {acol}={av!r} "
                     f"| {src} | {filt}")
            else:
                n_ok += 1
    log("global", "consolidated_mia_data.csv vs per-dataset attack CSVs",
        "values agreeing", n_ok,
        "Attack/{LiRA,MIA_Shokri,MIA_YEOM}/results/*/*.csv",
        f"all {len(m)} consolidated rows joined on dataset/model/variant/epsilon")


# ==========================================================================
# Driver
# ==========================================================================

VERIFIERS = [
    ("dataset_characteristics", v_dataset_characteristics),
    ("baseline_accuracy_matrix", v_baseline_accuracy_matrix),
    ("baseline_leakage_all_pairs", v_baseline_leakage_all_pairs),
    ("evaluation_set_sizes", v_evaluation_set_sizes),
    ("residual_leakage_eps1", v_residual_leakage_eps1),
    ("rho_by_epsilon", v_rho_by_epsilon),
    ("within_pair_correlations", v_within_pair_correlations),
    ("bound_violations", v_bound_violations),
    ("config_inventory", v_config_inventory),
    ("global: n_runs", v_n_runs),
    ("global: RQ-figure correlations", v_rq_correlations),
    ("global: consolidated vs per-dataset", v_consolidated_vs_per_dataset),
]


def main() -> int:
    for name, fn in VERIFIERS:
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001
            fail(f"{name} | - | - | verifier raised {type(exc).__name__}: {exc}")

    C.LOGS.mkdir(parents=True, exist_ok=True)
    out = []
    out.append("Verification of analysis/tables -- every numeric cell re-derived")
    out.append("independently from its primary source.")
    out.append(f"Tolerance: {TOL:g}.  Config: INCLUDE_LUNG_CANCER="
               f"{C.INCLUDE_LUNG_CANCER}, RANDOM_FPR={C.RANDOM_FPR}, "
               f"EPS_TARGET={C.EPS_TARGET}.")
    out.append("")
    out.append(f"CHECKS PASSED: {len(_checks)}")
    out.append(f"FAILURES:      {len(_failures)}")
    out.append(f"NOTES:         {len(_notes)}")
    out.append("")
    out.append("=" * 78)
    out.append("FAILURES" if _failures else "FAILURES: none")
    out.append("=" * 78)
    for f in _failures:
        out.append(f"FAIL  {f}")
    out.append("")
    out.append("=" * 78)
    out.append("NOTES (values that are absent at source rather than wrong)")
    out.append("=" * 78)
    for n in _notes:
        out.append(f"NOTE  {n}")
    out.append("")
    out.append("=" * 78)
    out.append("CHECKED VALUES")
    out.append("<table> | <row> | <column> | <value> | <source file> | <source row/filter>")
    out.append("=" * 78)
    out.extend(_checks)
    (C.LOGS / "verification.txt").write_text("\n".join(out) + "\n")

    # Machine-readable copy for gaps.md assembly.
    (C.LOGS / "_verify_notes.json").write_text(
        json.dumps({"failures": _failures, "notes": _notes,
                    "n_checks": len(_checks)}, indent=2) + "\n")

    print(f"checks={len(_checks)}  failures={len(_failures)}  notes={len(_notes)}")
    print(f"-> {C.LOGS / 'verification.txt'}")
    for f in _failures[:20]:
        print("FAIL ", f)
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
