"""Shared loading + correlation helpers for the RQ1 correlation analyses.

Authoritative inputs ONLY:
  Results/dataset_results/consolidated_data.csv    (utility; AL = `ACL`)
  Results/attack_results/consolidated_mia_data.csv (leakage)

Nothing here drops, winsorises, clips or floors a value. Negative Yeom
advantages are real and are carried through unmodified.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
UTILITY_CSV = REPO / "Results" / "dataset_results" / "consolidated_data.csv"
MIA_CSV = REPO / "Results" / "attack_results" / "consolidated_mia_data.csv"

# Paired nine-point grid (intersection of the utility sweep and the attack grid).
EPSILONS = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0, 10.0]

# utility model name -> attack model name
MODEL_MAP = {
    "DP-RF": "RF",
    "DP-LR": "LR",
    "DP-GNB": "GNB",
    "DP-SVM": "SVM",
    "DP-DNN": "DNN",
}

# (attack, column, label) — primary first.
LEAKAGE_METRICS = [
    ("yeom", "advantage_mean", "yeom_advantage"),
    ("lira", "attack_auc_mean", "lira_auc"),
    ("shokri", "attack_auc_mean", "shokri_auc"),
    ("lira", "tpr_at_1pct", "lira_tpr_at_1pct"),
]
PRIMARY_METRIC = "yeom_advantage"


def load_utility() -> pd.DataFrame:
    return pd.read_csv(UTILITY_CSV)


def load_mia() -> pd.DataFrame:
    return pd.read_csv(MIA_CSV)


def build_paired(utility: pd.DataFrame, mia: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, model, epsilon) with AL and every leakage metric.

    Join keys: dataset + model + epsilon, both sides filtered to variant=='dp'
    and to the nine-point grid. Model names normalised via MODEL_MAP.
    """
    u = utility[utility["variant"] == "dp"].copy()
    u = u[u["epsilon"].isin(EPSILONS)]
    u["model_key"] = u["model"].map(MODEL_MAP)
    if u["model_key"].isna().any():
        bad = sorted(u.loc[u["model_key"].isna(), "model"].unique())
        raise ValueError(f"unmapped utility model names: {bad}")

    base = u[["dataset", "dataset_dir", "model", "model_key", "epsilon",
              "accuracy_mean", "balanced_accuracy_mean", "ACL", "n_runs"]].rename(
        columns={"ACL": "AL", "n_runs": "utility_n_runs"}
    )
    # eps stored as float on both sides; round to kill any float-repr mismatch
    base["epsilon"] = base["epsilon"].round(6)

    a = mia[mia["variant"] == "dp"].copy()
    a = a[a["epsilon"].isin(EPSILONS)]
    a["epsilon"] = a["epsilon"].round(6)

    out = base
    for attack, col, label in LEAKAGE_METRICS:
        sub = a[a["attack"] == attack][["dataset", "model", "epsilon", col]].rename(
            columns={"model": "model_key", col: label}
        )
        dupes = sub.duplicated(subset=["dataset", "model_key", "epsilon"]).sum()
        if dupes:
            raise ValueError(f"{attack}/{col}: {dupes} duplicate join keys")
        out = out.merge(sub, on=["dataset", "model_key", "epsilon"], how="left")

    return out.sort_values(["dataset", "model", "epsilon"]).reset_index(drop=True)


def spearman_with_ci(x, y, alpha: float = 0.05):
    """Spearman rho with p-value and Fisher-z CI using the Bonett-Wright SE.

    SE_z = sqrt((1 + rho^2 / 2) / (n - 3)), the standard-error correction for
    Spearman's rho (Bonett & Wright 2000). Returns a dict; `note` is set
    whenever rho is undefined or the CI cannot be formed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(mask.sum())

    res = {"rho": np.nan, "n": n, "p": np.nan,
           "ci_low": np.nan, "ci_high": np.nan, "note": ""}

    if n < 3:
        res["note"] = f"undefined: only {n} paired points (<3)"
        return res
    if np.all(x == x[0]) and np.all(y == y[0]):
        res["note"] = "undefined: both series constant"
        return res
    if np.all(x == x[0]):
        res["note"] = "undefined: AL series constant (zero variance)"
        return res
    if np.all(y == y[0]):
        res["note"] = "undefined: leakage series constant (zero variance)"
        return res

    rho, p = stats.spearmanr(x, y)
    if not np.isfinite(rho):
        res["note"] = "undefined: spearmanr returned nan"
        return res
    res["rho"], res["p"] = float(rho), float(p)

    if n <= 3:
        res["note"] = "CI undefined: n <= 3 (Bonett-Wright SE needs n > 3)"
        return res
    if abs(rho) >= 1.0:
        # rho == +/-1 -> Fisher z is infinite; CI degenerates to the point itself.
        res["ci_low"] = res["ci_high"] = float(np.sign(rho))
        res["note"] = "perfect monotone rank agreement; CI degenerate at rho"
        return res

    z = math.atanh(rho)
    se = math.sqrt((1.0 + rho * rho / 2.0) / (n - 3))
    crit = stats.norm.ppf(1.0 - alpha / 2.0)
    res["ci_low"] = float(math.tanh(z - crit * se))
    res["ci_high"] = float(math.tanh(z + crit * se))
    return res


def summarise(rhos: pd.Series, ps: pd.Series) -> dict:
    """Median / IQR / min / max of computable rhos + significance split by sign."""
    ok = rhos.notna()
    r = rhos[ok]
    p = ps[ok]
    sig = p < 0.05
    q1, q3 = (np.nanpercentile(r, [25, 75]) if len(r) else (np.nan, np.nan))
    return {
        "n_pairs_total": int(len(rhos)),
        "n_computable": int(len(r)),
        "n_undefined": int((~ok).sum()),
        "median_rho": float(np.median(r)) if len(r) else np.nan,
        "iqr_low": float(q1),
        "iqr_high": float(q3),
        "min_rho": float(r.min()) if len(r) else np.nan,
        "max_rho": float(r.max()) if len(r) else np.nan,
        "n_significant_p05": int(sig.sum()),
        "n_significant_negative": int((sig & (r < 0)).sum()),
        "n_significant_positive": int((sig & (r > 0)).sum()),
    }
