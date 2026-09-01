"""Shared loading + correlation helpers for the RQ1 correlation analyses.

Authoritative inputs ONLY:
  Results/dataset_results/consolidated_data.csv
      (primary utility: AL_exported = `ACL_exported`;
       sensitivity utility: AL_mean = `ACL`)
  Results/attack_results/consolidated_mia_data.csv (leakage)

Nothing here drops, winsorises, clips or floors a value. Negative Yeom
advantages are real and are carried through unmodified.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import permutations
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

# Cross-sectional inference treats the six clinical datasets as the independent
# sampling units.  The seed is fixed so every generated table and figure is
# exactly reproducible.
CLUSTER_BOOTSTRAP_RESAMPLES = 20_000
INFERENCE_RANDOM_SEED = 20260901


def load_utility() -> pd.DataFrame:
    return pd.read_csv(UTILITY_CSV)


def load_mia() -> pd.DataFrame:
    return pd.read_csv(MIA_CSV)


def build_paired(utility: pd.DataFrame, mia: pd.DataFrame) -> pd.DataFrame:
    """One row per key with artifact-matched AL and every leakage metric.

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
              "accuracy_mean", "balanced_accuracy_mean", "ACL",
              "ACL_exported", "n_runs"]].rename(
        columns={"ACL": "AL_mean", "ACL_exported": "AL_exported",
                 "n_runs": "utility_n_runs"}
    )
    if base["AL_exported"].isna().any():
        missing = base.loc[base["AL_exported"].isna(),
                           ["dataset", "model", "epsilon"]]
        raise ValueError(f"missing artifact-matched ACL_exported rows:\n{missing}")
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


def _factorize_in_order(values: np.ndarray) -> np.ndarray:
    """Return stable integer codes without sorting potentially mixed labels."""
    mapping = {}
    codes = []
    for value in values.tolist():
        if value not in mapping:
            mapping[value] = len(mapping)
        codes.append(mapping[value])
    return np.asarray(codes, dtype=np.int16)


@lru_cache(maxsize=16)
def _cluster_bootstrap_plan(
    cluster_codes: tuple[int, ...], n_resamples: int, seed: int
) -> np.ndarray:
    """Indices for a pairs cluster bootstrap, padded with -1 when unbalanced."""
    codes = np.asarray(cluster_codes, dtype=int)
    unique = np.unique(codes)
    groups = [np.flatnonzero(codes == code) for code in unique]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(n_resamples, len(groups)))
    max_len = len(groups) * max(len(group) for group in groups)
    plan = np.full((n_resamples, max_len), -1, dtype=np.int32)
    for row, draw in enumerate(draws):
        sampled = np.concatenate([groups[i] for i in draw])
        plan[row, :len(sampled)] = sampled
    return plan


def _rowwise_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised Spearman correlations for matrices with optional NaN padding."""
    rx = stats.rankdata(x, axis=1, nan_policy="omit")
    ry = stats.rankdata(y, axis=1, nan_policy="omit")
    mx = np.nanmean(rx, axis=1, keepdims=True)
    my = np.nanmean(ry, axis=1, keepdims=True)
    dx, dy = rx - mx, ry - my
    denom = np.sqrt(np.nansum(dx * dx, axis=1) * np.nansum(dy * dy, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nansum(dx * dy, axis=1) / denom


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    *,
    alpha: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, int]:
    """Percentile CI from resampling whole datasets with replacement."""
    codes = _factorize_in_order(clusters)
    plan = _cluster_bootstrap_plan(tuple(codes.tolist()), n_resamples, seed)
    values = []
    # Batching bounds temporary memory for the 270-row pooled sensitivity.
    for start in range(0, len(plan), 2_000):
        idx = plan[start:start + 2_000]
        valid = idx >= 0
        safe = np.where(valid, idx, 0)
        xb = np.where(valid, x[safe], np.nan)
        yb = np.where(valid, y[safe], np.nan)
        values.append(_rowwise_spearman(xb, yb))
    boot = np.concatenate(values)
    boot = boot[np.isfinite(boot)]
    if not len(boot):
        return np.nan, np.nan, 0
    low, high = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high), int(len(boot))


def _cluster_robust_rank_p(
    x: np.ndarray, y: np.ndarray, clusters: np.ndarray
) -> float:
    """CR1 cluster-robust test of the slope between the two rank variables."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    design = np.column_stack([np.ones(len(rx)), rx])
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ ry
    residual = ry - design @ beta
    meat = np.zeros((2, 2), dtype=float)
    unique = list(dict.fromkeys(clusters.tolist()))
    for cluster in unique:
        mask = clusters == cluster
        score = design[mask].T @ residual[mask]
        meat += np.outer(score, score)
    n, k, g = len(x), design.shape[1], len(unique)
    correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = correction * bread @ meat @ bread
    se = math.sqrt(max(float(covariance[1, 1]), 0.0))
    if se == 0.0:
        return 0.0 if beta[1] != 0 else 1.0
    statistic = float(beta[1] / se)
    return float(2.0 * stats.t.sf(abs(statistic), df=g - 1))


def _balanced_cluster_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    strata: np.ndarray,
) -> tuple[float, int]:
    """Exact dataset-block permutation p-value, aligned by model/budget stratum."""
    cluster_order = list(dict.fromkeys(clusters.tolist()))
    stratum_sets = []
    rows = {}
    for cluster in cluster_order:
        mask = clusters == cluster
        cluster_strata = strata[mask].tolist()
        if len(cluster_strata) != len(set(cluster_strata)):
            raise ValueError(f"duplicate stratum inside cluster {cluster!r}")
        stratum_sets.append(set(cluster_strata))
        for xv, yv, stratum in zip(x[mask], y[mask], strata[mask]):
            rows[(cluster, stratum)] = (float(xv), float(yv))
    if any(s != stratum_sets[0] for s in stratum_sets[1:]):
        raise ValueError("exact cluster permutation requires identical strata in every cluster")
    stratum_order = sorted(stratum_sets[0], key=str)
    x_ordered = np.asarray([
        rows[(cluster, stratum)][0]
        for cluster in cluster_order for stratum in stratum_order
    ])
    y_observed = np.asarray([
        rows[(cluster, stratum)][1]
        for cluster in cluster_order for stratum in stratum_order
    ])
    observed = float(stats.spearmanr(x_ordered, y_observed).statistic)
    permuted = []
    for source_order in permutations(cluster_order):
        yp = np.asarray([
            rows[(source, stratum)][1]
            for source in source_order for stratum in stratum_order
        ])
        permuted.append(yp)
    yp = np.vstack(permuted)
    xp = np.broadcast_to(x_ordered, yp.shape)
    null = _rowwise_spearman(xp, yp)
    p = np.mean(np.abs(null) >= abs(observed) - 1e-12)
    return float(p), int(len(null))


def clustered_spearman_with_ci(
    x,
    y,
    clusters,
    strata=None,
    *,
    alpha: float = 0.05,
    n_resamples: int = CLUSTER_BOOTSTRAP_RESAMPLES,
    seed: int = INFERENCE_RANDOM_SEED,
):
    """Spearman rho with dataset-cluster-aware p-value and percentile CI.

    For the balanced study grid, the p-value exactly permutes all six outcome
    dataset blocks (6! = 720 assignments) while retaining model/budget strata.
    An unbalanced sensitivity subset uses a CR1 rank-regression test with
    ``G-1`` degrees of freedom.  In both cases the 95% interval resamples whole
    datasets, never individual dataset-model rows.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters, dtype=object)
    if strata is None:
        strata = np.asarray(["__row__"] * len(x), dtype=object)
    else:
        strata = np.asarray(strata, dtype=object)
    if not (len(x) == len(y) == len(clusters) == len(strata)):
        raise ValueError("x, y, clusters and strata must have equal lengths")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, clusters, strata = x[mask], y[mask], clusters[mask], strata[mask]
    n = int(len(x))
    n_clusters = len(set(clusters.tolist()))
    result = {
        "rho": np.nan, "n": n, "n_clusters": n_clusters, "p": np.nan,
        "ci_low": np.nan, "ci_high": np.nan,
        "p_method": "n/a", "ci_method": "n/a",
        "n_permutations": 0, "n_bootstrap_valid": 0, "note": "",
    }
    if n < 3 or n_clusters < 2:
        result["note"] = f"undefined: n={n}, dataset clusters={n_clusters}"
        return result
    if np.all(x == x[0]) or np.all(y == y[0]):
        result["note"] = "undefined: at least one series is constant"
        return result
    result["rho"] = float(stats.spearmanr(x, y).statistic)
    low, high, valid = _bootstrap_ci(
        x, y, clusters, alpha=alpha, n_resamples=n_resamples, seed=seed)
    result.update({
        "ci_low": low, "ci_high": high, "n_bootstrap_valid": valid,
        "ci_method": f"dataset-cluster percentile bootstrap ({n_resamples} resamples)",
    })

    cluster_strata = [set(strata[clusters == c].tolist())
                      for c in dict.fromkeys(clusters.tolist())]
    balanced = (all(s == cluster_strata[0] for s in cluster_strata[1:])
                and all(len(strata[clusters == c]) == len(cluster_strata[0])
                        for c in dict.fromkeys(clusters.tolist())))
    if balanced:
        p, n_perm = _balanced_cluster_permutation_p(x, y, clusters, strata)
        result.update({
            "p": p, "n_permutations": n_perm,
            "p_method": "exact dataset-block permutation",
        })
    else:
        result.update({
            "p": _cluster_robust_rank_p(x, y, clusters),
            "p_method": "CR1 dataset-cluster rank regression (t, df=G-1)",
            "note": "unbalanced cluster strata; exact block permutation unavailable",
        })
    return result


def exact_spearman_with_ci(
    x,
    y,
    *,
    alpha: float = 0.05,
    n_resamples: int = CLUSTER_BOOTSTRAP_RESAMPLES,
    seed: int = INFERENCE_RANDOM_SEED,
):
    """Exact permutation p-value and observation-bootstrap CI for small n."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    result = clustered_spearman_with_ci(
        x, y, np.arange(n), np.repeat("__single__", n),
        alpha=alpha, n_resamples=n_resamples, seed=seed)
    result["p_method"] = "exact observation-level permutation"
    result["ci_method"] = (
        f"observation-level percentile bootstrap ({n_resamples} resamples)")
    return result


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
