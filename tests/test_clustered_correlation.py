"""Inference tests for the six-dataset cross-sectional correlation design."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ANALYSIS_SRC = Path(__file__).resolve().parents[1] / "analysis" / "src"
sys.path.insert(0, str(ANALYSIS_SRC))

from common import clustered_spearman_with_ci, exact_spearman_with_ci  # noqa: E402


def test_six_point_exact_permutation_enumerates_all_assignments():
    x = np.arange(6, dtype=float)
    result = exact_spearman_with_ci(x, x, n_resamples=1_000, seed=7)

    assert result["rho"] == 1.0
    assert result["n"] == 6
    assert result["n_clusters"] == 6
    assert result["n_permutations"] == math.factorial(6)
    assert math.isclose(result["p"], 2 / math.factorial(6))
    assert result["p_method"] == "exact observation-level permutation"


def test_balanced_dataset_blocks_move_together():
    clusters = np.repeat([f"dataset-{i}" for i in range(6)], 5)
    models = np.tile(["RF", "LR", "GNB", "SVM", "DNN"], 6)
    x = np.arange(30, dtype=float)
    y = x + np.tile([0.0, 0.2, -0.1, 0.1, -0.2], 6)

    result = clustered_spearman_with_ci(
        x, y, clusters, models, n_resamples=1_000, seed=11)

    assert result["n"] == 30
    assert result["n_clusters"] == 6
    assert result["n_permutations"] == math.factorial(6)
    assert result["p_method"] == "exact dataset-block permutation"
    assert result["ci_method"].startswith("dataset-cluster percentile bootstrap")
    assert result["n_bootstrap_valid"] == 1_000
    assert -1 <= result["ci_low"] <= result["ci_high"] <= 1


def test_unbalanced_sensitivity_uses_cluster_robust_fallback():
    clusters = np.repeat([f"dataset-{i}" for i in range(6)], 5)
    models = np.tile(["RF", "LR", "GNB", "SVM", "DNN"], 6)
    keep = np.ones(30, dtype=bool)
    keep[0] = False
    x = np.arange(30, dtype=float)[keep]
    y = (np.arange(30, dtype=float) ** 1.1)[keep]

    result = clustered_spearman_with_ci(
        x, y, clusters[keep], models[keep], n_resamples=1_000, seed=13)

    assert result["n"] == 29
    assert result["n_clusters"] == 6
    assert result["n_permutations"] == 0
    assert result["p_method"].startswith("CR1 dataset-cluster rank regression")
    assert "unbalanced cluster strata" in result["note"]


def test_cluster_bootstrap_is_reproducible():
    clusters = np.repeat(np.arange(6), 5)
    models = np.tile(np.arange(5), 6)
    x = np.linspace(0, 1, 30)
    y = np.sin(x * 2.0)

    first = clustered_spearman_with_ci(
        x, y, clusters, models, n_resamples=1_000, seed=17)
    second = clustered_spearman_with_ci(
        x, y, clusters, models, n_resamples=1_000, seed=17)

    assert first["ci_low"] == second["ci_low"]
    assert first["ci_high"] == second["ci_high"]
    assert first["p"] == second["p"]
