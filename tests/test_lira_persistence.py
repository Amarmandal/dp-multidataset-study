"""Contracts for seed-level LiRA outcomes and reconstructible ROC data."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
LIRA_DIR = REPO / "Attack" / "LiRA"
sys.path.insert(0, str(LIRA_DIR))
sys.path.insert(0, str(REPO / "analysis" / "src"))

import lira  # noqa: E402
import run_lira as driver  # noqa: E402
import build_tables  # noqa: E402
import make_roc_grid  # noqa: E402


def test_low_fpr_operating_points_are_observed_integer_outcomes():
    member = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    score = np.array([0.95, 0.80, 0.70, 0.10, 0.90, 0.60, 0.30, 0.20])

    metrics = lira.attack_metrics(member, score)

    for label, target in lira.LOW_FPR_POINTS.items():
        prefix = f"op_{label}_"
        tp = metrics[prefix + "tp"]
        fp = metrics[prefix + "fp"]
        tn = metrics[prefix + "tn"]
        fn = metrics[prefix + "fn"]
        assert all(isinstance(value, int) for value in (tp, fp, tn, fn))
        assert tp + fn == metrics["n_members"]
        assert fp + tn == metrics["n_nonmembers"]
        assert metrics[prefix + "tpr"] == tp / (tp + fn)
        assert metrics[prefix + "fpr"] == fp / (fp + tn)
        assert metrics[prefix + "fpr"] <= target
        assert metrics[f"tpr_at_{label}"] == metrics[prefix + "tpr"]
        assert metrics[prefix + "selection_rule"] == (
            "max_tpr_with_empirical_fpr_le_target"
        )


def test_write_results_separates_summary_runs_and_roc(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "OUT_DIR", str(tmp_path))
    run = {
        "dataset": "TEST",
        "model": "LR",
        "variant": "standard",
        "epsilon": None,
        "run_seed": 0,
        "attack_auc": 0.75,
        "op_1pct_threshold": 0.8,
        "op_1pct_tp": 2,
        "op_1pct_fp": 0,
        "op_1pct_tn": 2,
        "op_1pct_fn": 0,
        "op_1pct_tpr": 1.0,
        "op_1pct_fpr": 0.0,
        "_roc_fpr": [0.0, 0.0, 1.0],
        "_roc_tpr": [0.0, 1.0, 1.0],
        "_roc_threshold": [np.inf, 0.8, 0.1],
    }
    summary = {
        "dataset": "TEST",
        "model": "LR",
        "variant": "standard",
        "epsilon": None,
        "n_runs": 1,
        "attack_auc": 0.75,
        "_roc_fpr": run["_roc_fpr"],
        "_roc_tpr": run["_roc_tpr"],
        "_roc_threshold": run["_roc_threshold"],
    }

    driver.write_results("TEST", [summary], [run])

    out = tmp_path / "TEST"
    summary_df = pd.read_csv(out / "TEST_lira_results.csv")
    runs_df = pd.read_csv(out / "TEST_lira_runs.csv")
    roc_df = pd.read_csv(out / "TEST_lira_roc.csv.gz")
    assert not any(column.startswith("_roc") for column in summary_df.columns)
    assert not any(column.startswith("_roc") for column in runs_df.columns)
    assert runs_df.loc[0, "op_1pct_tp"] == 2
    assert list(roc_df["point_index"]) == [0, 1, 2]
    assert list(roc_df["fpr"]) == [0.0, 0.0, 1.0]
    assert np.isinf(roc_df.loc[0, "threshold"])


def test_attack_runners_do_not_generate_reports_or_figures():
    runners = [
        REPO / "Attack" / "LiRA" / "run_lira.py",
        REPO / "Attack" / "MIA_YEOM" / "run_mia.py",
        REPO / "Attack" / "MIA_Shokri" / "run_mia.py",
    ]
    forbidden = ("analysis.md", "savefig(", "matplotlib", "build_report")
    for runner in runners:
        source = runner.read_text()
        assert not any(token in source for token in forbidden), runner


def test_low_fpr_uncertainty_uses_saved_run_counts(monkeypatch):
    runs = pd.DataFrame([
        {
            "dataset_dir": "BCP", "model": "LR", "variant": "dp",
            "epsilon": 1.0, "run_seed": 0,
            "op_1pct_target_fpr": 0.01, "op_1pct_selection_rule":
            "max_tpr_with_empirical_fpr_le_target",
            "op_1pct_tp": 1, "op_1pct_fp": 0,
            "op_1pct_tn": 100, "op_1pct_fn": 9,
            "op_1pct_tpr": 0.1, "op_1pct_fpr": 0.0,
            "op_1pct_fpr_resolvable": True,
        },
        {
            "dataset_dir": "BCP", "model": "LR", "variant": "dp",
            "epsilon": 1.0, "run_seed": 1,
            "op_1pct_target_fpr": 0.01, "op_1pct_selection_rule":
            "max_tpr_with_empirical_fpr_le_target",
            "op_1pct_tp": 3, "op_1pct_fp": 1,
            "op_1pct_tn": 99, "op_1pct_fn": 7,
            "op_1pct_tpr": 0.3, "op_1pct_fpr": 0.01,
            "op_1pct_fpr_resolvable": True,
        },
    ])
    monkeypatch.setattr(build_tables.L, "load_all_lira_runs", lambda: runs)

    row = build_tables._low_fpr_run_summary().iloc[0]

    assert row["lira_tpr_at_1pct"] == 0.2
    assert row["n_attack_runs"] == 2
    assert row["n_members_per_run"] == 10
    assert row["n_nonmembers_per_run"] == 100
    assert row["all_runs_fpr_resolvable"] == "yes"
    assert row["lira_tpr_at_1pct_ci_low"] <= 0.2
    assert row["lira_tpr_at_1pct_ci_high"] >= 0.2


def test_roc_grid_loader_reads_coordinates_not_images(tmp_path, monkeypatch):
    dataset = "BCP"
    result_dir = tmp_path / dataset
    result_dir.mkdir()
    pd.DataFrame({
        "dataset": [dataset, dataset, dataset],
        "model": ["LR", "LR", "LR"],
        "variant": ["standard", "standard", "standard"],
        "run_seed": [0, 0, 0],
        "point_index": [0, 1, 2],
        "threshold": [np.inf, 0.8, 0.1],
        "fpr": [0.0, 0.0, 1.0],
        "tpr": [0.0, 1.0, 1.0],
        "curve_selection": ["first_successful_standard_run"] * 3,
    }).to_csv(result_dir / f"{dataset}_lira_roc.csv.gz", index=False)
    monkeypatch.setattr(make_roc_grid, "LIRA_RESULTS", tmp_path)

    curves = make_roc_grid.load_curves(dataset)

    assert list(curves["point_index"]) == [0, 1, 2]
    assert not any(path.suffix == ".png" for path in tmp_path.rglob("*"))
