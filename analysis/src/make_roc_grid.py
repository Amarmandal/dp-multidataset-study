#!/usr/bin/env python3
"""Build the six-panel LiRA ROC figure directly from persisted curve data.

Run after LiRA has produced ``<DS>_lira_roc.csv.gz`` for all six datasets::

    python analysis/src/make_roc_grid.py

The PDF is genuinely vector: every curve is redrawn from FPR/TPR coordinates.
No attack-side PNG is read, copied, cropped, or embedded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIG = HERE.parent / "figures"
PNG_DIR = FIG / "png"
PDF_DIR = FIG / "pdf"
LIRA_RESULTS = REPO / "Attack" / "LiRA" / "results"

# Ordered by N so the grid reads small -> large.
PANELS = [
    ("GALLSTONE", "Gallstone Status", 319),
    ("BCP", "Breast Cancer Wisconsin", 569),
    ("CANCER_RISK", "Cancer Risk Prediction", 1_500),
    ("KIDNEY_STONE", "Kidney Stone Risk", 4_000),
    ("LUNG_CANCER", "Lung Cancer Risk Level", 50_000),
    ("DIABETES", "BRFSS 2015 Diabetes", 70_692),
]
LETTERS = "abcdef"
MODELS = ["LR", "RF", "GNB", "SVM", "DNN"]
COLORS = {
    "LR": "#1f77b4",
    "RF": "#d62728",
    "GNB": "#2ca02c",
    "SVM": "#9467bd",
    "DNN": "#ff7f0e",
}
REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "variant",
    "run_seed",
    "point_index",
    "threshold",
    "fpr",
    "tpr",
    "curve_selection",
}
LOWER_LIMIT = 1e-3
PNG_DPI = 600


def load_curves(dataset: str) -> pd.DataFrame:
    """Load and validate one dataset's representative standard-target curves."""
    path = LIRA_RESULTS / dataset / f"{dataset}_lira_roc.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; rerun LiRA for {dataset} before building the ROC grid"
        )

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: contains no ROC points")
    if set(frame["dataset"].astype(str)) != {dataset}:
        raise ValueError(f"{path}: dataset column does not contain only {dataset}")
    if set(frame["variant"].astype(str)) != {"standard"}:
        raise ValueError(f"{path}: ROC grid accepts standard-target curves only")
    if frame[["fpr", "tpr"]].isna().any().any():
        raise ValueError(f"{path}: FPR/TPR coordinates contain missing values")
    if not frame["fpr"].between(0, 1).all() or not frame["tpr"].between(0, 1).all():
        raise ValueError(f"{path}: FPR/TPR coordinates must lie in [0, 1]")

    for model, curve in frame.groupby("model", sort=False):
        if curve["run_seed"].nunique(dropna=False) != 1:
            raise ValueError(f"{path}: {model} contains more than one selected run")
        points = curve.sort_values("point_index")
        expected = np.arange(len(points))
        if not np.array_equal(points["point_index"].to_numpy(), expected):
            raise ValueError(f"{path}: {model} point_index is not contiguous from zero")
        if np.any(np.diff(points["fpr"].to_numpy(float)) < 0):
            raise ValueError(f"{path}: {model} FPR is not monotone")
    return frame


def draw_panel(ax, frame: pd.DataFrame) -> None:
    ax.plot(
        [LOWER_LIMIT, 1],
        [LOWER_LIMIT, 1],
        linestyle="--",
        color="grey",
        linewidth=1,
        label="random",
    )
    for model in MODELS:
        curve = frame[frame["model"] == model].sort_values("point_index")
        if curve.empty:
            continue
        ax.plot(
            np.clip(curve["fpr"].to_numpy(float), LOWER_LIMIT, 1),
            np.clip(curve["tpr"].to_numpy(float), LOWER_LIMIT, 1),
            color=COLORS[model],
            linewidth=1.4,
            label=model,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(LOWER_LIMIT, 1)
    ax.set_ylim(LOWER_LIMIT, 1)
    ax.grid(True, which="both", alpha=0.3)


def main() -> int:
    try:
        panels = {dataset: load_curves(dataset) for dataset, _, _ in PANELS}
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(10.7, 7.2), sharex=True, sharey=True)

    for ax, (dataset, label, n), letter in zip(axes.ravel(), PANELS, LETTERS):
        draw_panel(ax, panels[dataset])
        ax.set_title(f"({letter}) {label}  (N = {n:,})", fontsize=10.5, pad=5)

    for ax in axes[-1, :]:
        ax.set_xlabel("False positive rate (log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("True positive rate (log)")
    axes[0, 0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.7)

    pdf = PDF_DIR / "loglog_roc_grid.pdf"
    png = PNG_DIR / "loglog_roc_grid.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}  ({PNG_DPI} dpi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
