"""
Generate cross-dataset Yeom-MIA comparison figures.

Reads the per-dataset MIA result CSVs written by ``run_mia.py`` and writes a set
of colour-synced comparison plots into ``Attack/MIA/comparison/`` — the
average-case counterpart to ``Attack/LiRA/make_comparison_figures.py``.

Headline metric throughout: membership advantage (TPR − FPR; no-leak = 0.0).
Attack AUC (baseline 0.5) is shown alongside.

Usage:
    cd Attack/MIA && python3 make_comparison_figures.py
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "comparison")
os.makedirs(OUT, exist_ok=True)

# Palettes identical to run_mia.py / run_lira.py so every study figure matches.
MODEL_COLORS = {"LR": "#1f77b4", "RF": "#d62728", "GNB": "#2ca02c",
                "SVM": "#9467bd", "DNN": "#ff7f0e"}
MODELS = ["LR", "RF", "GNB", "SVM", "DNN"]

DATASETS = ["GALLSTONE", "BCP", "CANCER_RISK", "KIDNEY_STONE",
            "LUNG_CANCER", "DIABETES"]
DS_LABEL = {"GALLSTONE": "GALLSTONE\n(n=255)",
            "BCP": "BCP\n(n=455)",
            "CANCER_RISK": "CANCER_RISK\n(n=1,200)",
            "KIDNEY_STONE": "KIDNEY_STONE\n(n=3,200)",
            "LUNG_CANCER": "LUNG_CANCER\n(n=40,000)",
            "DIABETES": "DIABETES\n(n=56,553)"}
DS_COLORS = {"GALLSTONE": "#e41a1c", "BCP": "#4daf4a",
             "CANCER_RISK": "#ff7f00", "KIDNEY_STONE": "#377eb8",
             "LUNG_CANCER": "#984ea3", "DIABETES": "#a65628"}
N_TRAIN = {"GALLSTONE": 255, "BCP": 455, "CANCER_RISK": 1200,
           "KIDNEY_STONE": 3200, "LUNG_CANCER": 40000, "DIABETES": 56553}

BASELINE_ADV = 0.0   # no leakage
BASELINE_AUC = 0.5

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})


def load_all():
    frames = []
    for ds in DATASETS:
        p = os.path.join(RESULTS, ds, f"{ds}_mia_results.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit("No per-dataset MIA CSVs found. Run run_mia.py first.")
    df = pd.concat(frames, ignore_index=True)
    df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    return df


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")


# 1. Standard-model leakage: grouped bars, models x datasets
def fig_standard_leakage_bars(df):
    std = df[df["variant"] == "standard"]
    x = np.arange(len(DATASETS))
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, m in enumerate(MODELS):
        vals = []
        for ds in DATASETS:
            r = std[(std["dataset"] == ds) & (std["model"] == m)]
            vals.append(r["advantage"].values[0] if not r.empty else np.nan)
        offset = (i - (len(MODELS) - 1) / 2) * width
        bars = ax.bar(x + offset, np.nan_to_num(vals), width,
                      color=MODEL_COLORS[m], label=m, edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            if not np.isnan(v) and v > 0.05:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(BASELINE_ADV, ls=":", color="grey", lw=1.3, label="no leakage (0.0)")
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("Membership advantage (TPR − FPR)")
    ax.set_title("Standard (non-private) models under Yeom's MIA — average-case leakage")
    ax.legend(ncol=6, fontsize=9, loc="upper right")
    save(fig, "01_standard_leakage_bars.png")


# 2. Dataset-size effect: Standard advantage vs n_train (log x), per model
def fig_size_effect(df):
    std = df[df["variant"] == "standard"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in MODELS:
        xs, ys = [], []
        for ds in DATASETS:
            r = std[(std["dataset"] == ds) & (std["model"] == m)]
            if not r.empty and np.isfinite(r["advantage"].values[0]):
                xs.append(N_TRAIN[ds]); ys.append(r["advantage"].values[0])
        if xs:
            ax.plot(xs, ys, marker="o", ms=8, lw=2, color=MODEL_COLORS[m], label=m)
    ax.axhline(BASELINE_ADV, ls=":", color="grey", lw=1.3, label="no leakage (0.0)")
    ax.set_xscale("log")
    ax.set_xlabel("Training-set size  n  (log scale)")
    ax.set_ylabel("Membership advantage  (Standard models)")
    ax.set_title("Dataset size is the dominant privacy variable\n"
                 "Even without DP, average-case leakage decays toward 0 as n grows")
    ax.set_xticks(list(N_TRAIN.values()))
    ax.set_xticklabels([f"{v:,}" for v in N_TRAIN.values()])
    ax.legend(fontsize=9)
    save(fig, "02_dataset_size_effect.png")


# 3. advantage vs epsilon for DP-RF across datasets
def fig_rf_adv_vs_eps(df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for ds in DATASETS:
        d = df[(df["dataset"] == ds) & (df["model"] == "RF") & (df["variant"] == "dp")]
        d = d.sort_values("epsilon")
        if d.empty:
            continue
        ax.plot(d["epsilon"], d["advantage"], marker="o", lw=2,
                color=DS_COLORS[ds], label=DS_LABEL[ds].replace("\n", " "))
        s = df[(df["dataset"] == ds) & (df["model"] == "RF") & (df["variant"] == "standard")]
        if not s.empty:
            ax.axhline(s["advantage"].values[0], ls="--", lw=1,
                       color=DS_COLORS[ds], alpha=0.5)
    ax.axhline(BASELINE_ADV, ls=":", color="grey", lw=1.3, label="no leakage (0.0)")
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget  ε  (log scale)")
    ax.set_ylabel("Membership advantage  (DP-RF)")
    ax.set_title("DP-Random Forest advantage vs ε across datasets\n"
                 "Dashed = same-dataset Standard RF baseline")
    ax.legend(fontsize=9)
    save(fig, "03_dp_rf_advantage_vs_epsilon.png")


# 4. Attack-AUC heatmap: Standard models, models x datasets
def fig_auc_heatmap(df):
    std = df[df["variant"] == "standard"]
    grid = np.full((len(MODELS), len(DATASETS)), np.nan)
    for i, m in enumerate(MODELS):
        for j, ds in enumerate(DATASETS):
            r = std[(std["dataset"] == ds) & (std["model"] == m)]
            if not r.empty:
                grid[i, j] = r["attack_auc"].values[0]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=0.40, vmax=0.60, aspect="auto")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    for i in range(len(MODELS)):
        for j in range(len(DATASETS)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if (v > 0.565 or v < 0.435) else "black")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Attack AUC  (0.5 = random)")
    ax.set_title("Standard-model attack AUC (Yeom, average-case view)")
    ax.grid(False)
    save(fig, "04_standard_auc_heatmap.png")


def main():
    df = load_all()
    print(f"Loaded {len(df)} target rows from {df['dataset'].nunique()} datasets.")
    fig_standard_leakage_bars(df)
    fig_size_effect(df)
    fig_rf_adv_vs_eps(df)
    fig_auc_heatmap(df)
    print(f"\nAll figures written to {os.path.relpath(OUT, HERE)}/")


if __name__ == "__main__":
    main()
