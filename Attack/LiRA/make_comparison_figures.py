"""
Generate cross-dataset LiRA comparison figures.

Reads the per-dataset LiRA result CSVs and writes a set of clear, colour-synced
comparison plots into Attack/LiRA/comparison/.

Headline metric throughout: TPR @ 1% FPR (random baseline = 0.01) — the
worst-case membership leakage. Attack AUC (baseline 0.5) is shown alongside.

Usage:
    cd Attack/LiRA && python3 make_comparison_figures.py
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "comparison")
os.makedirs(OUT, exist_ok=True)

# ---- consistent palettes -------------------------------------------------- #
# Model colours: identical to run_lira.py so every figure in the study matches.
MODEL_COLORS = {"LR": "#1f77b4", "RF": "#d62728", "GNB": "#2ca02c",
                "SVM": "#9467bd", "DNN": "#ff7f0e"}
MODELS = ["LR", "RF", "GNB", "SVM", "DNN"]

# Dataset colours + display order (small -> large n_train).
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

BASELINE_TPR = 0.01   # random TPR @ 1% FPR
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
        p = os.path.join(RESULTS, ds, f"{ds}_lira_results.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    return df


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")


# --------------------------------------------------------------------------- #
# 1. Standard-model worst-case leakage: grouped bars, models x datasets
# --------------------------------------------------------------------------- #
def fig_standard_leakage_bars(df):
    std = df[df["variant"] == "standard"]
    x = np.arange(len(DATASETS))
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, m in enumerate(MODELS):
        vals = []
        for ds in DATASETS:
            r = std[(std["dataset"] == ds) & (std["model"] == m)]
            vals.append(r["tpr_at_1pct"].values[0] if not r.empty else np.nan)
        offset = (i - (len(MODELS) - 1) / 2) * width
        bars = ax.bar(x + offset, np.nan_to_num(vals), width,
                      color=MODEL_COLORS[m], label=m, edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            if not np.isnan(v) and v > 0.05:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(BASELINE_TPR, ls=":", color="grey", lw=1.3, label="random (0.01)")
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("TPR @ 1% FPR  (worst-case leakage)")
    ax.set_title("Standard (non-private) models under LiRA — worst-case membership leakage\n"
                 "Only Random Forest on the smallest dataset leaks; everything else sits near random")
    ax.legend(ncol=6, fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.05)
    save(fig, "01_standard_leakage_bars.png")


# --------------------------------------------------------------------------- #
# 2. Dataset-size effect: Standard TPR@1% vs n_train (log x), per model
# --------------------------------------------------------------------------- #
def fig_size_effect(df):
    std = df[df["variant"] == "standard"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in MODELS:
        xs, ys = [], []
        for ds in DATASETS:
            r = std[(std["dataset"] == ds) & (std["model"] == m)]
            if not r.empty and np.isfinite(r["tpr_at_1pct"].values[0]):
                xs.append(N_TRAIN[ds]); ys.append(r["tpr_at_1pct"].values[0])
        if xs:
            ax.plot(xs, ys, marker="o", ms=8, lw=2, color=MODEL_COLORS[m], label=m)
    ax.axhline(BASELINE_TPR, ls=":", color="grey", lw=1.3, label="random (0.01)")
    ax.set_xscale("log")
    ax.set_xlabel("Training-set size  n  (log scale)")
    ax.set_ylabel("TPR @ 1% FPR  (Standard models)")
    ax.set_title("Dataset size is the dominant privacy variable\n"
                 "Even without DP, worst-case leakage decays toward the random baseline as n grows")
    ax.set_xticks(list(N_TRAIN.values()))
    ax.set_xticklabels([f"{v:,}" for v in N_TRAIN.values()])
    ax.legend(fontsize=9)
    save(fig, "02_dataset_size_effect.png")


# --------------------------------------------------------------------------- #
# 3. Standard vs strongest-DP, GALLSTONE (the only dataset with real leakage)
# --------------------------------------------------------------------------- #
def fig_gallstone_std_vs_dp(df):
    g = df[df["dataset"] == "GALLSTONE"]
    models = [m for m in MODELS if m in g["model"].unique()]
    std_v, dp_v = [], []
    for m in models:
        s = g[(g["model"] == m) & (g["variant"] == "standard")]
        dpm = g[(g["model"] == m) & (g["variant"] == "dp")]
        std_v.append(s["tpr_at_1pct"].values[0] if not s.empty else 0)
        dp_v.append(dpm["tpr_at_1pct"].min() if not dpm.empty else 0)  # best protection
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    b1 = ax.bar(x - w / 2, std_v, w, color="#d62728", label="Standard (ε=∞)",
                edgecolor="white")
    b2 = ax.bar(x + w / 2, dp_v, w, color="#1f77b4", label="DP (best ε)",
                edgecolor="white")
    for bars, vals in [(b1, std_v), (b2, dp_v)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.axhline(BASELINE_TPR, ls=":", color="grey", lw=1.3, label="random (0.01)")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("TPR @ 1% FPR")
    ax.set_title("GALLSTONE (n=255): DP collapses the catastrophic RF leakage\n"
                 "Standard RF = 0.953 (memorisation) → DP RF ≈ noise floor")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    save(fig, "03_gallstone_std_vs_dp.png")


# --------------------------------------------------------------------------- #
# 4. TPR@1% vs epsilon for DP-RF across datasets (does DP curve differ by size?)
# --------------------------------------------------------------------------- #
def fig_rf_tpr_vs_eps(df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for ds in DATASETS:
        d = df[(df["dataset"] == ds) & (df["model"] == "RF") & (df["variant"] == "dp")]
        d = d.sort_values("epsilon")
        if d.empty:
            continue
        ax.plot(d["epsilon"], d["tpr_at_1pct"], marker="o", lw=2,
                color=DS_COLORS[ds], label=DS_LABEL[ds].replace("\n", " "))
        # mark the Standard baseline for this dataset
        s = df[(df["dataset"] == ds) & (df["model"] == "RF") & (df["variant"] == "standard")]
        if not s.empty:
            ax.axhline(s["tpr_at_1pct"].values[0], ls="--", lw=1,
                       color=DS_COLORS[ds], alpha=0.5)
    ax.axhline(BASELINE_TPR, ls=":", color="grey", lw=1.3, label="random (0.01)")
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget  ε  (log scale)")
    ax.set_ylabel("TPR @ 1% FPR  (DP-RF)")
    ax.set_title("DP-Random Forest leakage vs ε across datasets\n"
                 "Dashed = same-dataset Standard RF baseline; DP holds leakage at the floor for every ε")
    ax.legend(fontsize=9)
    save(fig, "04_dp_rf_tpr_vs_epsilon.png")


# --------------------------------------------------------------------------- #
# 5. Attack-AUC heatmap: Standard models, models x datasets
# --------------------------------------------------------------------------- #
def fig_auc_heatmap(df):
    std = df[df["variant"] == "standard"]
    grid = np.full((len(MODELS), len(DATASETS)), np.nan)
    for i, m in enumerate(MODELS):
        for j, ds in enumerate(DATASETS):
            r = std[(std["dataset"] == ds) & (std["model"] == m)]
            if not r.empty:
                grid[i, j] = r["attack_auc"].values[0]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # diverging map centred on 0.5 (random)
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
    ax.set_title("Standard-model attack AUC (LiRA, average-case view)\n"
                 "Note RF on GALLSTONE (0.998) — total memorisation invisible to weaker attacks")
    ax.grid(False)
    save(fig, "05_standard_auc_heatmap.png")


# --------------------------------------------------------------------------- #
# 6. Privacy vs utility for GALLSTONE-RF: TPR@1% and test-acc vs epsilon
# --------------------------------------------------------------------------- #
def fig_gallstone_rf_privacy_utility(df):
    g = df[(df["dataset"] == "GALLSTONE") & (df["model"] == "RF")]
    dp = g[g["variant"] == "dp"].sort_values("epsilon")
    std = g[g["variant"] == "standard"]
    if dp.empty:
        return
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    c_leak, c_util = "#d62728", "#1f77b4"
    ax1.plot(dp["epsilon"], dp["tpr_at_1pct"], marker="o", lw=2, color=c_leak,
             label="DP-RF leakage (TPR@1%)")
    if not std.empty:
        ax1.scatter([12], std["tpr_at_1pct"].values[0], color=c_leak, marker="*",
                    s=220, zorder=5, label="Standard RF leakage")
    ax1.axhline(BASELINE_TPR, ls=":", color="grey", lw=1.3)
    ax1.set_xscale("log")
    ax1.set_xlabel("Privacy budget  ε  (log scale; ★ = ε=∞ Standard)")
    ax1.set_ylabel("TPR @ 1% FPR  (leakage, lower better)", color=c_leak)
    ax1.tick_params(axis="y", labelcolor=c_leak)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(dp["epsilon"], dp["test_acc"], marker="s", lw=2, ls="--", color=c_util,
             label="DP-RF test accuracy")
    if not std.empty:
        ax2.scatter([12], std["test_acc"].values[0], color=c_util, marker="*",
                    s=220, zorder=5)
    ax2.set_ylabel("Test accuracy (utility, higher better)", color=c_util)
    ax2.tick_params(axis="y", labelcolor=c_util)
    ax2.set_ylim(0, 1.05)
    ax2.grid(False)

    lines = [Line2D([], [], color=c_leak, marker="o", label="DP-RF leakage (TPR@1%)"),
             Line2D([], [], color=c_util, marker="s", ls="--", label="DP-RF test accuracy"),
             Line2D([], [], color="grey", ls=":", label="random leakage (0.01)"),
             Line2D([], [], color="k", marker="*", ls="", label="Standard (ε=∞)")]
    ax1.legend(handles=lines, fontsize=9, loc="center right")
    ax1.set_title("GALLSTONE Random Forest: the privacy–utility trade-off LiRA exposes\n"
                  "DP trades ~11 pts of accuracy to erase a 0.953 → ~0.01 memorisation leak")
    save(fig, "06_gallstone_rf_privacy_utility.png")


def main():
    df = load_all()
    print(f"Loaded {len(df)} target rows from {df['dataset'].nunique()} datasets.")
    fig_standard_leakage_bars(df)
    fig_size_effect(df)
    fig_gallstone_std_vs_dp(df)
    fig_rf_tpr_vs_eps(df)
    fig_auc_heatmap(df)
    fig_gallstone_rf_privacy_utility(df)
    print(f"\nAll figures written to {os.path.relpath(OUT, HERE)}/")


if __name__ == "__main__":
    main()
