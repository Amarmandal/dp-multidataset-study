"""
Generate Standard-vs-DP comparison figures from shokri_mia_comparison.csv.

Produces, under Attack/MIA_Shokri/results/figures/:
  * <dataset>_advantage_vs_epsilon.png   attack advantage vs epsilon per model,
                                         with the Standard baseline as a dashed line
  * <dataset>_auc_vs_epsilon.png         attack AUC vs epsilon per model
  * std_vs_dp_advantage_bars.png         Standard vs best-privacy DP, all datasets
  * advantage_vs_gengap.png              attack advantage against generalisation gap

Usage:
    python plots.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, "results", "shokri_mia_comparison.csv")
FIG_DIR = os.path.join(HERE, "results", "figures")

MODELS = ["LR", "RF", "GNB", "SVM"]
COLORS = {"LR": "#1f77b4", "RF": "#d62728", "GNB": "#2ca02c", "SVM": "#9467bd"}


def load():
    df = pd.read_csv(CSV)
    df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")
    return df


def _per_dataset_curve(df, ds, metric, ylabel, fname):
    sub = df[df["dataset"] == ds]
    dp = sub[sub["variant"] == "dp"].sort_values("epsilon")
    std = sub[sub["variant"] == "standard"]
    if dp.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        d = dp[dp["model"] == m]
        if d.empty or d[f"{metric}_mean"].isna().all():
            continue
        ax.errorbar(d["epsilon"], d[f"{metric}_mean"], yerr=d[f"{metric}_std"],
                    marker="o", capsize=3, color=COLORS[m], label=f"DP-{m}")
        s = std[std["model"] == m]
        if not s.empty and not np.isnan(s[f"{metric}_mean"].values[0]):
            ax.axhline(s[f"{metric}_mean"].values[0], ls="--", lw=1,
                       color=COLORS[m], alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget  ε  (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ds}: {ylabel} vs ε  (dashed = Standard baseline)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close(fig)


def std_vs_dp_bars(df):
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4.5),
                             sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        models = [m for m in MODELS if m in sub["model"].unique()]
        std_vals, dp_vals = [], []
        for m in models:
            s = sub[(sub["model"] == m) & (sub["variant"] == "standard")]
            dp = sub[(sub["model"] == m) & (sub["variant"] == "dp")]
            std_vals.append(s["advantage_mean"].values[0] if not s.empty else 0)
            # strong-privacy DP = smallest epsilon
            dp_strong = dp.sort_values("epsilon").head(1)
            dp_vals.append(dp_strong["advantage_mean"].values[0] if not dp_strong.empty else 0)
        x = np.arange(len(models))
        ax.bar(x - 0.2, std_vals, 0.4, label="Standard", color="#d62728")
        ax.bar(x + 0.2, dp_vals, 0.4, label="DP (ε=min)", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(ds, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Membership advantage (TPR − FPR)")
    axes[-1].legend()
    fig.suptitle("Membership Inference: Standard vs strong-privacy DP", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "std_vs_dp_advantage_bars.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def advantage_vs_gengap(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, marker, label in [("standard", "o", "Standard"), ("dp", "x", "DP")]:
        s = df[df["variant"] == variant]
        ax.scatter(s["gen_gap_mean"], s["advantage_mean"], marker=marker,
                   alpha=0.6, label=label)
    ax.set_xlabel("Generalisation gap (train acc − test acc)")
    ax.set_ylabel("Membership advantage (TPR − FPR)")
    ax.set_title("Attack success tracks overfitting")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "advantage_vs_gengap.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load()
    for ds in df["dataset"].unique():
        _per_dataset_curve(df, ds, "advantage", "Membership advantage",
                           f"{ds}_advantage_vs_epsilon.png")
        _per_dataset_curve(df, ds, "attack_auc", "Attack AUC",
                           f"{ds}_auc_vs_epsilon.png")
    std_vs_dp_bars(df)
    advantage_vs_gengap(df)
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
