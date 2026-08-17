"""D-16 — RQ1 / RQ2 scatter figures pairing utility with leakage.

Every number is read from the two authoritative CSVs via ``common.py``:
  Results/dataset_results/consolidated_data.csv    (utility; AL = `ACL`)
  Results/attack_results/consolidated_mia_data.csv (leakage)

Nothing is dropped, clipped, floored or winsorised. Negative Yeom advantages
and sub-random AUCs are real and are plotted as they are.

Figures written to analysis/d16/ (both .pdf and .png):
  L4   baseline Yeom advantage      vs baseline LiRA AUC
  RQ1c AL at eps=1.0                vs DP residual LiRA TPR@1% at eps=1.0
  RQ2a baseline LiRA TPR@1%         vs DP leakage reduction at eps=1.0
  RQ2b baseline LiRA TPR@1% (log)   vs DP residual LiRA TPR@1% at eps=1.0
  L6   dataset size N (log)         vs standard-RF LiRA TPR@1%

RQ1c is emitted twice. `ACL` uses the 30-run mean; `ACL_exported` is the
accuracy of the exact run-0 artefact the attacks target (CLAUDE.md 12.3). Both
now carry all 30 pairs: the four DP-DNN rows that were blank (BCP, CANCER_RISK,
DIABETES, GALLSTONE) were recovered by scoring the exported artefacts directly
(Results/dataset_results/measure_exported_accuracy.py). Both rho values are
printed, and they still differ — the two columns are not interchangeable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import MODEL_MAP, load_mia, load_utility, spearman_with_ci  # noqa: E402

OUT = Path(__file__).resolve().parent / "d16"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "grid.linewidth": 0.4, "xtick.direction": "in", "ytick.direction": "in",
    "legend.framealpha": 0.9, "legend.edgecolor": "#cccccc",
    "mathtext.fontset": "cm",
})

MCOLOR = {"RF": "#009E73", "LR": "#D55E00", "GNB": "#CC79A7",
          "SVM": "#0072B2", "DNN": "#E69F00"}
MODELS = ["RF", "LR", "GNB", "SVM", "DNN"]
EPS_TARGET = 1.0


# --------------------------------------------------------------------------
# Data assembly — one row per (dataset, model): 30 pairs
# --------------------------------------------------------------------------
def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    u, m = load_utility(), load_mia()

    std = m[m["variant"] == "standard"]
    yeom_b = (std[std["attack"] == "yeom"][["dataset", "model", "advantage_mean"]]
              .rename(columns={"advantage_mean": "yeom_adv_base"}))
    lira_b = (std[std["attack"] == "lira"]
              [["dataset", "model", "attack_auc_mean", "tpr_at_1pct"]]
              .rename(columns={"attack_auc_mean": "lira_auc_base",
                               "tpr_at_1pct": "tpr1_base"}))

    dp = m[(m["variant"] == "dp") & (m["epsilon"] == EPS_TARGET)
           & (m["attack"] == "lira")][["dataset", "model", "tpr_at_1pct"]]
    dp = dp.rename(columns={"tpr_at_1pct": "tpr1_dp"})

    ut = u[(u["variant"] == "dp") & (u["epsilon"] == EPS_TARGET)].copy()
    ut["model"] = ut["model"].map(MODEL_MAP)
    ut = ut[["dataset", "model", "ACL", "ACL_exported"]]

    d = (yeom_b.merge(lira_b, on=["dataset", "model"])
              .merge(dp, on=["dataset", "model"])
              .merge(ut, on=["dataset", "model"]))
    d["reduction"] = d["tpr1_base"] - d["tpr1_dp"]

    sizes = u.groupby("dataset")["n_samples"].first().reset_index()
    d = d.merge(sizes, on="dataset")
    return d, sizes


def _scatter(ax, d, xcol, ycol):
    for model in MODELS:
        s = d[d["model"] == model]
        if s.empty:
            continue
        ax.scatter(s[xcol], s[ycol], c=MCOLOR[model], label=model, s=42,
                   edgecolors="#333333", linewidth=0.4, zorder=5)


def _annotate(ax, rows, xcol, ycol, labelcol="dataset"):
    for r in rows.itertuples():
        ax.annotate(getattr(r, labelcol),
                    (getattr(r, xcol), getattr(r, ycol)),
                    textcoords="offset points", xytext=(5, 2),
                    fontsize=6, color="#333333")


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}.pdf / .png")


# --------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, _ = build()
    d.to_csv(OUT / "d16_figure_input.csv", index=False)
    print(f"paired rows: {len(d)}\n")

    stats = []

    # ---- L4: average-case vs worst-case agreement (baselines) --------------
    r = spearman_with_ci(d["yeom_adv_base"], d["lira_auc_base"])
    stats.append(("L4  yeom_adv_base vs lira_auc_base", r))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    _scatter(ax, d, "yeom_adv_base", "lira_auc_base")
    ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1.0,
               zorder=10, label="random (0.5)")
    _annotate(ax, d.nlargest(4, "lira_auc_base"), "yeom_adv_base", "lira_auc_base")
    ax.set_xlabel("Yeom advantage (average-case)")
    ax.set_ylabel("LiRA AUC (worst-case)")
    ax.set_title(f"Average-case vs worst-case attack agreement "
                 f"($\\rho$={r['rho']:.2f})", fontsize=9.5)
    ax.grid(True, alpha=0.25, color="#cccccc")
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, "L4_avg_vs_worst_case")

    # ---- RQ1c: utility paid vs protection gained --------------------------
    for col, tag in (("ACL", ""), ("ACL_exported", "_ACLexported")):
        sub = d.dropna(subset=[col])
        r = spearman_with_ci(sub[col], sub["tpr1_dp"])
        stats.append((f"RQ1c {col} vs tpr1_dp", r))
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        _scatter(ax, sub, col, "tpr1_dp")
        ax.axhline(0.01, color="#333333", linestyle="--", linewidth=1.2,
                   zorder=10, label="random (1%)")
        ax.set_xlabel(f"{col} @ $\\varepsilon$=1.0 (utility paid)")
        ax.set_ylabel("DP residual leakage (TPR@1%)")
        ax.set_title(f"Utility paid $\\neq$ protection gained "
                     f"($\\rho$={r['rho']:.2f}, n={r['n']})", fontsize=9.5)
        ax.grid(True, alpha=0.25, color="#cccccc")
        ax.legend(fontsize=7, loc="upper left")
        _save(fig, f"RQ1c_utility_vs_protection{tag}")

    # ---- RQ2a: DP benefit scales with baseline leakage --------------------
    r = spearman_with_ci(d["tpr1_base"], d["reduction"])
    stats.append(("RQ2a tpr1_base vs reduction", r))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    lim = max(d["tpr1_base"].max(), d["reduction"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#333333", linestyle="--", linewidth=1.0,
            zorder=1, label="y=x (full reduction)")
    _scatter(ax, d, "tpr1_base", "reduction")
    _annotate(ax, d.nlargest(3, "tpr1_base"), "tpr1_base", "reduction")
    ax.set_xlabel("Baseline leakage (LiRA TPR@1%)")
    ax.set_ylabel("DP leakage reduction")
    ax.set_title(f"DP benefit scales with baseline leakage "
                 f"($\\rho$={r['rho']:.2f})", fontsize=9.5)
    ax.grid(True, alpha=0.25, color="#cccccc")
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, "RQ2a_benefit_vs_baseline")

    # ---- RQ2b: DP collapses every model to a common floor -----------------
    floor = float(d["tpr1_dp"].median())
    r = spearman_with_ci(d["tpr1_base"], d["tpr1_dp"])
    stats.append(("RQ2b tpr1_base vs tpr1_dp", r))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    _scatter(ax, d, "tpr1_base", "tpr1_dp")
    ax.axhline(floor, color="#117733", linestyle="--", linewidth=1.2, zorder=10,
               label=f"DP floor (median) = {floor:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Baseline leakage TPR@1% (log)")
    ax.set_ylabel("DP residual leakage TPR@1%")
    ax.set_title("DP collapses every model to a common floor", fontsize=9.5)
    ax.grid(True, alpha=0.25, color="#cccccc")
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, "RQ2b_common_floor")

    # ---- L6: baseline leakage shrinks with N (standard RF) ----------------
    rf = d[d["model"] == "RF"].sort_values("n_samples")
    r = spearman_with_ci(rf["n_samples"], rf["tpr1_base"])
    stats.append(("L6  n_samples vs std-RF tpr1_base", r))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(rf["n_samples"], rf["tpr1_base"], c=MCOLOR["RF"], s=55,
               edgecolors="#333333", linewidth=0.4, zorder=5)
    ax.axhline(0.01, color="#333333", linestyle="--", linewidth=1.2,
               zorder=10, label="random (1%)")
    _annotate(ax, rf, "n_samples", "tpr1_base")
    ax.set_xscale("log")
    ax.set_xlabel("$N$ (log)")
    ax.set_ylabel("Std-RF LiRA TPR@1%")
    ax.set_title(f"Baseline leakage shrinks with $N$ "
                 f"($\\rho$={r['rho']:.2f}, n={r['n']})", fontsize=9.5)
    ax.grid(True, alpha=0.25, color="#cccccc")
    ax.legend(fontsize=7, loc="upper right")
    _save(fig, "L6_baseline_leakage_vs_N")

    # ---- stats table ------------------------------------------------------
    rows = [{"figure": name, "rho": s["rho"], "n": s["n"], "p": s["p"],
             "ci_low": s["ci_low"], "ci_high": s["ci_high"], "note": s["note"]}
            for name, s in stats]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "d16_correlations.csv", index=False)

    print("\n=== D-16 correlations ===")
    print(df.to_string(index=False))
    print(f"\nRQ2b DP-residual TPR@1% spread: "
          f"[{d['tpr1_dp'].min():.4f}, {d['tpr1_dp'].max():.4f}], "
          f"median {floor:.4f}, mean {d['tpr1_dp'].mean():.4f}")


if __name__ == "__main__":
    main()
