"""Build the paper figures as PDF + PNG.

Matplotlib only.  No in-image titles (MDPI captions carry the title), no draft
codes in filenames or images, utility axes labelled with ``config.UTILITY_LABEL``.

The five RQ-figure figures are regenerated from ``analysis/figures/
figure_input.csv`` using the statistics already recorded in
``correlations.csv``; those statistics are asserted, never recomputed into
the figure.  A mismatch aborts the build.

Run:  python build_figures.py
"""

from __future__ import annotations

import json
import math

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

import config as C                       # noqa: E402
import loaders as L                      # noqa: E402

# Same house style as analysis/figures_rq_figures.py, minus the titles.
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

_gaps: list[str] = []


def gap(msg: str) -> None:
    _gaps.append(msg)
    print(f"  GAP: {msg}")


def save(fig, name: str) -> None:
    C.FIGURES_PDF.mkdir(parents=True, exist_ok=True)
    C.FIGURES_PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_PDF / f"{name}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(C.FIGURES_PNG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/pdf/{name}.pdf and figures/png/{name}.png")


# ==========================================================================
# Label placement
# ==========================================================================

def place_labels(ax, xs, ys, labels, fontsize=6.0, color="#333333",
                 reserve=()) -> None:
    """Annotate points with a greedy collision-avoiding offset.

    The original RQ figures used a fixed (5, 2) offset for every label, which
    collided wherever points clustered.  Here each label is tried in a set of
    candidate directions and takes the first placement whose bounding box
    overlaps neither an already-placed label, nor a data point, nor any
    artist passed in ``reserve`` (the legend and the statistics box).
    """
    fig = ax.figure
    fig.canvas.draw()
    trans = ax.transData
    # transData is in device pixels; offsets and font sizes are in points.
    s = fig.dpi / 72.0

    placed: list[tuple[float, float, float, float]] = []
    for artist in reserve:
        if artist is None:
            continue
        bb = artist.get_window_extent(fig.canvas.get_renderer())
        placed.append((bb.x0, bb.y0, bb.x1, bb.y1))

    pts = [trans.transform((x, y)) for x, y in zip(xs, ys)]
    # Treat each data marker as a small keep-out box.
    for px, py in pts:
        placed.append((px - 4 * s, py - 4 * s, px + 4 * s, py + 4 * s))

    offsets = [(6, 3), (6, -9), (-6, 3), (-6, -9), (6, 10), (-6, 10),
               (0, 12), (0, -14), (6, 18), (-6, 18), (6, -20), (-6, -20),
               (0, 26), (0, -30), (12, 30), (-12, 30), (12, -34), (-12, -34)]

    def overlaps(box):
        ax0, ay0, ax1, ay1 = box
        for bx0, by0, bx1, by1 in placed:
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                return True
        return False

    for (px, py), label in zip(pts, labels):
        w = 0.52 * fontsize * len(str(label)) * s
        h = 1.2 * fontsize * s
        chosen = offsets[0]
        for dx, dy in offsets:
            x0 = px + dx * s if dx >= 0 else px + dx * s - w
            box = (x0, py + dy * s, x0 + w, py + dy * s + h)
            if not overlaps(box):
                chosen = (dx, dy)
                placed.append(box)
                break
        else:
            dx, dy = offsets[0]
            placed.append((px + dx * s, py + dy * s,
                           px + dx * s + w, py + dy * s + h))
        data_xy = trans.inverted().transform((px, py))
        ax.annotate(str(label), xy=data_xy, xycoords="data",
                    textcoords="offset points", xytext=chosen,
                    fontsize=fontsize, color=color, zorder=20,
                    ha="left" if chosen[0] >= 0 else "right")


def scatter_by_family(ax, d, xcol, ycol, families=None, size=42):
    for fam in (families or L.FAMILIES):
        s = d[d["model"] == fam]
        if s.empty:
            continue
        ax.scatter(s[xcol], s[ycol], c=MCOLOR[fam], label=fam, s=size,
                   edgecolors="#333333", linewidth=0.4, zorder=5)


def stat_text(rho, n, p) -> str:
    p_txt = "$p < 10^{-4}$" if p < 1e-4 else f"$p = {p:.4f}$"
    return f"Spearman $\\rho = {rho:.3f}$, $n = {n}$, {p_txt}"


def annotate_stats(ax, text, loc="lower right"):
    xy = {"lower right": (0.98, 0.03, "right", "bottom"),
          "upper left": (0.02, 0.97, "left", "top"),
          "lower left": (0.02, 0.03, "left", "bottom")}[loc]
    return ax.text(xy[0], xy[1], text, transform=ax.transAxes, fontsize=7.5,
                   ha=xy[2], va=xy[3],
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                             edgecolor="#cccccc", alpha=0.92), zorder=25)


# ==========================================================================
# 1. gap_vs_leakage  /  gap_vs_leakage_vs_N                        [51][65]
# ==========================================================================

def _gap_frame() -> pd.DataFrame:
    """train-test gap joined to baseline LiRA leakage, per (dataset, family)."""
    acc = pd.read_csv(C.TABLES_CSV / "baseline_accuracy_matrix.csv")
    leak = pd.read_csv(C.TABLES_CSV / "baseline_leakage_all_pairs.csv")
    leak = leak.rename(columns={"model": "family"})
    d = acc.merge(leak, on=["dataset", "family"])
    d["model"] = d["family"]
    u = L.load_utility()
    sizes = u.groupby("dataset")["n_samples"].first().reset_index()
    return d.merge(sizes, on="dataset")


def fig_gap_vs_leakage(d: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))

    for ax, ycol, ylabel, ref in (
            (axes[0], "lira_tpr_at_1pct", "Baseline LiRA TPR@1%", C.RANDOM_FPR),
            (axes[1], "lira_auc", "Baseline LiRA AUC", 0.5)):
        r = spearmanr(d["train_test_gap"], d[ycol])
        scatter_by_family(ax, d, "train_test_gap", ycol)
        ax.axhline(ref, color="#333333", linestyle="--", linewidth=1.1, zorder=10,
                   label=f"chance ({ref:g})")
        ax.set_xlabel("Train $-$ test accuracy gap (non-private)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, color="#cccccc")
        ax.margins(x=0.12)
        leg = ax.legend(fontsize=7, loc="upper left")
        box = annotate_stats(ax, stat_text(r.statistic, len(d), r.pvalue))
        # Annotate the datasets that sit furthest out on either axis, keeping
        # clear of the legend and the statistics box.
        out = pd.concat([d.nlargest(3, ycol), d.nlargest(3, "train_test_gap")])
        out = out.drop_duplicates(subset=["dataset", "family"])
        place_labels(ax, out["train_test_gap"], out[ycol],
                     [f"{r_.dataset} {r_.family}" for r_ in out.itertuples()],
                     reserve=(leg, box))

    fig.tight_layout()
    save(fig, "gap_vs_leakage")


def fig_gap_vs_leakage_vs_N(d: pd.DataFrame) -> None:
    """The gap-based relationship beside the N-based one, same y axis."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), sharey=True)

    r_gap = spearmanr(d["train_test_gap"], d["lira_tpr_at_1pct"])
    scatter_by_family(axes[0], d, "train_test_gap", "lira_tpr_at_1pct")
    axes[0].set_xlabel("Train $-$ test accuracy gap (non-private)")
    axes[0].set_ylabel("Baseline LiRA TPR@1%")
    annotate_stats(axes[0], stat_text(r_gap.statistic, len(d), r_gap.pvalue))

    r_n = spearmanr(d["n_samples"], d["lira_tpr_at_1pct"])
    scatter_by_family(axes[1], d, "n_samples", "lira_tpr_at_1pct")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("$N$ (log scale)")
    annotate_stats(axes[1], stat_text(r_n.statistic, len(d), r_n.pvalue))

    for ax in axes:
        ax.axhline(C.RANDOM_FPR, color="#333333", linestyle="--", linewidth=1.1,
                   zorder=10, label=f"chance ({C.RANDOM_FPR:g})")
        ax.grid(True, alpha=0.25, color="#cccccc")
        ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    save(fig, "gap_vs_leakage_vs_N")


# ==========================================================================
# 2. residual_floor_ci                                             [14][15]
# ==========================================================================

def fig_residual_floor_ci() -> None:
    t = pd.read_csv(C.TABLES_CSV / "residual_leakage_eps1.csv")
    # The table carries both `model` (DP-RF) and `family` (RF); the baseline
    # table is keyed on the family code, so key on that.
    t = t.drop(columns=["model"]).rename(columns={"family": "model"})

    floor = float(t["lira_tpr_at_1pct"].median())
    q1 = float(t["lira_tpr_at_1pct"].quantile(0.25))
    q3 = float(t["lira_tpr_at_1pct"].quantile(0.75))

    # Baseline leakage on the x axis, as in the RQ2b original.
    base = pd.read_csv(C.TABLES_CSV / "baseline_leakage_all_pairs.csv")
    t = t.merge(base[["dataset", "model", "lira_tpr_at_1pct"]]
                .rename(columns={"lira_tpr_at_1pct": "tpr1_base"}),
                on=["dataset", "model"])

    lo = t["lira_tpr_at_1pct"] - t["lira_tpr_at_1pct_ci_low"]
    hi = t["lira_tpr_at_1pct_ci_high"] - t["lira_tpr_at_1pct"]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(t["tpr1_base"], t["lira_tpr_at_1pct"],
                yerr=[lo, hi], fmt="none", ecolor="#888888",
                elinewidth=0.8, capsize=2, zorder=3)
    scatter_by_family(ax, t, "tpr1_base", "lira_tpr_at_1pct")
    ax.axhline(floor, color="#117733", linestyle="--", linewidth=1.3, zorder=10,
               label=f"DP floor (median) = {floor:.4f}")
    ax.axhspan(q1, q3, color="#117733", alpha=0.10, zorder=1,
               label=f"IQR [{q1:.4f}, {q3:.4f}]")
    ax.axhline(C.RANDOM_FPR, color="#333333", linestyle=":", linewidth=1.1,
               zorder=10, label=f"chance ({C.RANDOM_FPR:g})")
    ax.set_xscale("log")
    ax.set_xlabel("Baseline LiRA TPR@1% (log scale)")
    ax.set_ylabel(f"DP residual LiRA TPR@1% at $\\varepsilon={C.EPS_TARGET:g}$")
    ax.grid(True, alpha=0.25, color="#cccccc")
    ax.legend(fontsize=7, loc="upper left")
    annotate_stats(
        ax,
        "Floor estimator: median of the 30 DP residual rates\n"
        f"= {floor:.4f}, IQR [{q1:.4f}, {q3:.4f}].\n"
        "Error bars: 95% run-level bootstrap CI\n"
        "from saved integer outcomes at observed ROC thresholds.")
    fig.tight_layout()
    save(fig, "residual_floor_ci")


# ==========================================================================
# 3. the five RQ figures                                      [72][73][75]
# ==========================================================================

RQ_FIGURE_SPEC = {
    "l4_avg_vs_worst_case": "L4  yeom_adv_base vs lira_auc_base",
    "rq1c_utility_vs_protection_exported": "RQ1c ACL_exported vs tpr1_dp",
    "rq2a_benefit_vs_baseline": "RQ2a tpr1_base vs reduction",
    "l6_baseline_leakage_vs_N": "L6  n_samples vs std-RF tpr1_base",
}


def _rq_stats(corr: pd.DataFrame, key: str, x, y) -> tuple:
    """Look up the recorded statistic and assert it reproduces from the data."""
    row = corr[corr["figure"] == key]
    if len(row) != 1:
        raise AssertionError(
            f"correlations.csv has {len(row)} rows for {key!r}")
    rec_rho, rec_n, rec_p = (float(row.iloc[0]["rho"]), int(row.iloc[0]["n"]),
                             float(row.iloc[0]["p"]))
    res = spearmanr(x, y)
    if not (math.isclose(rec_rho, float(res.statistic), abs_tol=1e-6)
            and rec_n == len(x)
            and math.isclose(rec_p, float(res.pvalue), abs_tol=1e-6)):
        raise AssertionError(
            f"{key}: correlations.csv records rho={rec_rho!r}, n={rec_n}, "
            f"p={rec_p!r}, but the same statistic computed from "
            f"figure_input.csv is rho={float(res.statistic)!r}, "
            f"n={len(x)}, p={float(res.pvalue)!r}. Refusing to plot.")
    return rec_rho, rec_n, rec_p


def build_rq_figures() -> None:
    d = L.load_figure_input()
    corr = L.load_correlations()

    if not C.INCLUDE_LUNG_CANCER:
        d = L.filter_datasets(d)
        gap("RQ figures: INCLUDE_LUNG_CANCER=False drops Lung Cancer from "
            "figure_input.csv, so the correlations recorded in "
            "correlations.csv (which were computed over all 30 pairs) no "
            "longer apply. The statistics annotated on these five figures were "
            "recomputed on the reduced set and are NOT the published values.")

    def stats(key, x, y):
        if C.INCLUDE_LUNG_CANCER:
            return _rq_stats(corr, RQ_FIGURE_SPEC[key], x, y)
        res = spearmanr(x, y)
        return float(res.statistic), len(x), float(res.pvalue)

    # ---- L4: agreement between the two average-case attacks (baselines) ----
    # Both axes are average-case summaries: Yeom advantage is TPR-FPR at a single
    # loss threshold, and AUC aggregates over all thresholds. LiRA's worst-case
    # metric is TPR at a low fixed FPR, which is plotted in residual_floor_ci and
    # l6_baseline_leakage_vs_N -- not here. Do not label this axis "worst-case".
    rho, n, p = stats("l4_avg_vs_worst_case", d["yeom_adv_base"], d["lira_auc_base"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    scatter_by_family(ax, d, "yeom_adv_base", "lira_auc_base")
    ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1.1, zorder=10,
               label="chance AUC (0.5)")
    ax.set_xlabel("Yeom advantage (average-case)")
    ax.set_ylabel("LiRA AUC (average-case)")
    ax.grid(True, alpha=0.25, color="#cccccc")
    ax.margins(x=0.12)
    leg = ax.legend(fontsize=7, loc="upper left")
    box = annotate_stats(ax, stat_text(rho, n, p))
    out = d.nlargest(4, "lira_auc_base")
    place_labels(ax, out["yeom_adv_base"], out["lira_auc_base"], out["dataset"],
                 reserve=(leg, box))
    fig.tight_layout()
    save(fig, "l4_avg_vs_worst_case")

    # ---- RQ1c: utility paid vs protection gained --------------------------
    for key, col in (("rq1c_utility_vs_protection_exported", "ACL_exported"),):
        sub = d.dropna(subset=[col])
        if len(sub) < len(d):
            gap(f"{key}: {len(d) - len(sub)} of {len(d)} pairs have no "
                f"{col} and are absent from this figure: "
                f"{sorted(map(tuple, d[d[col].isna()][['dataset', 'model']].values))}. "
                f"exported_model_accuracy was blank on the DP-DNN rows of BCP, "
                f"CANCER_RISK, DIABETES and GALLSTONE until 2026-08-16, when "
                f"those 36 cells were recovered by scoring the exported run-0 "
                f"artefacts directly (see Results/dataset_results/"
                f"measure_exported_accuracy.py); if this gap has reappeared, "
                f"consolidated_data.csv has been regenerated without that step.")
        rho, n, p = stats(key, sub[col], sub["tpr1_dp"])
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        scatter_by_family(ax, sub, col, "tpr1_dp")
        ax.axhline(C.RANDOM_FPR, color="#333333", linestyle="--", linewidth=1.2,
                   zorder=10, label=f"chance ({C.RANDOM_FPR:g})")
        label = C.UTILITY_LABEL if col == "ACL" else f"{C.UTILITY_LABEL} (exported)"
        ax.set_xlabel(f"{label} at $\\varepsilon={C.EPS_TARGET:g}$ (utility paid)")
        ax.set_ylabel(f"DP residual leakage, LiRA TPR@1% at "
                      f"$\\varepsilon={C.EPS_TARGET:g}$")
        ax.grid(True, alpha=0.25, color="#cccccc")
        ax.legend(fontsize=7, loc="upper left")
        annotate_stats(ax, stat_text(rho, n, p))
        fig.tight_layout()
        save(fig, key)

    # ---- RQ2a: DP benefit scales with baseline leakage --------------------
    rho, n, p = stats("rq2a_benefit_vs_baseline", d["tpr1_base"], d["reduction"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    lim = max(d["tpr1_base"].max(), d["reduction"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#333333", linestyle="--", linewidth=1.0,
            zorder=1, label="$y = x$ (leakage fully removed)")
    scatter_by_family(ax, d, "tpr1_base", "reduction")
    ax.set_xlabel("Baseline leakage, LiRA TPR@1%")
    ax.set_ylabel(f"Leakage reduction under DP at $\\varepsilon={C.EPS_TARGET:g}$")
    ax.grid(True, alpha=0.25, color="#cccccc")
    ax.margins(x=0.12)
    leg = ax.legend(fontsize=7, loc="upper left")
    box = annotate_stats(ax, stat_text(rho, n, p))
    out = d.nlargest(4, "tpr1_base")
    place_labels(ax, out["tpr1_base"], out["reduction"], out["dataset"],
                 reserve=(leg, box))
    fig.tight_layout()
    save(fig, "rq2a_benefit_vs_baseline")

    # ---- L6: baseline leakage shrinks with N (standard RF) ----------------
    rf = d[d["model"] == "RF"].sort_values("n_samples")
    rho, n, p = stats("l6_baseline_leakage_vs_N", rf["n_samples"], rf["tpr1_base"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(rf["n_samples"], rf["tpr1_base"], c=MCOLOR["RF"], s=55,
               edgecolors="#333333", linewidth=0.4, zorder=5, label="RF")
    ax.axhline(C.RANDOM_FPR, color="#333333", linestyle="--", linewidth=1.2,
               zorder=10, label=f"chance ({C.RANDOM_FPR:g})")
    ax.set_xscale("log")
    ax.set_xlabel("$N$ (log scale)")
    ax.set_ylabel("Non-private RF leakage, LiRA TPR@1%")
    ax.grid(True, alpha=0.25, color="#cccccc")
    ax.margins(x=0.18, y=0.12)
    leg = ax.legend(fontsize=7, loc="upper right")
    box = annotate_stats(ax, stat_text(rho, n, p), loc="lower left")
    place_labels(ax, rf["n_samples"], rf["tpr1_base"], rf["dataset"],
                 fontsize=6.5, reserve=(leg, box))
    fig.tight_layout()
    save(fig, "l6_baseline_leakage_vs_N")


# ==========================================================================
# Driver
# ==========================================================================

def main() -> int:
    print("Building figures...")
    d = _gap_frame()
    fig_gap_vs_leakage(d)
    fig_gap_vs_leakage_vs_N(d)
    fig_residual_floor_ci()
    build_rq_figures()

    C.LOGS.mkdir(parents=True, exist_ok=True)
    (C.LOGS / "_figure_gaps.json").write_text(
        json.dumps({"gaps": _gaps}, indent=2) + "\n")
    print(f"\n{len(_gaps)} figure gap(s) -> logs/_figure_gaps.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
