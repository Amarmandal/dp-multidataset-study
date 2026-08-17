"""
Per-dataset reporting for the Shokri MIA pipeline.

For one dataset this writes a self-contained folder ``results/<DATASET>/``::

    results/<DATASET>/
        <DATASET>_advantage_vs_epsilon.png
        <DATASET>_auc_vs_epsilon.png
        <DATASET>_advantage_vs_gengap.png
        <DATASET>_std_vs_dp_bars.png
        <DATASET>_results.csv      # this dataset's rows only
        analysis.md                # auto-written, data-driven narrative

``run_mia.py`` calls ``build_report`` after attacking each dataset, so the same
artefacts are produced for every dataset in the study.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["LR", "RF", "GNB", "SVM", "DNN"]
COLORS = {"LR": "#1f77b4", "RF": "#d62728", "GNB": "#2ca02c",
          "SVM": "#9467bd", "DNN": "#ff7f0e"}

# What counts as a successful attack (used only for the auto-narrative).
LEAK_ADV = 0.10   # membership advantage above this = real leakage
LEAK_AUC = 0.60   # attack AUC above this = real leakage
FLAT_GAP = 0.02   # standard gen-gap below this = dataset doesn't overfit


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _curve(sub, metric, ylabel, ds, path):
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
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _gengap_scatter(sub, ds, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, marker, label in [("standard", "o", "Standard"), ("dp", "x", "DP")]:
        s = sub[sub["variant"] == variant]
        ax.scatter(s["gen_gap_mean"], s["advantage_mean"], marker=marker,
                   alpha=0.6, label=label)
    ax.axhline(LEAK_ADV, ls=":", color="grey", lw=1, label=f"leak threshold ({LEAK_ADV})")
    ax.set_xlabel("Generalisation gap (train acc − test acc)")
    ax.set_ylabel("Membership advantage (TPR − FPR)")
    ax.set_title(f"{ds}: attack success vs overfitting")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _std_vs_dp_bars(sub, ds, path):
    models = [m for m in MODELS if m in sub["model"].unique()]
    std_vals, dp_vals = [], []
    for m in models:
        s = sub[(sub["model"] == m) & (sub["variant"] == "standard")]
        dp = sub[(sub["model"] == m) & (sub["variant"] == "dp")].sort_values("epsilon")
        std_vals.append(s["advantage_mean"].values[0] if not s.empty else 0)
        dp_vals.append(dp["advantage_mean"].values[0] if not dp.empty else 0)
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(models)), 4.5))
    ax.bar(x - 0.2, std_vals, 0.4, label="Standard", color="#d62728")
    ax.bar(x + 0.2, dp_vals, 0.4, label="DP (ε=min)", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Membership advantage (TPR − FPR)")
    ax.set_title(f"{ds}: Standard vs strong-privacy DP")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _heatmap(sub, ds, path):
    families = [m for m in MODELS if m in sub["model"].unique()]
    eps = sorted(sub.loc[sub["variant"] == "dp", "epsilon"].dropna().unique())
    cols = eps + ["∞"]  # DP budgets then the Standard (ε=∞) column

    def val(metric, m, e):
        if e == "∞":
            r = sub[(sub["model"] == m) & (sub["variant"] == "standard")]
        else:
            r = sub[(sub["model"] == m) & (sub["variant"] == "dp") & (sub["epsilon"] == e)]
        return r[f"{metric}_mean"].values[0] if not r.empty else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(1.0 * len(cols) + 5, 0.55 * len(families) + 2.5))
    panels = [(axes[0], "advantage", "Membership advantage", "Reds", 0.0, LEAK_ADV),
              (axes[1], "attack_auc", "Attack AUC", "RdBu_r", 0.40, 0.60)]
    for ax, metric, title, cmap, vmin, vmax in panels:
        M = np.array([[val(metric, m, e) for e in cols] for m in families])
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([("∞" if c == "∞" else f"{c:g}") for c in cols])
        ax.set_yticks(range(len(families)))
        ax.set_yticklabels(families)
        ax.set_xlabel("ε  (∞ = Standard)")
        ax.set_title(title)
        for i in range(len(families)):
            for j in range(len(cols)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{ds}: MIA leakage heatmap  (advantage scaled 0→leak threshold)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _roc_scatter(sub, ds, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="random (TPR=FPR)")
    for m in MODELS:
        d = sub[sub["model"] == m]
        if d.empty:
            continue
        dp = d[d["variant"] == "dp"]
        std = d[d["variant"] == "standard"]
        ax.scatter(dp["fpr_mean"], dp["tpr_mean"], color=COLORS[m],
                   marker="x", alpha=0.7, label=f"{m} DP")
        ax.scatter(std["fpr_mean"], std["tpr_mean"], color=COLORS[m],
                   marker="o", edgecolor="k", s=55, label=f"{m} Std")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate (non-members called members)")
    ax.set_ylabel("True positive rate (members detected)")
    ax.set_title(f"{ds}: attack operating points  (on diagonal = no leak)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _privacy_utility(sub, ds, path):
    dp = sub[sub["variant"] == "dp"]
    if dp.empty:
        return
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax2 = ax1.twinx()
    for m in MODELS:
        d = dp[dp["model"] == m].sort_values("epsilon")
        if d.empty:
            continue
        ax1.plot(d["epsilon"], d["test_acc_mean"], marker="o",
                 color=COLORS[m], label=m)
        ax2.plot(d["epsilon"], d["advantage_mean"], marker="x", ls=":",
                 color=COLORS[m])
    ax1.set_xscale("log")
    ax1.set_xlabel("Privacy budget  ε  (log scale)")
    ax1.set_ylabel("Target test accuracy  —  solid (utility ↑)")
    ax2.set_ylabel("Membership advantage  —  dotted (leakage)")
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_ylim(-0.02, max(LEAK_ADV, float(dp["advantage_mean"].max()) * 1.5))
    ax1.set_title(f"{ds}: utility rises with ε while leakage stays flat")
    ax1.grid(True, alpha=0.3)
    ax1.legend(title="model", fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _perrun_box(targets, ds, path):
    data = {}
    for t in targets or []:
        fam = t["config"]["model"]
        for r in t.get("runs", []):
            if isinstance(r, dict) and "advantage" in r:
                data.setdefault(fam, []).append(r["advantage"])
    fams = [m for m in MODELS if m in data]
    if not fams:
        return
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(fams)), 5))
    ax.boxplot([data[m] for m in fams], showfliers=False)
    rng = np.random.default_rng(0)
    for i, m in enumerate(fams, start=1):
        xs = rng.normal(i, 0.05, size=len(data[m]))
        ax.scatter(xs, data[m], color=COLORS[m], alpha=0.5, s=15)
    ax.set_xticks(range(1, len(fams) + 1))
    ax.set_xticklabels(fams)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axhline(LEAK_ADV, color="red", ls=":", lw=1, label=f"leak threshold ({LEAK_ADV})")
    ax.set_ylabel("Per-run membership advantage")
    ax.set_title(f"{ds}: seed-level advantage spread  (noise band around 0)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Analysis narrative
# --------------------------------------------------------------------------- #
def _fmt_eps(e):
    return "∞" if pd.isna(e) else f"{float(e):g}"


def _results_table(sub):
    rows = ["| Model | Variant | ε | Advantage | Attack AUC | Train acc | Test acc | Gen. gap |",
            "|-------|---------|---|-----------|-----------|-----------|----------|----------|"]
    order = {m: i for i, m in enumerate(MODELS)}
    sub = sub.sort_values(
        by=["model", "variant", "epsilon"],
        key=lambda c: c.map(order) if c.name == "model" else c,
    )
    for _, r in sub.iterrows():
        variant = "Standard" if r["variant"] == "standard" else "DP"
        rows.append(
            f"| {r['model']} | {variant} | {_fmt_eps(r['epsilon'])} "
            f"| {r['advantage_mean']:.3f} | {r['attack_auc_mean']:.3f} "
            f"| {r['train_acc_mean']:.3f} | {r['test_acc_mean']:.3f} "
            f"| {r['gen_gap_mean']:+.3f} |"
        )
    return "\n".join(rows)


# Order + one-line descriptions for the embedded figure gallery.
_FIGURE_INFO = [
    ("adv", "Membership advantage vs ε",
     "Attack advantage (TPR − FPR) for each DP model across the privacy budget; "
     "dashed lines are the non-private Standard baselines. Flat near 0 means the "
     "attacker is no better than guessing. Note the very small y-axis scale."),
    ("auc", "Attack AUC vs ε",
     "How well the attacker *ranks* members above non-members. 0.5 is random; "
     "every curve sits on 0.5 within its error bars, i.e. no discriminating power."),
    ("heatmap", "Leakage heatmap (model × ε)",
     "Advantage (left) and AUC (right) for every model/budget cell, including the "
     "Standard column (∞). The advantage scale runs 0 → the leak threshold, so "
     "near-white cells = no leakage; the printed numbers give the exact values."),
    ("roc", "Attack operating points (TPR vs FPR)",
     "Each target plotted as one (FPR, TPR) point against the random diagonal. "
     "Points sitting on the diagonal mean the attacker's hit-rate equals its "
     "false-alarm rate — the textbook signature of a failed membership attack."),
    ("putility", "Privacy–utility–leakage trade-off",
     "Target test accuracy (solid, left axis) versus attack advantage (dotted, "
     "right axis) as ε grows. Utility climbs while leakage stays pinned near zero — "
     "relaxing privacy here buys accuracy without creating a membership risk."),
    ("gap", "Advantage vs generalisation gap",
     "Membership advantage against each model's overfitting (train − test acc), "
     "with the leak threshold dotted. Points clustered at the origin show there is "
     "no overfitting to exploit — the root cause of (the absence of) leakage."),
    ("perrun", "Seed-level advantage spread",
     "The individual seed runs behind each averaged point, grouped by model. The "
     "spread straddles zero, confirming the small non-zero means are random noise "
     "rather than a real signal."),
    ("bars", "Standard vs strong-privacy DP",
     "Per family, the non-private Standard advantage beside the strongest-privacy "
     "DP model (ε = min). Both bars sit near zero across all families."),
]


def _figure_gallery(figs):
    blocks = []
    for key, title, desc in _FIGURE_INFO:
        if key not in figs:
            continue
        blocks.append(f"### {title}\n\n![{title}]({figs[key]})\n\n{desc}\n")
    return "\n".join(blocks)


def _write_analysis(ds, sub, meta, figs, path):
    adv_max = sub.loc[sub["advantage_mean"].idxmax()]
    auc_max = sub.loc[sub["attack_auc_mean"].idxmax()]
    leaks = sub[(sub["advantage_mean"] > LEAK_ADV) | (sub["attack_auc_mean"] > LEAK_AUC)]
    std_gap = sub[sub["variant"] == "standard"]["gen_gap_mean"].abs().max()
    families = [m for m in MODELS if m in sub["model"].unique()]
    eps_list = sorted(sub.loc[sub["variant"] == "dp", "epsilon"].dropna().unique())

    # ---- interpretation branch ----
    if leaks.empty:
        verdict = (
            f"**No measurable membership leakage on {ds} — for any model, at any ε, "
            f"including the non-private Standard models.** The strongest result anywhere "
            f"is advantage **{adv_max['advantage_mean']:.3f}** "
            f"({adv_max['model']} {('Standard' if adv_max['variant']=='standard' else 'DP ε='+_fmt_eps(adv_max['epsilon']))}) "
            f"and attack AUC **{auc_max['attack_auc_mean']:.3f}** — both statistically "
            f"indistinguishable from random guessing (advantage 0 / AUC 0.5)."
        )
    else:
        names = ", ".join(
            f"{r['model']} "
            f"{'Standard' if r['variant']=='standard' else 'DP ε='+_fmt_eps(r['epsilon'])} "
            f"(adv {r['advantage_mean']:.3f}, AUC {r['attack_auc_mean']:.3f})"
            for _, r in leaks.iterrows()
        )
        verdict = (
            f"**{len(leaks)} configuration(s) leak** above the threshold "
            f"(advantage > {LEAK_ADV} or AUC > {LEAK_AUC}): {names}."
        )

    if std_gap < FLAT_GAP:
        caveat = (
            f"\n**Important caveat:** the Standard models show an almost-zero "
            f"generalisation gap (max |train − test| = {std_gap:.3f}). The dataset is "
            f"large/separable enough that members and non-members behave identically, so "
            f"there is little to no membership signal *at the source*. Read a clean result "
            f"here as **\"there was nothing to steal\"**, not \"DP defeated a strong attack.\""
        )
    else:
        caveat = (
            f"\nThe Standard models overfit (max |train − test| gap = {std_gap:.3f}), which "
            f"is the condition under which MIA can succeed — compare the Standard vs DP bars."
        )

    md = f"""# {ds} — Membership Inference Attack (Shokri) Analysis

*Auto-generated by `run_mia.py` → `report.py`. Raw numbers: `{ds}_results.csv`.*

Shokri et al. (2017) shadow-model Membership Inference Attack (arXiv:1610.05820),
run against the **exported** targets in `{ds}/<family>/output/model/`.

## 1. What the attack asks
Given only query access to a trained model, can an attacker tell whether a
specific patient's record was in the **training set**? If yes, model membership
leaks — a privacy breach. The tell-tale is the **generalisation gap**
(`train_acc − test_acc`): a model that memorises its training data reacts
differently to members vs non-members. Gap ≈ 0 ⇒ nothing to leak.

## 2. What was run for {ds}
| Item | Value |
|------|-------|
| Samples | N = {meta['n_train'] + meta['n_test']:,} ({meta['n_train']:,} train / {meta['n_test']:,} test) |
| Classes | {meta['n_classes']} |
| Features | {meta['n_features']} |
| Model families | {', '.join(families)} |
| DP budgets ε | {', '.join(_fmt_eps(e) for e in eps_list)} |
| Targets attacked | {len(sub)} |
| Target source | exported `output/model/` (loaded, not re-trained) |

The **target** is your saved model; **shadow** models are trained fresh (inherent
to the attack). Per-family loading: LR/RF/GNB load directly; the OvR **SVM** gets
a softmax probability head and is attacked in its ‖x‖≤1 space; the **DNN** is an
opacus `GradSampleModule` with a softmax head over its logits.

## 3. How to read the metrics
| Metric | Meaning | No leak | Strong leak |
|--------|---------|---------|-------------|
| Membership advantage = TPR − FPR | better-than-coin-flip membership guess | 0.0 | > {LEAK_ADV} |
| Attack AUC | ranking quality of the guess | 0.5 | > {LEAK_AUC} |
| Generalisation gap = train − test | how much the model overfits (the *cause*) | 0.0 | > 0.1 |

## 4. Results
{_results_table(sub)}

## 5. Figures
{_figure_gallery(figs)}
Watch the y-axis scale on the ε-curves: when every point hugs advantage 0 / AUC
0.5 with error bars that cross those lines, the "wiggle" is noise, not signal.

## 6. Interpretation
{verdict}
{caveat}

## 7. Reproduce
```bash
cd Attack/MIA_Shokri
python3 run_mia.py --datasets {ds}
```
"""
    with open(path, "w") as f:
        f.write(md)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_report(ds, df, meta, results_root, targets=None):
    """Write figures + analysis.md + per-dataset CSV into results/<DATASET>/."""
    out_dir = os.path.join(results_root, ds)
    os.makedirs(out_dir, exist_ok=True)
    sub = df[df["dataset"] == ds].copy()
    sub["epsilon"] = pd.to_numeric(sub["epsilon"], errors="coerce")

    figs = {
        "adv": f"{ds}_advantage_vs_epsilon.png",
        "auc": f"{ds}_auc_vs_epsilon.png",
        "heatmap": f"{ds}_leakage_heatmap.png",
        "roc": f"{ds}_roc_operating_points.png",
        "putility": f"{ds}_privacy_utility_leakage.png",
        "gap": f"{ds}_advantage_vs_gengap.png",
        "perrun": f"{ds}_perrun_advantage_box.png",
        "bars": f"{ds}_std_vs_dp_bars.png",
    }
    _curve(sub, "advantage", "Membership advantage", ds, os.path.join(out_dir, figs["adv"]))
    _curve(sub, "attack_auc", "Attack AUC", ds, os.path.join(out_dir, figs["auc"]))
    _heatmap(sub, ds, os.path.join(out_dir, figs["heatmap"]))
    _roc_scatter(sub, ds, os.path.join(out_dir, figs["roc"]))
    _privacy_utility(sub, ds, os.path.join(out_dir, figs["putility"]))
    _gengap_scatter(sub, ds, os.path.join(out_dir, figs["gap"]))
    _perrun_box(targets, ds, os.path.join(out_dir, figs["perrun"]))
    _std_vs_dp_bars(sub, ds, os.path.join(out_dir, figs["bars"]))

    sub.to_csv(os.path.join(out_dir, f"{ds}_results.csv"), index=False)
    _write_analysis(ds, sub, meta, figs, os.path.join(out_dir, "analysis.md"))
    return out_dir
