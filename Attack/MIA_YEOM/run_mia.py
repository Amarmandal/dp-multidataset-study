"""
Driver: run Yeom's loss-threshold membership inference attack (Yeom et al. 2018)
against the exported Standard vs DP targets for **every** model family, and write
a per-dataset report whose headline metric is the **membership advantage**
(TPR − FPR) — the average-case leakage number, the companion to the per-example
TPR@low-FPR that ``Attack/LiRA`` reports.

It reuses the Shokri pipeline's target loaders and feature-space views
(``Attack/MIA_Shokri/exported_models.py``), exactly like ``Attack/LiRA/run_lira.py``,
so a single command attacks LR / RF / GNB / SVM / DNN, Standard and DP, across
all six study datasets. The attack itself lives in ``mia.py`` and is unchanged from
the original per-model ``*_mia.py`` scripts — only the way it is *run* is now
modular.

The target is always the **exported** artifact (``std_<fam>_model.pkl`` /
``dp_<fam>_model_eps_<eps>.pkl``) — the very same file whose accuracy produced the
ACL number in the utility pipeline. It is never retrained here. Together with the
fixed 80/20 member / non-member split (``random_state=42``, baked into
``processed_data.pkl``) that makes Yeom **fully deterministic**: one run per
target, every ``*_std`` column 0.0, and an explicit ``deterministic=True`` column
recording that the zero dispersion is intended, not a seeding bug.

Outputs (under Attack/MIA_YEOM/results/):
  * mia_comparison.csv                 tidy table across datasets
  * results/<dataset>/                 per-dataset CSV (ground truth) +
                                       figures + analysis.md
  No JSON is written.

Usage:
    python run_mia.py --datasets KIDNEY_STONE
    python run_mia.py                                 # all datasets, all models
    python run_mia.py --verify-determinism            # run twice, assert identical
    python run_mia.py --quick                         # fast smoke test
"""

import argparse
import os
import pickle
import sys
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Reuse the Shokri folder's loaders / feature-space views (same as LiRA). The
# repo root goes on the path too: the SVM notebooks import their DP estimator
# from ``common_svm``, so exported targets pickle as
# ``common_svm.DifferentiallyPrivateSVM`` and pickle.load must import it.
SHOKRI_DIR = os.path.join(os.path.dirname(__file__), "..", "MIA_Shokri")
sys.path.insert(0, os.path.abspath(SHOKRI_DIR))
sys.path.insert(0, REPO)
import exported_models as em  # noqa: E402
from mia import yeom_mia  # noqa: E402

warnings.filterwarnings("ignore")


DATASETS = [
    "GALLSTONE",
    "KIDNEY_STONE",
    "LUNG_CANCER",
    "BCP",
    "CANCER_RISK",
    "DIABETES",
]
DEFAULT_EPSILONS = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0, 10.0]
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")

MODELS = ["LR", "RF", "GNB", "SVM", "DNN"]
COLORS = {
    "LR": "#1f77b4",
    "RF": "#d62728",
    "GNB": "#2ca02c",
    "SVM": "#9467bd",
    "DNN": "#ff7f0e",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_dataset(name):
    path = os.path.join(REPO, name, "data", "processed_data.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    data = {
        "X_train": np.asarray(d["X_train"], dtype=float),
        "X_test": np.asarray(d["X_test"], dtype=float),
        "y_train": np.asarray(d["y_train"]).ravel().astype(int),
        "y_test": np.asarray(d["y_test"]).ravel().astype(int),
    }
    n_classes = int(d.get("n_classes", len(np.unique(data["y_train"]))))
    low, high = d["bounds"]
    return data, n_classes, np.asarray(low, float), np.asarray(high, float)


def target_test_acc(target, view):
    return float((target.predict(view["X_test"]) == view["y_test"]).mean())


# --------------------------------------------------------------------------- #
# Attack one exported target
# --------------------------------------------------------------------------- #
# Aggregated across runs: bare name carries the mean, ``<name>_std`` the spread.
# The bare names are what prepare_mia_data.py already aliases to ``*_mean``.
AGG_KEYS = [
    "advantage",
    "attack_auc",
    "tpr",
    "fpr",
    "test_acc",
    "threshold_tau",
    "loss_gap",
]
# Sample-level statistics from a single attack; averaged across seeds, no std.
PASS_KEYS = ["train_loss_mean", "train_loss_std", "test_loss_mean", "test_loss_std"]
COUNT_KEYS = ["n_members", "n_nonmembers"]


def aggregate(runs, family, variant, epsilon):
    """Mean/std across runs. Mirrors Attack/MIA_Shokri/run_mia.py.

    Yeom is deterministic (see ``attack_target``), so ``runs`` holds exactly one
    entry and every ``<key>_std`` comes out 0.0. The schema is kept identical to
    the stochastic attacks so downstream consumers do not need a special case.
    """
    ok = [r for r in runs if "error" not in r]
    row = {"model": family, "variant": variant, "epsilon": epsilon, "n_runs": len(runs)}
    if not ok:
        row["error"] = runs[0].get("error", "unknown")
        row["deterministic"] = True
        return row
    for k in AGG_KEYS:
        vals = [r[k] for r in ok if k in r]
        row[k] = float(np.mean(vals)) if vals else None
        row[f"{k}_std"] = float(np.std(vals)) if vals else None
    for k in PASS_KEYS:
        vals = [r[k] for r in ok if k in r]
        row[k] = float(np.mean(vals)) if vals else None
    for k in COUNT_KEYS:
        row[k] = ok[0].get(k)
    # Explicit methodological marker: the target is the fixed exported artifact
    # and the member/non-member partition is the fixed 80/20 split, so a repeat
    # call reproduces these numbers exactly. std=0.0 is expected, not a bug.
    row["deterministic"] = True
    return row


def attack_target(fam, variant, data, n_classes, low, high, ds, epsilon=None):
    """Attack the **exported** target once. No retraining, no seeds.

    The target is loaded from ``<DATASET>/<FAM>/output/model/`` — the same
    artifact the utility pipeline measured for ACL. Members are ``X_train`` and
    non-members ``X_test`` from ``processed_data.pkl`` (stratified 80/20,
    random_state=42); neither is resampled here. Yeom's rule (loss <= mean
    training loss) has no randomness of its own, so the whole procedure is
    deterministic and one run is the complete result.
    """
    view, lo, hi = em.family_dataview(fam, data, low, high)

    try:
        target = em.load_target(REPO, ds, fam, variant, epsilon, n_classes)
        m = yeom_mia(target, view)
        m["test_acc"] = target_test_acc(target, view)
    except Exception as exc:
        m = {"error": str(exc)}

    row = aggregate([m], fam, variant, epsilon)
    row["dataset"] = ds
    return row


def run_dataset(ds, args, rows):
    data, n_classes, low, high = load_dataset(ds)
    print(
        f"\n{'=' * 72}\n{ds}  (classes={n_classes}, n_train={len(data['y_train'])}, "
        f"n_features={data['X_train'].shape[1]})  [Yeom MIA]\n{'=' * 72}"
    )
    results = []

    for fam in args.models:
        if fam not in em.MODEL_FAMILIES or not em.model_exists(
            REPO, ds, fam, "standard"
        ):
            print(f"  [{fam}] skipped (no exported model)")
            continue

        t0 = time.time()
        m = attack_target(fam, "standard", data, n_classes, low, high, ds)
        _verify_determinism(args, fam, "standard", data, n_classes, low, high, ds, m)
        results.append(m)
        _print_row(fam, "STD  ", m)

        for eps in args.epsilons:
            if not em.model_exists(REPO, ds, fam, "dp", eps):
                continue
            m = attack_target(
                fam, "dp", data, n_classes, low, high, ds, epsilon=eps
            )
            _verify_determinism(
                args, fam, "dp", data, n_classes, low, high, ds, m, epsilon=eps
            )
            results.append(m)
            _print_row(fam, f"eps={eps:<4}", m)
        print(f"  [{fam}] done in {time.time() - t0:.1f}s")

    df = pd.DataFrame(results)
    rows.extend(df.to_dict("records"))
    build_report(ds, df, n_classes, data)


def _verify_determinism(
    args, fam, variant, data, n_classes, low, high, ds, first, epsilon=None
):
    """--verify-determinism: attack the same target twice, assert identical rows.

    This is the repeat check that justifies ``n_runs=1``: if a second, entirely
    independent call reproduces every metric bit-for-bit, the zero std is a
    property of the method, not of a missing seed.
    """
    if not args.verify_determinism:
        return
    second = attack_target(
        fam, variant, data, n_classes, low, high, ds, epsilon=epsilon
    )
    keys = [k for k in AGG_KEYS + PASS_KEYS if k in first]
    deltas = {
        k: abs(float(first[k]) - float(second[k]))
        for k in keys
        if first.get(k) is not None and second.get(k) is not None
    }
    # Tolerance, not exact equality. sklearn's RandomForest averages tree
    # posteriors across threads (n_jobs=-1), so the summation order varies and
    # predict_proba differs at ~1e-16 between two calls on the *same* object.
    # That floating-point noise reaches the mean-loss keys (threshold_tau,
    # loss_gap, *_loss_mean) while every rate metric stays bit-identical. A real
    # seeding leak -- a target actually being retrained -- moves these by 1e-3
    # or more, so 1e-9 separates the two cases with many orders of magnitude to
    # spare.
    tol = 1e-9
    bad = {k: d for k, d in deltas.items() if d > tol}
    tag = "STD" if variant == "standard" else f"eps={epsilon:g}"
    if bad:
        detail = ", ".join(f"{k} Δ={d:.3e}" for k, d in sorted(bad.items()))
        raise SystemExit(
            f"DETERMINISM CHECK FAILED [{ds}/{fam}/{tag}]: {detail}"
        )
    worst = max(deltas.values()) if deltas else 0.0
    print(
        f"  [{fam}] {tag} determinism check OK "
        f"(repeat call identical, max Δ={worst:.1e})"
    )


def _print_row(fam, tag, m):
    if "error" in m:
        print(f"  [{fam}] {tag} ERROR: {m['error']}")
    else:
        print(
            f"  [{fam}] {tag} adv={m['advantage']:.3f}  AUC={m['attack_auc']:.3f}  "
            f"TPR={m['tpr']:.3f}  FPR={m['fpr']:.3f}  acc={m.get('test_acc', float('nan')):.3f}"
        )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _curve(df, metric, ylabel, ds, path, baseline=None):
    dp = df[df["variant"] == "dp"].sort_values("epsilon")
    std = df[df["variant"] == "standard"]
    if dp.empty or metric not in dp:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        d = dp[dp["model"] == m]
        if d.empty or d[metric].isna().all():
            continue
        ax.plot(d["epsilon"], d[metric], marker="o", color=COLORS[m], label=f"DP-{m}")
        s = std[std["model"] == m]
        if not s.empty and np.isfinite(s[metric].values[0]):
            ax.axhline(s[metric].values[0], ls="--", lw=1, color=COLORS[m], alpha=0.6)
    if baseline is not None:
        ax.axhline(baseline, ls=":", color="grey", lw=1, label="random baseline")
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget  ε  (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ds}: {ylabel} vs ε  (dashed = Standard baseline)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _adv_bars(df, ds, path):
    models = [m for m in MODELS if m in df["model"].unique()]
    std_v, dp_v = [], []
    for m in models:
        s = df[(df["model"] == m) & (df["variant"] == "standard")]
        dp = df[(df["model"] == m) & (df["variant"] == "dp")].sort_values("epsilon")
        std_v.append(s["advantage"].values[0] if not s.empty else 0)
        dp_v.append(dp["advantage"].min() if not dp.empty else 0)  # best protection
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(models)), 4.5))
    ax.bar(x - 0.2, std_v, 0.4, label="Standard (ε=∞)", color="#d62728")
    ax.bar(x + 0.2, dp_v, 0.4, label="DP (best ε)", color="#1f77b4")
    ax.axhline(0.0, ls=":", color="grey", lw=1, label="no leakage (0.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Membership advantage (TPR − FPR)")
    ax.set_title(f"{ds}: Yeom membership advantage, Standard vs strongest DP")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _loss_gap(df, ds, path):
    """Generalisation gap (test − train loss) — what Yeom's attack exploits."""
    dp = df[df["variant"] == "dp"].sort_values("epsilon")
    std = df[df["variant"] == "standard"]
    if dp.empty or "loss_gap" not in dp:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in MODELS:
        d = dp[dp["model"] == m]
        if d.empty or d["loss_gap"].isna().all():
            continue
        ax.plot(
            d["epsilon"], d["loss_gap"], marker="o", color=COLORS[m], label=f"DP-{m}"
        )
        s = std[std["model"] == m]
        if not s.empty and np.isfinite(s["loss_gap"].values[0]):
            ax.axhline(
                s["loss_gap"].values[0], ls="--", lw=1, color=COLORS[m], alpha=0.6
            )
    ax.axhline(0.0, ls=":", color="grey", lw=1, label="no gap")
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget  ε  (log scale)")
    ax.set_ylabel("Generalisation gap  (mean test − train loss)")
    ax.set_title(f"{ds}: loss gap vs ε  (the signal Yeom exploits; dashed = Standard)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(ds, df, n_classes, data):
    out = os.path.join(OUT_DIR, ds)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{ds}_mia_results.csv"), index=False)

    if "advantage" in df:
        _curve(
            df,
            "advantage",
            "Membership advantage (TPR − FPR)",
            ds,
            os.path.join(out, f"{ds}_advantage_vs_epsilon.png"),
            baseline=0.0,
        )
        _curve(
            df,
            "attack_auc",
            "Attack AUC",
            ds,
            os.path.join(out, f"{ds}_auc_vs_epsilon.png"),
            baseline=0.5,
        )
        _adv_bars(df, ds, os.path.join(out, f"{ds}_advantage_std_vs_dp_bars.png"))
        _loss_gap(df, ds, os.path.join(out, f"{ds}_loss_gap_vs_epsilon.png"))

    _write_analysis(ds, df, n_classes, data, out)
    print(f"  -> figures + analysis.md written to {out}/")


def _write_analysis(ds, df, n_classes, data, out):
    std = df[df["variant"] == "standard"]
    dp = df[df["variant"] == "dp"]
    eps_list = sorted(dp["epsilon"].dropna().unique())

    def fmt(v, p=3):
        return (
            "—"
            if v is None or (isinstance(v, float) and not np.isfinite(v))
            else f"{v:.{p}f}"
        )

    leak = std.dropna(subset=["advantage"]) if "advantage" in std else std
    worst = leak.loc[leak["advantage"].idxmax()] if not leak.empty else None

    lines = []
    A = lines.append
    A(f"# {ds} — Yeom MIA (loss-threshold) Analysis\n")
    A(f"*Auto-generated by `run_mia.py`. Raw numbers: `{ds}_mia_results.csv`.*\n")
    A(
        "Yeom, Giacomelli, Fredrikson, Jha (2018), *Privacy Risk in Machine "
        "Learning* (IEEE CSF) — run against the **exported** targets in "
        f"`{ds}/<family>/output/model/`.\n"
    )

    A("## 1. Why Yeom, alongside LiRA")
    A(
        "Yeom's attack is the **average-case** view: one global loss threshold τ "
        "(the mean training loss) applied to every record, scoring a model by the "
        "membership advantage TPR − FPR. It is the natural companion to the "
        "per-example LiRA (`Attack/LiRA`), which surfaces the *worst-case* records "
        "via TPR at a low fixed FPR. The leakage Yeom measures is driven by the "
        "model's generalisation gap, so the loss-gap figure is included as a "
        "mechanism plot.\n"
    )

    A("## 2. What was run")
    A("| Item | Value |")
    A("|------|-------|")
    A(
        f"| Samples | N = {len(data['y_train']) + len(data['y_test'])} "
        f"({len(data['y_train'])} train / {len(data['y_test'])} test) |"
    )
    A(f"| Classes | {n_classes} |")
    A(f"| Features | {data['X_train'].shape[1]} |")
    A(f"| DP budgets ε | {', '.join(f'{e:g}' for e in eps_list)} |")
    A("| Membership ground truth | record ∈ target's training set (X_train) |")
    A("| Per-sample signal | true-class cross-entropy −log p_true |")
    A("| Rule | predict IN if loss ≤ τ = mean training loss |\n")

    A("## 3. How to read it")
    A("| Metric | Meaning | No leak | Leak |")
    A("|--------|---------|---------|------|")
    A("| Advantage | TPR − FPR at τ (Yeom headline) | 0.0 | > 0.1 |")
    A("| Attack AUC | average-case ranking quality | 0.5 | > 0.6 |")
    A("| Loss gap | mean(test loss) − mean(train loss) | 0.0 | ≫ 0.0 |\n")

    A("## 4. Results")
    A("| Model | Variant | ε | Advantage | AUC | TPR | FPR | Loss gap | Test acc |")
    A("|-------|---------|---|-----------|-----|-----|-----|----------|----------|")
    for _, r in df.iterrows():
        eps = "∞" if r["variant"] == "standard" else f"{r['epsilon']:g}"
        if isinstance(r.get("error"), str):
            A(f"| {r['model']} | {r['variant']} | {eps} | ERROR: {r['error']} |")
            continue
        A(
            f"| {r['model']} | {r['variant'].upper() if r['variant'] == 'dp' else 'Standard'} "
            f"| {eps} | {fmt(r.get('advantage'))} | {fmt(r.get('attack_auc'))} "
            f"| {fmt(r.get('tpr'))} | {fmt(r.get('fpr'))} | {fmt(r.get('loss_gap'))} "
            f"| {fmt(r.get('test_acc'))} |"
        )
    A("")

    A("## 5. Figures")
    for title, fn, cap in [
        (
            "Membership advantage vs ε",
            f"{ds}_advantage_vs_epsilon.png",
            "Yeom advantage for each DP model across the budget; dashed = Standard "
            "baseline, dotted = no-leakage (0.0). Lower is more private.",
        ),
        (
            "Attack AUC vs ε",
            f"{ds}_auc_vs_epsilon.png",
            "Average-case ranking quality; 0.5 is random.",
        ),
        (
            "Advantage: Standard vs strong DP",
            f"{ds}_advantage_std_vs_dp_bars.png",
            "Per family, non-private vs strongest-privacy DP membership advantage.",
        ),
        (
            "Generalisation gap vs ε",
            f"{ds}_loss_gap_vs_epsilon.png",
            "The signal Yeom exploits: test − train loss. DP shrinks the gap, which "
            "is why it suppresses the advantage.",
        ),
    ]:
        A(f"### {title}\n")
        A(f"![{title}]({fn})\n")
        A(cap + "\n")

    A("## 6. Interpretation")
    if worst is not None:
        A(
            f"Strongest average-case leakage among the **Standard** models: "
            f"**{worst['model']}** at advantage = **{fmt(worst['advantage'])}** "
            f"(AUC {fmt(worst['attack_auc'])}), vs the 0.0 no-leakage baseline."
        )
    dp_leak = dp.dropna(subset=["advantage"]) if "advantage" in dp else dp
    if not dp_leak.empty:
        worst_dp = dp_leak.loc[dp_leak["advantage"].idxmax()]
        A(
            f"\nUnder DP, the worst case drops to advantage = "
            f"**{fmt(worst_dp['advantage'])}** "
            f"({worst_dp['model']}, ε={worst_dp['epsilon']:g}) — "
            "quantifying the protection DP buys against the average-case attack."
        )
    A(
        "\n> For the worst-case, per-record leakage on the same targets see the "
        "LiRA report in `Attack/LiRA/results/`.\n"
    )

    A("## 7. Reproduce")
    A("```bash")
    A("cd Attack/MIA")
    A(f"python3 run_mia.py --datasets {ds}")
    A("```")

    with open(os.path.join(out, "analysis.md"), "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Yeom MIA: Standard vs DP, membership advantage"
    )
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=em.MODEL_FAMILIES)
    ap.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    ap.add_argument(
        "--verify-determinism",
        action="store_true",
        help="attack every target twice and assert the two runs are identical",
    )
    ap.add_argument("--quick", action="store_true", help="tiny config for smoke test")
    args = ap.parse_args()

    if args.quick:
        args.epsilons = [0.1, 1.0]

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for ds in args.datasets:
        run_dataset(ds, args, rows)

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "mia_comparison.csv"), index=False)
    print(f"\nSaved comparison table -> {os.path.join(OUT_DIR, 'mia_comparison.csv')}")


if __name__ == "__main__":
    main()
