"""
Driver: run the LiRA (Carlini et al. 2022) per-example membership inference
attack against the exported Standard vs DP targets, and write a report whose
headline metric is **TPR at a low fixed FPR** — the worst-case leakage number
that the average-case Shokri / Yeom attacks already in the study cannot see.

It reuses the Shokri pipeline's target loaders and shadow factory
(``Attack/MIA_Shokri/exported_models.py``), but trains shadows on **real data
splits** rather than synthetic records.

The target is always the **exported** artifact (``std_<fam>_model.pkl`` /
``dp_<fam>_model_eps_<eps>.pkl``) — the same file whose accuracy produced the ACL
number in the utility pipeline. It is never retrained and never overwritten. The
only randomness in a run is the reseeding of the 32 shadow models, so the run-to-run
spread reported in ``*_std`` is purely shadow-calibration noise around one fixed
target.

Outputs (under Attack/LiRA/results/):
  * lira_comparison.csv                tidy table across datasets
  * results/<dataset>/                 per-dataset CSV (ground truth) +
                                       figures + analysis.md
  No JSON is written.

Usage:
    python run_lira.py --datasets KIDNEY_STONE
    python run_lira.py --datasets KIDNEY_STONE --n-shadow 64
    python run_lira.py --datasets KIDNEY_STONE --balance   # 1:1 members:non-members
    python run_lira.py --runs 30 --dnn-runs 5         # per-family run counts
    python run_lira.py --runs 30 --skip-dnn            # SVM stays capped at its default 5 runs
    python run_lira.py --datasets DIABETES --models SVM --workers 3
    python run_lira.py --quick                       # fast smoke test
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# Reuse the Shokri folder's loaders / shadow factory. The repo root goes on the
# path too: the SVM notebooks import their DP estimator from ``common_svm``, so
# the exported targets pickle as ``common_svm.DifferentiallyPrivateSVM`` and
# pickle.load has to be able to import that module.
SHOKRI_DIR = os.path.join(os.path.dirname(__file__), "..", "MIA_Shokri")
sys.path.insert(0, os.path.abspath(SHOKRI_DIR))
sys.path.insert(0, REPO)
import exported_models as em  # noqa: E402
from lira import run_lira  # noqa: E402

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
# TPR@1%FPR above this = real worst-case leakage (random baseline is 0.01).
LEAK_TPR1 = 0.05


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
def family_runs(fam, args):
    """Return the configured run count for one model family.

    A LiRA run trains ``--n-shadow`` (32) shadows, so the DNN is orders of
    magnitude more expensive than LR/RF/GNB. SVM is also capped separately
    because RBF scoring is costly on non-separable data.
    """
    if fam == "DNN":
        return args.dnn_runs
    if fam == "SVM" and args.svm_runs is not None:
        return args.svm_runs
    return args.runs


# Aggregated across seeds: bare name carries the mean, ``<name>_std`` the spread.
# The bare names are what prepare_mia_data.py already aliases to ``*_mean``.
AGG_KEYS = [
    "attack_auc",
    "advantage",
    "tpr_at_10pct",
    "tpr_at_1pct",
    "tpr_at_0p1pct",
    "sd_in",
    "sd_out",
    "test_acc",
]
# Constant across seeds; taken from the first successful run.
PASS_KEYS = ["n_members", "n_nonmembers", "n_shadow_trained", "online", "balanced"]


def aggregate(runs, family, variant, epsilon):
    """Mean/std across seeds. Mirrors Attack/MIA_Shokri/run_mia.py:133-154."""
    ok = [r for r in runs if "error" not in r]
    row = {"model": family, "variant": variant, "epsilon": epsilon, "n_runs": len(runs)}
    if not ok:
        row["error"] = runs[0].get("error", "unknown")
        return row
    for k in AGG_KEYS:
        vals = [r[k] for r in ok if k in r]
        row[k] = float(np.mean(vals)) if vals else None
        row[f"{k}_std"] = float(np.std(vals)) if vals else None
    for k in PASS_KEYS:
        row[k] = ok[0].get(k)
    # The log-log ROC figure (standard targets only) and the JSON both need the
    # per-target curve; carry the first run's. The CSV strips _roc* downstream.
    for k in ("_roc_fpr", "_roc_tpr"):
        if k in ok[0]:
            row[k] = ok[0][k]
    return row


def attack_target(
    fam, variant, data, n_classes, low, high, args, ds, epsilon=None, seeds=(0,)
):
    view, lo, hi = em.family_dataview(fam, data, low, high)
    n_features = view["X_train"].shape[1]

    # The exported artifact is the fixed target: loaded once, before the seed
    # loop, so every run attacks the identical object. It is never retrained and
    # never written back. Only the shadow models are reseeded per run.
    try:
        target = em.load_target(REPO, ds, fam, variant, epsilon, n_classes)
    except Exception as exc:  # unloadable target -> one error row, run continues
        row = aggregate([{"error": str(exc)}], fam, variant, epsilon)
        row["dataset"] = ds
        return row

    runs = []
    for seed in tuple(seeds):
        try:
            make_shadow = em.make_shadow_factory(
                fam,
                variant,
                epsilon=epsilon,
                bounds=(lo, hi),
                n_features=n_features,
                n_classes=n_classes,
                base_seed=seed,
                dataset=ds,
            )
            m = run_lira(
                target,
                make_shadow,
                view,
                n_classes,
                n_shadow=args.n_shadow,
                online=not args.offline,
                seed=seed,
                balance=args.balance,
            )
            if "error" not in m:
                m["test_acc"] = target_test_acc(target, view)
        except Exception as exc:  # a single bad seed shouldn't sink the run
            m = {"error": str(exc)}
        runs.append(m)

    row = aggregate(runs, fam, variant, epsilon)
    row["dataset"] = ds
    return row


def _attack_target_worker(ds, fam, variant, epsilon, args, seeds):
    """Run one independent target configuration in a child process."""
    label = "standard" if variant == "standard" else f"epsilon={epsilon}"
    print(f"  [{fam}] worker started {label}", flush=True)
    data, n_classes, low, high = load_dataset(ds)
    return attack_target(
        fam,
        variant,
        data,
        n_classes,
        low,
        high,
        args,
        ds,
        epsilon=epsilon,
        seeds=seeds,
    )


def _parallel_targets(ds, fam, args, seeds):
    """Evaluate DP targets and individual standard seeds concurrently."""
    dp_epsilons = [
        eps
        for eps in args.epsilons
        if em.model_exists(REPO, ds, fam, "dp", eps)
    ]
    dp_results = {}
    standard_runs = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {}

        # Submit the cheap DP targets first so their progress is visible while
        # the much slower standard RBF-SVM seeds occupy the pool afterward.
        for epsilon in dp_epsilons:
            future = pool.submit(
                _attack_target_worker,
                ds,
                fam,
                "dp",
                epsilon,
                args,
                seeds,
            )
            pending[future] = ("dp", epsilon)

        # Each standard LiRA seed is independent. Splitting them here preserves
        # the exact seed-level metrics and aggregates them in seed order below.
        for seed in seeds:
            future = pool.submit(
                _attack_target_worker,
                ds,
                fam,
                "standard",
                None,
                args,
                (seed,),
            )
            pending[future] = ("standard_seed", seed)

        for future in as_completed(pending):
            kind, value = pending[future]
            result = future.result()
            if kind == "dp":
                epsilon = value
                dp_results[epsilon] = result
                _print_row(fam, f"eps={epsilon:<4}", result)
                print(f"  [{fam}] completed epsilon={epsilon}", flush=True)
            else:
                seed = value
                standard_runs[seed] = result
                print(
                    f"  [{fam}] completed standard seed={seed} "
                    f"({len(standard_runs)}/{len(seeds)})",
                    flush=True,
                )

    standard = aggregate(
        [standard_runs[seed] for seed in seeds], fam, "standard", None
    )
    standard["dataset"] = ds
    _print_row(fam, "STD  ", standard)
    print(f"  [{fam}] completed standard aggregate", flush=True)
    return [standard] + [dp_results[epsilon] for epsilon in dp_epsilons]


def run_dataset(ds, args, rows):
    data, n_classes, low, high = load_dataset(ds)
    print(
        f"\n{'=' * 72}\n{ds}  (classes={n_classes}, n_train={len(data['y_train'])}, "
        f"n_features={data['X_train'].shape[1]})  [LiRA, {args.n_shadow} shadows]\n{'=' * 72}"
    , flush=True)
    results = []

    for fam in args.models:
        if fam not in em.MODEL_FAMILIES or not em.model_exists(
            REPO, ds, fam, "standard"
        ):
            print(f"  [{fam}] skipped (no exported model)", flush=True)
            continue

        t0 = time.time()
        # Shadow seeds: 0..n-1, distinct per run and per family run count.
        seeds = tuple(range(family_runs(fam, args)))
        print(
            f"  [{fam}] {len(seeds)} run(s), shadow seeds {seeds[0]}..{seeds[-1]}",
            flush=True,
        )
        if ds == "DIABETES" and fam == "SVM" and args.workers > 1:
            print(
                f"  [{fam}] processing standard + {len(args.epsilons)} epsilon "
                f"targets with {args.workers} workers",
                flush=True,
            )
            results.extend(_parallel_targets(ds, fam, args, seeds))
        else:
            print(f"  [{fam}] processing standard", flush=True)
            m = attack_target(
                fam,
                "standard",
                data,
                n_classes,
                low,
                high,
                args,
                ds,
                seeds=seeds,
            )
            results.append(m)
            _print_row(fam, "STD  ", m)

            for eps in args.epsilons:
                if not em.model_exists(REPO, ds, fam, "dp", eps):
                    continue
                print(f"  [{fam}] processing epsilon={eps}", flush=True)
                m = attack_target(
                    fam,
                    "dp",
                    data,
                    n_classes,
                    low,
                    high,
                    args,
                    ds,
                    epsilon=eps,
                    seeds=seeds,
                )
                results.append(m)
                _print_row(fam, f"eps={eps:<4}", m)
        print(f"  [{fam}] done in {time.time() - t0:.1f}s", flush=True)

    df = pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_roc")} for r in results]
    )
    rows.extend(df.to_dict("records"))
    build_report(ds, df, results, n_classes, data)


def _print_row(fam, tag, m):
    if "error" in m:
        print(f"  [{fam}] {tag} ERROR: {m['error']}", flush=True)
    else:
        print(
            f"  [{fam}] {tag} AUC={m['attack_auc']:.3f}  "
            f"TPR@1%={m['tpr_at_1pct']:.3f}  TPR@.1%={m['tpr_at_0p1pct']:.3f}  "
            f"adv={m['advantage']:.3f}",
            flush=True,
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


def _loglog_roc(targets, ds, path):
    """The signature LiRA plot: log-log ROC for the Standard targets."""
    fig, ax = plt.subplots(figsize=(6, 6))
    lo = 1e-3
    ax.plot([lo, 1], [lo, 1], ls="--", color="grey", lw=1, label="random")
    for t in targets:
        if t.get("variant") != "standard" or "_roc_fpr" not in t:
            continue
        fpr = np.asarray(t["_roc_fpr"])
        tpr = np.asarray(t["_roc_tpr"])
        ax.plot(
            np.clip(fpr, lo, 1),
            np.clip(tpr, lo, 1),
            color=COLORS.get(t["model"], "k"),
            label=t["model"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, 1)
    ax.set_ylim(lo, 1)
    ax.set_xlabel("False positive rate (log)")
    ax.set_ylabel("True positive rate (log)")
    ax.set_title(
        f"{ds}: LiRA ROC (Standard targets)\npoints above the diagonal at low FPR = worst-case leakage"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _tpr_bars(df, ds, path):
    models = [m for m in MODELS if m in df["model"].unique()]
    std_v, dp_v = [], []
    for m in models:
        s = df[(df["model"] == m) & (df["variant"] == "standard")]
        dp = df[(df["model"] == m) & (df["variant"] == "dp")].sort_values("epsilon")
        std_v.append(s["tpr_at_1pct"].values[0] if not s.empty else 0)
        dp_v.append(dp["tpr_at_1pct"].values[0] if not dp.empty else 0)
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(models)), 4.5))
    ax.bar(x - 0.2, std_v, 0.4, label="Standard", color="#d62728")
    ax.bar(x + 0.2, dp_v, 0.4, label="DP (ε=min)", color="#1f77b4")
    ax.axhline(0.01, ls=":", color="grey", lw=1, label="random (FPR=1%)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("TPR @ 1% FPR")
    ax.set_title(f"{ds}: worst-case leakage (TPR@1%FPR), Standard vs strong DP")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(ds, df, targets, n_classes, data):
    out = os.path.join(OUT_DIR, ds)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{ds}_lira_results.csv"), index=False)

    _curve(
        df,
        "tpr_at_1pct",
        "TPR @ 1% FPR",
        ds,
        os.path.join(out, f"{ds}_tpr1_vs_epsilon.png"),
        baseline=0.01,
    )
    _curve(
        df,
        "attack_auc",
        "Attack AUC",
        ds,
        os.path.join(out, f"{ds}_auc_vs_epsilon.png"),
        baseline=0.5,
    )
    _loglog_roc(targets, ds, os.path.join(out, f"{ds}_loglog_roc.png"))
    _tpr_bars(df, ds, os.path.join(out, f"{ds}_tpr1_std_vs_dp_bars.png"))

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

    # worst standard leaker by TPR@1%FPR
    leak = std.dropna(subset=["tpr_at_1pct"])
    worst = leak.loc[leak["tpr_at_1pct"].idxmax()] if not leak.empty else None
    n_shadow = (
        int(df["n_shadow_trained"].dropna().max()) if "n_shadow_trained" in df else None
    )
    balanced = bool(df["balanced"].dropna().any()) if "balanced" in df else False
    n_mem = int(df["n_members"].dropna().max()) if "n_members" in df else None
    n_non = int(df["n_nonmembers"].dropna().max()) if "n_nonmembers" in df else None

    lines = []
    A = lines.append
    A(f"# {ds} — LiRA (Likelihood Ratio Attack) Analysis\n")
    A(f"*Auto-generated by `run_lira.py`. Raw numbers: `{ds}_lira_results.csv`.*\n")
    A(
        "LiRA — Carlini, Tramèr, Terzis, Song, Steinke, Jagielski, Erlingsson, "
        "Oprea et al. (2022), *Membership Inference Attacks From First Principles* "
        "(IEEE S&P, arXiv:2112.03404) — run against the **exported** targets in "
        f"`{ds}/<family>/output/model/`.\n"
    )

    A("## 1. Why LiRA, on top of Shokri + Yeom")
    A(
        "Shokri (shadow models) and Yeom (loss threshold) are **average-case** "
        "attacks: a single rule for every record, summarised by AUC / advantage "
        "near 0.5. LiRA is **per-example** — for each record it asks whether the "
        "target's confidence is more consistent with models trained *with* that "
        "record (IN) or *without* it (OUT), via a Gaussian likelihood-ratio test "
        "on logit-scaled confidence. This surfaces the few records that leak "
        "strongly even when the average AUC is ~0.5, so the headline metric is "
        "**TPR at a low fixed FPR**, not AUC.\n"
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
    A(f"| Shadow models / target | {n_shadow} (trained on **real** random halves) |")
    A(f"| DP budgets ε | {', '.join(f'{e:g}' for e in eps_list)} |")
    A("| Membership ground truth | record ∈ target's training set (X_train) |")
    if n_mem and n_non:
        ratio = f"{n_mem / n_non:.2g}:1"
        note = (
            " — members subsampled to the non-member count (`--balance`)"
            if balanced
            else " — the raw train/test split"
        )
        A(f"| Evaluation set | {n_mem} members : {n_non} non-members ({ratio}){note} |")
    A("| Score | online LiRA, log N(φ;μ_in,σ) − log N(φ;μ_out,σ), fixed σ |\n")

    A("## 3. How to read it")
    A("| Metric | Meaning | No leak | Leak |")
    A("|--------|---------|---------|------|")
    A(
        "| TPR @ 1% FPR | members caught while wrongly flagging 1% of non-members | 0.01 | ≫ 0.01 |"
    )
    A(
        "| TPR @ 0.1% FPR | the strict worst-case (limited by #non-members) | 0.001 | ≫ 0.001 |"
    )
    A("| Attack AUC | average-case ranking quality | 0.5 | > 0.6 |")
    A("| Advantage | max(TPR − FPR) over thresholds | 0.0 | > 0.1 |\n")

    A("## 4. Results")
    A(
        "| Model | Variant | ε | AUC | TPR@10% | TPR@1% | TPR@0.1% | Advantage | Test acc |"
    )
    A(
        "|-------|---------|---|-----|---------|--------|----------|-----------|----------|"
    )
    for _, r in df.iterrows():
        eps = "∞" if r["variant"] == "standard" else f"{r['epsilon']:g}"
        if "error" in r and isinstance(r.get("error"), str):
            A(f"| {r['model']} | {r['variant']} | {eps} | ERROR: {r['error']} |")
            continue
        A(
            f"| {r['model']} | {r['variant'].upper() if r['variant'] == 'dp' else 'Standard'} "
            f"| {eps} | {fmt(r.get('attack_auc'))} | {fmt(r.get('tpr_at_10pct'))} "
            f"| {fmt(r.get('tpr_at_1pct'))} | {fmt(r.get('tpr_at_0p1pct'))} "
            f"| {fmt(r.get('advantage'))} | {fmt(r.get('test_acc'))} |"
        )
    A("")

    A("## 5. Figures")
    for title, fn, cap in [
        (
            "LiRA ROC (log-log, Standard targets)",
            f"{ds}_loglog_roc.png",
            "The diagnostic LiRA view. A curve hugging the diagonal at low FPR means "
            "no worst-case leakage; a curve bowing up on the left means specific "
            "records are reliably identified as members.",
        ),
        (
            "TPR@1%FPR vs ε",
            f"{ds}_tpr1_vs_epsilon.png",
            "Worst-case leakage for each DP model across the budget; dashed = Standard "
            "baseline, dotted = random (0.01). Lower is more private.",
        ),
        (
            "Attack AUC vs ε",
            f"{ds}_auc_vs_epsilon.png",
            "Average-case ranking quality; 0.5 is random. Provided for continuity with "
            "the Shokri/Yeom results.",
        ),
        (
            "TPR@1%FPR: Standard vs strong DP",
            f"{ds}_tpr1_std_vs_dp_bars.png",
            "Per family, non-private vs strongest-privacy DP worst-case leakage.",
        ),
    ]:
        A(f"### {title}\n")
        A(f"![{title}]({fn})\n")
        A(cap + "\n")

    A("## 6. Interpretation")
    if worst is not None:
        A(
            f"Strongest worst-case leakage among the **Standard** models: "
            f"**{worst['model']}** at TPR@1%FPR = **{fmt(worst['tpr_at_1pct'])}** "
            f"(AUC {fmt(worst['attack_auc'])}), vs the 0.01 random baseline. "
            "LiRA is a strictly stronger attack than the average-case Shokri/Yeom "
            "runs, so this is the most adversarial membership estimate in the study."
        )
    dp_leak = dp.dropna(subset=["tpr_at_1pct"])
    if not dp_leak.empty:
        worst_dp = dp_leak.loc[dp_leak["tpr_at_1pct"].idxmax()]
        A(
            f"\nUnder DP, the worst case drops to TPR@1%FPR = "
            f"**{fmt(worst_dp['tpr_at_1pct'])}** "
            f"({worst_dp['model']}, ε={worst_dp['epsilon']:g}) — "
            "quantifying the protection DP buys at the per-record level."
        )
    A(
        "\n> Caveat: TPR at very low FPR is bounded by the number of non-members "
        f"({len(data['y_test'])} here), and the DNN shadows are non-private even for "
        "DP-DNN targets (an approximation inherited from the shadow factory). Treat "
        "0.1% FPR as indicative, 1% FPR as the robust worst-case figure.\n"
    )

    A("## 7. Reproduce")
    A("```bash")
    A("cd Attack/LiRA")
    A(
        f"python3 run_lira.py --datasets {ds} --n-shadow {n_shadow or 32}"
        f"{' --balance' if balanced else ''}"
    )
    A("```")

    with open(os.path.join(out, "analysis.md"), "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="LiRA: Standard vs DP, TPR@low-FPR")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=em.MODEL_FAMILIES)
    ap.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    ap.add_argument("--n-shadow", type=int, default=32, help="shadow models per target")
    ap.add_argument(
        "--offline", action="store_true", help="offline LiRA (OUT-only, no IN models)"
    )
    ap.add_argument(
        "--balance",
        action="store_true",
        help="subsample members to the non-member count (1:1 evaluation set)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--runs",
        type=int,
        default=30,
        help="shadow-model reseeds per target for LR/RF/GNB/SVM (target is fixed)",
    )
    ap.add_argument(
        "--dnn-runs",
        type=int,
        default=5,
        help="shadow-model reseeds per DNN target (32 shadows/run makes it costly)",
    )
    ap.add_argument(
        "--svm-runs",
        type=int,
        default=5,
        help="shadow-model reseeds per SVM target (default: 5; overrides --runs)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel target configurations for DIABETES SVM only (default: 1)",
    )
    ap.add_argument(
        "--skip-dnn",
        action="store_true",
        help="drop the DNN family entirely (5 seeds x 32 shadows is ~8.5h)",
    )
    ap.add_argument("--quick", action="store_true", help="tiny config for smoke test")
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be at least 1")

    if args.skip_dnn:
        args.models = [m for m in args.models if m != "DNN"]

    if args.quick:
        args.epsilons = [0.1, 1.0]
        args.n_shadow = 8
        args.runs = min(args.runs, 3)
        args.dnn_runs = min(args.dnn_runs, 2)
        args.svm_runs = min(args.svm_runs, 3)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for ds in args.datasets:
        run_dataset(ds, args, rows)

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "lira_comparison.csv"), index=False)
    print(
        f"\nSaved comparison table -> {os.path.join(OUT_DIR, 'lira_comparison.csv')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
