"""Within-pair correlations — Spearman correlation between AL and leakage.

For each of the 30 dataset x model pairs, correlate AL against each leakage
metric ACROSS the nine paired epsilons (n=9 per pair). Repeated for all four
leakage metrics. Nothing is dropped or adjusted.

Outputs: analysis/within_pair/pair_correlations.csv, analysis/within_pair/summary.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import (EPSILONS, LEAKAGE_METRICS, build_paired, load_mia,
                    load_utility, spearman_with_ci, summarise)

OUT = Path(__file__).resolve().parents[1] / "stats" / "within_pair"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paired = build_paired(load_utility(), load_mia())
    paired.to_csv(OUT / "paired_input.csv", index=False)

    rows = []
    for (dataset, model), g in paired.groupby(["dataset", "model"], sort=True):
        g = g.sort_values("epsilon")
        for _, _, label in LEAKAGE_METRICS:
            n_expected = len(EPSILONS)
            n_present = int(g[label].notna().sum())
            res = spearman_with_ci(g["AL"], g[label])
            note = res["note"]
            if n_present < n_expected:
                miss = f"{n_expected - n_present} of {n_expected} epsilons missing in {label}"
                note = f"{note}; {miss}" if note else miss
            rows.append({
                "dataset": dataset,
                "model": model,
                "metric": label,
                "rho": res["rho"],
                "n": res["n"],
                "p": res["p"],
                "ci_low": res["ci_low"],
                "ci_high": res["ci_high"],
                "note": note,
            })

    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT / "pair_correlations.csv", index=False)

    summary = []
    for _, _, label in LEAKAGE_METRICS:
        sub = pairs[pairs["metric"] == label]
        s = {"metric": label}
        s.update(summarise(sub["rho"], sub["p"]))
        summary.append(s)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== within-pair correlations summary (per leakage metric, over 30 dataset x model pairs) ===")
    print(summary.to_string(index=False))

    print("\n=== within-pair correlations cross-check anchors (yeom_advantage) ===")
    y = pairs[pairs["metric"] == "yeom_advantage"]
    for ds, md in [("Gallstone", "DP-DNN"), ("Cancer Risk", "DP-LR"),
                   ("Kidney Stone", "DP-DNN")]:
        r = y[(y["dataset"] == ds) & (y["model"] == md)]
        if r.empty:
            print(f"{ds} {md}: NOT PRESENT")
        else:
            r = r.iloc[0]
            print(f"{ds} {md}: rho={r['rho']:+.6f} n={r['n']} p={r['p']:.6g} "
                  f"CI=[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] note={r['note'] or '-'}")

    print("\n=== within-pair correlations all pairs, yeom_advantage ===")
    print(y[["dataset", "model", "rho", "n", "p", "ci_low", "ci_high", "note"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
