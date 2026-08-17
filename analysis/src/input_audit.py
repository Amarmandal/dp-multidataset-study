"""Step 0 — input provenance and repetition-count audit."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from common import MIA_CSV, UTILITY_CSV, load_mia, load_utility


def mtime(p):
    return dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def main():
    u, a = load_utility(), load_mia()
    print("=== Step 0: input files ===")
    for p, df in ((UTILITY_CSV, u), (MIA_CSV, a)):
        print(f"{p.relative_to(p.parents[2])}: mtime={mtime(p)}  rows={len(df)}  cols={df.shape[1]}")

    print("\n=== Step 0: yeom repetition-count audit ===")
    y = a[a["attack"] == "yeom"]
    ydp = y[y["variant"] == "dp"]
    for label, df in (("yeom (all variants)", y), ("yeom (variant==dp)", ydp)):
        nr = int(df["n_runs"].notna().sum())
        ns = int(df["advantage_std"].notna().sum())
        print(f"{label}: rows={len(df)}  non-null n_runs={nr}  non-null advantage_std={ns}")

    nr = int(y["n_runs"].notna().sum())
    ns = int(y["advantage_std"].notna().sum())
    if nr == 0 or ns == 0:
        print("\n*** WARNING ***")
        print("The yeom subset reports NO repetition count and NO dispersion:")
        print(f"  n_runs non-null        = {nr}/{len(y)}")
        print(f"  advantage_std non-null = {ns}/{len(y)}")
        print("Every leakage point is therefore a SINGLE unreplicated estimate with")
        print("no measurable sampling error. All confidence intervals reported below")
        print("propagate only the rank-correlation sampling error across epsilons; they")
        print("IGNORE the (unknown, unreported) measurement error in each leakage value")
        print("and are consequently UNDERSTATED.")

    print("\n=== Step 0: negative-advantage check (must not be floored) ===")
    neg = int((ydp["advantage_mean"] < 0).sum())
    print(f"yeom dp rows with advantage_mean < 0: {neg}/{len(ydp)}  "
          f"range=[{ydp['advantage_mean'].min():.6f}, {ydp['advantage_mean'].max():.6f}]")


if __name__ == "__main__":
    main()
