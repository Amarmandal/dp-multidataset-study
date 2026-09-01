# `analysis/` — statistics, tables and figures for the manuscript

Turns the consolidated experiment results into **nine core tables**, **three
upload-ready manuscript tables**, and **eight figures**, plus the correlation
statistics behind them.

It is a pure post-processing stage. It trains nothing, attacks nothing, loads no
model, and touches nothing outside `analysis/`. Everything it reads is already
on disk when it starts.

```
analysis/
  src/            every script (13 files)
  stats/          intermediate statistics
    within_pair/          per-pair correlations
    correlation_stats/    clustered cross-sectional inference, rho-vs-epsilon,
                          within-pair and robustness analyses
    rq/                   RQ figure input + recorded correlations
  tables/csv/     nine core + three upload-ready manuscript tables
  figures/pdf/    paper figures, vector
  figures/png/    paper figures, 300 dpi raster
  logs/           verification.txt, gaps.md
  MANIFEST.csv    one row per generated artifact, with its sources
```

---

## Preconditions — read this before running

**Nothing here generates its own inputs.** Every script fails, or silently
produces a thinner table, if an upstream artifact is missing. Check these first.

### 1. The two consolidated CSVs must exist and be current

| Required file | Produced by | Expected shape |
|---|---|---|
| `Results/dataset_results/consolidated_data.csv` | `Results/dataset_results/prepare_data.py` | 444 rows (30 baseline + 414 DP) × 31 cols |
| `Results/attack_results/consolidated_mia_data.csv` | `Results/attack_results/prepare_mia_data.py` | 900 rows = 3 attacks × 300 |

These are the whole numeric basis of this stage. If either is stale, everything
downstream is stale and **nothing here will warn you** — the scripts check
shape, not freshness. Re-run the two `prepare_*.py` scripts whenever any
`<DATASET>/<FAMILY>/output/*.csv` or `Attack/*/results/*.csv` has changed.

### 2. The per-dataset attack results must exist

| Required | Produced by |
|---|---|
| `Attack/LiRA/results/<DS>/<DS>_lira_results.csv` | `Attack/LiRA/run_lira.py` |
| `Attack/LiRA/results/<DS>/<DS>_lira_runs.csv` | `Attack/LiRA/run_lira.py` |
| `Attack/LiRA/results/<DS>/<DS>_lira_roc.csv.gz` | `Attack/LiRA/run_lira.py` |
| `Attack/MIA_Shokri/results/<DS>/<DS>_results.csv` | `Attack/MIA_Shokri/run_mia.py` |
| `Attack/MIA_YEOM/results/<DS>/<DS>_mia_results.csv` | `Attack/MIA_YEOM/run_mia.py` |

`<DS>` ∈ {BCP, CANCER_RISK, DIABETES, GALLSTONE, KIDNEY_STONE, LUNG_CANCER} —
**all six**. A partial attack run truncates these and the tables shrink silently.

### 3. The training artifacts must exist

| Required | Produced by | Used for |
|---|---|---|
| `<DS>/data/processed_data.pkl` | `<DS>/data_preprocessor.ipynb` | class balance, `n_train`, `n_test` — nothing else |
| `<DS>/<FAMILY>/output/std_*_report.json`, `dp_*_report.json` | the `STD_`/`DP_` notebooks | hyperparameters, batch size, runtime |

Exported models under `output/model/` are **not** needed here — this stage never
opens one. That matters because they are gitignored: a fresh clone can run
`analysis/` as soon as the two consolidated CSVs are present, without retraining.

`processed_data.pkl` is read through a **restricted unpickler** that returns an
inert stub for every non-numpy class, so the fitted `MinMaxScaler` and
`LabelEncoder` inside are never reconstructed and scikit-learn is never
imported. Only `y_train`, `y_test` and the `X` shapes are touched.

### 4. Ordering within this stage

`build_figures.py` and `build_tables.py` consume `stats/`, so the three
statistics scripts must run **first**. `make_roc_grid.py` reads the six
compressed LiRA ROC-coordinate CSVs directly and can run after LiRA has been
rerun for all datasets.

---

## How to run

From `analysis/src/`, in this order:

```bash
cd analysis/src

# --- 1. statistics -> stats/ ---
uv run python input_audit.py                # provenance + repetition audit, no writes
uv run python within_pair_correlations.py   # -> ../stats/within_pair/
uv run python correlation_statistics.py     # -> ../stats/correlation_stats/
uv run python rq_figures.py                 # -> ../stats/rq/

# --- 2. artifacts -> tables/, figures/ ---
uv run python build_tables.py               # -> ../tables/csv/*.csv
uv run python verify.py                     # -> ../logs/verification.txt   (non-zero exit on failure)
uv run python build_figures.py              # -> ../figures/pdf/*.pdf, ../figures/png/*.png
uv run python make_roc_grid.py              # -> ../figures/{pdf,png}/loglog_roc_grid.*
uv run python verify.py                     # re-check after the figures
uv run python make_manifest.py              # -> ../MANIFEST.csv
```

`loaders.py` prints a read-only shape/coverage report if run directly and writes
nothing — a cheap way to confirm the preconditions above are met.
`extract_provenance.py` is imported by `build_tables.py`; run directly it prints
the parsed configuration inventory.

Needs only `pandas`, `numpy`, `scipy`, `matplotlib` and `pillow`. No GPU, no
torch, no scikit-learn.

### `verify.py` is the gate

It re-derives **every numeric table cell** independently from its primary source
and exits non-zero if any disagree beyond 1e-06. Do not treat a table as final
while it fails.

Current state: `checks=2121  failures=186  notes=1`. **All 186 failures are the
same known issue** — `config_inventory` software versions and
`dataset_characteristics` real/synthetic flags are author-supplied constants,
while `verify.py` still asserts they equal `[MISSING: …]`. No measured quantity
disagrees. Fixing it means either checking those constants "verbatim as
supplied" (the pattern `source_url` already uses) or reverting the injection.

---

## What it produces

### Figures — `figures/pdf/<name>.pdf`, `figures/png/<name>.png`

| File | Manuscript |
|---|---|
| `l4_avg_vs_worst_case` (legacy filename; both axes are average-case) | **Figure 7** |
| `loglog_roc_grid` | **Figure 8** |
| `rq1c_utility_vs_protection_exported` | **Figure 9** |
| `residual_floor_ci` | **Figure 10** |
| `rq2a_benefit_vs_baseline` | **Figure 11** |
| `gap_vs_leakage_vs_N` | **Figure 12** |
| `l6_baseline_leakage_vs_N` | **Figure 13** |
| `gap_vs_leakage` | supporting, not in the manuscript |

Figure 9 uses **`ACL_exported`**, not `ACL`. The attacks target the exported
run-0 artifact, so pairing leakage against the 30-run mean would describe two
different models. Both variants are emitted so the difference is inspectable.

No in-image titles — the manuscript captions carry them. `loglog_roc_grid` is
drawn by this pipeline from the persisted LiRA FPR/TPR coordinates and covers
the representative standard-target curves.

### Tables — `tables/csv/<name>.csv`

`dataset_characteristics` · `baseline_accuracy_matrix` ·
`baseline_leakage_all_pairs` · `evaluation_set_sizes` · `residual_leakage_eps1` ·
`rho_by_epsilon` · `within_pair_correlations` · `bound_violations` ·
`config_inventory`

The primary RQ1 tables `rho_by_epsilon` and `within_pair_correlations` use
`ACL_exported`, paired with leakage from the same exported run-0 artifact.
Thirty-run mean ACL is not used in these primary tables; it is retained only
for the standalone utility landscape and explicitly labeled sensitivity work.

### Correlation inference

Cross-sectional rows contain 30 dataset-model observations but only six
independent datasets. Their descriptive Spearman rho still uses all 30 rows,
while uncertainty treats the dataset as the sampling unit:

- `p`: exact permutation of the six complete outcome-dataset blocks
  (`6! = 720` assignments), retaining model or model-budget strata;
- `ci_low`, `ci_high`: percentile bootstrap that resamples all rows of a
  dataset together (20,000 resamples, seed `20260901`);
- the 29-row near-majority sensitivity subset has unequal blocks and therefore
  uses a labeled CR1 dataset-cluster rank-regression test with `G-1` degrees of
  freedom, while retaining the dataset-cluster bootstrap interval;
- the six-point RF dataset-size analysis uses all `6! = 720` observation-level
  permutations and an observation-level percentile bootstrap interval.

The inference method, number of dataset clusters, permutation count and valid
bootstrap count are persisted in `stats/correlation_stats/*.csv` and
`stats/rq/correlations.csv`. Significance markers remain uncorrected across
the family of budgets and metrics.

### Upload-ready manuscript CSVs

The following files are presentation-ready sources named after the manuscript
table numbers:

| Manuscript table | Upload-ready CSV |
|---|---|
| Table 11 | `tables/csv/table_11_artifact_matched_acl_exported_epsilon_1.csv` |
| Table 12 | `tables/csv/table_12_artifact_matched_acl_exported_all_budgets.csv` |
| Table 13 | `tables/csv/table_13_artifact_matched_acl_exported_within_pair.csv` |

`build_tables.py` regenerates all three upload-ready tables. Table 11 includes
the dataset count and inference-method columns; Table 12 significance stars use
the dataset-block permutation p-values.

---

## Guarantees

- **No invented values.** A quantity not on disk is written literally as
  `[MISSING: <what is needed>]`. Nothing is defaulted, estimated, interpolated
  or back-filled.
- **No silent tie-breaking.** Where two inputs disagree, both are recorded under
  `CONFLICTS` in `logs/gaps.md` and the cell reads `[CONFLICT: see gaps.md]`.
- **Nothing is winsorised, clipped or floored.** Negative Yeom advantages and
  sub-random AUCs are real and are carried through unmodified.
- **Recorded statistics are asserted, not recomputed.** For the RQ figures a
  mismatch against `stats/rq/correlations.csv` aborts the build rather than
  silently replotting.

## `config.py` switches

| Switch | Default | Effect |
|---|---|---|
| `INCLUDE_LUNG_CANCER` | `True` | When `False`, Lung Cancer drops from every table and figure. The 30-pair tables become 25-pair, and the assertion against `correlations.csv` (computed over all 30 pairs) no longer holds, so the RQ figures annotate **recomputed** statistics and say so in `gaps.md`. |
| `UTILITY_LABEL` | `"ACL"` | Display label on utility axes only. Does **not** rename the `ACL` column in any CSV. |
| `RANDOM_FPR` | `0.01` | Reference FPR for TPR@1%. Drives the chance lines, the `ci_excludes_random` flag and the `e^ε · FPR` bound in `bound_violations`. |

Canonical naming — dataset display names, directory names, model family names —
lives in `loaders.py` and nowhere else.

## Gaps

[`logs/gaps.md`](logs/gaps.md) records, with reasons, everything that could not
be produced without re-running experiments: the imbalanced evaluation sets, the
missing DP-variant ROC curves, the single train/test split, the non-DP-matched
DNN shadow models, and every field that resolved to `[MISSING: …]`.

Statistical write-up of the correlation results is in [`RESULTS.md`](RESULTS.md).
