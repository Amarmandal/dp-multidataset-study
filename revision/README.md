# `revision/` — artifact pipeline for the MDPI major revision

Regenerates the tables, figures and provenance records requested by the
referees, **entirely from data already on disk**. Nothing outside `revision/` is
created, modified, moved or deleted; the rest of the repository, including
`analysis/`, is treated as read-only input.

---

## What this pipeline does

- Reads the two consolidated result CSVs, the eighteen per-dataset attack CSVs,
  the reused `analysis/` outputs, the per-model report JSONs, and the pickled
  train/test splits.
- Emits **nine tables**, each as a CSV and as an `\input`-able MDPI `booktabs`
  LaTeX fragment carrying a `\label{tab:…}`.
- Emits **eight figures** as both PDF and PNG, plus **six relocated** LiRA ROC
  PNGs.
- Re-derives every numeric table cell independently from its primary source and
  logs each check.
- Records everything it could not produce, with reasons, in
  [`logs/gaps.md`](logs/gaps.md).

## What this pipeline does not do

- **No training, no attack runs, no notebook execution, no model loading.** It
  does not import torch, opacus, diffprivlib or any scikit-learn estimator, and
  never opens a `.pkl` under a `model/` directory. Where a deliverable required
  any of those, it was not attempted — it was recorded in `gaps.md`.
- **No invented values.** A quantity that is not on disk is written as the
  literal string `[MISSING: <what is needed>]`. Nothing is defaulted,
  estimated, interpolated or back-filled.
- **No silent tie-breaking.** Where two inputs disagree about the same quantity,
  both are recorded under `CONFLICTS` in `gaps.md` and the cell reads
  `[CONFLICT: see gaps.md]`. (No conflicts were found in this run.)
- **No web access, and no citations, authors or titles of any kind.**
- It does not recompute the `analysis/` statistics it reuses. For the five
  d16-derived figures the recorded correlations are **asserted**; a mismatch
  aborts the build rather than silently replotting.

`DIABETES/` is read only for class counts and split sizes, from
`DIABETES/data/processed_data.pkl`, and for nothing else.

---

## How to re-run

From `revision/src/`, in order:

```bash
cd revision/src

python loaders.py            # optional: shape/coverage report, no writes
python build_tables.py       # -> ../tables/csv/*.csv, ../tables/tex/*.tex
python verify.py             # -> ../logs/verification.txt   (exits non-zero on failure)
python build_figures.py      # -> ../figures/pdf/*.pdf, ../figures/png/*.png
python verify.py             # re-check
python make_manifest.py      # -> ../MANIFEST.csv
```

`extract_provenance.py` is imported by `build_tables.py`; running it directly
prints the parsed configuration inventory. Requires `pandas`, `numpy`, `scipy`,
`matplotlib` and `pillow` — no other dependencies, and no GPU.

`verify.py` is the gate: it exits non-zero if any check fails, and lists the
failures at the top of `logs/verification.txt`. **Do not proceed past a failing
verify.**

Current status of this build:

```
checks = 1999    failures = 0    notes = 1
```

---

## Input inventory

| Path | Used for |
|---|---|
| `Results/dataset_results/consolidated_data.csv` | utility, ACL, ACL_exported, train accuracy, train_time |
| `Results/attack_results/consolidated_mia_data.csv` | all leakage metrics, all three attacks |
| `Attack/LiRA/results/<DS>/<DS>_lira_results.csv` | per-dataset LiRA ground truth |
| `Attack/MIA_Shokri/results/<DS>/<DS>_results.csv` | per-dataset Shokri ground truth |
| `Attack/MIA_YEOM/results/<DS>/<DS>_mia_results.csv` | per-dataset Yeom ground truth |
| `analysis/d14/d14_summary.csv`, `d14_pair_correlations.csv` | reused verbatim |
| `analysis/d15/d15_rho_by_epsilon.csv`, `d15_robustness_excluded.csv`, `d15_balanced_accuracy.csv`, `d15_majority_class_accuracy.csv` | reused verbatim |
| `analysis/d16/d16_figure_input.csv`, `d16_correlations.csv` | reused verbatim |
| `<DS>/data/processed_data.pkl` | class balance, `n_train`, `n_test` **only** |
| `<DS>/<FAMILY>/**/std_*_report.json`, `dp_*_report.json` | hyperparameters, batch size, runtime |
| `Attack/LiRA/results/<DS>/<DS>_loglog_roc.png` | existing figure, relocated |

`<DS>` ∈ {BCP, CANCER_RISK, DIABETES, GALLSTONE, KIDNEY_STONE, LUNG_CANCER}.

Two loading details worth knowing:

- **Report JSONs sit at inconsistent depths** (`BCP/RandomForest/`,
  `BCP/LR/data/`, `*/SVM/output/`), so they are located by globbing beneath
  `<DS>/<FAMILY>/`. Anchoring on the family directory means the stale duplicates
  under `BCP/Result/data/` can never be picked up, and `old_`-prefixed files are
  skipped explicitly.
- **`processed_data.pkl` is read through a restricted unpickler** that returns
  an inert stub for every non-numpy class. The fitted `MinMaxScaler` and
  `LabelEncoder` inside are therefore never reconstructed and scikit-learn is
  never imported. Only `y_train`, `y_test` and the `X` shapes are touched.

---

## Artifacts

`MANIFEST.csv` carries the authoritative list, one row per file, with sources
and manuscript location. Summary:

### Tables — `tables/csv/<name>.csv` and `tables/tex/<name>.tex`

| Table | Rows | Referee item |
|---|---|---|
| `dataset_characteristics` | 6 | [48][49][50] |
| `baseline_accuracy_matrix` | 30 | [11][51] |
| `baseline_leakage_all_pairs` | 30 | [10] |
| `evaluation_set_sizes` | 6 | [8] |
| `residual_leakage_eps1` | 30 | [14][15] |
| `rho_by_epsilon` | 12 | [19] |
| `within_pair_correlations` | 4 | [18] |
| `bound_violations` | 41 | [3][33] |
| `config_inventory` | 30 | [9][59][61] |

The `.tex` files are fragments: `\begin{table}[H]` … `\end{table}` with
`booktabs` rules and a `tablenotes` block, no document preamble. They require
`booktabs` and `url`. Where the LaTeX rendering differs from the CSV (URL
macros, significance stars) the CSV holds the plain value and the `.tex` holds
the marked-up one — no value appears in one that is absent from the other.

### Figures — `figures/pdf/<name>.pdf` and `figures/png/<name>.png`

| Figure | Referee item |
|---|---|
| `gap_vs_leakage` | [51][65] |
| `gap_vs_leakage_vs_N` | [51][65] |
| `residual_floor_ci` | [14][15] |
| `rq1c_utility_vs_protection` | [72][73][75] |
| `rq1c_utility_vs_protection_exported` | [72][73][75] |
| `rq2a_benefit_vs_baseline` | [72][73][75] |
| `l4_avg_vs_worst_case` | [72][73][75] |
| `l6_baseline_leakage_vs_N` | [72][73][75] |
| `loglog_roc_<DS>` ×6 (PNG only, relocated) | [62][78] |

Matplotlib only. No in-image titles on the generated figures — MDPI captions
carry the title — and no draft codes in filenames or images. Utility axes are
labelled with `UTILITY_LABEL`.

The six `loglog_roc_<DS>` PNGs are the **only** artifacts not produced by this
pipeline's own plotting code. They are copies of the existing LiRA figures,
**unchanged**: the title crop was attempted, the result inspected, and found
unsafe, so the two-line in-image title remains. They also cover **standard
(non-private) targets only**. Both points are recorded in `gaps.md` §1.2–1.3 and
on every relevant `MANIFEST.csv` row.

---

## `config.py` switches

| Switch | Default | Effect |
|---|---|---|
| `INCLUDE_LUNG_CANCER` | `True` | When `False`, Lung Cancer is dropped from **every** table and figure, and each `.tex` gains a footnote line stating that it was excluded and why. Note this also drops it from `dataset_characteristics`, and it changes the row counts above (the 30-pair tables become 25-pair). It additionally invalidates the assertion against `d16_correlations.csv`, whose values were computed over all 30 pairs; the five d16 figures then annotate **recomputed** statistics and say so in `gaps.md`. |
| `UTILITY_LABEL` | `"ACL"` | Display label on utility axes only. It does **not** rename the `ACL` column in any CSV. |
| `RANDOM_FPR` | `0.01` | Reference FPR for TPR@1%. Drives the chance lines on the figures, the `min_resolvable_fpr_exceeds_1pct` flag, the `ci_excludes_random` flag, and the `e^ε · FPR` bound in `bound_violations`. |

Canonical naming — dataset display names, directory names and model family names
— lives in `loaders.py` and nowhere else. Do not duplicate it.

---

## Gaps

Read [`logs/gaps.md`](logs/gaps.md) before drafting the response letter. It
documents, with reasons, everything that could not be produced without
re-running experiments: the imbalanced evaluation sets [8], the missing
DP-variant ROC curves [62], the single train/test split [22], the non-DP-matched
DNN shadow models [6], the capacity-vs-noise decomposition [5], every field that
resolved to `[MISSING: …]`, and the `CONFLICTS` section (empty in this run). It
is written to be lifted more or less directly into the letter.
