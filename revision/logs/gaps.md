# Gaps: what this revision package could not produce without re-running experiments

Every item below is something the referee asked for, or something a generated
artifact would ordinarily contain, that is **not derivable from the data on
disk**. Each entry states what is missing, why, and what it would take to fix.
Nothing here was estimated, interpolated, or filled with a plausible value:
unavailable cells carry the literal string `[MISSING: …]`.

The pipeline that produced this package performs no training, no attack runs, no
notebook execution and no model loading. Where a deliverable required any of
those, it was not attempted.

---

## 1. Requires compute — cannot be corrected post hoc

### 1.1 Balanced 1:1 evaluation sets  [8]

**Status:** not available; the reported leakage metrics come from 4:1 imbalanced
evaluation sets.

Members are the full training split and non-members the full held-out test
split, so every attack was evaluated at a member:non-member ratio of
approximately 4:1 (exactly 4.00 for Cancer Risk, Kidney Stone and Lung Cancer;
3.98 for Gallstone, 3.99 for Breast Cancer, 4.00 for Diabetes — see
`tables/csv/evaluation_set_sizes.csv`).

A `balance_members()` helper exists in `Attack/LiRA/lira.py`, but `--balance`
was never passed on any run: the `balanced` column is `False` on all 300 rows of
`Attack/LiRA/results/*/*_lira_results.csv` (verified — see
`verification.txt`, `evaluation_set_sizes | all | balanced==False on all rows`).

This **cannot be corrected post hoc**. Re-deriving metrics on a balanced subset
requires the per-record membership scores, and those are not persisted — the
attack CSVs store only aggregated means and standard deviations. Fixing this
requires re-running all three attacks with `--balance`.

**Related consequence — the resolution floor on TPR@1%.** Because the
non-member set is the test split, the smallest non-zero false-positive rate that
can be observed is `1/n_nonmembers`. For **Gallstone** that is
1/64 = 0.0156, which **exceeds the 1% reference FPR**: TPR@1% cannot be measured
at its nominal operating point for that dataset, and its values there are the
result of interpolation inside the attack code rather than a directly observed
rate. Breast Cancer is the next tightest at 1/114 = 0.0088, just inside 1%. This
is flagged in the `min_resolvable_fpr_exceeds_1pct` column.

### 1.2 DP-variant log-log ROC curves  [62]

**Status:** the six `loglog_roc_<DS>` figures cover **non-private (standard)
targets only**. No DP-variant ROC curve exists for any dataset.

`_loglog_roc()` in `Attack/LiRA/run_lira.py` explicitly skips every non-standard
target (`if t.get("variant") != "standard" or "_roc_fpr" not in t: continue`),
and the underlying `_roc_fpr` / `_roc_tpr` arrays are stripped from the target
records before the per-dataset CSV is written. Neither the CSVs nor the JSON
reports retain them anywhere in the repository.

Producing DP ROC curves therefore requires re-running LiRA (~50 min/dataset).
The six existing PNGs were relocated rather than regenerated for exactly this
reason.

### 1.3 In-image titles on the six ROC figures  [62][78]

**Status:** attempted, verified unsafe, **originals kept unchanged**.

MDPI will not want the two-line in-image title these PNGs carry. A crop was
attempted with PIL and the result was opened and inspected. It is not safe:

- The title's upper line occupies rows 26–49, followed by a blank band.
- The title's **lower** line occupies rows 56–79.
- The topmost y-axis tick label (`10⁰`) begins at row 77 and runs to the top
  axes spine at row 87.

Rows 77–79 contain **both** the lower title line's descenders and the top of the
tick label, and there is no fully blank row anywhere between them. Consequently
no horizontal crop can remove the whole title without clipping the topmost tick
label — plot content. A first attempt cut at row 52 and removed only the upper
line, leaving the figure still titled; this was confirmed visually and reverted.

All six PNGs are therefore copied byte-for-byte from
`Attack/LiRA/results/<DS>/<DS>_loglog_roc.png` and **still carry their titles**.
The detector records the specific geometry per file in `_figure_gaps.json`.

Options, none of which this pipeline may take: re-run LiRA with the title
suppressed; or crop in an image editor and accept the clipped tick label; or
mask the title region to white rather than cropping (this alters pixels the
pipeline did not generate, so it was not done unilaterally).

### 1.4 Repeated train/test splits  [22]

**Status:** not available. Every result in this study rests on a **single**
80/20 stratified split per dataset, fixed at `random_state=42`.

The 30 repeats recorded in `n_runs` re-seed the *model* (seeds `run*10+42`), not
the split: the same `processed_data.pkl` train/test partition is reused by every
run, every model family and every attack. The reported standard deviations
therefore capture model and mechanism stochasticity **only**, and understate
total variance — they contain no split-induced component.

Quantifying split variance requires re-running preprocessing, training and
attacks across multiple splits: the full pipeline, per split.

### 1.5 DP-matched DNN shadow models  [6]

**Status:** not available.

Shadow models for the Shokri and LiRA attacks are trained non-privately, while
the DP-DNN targets are trained with DP-SGD. The attack therefore models the
non-private loss distribution and applies it to a DP target, which biases the
DP-target leakage estimates — most likely downward, since the shadow
distribution is sharper than the target's.

Correcting this means training DP-matched shadow models (32 shadows per target
for LiRA), which is the single most expensive operation in the study.

### 1.6 Capacity-vs-noise decomposition  [5]

**Status:** not produced; superseded by the A-09 capacity match.

Separating "utility lost to reduced model capacity" from "utility lost to DP
noise" requires training the non-private baseline at DP capacity and the DP
model at baseline capacity — a new arm of experiments. The existing design
instead capacity-matches the non-private baselines to their DP counterparts
(e.g. RF at `n_estimators=20`, `max_depth=4` in both variants), so the reported
ACL already excludes the capacity term by construction. That is a design answer,
not a measurement, and it is worth saying so explicitly in the response letter.

---

## 2. Provenance fields absent from the recorded outputs

These are not compute-blocked — the information was simply never written down.

### 2.1 Software versions  [9][59][61]

**Status:** resolved 2026-08-13 for five of six fields, from this repository's
own `pyproject.toml` / `uv.lock` rather than from any per-model report (no
report JSON records a library or interpreter version):

| Field | Value | Source |
|---|---|---|
| `version_numpy` | 2.4.6 | `pyproject.toml` `==` pin, confirmed in `uv.lock` |
| `version_scikit_learn` | 1.6.1 | `pyproject.toml` `==` pin |
| `version_pytorch` | 2.11.0 | `pyproject.toml` `==` pin |
| `version_opacus` | 1.6.0 | `pyproject.toml` `==` pin |
| `version_diffprivlib` | 0.6.6 | `pyproject.toml` `==` pin |
| `version_python` | 3.13 | `requires-python = ">=3.13"` is a floor only, and `uv.lock` resolves against both cp313 and cp314 wheel sets without recording which one actually ran — confirmed directly by the authors (2026-08-13) as 3.13 |

The five library versions are treated as authoritative because every one is an
exact (`==`) pin, not a range — the lockfile has only one value to resolve to.
`version_python` is not derivable from the lockfile alone (it resolves against
two wheel sets) and is instead recorded on the authors' direct confirmation of
the interpreter used. All six values still describe the *current*
repository's pinned environment, not a captured record of the exact
environment each experiment ran in; if the pins were ever bumped after the
experiments completed, the library versions would be wrong. Written into
`config_inventory` by `revision/src/extract_provenance.py::SOFTWARE_VERSIONS`.

### 2.2 Non-private hyperparameters  [9]

**Status:** resolved 2026-08-13. Was `[MISSING: hyperparameters not recorded in
report JSON]` on **11 of 30** configurations (Gallstone all five; Lung Cancer
RF, LR, GNB, SVM; Cancer Risk GNB; Kidney Stone GNB) — those `std_*_report.json`
files carry metrics but no `parameters` block.

The values were recovered by **reading the corresponding `STD_*.ipynb` as
source** — transcribing the model constructor call. No notebook was executed,
so the no-compute constraint still holds. Recorded in
`revision/src/extract_provenance.py::NOTEBOOK_PARAMS` and consulted only where
the report JSON has nothing; the report always wins where it has a value.

Every cell now carries a `nonprivate_hyperparameters_source` column reading
either `report JSON` or `notebook source (<path>)`, so a referee can separate
what was recorded at run time from what was read back off the code. Current
split: 19 from report JSON, 11 from notebook source, 0 missing.

**Caveat worth keeping:** a notebook records what the code *would* do on its
next run, not a captured log of the run that produced these results. If a
notebook was edited after its results were generated, the transcribed value
would be wrong. Cells sourced from a report JSON do not carry that risk. Two
independent checks support the transcription (see §2.4).

### 2.3 DP hyperparameters  [9]

**Status:** resolved 2026-08-13, same method and same caveat as §2.2. Was
`[MISSING: …]` on **7 of 30** configurations — Gallstone RF, LR, GNB and SVM;
Kidney Stone RF, LR and GNB — all from a missing `parameters` block in the
corresponding `dp_*_report.json`. Recovered from the `DP_*.ipynb` constructor
calls. Current split: 23 from report JSON, 7 from notebook source, 0 missing.

### 2.4 DNN batch sizes  [59]

**Status:** resolved 2026-08-13.

| Configuration | STD batch | DP batch | Source |
|---|---|---|---|
| Gallstone DP-DNN | 32 | 31 | STD from `GALLSTONE/DNN/STD_DNN.ipynb`; DP already in report |
| Lung Cancer DP-DNN | 256 | 256 | both from `LUNG_CANCER/DNN/{STD,DP}_DNN.ipynb` |

The DP notebooks set `batch_size = min(256, max(16, n_train // 8))`. Evaluated
at the `n_train` recorded in each `processed_data.pkl`, that rule gives
**Gallstone 255 → 31** and **Lung Cancer 40 000 → 256**.

The Gallstone figure is a genuine cross-check rather than a restatement: 31 is
independently present in `dp_dnn_report.json`, and the notebook rule reproduces
it exactly. That the two agree where they overlap is the main evidence that the
notebooks match the runs that produced these results. Non-DNN families remain
`n/a (not a minibatch learner)`.

### 2.4a Lung Cancer DP-DNN trains for 30 epochs, not 50  ← **new finding**

Reading these notebooks surfaced a discrepancy that no report JSON records.
`LUNG_CANCER/DNN/DP_DNN.ipynb` sets `num_epochs = 30`, commented
"Reduced for DP training efficiency". Every other DP-DNN notebook, and Lung
Cancer's own `STD_DNN.ipynb`, uses `num_epochs = 50`.

Lung Cancer's DP-DNN is therefore **not** trained under the same budget as its
own non-private baseline, so its ACL confounds the DP noise penalty with 20
fewer epochs of training. This is not captured anywhere in the results files
and is not currently stated in the manuscript. It should either be disclosed as
a deliberate protocol deviation or corrected by a re-run. See §10 of
`CLAUDE.md` — this belongs on the open-issues list.

### 2.5 Dataset provenance: real vs synthetic  [48][49][50]

**Status:** resolved 2026-08-13, supplied directly by the authors rather than
inferred from the source URL (a UCI or Kaggle URL alone establishes neither):

| Dataset | Synthetic? | Provenance |
|---|---|---|
| Gallstone | No | Ankara VM Medical Park Hospital (Clinical Data) |
| Breast Cancer Wisconsin (BCP) | No | University of Wisconsin (FNA Images) |
| Cancer Risk | Yes | Synthetically generated by author |
| Kidney Stone Risk | Unknown | Unknown |
| Lung Cancer | Yes | Unverified (widely considered synthetic) |
| Diabetes (BRFSS 2015) | No | CDC (Behavioral Risk Factor Surveillance System) |

Kidney Stone Risk is recorded as `Unknown` on both columns because the authors
themselves do not know its provenance — that is their answer, not a residual
gap in this pipeline. Written into `dataset_characteristics` by
`revision/src/loaders.py::DATASET_PROVENANCE`, consumed in
`build_tables.py::t_dataset_characteristics`.

---

## 3. Incomplete derived columns

### 3.1 `exported_model_accuracy` / `ACL_exported` blank on 36 DP rows

**Status: RESOLVED 2026-08-16 by direct measurement.** All 414 DP rows of
`consolidated_data.csv` now carry both columns (378 from the report JSON, 36
measured). `n_runs` remains absent on the same 36 rows — see below.

Was blank on the DP-DNN rows of **BCP, Cancer Risk, Diabetes and Gallstone** —
9 epsilons each, 36 of the 414 DP rows. Those four `dp_dnn_report.json` files
predate the notebook line that writes `per_run_accuracies`, and no
`dp_*results.csv` carries per-run values (mean and std only), so the report JSON
was the only *recorded* source.

The value was nonetheless recoverable **without retraining**, because the run-0
draw is not merely described by the report — it is on disk. Every `DP_*.ipynb`
calls `export_model()` inside `if run == 0:`, in the same loop iteration that
appends to `per_run_accuracies`, so `dp_dnn_model_eps_<eps>.{pt,pkl}` *is* the
model whose accuracy `per_run_accuracies[0]` would have recorded. Loading it and
scoring it on the same test split is a **measurement of that same object**, not
a reconstruction, an interpolation or an estimate — the distinction this
document draws elsewhere.

`Results/dataset_results/measure_exported_accuracy.py` performs it, reproducing
the notebooks' own evaluation exactly (`model.eval()`, forward under `no_grad`,
argmax over logits, `accuracy_score` against `y_test` from
`processed_data.pkl`). Output: `exported_model_accuracy_measured.csv`.

**Run-0 provenance is author-confirmed** (2026-08-16), on the same basis as
§2.5: the authors confirm that the exported artifact is the run-0 draw for all
six datasets, which the code independently bears out (`export_model()` is called
inside `if run == 0:` in every `DP_*.ipynb`). The measured CSV therefore carries
`recorded_run0_accuracy = True` on all 54 rows — the flag asserts *what the
number is*, not whether a given run's report happened to log it. A missing log
entry is an absent record, not evidence against the property. The 18 logged
floats are retained separately in `report_json_run0_accuracy` as corroboration.

**The procedure is self-validating.** Kidney Stone and Lung Cancer *do* record
`per_run_accuracies`, so the script measures all six datasets and checks itself
against the recorded value wherever one exists. All **18** such control
configurations reproduced the recorded `per_run_accuracies[0]` to within 1e-9 —
in fact exactly, at a floating-point difference of 0. The script refuses to
write its CSV if any control disagrees. `prepare_data.py` consults the measured
CSV **only** where the report JSON has nothing; the report always wins where it
has a value, and every cell now carries an `exported_model_accuracy_source`
column reading either `report JSON (per_run_accuracies[0])` or `measured from
exported artifact`, so a referee can separate the two.

**Consequence for the figures:** `rq1c_utility_vs_protection_exported` now
carries **30 of 30** pairs, up from 26. Re-running `analysis/d16_rq_figures.py`
moves that correlation from ρ=0.538 (n=26, p=0.0045) to **ρ=0.470 (n=30,
p=0.0088)** — still significant, and still distinct from the `ACL` variant's
ρ=0.557, since the two columns remain non-interchangeable.

The `revision/` package was rebuilt on 2026-08-16 (`build_tables.py` →
`verify.py` → `build_figures.py` → `make_manifest.py`). All nine tables came out
**byte-identical**, so no table in this package depended on the recovered column.
Exactly one figure changed — `rq1c_utility_vs_protection_exported`, now 30
points annotated ρ=0.470 — and the corresponding entry disappeared from
`_figure_gaps.json`, which drops from 8 recorded gaps to 7.

**Residual gap:** `n_runs` is still absent on those 36 rows. Unlike the
accuracy, it is not recoverable by measurement — no artifact encodes it. All
four notebooks set `N_RUNS = 30`, so it could be transcribed from notebook
source under the §2.2 method and caveat, but that is a weaker class of evidence
and was not done unilaterally.

### 3.2 Evaluation-set sizes absent from the Shokri outputs

**Status:** the Shokri CSVs record no `n_members` / `n_nonmembers` column at
all — the field is null on all 300 Shokri rows of `consolidated_mia_data.csv`.

`evaluation_set_sizes` is therefore populated from the LiRA and Yeom CSVs, which
agree with each other **and** with each dataset's `processed_data.pkl`
(`n_train`, `n_test`) on all six datasets. This is a documentation gap, not a
discrepancy: all three attacks demonstrably evaluate on the same splits.

---

## 4. Interpretation notes attached to generated artifacts

Not gaps, but points where a reader could otherwise be misled.

### 4.1 Two different "non-private test accuracy" figures

`consolidated_data.csv` reports `accuracy_mean`, the mean over 30 seeded runs.
The attack CSVs report `test_acc_mean`, the accuracy of the **single exported
run-0 artefact** the attacks actually target. These differ by at most 0.0099
(largest: Breast Cancer RF, 0.9573 vs 0.9474).

They are different estimands, not a conflict, so neither was chosen for the
reader: `baseline_accuracy_matrix` carries **both**, as `test_accuracy` /
`train_accuracy` and `test_accuracy_exported` / `train_accuracy_exported`, with
the train−test gap computed within each source. Any claim pairing utility with
leakage should use the exported columns.

### 4.2 `bound_violations` is not evidence of a broken guarantee

41 of the 270 DP LiRA configurations show a measured TPR@1% above
`e^ε · 0.01`. This should **not** be read as a violated privacy guarantee, and
the table note says so. The bound is stated on one attack's true-positive rate
at one operating point; the reported TPR is a mean over repeated attack runs;
and the finite non-member set limits the resolution of the 1% threshold (§1.1).
The `ci_low_exceeds_bound` column marks the subset where even the lower exact
confidence limit sits above the bound — the only rows worth discussing.

### 4.3 Uncorrected significance

The `p < 0.05` markers and significance counts in `rho_by_epsilon` and
`within_pair_correlations` are reproduced verbatim from the `analysis/` outputs
and are **uncorrected for multiple comparisons**. `rho_by_epsilon` alone reports
108 tests.

---

## CONFLICTS

**None found.**

Every cross-checked quantity agreed to within 1e-6 across its independent
sources. Specifically verified, with 1999 individual cell checks logged in
`verification.txt`:

- All 7800 comparable values in `consolidated_mia_data.csv` match the
  corresponding rows of the 18 per-dataset attack CSVs under `Attack/*/results/`.
- `n_samples`, `n_features`, `n_train` and `n_test` agree between each
  `processed_data.pkl`, `consolidated_data.csv`, and
  `analysis/d15/d15_majority_class_accuracy.csv`.
- Every non-private accuracy in `baseline_accuracy_matrix` matches its
  `std_*_report.json`.
- Evaluation-set sizes agree between the LiRA CSVs, the Yeom CSVs and the
  pickled splits.
- All six correlations in `analysis/d16/d16_correlations.csv` reproduce from
  `analysis/d16/d16_figure_input.csv`.
- 378 DP utility rows carry `n_runs == 30`; the other 36 carry no `n_runs` at
  all (§3.1) — an absent field, not a disagreeing value.

Two stale-duplicate report JSONs exist at `BCP/Result/data/std_lr_report.json`
and `BCP/Result/data/std_rf_report.json` with accuracies (0.9649, 0.9737)
differing from the current `BCP/LR/data/` and `BCP/RandomForest/` files (0.9123,
0.9573). These are **not** treated as a conflict: they sit outside the
model-family directories, are not among this package's declared inputs, and the
loader anchors its glob on `<DATASET>/<FAMILY>/` so they can never be read. They
are noted here only so that nobody later mistakes them for a second opinion.
