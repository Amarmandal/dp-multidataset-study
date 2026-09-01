# The Privacy–Utility Trade-off in Differentially Private Classifiers

Code and results for the **journal extension** of _"An Empirical Study of the
Privacy-Utility Trade-off in Differentially Private Classifiers"_ (CIIT 2026).

The study trains **5 model families** on **6 medical datasets** across a sweep of
**privacy budgets**, then attacks every resulting model with **3 membership inference attacks**
to measure whether Differential Privacy actually reduces leakage — rather than
assuming it does.

|                            |                                                                                                                                                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Datasets**               | 6, spanning N = 319 → 70,692                                                                                                                                                                                              |
| **Model families**         | DP-RF, DP-LR, DP-GNB, DP-SVM, DP-DNN                                                                                                                                                                                      |
| **Privacy budgets**        | DP-RF/LR/GNB/SVM: 15 values, ε ∈ {0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 4.0, 6.0, 8.0, 10.0}<br>DP-DNN: the 9-value subset {0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0, 10.0}, which is also the attack grid |
| **Repeats**                | 30 runs per (dataset, model, ε), seeds `run × 10 + 42`                                                                                                                                                                    |
| **Attacks**                | Yeom (2018), Shokri (2017), LiRA (2022)                                                                                                                                                                                   |
| **Primary utility metric** | ACL (Accuracy Loss) = 1 − Accuracy(M, ε) / Accuracy(M, ε=∞)                                                                                                                                                               |

**Research questions**

1. What is the relationship between accuracy loss and privacy leakage under different privacy budgets?
2. Does the privacy leakage of the non-private model predict or relate to the reduction in privacy leakage achieved by applying DP?

---

## Setup

Python 3.13, managed with [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd code
uv sync                      # creates .venv/ from pyproject.toml + uv.lock
```

Run anything in the project with `uv run`:

```bash
uv run python <script>.py
uv run jupyter lab           # for the notebooks
```

Pinned versions live in [pyproject.toml](pyproject.toml); `uv.lock` fixes the exact
resolution. Key libraries: `diffprivlib` 0.6.6 (DP-RF/LR/GNB), `torch` 2.11 +
`opacus` 1.6 (DP-SGD for the DNN), `scikit-learn` 1.6.1.

> **Trained models are not in this repository.** Every artifact under
> `<DATASET>/<FAMILY>/output/model/` is gitignored — 210 MB of `.pkl`/`.pt` that
> is reproducible from the notebooks. The **results** those models produced (CSV,
> JSON, figures) _are_ tracked, so all published numbers are inspectable without
> them. But the attacks in [Attack/](Attack/) load those model files, so a fresh
> clone must complete Step 1 for a dataset before Step 2 can run on it.

---

## Repository layout

```
<DATASET>/                     BCP, CANCER_RISK, DIABETES, GALLSTONE,
                               KIDNEY_STONE, LUNG_CANCER
  data_preprocessor.ipynb      raw CSV -> data/processed_data.pkl
  data/                        raw dataset + the processed split (tracked)
  RandomForest/ LR/ GaussianNB/ SVM/ DNN/
      STD_<FAM>.ipynb          non-private baseline  (the ε=∞ ceiling)
      DP_<FAM>.ipynb           DP sweep over ε
      output/                  baseline_results.csv     metrics of the STD run
                               dp_results.csv           metrics per ε
                               std_<fam>_report.json    STD run detail
                               dp_<fam>_report.json     DP run detail
        model/                 exported target models   (GITIGNORED)

Attack/                        membership inference, run against exported models
  MIA_YEOM/  MIA_Shokri/  LiRA/
Results/                       cross-dataset consolidation + paper figures
  dataset_results/             UTILITY:  ACL / accuracy / F1 vs ε
  attack_results/              PRIVACY:  leakage vs ε
analysis/                      the paper's tables and figures, + the statistics behind them
  src/                         all 13 scripts
  stats/                       intermediate correlation statistics
  tables/csv/                 the nine paper tables
  figures/pdf/ figures/png/    the paper figures (Fig. 7-13)
  logs/                        verification.txt, gaps.md
  MANIFEST.csv                 one row per artifact, with its sources
notebooks/                     exploratory, read-only scratch analyses

All six datasets share this exact structure, and all 30 dataset x family
directories contain exactly the two notebooks plus output/. The DNN family is
the one naming exception: its metrics files are baseline_dnn_results.csv and
dp_dnn_results.csv.

common_svm.py                  DifferentiallyPrivateSVM, DPSVMOneVsRest
metrics_utils.py               ExtraMetrics (train acc, balanced acc, AUROC, time)
model_exporter.py              export_model() -> output/model/*.pkl
```

---

## How to run — the pipeline, in order

Each step consumes the previous step's output. Run them in this sequence.

### Step 0 — Preprocess

Once per dataset. Produces the split every later step reads.

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    GALLSTONE/data_preprocessor.ipynb
```

Identical procedure in all six datasets: drop identifiers → encode target →
**80/20 stratified split (`random_state=42`), before scaling** →
**`MinMaxScaler(feature_range=(-1, 1))` fit on train only** → set data-independent
DP bounds (`lower = −1 × d`, `upper = +1 × d`) → pickle to
`<DATASET>/data/processed_data.pkl`.

> **Why MinMax and not StandardScaler** — this is a methodological point, not a
> preference. MinMax bounds every feature to [−1, 1] _by construction_, so the
> feature bounds that `diffprivlib` requires cost no privacy budget (under
> StandardScaler they are data-dependent statistics, i.e. an unaccounted leak),
> and `max ‖x‖₂ = √d` is known exactly, making sensitivity calibration exact.

### Step 1 — Train the models

For every **dataset × model family**, run both notebooks:

| Notebook          | Purpose                                                                                                            | Produces                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `STD_<FAM>.ipynb` | non-private **baseline** — how the model performs with no DP; this is the ε=∞ ceiling that ACL is measured against | `baseline_results.csv`, `std_<fam>_report.json`, `output/model/std_<fam>_model.pkl` |
| `DP_<FAM>.ipynb`  | **DP sweep** — the same model across its ε grid (15 values; DP-DNN uses the 9-value subset), 30 runs each          | `dp_results.csv`, `dp_<fam>_report.json`, `output/model/dp_<fam>_model_eps_<ε>.pkl` |

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    GALLSTONE/RandomForest/STD_RF.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace \
    GALLSTONE/RandomForest/DP_RF.ipynb
```

That is 30 (dataset, family) pairs × 2 notebooks. **Run STD before DP** — the DP
notebook reads the baseline accuracy to compute ACL.

**The exported model is the run-0 draw.** Every DP notebook calls `export_model()`
inside `if run == 0:`, so `dp_<fam>_model_eps_<ε>.pkl` is the seed-42 model while
`accuracy_mean` averages all 30 runs. This matters in Step 3: the attacks target
the exported artifact, so leakage must be paired with `ACL_exported`, not `ACL`.

### Step 2 — Run the attacks

Only after **all five families** of a dataset have completed Step 1 — the runners
load every exported model for that dataset.

```bash
uv run python -u Attack/MIA_YEOM/run_mia.py   --datasets GALLSTONE   # ~1 min
uv run python -u Attack/MIA_Shokri/run_mia.py --datasets GALLSTONE   # ~3 min
uv run python -u Attack/LiRA/run_lira.py      --datasets GALLSTONE   # ~50 min
```

> ⚠️ **Scoped-output warning.** Each runner rebuilds the selected dataset's
> authoritative result CSV from only the models in that invocation. A partial
> `--models DNN` run therefore replaces that dataset CSV with DNN rows only.

Use `python -u` — stdout is block-buffered otherwise, hiding progress for the
entire run.

### Step 3 — Consolidate and plot

Aggregation only; nothing is retrained. **All consolidation happens here, at the
repository root** — there is deliberately no per-dataset aggregation step.
`prepare_data.py` walks every `<DATASET>/<FAMILY>/output/` directly and emits one
table covering all six datasets, so `consolidated_data.csv` is the single source
of truth for utility results.

```bash
# cross-dataset UTILITY
cd Results/dataset_results && uv run python prepare_data.py
uv run jupyter nbconvert --to notebook --execute --inplace graph_construction.ipynb

# cross-dataset PRIVACY  (set ATTACK in the notebook's paths cell, one at a time)
cd Results/attack_results && uv run python prepare_mia_data.py
uv run jupyter nbconvert --to notebook --execute --inplace mia_graph_construction.ipynb
```

### Step 4 — Paper tables and figures

Everything the manuscript prints is produced here, from the two consolidated
CSVs of Step 3 plus the per-dataset attack results of Step 2.

```bash
cd analysis/src

uv run python input_audit.py                # audit inputs, no writes
uv run python within_pair_correlations.py   # -> ../stats/
uv run python correlation_statistics.py
uv run python rq_figures.py

uv run python build_tables.py               # -> ../tables/csv/
uv run python verify.py                     # gate: non-zero exit on mismatch
uv run python build_figures.py              # -> ../figures/{pdf,png}/
uv run python make_roc_grid.py
uv run python make_manifest.py              # -> ../MANIFEST.csv
```

**`verify.py` is a gate** — it re-derives every numeric table cell from its
primary source and exits non-zero on any mismatch beyond 1e-06. It currently
reports `checks=2121 failures=186`; all 186 are one known issue with
author-supplied constants, documented in [analysis/README.md](analysis/README.md).
No measured quantity disagrees.

[analysis/README.md](analysis/README.md) lists the **preconditions** — which
upstream files must exist before this stage will produce a complete result, and
which failures are silent. [analysis/RESULTS.md](analysis/RESULTS.md) is the
statistical write-up.

Note this stage never opens an exported model, so it runs on a fresh clone as
soon as the Step 3 CSVs exist — no retraining required.

---

## Where the results live

**Read these files. Do not quote remembered numbers, and prefer CSV over the
`.json` siblings, which have been observed lagging.**

| File                                                                                                 | Contents                                                                           |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| [Results/dataset_results/consolidated_data.csv](Results/dataset_results/consolidated_data.csv)       | **the utility table** — 444 rows (414 DP + 30 baselines) × 31 cols, all 6 datasets |
| [Results/attack_results/consolidated_mia_data.csv](Results/attack_results/consolidated_mia_data.csv) | **the leakage table** — 900 rows = 3 attacks × 300                                 |
| `<DATASET>/<FAMILY>/output/*.csv`, `*.json`                                                          | raw per-model metrics, the input to the two tables above                           |
| `Attack/<ATTACK>/results/<DATASET>/*.csv`                                                            | authoritative per-dataset attack results; LiRA also stores run and ROC coordinates |
| [analysis/tables/csv/](analysis/tables/csv/)                                                         | the nine paper tables, re-derived and verified                                     |
| [analysis/figures/png/](analysis/figures/png/)                                                       | the paper figures (Fig. 7–13)                                                      |
| [analysis/MANIFEST.csv](analysis/MANIFEST.csv)                                                       | every generated artifact and its provenance                                        |

Filter `consolidated_data.csv` on `variant` (`dp` / `standard`) — baselines are
rows in the same file, not a separate block.

### `ACL` vs `ACL_exported` — pick the one that matches the question

| Column         | Numerator                                     | Use for                                           |
| -------------- | --------------------------------------------- | ------------------------------------------------- |
| `ACL`          | `accuracy_mean`, the 30-run mean              | utility-sweep claims — DP's average accuracy cost |
| `ACL_exported` | `exported_model_accuracy`, the run-0 artifact | **any figure pairing utility with leakage**       |

They are not interchangeable. 14% of DP rows differ by >0.05, 5% by >0.10, and
**28 rows disagree on the sign of ACL**. Quoting `ACL` next to an attack number
describes two different models.

---

## Conventions

- **DP** = Differential Privacy; **ACL** = Accuracy Loss
- Greek **ε**, **δ**, **∆f** in formulas — never "epsilon"
- Model names: **DP-RF / DP-LR / DP-GNB / DP-SVM / DP-DNN**
- Non-private models are the **"performance ceiling" / "utility upper bound"**
- Budgets: "relaxed" (ε=10.0) → "strong/strict" (ε ≤ 1.0) → "high-privacy mandate" (ε=0.1)
- Scaling is **MinMaxScaler to [−1, 1]** — never write "StandardScaler" or "µ=0, σ=1"
- Always report **N** when comparing datasets — it is the dominant explanatory variable
- **LUNG_CANCER is the only multi-class dataset** (3 classes). It changes code
  paths: DP-SVM uses One-vs-Rest and splits ε across classes, and attacks pass
  `n_classes=3`. Check it before assuming binary.

---

## Citation

Journal extension of the CIIT 2026 conference paper. Authors: Amar Kumar Mandal,
S M Dedar Alam, Bimbo Lawrence Damitan, Bisola Favour Adediji (Kadir Has
University, Istanbul); Zivko Atanaskoski, Zorica Karapancheva, Mila Dodevska,
Vesna Dimitrova (Ss. Cyril and Methodius University, Skopje).
