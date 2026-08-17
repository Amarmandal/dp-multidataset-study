# Shokri Shadow-Model Membership Inference Attack (Standard vs DP)

Implementation of the membership inference attack of **Shokri, Stronati, Song &
Shmatikov, *"Membership Inference Attacks Against Machine Learning Models"*,
IEEE S&P 2017** (arXiv:1610.05820 — see `paper/1610.05820v2 (1).pdf`), applied to
the six medical datasets in this repo:

`GALLSTONE`, `BCP`, `CANCER_RISK`, `KIDNEY_STONE`, `LUNG_CANCER`, `DIABETES`.

The goal is to quantify **how much harder Differential Privacy makes membership
inference** compared with the non-private ("Standard") models, for every model
family studied in the paper.

> This is a *different, stronger* attack than the loss-threshold (Yeom) attack in
> `Attack/MIA/`. It trains shadow models and a learned attack classifier, exactly
> as in Shokri et al., including **Algorithm 1 (data synthesis using the target
> model)**.

---

## Attack pipeline

| Stage | File / function | What it does |
|-------|-----------------|--------------|
| 1. Synthesis | `shokri_mia.synthesize_pool` | **Algorithm 1**: black-box hill-climbing that queries the target's posteriors to grow records it classifies confidently. Run as parallel chains for speed. |
| 2. Shadow models | `shokri_mia.train_shadow_attack_data` | Trains *k* shadow models **of the same kind as the target** on disjoint slices of the synthetic pool; labels each query as `in` (member) / `out` (non-member). |
| 3. Attack model | `shokri_mia.train_attack_models` | One binary `in/out` classifier **per class**, taking the posterior vector as input. |
| 4. Evaluation | `shokri_mia.evaluate_attack` | Applies the attack to the target's **real** members (`X_train`) vs non-members (`X_test`). |

### Algorithm 1 details

`synthesize_pool` reproduces the paper's `SYNTHESIZE(class : c)` routine:
random initialisation (`RandRecord`), accept-if-improved hill climbing, sampling
on high confidence, and `k`-feature re-randomisation that halves `k` after
`rej_max` consecutive rejects. Because medical features here are continuous and
already scaled, `RandRecord` samples uniformly within the per-feature `bounds`
stored in each `processed_data.pkl`.

When a target is so noisy (typically DP at small ε) that hill-climbing cannot
reach `conf_min`, a **relaxed top-up** keeps random records whose target
*argmax* matches the needed class, and — as a last resort — the records with the
highest posterior for that class. This guarantees a usable, non-single-class
shadow pool while remaining "synthesis using the target model".

---

## Models (Standard vs DP)

**Targets** are the already-trained models exported by each dataset's notebook
into `<DATASET>/<Family>/output/model/`, loaded by `exported_models.load_target`
— the attack hits the exact artifacts the paper reports on, never a re-trained
stand-in. **Shadows** are fresh estimators of the same kind, built by
`exported_models.make_shadow_factory`:

| Family | Standard | DP |
|--------|----------|----|
| `LR`  | sklearn `LogisticRegression` | diffprivlib `LogisticRegression` |
| `RF`  | sklearn `RandomForestClassifier` | diffprivlib `RandomForestClassifier` (exponential-mechanism splits) |
| `GNB` | sklearn `GaussianNB` | diffprivlib `GaussianNB` |
| `SVM` | sklearn `SVC(kernel='rbf')` | `common_svm.DifferentiallyPrivateSVM` (Chaudhuri Alg. 2 objective perturbation), or `DPSVMOneVsRest` when multiclass |
| `DNN` | torch MLP | torch MLP (see caveat below) |

DP-SVM shadows come from the same `common_svm` module the notebooks train with,
at the same budget and with the same `normalize=True` / intercept settings, so a
shadow is distributed like the target it calibrates.

The DNN shadows are **non-private even when the target is a DP-DNN** (Opacus
shadow training is prohibitively expensive); treat DNN rows accordingly.

---

## Metrics

| Metric | Meaning | Privacy reading |
|--------|---------|-----------------|
| **Membership advantage** = TPR − FPR | core MIA metric (Yeom def.) | 0 = no leakage, 1 = perfect inference |
| **Attack AUC** | ranking quality of the in/out score | 0.5 = random guessing |
| **Attack accuracy** | balanced member/non-member accuracy | 0.5 = no leakage |
| **Generalisation gap** = train acc − test acc | target overfitting | reported for context — MIA tracks it |

Lower advantage / AUC ⇒ stronger privacy.

---

## Usage

```bash
cd Attack/MIA_Shokri

# Full sweep: all datasets x {LR,RF,GNB,SVM} x {Standard, DP at 6 epsilons}
python run_mia.py

# Subsets / custom config
python run_mia.py --datasets GALLSTONE KIDNEY_STONE --models RF LR
python run_mia.py --epsilons 0.1 1.0 10.0 --dp-runs 5 --n-per-class 300
python run_mia.py --quick          # fast smoke test

# Figures (after run_mia.py)
python plots.py
```

### Outputs (`results/`)

- `shokri_mia_comparison.csv` — tidy table (one row per target config), mean ± std
  over `--dp-runs` seeds.
- `<dataset>_mia.json` — full metrics incl. per-seed runs and target utility.
- `figures/` — advantage/AUC-vs-ε curves per dataset, Standard-vs-DP bar charts,
  and advantage-vs-overfitting scatter (from `plots.py`).

### Key parameters

| Flag | Default | Notes |
|------|---------|-------|
| `--epsilons` | `0.1 0.4 0.8 1.0 2.0 10.0` | matches the paper's budget grid |
| `--dp-runs` | `3` | seeds per DP target (DP is stochastic) |
| `--n-per-class` | `200` | synthetic records per class for the shadow pool |
| `--n-shadow` | `5` | number of shadow models |
| `--conf-min` | `0.4` | Algorithm 1 acceptance confidence |
