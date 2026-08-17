# A-14 — RQ1 correlation analyses (D-14 + D-15)

Scripts: `analysis/common.py`, `analysis/step0_check.py`, `analysis/d14_pair_correlations.py`,
`analysis/d15_correlations.py`. Run with `uv run --project .. python <script>` from `analysis/`.

Inputs (only these two):

| File | mtime | data rows |
|---|---|---|
| `Results/dataset_results/consolidated_data.csv` | 2026-08-12 07:41:29 | 444 (28 cols) |
| `Results/attack_results/consolidated_mia_data.csv` | 2026-08-12 07:41:30 | 900 (45 cols) |

Join: `dataset` + `model` + `epsilon`, both sides `variant=="dp"`, model names mapped
DP-RF/LR/GNB/SVM/DNN → RF/LR/GNB/SVM/DNN, restricted to ε ∈ {0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0, 10.0}.
Result: 270 paired points = 30 dataset × model pairs × 9 budgets, **zero missing cells** in
AL or in any of the four leakage metrics (`d15/d15_paired_input.csv`). AL = the `ACL` column.

---

## Step 0 — repetition-count audit ⚠️

| Subset | rows | non-null `n_runs` | non-null `advantage_std` |
|---|---|---|---|
| yeom, all variants | 300 | **0** | **0** |
| yeom, `variant=="dp"` | 270 | **0** | **0** |

*Source: `step0_check.py` over `consolidated_mia_data.csv`.*

**Loudly: the Yeom leakage points carry no reported repetition count and no dispersion.**
Every leakage value is a single unreplicated estimate with no measurable sampling error.
All CIs in this document propagate only the rank-correlation sampling error across the
nine budgets; they ignore the unknown measurement error inside each leakage point and are
therefore **understated**.

Negatives preserved, not floored: 105 of 270 Yeom dp rows have `advantage_mean < 0`;
range [−0.075735, +0.158885].

### Root cause — traced to the attack drivers, not to consolidation

Verified against the authoritative per-dataset CSVs under each attack's `results/`
folder (the `.json` siblings were not consulted):

| Attack | run count in source CSV | run-level std in source CSV | driver |
|---|---|---|---|
| Yeom | **absent** | **absent** | one `yeom_mia()` call per target, no loop — `Attack/MIA_YEOM/run_mia.py:102`, `:126-131` |
| LiRA | `n_shadow_trained` only | **absent** | one `run_lira()` call per target at fixed `--seed` default 0 — `Attack/LiRA/run_lira.py:116-125`, `:486`, `:495` |
| Shokri | `n_runs` (3 dp / 1 baseline) | full set | `seeds = tuple(range(args.dp_runs))`, `--dp-runs` default 3 — `Attack/MIA_Shokri/run_mia.py:211`, `:263`, aggregated `:146-153` |

`Results/attack_results/prepare_mia_data.py` does **not** drop these columns: every
Yeom and LiRA source column is mapped (`:142-156`, `:242-251`) and `n_runs` is simply
absent from their headers (`:239-240`). The blanks in `consolidated_mia_data.csv` are
faithful to the source.

Two std-like columns that Yeom and LiRA *do* carry are **not** run dispersion and were
not used as such: Yeom `train_loss_std`/`test_loss_std` are spread across *samples*
within one attack; LiRA `sd_in`/`sd_out` are fitted per-example Gaussian widths.
LiRA's 32 shadows sit *inside* a single attack and do not replicate it.

**Consequence:** the dispersion does not exist anywhere in the repository and cannot be
recovered without re-running the Yeom and LiRA attacks over multiple seeds. The primary
metric and two of the three secondary metrics (lira_auc, lira_tpr_at_1pct) rest on
unreplicated leakage points; only shokri_auc rests on replicated ones, and those at n=3.

---

## D-14 — within-pair Spearman (n=9 per pair)

`d14/d14_summary.csv`, over the 30 dataset × model pairs:

| Leakage metric | computable | median ρ | IQR | min ρ | max ρ | sig p<0.05 | sig − | sig + |
|---|---|---|---|---|---|---|---|---|
| yeom `advantage_mean` (primary) | 30/30 | +0.1339 | [−0.6100, +0.4750] | −1.0000 | +0.9052 | 10 | 7 | 3 |
| lira `attack_auc_mean` | 30/30 | −0.1344 | [−0.4208, +0.4658] | −0.9167 | +0.9192 | 8 | 4 | 4 |
| shokri `attack_auc_mean` | 29/30 | −0.1833 | [−0.3667, +0.1833] | −0.8167 | +0.9154 | 2 | 1 | 1 |
| lira `tpr_at_1pct` | 30/30 | −0.0389 | [−0.4167, +0.3084] | −0.7931 | +0.7782 | 4 | 3 | 1 |

Per-pair ρ, n, p, 95% CI (Fisher z, Bonett-Wright SE `sqrt((1+ρ²/2)/(n−3))`) and notes:
`d14/d14_pair_correlations.csv`.

One undefined correlation: **Gallstone DP-DNN / shokri_auc** — the Shokri AUC series is
constant across all nine budgets, so ρ is undefined (`note` column). No other pair is
undefined for any metric.

**Cross-check against your existing run (yeom, primary): matches exactly.**

| Check | Yours | Computed here |
|---|---|---|
| median ρ | +0.134 | +0.133938 |
| IQR | [−0.610, +0.475] | [−0.609978, +0.475000] |
| computable | 30/30 | 30/30 |
| significant | 10 (7 −, 3 +) | 10 (7 −, 3 +) |
| Gallstone DP-DNN | +0.905 | +0.905204, p=7.86e-4, CI [+0.5007, +0.9852] |
| Cancer Risk DP-LR | −0.962 | −0.962352, p=3.29e-5, CI [−0.9945, −0.7653] |
| Kidney Stone DP-DNN | −1.000 | −1.000000, p=0, CI degenerate at −1 |

No disagreement to report.

---

## D-15

### (a) Every correlation with full stats

`d15/d15_all_correlations.csv` — 500 correlations, each with ρ, n, p and 95% CI;
3 undefined (all the Gallstone DP-DNN / shokri_auc constant series, under the three AL
definitions). No bare ρ is emitted anywhere.

Pooled over all 270 paired points (`d15/d15_balanced_accuracy.csv`; note the points are
not independent — 9 budgets nested in 30 pairs):

| Metric | pooled ρ | n | p | 95% CI |
|---|---|---|---|---|
| yeom `advantage_mean` | −0.0707 | 270 | 0.2469 | [−0.1886, +0.0492] |
| lira `attack_auc_mean` | +0.0530 | 270 | 0.3857 | [−0.0669, +0.1714] |
| shokri `attack_auc_mean` | −0.0727 | 270 | 0.2335 | [−0.1906, +0.0472] |
| lira `tpr_at_1pct` | +0.0509 | 270 | 0.4044 | [−0.0689, +0.1694] |

### (b) ρ versus ε — AL vs leakage across the 30 pairs, one row per budget

`d15/d15_rho_by_epsilon.csv`. Primary metric (yeom `advantage_mean`), all 30 pairs:

| ε | ρ | n | p | 95% CI |
|---|---|---|---|---|
| 0.1 | +0.0598 | 30 | 0.7534 | [−0.3074, +0.4115] |
| 0.2 | +0.0607 | 30 | 0.7499 | [−0.3066, +0.4123] |
| 0.4 | +0.0474 | 30 | 0.8036 | [−0.3185, +0.4010] |
| 0.8 | +0.0316 | 30 | 0.8683 | [−0.3325, +0.3875] |
| 1.0 | −0.2942 | 30 | 0.1146 | [−0.5969, +0.0820] |
| 2.0 | +0.1075 | 30 | 0.5718 | [−0.2640, +0.4512] |
| 4.0 | −0.1596 | 30 | 0.3996 | [−0.4934, +0.2152] |
| 8.0 | −0.0563 | 30 | 0.7676 | [−0.4085, +0.3105] |
| 10.0 | −0.0550 | 30 | 0.7729 | [−0.4074, +0.3117] |

The other three metrics are in the same CSV. The only per-budget cell significant at
p<0.05 across all four metrics × nine budgets is lira_auc at ε=0.8 (ρ=+0.3731, p=0.0423,
CI [+0.0019, +0.6539]) — 1 of 36, at the rate expected by chance.

Trend test (`d15/d15_rho_epsilon_trend.csv`), Spearman of the per-budget ρ against ε, n=9:

| Metric | ρ of ρ-vs-ε | p | ρ of \|ρ\|-vs-ε | p |
|---|---|---|---|---|
| yeom `advantage_mean` | −0.5167 | 0.1544 | +0.0667 | 0.8647 |
| lira `attack_auc_mean` | +0.0167 | 0.9661 | −0.0167 | 0.9661 |
| shokri `attack_auc_mean` | −0.3667 | 0.3317 | +0.6833 | 0.0424 |
| lira `tpr_at_1pct` | −0.1667 | 0.6682 | −0.4833 | 0.1875 |

### (c) Robustness — pairs near the majority-class baseline

Majority-class accuracy computed from the label distribution in each
`<DATASET>/data/processed_data.pkl` (`d15/d15_majority_class_accuracy.csv`); the test-split
rate is used, since baseline accuracy in `consolidated_data.csv` is a held-out test number.

| Dataset dir | majority acc (test) | n_test | test class counts |
|---|---|---|---|
| BCP | 0.631579 | 114 | 0:72; 1:42 |
| CANCER_RISK | 0.630000 | 300 | 0:189; 1:111 |
| DIABETES | 0.500035 | 14139 | 0:7070; 1:7069 |
| GALLSTONE | 0.500000 | 64 | 0:32; 1:32 |
| KIDNEY_STONE | 0.608750 | 800 | 0:487; 1:313 |
| LUNG_CANCER | 0.368600 | 10000 | 0:3023; 1:3291; 2:3686 |

Rule: exclude a pair if |non-private baseline accuracy − majority-class accuracy| < 0.05.
**Excluded: 1 of 30 — Gallstone DP-GNB** (baseline 0.546875, majority 0.500000, margin
0.046875). All other 29 pairs clear the threshold by ≥ 0.20. Full per-pair table with
margins and reasons: `d15/d15_robustness_excluded.csv`.

Recomputed ρ-vs-ε on the 29 remaining pairs, primary metric
(`d15/d15_rho_by_epsilon.csv`, `subset=="robustness_excluded_near_majority"`):

| ε | ρ (all 30) | ρ (29 retained) |
|---|---|---|
| 0.1 | +0.0598 | +0.1123 |
| 0.2 | +0.0607 | +0.0709 |
| 0.4 | +0.0474 | +0.0532 |
| 0.8 | +0.0316 | +0.0441 |
| 1.0 | −0.2942 | −0.3119 |
| 2.0 | +0.1075 | +0.0848 |
| 4.0 | −0.1596 | −0.1986 |
| 8.0 | −0.0563 | −0.0488 |
| 10.0 | −0.0550 | −0.0478 |

No per-budget correlation changes sign or crosses p=0.05 except lira_auc at ε=0.8, which
stays significant (ρ=+0.3803, p=0.0418).

### (d) Balanced-accuracy AL

AL rebuilt as (baseline `balanced_accuracy_mean` − dp `balanced_accuracy_mean`) /
baseline `balanced_accuracy_mean`, baseline taken from the `variant=="standard"` rows of
`consolidated_data.csv`. All 270 rows resolve a baseline; no nulls.

Headline RQ1 (within-pair Spearman) side by side (`d15/d15_balanced_accuracy.csv`):

| Metric | AL def. | median ρ | IQR | sig p<0.05 | sig − | sig + |
|---|---|---|---|---|---|---|
| yeom `advantage_mean` | accuracy | +0.1339 | [−0.6100, +0.4750] | 10 | 7 | 3 |
| yeom `advantage_mean` | balanced | +0.1172 | [−0.6605, +0.4750] | 11 | 8 | 3 |
| lira `attack_auc_mean` | accuracy | −0.1344 | [−0.4208, +0.4658] | 8 | 4 | 4 |
| lira `attack_auc_mean` | balanced | −0.1427 | [−0.4208, +0.5054] | 8 | 3 | 5 |
| shokri `attack_auc_mean` | accuracy | −0.1833 | [−0.3667, +0.1833] | 2 | 1 | 1 |
| shokri `attack_auc_mean` | balanced | −0.2167 | [−0.3880, +0.1333] | 2 | 1 | 1 |
| lira `tpr_at_1pct` | accuracy | −0.0389 | [−0.4167, +0.3084] | 4 | 3 | 1 |
| lira `tpr_at_1pct` | balanced | −0.0578 | [−0.4167, +0.2808] | 5 | 4 | 1 |

Pooled, primary metric: accuracy AL ρ=−0.0707 (p=0.2469) vs balanced AL ρ=−0.0616
(p=0.3130), both n=270.

---

## Three plain statements

**1. Within-pair ρ neither supports nor contradicts RQ1 — it fails to establish a
consistent relationship.** The primary-metric median ρ is +0.1339 with an IQR spanning
[−0.6100, +0.4750], i.e. the middle half of the 30 pairs runs from strongly negative to
strongly positive. Ten pairs are individually significant, but they point both ways
(7 negative, 3 positive), including two near-perfect opposite extremes — Gallstone DP-DNN
+0.905 and Kidney Stone DP-DNN −1.000. The other three leakage metrics give medians of
−0.1344, −0.1833 and −0.0389, i.e. the sign of the central tendency is not even stable
across the leakage measure chosen. Pooled over all 270 points, no metric reaches
significance (|ρ| ≤ 0.073, all p ≥ 0.23, all CIs straddling zero). This is a null/mixed
result, not a directional one.

**2. ρ-versus-ε shows no strengthening or weakening with budget.** For the primary metric,
ρ drifts from +0.0598 at ε=0.1 to −0.0550 at ε=10 with a non-monotone excursion to −0.2942
at ε=1.0; the trend of ρ against ε is −0.5167 (p=0.1544) and of |ρ| against ε is +0.0667
(p=0.8647) — neither significant. Not one of the nine per-budget correlations for the
primary metric is significant at p<0.05. Across all four metrics × nine budgets, exactly
one of 36 cells is significant (lira_auc at ε=0.8), which is what chance alone would
produce. The single significant trend statistic (shokri |ρ| vs ε, +0.6833, p=0.0424) is
one test among 24 in that table and is not treated as evidence.

**3. Neither robustness pass changes the headline.** Excluding the one pair whose
non-private baseline sits within 0.05 of majority-class accuracy (Gallstone DP-GNB) shifts
every per-budget ρ by less than 0.04 and changes no sign or significance verdict.
Substituting balanced-accuracy AL moves the primary-metric median ρ from +0.1339 to
+0.1172, the significant count from 10 (7−/3+) to 11 (8−/3+), and the pooled ρ from −0.0707
to −0.0616 — all within the noise of the original estimate. The mixed, non-significant
result is not an artefact of class imbalance or of using raw accuracy.

**Caveat carried forward from Step 0:** the Yeom and LiRA attacks were each run once per
target at a fixed seed, so the leakage side of the primary metric and of both LiRA
secondaries is unreplicated and carries no measurable sampling error; only shokri_auc is
replicated, at n=3. All reported CIs are therefore narrower than the truth. Widening them
can only weaken the conclusions above, not strengthen them — the mixed/null result is
robust to this limitation, but any *positive* claim drawn from these correlations would
not be.
