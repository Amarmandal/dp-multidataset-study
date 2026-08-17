# LiRA Membership Inference: Methodology, Implementation, and Findings

*An evaluation of the privacy leakage of Standard vs. Differentially Private (DP) classifiers under the strongest known membership inference attack.*

This document accompanies the study **"An Empirical Study of the Privacy-Utility Trade-off in Differentially Private Classifiers"** (CIIT 2026). It explains what LiRA is, how it works, how we implemented it against our exported models, and what the results tell us about which models leak — and how much DP buys back.

---

## 1. What is LiRA?

**LiRA — the Likelihood Ratio Attack** — was introduced by Carlini, Tramèr, Terzis, Song, Steinke, Jagielski, Erlingsson, Oprea et al. in *Membership Inference Attacks From First Principles* (IEEE S&P 2022, arXiv:2112.03404). It is currently the strongest practical **membership inference attack (MIA)**.

A membership inference attack answers one question: *was this specific record part of the model's training set?* In a medical setting this is a direct privacy breach — confirming that a patient's record was used to train a "gallstone diagnosis" model reveals that the patient was a gallstone patient. This is exactly the threat model (MIA) that Differential Privacy is designed to defend against.

### Why LiRA, on top of the Shokri / Yeom attacks already in the study

The earlier attacks in this study are **average-case**:

| Attack | Idea | Summary metric |
|--------|------|----------------|
| **Yeom (2018)** | Threshold on the per-record loss | AUC / advantage |
| **Shokri (2017)** | Train shadow models, then an attack classifier on confidence vectors | AUC / advantage |

Both apply *one rule to every record* and report a single number near 0.5. They answer "on average, can you guess membership?" — and the honest answer is usually "barely."

LiRA is **per-example**. For each individual record it builds a calibrated statistical test, asking whether the model's confidence on that record looks like a model trained **with** the record (IN) or **without** it (OUT). This surfaces the handful of records that leak *catastrophically* even when the average AUC sits at 0.5 — the patients who are genuinely exposed. That is why the headline metric here is **not AUC** but:

> **TPR @ a low fixed FPR** — how many true members the attacker confirms while almost never falsely accusing a non-member. The random-guessing baseline at 1% FPR is **0.01**; anything well above that is real, worst-case leakage.

---

## 2. How LiRA works (methodology)

For a target model `f` and a candidate record `(x, y)`:

1. **Train shadow models.** Train many models of the *same family* on random halves of the real data. For each shadow, the record `x` is IN the training half (≈ half the shadows) or OUT (the other half).
2. **Collect a confidence signal.** For every shadow, compute the model's confidence on the true class for `x` and map it to a stable scale with the **logit transform**
   `φ = log( p / (1 − p) )`,
   where `p` is the predicted probability of the correct class. The logit makes the distribution of confidences approximately Gaussian.
3. **Fit two Gaussians.** From the shadow confidences, estimate
   - `N(μ_in, σ)` — the distribution of `φ` when `x` **was** trained on, and
   - `N(μ_out, σ)` — the distribution when it **was not**.
4. **Likelihood-ratio test on the real target.** Query the actual target model to get its confidence `φ_target` on `x`, and score the record by
   `Λ = log N(φ_target; μ_in, σ) − log N(φ_target; μ_out, σ)`.
   A high `Λ` means the target behaves like it memorized `x` ⇒ likely a member.
5. **Sweep the decision threshold** over all records to trace an ROC curve, and read off **TPR at fixed low FPR (1%, 0.1%)**, the attack AUC, and the membership advantage.

This is the **online** variant (both IN and OUT shadows are trained), which is the most powerful configuration.

---

## 3. How we implemented it here

The driver is [`run_lira.py`](run_lira.py); the attack core is `lira.py`. Key implementation choices:

| Aspect | Choice in this study |
|--------|----------------------|
| **Targets attacked** | The **exported, already-trained** models from each dataset's `<family>/output/model/` directory — i.e. the exact artifacts the paper reports on, not re-trained copies. |
| **Model families** | LR, RF, GNB, SVM, and (where exported) DNN — both the Standard (ε = ∞) and DP variants. |
| **Shadow training** | 32 shadow models per target, trained on **real random halves** of the dataset (not synthetic records). Shadows reuse the Shokri pipeline's shadow factory (`Attack/MIA_Shokri/exported_models.py`). |
| **DP budgets ε** | 0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0, 10.0 — the same grid as the main study. |
| **Membership ground truth** | A record is a *member* iff it is in the target's `X_train`. Non-members are drawn from the held-out `X_test`. |
| **Confidence → score** | Online LiRA, logit-scaled confidence, fixed shared σ. |
| **Mode** | Online (IN + OUT shadows). |

**One honest caveat baked into the numbers:** the DNN shadow models are non-private even when attacking a DP-DNN target (an approximation inherited from the shared shadow factory), and TPR at 0.1% FPR is bounded by the number of non-members. We therefore treat **1% FPR as the robust worst-case figure** and 0.1% FPR as merely indicative.

### Datasets attacked

| Dataset | Train / Test | Classes | Features |
|---------|--------------|---------|----------|
| GALLSTONE | 255 / 64 | 2 | 38 |
| BCP | 455 / 114 | 2 | 30 |
| CANCER_RISK | 1,200 / 300 | 2 | 8 |
| KIDNEY_STONE | 3,200 / 800 | 2 | 23 |
| LUNG_CANCER | 40,000 / 10,000 | 3 | 23 |
| DIABETES | 56,553 / 14,139 | 2 | 21 |

Note the ~222× spread in training-set size from GALLSTONE to DIABETES — this turns out to be the single most important variable in the results.

### Outputs produced

- `results/<DATASET>_lira.json` — full per-target metrics incl. ROC arrays
- `results/lira_comparison.csv` — tidy cross-target table
- `results/<DATASET>/` — `*_lira_results.csv`, four figures (log-log ROC, TPR@1% vs ε, AUC vs ε, Std-vs-DP bars), and a per-dataset `analysis.md`

---

## 4. Results

> **Results pending regeneration.**
>
> The dataset set for this study changed: KIDNEY_FUNCTION, STROKE_RISK and
> HERAT_DISEASE were removed, and BCP, CANCER_RISK and DIABETES are now part of
> the study. Every result table, key finding and comparison figure that
> previously lived in this section was computed on the earlier four-dataset run
> and no longer describes what the pipeline attacks, so they have been removed
> rather than left in place to be misread.
>
> Re-run the attack across the six datasets (Section 7) and the comparison
> figures (`make_comparison_figures.py`), then write this section and the key
> findings from `results/lira_comparison.csv`.

---

## 5. Reproduce

```bash
cd Attack/LiRA
python3 run_lira.py --n-shadow 32          # all six study datasets (the default)
python3 make_comparison_figures.py         # cross-dataset comparison plots
```

Per-dataset figures and auto-generated notes live in `results/<DATASET>/analysis.md`. Raw metrics are in `results/<DATASET>/<DATASET>_lira_results.csv` and the JSON files.

---

*Reference: N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, F. Tramèr. "Membership Inference Attacks From First Principles." IEEE S&P 2022. arXiv:2112.03404.*
