# Membership Inference Attack (Shokri) — Findings Report

> **Attack:** Shokri et al. (2017) Shadow-Model MIA · arXiv:1610.05820
> **Datasets:** GALLSTONE · KIDNEY\_STONE · LUNG\_CANCER
> **Models:** LR · RF · GNB · SVM · DNN (Standard + DP variants)
> **DP budgets ε tested:** 0.1, 0.4, 0.8, 1.0, 2.0, 10.0

---

## What Is a Membership Inference Attack?

The attacker queries a trained model (black-box) and tries to answer:

> *"Was this patient's record used to train this model?"*

If yes → **privacy breach**. The attack trains shadow models on synthetic data and learns to classify a record as *member* or *non-member* based on the model's confidence scores.

### Key Metrics

| Metric | No Leak | Strong Leak |
| --- | --- | --- |
| **Attack AUC** | 0.50 (random) | > 0.60 |
| **Membership Advantage** (TPR − FPR) | 0.0 | > 0.10 |
| **Generalisation Gap** (train acc − test acc) | ≈ 0.0 | > 0.10 |

The generalisation gap is the **root cause** — a model that memorises training data reacts differently to members vs. non-members, giving the attacker a signal to exploit.

---

## Dataset Overview

| Dataset | N train | N test | Classes | Features |
| --- | --- | --- | --- | --- |
| GALLSTONE | 255 | 64 | 2 | 38 |
| KIDNEY\_STONE | 3,200 | 800 | 2 | 23 |
| LUNG\_CANCER | 40,000 | 10,000 | 3 | 23 |

---

## Result 1 — GALLSTONE (N = 255, small dataset)

### Standard Model Vulnerability

| Model | Attack AUC | Advantage | Gen. Gap | Verdict |
| --- | --- | --- | --- | --- |
| **RF** | **0.869** | **0.616** | **+0.203** | 🔴 Severely vulnerable |
| DNN | 0.566 | 0.101 | +0.128 | 🟠 Vulnerable |
| SVM | 0.530 | 0.122 | −0.006 | 🟠 Marginal leak |
| GNB | 0.505 | 0.000 | +0.030 | 🟢 No leak |
| LR | 0.431 | 0.000 | −0.001 | 🟢 No leak |

**RF on GALLSTONE is the most vulnerable configuration across all three datasets.** The RF memorises training data completely (train acc = 100%, gen gap = 0.203), giving the attacker a strong signal.

### Effect of DP on RF (the critical case)

| Variant | ε | Attack AUC | Advantage |
| --- | --- | --- | --- |
| Standard | ∞ | 0.869 | 0.616 |
| DP | 0.1 | 0.448 | 0.000 |
| DP | 0.4 | 0.484 | 0.004 |
| DP | 0.8 | 0.467 | 0.002 |
| DP | 1.0 | 0.500 | 0.030 |
| DP | 2.0 | 0.493 | 0.000 |
| DP | 10.0 | 0.480 | 0.000 |

**DP eliminates the attack entirely at ε = 0.1.** AUC drops from 0.869 → 0.448 (below random), and advantage collapses from 0.616 → 0.000. This is DP working exactly as designed — by forcing the model not to memorise, the attacker's signal disappears.

---

## Result 2 — KIDNEY\_STONE (N = 3,200, medium dataset)

### Standard Model Vulnerability

| Model | Attack AUC | Advantage | Gen. Gap | Verdict |
| --- | --- | --- | --- | --- |
| RF | 0.552 | 0.057 | +0.006 | 🟢 No meaningful leak |
| LR | 0.505 | 0.020 | −0.003 | 🟢 No leak |
| GNB | 0.502 | 0.010 | +0.010 | 🟢 No leak |
| DNN | 0.504 | 0.007 | +0.022 | 🟢 No leak |
| SVM | 0.500 | 0.005 | +0.014 | 🟢 No leak |

**No measurable leakage on any model.** The strongest result (RF AUC = 0.552) is statistically indistinguishable from random guessing.

### DP Effect

All DP variants remain at AUC ≈ 0.50 across all ε. Since the baseline is already random, DP adds no measurable additional protection — there was nothing to protect against here.

---

## Result 3 — LUNG\_CANCER (N = 40,000, large dataset)

### Standard Model Vulnerability

| Model | Attack AUC | Advantage | Gen. Gap | Verdict |
| --- | --- | --- | --- | --- |
| LR | 0.502 | 0.003 | 0.000 | 🟢 No leak |
| RF | 0.500 | 0.000 | 0.000 | 🟢 No leak |
| GNB | 0.503 | 0.008 | −0.002 | 🟢 No leak |
| SVM | 0.500 | 0.000 | −0.002 | 🟢 No leak |
| DNN | 0.500 | 0.000 | 0.000 | 🟢 No leak |

**Absolute immunity — every model, every ε, AUC = 0.50.** Zero generalisation gap across the board means there is no membership signal to extract at all.

---

## Cross-Dataset Comparison — Standard RF (the most informative model)

| Dataset | N train | RF AUC | RF Advantage | RF Gen. Gap |
| --- | --- | --- | --- | --- |
| GALLSTONE | 255 | **0.869** | **0.616** | **+0.203** |
| KIDNEY\_STONE | 3,200 | 0.552 | 0.057 | +0.006 |
| LUNG\_CANCER | 40,000 | 0.500 | 0.000 | 0.000 |

Dataset size drives vulnerability. As N grows, per-record sensitivity decreases, models stop memorising, and the attack collapses to random guessing.

---

## Why Standard Models Are "Not Affected Much" — The Key Insight

This is the most important conceptual finding:

**The Shokri attack needs overfitting to work. No overfitting → no signal → no attack.**

On KIDNEY\_STONE and LUNG\_CANCER, models generalise well regardless of whether DP is applied. An attacker querying these models gets the same confidence scores for members and non-members — there is nothing to learn from.

> ⚠️ **Critical caveat:** AUC = 0.50 on a standard model does NOT mean the model is private. It means *this specific attack found no signal on this dataset.* Formal DP guarantees are still needed — LiRA (likelihood-ratio attack) is more sensitive and may reveal leakage where Shokri cannot.

---

## Summary Table — Worst-Case Standard Model per Dataset

| Dataset | Most Vulnerable Model | AUC | Advantage | After DP (ε = 0.1) AUC |
| --- | --- | --- | --- | --- |
| GALLSTONE | RF | 0.869 | 0.616 | 0.448 ✅ |
| KIDNEY\_STONE | RF | 0.552 | 0.057 | 0.499 ✅ |
| LUNG\_CANCER | GNB | 0.503 | 0.008 | 0.500 ✅ |

---

## Key Takeaways for the Paper

1. **Small datasets are the danger zone.** GALLSTONE (N = 255) with RF shows AUC = 0.869 — a severe privacy breach that DP at ε = 0.1 fully eliminates.

2. **Dataset size is a natural MIA defence** — not a substitute for DP. Larger datasets reduce per-record sensitivity, but formal guarantees require DP.

3. **Model architecture matters.** On small data, RF memorises aggressively (ensemble trees, no regularisation pressure) while LR and GNB maintain near-zero gen gaps and stay safe even without DP.

4. **DP is most valuable in the low-data, complex-model regime** — exactly the GALLSTONE/RF scenario common in clinical settings.

5. **Empirical MIA resistance ≠ formal DP guarantee.** A clean Shokri result on large datasets reflects the attack's limitations, not the model's privacy. For a complete privacy audit, pair Shokri with LiRA results (already run — see `Attack/LiRA/results/`).

---

## References

- Shokri, R. et al. (2017). *Membership Inference Attacks Against Machine Learning Models.* IEEE S\&P. arXiv:1610.05820
- Dwork, C. & Roth, A. (2013). *The Algorithmic Foundations of Differential Privacy.*
- Carlini, N. et al. (2022). *Membership Inference Attacks From First Principles (LiRA).* IEEE S\&P.
