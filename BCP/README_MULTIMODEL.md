# Multi-Model Differential Privacy Analysis - BCP Dataset

## Project Overview

This repository contains the complete experimental framework for evaluating differential privacy mechanisms across **four machine learning architectures** on the Breast Cancer Wisconsin (Diagnostic) dataset.

### Research Evolution

- **Conference Paper (CIIT 2026)**: LR vs RF comparison
- **Journal Extension**: Complete 4-model analysis (LR, SVM, RF, GBDT)

---

## Repository Structure

```
BCP/
│
├── 📁 LR/                          # Logistic Regression (Original)
│   ├── STD_LR.ipynb               # Baseline
│   ├── DP_LR.ipynb                # DP analysis
│   └── COMPARE.ipynb              # Comparison plots
│
├── 📁 SVM/                         # Support Vector Machine (NEW ✨)
│   ├── STD_SVM.ipynb              # Baseline
│   ├── DP_SVM.ipynb               # DP analysis (GaussianNB proxy)
│   └── README.md                  # Implementation notes
│
├── 📁 RandomForest/                # Random Forest (Original)
│   ├── STD_RF.ipynb               # Baseline
│   ├── DP_RF.ipynb                # DP analysis
│   └── instruction.md             # Usage guide
│
# Implementation notes
│
├── 📁 Result/                      # Comparative analysis
│   ├── combined_graphs.ipynb     # Multi-model plots
│   └── graph_data.ipynb          # Data aggregation
│
├── 📁 data/                        # Preprocessed data
│   └── processed_data.pkl        # Scaled train/test split
│
├── 📄 Breast_cancer_dataset.csv   # Raw data (UCI ML Repository)
├── 📄 data_preprocessor.ipynb     # Data pipeline
├── 📄 MULTI_MODEL_EXTENSION.md    # Detailed extension plan
├── 📄 EXECUTION_GUIDE.md          # Step-by-step instructions
└── 📄 README_MULTIMODEL.md        # This file
```

---

## Quick Start

### 1. Environment Setup

```bash
# Navigate to project root
cd /path/to/securing-ml-model

# Activate environment
source .venv/bin/activate

# Verify dependencies
pip list | grep -E "diffprivlib|scikit-learn|numpy|pandas"
```

### 2. Run New Models (SVM + GBDT)

#### SVM Analysis (~45 min)

```bash
cd BCP/SVM

# Baseline (5 min)
jupyter nbconvert --to notebook --execute STD_SVM.ipynb --inplace

# DP Sweep (40 min)
jupyter nbconvert --to notebook --execute DP_SVM.ipynb --inplace
```

#### GBDT Analysis (~70 min)

```bash
cd ../GBDT

# Baseline (8 min)
jupyter nbconvert --to notebook --execute STD_GBDT.ipynb --inplace

# DP Sweep (60 min)
jupyter nbconvert --to notebook --execute DP_GBDT.ipynb --inplace
```

### 3. Verify Outputs

```bash
# Check all results exist
find BCP/{SVM,GBDT} -name "*.csv" -o -name "*.json" -o -name "*.png"
```

Expected: 6 CSVs, 6 JSONs, 4 PNGs

---

## Model Comparison Matrix

| Aspect                   | LR                               | SVM                        | RF                                   | GBDT                                     |
| ------------------------ | -------------------------------- | -------------------------- | ------------------------------------ | ---------------------------------------- |
| **Architecture**         | Linear                           | Kernel (RBF)               | Bagging Ensemble                     | Boosting Ensemble                        |
| **DP Mechanism**         | Objective Perturbation           | Bounded Sensitivity        | Exponential Mechanism                | Gradient Perturbation                    |
| **Implementation**       | `diffprivlib.LogisticRegression` | `diffprivlib.GaussianNB`\* | `diffprivlib.RandomForestClassifier` | `diffprivlib.GradientBoostingClassifier` |
| **Expected ACL (ε=1.0)** | ~26%                             | ~15%?                      | ~10%                                 | ~12%?                                    |
| **Optimal ε Range**      | 2.0-10.0                         | TBD                        | 0.8-2.0                              | TBD                                      |
| **Resilience Tier**      | Low                              | Medium                     | High                                 | Medium-High                              |

\*Note: SVM uses GaussianNB as proxy due to lack of native DP-SVM in diffprivlib.

---

## Key Research Questions

### RQ1: Architecture Hierarchy

**Does DP resilience follow a predictable pattern by model type?**

Hypothesis: Ensemble > Non-linear > Linear

Expected ranking:

1. RF (parallel ensemble + output perturbation)
2. GBDT (sequential ensemble + gradient perturbation)
3. SVM (kernel method + bounded sensitivity)
4. LR (linear + objective perturbation)

### RQ2: Ensemble Strategy

**How does parallel bagging (RF) compare to sequential boosting (GBDT) under DP?**

Key differences:

- **RF**: Independent trees → noise averaging → high resilience
- **GBDT**: Dependent trees → error propagation → moderate resilience

### RQ3: Privacy Budget Universality

**Is ε ∈ [0.8, 2.0] optimal across all architectures?**

Current finding (conference paper):

- **LR**: Needs ε ≥ 2.0 for acceptable utility
- **RF**: Optimal at ε ∈ [0.8, 2.0]

Journal extension investigates:

- **SVM**: Expected ε ∈ [1.0, 4.0]?
- **GBDT**: Expected ε ∈ [1.0, 2.0]?

---

## Experimental Protocol

### Common Parameters

```python
# Dataset
N_samples = 569
Train_size = 455 (80%)
Test_size = 114 (20%)
Random_seed = 42

# Preprocessing
Scaler = StandardScaler()  # μ=0, σ=1
Features = 30 (FNA-derived)

# DP Evaluation
Epsilon_values = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4,
                  1.6, 1.8, 2.0, 4.0, 6.0, 8.0, 10.0]
N_runs_per_epsilon = 30
Total_runs = 15 × 30 = 450 per model
```

### Feature Bounds (Shared Across All Models)

Manually specified per-feature lower/upper bounds based on breast cancer clinical ranges.

See: `BCP/{SVM,GBDT}/DP_*.ipynb` → "Define Feature Bounds" section

---

## Output Files

### Per-Model Outputs

#### Baseline Files

- `baseline_results.csv`: Single-row summary (accuracy, F1, precision, recall, ROC-AUC)
- `std_[model]_report.json`: Detailed report with parameters and confusion matrix

#### DP Files

- `dp_results.csv`: 15-row summary (one per epsilon) with mean/std metrics
- `dp_[model]_report.json`: Comprehensive report with per-run data for all epsilons
- `privacy_accuracy_tradeoff.png`: Dual plot (ε vs Accuracy, ε vs ACL)
- `all_metrics_vs_epsilon.png`: Multi-metric line plot with confidence bands

---

## Results Summary (Preliminary)

### Baseline Performance (Non-Private)

| Model | Accuracy | F1-Score | Precision | Recall  | ROC-AUC |
| ----- | -------- | -------- | --------- | ------- | ------- |
| LR    | 0.9649   | 0.9600   | 0.9700    | 0.9238  | 0.9920  |
| SVM   | **TBD**  | **TBD**  | **TBD**   | **TBD** | **TBD** |
| RF    | 0.9675   | 0.9539   | 0.9983    | 0.9135  | 0.9942  |
| GBDT  | **TBD**  | **TBD**  | **TBD**   | **TBD** | **TBD** |

All models expected to be within ~96-97% accuracy range (performance parity).

### ACL at Key Epsilon Values

| Model | ε=0.1   | ε=0.8   | ε=1.0   | ε=2.0   | ε=10.0  |
| ----- | ------- | ------- | ------- | ------- | ------- |
| LR    | 0.354   | 0.301   | 0.263   | 0.263   | 0.127   |
| SVM   | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| RF    | 0.132   | 0.079   | 0.105   | 0.088   | 0.105   |
| GBDT  | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

**Key Insight**: RF maintains <10% ACL even at ε=0.8, while LR requires ε≥2.0 for similar performance.

---

## Statistical Validation Plan

### Tests to Perform

1. **Paired t-test / Wilcoxon**: Compare ACL at same ε across models
2. **Kruskal-Wallis**: Test if median ACL differs significantly across all 4 models
3. **Effect Size (Cohen's d)**: Quantify magnitude of differences
4. **Bonferroni Correction**: Adjust α for multiple comparisons (α' = 0.05/6 = 0.0083)

### Hypotheses

- **H1**: RF has significantly lower ACL than LR at ε < 1.0 (p < 0.05) ✅ **Confirmed**
- **H2**: GBDT has lower ACL than LR but higher than RF at ε < 1.0 (p < 0.05) **Pending**
- **H3**: SVM has intermediate ACL between GBDT and LR (p < 0.05) **Pending**
- **H4**: All models converge to similar ACL at ε ≥ 10.0 (p > 0.05) **Pending**

---

## Journal Paper Contributions

### New Tables

#### Table IV: Multi-Model Baseline Comparison

Complete performance matrix for all 4 models.

#### Table V: ACL Heatmap

Models (rows) × Epsilon values (columns) showing color-coded ACL.

#### Table VI: Optimal Privacy Budget Recommendations

Architecture-specific ε ranges with justifications.

### New Figures

#### Figure 6: Multi-Model ACL Comparison

4-curve line plot showing ACL vs ε for LR, SVM, RF, GBDT on log-scale x-axis.

#### Figure 7: Architectural Resilience Taxonomy

Tiered diagram categorizing models by DP resilience (Tier 1: RF, Tier 2: GBDT/SVM, Tier 3: LR).

#### Figure 8: F1-Score Degradation

Grouped bar chart comparing F1 loss at ε ∈ {0.1, 1.0, 10.0}.

### New Sections

- **Section 4.4**: Extended Model Set (SVM & GBDT architectures)
- **Section 5.3**: Multi-Model Comparative Analysis
- **Section 6.4**: Architectural Taxonomy for DP Classifiers
- **Section 7.2**: Practical Model Selection Guidelines

---

## Implementation Status

| Task                | Status      | Date       |
| ------------------- | ----------- | ---------- |
| LR notebooks        | ✅ Complete | 2024 Q1    |
| RF notebooks        | ✅ Complete | 2024 Q1    |
| SVM notebooks       | ✅ Complete | 2026-05-10 |
| GBDT notebooks      | ✅ Complete | 2026-05-10 |
| Documentation       | ✅ Complete | 2026-05-10 |
| SVM execution       | ⏳ Pending  | TBD        |
| GBDT execution      | ⏳ Pending  | TBD        |
| Results aggregation | ⏳ Pending  | TBD        |
| Statistical tests   | ⏳ Pending  | TBD        |
| Paper draft         | ⏳ Pending  | TBD        |

---

## Known Limitations

### SVM

⚠️ **No native DP-SVM in diffprivlib**: Current implementation uses GaussianNB as a bounded-sensitivity classifier proxy. For production DP-SVM, custom implementations are required (see Chaudhuri et al., 2011; Rubinstein et al., 2012).

### GBDT

⚠️ **Sequential dependency**: Unlike RF's independent trees, GBDT trees are sequential. Privacy budget composition is cumulative, and noise propagates through the error-correction chain. This may lead to higher variance in later boosting rounds.

### Dataset Size

⚠️ **Small sample (N=569)**: High per-record sensitivity due to limited training data. Larger datasets would reduce the "privacy tax" through better noise averaging.

---

## Reproducibility

All experiments are fully reproducible via:

1. **Fixed random seeds**: `seed = run * 10 + 3` for all models
2. **Identical data split**: `train_test_split(random_state=42)`
3. **Shared preprocessing**: All models load from `data/processed_data.pkl`
4. **Version control**: Git commits track all notebook changes
5. **Environment**: Python 3.10+, dependencies in `requirements.txt`

To reproduce:

```bash
git clone <repo>
cd securing-ml-model
pip install -r requirements.txt
bash BCP/EXECUTION_GUIDE.md  # Follow step-by-step
```

---

## Citation

### Conference Paper (CIIT 2026)

```bibtex
@inproceedings{mandal2026empirical,
  title={An Empirical Study of the Privacy-Utility Trade-off in Differentially Private Classifiers},
  author={Mandal, Amar Kumar and Alam, S M Dedar and Damitan, Bimbo Lawrence and Adediji, Bisola Favour and Atanaskoski, Zivko and Karapancheva, Zorica and Dodevska, Mila and Dimitrova, Vesna},
  booktitle={Proceedings of CIIT 2026},
  year={2026},
  organization={Kadir Has University \& Ss. Cyril and Methodius University}
}
```

### Journal Extension (In Preparation)

```bibtex
@article{mandal2026multimodel,
  title={A Multi-Model Analysis of Differential Privacy Resilience Across Machine Learning Architectures},
  author={Mandal, Amar Kumar and [co-authors TBD]},
  journal={[Target Journal TBD]},
  year={2026},
  note={Extended from CIIT 2026 conference paper}
}
```

---

## Contact & Support

**Primary Author**: Amar Kumar Mandal  
**Affiliation**: Kadir Has University, Istanbul, Turkey  
**Email**: [Your email]

**For Technical Issues**:

- Implementation questions → See `EXECUTION_GUIDE.md`
- Model-specific details → Check `{SVM,GBDT}/README.md`
- General questions → Open GitHub issue or contact author

**Project Timeline**:

- Conference submission: [Date]
- Conference acceptance: [Date]
- Journal extension start: 2026-05-10
- Expected journal submission: [Target date]

---

## Acknowledgments

- **Dataset**: UCI Machine Learning Repository (Breast Cancer Wisconsin)
- **DP Library**: IBM diffprivlib (https://github.com/IBM/differential-privacy-library)
- **Funding**: [If applicable]

---

## License

[Specify license: MIT, Apache 2.0, or Academic use only]

---

**Last Updated**: 2026-05-10  
**Version**: 2.0 (Multi-Model Extension)  
**Status**: 🟢 Implementation complete, experiments pending
