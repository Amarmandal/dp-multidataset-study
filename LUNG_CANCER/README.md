# Lung Cancer Dataset - Differential Privacy Study

This directory contains the implementation of differentially private classifiers for the **Lung Cancer Risk Level Dataset**.

## Dataset Information

- **Name**: Lung Cancer Risk Level Dataset
- **Source**: `lung_cancer_clean_50k.csv`
- **Instances**: 50,000 samples
- **Features**: 23 features (demographic, lifestyle, and symptom indicators)
- **Target**: `Level` (0 = Low Risk, 1 = Medium Risk, 2 = High Risk)
- **Class Distribution**: Approximately balanced (~30% each)
- **Split**: 80/20 train-test, stratified, random seed = 42

## Features (23 total)

| Feature | Description |
|---|---|
| Age | Patient age |
| Gender | Biological sex (1=Male, 2=Female) |
| Air Pollution | Air pollution exposure level (1-8) |
| Alcohol use | Alcohol consumption (1-8) |
| Dust Allergy | Dust allergy severity (1-8) |
| OccuPational Hazards | Occupational hazard exposure (1-8) |
| Genetic Risk | Genetic risk level (1-7) |
| chronic Lung Disease | Chronic lung disease severity (1-7) |
| Balanced Diet | Diet quality score (1-7) |
| Obesity | Obesity level (1-7) |
| Smoking | Smoking status/severity (1-8) |
| Passive Smoker | Passive smoking exposure (1-8) |
| Chest Pain | Chest pain severity (1-9) |
| Coughing of Blood | Blood in cough severity (1-9) |
| Fatigue | Fatigue level (1-9) |
| Weight Loss | Weight loss severity (1-8) |
| Shortness of Breath | Breathing difficulty (1-9) |
| Wheezing | Wheezing severity (1-8) |
| Swallowing Difficulty | Dysphagia severity (1-8) |
| Clubbing of Finger Nails | Nail clubbing level (1-9) |
| Frequent Cold | Frequency of colds (1-7) |
| Dry Cough | Dry cough severity (1-7) |
| Snoring | Snoring severity (1-7) |

## Models Implemented

### 1. Random Forest (RF)
- **Standard RF**: `RandomForest/STD_RF.ipynb`
- **DP-RF**: `RandomForest/DP_RF.ipynb` — uses `diffprivlib`

### 2. Logistic Regression (LR)
- **Standard LR**: `LR/STD_LR.ipynb`
- **DP-LR**: `LR/DP_LR.ipynb` — uses `diffprivlib`

### 3. Gaussian Naive Bayes (GNB)
- **Standard GNB**: `GaussianNB/STD_GNB.ipynb`
- **DP-GNB**: `GaussianNB/DP_GNB.ipynb` — uses `diffprivlib`

### 4. Support Vector Machine (SVM)
- **Standard SVM**: `SVM/STD_SVM.ipynb` — uses `LinearSVC` (one-vs-rest)
- **DP-SVM**: `SVM/DP_SVM.ipynb` — custom objective perturbation + one-vs-rest multi-class

### 5. Deep Neural Network (DNN)
- **Standard DNN**: `DNN/STD_DNN.ipynb`
- **DP-DNN**: `DNN/DP_DNN.ipynb` — uses `Opacus` (DP-SGD)

## Privacy Budget Range

ε ∈ {0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 4.0, 6.0, 8.0, 10.0}

> Note: DP-DNN uses a reduced epsilon range {0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0, 10.0} due to computational cost.

## Workflow

1. **Data Preprocessing**: Run `data_preprocessor.ipynb` first
   - Drops identifier columns (`index`, `Patient Id`)
   - Scales features to [-1, 1] (MinMaxScaler)
   - Saves `data/processed_data.pkl`
2. **Baseline Training**: Train standard (non-private) models (`STD_*.ipynb`)
3. **DP Training**: Train differentially private models across epsilon values (`DP_*.ipynb`)
4. **Evaluation**: Compare metrics (Accuracy, F1, Precision, Recall, ACL)
5. **Consolidation**: Aggregate across all datasets with `Results/dataset_results/prepare_data.py`, then plot with `graph_construction.ipynb`

## Key Differences from CANCER_RISK

| Property | CANCER_RISK | LUNG_CANCER |
|---|---|---|
| Samples | 1,500 | 50,000 |
| Features | 8 | 24 |
| Classes | 2 (binary) | 3 (multi-class) |
| Target | Diagnosis (0/1) | Level (0/1/2) |
| Class Balance | Imbalanced (63/37%) | Balanced (~33% each) |
| Metrics | Standard F1/Precision/Recall | Weighted F1/Precision/Recall |
| SVM DP | Binary objective perturbation | One-vs-Rest multi-class |
| DNN Architecture | [64, 32] hidden | [128, 64] hidden |
| DNN Batch Size | 32 | 256 |

## Results Location

All results, plots, and comparative analysis are saved in each model's `output/` directory and `Result/` directory.

## Notes

- All multi-class metrics use `average='weighted'` to account for class frequencies
- DP-SVM uses One-vs-Rest (OvR) strategy with individual DP budgets per binary classifier
- Dataset is much larger (50k vs 1.5k) → lower sensitivity per record → better DP utility at same ε
- Per-record sensitivity: ∆f/n where n ≈ 40,000 (training samples)
- Each DP model runs 30 times per epsilon value for statistical robustness
- DNN uses DP-SGD via Opacus; SVM uses custom objective perturbation
- All other models use IBM's diffprivlib
