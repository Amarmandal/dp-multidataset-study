# Kidney Stone Risk Dataset - Differential Privacy Study

This directory contains the implementation of differentially private classifiers for the **Kidney Stone Risk Prediction Dataset**.

## Dataset Information

- **Name**: Kidney Stone Risk Prediction Dataset
- **Source**: `cleaned_stone.csv`
- **Instances**: 4,000 samples
- **Features**: 23 features (biological, demographic, and lifestyle indicators)
- **Target**: `stone_risk` (0 = No Stone, 1 = Stone)
- **Class Distribution**: Imbalanced (~61% No Stone, ~39% Stone)
- **Split**: 80/20 train-test, stratified, random seed = 42

## Features

The dataset contains 23 preprocessed feature columns:
1. **serum_creatinine**: Blood creatinine levels (continuous)
2. **gfr**: Glomerular filtration rate (continuous)
3. **bun**: Blood urea nitrogen (continuous)
4. **serum_calcium**: Blood calcium levels (continuous)
5. **ana**: Antinuclear antibody status (categorical)
6. **c3_c4**: Complement proteins (continuous)
7. **hematuria**: Blood in urine (categorical/binary)
8. **oxalate_levels**: Urinary oxalate concentration (continuous)
9. **urine_ph**: Urine pH level (continuous)
10. **blood_pressure**: Blood pressure status (categorical)
11. **physical_activity**: Physical activity level (categorical)
12. **diet**: Dietary habits (categorical)
13. **water_intake**: Average daily water intake (continuous)
14. **smoking**: Smoking status (binary)
15. **alcohol**: Alcohol intake status (binary)
16. **painkiller_usage**: Usage of painkillers (binary)
17. **family_history**: Family history of kidney stones (binary)
18. **weight_changes**: Weight changes status (binary)
19. **stress_level**: Self-reported stress level (categorical)
20. **months**: Time indicator (continuous)
21. **cluster**: Clustering label (categorical)
22. **ckd_pred**: Chronic Kidney Disease prediction status (categorical)
23. **ckd_stage**: Chronic Kidney Disease stage (categorical)

## Models Implemented

### 1. Random Forest (RF)
- **Standard RF**: `RandomForest/STD_RF.ipynb`
- **DP-RF**: `RandomForest/DP_RF.ipynb`

### 2. Logistic Regression (LR)
- **Standard LR**: `LR/STD_LR.ipynb`
- **DP-LR**: `LR/DP_LR.ipynb`

### 3. Gaussian Naive Bayes (GNB)
- **Standard GNB**: `GaussianNB/STD_GNB.ipynb`
- **DP-GNB**: `GaussianNB/DP_GNB.ipynb`

### 4. Support Vector Machine (SVM)
- **Standard SVM**: `SVM/STD_SVM.ipynb`
- **DP-SVM**: `SVM/DP_SVM.ipynb`

### 5. Deep Neural Network (DNN)
- **Standard DNN**: `DNN/STD_DNN.ipynb`
- **DP-DNN**: `DNN/DP_DNN.ipynb`

## Privacy Budget Range

The DP models are evaluated across **15 epsilon values** (9 for DNN):
ε ∈ {0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 4.0, 6.0, 8.0, 10.0}

## Workflow

1. **Data Preprocessing**: Run `data_preprocessor.ipynb` (standardization via StandardScaler, label encoding, bounds computation)
2. **Baseline Training**: Train standard (non-private) models (`STD_*.ipynb`)
3. **DP Training**: Train differentially private models across epsilon values (`DP_*.ipynb`)
4. **Evaluation**: Compare metrics (Accuracy, F1, Precision, Recall, ACL)
5. **Consolidation**: Aggregate across all datasets with `Results/dataset_results/prepare_data.py`, then plot with `graph_construction.ipynb`

## Results Location

All results, plots, and comparative analysis are saved in the model `output/` subdirectories and aggregate results are produced in the `Result/` directory.

## Notes

- Dataset has 4,000 samples (larger than CANCER_RISK at 1,500 and smaller than HERAT_DISEASE at 70,000)
- Imbalanced classes (61/39) — stratified split maintains proportions
- Per-record sensitivity: ∆f/n where n ≈ 3,200 (training samples)
- Each DP model runs 30 times per epsilon value for statistical robustness
- DNN uses DP-SGD via Opacus; SVM uses custom objective perturbation
- All other models use IBM's diffprivlib
