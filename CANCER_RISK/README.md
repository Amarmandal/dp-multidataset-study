# Cancer Risk Dataset - Differential Privacy Study

This directory contains the implementation of differentially private classifiers for the **Cancer Risk Prediction Dataset**.

## Dataset Information

- **Name**: Cancer Risk Prediction Dataset
- **Source**: The_Cancer_data_1500_V2.csv
- **Instances**: 1,500 samples
- **Features**: 8 features (demographic and health indicators)
- **Target**: Diagnosis (0 = No Cancer, 1 = Cancer)
- **Class Distribution**: Imbalanced (~63% No Cancer, ~37% Cancer)
- **Split**: 80/20 train-test, stratified, random seed = 42

## Features

The dataset contains 8 features:
1. **Age**: Patient age (continuous, 20-80)
2. **Gender**: Biological sex (0 = female, 1 = male)
3. **BMI**: Body Mass Index (continuous)
4. **Smoking**: Smoking status (0/1)
5. **GeneticRisk**: Genetic risk level (0 = Low, 1 = Medium, 2 = High)
6. **PhysicalActivity**: Physical activity hours per week (continuous)
7. **AlcoholIntake**: Alcohol intake units per week (continuous)
8. **CancerHistory**: Family cancer history (0/1)

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

ε ∈ {0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 4.0, 6.0, 8.0, 10.0}

## Workflow

1. **Data Preprocessing**: Run `data_preprocessor.ipynb` first (scaling via MinMaxScaler to [-1, 1])
2. **Baseline Training**: Train standard (non-private) models (`STD_*.ipynb`)
3. **DP Training**: Train differentially private models across epsilon values (`DP_*.ipynb`)
4. **Evaluation**: Compare metrics (Accuracy, F1, Precision, Recall, ACL)
5. **Consolidation**: Aggregate across all datasets with `Results/dataset_results/prepare_data.py`, then plot with `graph_construction.ipynb`

## Results Location

All results, plots, and comparative analysis will be saved in the `Result/` directory.

## Notes

- Dataset has 1,500 samples (between BCP at 569 and Diabetes at 70K)
- Imbalanced classes (63/37) — stratified split maintains proportions
- Per-record sensitivity: ∆f/n where n ≈ 1200 (training samples)
- Each DP model runs 30 times per epsilon value for statistical robustness
- DNN uses DP-SGD via Opacus; SVM uses custom objective perturbation
- All other models use IBM's diffprivlib
