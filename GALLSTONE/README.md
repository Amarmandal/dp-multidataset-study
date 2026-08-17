# Gallstone Dataset - Differential Privacy Study

This directory contains the implementation of differentially private classifiers for the **Gallstone Status Dataset**.

## Dataset Information

- **Name**: Gallstone Status Dataset
- **Source**: `dataset-uci.csv`
- **Instances**: 319 samples
- **Features**: 38 clinical and demographic parameters
- **Target**: `Gallstone Status` (0 = No Gallstone, 1 = Gallstone)
- **Class Distribution**: 161 (0) / 158 (1) - perfectly balanced
- **Split**: 80/20 train-test, stratified, random seed = 42

## Features

The dataset contains 38 features covering demographic details, body composition indicators, and clinical laboratory metrics:
1. **Age**: Patient age
2. **Gender**: Biological sex (0/1)
3. **Comorbidity**: Presence of other comorbidities
4. **Coronary Artery Disease (CAD)**: CAD status
5. **Hypothyroidism**: Hypothyroidism status
6. **Hyperlipidemia**: Hyperlipidemia status
7. **Diabetes Mellitus (DM)**: Diabetes status
8. **Height**: Height in cm
9. **Weight**: Weight in kg
10. **Body Mass Index (BMI)**: BMI value
11. **Total Body Water (TBW)**
12. **Extracellular Water (ECW)**
13. **Intracellular Water (ICW)**
14. **Extracellular Fluid/Total Body Water (ECF/TBW)**
15. **Total Body Fat Ratio (TBFR) (%)**
16. **Lean Mass (LM) (%)**
17. **Body Protein Content (Protein) (%)**
18. **Visceral Fat Rating (VFR)**
19. **Bone Mass (BM)**
20. **Muscle Mass (MM)**
21. **Obesity (%)**
22. **Total Fat Content (TFC)**
23. **Visceral Fat Area (VFA)**
24. **Visceral Muscle Area (VMA) (Kg)**
25. **Hepatic Fat Accumulation (HFA)**
26. **Glucose**
27. **Total Cholesterol (TC)**
28. **Low Density Lipoprotein (LDL)**
29. **High Density Lipoprotein (HDL)**
30. **Triglyceride**
31. **Aspartate Aminotransferase (AST)**
32. **Alanine Aminotransferase (ALT)**
33. **Alkaline Phosphatase (ALP)**
34. **Creatinine**
35. **Glomerular Filtration Rate (GFR)**
36. **C-Reactive Protein (CRP)**
37. **Hemoglobin (HGB)**
38. **Vitamin D**

## Models Implemented

### 1. Random Forest (RF)
- **Standard RF**: `RandomForest/STD_RF.ipynb`
- **DP-RF**: `RandomForest/DP_RF.ipynb` (using IBM diffprivlib)

### 2. Logistic Regression (LR)
- **Standard LR**: `LR/STD_LR.ipynb`
- **DP-LR**: `LR/DP_LR.ipynb` (using IBM diffprivlib)

### 3. Gaussian Naive Bayes (GNB)
- **Standard GNB**: `GaussianNB/STD_GNB.ipynb`
- **DP-GNB**: `GaussianNB/DP_GNB.ipynb` (using IBM diffprivlib)

### 4. Support Vector Machine (SVM)
- **Standard SVM**: `SVM/STD_SVM.ipynb`
- **DP-SVM**: `SVM/DP_SVM.ipynb` (using Huber loss + objective perturbation)

### 5. Deep Neural Network (DNN)
- **Standard DNN**: `DNN/STD_DNN.ipynb`
- **DP-DNN**: `DNN/DP_DNN.ipynb` (using PyTorch + Opacus DP-SGD)

## Privacy Budget Range

ε ∈ {0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 4.0, 6.0, 8.0, 10.0}

## Workflow

1. **Data Preprocessing**: Run `data_preprocessor.ipynb` first (standardization via StandardScaler, split, and feature bounds computation).
2. **Baseline Training**: Train standard (non-private) models (`STD_*.ipynb`) to establish baseline parameters.
3. **DP Training**: Train differentially private models across epsilon values (`DP_*.ipynb`).
4. **Evaluation**: Compare metrics (Accuracy, F1, Precision, Recall, ACL).
5. **Visualization**: Run `Result/compare_all_models.py` for comprehensive comparison across architectures.

## Results Location

All results, plots, and comparative analysis are saved in the `Result/output/` directory.

## Notes

- Dataset is small (319 samples) but perfectly balanced (161 / 158).
- Each DP model runs 30 times per epsilon value for statistical robustness.
- DNN uses DP-SGD via Opacus; SVM uses custom objective perturbation.
- All other models use IBM's diffprivlib.
