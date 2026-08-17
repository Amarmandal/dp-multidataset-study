# Diabetes Dataset - Differential Privacy Study

This directory contains the implementation of differentially private classifiers for the **BRFSS 2015 Diabetes Dataset**.

## Dataset Information

- **Name**: Behavioral Risk Factor Surveillance System (BRFSS) 2015
- **Source**: CDC BRFSS 2015 Survey (balanced 50-50 split)
- **Instances**: 70,692 samples
- **Features**: 21 features (health indicators)
- **Target**: Diabetes_binary (0 = No diabetes, 1 = Diabetes/Prediabetes)
- **Split**: 80/20 train-test, random seed = 42

## Features

The dataset contains 21 health-related features:
1. **HighBP**: High blood pressure (0/1)
2. **HighChol**: High cholesterol (0/1)
3. **CholCheck**: Cholesterol check in past 5 years (0/1)
4. **BMI**: Body Mass Index (continuous)
5. **Smoker**: Ever smoked 100 cigarettes (0/1)
6. **Stroke**: History of stroke (0/1)
7. **HeartDiseaseorAttack**: History of coronary heart disease or MI (0/1)
8. **PhysActivity**: Physical activity in past 30 days (0/1)
9. **Fruits**: Consume fruit ≥1 per day (0/1)
10. **Veggies**: Consume vegetables ≥1 per day (0/1)
11. **HvyAlcoholConsump**: Heavy alcohol consumption (0/1)
12. **AnyHealthcare**: Any health care coverage (0/1)
13. **NoDocbcCost**: Could not see doctor due to cost (0/1)
14. **GenHlth**: General health (1-5 scale)
15. **MentHlth**: Mental health days in past 30 (0-30)
16. **PhysHlth**: Physical health days in past 30 (0-30)
17. **DiffWalk**: Difficulty walking or climbing stairs (0/1)
18. **Sex**: Biological sex (0 = female, 1 = male)
19. **Age**: Age category (1-13 scale)
20. **Education**: Education level (1-6 scale)
21. **Income**: Income level (1-8 scale)

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

1. **Data Preprocessing**: Load and preprocess data (standardization via StandardScaler)
2. **Baseline Training**: Train standard (non-private) models
3. **DP Training**: Train differentially private models across epsilon values
4. **Evaluation**: Compare metrics (Accuracy, F1, Precision, Recall, ACL)
5. **Visualization**: Generate privacy-utility tradeoff plots

## Results Location

All results, plots, and comparative analysis will be saved in the `Result/` directory.

## Notes

- Dataset is significantly larger (70K vs 569 samples for breast cancer)
- Larger dataset should result in lower per-record sensitivity
- Expected: Better privacy-utility tradeoff compared to breast cancer dataset
- All models use StandardScaler preprocessing
- Each DP model runs 30 times per epsilon value for statistical robustness
