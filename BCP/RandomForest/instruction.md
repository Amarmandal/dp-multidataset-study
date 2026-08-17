## Steps for Standard (non-differentially private) Random Classifier

1. First we will import the dataset from ucimlrepo breastcancer

```py
from ucimlrepo import fetch_ucirepo

# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# metadata
print(breast_cancer_wisconsin_diagnostic.metadata)

# variable information
print(breast_cancer_wisconsin_diagnostic.variables)

```

2. Do the pre-processing steps and prepare it for Standard Random Forest model
3. Train the standard model and evaluate it using standard metrics used in classification like Confusion matrix, F1-score and so on (LLM can choose other relevant matrix)
4. Finally we will save it's result as baseline performance

## Steps for Differentially Private Random Classifier

**Note: we are using diffprivlib for it.**

Sample code snippet

```py
from sklearn.datasets import make_classification
from diffprivlib.models import RandomForestClassifier
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0]]))
```

1. Our job is to prepare winconsin dataset from ucimlrepo for differentially private version.
2. Do all the pre-processing necessary
3. Train using the DF-version.
4. Test and evaluate the performance. Evaluate the performance relative to the standard model (non-private) version.
5. Also, we have to see how the accuracy drops as we try different value of privacy budget (epsilon).
6. Privacy budget on X-axis and Accuracy Loss (Y-axis) is very import.
