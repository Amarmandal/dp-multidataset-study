# Referee-Item Resolution Tracker

Status labels: `Not started`, `In progress`, `Blocked`, `Complete`.

## 1. Repair or re-prove the DP-SVM mechanism, use data-independent preprocessing, and rerun affected models

**Status:** In progress  
**Last updated:** 2026-08-19

### Referee requirement

Repair or re-prove the DP-SVM mechanism, use data-independent preprocessing,
and rerun affected models.

### Completed

- Audited the DP-SVM and standard-SVM preprocessing paths for BCP, Cancer
  Risk, Diabetes, Gallstone, Kidney Stone, and Lung Cancer.
- Verified that each dataset preprocessor fits
  `MinMaxScaler(feature_range=(-1, 1))` on `X_train` and transforms `X_test`
  with the fitted scaler.
- Verified that all six DP-SVM notebooks independently project every training
  and test row onto the unit L2 ball:

  ```python
  X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1.0)
  ```

- Confirmed mathematically and against the stored data that the projected
  training and test records satisfy `||x_i||_2 <= 1` (up to floating-point
  tolerance).
- Identified and removed the additional data-dependent normalization in
  `common_svm.py` that divided all records by the largest training-row norm.
- Changed `DifferentiallyPrivateSVM(normalize=True)` to use the deterministic,
  record-wise unit-ball projection during both fitting and inference.
- Removed the global-scale adjustment of learned weights. Retained `scale_ =
  1.0` and the legacy privacy-report field as compatibility aids.
- Added an Algorithm 2 conformance test suite covering the Huber loss,
  `c = 1/(2h)`, epsilon-prime branches, Delta fallback, noise-radius
  distribution, analytic gradient, unit-ball enforcement, record-independent
  projection, and consistent inference projection.
- Validation result: `12 passed` using `.venv/bin/pytest -q`.

### Current preprocessing path

```text
raw data
-> train/test split
-> MinMaxScaler fitted on training data
-> row-wise unit-L2-ball projection
-> common_svm.py record-wise projection (idempotent safety check)
-> DP-SVM model
```

### Remaining work

- Resolve the separate unregularized-intercept/strong-convexity issue before
  claiming direct conformance with the theorem (for example, use
  `fit_intercept=False` in the theorem-backed implementation).
- Rerun all affected DP-SVM notebooks for the six datasets.
- Regenerate exported DP-SVM model artifacts because the normalization
  semantics in `common_svm.py` changed.
- Regenerate DP-SVM metrics, privacy reports, plots, and downstream attack
  results that depend on those model artifacts.
- Update manuscript methods and referee-response text with the repaired
  preprocessing definition, theorem assumptions, test evidence, and rerun
  results.
- Mark this item `Complete` only after the implementation issue, reruns, and
  manuscript updates are finished and verified.

### Evidence and files

- `common_svm.py`
- `tests/test_dp_svm_algorithm2.py`
- `tests/test_dp_svm_huber.py`
- `tests/test_dp_svm_assumptions.py`
- `tests/conftest.py`
- `BCP/SVM/DP_SVM.ipynb`
- `CANCER_RISK/SVM/DP_SVM.ipynb`
- `DIABETES/SVM/DP_SVM.ipynb`
- `GALLSTONE/SVM/DP_SVM.ipynb`
- `KIDNEY_STONE/SVM/DP_SVM.ipynb`
- `LUNG_CANCER/SVM/DP_SVM.ipynb`

### Decisions and notes

- Tests verify implementation-facing equations and assumptions; they do not,
  by themselves, prove differential privacy.
- Notebook-level and model-level row projection are intentionally idempotent.
  Keeping the model-level projection makes inference consistent for callers
  that provide MinMax-scaled but not yet projected records.
- Previously exported DP-SVM models should not be treated as repaired
  artifacts; they must be regenerated.

---

## 2. Decide whether the paper claims end-to-end DP

**Status:** Not started

If yes, privatize/fix preprocessing and define the neighboring-world audit. If
no, remove the formal TPR-ceiling interpretation.

### Completed

- None yet.

### Remaining work

- To be added.

---

## 3. Recompute RQ1 with artifact-matched utility and leakage measurements

**Status:** Not started

### Completed

- None yet.

### Remaining work

- To be added.

---

## 4. Rebuild low-FPR uncertainty from per-run outcomes

**Status:** Not started

Withdraw the 87% and three-exceedance claims until that analysis is valid.

### Completed

- None yet.

### Remaining work

- To be added.

---

## 5. Redo correlation inference with dataset clustering

**Status:** Not started

Tone down causal and “worst-case” language.

### Completed

- None yet.

### Remaining work

- To be added.

---

## 6. Complete reproducibility and submission metadata

**Status:** Not started

Complete reproducibility parameters, provenance, ethics, data availability,
and other submission metadata; then perform the listed proofing fixes.

### Completed

- None yet.

### Remaining work

- To be added.
