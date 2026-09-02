# Reproducibility Record

This file preserves implementation and environment details moved out of the
main manuscript for readability. It does not change the experimental protocol
or reported results.

## Data splits and seeds

- Each dataset uses one stratified 80/20 train--test split with
  `random_state = 42`.
- The split is fixed across models, privacy budgets, and attacks.
- Training is repeated 30 times with model seed `10 * run + 42`.
- Features are transformed with a training-fitted `MinMaxScaler` to `[-1, 1]`.
- LR, SVM, and their non-private baselines additionally use the row-wise
  unit-L2-ball projection described in the manuscript.

## Repetitions and privacy-budget grids

- Classical DP configurations use 30 training runs.
- Yeom is deterministic given the target and is run once.
- Shokri and LiRA use 30 attack repetitions for classical targets and 5 for
  DNN targets; Diabetes SVM LiRA also uses 5.
- The RQ analyses use the shared nine-point epsilon grid
  `{0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0, 10.0}`.
- The standalone classical-model utility landscape additionally includes
  `{0.6, 1.2, 1.4, 1.6, 1.8, 6.0}`.

## Software environment

- Python 3.13
- diffprivlib 0.6.6
- Opacus 1.6.0
- PyTorch 2.11.0
- scikit-learn 1.6.1
- NumPy 2.4.6
- SciPy 1.17.1
- pandas 3.0.3

The original environment used exact package pins in the committed `uv.lock`
file and was restored with `uv sync`.

## DP-SVM conformance checks

The focused Algorithm 2 test suite checks the Huber loss and analytic gradient,
`c = 1/(2h)`, both epsilon-prime branches and the Delta fallback, the
noise-radius distribution, row-wise unit-ball enforcement, independence from
other training records, consistent inference projection, and the no-intercept
configuration. All 13 focused tests passed. The six executed DP-SVM notebooks
exported 90 private models with `fit_intercept=False`, and their executed
outputs contained no errors.

## Hardware and runtime record

Experiments ran on one Apple M5 system with 24 GB unified memory under macOS
Tahoe 26.5.2. Mean per-fit training time for classical DP models remained below
0.11 s on every dataset. DP-DNN ranged from 0.30 s for Gallstone to 22.94 s for
Diabetes per fit. The largest non-private fit was the Diabetes SVM baseline at
140.3 s. Per-configuration means are recorded in all 444 aggregated
dataset--model--variant--budget rows of the consolidated utility table.

## Attack-time settings and code

DP-DNN was placed in evaluation mode with gradient tracking disabled during
posterior queries. The attack implementations (`mia.py`, `shokri_mia.py`, and
`lira.py`) are documented in the
[project repository](https://github.com/Amarmandal/dp-multidataset-study).
