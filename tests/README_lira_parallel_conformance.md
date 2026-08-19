# LiRA multiprocessing conformance

## Why this test exists

LiRA was too slow for the DIABETES SVM targets when executed sequentially.
The standard target is especially expensive: five LiRA runs train and score
`5 × 32 = 160` RBF-SVM shadow models over 70,692 records.

`Attack/LiRA/run_lira.py` therefore supports multiprocessing for DIABETES SVM:

- DP targets run independently by epsilon.
- Standard-target runs execute independently by seed.
- Results are aggregated and restored to canonical CSV order afterward.

Parallel execution must change only scheduling. It must not change any metric,
seed, shadow model, ROC curve, or output ordering.

## What the test proves

`test_lira_parallel_conformance.py` uses BCP because it exercises the same code
path quickly. It evaluates the real exported BCP SVM targets twice:

1. Sequentially.
2. Through the multiprocessing path.

Both executions use:

- the standard target and all nine DP epsilon targets;
- 10 runs per target, with seeds `0..9`;
- 32 shadows per run;
- the same data, exported targets, and LiRA implementation.

The test requires exact equality for every aggregate metric, standard
deviation, metadata field, ROC array, and row position. A single difference
fails the test.

The test is read-only: it does not replace models, attack CSVs, figures, or BCP
results.

## Run it

From the repository root:

```bash
MPLCONFIGDIR=/tmp/lira-test .venv/bin/pytest -q \
  tests/test_lira_parallel_conformance.py
```

Expected result:

```text
1 passed
```

## Scope

Passing proves numerical equivalence between the sequential and parallel LiRA
execution paths. DIABETES output still requires normal artifact checks after a
full run: 10 ordered SVM rows, five runs per row, the complete epsilon grid,
and no errors or missing metrics.
