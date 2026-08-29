# LiRA rerun

Protocol:

- LR, RF, GNB and non-Diabetes SVM: 30 runs.
- DNN and Diabetes SVM: 5 runs.
- Each run uses 32 shadows.
- Diabetes SVM uses 3 workers.

```bash
uv run python -u Attack/LiRA/run_lira.py \
  --datasets GALLSTONE KIDNEY_STONE LUNG_CANCER BCP CANCER_RISK \
  --runs 30 --svm-runs 30 --dnn-runs 5

uv run python -u Attack/LiRA/run_lira.py \
  --datasets DIABETES \
  --runs 30 --svm-runs 5 --dnn-runs 5 --workers 3
```

Always run all five model families together. A `--models` subset replaces that
dataset's CSVs with partial results.

Outputs per dataset: `_lira_results.csv`, `_lira_runs.csv`,
`_lira_roc.csv.gz`, and `_lira_runtimes.csv`. The runtime CSV records wall time
for every model family; use it to update `Attack/SVM_ATTACK_RUNTIMES.md` after a
complete rerun.
