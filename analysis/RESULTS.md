# Cluster-aware correlation results

Cross-sectional Spearman coefficients are descriptive statistics over the 30
dataset-model rows. Their p-values and confidence intervals treat the six
datasets—not the 30 rows—as the independent sampling units. Complete 6×5 grids
use all 720 exact dataset-block permutations and 20,000 dataset-cluster
bootstrap resamples. The six-point RF analysis uses an exact 720-permutation
test. These results describe associations and do not establish causal
mechanisms.

## Artifact-matched utility and leakage

At `epsilon = 1`, artifact-matched `ACL_exported` had the following
cross-sectional associations:

| Leakage metric | Spearman rho | Exact block p | Dataset-cluster bootstrap 95% CI |
|---|---:|---:|---:|
| Yeom advantage | -0.1567 | 0.3694 | [-0.2771, 0.0536] |
| Shokri AUC | -0.2347 | 0.3958 | [-0.7615, 0.4092] |
| LiRA AUC | 0.2957 | 0.1514 | [-0.1187, 0.7159] |
| Low-FPR LiRA TPR | 0.3108 | 0.2569 | [-0.1584, 0.7526] |

The LiRA TPR coefficient is numerically unchanged from the earlier analysis;
only its inference changed because the five models within each dataset are no
longer treated as independent replicates.

Across the nine privacy budgets, low-FPR LiRA TPR was significant at two
uncorrected operating points: `epsilon = 0.2` (`rho = 0.5301`, `p = 0.0417`)
and `epsilon = 0.8` (`rho = 0.6263`, `p = 0.0222`). The earlier four-marker
claim is therefore superseded. Because these markers are uncorrected across
budgets and metrics, they should be described as exploratory.

## Other cross-sectional associations

- Yeom advantage and LiRA AUC on non-private targets were positively associated
  (`rho = 0.6018`), but the exact dataset-block p-value was `0.0611`; this does
  not meet the uncorrected 0.05 threshold.
- Baseline low-FPR LiRA TPR and its reduction under DP were strongly positively
  associated (`rho = 0.8897`, exact block `p = 0.0014`, bootstrap 95% CI
  `[0.7485, 0.9647]`). This is partly arithmetically coupled because reduction
  contains baseline leakage; it must not be presented as a causal mechanism.
- Baseline and residual low-FPR LiRA TPR had `rho = 0.3677`, exact block
  `p = 0.1361`, and bootstrap 95% CI `[-0.1143, 0.7817]`.
- For the six non-private RF points, dataset size and low-FPR LiRA TPR had
  `rho = -0.8857`, exact permutation `p = 0.0333`, and observation-bootstrap
  95% CI `[-1.0000, -0.2000]`.

With only six independent datasets, all cross-sectional inference remains
fragile. The exact permutation p-values are discrete, and the bootstrap
intervals describe resampling across these six observed datasets. Neither
method incorporates alternative train/test splits, target retraining, or new
clinical datasets.

## Manuscript wording

Use "associated with," "positively related to," or "consistent with" instead
of "driven primarily by," "caused by," or "a property of overfitting." Use
"low-FPR LiRA TPR" or "tail-oriented aggregate leakage" instead of
"worst-case leakage." TPR at a fixed low FPR aggregates records at one
operating point; it is not the maximum individual privacy loss.

Authoritative machine-readable results are in:

- `stats/correlation_stats/rho_by_epsilon.csv`
- `stats/correlation_stats/all_correlations.csv`
- `stats/rq/correlations.csv`
- `tables/csv/table_11_artifact_matched_acl_exported_epsilon_1.csv`
- `tables/csv/table_12_artifact_matched_acl_exported_all_budgets.csv`
- `tables/csv/table_13_artifact_matched_acl_exported_within_pair.csv`
