# Results

Two independent pipelines. Both read CSV; both write into their own `figures/`.

```
dataset_results/   utility — ACL / accuracy / F1 vs ε, across datasets and models
attack_results/    privacy — membership inference leakage (LiRA, Shokri, Yeom)
```

## Utility figures

```bash
cd dataset_results
python prepare_data.py                                   # -> consolidated_data.csv
jupyter nbconvert --to notebook --execute --inplace graph_construction.ipynb
```

- **Reads:** `consolidated_data.csv` — 444 rows, one per (dataset, model, variant, epsilon).
  Baselines are rows with `variant == 'standard'`, not a separate file.
- **Writes:** `figures/fig1…fig4b` (`.png` + `.pdf`)
- Source of truth is the CSV. `consolidated_data.json` is a stale leftover — do not use.

## Attack figures

```bash
cd attack_results
python prepare_mia_data.py                                     # -> consolidated_mia_data.csv
jupyter nbconvert --to notebook --execute --inplace mia_graph_construction.ipynb
```

- **Reads:** `consolidated_mia_data.csv` — 900 rows = 3 attacks × 300.
- **Writes:** `figures/<attack>_fig1…fig5b`
- The notebook plots **one attack at a time**, set by `ATTACK` in the paths cell
  (`'lira'` by default). Change it and re-run to regenerate another set; filenames
  are prefixed by attack, so the three sets coexist.

## Regenerate after results change

Both `prepare_*.py` scripts only aggregate — they do not retrain. Re-run the one
whose upstream changed, then its notebook:

| Changed | Re-run |
|---|---|
| any `<DATASET>/<FAM>/output/*.csv` | `dataset_results/prepare_data.py` → notebook |
| any `Attack/*/results/*.csv` | `attack_results/prepare_mia_data.py` → notebook |

Run scripts from their own directory; each resolves the repo root relative to its
own location and writes its CSV alongside itself.
