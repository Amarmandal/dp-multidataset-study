# MIA_YEOM — Yeom's Loss-Threshold Membership Inference Attack

Average-case membership inference against the **exported** Standard and DP
targets, for every model family on every study dataset.

Reference: Yeom, Giacomelli, Fredrikson & Jha (2018), *Privacy Risk in Machine
Learning: Analyzing the Connection to Overfitting*, IEEE CSF.

---

## 1. Attack method

A model that memorises its training set assigns **lower loss to members** than
to non-members. Yeom's attack turns that gap into a decision rule with a single
global threshold:

| Element | Definition |
|---------|------------|
| Per-sample signal | true-class cross-entropy, φ = −log p<sub>true</sub> |
| Threshold τ | mean training loss |
| Rule | predict IN (member) if loss ≤ τ |
| Headline metric | **membership advantage = TPR − FPR** |

**Advantage is not clamped.** 0.0 means no leakage; a negative value is
meaningful signal (the model fits the test split better than the train split),
so squashing it to [0, 1] would hide a real result. Attack AUC (0.5 = random)
is reported alongside for continuity with the LiRA results.

The mechanism Yeom exploits is the **generalisation gap** — mean test loss minus
mean train loss. That is why `loss_gap` is recorded per target and plotted: DP
shrinks the gap, which is precisely why it suppresses the advantage.

### Relationship to the other attacks

| Directory | Question it answers | Metric |
|-----------|--------------------|--------|
| `Attack/MIA_YEOM` | Does this model leak **on average**? | advantage (TPR − FPR) |
| `Attack/LiRA` | Which **individual** records leak? | TPR @ low fixed FPR |
| `Attack/MIA_Shokri` | Can a shadow-trained classifier infer membership? | attack accuracy / AUC |

Yeom is the average-case view; low-FPR LiRA TPR is the tail-oriented aggregate
companion on the same targets.

---

## 2. Layout

```
MIA_YEOM/
├── mia.py                      # the attack itself (target-agnostic, ~90 lines)
├── run_mia.py                  # driver: all families × all datasets
└── results/                    # authoritative per-dataset CSV files
```

There are **no per-family subdirectories, and none are needed.** One driver
covers LR / RF / GNB / SVM / DNN — see §4.

---

## 3. Usage

```bash
cd Attack/MIA_YEOM

python3 run_mia.py                          # all datasets, all families
python3 run_mia.py --datasets KIDNEY_STONE  # one dataset
python3 run_mia.py --models GNB RF          # subset of families
python3 run_mia.py --quick                  # smoke test (ε ∈ {0.1, 1.0})
```

Defaults: datasets `GALLSTONE, KIDNEY_STONE, LUNG_CANCER, BCP, CANCER_RISK,
DIABETES`; families `LR, RF, GNB, SVM, DNN`; budgets ε ∈ {0.1, 0.4, 0.8, 1.0,
2.0, 10.0}.

### Outputs

| Path | Contents |
|------|----------|
| `results/<dataset>/<dataset>_mia_results.csv` | tidy per-target table |

A single failed target is caught and recorded as an `error` row rather than
sinking the whole run, so a partial export still produces a usable CSV.

---

## 4. Why one driver covers every family

`run_mia.py` attacks the **exported** targets — it does not re-train anything.
Loading is table-driven through `Attack/MIA_Shokri/exported_models.py`:

```python
FAMILIES = {
    "LR":  ("LR",           "lr",  ".pkl"),
    "RF":  ("RandomForest", "rf",  ".pkl"),
    "GNB": ("GaussianNB",   "gnb", ".pkl"),
    "SVM": ("SVM",          "svm", ".pkl"),
    "DNN": ("DNN",          "dnn", ".pt"),
}
```

which resolves the standard export layout

```
<DATASET>/<FamilyDir>/output/model/std_<stem>_model<ext>
<DATASET>/<FamilyDir>/output/model/dp_<stem>_model_eps_<ε><ext>
```

`yeom_mia(target, view)` then needs only `predict_proba`. The three places
families genuinely differ are already centralised in `exported_models.py`:

- **Posterior head** — LR/RF/GNB expose `predict_proba` natively; `_SVMProba`
  softmaxes the DP-SVM's `decision_function`; `_TorchProba` softmaxes DNN logits.
- **Feature space** — `family_dataview()` reproduces the unit-L2-row clip the LR
  and SVM notebooks apply; other families use the MinMax-scaled features.
- **Unpickling** — the DP-SVM classes come from the repo-root `common_svm`
  module (the same objects the notebooks pickled); the DNN module classes are
  re-registered so `torch.load` resolves them.

Adding a sixth family means adding one row to `FAMILIES` (plus a posterior
wrapper if it lacks `predict_proba`) — not a new directory.

---

## 5. History

This directory previously held ten standalone scripts in per-family
subdirectories (`LR/std_lr_mia.py`, `GaussiaNB/dp_gnb_mia.py`, …). They were
removed in favour of the shared driver. **The attack logic is unchanged** — only
the plumbing is now shared. They were superseded rather than merely duplicated:

1. **They re-trained their targets in-process**, while `run_mia.py` attacks the
   exported artefacts that LiRA and Shokri also attack. Different targets, so
   old and new numbers are not comparable.
2. **`GaussiaNB/dp_gnb_mia.py` had drifted on preprocessing**, drifting from the
   pipeline's `MinMaxScaler(feature_range=(-1,1))` with bounds `[-1,1]`, so its
   DP noise was not calibrated against the scaler actually in use.
3. **They clamped advantage to [0, 1]**, discarding the negative-advantage
   signal that `mia.py` preserves.

Recover them from git history if ever needed:

```bash
git log --diff-filter=D -- 'Attack/MIA_YEOM/LR/*'
```

---

## 6. Reading the numbers

| Metric | Meaning | No leak | Leak |
|--------|---------|---------|------|
| Advantage | TPR − FPR at τ (headline) | 0.0 | > 0.1 |
| Attack AUC | average-case ranking quality | 0.5 | > 0.6 |
| Loss gap | mean(test loss) − mean(train loss) | 0.0 | ≫ 0.0 |

Expected behaviour: Standard models show the highest advantage (no privacy
protection); DP at low ε drives it toward 0.0; DP at high ε converges back
toward the Standard baseline. Dataset size matters — small-N datasets overfit
harder and therefore leak more, so advantage should fall as N grows.

> Re-run this attack whenever targets are re-exported: the results are only
> valid for the `.pkl` / `.pt` files present at run time.
