"""
MIA — Yeom's loss-threshold Membership Inference Attack (Yeom, Giacomelli,
Fredrikson, Jha 2018, *Privacy Risk in Machine Learning*, IEEE CSF) run against
the **exported** Standard vs DP targets.

This is the average-case companion to the per-example LiRA in ``Attack/LiRA``:
LiRA asks "which *individual* records leak?"; Yeom asks "does this model leak on
average?" via a single global loss threshold. The headline metric is the
**membership advantage**, TPR − FPR at that threshold.

The attack is **target-agnostic** — exactly like ``lira.py`` it needs only a
fitted target exposing ``predict_proba`` (provided by ``exported_models.py``)
and a data ``view`` holding the target's real train/test split. The per-sample
signal is the true-class cross-entropy loss φ = −log p_true, computed
identically for every family through its posterior head:

  * LR / RF / GNB  -> binary cross-entropy on ``predict_proba`` (unchanged from
    the original per-model scripts);
  * SVM / DNN      -> cross-entropy on the softmax posterior head that
    ``exported_models`` wraps around the decision scores / logits.

Yeom's rule: a record is predicted IN (member) if its loss is ≤ τ, the mean
training loss. The attack logic itself is unchanged from the original
``*_mia.py`` scripts — only the *plumbing* is now shared so one driver can run
every family on every dataset.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# Per-sample loss signal
# --------------------------------------------------------------------------- #
def per_sample_ce(proba, y, eps=1e-15):
    """True-class cross-entropy loss per record: −log p_true.

    For binary targets this is identical to the binary-cross-entropy signal the
    original LR/RF/GNB scripts used; for SVM/DNN it is the cross-entropy on the
    softmax posterior head exposed by ``exported_models``.
    """
    proba = np.asarray(proba, float)
    p = proba[np.arange(len(y)), np.asarray(y, int)]
    return -np.log(np.clip(p, eps, 1.0))


# --------------------------------------------------------------------------- #
# Yeom's loss-threshold attack
# --------------------------------------------------------------------------- #
def yeom_mia(target, view):
    """Run Yeom's loss-threshold MIA against ``target`` on its real split.

    ``view`` holds X_train/X_test/y_train/y_test in the family's feature space.
    Membership ground truth is "record ∈ the target's training set". τ is the
    mean training loss; members are records with loss ≤ τ; the headline metric
    is advantage = TPR − FPR (Yeom et al. 2018). Attack AUC (ranking quality,
    0.5 = random) is reported alongside for continuity with the LiRA results.
    """
    Xtr, Xte = view["X_train"], view["X_test"]
    ytr, yte = view["y_train"], view["y_test"]

    train_loss = per_sample_ce(target.predict_proba(Xtr), ytr)
    test_loss = per_sample_ce(target.predict_proba(Xte), yte)

    tau = float(np.mean(train_loss))
    tpr = float(np.mean(train_loss <= tau))
    fpr = float(np.mean(test_loss <= tau))
    advantage = float(tpr - fpr)

    # Average-case ranking quality: lower loss => more member-like.
    member = np.concatenate([np.ones(len(ytr)), np.zeros(len(yte))]).astype(int)
    score = -np.concatenate([train_loss, test_loss])
    ok = np.isfinite(score)
    try:
        auc = float(roc_auc_score(member[ok], score[ok]))
    except ValueError:
        auc = float("nan")

    return {
        "threshold_tau": tau,
        "tpr": tpr,
        "fpr": fpr,
        "advantage": advantage,
        "attack_auc": auc,
        "train_loss_mean": float(np.mean(train_loss)),
        "train_loss_std": float(np.std(train_loss)),
        "test_loss_mean": float(np.mean(test_loss)),
        "test_loss_std": float(np.std(test_loss)),
        "loss_gap": float(np.mean(test_loss) - np.mean(train_loss)),
        "n_members": int(len(ytr)),
        "n_nonmembers": int(len(yte)),
    }
