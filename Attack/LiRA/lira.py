"""
LiRA — Likelihood Ratio Attack (Carlini et al. 2022, IEEE S&P,
"Membership Inference Attacks From First Principles", arXiv:2112.03404).

Why this is *different* from the Shokri shadow attack and the Yeom loss threshold
already in the study:

  * Shokri / Yeom are **average-case**: a single global rule (an attack model over
    posteriors, or one loss threshold) applied to every record. They answer
    "does this model leak on average?" — and report AUC / advantage near 0.5.
  * LiRA is **per-example**: for each record z it calibrates against *that record's*
    own difficulty. It trains many shadow models, splits them into those trained
    WITH z (IN) and WITHOUT z (OUT), fits a Gaussian to the logit-scaled
    confidence under each, and applies a likelihood-ratio test to the target's
    confidence on z. This exposes the handful of records that leak strongly even
    when the average AUC is ~0.5 — the number that actually matters for a medical
    privacy claim.

Key differences from the Shokri module in this repo:
  * Shadows are trained on **real data splits** (random halves of X_train ∪ X_test),
    not on synthetic records hill-climbed from the target. This removes the
    synthetic-pool distribution shift that handicaps the Shokri variant.
  * The headline metric is **TPR at a low fixed FPR** (0.1 % / 1 %), reported on a
    log-log ROC, exactly as Carlini et al. argue MIAs should be evaluated.

The module is target-agnostic: it needs a fitted target exposing predict_proba
and a zero-arg factory that builds fresh shadow estimators of the same kind
(both provided by ``exported_models.py``).
"""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve


# --------------------------------------------------------------------------- #
# Confidence transform
# --------------------------------------------------------------------------- #
def logit_conf(proba, y, eps=1e-6):
    """Logit-scaled confidence of the *true* class: phi = log(p / (1 - p)).

    Carlini et al. show the model's confidence on the true label is roughly
    Gaussian *after* this logit transform, which is what makes the per-example
    Gaussian likelihood-ratio test well-calibrated.
    """
    proba = np.asarray(proba, float)
    p = proba[np.arange(len(y)), np.asarray(y, int)]
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


# --------------------------------------------------------------------------- #
# Metrics — TPR at low FPR is the headline
# --------------------------------------------------------------------------- #
def attack_metrics(member, score):
    """ROC-based metrics for a membership score (higher = more member-like)."""
    member = np.asarray(member, int)
    score = np.asarray(score, float)
    ok = np.isfinite(score)
    member, score = member[ok], score[ok]

    if len(np.unique(member)) < 2:
        return {"error": "only one membership class present"}

    auc = float(roc_auc_score(member, score))
    fpr, tpr, _ = roc_curve(member, score)

    def tpr_at(target_fpr):
        return float(np.interp(target_fpr, fpr, tpr))

    return {
        "attack_auc": auc,
        "advantage": float(np.max(tpr - fpr)),       # KS statistic (max TPR-FPR)
        "tpr_at_10pct": tpr_at(0.10),
        "tpr_at_1pct": tpr_at(0.01),
        "tpr_at_0p1pct": tpr_at(0.001),
        "n_members": int((member == 1).sum()),
        "n_nonmembers": int((member == 0).sum()),
        "_roc_fpr": fpr.tolist(),                    # kept for log-log ROC plot
        "_roc_tpr": tpr.tolist(),
    }


# --------------------------------------------------------------------------- #
# Evaluation-set balancing
# --------------------------------------------------------------------------- #
def balance_members(member, score, seed=0):
    """Subsample members down to the non-member count -> a 1:1 evaluation set.

    Membership ground truth here is "record ∈ X_train", so an 80/20 target split
    gives 4 members per non-member. The standard MIA threat model instead assumes
    an adversary with a 50/50 prior (Yeom et al. 2018; Carlini et al. 2022 evaluate
    on balanced IN/OUT sets), and a 4:1 pool makes the reported class counts —
    and anything read off them, such as the granularity of the FPR axis — differ
    from dataset to dataset for reasons unrelated to leakage.

    LiRA scores are computed per record and independently, so dropping members
    changes only *which* records are scored, never the score of a retained one.
    Returns (member, score) restricted to the balanced subset.
    """
    member = np.asarray(member, int)
    score = np.asarray(score, float)

    idx_in = np.where(member == 1)[0]
    idx_out = np.where(member == 0)[0]
    if len(idx_in) <= len(idx_out):
        return member, score

    keep = np.random.default_rng(seed).choice(idx_in, size=len(idx_out), replace=False)
    sel = np.sort(np.concatenate([keep, idx_out]))
    return member[sel], score[sel]


# --------------------------------------------------------------------------- #
# LiRA
# --------------------------------------------------------------------------- #
def run_lira(target, make_shadow, view, n_classes, *,
             n_shadow=32, online=True, seed=0, balance=False):
    """Run LiRA against ``target`` using shadows from ``make_shadow``.

    ``view`` holds the target's real split (X_train/X_test/y_train/y_test) in the
    family's feature space. Membership ground truth is "was z in the target's
    training set" (i.e. z ∈ X_train). Shadows are trained on random halves of the
    pooled records; for record z the shadows that included it form its IN
    distribution and the rest its OUT distribution.

    online=True  -> two-sided LR test  log N(phi_t; mu_in,sd_in) - log N(phi_t; mu_out,sd_out)
    online=False -> offline one-sided test  log P(phi >= phi_t | OUT)  (no IN models needed)

    balance=True subsamples members down to the non-member count before scoring,
    so metrics are reported on a 1:1 member/non-member set (see ``balance_members``).

    Returns a metrics dict (see ``attack_metrics``) plus diagnostics.
    """
    rng = np.random.default_rng(seed)

    X = np.vstack([view["X_train"], view["X_test"]]).astype(float)
    y = np.concatenate([view["y_train"], view["y_test"]]).astype(int)
    member = np.concatenate([
        np.ones(len(view["y_train"])), np.zeros(len(view["y_test"]))
    ]).astype(int)
    n = len(y)

    # ----- train shadows on real random halves; record confidences & membership
    phi = np.full((n_shadow, n), np.nan)        # phi[i, j] = shadow i's logit conf on j
    incl = np.zeros((n_shadow, n), dtype=bool)  # incl[i, j] = j was in shadow i's train set
    n_ok = 0
    for i in range(n_shadow):
        mask = rng.random(n) < 0.5
        if len(np.unique(y[mask])) < 2:         # degenerate split — guarantee 2 classes
            for c in np.unique(y):
                mask[rng.choice(np.where(y == c)[0])] = True
        shadow = make_shadow()
        try:
            shadow.fit(X[mask], y[mask])
            phi[i] = logit_conf(shadow.predict_proba(X), y)
            incl[i] = mask
            n_ok += 1
        except Exception:
            continue  # a single failed shadow shouldn't sink the run

    if n_ok < 2:
        return {"error": f"only {n_ok} shadow(s) trained"}

    # ----- per-record IN/OUT means; global (pooled) variances for stability
    # Carlini's "fixed variance" LiRA: per-example means but a shared sigma,
    # which is far more robust at modest shadow counts than per-example sigma.
    mu_in = np.full(n, np.nan)
    mu_out = np.full(n, np.nan)
    in_resid, out_resid = [], []
    for j in range(n):
        col = phi[:, j]
        pin = col[incl[:, j] & np.isfinite(col)]
        pout = col[~incl[:, j] & np.isfinite(col)]
        if len(pin):
            mu_in[j] = pin.mean()
            in_resid.append(pin - mu_in[j])
        if len(pout):
            mu_out[j] = pout.mean()
            out_resid.append(pout - mu_out[j])

    sd_in = max(float(np.std(np.concatenate(in_resid))) if in_resid else 1.0, 1e-3)
    sd_out = max(float(np.std(np.concatenate(out_resid))) if out_resid else 1.0, 1e-3)

    # ----- target confidence on every record
    phi_t = logit_conf(target.predict_proba(X), y)

    # ----- likelihood-ratio score (higher => more member-like)
    with np.errstate(invalid="ignore"):
        if online:
            score = norm.logpdf(phi_t, mu_in, sd_in) - norm.logpdf(phi_t, mu_out, sd_out)
            # records with no IN (or no OUT) shadow fall back to the offline test
            bad = ~np.isfinite(score)
            score[bad] = norm.logsf(phi_t[bad], mu_out[bad], sd_out)
        else:
            score = norm.logsf(phi_t, mu_out, sd_out)

    # any remaining non-finite (e.g. record in *every* shadow) -> neutral score
    score[~np.isfinite(score)] = np.nanmedian(score[np.isfinite(score)])

    if balance:
        member, score = balance_members(member, score, seed=seed)

    metrics = attack_metrics(member, score)
    metrics.update({
        "n_shadow_trained": int(n_ok),
        "online": bool(online),
        "balanced": bool(balance),
        "sd_in": sd_in,
        "sd_out": sd_out,
    })
    return metrics
