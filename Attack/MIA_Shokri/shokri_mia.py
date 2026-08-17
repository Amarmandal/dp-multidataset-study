"""
Shokri et al. (2017) shadow-model Membership Inference Attack.

Reference: R. Shokri, M. Stronati, C. Song, V. Shmatikov,
"Membership Inference Attacks Against Machine Learning Models", IEEE S&P 2017
(arXiv:1610.05820). See ``paper/1610.05820v2 (1).pdf``.

Pipeline
--------
1. **Algorithm 1 — data synthesis using the target model** (``synthesize_pool``):
   black-box hill-climbing that queries the target's posteriors to grow records
   the target classifies confidently. Run as B parallel chains for speed; each
   chain reproduces the sequential logic of the paper's Algorithm 1.
2. **Shadow models** (``train_shadow_attack_data``): k models of the *same kind*
   as the target are trained on disjoint slices of the synthetic pool. Querying
   each shadow on its own members (``in``) and held-out non-members (``out``)
   yields labelled (posterior vector, true class, membership) examples.
3. **Attack model** (``train_attack_models``): one binary classifier per class
   maps a posterior vector to in/out.
4. **Evaluation** (``evaluate_attack``): apply the attack to the *target's* real
   members (X_train) and non-members (X_test); report membership advantage
   (TPR - FPR), balanced attack accuracy and AUC.

The module is target-agnostic: it only needs a fitted estimator exposing
``predict_proba`` plus a factory that builds fresh shadow estimators of the same
kind. See ``exported_models.make_shadow_factory``.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# Posterior alignment
# --------------------------------------------------------------------------- #
def aligned_proba(model, X, n_classes):
    """predict_proba aligned to the global class set [0 .. n_classes-1].

    A shadow trained on a slice may not see every class; this pads missing
    columns with zeros so every posterior vector has a consistent length.
    """
    proba = model.predict_proba(X)
    classes = np.asarray(getattr(model, "classes_", np.arange(proba.shape[1])))
    if proba.shape[1] == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return proba
    full = np.zeros((proba.shape[0], n_classes))
    for j, c in enumerate(classes):
        ci = int(c)
        if 0 <= ci < n_classes:
            full[:, ci] = proba[:, j]
    return full


# --------------------------------------------------------------------------- #
# Algorithm 1: data synthesis using the target model
# --------------------------------------------------------------------------- #
def _rand_records(n, low, high, rng):
    return rng.uniform(low, high, size=(n, low.shape[0]))


def _randomize_k(x, k, low, high, rng):
    """Randomise ``k[i]`` features of each row x[i] within bounds (RandRecord).

    Vectorised: a random matrix is ranked per row and the lowest-ranked ``k[i]``
    features are resampled, giving exactly ``k[i]`` randomised features per row
    without a Python loop.
    """
    B, d = x.shape
    k = np.clip(k, 1, d).astype(int)
    ranks = rng.random((B, d)).argsort(axis=1).argsort(axis=1)
    mask = ranks < k[:, None]
    new_vals = rng.uniform(low, high, size=(B, d))
    return np.where(mask, new_vals, x)


def synthesize_pool(
    target, n_classes, low, high, *,
    n_per_class=200, k_max=None, k_min=1, conf_min=0.4, rej_max=5,
    n_chains=64, max_total_iters=8000, rng=None, allow_relaxed_topup=True,
):
    """Synthesize a labelled pool by querying ``target`` (Algorithm 1).

    Returns ``(X_pool, y_pool)`` where ``y_pool`` is the class the target was
    driven toward. ``k_max`` defaults to the feature count.

    If hill-climbing cannot fill a class (common when DP flattens posteriors),
    a relaxed top-up keeps any random record whose target argmax matches the
    needed class. This guarantees usable shadow data while still being
    "synthesis using the target model".
    """
    rng = rng or np.random.default_rng(0)
    d = low.shape[0]
    k_max = k_max or d

    collected = {c: [] for c in range(n_classes)}
    need = lambda: [c for c in range(n_classes) if len(collected[c]) < n_per_class]

    B = n_chains
    c = np.array([need()[i % max(1, len(need()))] for i in range(B)]) if need() else np.zeros(B, int)
    x = _rand_records(B, low, high, rng)
    x_star = x.copy()
    yc_star = np.zeros(B)
    j = np.zeros(B, int)
    k = np.full(B, k_max, int)

    total = 0
    while need() and total < max_total_iters:
        total += 1
        probs = aligned_proba(target, x, n_classes)          # query target
        rows = np.arange(B)
        yc = probs[rows, c]
        argmax = probs.argmax(1)

        accept = yc >= yc_star
        success = accept & (yc > conf_min) & (argmax == c) & (rng.random(B) < yc)

        # store successes, then reset those chains onto classes still needed
        reset = np.zeros(B, bool)
        for i in np.where(success)[0]:
            ci = int(c[i])
            if len(collected[ci]) < n_per_class:
                collected[ci].append(x[i].copy())
            reset[i] = True

        # accepted (non-success): adopt as new best
        adopt = accept & ~success
        x_star[adopt] = x[adopt]
        yc_star[adopt] = yc[adopt]
        j[adopt] = 0

        # rejected: count consecutive rejects, shrink k after rej_max
        rej = ~accept
        j[rej] += 1
        shrink = rej & (j > rej_max)
        k[shrink] = np.maximum(k_min, np.ceil(k[shrink] / 2).astype(int))
        j[shrink] = 0

        # propose next records
        x = _randomize_k(x_star, k, low, high, rng)

        # reinitialise reset chains onto a still-needed class
        pending = need()
        if reset.any() and pending:
            ridx = np.where(reset)[0]
            new_c = np.array([pending[t % len(pending)] for t in range(len(ridx))])
            c[ridx] = new_c
            x[ridx] = _rand_records(len(ridx), low, high, rng)
            x_star[ridx] = x[ridx]
            yc_star[ridx] = 0.0
            j[ridx] = 0
            k[ridx] = k_max

    # relaxed top-up for classes hill-climbing could not fill
    if allow_relaxed_topup and need():
        for _ in range(80):
            if not need():
                break
            cand = _rand_records(2048, low, high, rng)
            pred = aligned_proba(target, cand, n_classes).argmax(1)
            for ci in need():
                want = n_per_class - len(collected[ci])
                hit = cand[pred == ci][:want]
                collected[ci].extend(list(hit))

        # last resort: if a class never wins argmax (e.g. a DP target that
        # collapses to one label), keep the records with the highest posterior
        # for that class so the pool is never single-class.
        for ci in need():
            want = n_per_class - len(collected[ci])
            cand = _rand_records(max(4096, want * 8), low, high, rng)
            score = aligned_proba(target, cand, n_classes)[:, ci]
            top = cand[np.argsort(score)[::-1][:want]]
            collected[ci].extend(list(top))

    X_pool, y_pool = [], []
    for ci in range(n_classes):
        for rec in collected[ci]:
            X_pool.append(rec)
            y_pool.append(ci)
    if not X_pool:  # absolute fallback: pure random with target labels
        X_pool = _rand_records(n_per_class * n_classes, low, high, rng)
        y_pool = aligned_proba(target, X_pool, n_classes).argmax(1)
        return np.asarray(X_pool), np.asarray(y_pool)
    return np.asarray(X_pool), np.asarray(y_pool)


# --------------------------------------------------------------------------- #
# Shadow models -> attack training data
# --------------------------------------------------------------------------- #
def train_shadow_attack_data(
    make_shadow, X_pool, y_pool, n_classes, *,
    n_shadow=5, shadow_train_size=None, rng=None,
):
    """Train ``n_shadow`` shadow models and collect labelled attack examples.

    ``make_shadow`` is a zero-arg callable returning a fresh shadow estimator of
    the same kind as the target. Returns ``(A_probs, A_class, A_member)``.
    """
    rng = rng or np.random.default_rng(0)
    n = X_pool.shape[0]
    half = n // 2
    st = min(shadow_train_size or half, half)

    probs_all, class_all, member_all = [], [], []
    for s in range(n_shadow):
        perm = rng.permutation(n)
        tr_idx, te_idx = perm[:st], perm[st:st + st]
        if len(np.unique(y_pool[tr_idx])) < 2:
            continue  # shadow needs >=2 classes to be trainable

        shadow = make_shadow()
        shadow.fit(X_pool[tr_idx], y_pool[tr_idx])

        for idx, member in ((tr_idx, 1), (te_idx, 0)):
            p = aligned_proba(shadow, X_pool[idx], n_classes)
            probs_all.append(p)
            class_all.append(y_pool[idx])
            member_all.append(np.full(len(idx), member))

    if not probs_all:
        return None
    return (np.vstack(probs_all), np.concatenate(class_all), np.concatenate(member_all))


def train_attack_models(A_probs, A_class, A_member, n_classes, rng=None):
    """One in/out classifier per class, keyed by class index."""
    rng = rng or np.random.default_rng(0)
    seed = int(rng.integers(1e6))
    models = {}
    for c in range(n_classes):
        mask = A_class == c
        Xc, yc = A_probs[mask], A_member[mask]
        if len(yc) < 4 or len(np.unique(yc)) < 2:
            continue
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=seed, n_jobs=-1
        )
        clf.fit(Xc, yc)
        models[c] = clf
    return models


# --------------------------------------------------------------------------- #
# Evaluate the attack against the target's real members / non-members
# --------------------------------------------------------------------------- #
def evaluate_attack(attack_models, target, X_train, X_test, y_train, y_test, n_classes):
    """Apply per-class attack models to target members vs non-members."""
    Xm = np.vstack([X_train, X_test])
    ym = np.concatenate([y_train, y_test])
    member = np.concatenate([np.ones(len(X_train)), np.zeros(len(X_test))])

    probs = aligned_proba(target, Xm, n_classes)
    in_score = np.full(len(Xm), np.nan)
    for c, clf in attack_models.items():
        rows = np.where(ym == c)[0]
        if len(rows):
            in_score[rows] = clf.predict_proba(probs[rows])[:, 1]

    valid = ~np.isnan(in_score)
    in_score = in_score[valid]
    member = member[valid]
    pred = (in_score >= 0.5).astype(int)

    members = member == 1
    nonmembers = member == 0
    tpr = float(pred[members].mean()) if members.any() else 0.0
    fpr = float(pred[nonmembers].mean()) if nonmembers.any() else 0.0
    advantage = tpr - fpr
    balanced_acc = 0.5 * (tpr + (1.0 - fpr))

    try:
        auc = float(roc_auc_score(member, in_score)) if len(np.unique(member)) > 1 else 0.5
    except ValueError:
        auc = 0.5

    precision = (
        float(pred[members].sum()) / float(pred.sum()) if pred.sum() > 0 else 0.0
    )

    return {
        "tpr": tpr,
        "fpr": fpr,
        "advantage": advantage,
        "attack_accuracy": float(balanced_acc),
        "attack_precision": precision,
        "attack_auc": auc,
        "n_members": int(members.sum()),
        "n_nonmembers": int(nonmembers.sum()),
        "n_evaluated": int(valid.sum()),
    }


# --------------------------------------------------------------------------- #
# End-to-end driver for a single target
# --------------------------------------------------------------------------- #
def run_shokri_attack(
    target, make_shadow, data, n_classes, low, high, *,
    n_per_class=200, n_shadow=5, conf_min=0.4, shadow_train_size=None, seed=0,
):
    """Full attack against an already-fitted ``target``. Returns metrics dict.

    ``data`` is a dict with X_train/X_test/y_train/y_test (the target's real
    split). ``make_shadow`` builds fresh shadow estimators of the target's kind.
    """
    rng = np.random.default_rng(seed)

    X_pool, y_pool = synthesize_pool(
        target, n_classes, low, high,
        n_per_class=n_per_class, conf_min=conf_min, rng=rng,
    )

    attack_data = train_shadow_attack_data(
        make_shadow, X_pool, y_pool, n_classes,
        n_shadow=n_shadow, shadow_train_size=shadow_train_size, rng=rng,
    )
    if attack_data is None:
        return {"error": "shadow training produced no attack data"}

    attack_models = train_attack_models(*attack_data, n_classes, rng=rng)
    if not attack_models:
        return {"error": "no attack models could be trained"}

    metrics = evaluate_attack(
        attack_models, target,
        data["X_train"], data["X_test"], data["y_train"], data["y_test"],
        n_classes,
    )
    metrics["synth_pool_size"] = int(len(y_pool))
    metrics["n_shadow_used"] = int(len(np.unique(attack_data[2])) and n_shadow)
    metrics["n_attack_models"] = int(len(attack_models))
    return metrics
