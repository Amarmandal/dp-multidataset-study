"""Shared evaluation helpers for the DP notebooks.

Every model family (RF, LR, GNB, SVM, DNN) reports the same metric set so the
per-epsilon reports stay comparable across datasets:

    accuracy, train_accuracy, balanced_accuracy, roc_auc (AUROC),
    precision, recall, F1.

Accuracy, balanced accuracy, precision, recall and F1 come straight from
scikit-learn.  AUROC is the awkward one: it needs a *continuous* score rather
than a hard label, and the estimators used here expose that score in three
different ways --

    * ``predict_proba``      -- sklearn / diffprivlib RF, LR, GNB, SVC(probability=True)
    * ``decision_function``  -- LinearSVC and the DP-SVM of ``common_svm``
    * neither                -- the PyTorch DNNs, which produce their scores
                                inside ``evaluate_model`` and pass them here
                                directly

``positive_class_scores`` normalises the first two cases; ``auroc`` then wraps
``sklearn.metrics.roc_auc_score`` so the binary datasets and the 3-class
LUNG_CANCER dataset can share one call site.
"""

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

__all__ = ["positive_class_scores", "auroc", "model_auroc", "ExtraMetrics"]


def positive_class_scores(model, X):
    """Continuous scores suitable for :func:`auroc`.

    Returns a 1-D array of positive-class scores for a binary estimator and an
    ``(n_samples, n_classes)`` matrix for a multiclass one.  ``predict_proba``
    is preferred; estimators that only expose ``decision_function`` (LinearSVC,
    ``common_svm.DifferentiallyPrivateSVM``) fall back to the margin, which is
    a monotone transform of the probability and so leaves AUROC unchanged.
    """
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(X))
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
    else:
        raise TypeError(
            f"{type(model).__name__} exposes neither predict_proba nor "
            "decision_function, so AUROC cannot be computed from it; give the "
            "estimator a decision_function, or pass y_score to ExtraMetrics.add()."
        )

    if scores.ndim == 2 and scores.shape[1] == 2:
        # Collapse a two-column binary score to the positive class.  For
        # probabilities that is column 1; for a one-vs-rest margin pair the
        # difference preserves the ranking.
        return scores[:, 1] if hasattr(model, "predict_proba") else scores[:, 1] - scores[:, 0]
    return scores


def auroc(y_true, scores):
    """``roc_auc_score`` for both the binary and the multiclass notebooks.

    ``scores`` is either a 1-D positive-class score (binary) or an
    ``(n_samples, n_classes)`` matrix (multiclass, scored one-vs-rest with a
    macro average).  Multiclass ``roc_auc_score`` insists the rows sum to one,
    so raw decision-function margins are softmaxed first.

    Returns ``nan`` when ``y_true`` holds a single class, where AUROC is
    undefined -- that keeps a degenerate DP run from aborting a whole sweep.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    if scores.ndim == 2 and scores.shape[1] == 2:
        # A two-column score for a binary problem: roc_auc_score wants the
        # positive column, not the matrix.
        scores = scores[:, 1]

    if scores.ndim == 1:
        return float(roc_auc_score(y_true, scores))

    if not np.allclose(scores.sum(axis=1), 1.0):
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        scores = exp / exp.sum(axis=1, keepdims=True)
    return float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro"))


def model_auroc(y_true, model, X):
    """Convenience wrapper: score ``X`` with ``model`` and compute AUROC."""
    return auroc(y_true, positive_class_scores(model, X))


class ExtraMetrics:
    """Per-run accumulator for the metrics added on top of the original set.

    The notebooks already track accuracy, precision, recall and F1 in bespoke
    lists.  This collects only what was missing -- training accuracy, balanced
    accuracy and AUROC -- so it can be merged into the existing ``results``
    dicts and JSON reports without colliding with them.

    Typical use inside an epsilon sweep::

        extra = ExtraMetrics()
        for run in range(N_RUNS):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            extra.add(y_test, y_pred, model=model, X=X_test)

        results_json[str(eps)].update(extra.summary())
        results_json[str(eps)].update(extra.per_run())

    ``train_accuracy`` is only populated when a training partition is passed,
    which the notebooks do for the non-private fits.
    """

    METRICS = ("train_accuracy", "balanced_accuracy", "roc_auc", "train_time")

    def __init__(self):
        self.train_accuracy = []
        self.balanced_accuracy = []
        self.roc_auc = []
        self.train_time = []

    def add(self, y_true, y_pred, model=None, X=None, y_score=None,
            train_true=None, train_pred=None, train_time=None):
        """Record one run.

        Supply either ``model`` + ``X`` (the estimator is scored here) or a
        precomputed ``y_score`` (the DNNs, whose scores come out of their own
        evaluation loop).  Pass ``train_true`` / ``train_pred`` to also log the
        training-partition accuracy, and ``train_time`` (seconds, measured with
        ``time.perf_counter`` around the fit) to log the training runtime.
        """
        self.balanced_accuracy.append(float(balanced_accuracy_score(y_true, y_pred)))

        if y_score is None and model is not None and X is not None:
            y_score = positive_class_scores(model, X)
        self.roc_auc.append(auroc(y_true, y_score) if y_score is not None else float("nan"))

        if train_true is not None and train_pred is not None:
            self.train_accuracy.append(float(accuracy_score(train_true, train_pred)))

        if train_time is not None:
            self.train_time.append(float(train_time))
        return self

    def _active(self, exclude):
        exclude = set(exclude or ())
        return [n for n in self.METRICS if n not in exclude and getattr(self, n)]

    def means(self, exclude=()):
        """Plain-named means, for the single-row baseline/summary dicts."""
        return {n: float(np.nanmean(getattr(self, n))) for n in self._active(exclude)}

    def summary(self, exclude=()):
        """``*_mean`` / ``*_std`` pairs, matching the per-epsilon report keys.

        ``exclude`` drops metrics a notebook already tracks itself, so merging
        this into an existing results dict cannot double-count a column.
        """
        out = {}
        for name in self._active(exclude):
            vals = getattr(self, name)
            out[f"{name}_mean"] = float(np.nanmean(vals))
            out[f"{name}_std"] = float(np.nanstd(vals))
        return out

    def per_run(self, exclude=()):
        """Raw per-run values, matching the ``per_run_*`` report keys."""
        return {f"per_run_{n}": [float(v) for v in getattr(self, n)]
                for n in self._active(exclude)}

    def describe(self, indent="  "):
        """One line per metric, for the notebooks' summary printouts."""
        labels = {"train_accuracy": "Train Acc", "balanced_accuracy": "Balanced Acc",
                  "roc_auc": "AUROC", "train_time": "Train Time (s)"}
        lines = []
        for name in self.METRICS:
            vals = getattr(self, name)
            if not vals:
                continue
            lines.append(f"{indent}{labels[name] + ':':<16}"
                         f"{np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")
        return "\n".join(lines)

    def last(self, name):
        """Most recent value of ``name``, for per-run progress prints."""
        vals = getattr(self, name)
        return vals[-1] if vals else float("nan")
