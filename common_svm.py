import numpy as np
from scipy.optimize import minimize
from sklearn.svm import LinearSVC


class StandardSVM:
    """Wrapper for a standard linear SVM (LinearSVC).

    Usage:
        clf = StandardSVM(C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    """

    def __init__(self, C=1.0, max_iter=2000, random_state=None, multi_class="ovr"):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.multi_class = multi_class
        self.model = None

    def fit(self, X, y):
        kwargs = dict(C=self.C, max_iter=self.max_iter, random_state=self.random_state)
        # `multi_class` is deprecated in scikit-learn >= 1.5 (removed in 1.7);
        # "ovr" is the only remaining behaviour, so only pass it when overridden.
        if self.multi_class != "ovr":
            kwargs["multi_class"] = self.multi_class
        self.model = LinearSVC(**kwargs)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def decision_function(self, X):
        return self.model.decision_function(X)


class DifferentiallyPrivateSVM:
    """DP linear SVM: Huber loss + objective perturbation.

    Strict implementation of Algorithm 2 of Chaudhuri, Monteleoni & Sarwate
    (2011), *Differentially Private Empirical Risk Minimization*, JMLR 12.

    Mechanism
    ---------
    The hinge loss is replaced by the Huber surrogate of Eq. (7), which is
    convex and doubly differentiable with second-derivative bound c = 1/(2h).
    A random linear term b'w/n is added to the regularized objective, with b
    drawn from the spherically symmetric density nu(b) ~ exp(-beta ||b||_2),
    beta = eps_prime / 2.  Sampling is done as (uniform direction) x (radius
    ~ Gamma(d, 1/beta)).

    Algorithm 2 first computes

        eps_prime = eps - log(1 + 2c/(n*Lambda) + c^2/(n^2 * Lambda^2)).

    If eps_prime > 0 no extra regularization is needed (Delta = 0).  Otherwise
    the mechanism falls back to Delta = c / (n * (exp(eps/4) - 1)) - Lambda and
    eps_prime = eps/2.  Whether that branch fired is recorded in
    ``fallback_triggered_`` -- report it, because on small n it means the
    effective budget is eps/2, not eps.

    Norm precondition
    -----------------
    The guarantee assumes ||x_i||_2 <= 1 for every record.  Per-coordinate
    scaling to [-1, 1] gives only ||x||_2 <= sqrt(d), so pass
    ``normalize=True`` to project each record independently onto the unit L2
    ball: x -> x / max(1, ||x||_2).  The same deterministic projection is
    applied by ``decision_function`` / ``predict`` at inference time.

    Labels for the binary problem must be {-1, +1}.
    """

    def __init__(
        self,
        epsilon=1.0,
        Lambda=0.01,
        h=0.5,
        fit_intercept=True,
        normalize=False,
        random_state=None,
    ):
        self.epsilon = epsilon
        self.Lambda = Lambda
        self.h = h
        self.c = 1.0 / (2.0 * h)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.random_state = random_state

        # Learned parameters.
        self.w = None
        self.b = 0.0
        # Retained for compatibility with existing reports and pickled models.
        # Row-wise projection has no single global scale factor.
        self.scale_ = 1.0

        # Diagnostics (Chaudhuri Algorithm 2 bookkeeping).
        self.eps_prime_ = None
        self.Delta_ = None
        self.fallback_triggered_ = None
        self.max_row_norm_ = None

    # ------------------------------------------------------------------ #
    # Huber loss, Eq. (7)
    # ------------------------------------------------------------------ #
    def huber_loss(self, margin):
        loss = np.zeros_like(margin)
        region2 = np.abs(1 - margin) <= self.h
        region3 = margin < (1 - self.h)
        loss[region2] = ((1 + self.h - margin[region2]) ** 2) / (4 * self.h)
        loss[region3] = 1 - margin[region3]
        return loss

    def huber_grad(self, margin):
        """d(loss)/d(margin)."""
        grad = np.zeros_like(margin)
        region2 = np.abs(1 - margin) <= self.h
        region3 = margin < (1 - self.h)
        grad[region2] = -(1 + self.h - margin[region2]) / (2 * self.h)
        grad[region3] = -1.0
        return grad

    # ------------------------------------------------------------------ #
    # Perturbed objective.  params = [w, b] when fit_intercept else [w].
    # The intercept is NOT regularized and NOT part of ||w||^2.
    # ------------------------------------------------------------------ #
    def _split(self, params):
        if self.fit_intercept:
            return params[:-1], params[-1]
        return params, 0.0

    def objective(self, params, X, y, noise, Delta):
        w, b = self._split(params)
        n = X.shape[0]
        margin = y * (X @ w + b)
        return (
            np.mean(self.huber_loss(margin))
            + 0.5 * self.Lambda * np.dot(w, w)
            + 0.5 * Delta * np.dot(w, w)
            + np.dot(noise, params) / n
        )

    def grad_objective(self, params, X, y, noise, Delta):
        w, b = self._split(params)
        n = X.shape[0]
        margin = y * (X @ w + b)
        dL = self.huber_grad(margin) * y  # d(loss)/dz

        grad_w = (X.T @ dL) / n + (self.Lambda + Delta) * w
        if self.fit_intercept:
            grad_b = np.mean(dL)
            grad = np.concatenate([grad_w, [grad_b]])
        else:
            grad = grad_w
        return grad + noise / n

    # ------------------------------------------------------------------ #
    # Noise: direction uniform on the unit sphere, radius ~ Gamma(d, 1/beta)
    # with beta = eps_prime / 2, giving density ~ exp(-beta ||b||_2).
    # ------------------------------------------------------------------ #
    def _draw_noise(self, d, eps_prime, rng):
        beta = eps_prime / 2.0
        radius = rng.gamma(shape=d, scale=1.0 / beta)
        direction = rng.standard_normal(d)
        nrm = np.linalg.norm(direction)
        if nrm == 0.0:
            direction = np.ones(d)
            nrm = np.linalg.norm(direction)
        return radius * (direction / nrm)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _project_unit_ball(X):
        """Project each row independently onto the closed unit L2 ball."""
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(row_norms, 1.0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        row_norms = np.linalg.norm(X, axis=1)
        self.max_row_norm_ = float(np.max(row_norms)) if len(row_norms) else 0.0

        if self.normalize:
            X = self._project_unit_ball(X)
        self.scale_ = 1.0

        assert np.all(np.linalg.norm(X, axis=1) <= 1 + 1e-9), (
            "||x||_2 <= 1 is required for the Chaudhuri Algorithm 2 guarantee; "
            "pass normalize=True or row-normalize the features before fit()."
        )

        rng = np.random.default_rng(self.random_state)
        n, n_features = X.shape
        d = n_features + 1 if self.fit_intercept else n_features
        Lambda, c, eps = self.Lambda, self.c, self.epsilon

        # --- Algorithm 2, step 1: eps_prime and the Delta fallback ---------
        adjustment = np.log(
            1.0 + (2.0 * c) / (n * Lambda) + (c**2) / (n**2 * Lambda**2)
        )
        eps_prime = eps - adjustment

        if eps_prime > 0:
            Delta = 0.0
            self.fallback_triggered_ = False
        else:
            Delta = c / (n * (np.exp(eps / 4.0) - 1.0)) - Lambda
            eps_prime = eps / 2.0
            self.fallback_triggered_ = True

        Delta = max(Delta, 0.0)
        self.eps_prime_ = float(eps_prime)
        self.Delta_ = float(Delta)

        noise = self._draw_noise(d, eps_prime, rng)

        result = minimize(
            self.objective,
            np.zeros(d),
            args=(X, y, noise, Delta),
            jac=self.grad_objective,
            method="L-BFGS-B",
            options={"maxiter": 1000},
        )
        if not result.success:
            print(f"Optimization warning: {result.message}")

        w_hat, b_hat = self._split(result.x)
        self.w = w_hat
        self.b = float(b_hat)
        return self

    # ------------------------------------------------------------------ #
    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if self.normalize:
            X = self._project_unit_ball(X)
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def privacy_report(self):
        """Diagnostics for the methodology section / privacy audit."""
        return {
            "epsilon": self.epsilon,
            "Lambda": self.Lambda,
            "h": self.h,
            "c": self.c,
            "eps_prime": self.eps_prime_,
            "Delta": self.Delta_,
            "fallback_triggered": self.fallback_triggered_,
            "max_row_norm_before_projection": self.max_row_norm_,
            # Backward-compatible alias for existing report consumers.
            "max_row_norm_before_scaling": self.max_row_norm_,
            "normalize": self.normalize,
            "scale_": self.scale_,
            "fit_intercept": self.fit_intercept,
        }


class DPSVMOneVsRest:
    """One-vs-Rest wrapper for multi-class DP-SVM.

    Every record participates in all m binary problems, so the budget composes
    sequentially and epsilon is split evenly: each constituent classifier is
    trained at epsilon/m.  State this in the methodology when reporting the
    multi-class results.
    """

    def __init__(
        self,
        epsilon=1.0,
        Lambda=0.01,
        h=0.5,
        fit_intercept=True,
        normalize=False,
        random_state=None,
    ):
        self.epsilon = epsilon
        self.Lambda = Lambda
        self.h = h
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.random_state = random_state
        self.classifiers = {}
        self.classes_ = None
        self.epsilon_per_classifier_ = None

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = sorted(set(y.tolist()))
        m = len(self.classes_)
        per_eps = float(self.epsilon) / max(1, m)
        self.epsilon_per_classifier_ = per_eps

        for i, cls in enumerate(self.classes_):
            y_binary = np.where(y == cls, 1, -1)
            seed = (self.random_state or 0) + i * 100
            clf = DifferentiallyPrivateSVM(
                epsilon=per_eps,
                Lambda=self.Lambda,
                h=self.h,
                fit_intercept=self.fit_intercept,
                normalize=self.normalize,
                random_state=seed,
            )
            clf.fit(X, y_binary)
            self.classifiers[cls] = clf
        return self

    def decision_function(self, X):
        return np.column_stack(
            [self.classifiers[cls].decision_function(X) for cls in self.classes_]
        )

    def predict(self, X):
        scores = self.decision_function(X)
        return np.array(self.classes_)[np.argmax(scores, axis=1)]

    def privacy_report(self):
        return {
            "epsilon_total": self.epsilon,
            "n_classifiers": len(self.classes_) if self.classes_ else None,
            "epsilon_per_classifier": self.epsilon_per_classifier_,
            "any_fallback_triggered": any(
                c.fallback_triggered_ for c in self.classifiers.values()
            )
            if self.classifiers
            else None,
            "per_class": {
                cls: clf.privacy_report() for cls, clf in self.classifiers.items()
            },
        }


__all__ = [
    "StandardSVM",
    "DifferentiallyPrivateSVM",
    "DPSVMOneVsRest",
]
