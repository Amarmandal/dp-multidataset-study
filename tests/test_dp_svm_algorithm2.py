# tests/test_dp_svm_algorithm2.py

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import gamma, kstest

from common_svm import DifferentiallyPrivateSVM


def test_epsilon_prime_without_fallback(binary_unit_ball_data):
    epsilon = 10.0
    Lambda = 1.0
    h = 0.5

    X, y = binary_unit_ball_data

    svm = DifferentiallyPrivateSVM(
        epsilon=epsilon,
        Lambda=Lambda,
        h=h,
        fit_intercept=False,
        normalize=False,
        random_state=42,
    )

    svm.fit(X, y)

    n = len(X)
    c = 1 / (2 * h)

    expected = epsilon - np.log(
        1
        + 2 * c / (n * Lambda)
        + c**2 / (n**2 * Lambda**2)
    )

    assert expected > 0
    assert svm.fallback_triggered_ is False
    assert svm.Delta_ == pytest.approx(0.0)
    assert svm.eps_prime_ == pytest.approx(expected)


def test_delta_fallback():
    epsilon = 0.1
    regularization = 0.01
    h = 0.5
    X = np.array([[0.1, 0.0], [0.0, 0.1], [-0.1, 0.0], [0.0, -0.1]])
    y = np.array([1, 1, -1, -1])
    svm = DifferentiallyPrivateSVM(
        epsilon=epsilon,
        Lambda=regularization,
        h=h,
        fit_intercept=False,
        normalize=False,
        random_state=42,
    )

    svm.fit(X, y)

    n = len(X)
    c = 1 / (2 * h)
    expected_delta = c / (n * (np.exp(epsilon / 4) - 1)) - regularization
    assert svm.fallback_triggered_ is True
    assert svm.eps_prime_ == pytest.approx(epsilon / 2)
    assert svm.Delta_ == pytest.approx(max(expected_delta, 0.0))


def test_noise_radius_matches_gamma_distribution():
    svm = DifferentiallyPrivateSVM()
    dimensions = 5
    eps_prime = 2.0
    beta = eps_prime / 2
    rng = np.random.default_rng(12345)
    samples = np.array(
        [svm._draw_noise(dimensions, eps_prime, rng) for _ in range(10_000)]
    )
    radii = np.linalg.norm(samples, axis=1)

    _, pvalue = kstest(radii, gamma(a=dimensions, scale=1 / beta).cdf)

    assert pvalue > 0.01


def test_analytic_gradient_matches_numerical_gradient():
    svm = DifferentiallyPrivateSVM(
        epsilon=1.0,
        Lambda=0.1,
        h=0.5,
        fit_intercept=False,
    )
    X = np.array([[0.2, 0.1], [-0.3, 0.2], [0.1, -0.4]])
    y = np.array([1, -1, 1])
    params = np.array([0.15, -0.2])
    noise = np.array([0.3, -0.1])
    delta = 0.05

    analytic = svm.grad_objective(params, X, y, noise, delta)
    numerical = approx_fprime(
        params,
        lambda candidate: svm.objective(candidate, X, y, noise, delta),
        epsilon=1e-7,
    )

    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-5)
