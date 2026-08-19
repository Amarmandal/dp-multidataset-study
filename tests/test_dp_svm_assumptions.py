import numpy as np
import pytest

from common_svm import DifferentiallyPrivateSVM


def test_rejects_data_outside_unit_ball_without_normalization():
    X = np.array([[3.0, 4.0], [0.1, 0.2]])
    y = np.array([1, -1])
    svm = DifferentiallyPrivateSVM(fit_intercept=False, normalize=False)

    with pytest.raises(AssertionError, match=r"\|\|x\|\|_2 <= 1"):
        svm.fit(X, y)


def test_normalization_projects_each_record_independently():
    dataset = np.array([[3.0, 4.0], [0.6, 0.8]])
    neighboring_dataset = np.array([[6.0, 8.0], [0.6, 0.8]])
    projected = DifferentiallyPrivateSVM._project_unit_ball(dataset)
    neighboring_projected = DifferentiallyPrivateSVM._project_unit_ball(
        neighboring_dataset
    )

    np.testing.assert_allclose(projected, [[0.6, 0.8], [0.6, 0.8]])
    np.testing.assert_allclose(neighboring_projected, [[0.6, 0.8], [0.6, 0.8]])
    np.testing.assert_allclose(projected[1], neighboring_projected[1])
    assert np.all(np.linalg.norm(projected, axis=1) <= 1 + 1e-12)
    assert np.all(np.linalg.norm(neighboring_projected, axis=1) <= 1 + 1e-12)


def test_prediction_applies_the_same_row_projection_as_training():
    svm = DifferentiallyPrivateSVM(normalize=True, fit_intercept=False)
    svm.w = np.array([2.0, -1.0])
    X = np.array([[3.0, 4.0], [0.6, 0.8]])

    expected = DifferentiallyPrivateSVM._project_unit_ball(X) @ svm.w

    np.testing.assert_allclose(svm.decision_function(X), expected)


def test_unregularized_intercept_has_no_regularizer_curvature():
    """Document why fit_intercept=True is outside the direct theorem setting."""
    svm = DifferentiallyPrivateSVM(Lambda=1.0, h=0.5, fit_intercept=True)
    X = np.zeros((2, 1))
    y = np.ones(2)
    noise = np.zeros(2)
    params_at_intercept_2 = np.array([0.0, 2.0])
    params_at_intercept_3 = np.array([0.0, 3.0])

    objective_at_2 = svm.objective(params_at_intercept_2, X, y, noise, Delta=0)
    objective_at_3 = svm.objective(params_at_intercept_3, X, y, noise, Delta=0)

    assert objective_at_2 == pytest.approx(objective_at_3)
