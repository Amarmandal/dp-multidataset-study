import numpy as np
import pytest

from common_svm import DifferentiallyPrivateSVM


def test_huber_loss_matches_equation_7():
    svm = DifferentiallyPrivateSVM(h=0.5)
    margins = np.array([2.0, 1.0, 0.0])

    loss = svm.huber_loss(margins)

    expected = np.array([0.0, ((1 + 0.5 - 1.0) ** 2) / (4 * 0.5), 1.0])
    np.testing.assert_allclose(loss, expected)

# test c = 1/2h
@pytest.mark.parametrize(
    ("h", "expected_c"),
    [(0.5, 1.0), (1.0, 0.5), (0.25, 2.0)],
)
def test_huber_second_derivative_bound(h, expected_c):
    svm = DifferentiallyPrivateSVM(h=h)

    assert svm.c == pytest.approx(expected_c)
