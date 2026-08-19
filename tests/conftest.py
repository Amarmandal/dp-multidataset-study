import numpy as np
import pytest


@pytest.fixture
def binary_unit_ball_data():
    """Small binary dataset satisfying Algorithm 2's unit-ball assumption."""
    X = np.array([[0.1, 0.2], [-0.2, 0.1], [0.3, -0.1], [-0.1, -0.2]])
    y = np.array([1, -1, 1, -1])
    return X, y
