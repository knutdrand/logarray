import numpy as np
from logarray import log_array
from logarray.testing import assert_logarray_allclose
import pytest


@pytest.fixture
def row_vector():
    return np.array([[1, 2]])


@pytest.fixture
def col_vector():
    return np.array([[2], [3]])


@pytest.fixture
def matrix():
    return np.array([[3, 4, 5], [5, 6, 7]])


def test_matmul(row_vector, col_vector):
    result = row_vector @ col_vector
    my_result = log_array(row_vector) @ log_array(col_vector)
    assert_logarray_allclose(result, my_result)
