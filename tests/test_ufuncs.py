import numpy as np
from logarray import log_array
from logarray.testing import assert_logarray_allclose
import pytest


@pytest.fixture
def vector():
    return np.array([4, 5])


@pytest.fixture
def row_vector():
    return np.array([[1, 2]])


@pytest.fixture
def col_vector():
    return np.array([[2], [3]])


@pytest.fixture
def matrix23():
    return np.array([[3, 4, 5], [5, 6, 7]])


@pytest.fixture
def matrices():
    return np.arange(4*3*2).reshape(4, 2, 3)


def test_matmul(row_vector, col_vector):
    result = row_vector @ col_vector
    my_result = log_array(row_vector) @ log_array(col_vector)
    assert_logarray_allclose(result, my_result)


def test_matmul_outer(row_vector, col_vector):
    result = col_vector @ row_vector
    my_result = log_array(col_vector) @ log_array(row_vector)
    assert_logarray_allclose(result, my_result)



def test_matmul2(row_vector, matrix23):
    result = row_vector @ matrix23
    my_result = log_array(row_vector) @ log_array(matrix23)
    assert_logarray_allclose(result, my_result)


def test_matmuls(row_vector, matrices):
    result = row_vector @ matrices
    my_result = log_array(row_vector) @ log_array(matrices)
    assert_logarray_allclose(result, my_result)


def test_matmul_vec(vector):
    result = vector @ vector
    my_result = log_array(vector) @ log_array(vector)
    assert_logarray_allclose(result, my_result)
