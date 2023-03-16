import numpy as np
from logarray import log_array
from logarray.testing import assert_logarray_allclose
import pytest


@pytest.fixture
def vector():
    return np.array([4, 5])


@pytest.fixture
def mixed_vector():
    return np.array([4, -5])


@pytest.fixture
def vector3():
    return np.array([4, 5, 6])


@pytest.fixture
def mixed_vector3():
    return np.array([-4, 5, -6])


@pytest.fixture
def row_vector():
    return np.array([[1, 2]])


@pytest.fixture
def col_vector():
    return np.array([[2], [3]])


@pytest.fixture
def col_vector3():
    return np.array([[2], [3], [4]])


@pytest.fixture
def matrix23():
    return np.array([[3, 4, 5], [5, 6, 7]])


@pytest.fixture
def matrices():
    return np.arange(4*3*2).reshape(4, 2, 3)


#n@pytest.fixture
#def mixed_matrices(matrices):
#    matrices[np.arange(matrices).size % 3] == 



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


def test_matmuls_col(col_vector3, matrices):
    result = matrices @ col_vector3
    my_result = log_array(matrices) @ log_array(col_vector3)
    assert_logarray_allclose(result, my_result)


def test_matmul_vec(vector):
    result = vector @ vector
    my_result = log_array(vector) @ log_array(vector)
    assert_logarray_allclose(result, my_result)


def test_matmul_vec_2(row_vector, vector):
    result = row_vector @ vector
    my_result = log_array(row_vector) @ log_array(vector)
    assert_logarray_allclose(result, my_result)


def test_matmul_vec_2_mixed(row_vector, mixed_vector):
    result = row_vector @ mixed_vector
    my_result = log_array(row_vector) @ log_array(mixed_vector)
    assert_logarray_allclose(result, my_result)


def test_matmul_vec_3(col_vector, vector):
    result = vector @ col_vector
    my_result = log_array(vector) @ log_array(col_vector)
    assert_logarray_allclose(result, my_result)


def test_matmuls_vec(matrices, vector3):
    result = matrices @ vector3
    my_result = log_array(matrices) @ log_array(vector3)
    assert_logarray_allclose(result, my_result)


def test_matmuls_vec_mixed(matrices, mixed_vector3):
    result = matrices @ mixed_vector3
    my_result = log_array(matrices) @ log_array(mixed_vector3)
    assert_logarray_allclose(result, my_result)
