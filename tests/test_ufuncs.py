from .fixtures import a, b, matrix, col_vector, row_vector
from logarray import log_array
from logarray.testing import assert_logarray_allclose


def test_matmul(row_vector, col_vector):
    result = row_vector @ col_vector
    my_result = log_array(row_vector) @ log_array(col_vector)
    assert_logarray_allclose(result, my_result)
