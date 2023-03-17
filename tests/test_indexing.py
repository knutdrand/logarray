import numpy as np

from logarray import log_array
from logarray.testing import assert_logarray_allclose


def test_setitem():
    a = np.array([10, 20, 30])
    b = log_array(a)

    b[0] = 11
    assert_logarray_allclose(np.array([11, 20, 30]), b)

    b[1:] = [0.1, 3.2]
    assert_logarray_allclose(np.array([11, 0.1, 3.2]), b)

    b[1:] = log_array(np.array([5, 6]))
    assert_logarray_allclose(np.array([11, 5, 6]), b)
