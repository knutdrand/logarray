from numpy.testing import assert_allclose
from . import LogArray
import numpy as np


def assert_logarray_allclose(*arrays, rtol=1e-7, atol=0):
    arrays = [array.to_array() if isinstance(array, LogArray)
              else array for array in arrays]
    assert_allclose(*arrays, rtol=rtol, atol=atol)


def assert_logarray_allclose_log_space(array1, array2):
    assert_allclose(np.log(array1), np.log(array2))
