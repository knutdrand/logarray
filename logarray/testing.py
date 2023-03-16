from numpy.testing import assert_allclose
from . import LogArray
import numpy as np


relative_tolerance = 1e-5
max_value = 9.999999944957273e+32


def assert_logarray_allclose(*arrays, rtol=relative_tolerance, atol=0):
    arrays = [array.to_array() if isinstance(array, LogArray)
              else array for array in arrays]
    assert all(a.shape == arrays[0].shape for a in arrays), arrays
    assert_allclose(*arrays, rtol=rtol, atol=atol)


def assert_logarray_allclose_log_space(array1, array2):
    assert_allclose(np.log(array1), np.log(array2))
