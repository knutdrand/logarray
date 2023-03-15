from numpy.testing import assert_allclose
import numpy as np


def assert_logarray_allclose(array1, array2):
    assert_allclose(np.log(array1), np.log(array2))
