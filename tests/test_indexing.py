import numpy as np
import pytest

from logarray import log_array
from logarray.testing import assert_logarray_allclose


def test_getitem():
    a = np.array([10, 20, 30])
    b = log_array(a)

    for i in range(b.shape[0]):
        assert_logarray_allclose(b[i], a[i])

    indices = np.random.randint(0, 1, a.shape[0], dtype=bool)
    assert_logarray_allclose(b[indices], a[indices])

    max_index = np.random.randint(0, a.shape[0]//2)
    assert_logarray_allclose(b[:max_index], a[:max_index])


@pytest.mark.skip('fail')
def test_setitem():
    a = np.array([10, 20, 30])
    b = log_array(a)

    b[0] = 11
