import numpy as np
from hypothesis import given
import hypothesis.extra.numpy as stnp
from hypothesis.strategies import composite, floats

from logarray import log_array
from logarray.testing import assert_logarray_allclose, max_value


@composite
def valid_float_arrays(draw):
    return draw(stnp.arrays(shape=(4,), dtype=stnp.floating_dtypes(sizes=[32, 64]),
                            elements=floats(0, max_value, allow_nan=False, width=32, exclude_min=True)))


@given(valid_float_arrays())
def test_getitem(a):
    b = log_array(a)

    for i in range(b.shape[0]):
        assert_logarray_allclose(b[i], a[i])

    indices = np.random.randint(0, 1, a.shape[0], dtype=bool)
    assert_logarray_allclose(b[indices], a[indices])

    max_index = np.random.randint(0, a.shape[0] // 2)
    assert_logarray_allclose(b[:max_index], a[:max_index])
