import pytest
from hypothesis import given, example
import hypothesis.extra.numpy as stnp
from hypothesis.strategies import composite
from logarray import log_array
from logarray.testing import assert_logarray_allclose
import numpy as np

VALID_LOGARRAY_DTYPES = [np.integer, np.floating]
def is_valid_array(array):
    return any(np.issubdtype(array.dtype, valid_type) for valid_type in VALID_LOGARRAY_DTYPES)

def test_is_valid_array():
    assert is_valid_array(np.array([1,2,3],dtype=int))
    assert not is_valid_array(np.array([False, True], dtype=bool))
@composite
def valid_arrays(draw):
    return draw(stnp.arrays(shape=(4, ), dtype=stnp.floating_dtypes()))

@given(valid_arrays())
@example(np.array([1,1,1,1]))
@pytest.mark.skip("Failing, due to lacking support for negative numbers")
def test_sum(regular):
    mylogarray = log_array(regular)

    logarray_sum = np.sum(mylogarray)
    regular_sum = np.sum(regular)
    assert_logarray_allclose(logarray_sum, regular_sum, rtol=1e-2 if regular.dtype=='float16' else 1e-5)

print(valid_arrays().example())