import numpy as np

from logarray import log_array
from logarray.testing import assert_logarray_allclose
import pytest

array_functions = [np.sum]
array_functions_future = [np.product]
test_arrays = [np.array([1,2,3]), np.array([-1,-2,-3]), np.array([1,2,-3])]
@pytest.mark.skip('fail')
@pytest.mark.parametrize("regular", test_arrays)
def test_sum(regular):
    mylogarray = log_array(regular)
    assert np.sum(mylogarray), regular.sum()
    logarray_sum = np.log(np.sum(mylogarray))
    regular_sum = np.log(np.sum(regular))
    assert_logarray_allclose(logarray_sum, regular_sum)

def _test_array_function(regular, array_func):
    mylogarray = log_array(regular)

    logarray_product = array_func(mylogarray)
    regular_product = array_func(regular)
    assert_logarray_allclose(logarray_product, regular_product, rtol=1e-2 if regular.dtype=='float16' else 1e-5)



@pytest.mark.parametrize("array_func", array_functions)
@pytest.mark.parametrize("regular", test_arrays)
def test_array_function(regular, array_func):
    _test_array_function(regular, array_func)

@pytest.mark.parametrize("array_func", array_functions_future)
@pytest.mark.parametrize("regular", test_arrays)
@pytest.mark.xfail('fail')
def test_array_function_future(regular, array_func):
    _test_array_function(regular, array_func)