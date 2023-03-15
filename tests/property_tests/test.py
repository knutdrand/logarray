from hypothesis import given, example
import hypothesis.extra.numpy as stnp
import numpy as np
import pytest
from logarray import log_array
from logarray.testing import assert_logarray_allclose_log_space


@pytest.mark.skip('failing')
@given(stnp.arrays(shape=stnp.array_shapes(), dtype=float))
def test_add_scalar(array):
    assert_logarray_allclose_log_space(
        array+10,
        log_array(array)+10)
