import numpy as np
from logarray.testing import assert_logarray_allclose
from logarray import log_array
import pytest


@pytest.fixture
def simple_array():
    return np.arange(4)


# @pytest.mark.skip('fail')
def test_zeros_like(simple_array):
    assert_logarray_allclose(
        np.zeros_like(simple_array),
        np.zeros_like(log_array(simple_array)))


def test_ones_like(simple_array):
    assert_logarray_allclose(
        np.ones_like(simple_array),
        np.ones_like(log_array(simple_array)))


def test_full_like(simple_array):
    assert_logarray_allclose(
        np.full_like(simple_array, 10),
        np.full_like(log_array(simple_array), 10))
