#!/usr/bin/env python

"""Tests for `logarray` package."""

import pytest
from logarray.logarray import log_array
from logarray.testing import assert_logarray_allclose, assert_logarray_allclose_log_space
from numpy.testing import assert_allclose
import numpy as np


@pytest.fixture
def a():
    return np.array([1, 2, 3])


@pytest.fixture
def b():
    return np.array([10, 20, 30])


def test_add_logarray(a, b):
    true = a + b
    result = log_array(a) + log_array(b)
    assert_logarray_allclose(result, true)


def test_multiply_logarray(a, b):
    true = a * b
    result = log_array(a) * log_array(b)
    assert_logarray_allclose(result, true)


def test_divide_logarray(a, b):
    true = a / b
    result = log_array(a) / log_array(b)
    assert_logarray_allclose(result, true)


def test_subtract_logarray(a, b):
    true = a - b
    result = log_array(a) - log_array(b)
    assert_logarray_allclose(result, true)


def test_log(a):
    assert_allclose(np.log(log_array(a)), np.log(a))


def test_sum(a):
    assert_allclose(np.sum(a), np.sum(log_array(a)).to_array())
