#!/usr/bin/env python

"""Tests for `logarray` package."""

import pytest
from logarray.logarray import log_array
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
    assert_allclose(result.to_array(), true)


def test_multiply_logarray(a, b):
    true = a * b
    result = log_array(a) * log_array(b)
    assert_allclose(result.to_array(), true)


def test_divide_logarray(a, b):
    true = a / b
    result = log_array(a) / log_array(b)
    assert_allclose(result.to_array(), true)


def test_subtract_logarray(a, b):
    true = a - b
    result = log_array(a) - log_array(b)
    assert_allclose(result.to_array(), true)
