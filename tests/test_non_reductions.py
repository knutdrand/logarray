import numpy as np
import pytest


@pytest.fixture
def simple_array():
    return np.arange(4)

