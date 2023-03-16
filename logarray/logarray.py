import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp
from typing import List, Dict
from .util import signed_log, add_log_space, sub_log_space, mul_log_space, div_log_space, power_log_space, sign_collapse

# Will be filled in by methods decorated with implements
HANDLED_FUNCTIONS = {}


def implements(np_function):
    "Register an __array_function__ implementation for RaggedArray objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class LogArray(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, log_values=np.ndarray, signs: np.ndarray = 1):
        self._log_values = log_values
        self._signs = signs

    @property
    def shape(self):
        return self._log_values.shape

    @property
    def size(self):
        return self._log_values.size

    def copy(self):
        return self.__class__(self._log_values.copy(), self._signs)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return self.__class__(self._log_values[idx], self._signs)

    def __str__(self):
        return f'log_array({self.to_array()})'

    def __repr__(self):
        return f'log_{repr(self.to_array())}'

    def __array_ufunc__(self, ufunc: callable, method: str, *inputs, **kwargs):
        """Handle numpy unfuncs called on the runlength array
        
        Currently only handles '__call__' modes and unary and binary functions

        Parameters
        ----------
        ufunc : callable
        method : str
        *inputs :
        **kwargs :
        """
        if method not in ("__call__"):
            return NotImplemented
        if ufunc == np.log:
            return np.where(self._signs == 1, self._log_values, np.NaN)
        if ufunc == np.power:
            exponent = inputs[1]
            log_values, signs = power_log_space(self._log_values, self._signs, exponent)
            return self.__class__(log_values, signs)

        assert len(inputs) == 2, f"Only unary and binary operations supported for runlengtharray {len(inputs)}"
        inputs = [as_log_array(i) for i in inputs]
        a, signa, b, signb = inputs[0]._log_values, inputs[0]._signs, inputs[1]._log_values, inputs[1]._signs
        if ufunc == np.add:
            log_values, signs = add_log_space(a, signa, b, signb)
            return self.__class__(log_values, signs)
        if ufunc == np.multiply:
            log_values, signs = mul_log_space(a, signa, b, signb)
            return self.__class__(log_values, signs)
        if ufunc == np.divide:
            log_values, signs = div_log_space(a, signa, b, signb)
            return self.__class__(log_values, signs)
        if ufunc == np.subtract:
            log_values, signs = sub_log_space(a, signa, b, signb)
            return self.__class__(log_values, signs)
        if ufunc == np.matmul:
            return self.__class__(
                logsumexp(a[..., np.newaxis]+b[..., np.newaxis, :, :], axis=-2))
        return NotImplemented

    def __array_function__(self, func: callable, types: List, args: List, kwargs: Dict):
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)
        return NotImplemented
        
        print(func, types, args, kwargs)
        return NotImplemented

    def to_array(self):
        return self._signs * np.exp(self._log_values)


@implements(np.pad)
def pad(array, pad_width, mode='constant', constant_values=0):
    assert mode == 'constant', mode
    return array.__class__(np.pad(array._log_values, pad_width, mode, constant_values=as_log_array(constant_values)._log_values))


def as_log_array(array):
    if isinstance(array, LogArray):
        return array
    return log_array(array)


def log_array(array: ArrayLike) -> LogArray:
    """Create a `LogArray` from an array

    If `array` is a LogArray, make a copy of it

    Parameters
    ----------
    array : ArrayLike
        Array with values in linear space

    Returns
    -------
    LogArray
        LogArray representing the values in `array`

    Examples
    --------
    >>> from logarray import log_array
    >>> a = log_array([1., 2., 3.])
    >>> a
    log_array([1., 2., 3.])
    """

    if isinstance(array, LogArray):
        return array.copy()
    log_values, signs = signed_log(np.asanyarray(array))
    return LogArray(log_values, signs)


@implements(np.sum)
def sum(array, **kwargs):
    # TODO This function is not called when you sum a list of LogArrays, and does not work for array of LogArrays
    s, signs = logsumexp(array._log_values, b = array._signs, return_sign = True, **kwargs)
    return LogArray(s, sign_collapse(signs))


@implements(np.zeros_like)
def zeros_like(array):
    return LogArray(np.full_like(array._log_values, -np.inf), 1)


@implements(np.ones_like)
def ones_like(array):
    return LogArray(np.zeros_like(array._log_values), 1)


@implements(np.full_like)
def full_like(array, value):
    return LogArray(np.full_like(array._log_values, np.log(np.abs(value))), sign_collapse(np.sign(value)))
