import numpy as np
from scipy.special import logsumexp
from typing import List, Dict


HANDLED_FUNCTIONS = {}


def implements(np_function):
    "Register an __array_function__ implementation for RaggedArray objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class LogArray(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, log_values=np.ndarray, sign: np.ndarray = 1):
        self._log_values = log_values
        self._sign = sign

    @property
    def shape(self):
        return self._log_values.shape

    @property
    def size(self):
        return self._log_values.size

    def copy(self):
        return self.__class__(self._log_values.copy(), self._sign)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return self.__class__(self._log_values[idx], self._sign)

    def __str__(self):
        return f'log_array({self._sign}{np.exp(self._log_values)})'

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
            return self._log_values
        assert len(inputs) == 2, f"Only unary and binary operations supported for runlengtharray {len(inputs)}"
        inputs = [as_log_array(i)._log_values for i in inputs]
        if ufunc == np.add:
            return self.__class__(
                np.logaddexp(*inputs, **kwargs))
        if ufunc == np.multiply:
            return self.__class__(
                np.add(*inputs, **kwargs))
        if ufunc == np.divide:
            return self.__class__(
                np.subtract(*inputs, **kwargs))
        if ufunc == np.subtract:
            print(inputs)
            return self.__class__(
                *logsumexp(inputs, b=np.array([1, -1]).reshape((2, ) + tuple(1 for d in self.shape)), return_sign=True, axis=0))
        return NotImplemented

    def __array_function__(self, func: callable, types: List, args: List, kwargs: Dict):
        if func == np.pad:
            return pad(*args, **kwargs)

        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)
        return NotImplemented
        
        print(func, types, args, kwargs)
        return NotImplemented

    def to_array(self):
        return self._sign*np.exp(self._log_values)


def pad(array, pad_width, mode='constant', constant_values=0):
    assert mode == 'constant', mode
    return array.__class__(np.pad(array._log_values, pad_width, mode, constant_values=as_log_array(constant_values)._log_values))


def as_log_array(array):
    if isinstance(array, LogArray):
        return array
    return log_array(array)


def log_array(array):
    if isinstance(array, LogArray):
        return array.copy()
    print('a', array)
    return LogArray(np.log(array))


@implements(np.sum)
def sum(*args, **kwargs):
    arrays = [as_log_array(a) for a in args]
    args = [a._log_values for a in arrays]
    return LogArray(logsumexp(*args, **kwargs))
