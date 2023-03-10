import numpy as np
from scipy.special import logsumexp


class LogArray(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, log_values=np.ndarray, sign: int = 1):
        self._log_values = log_values
        self._sign = sign

    @property
    def shape(self):
        return self._log_values.shape

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
        if len(inputs) == 1:
            return self.__class__(self._events, ufunc(self._values))
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

    def to_array(self):
        return self._sign*np.exp(self._log_values)


def as_log_array(array):
    if isinstance(array, LogArray):
        return array
    return log_array(array)


def log_array(array):
    if isinstance(array, LogArray):
        return array.copy()
    return LogArray(np.log(array))
