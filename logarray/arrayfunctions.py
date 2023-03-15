import numpy.typing as npt
import numpy as np

HANDLED_FUNCTIONS = {}


def implements(np_function):
    "Register an __array_function__ implementation for RaggedArray objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator
