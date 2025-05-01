import numpy as np
gpu_enable = True
try:
    #import cupy as cp
    #cupy = cp
    raise ImportError
except ImportError:
    gpu_enable = False
from dezero import Variable


def get_array_module(x):
    """Returns the array module for `x`.

    Args:
        x (dezero.Variable or numpy.ndarray or cupy.ndarray): Values to
            determine whether NumPy or CuPy should be used.

    Returns:
        module: `cupy` or `numpy` is returned based on the argument.
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = np.get_array_module(x)
    return xp


def as_numpy(x):
    """Convert to `numpy.ndarray`.

    Args:
        x (`numpy.ndarray` or `cupy.ndarray`): Arbitrary object that can be
            converted to `numpy.ndarray`.
    Returns:
        `numpy.ndarray`: Converted array.
    """
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return np.asnumpy(x)


def as_cupy(x):
    """Convert to `cupy.ndarray`.

    Args:
        x (`numpy.ndarray` or `cupy.ndarray`): Arbitrary object that can be
            converted to `cupy.ndarray`.
    Returns:
        `cupy.ndarray`: Converted array.
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception('CuPy cannot be loaded. Install CuPy!')
    return np.asarray(x)