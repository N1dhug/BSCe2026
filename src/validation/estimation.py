"""Finite scalar/vector and response-grid contracts used by deterministic G."""
import numpy as np
from numpy.typing import ArrayLike,NDArray

def finite_vector(values: ArrayLike, name: str, *, minimum_size: int = 1) -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise ValueError(f'{name} must be real-valued.')
    array = np.asarray(values, dtype=float)
    if (array.ndim != 1 or array.size < minimum_size
            or not np.all(np.isfinite(array))):
        raise ValueError(f'{name} must be a finite one-dimensional vector of size >= {minimum_size}.')
    return array.copy()

def finite_scalar(value: float, name: str, *, positive: bool = False) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind in ('b', 'c'):
        raise TypeError(f'{name} must be a real scalar, not an array or bool.')
    try:
        scalar = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f'{name} must be a real scalar.') from exc
    if not np.isfinite(scalar) or (positive and scalar <= 0.0):
        qualifier = 'finite and strictly positive' if positive else 'finite'
        raise ValueError(f'{name} must be {qualifier}.')
    return scalar

def positive_integer(value, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be an integer.')
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind in ('b', 'c'):
        raise TypeError(f'{name} must be an integer.')
    try:
        integer = int(array)
        same = float(integer) == float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f'{name} must be an integer.') from exc
    if not same or integer < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return integer

def increasing_grid(values: ArrayLike, name: str, *, minimum_size: int = 2) -> NDArray[np.float64]:
    array = finite_vector(values, name, minimum_size=minimum_size)
    if np.any(np.diff(array) <= 0.0):
        raise ValueError(f'{name} must be strictly increasing.')
    return array
