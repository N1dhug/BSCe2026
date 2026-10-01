"""Strict input contracts for F4 estimation primitives; no models or solvers.

Unlike physical states, Gaussian demand observations need not lie in [0, 1].
Invalid probability input is rejected, never clipped or silently normalized.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

MASS_ATOL = 1e-10


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


def probability_mass(values: ArrayLike, name: str = 'mass') -> NDArray[np.float64]:
    mass = finite_vector(values, name)
    if np.any(mass < 0.0) or abs(float(np.sum(mass)) - 1.0) > MASS_ATOL:
        raise ValueError(f'{name} must be non-negative and sum to one; no repair is applied.')
    return mass


def probability_rows(values, name='mass histories', *, atol=MASS_ATOL):
    """Validate and own a matrix of probability vectors without row-copy loops."""
    atol = finite_scalar(atol, 'atol')
    if atol < 0.0:
        raise ValueError('atol must be non-negative.')
    if np.iscomplexobj(values):
        raise ValueError(f'{name} must be real-valued.')
    rows = np.asarray(values, dtype=float)
    if (rows.ndim != 2 or min(rows.shape) < 1
            or not np.all(np.isfinite(rows)) or np.any(rows < 0.0)
            or np.any(np.abs(np.sum(rows, axis=1) - 1.0) > atol)):
        raise ValueError(f'{name} must contain finite non-negative unit-sum rows; no repair is applied.')
    return rows.copy()


def log_probability_mass(values: ArrayLike, name: str = 'log_mass') -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise ValueError(f'{name} must be real-valued.')
    logs = np.asarray(values, dtype=float)
    if (logs.ndim != 1 or logs.size == 0 or np.any(np.isnan(logs))
            or np.any(np.isposinf(logs)) or not np.any(np.isfinite(logs))):
        raise ValueError(f'{name} must contain finite log probabilities or -inf, with nonempty support.')
    if abs(float(np.logaddexp.reduce(logs))) > MASS_ATOL:
        raise ValueError(f'{name} must already be normalized (logsumexp = 0).')
    return logs.copy()


def point_observations(times: ArrayLike, values: ArrayLike, mask: ArrayLike | None):
    """Validate point observations, permitting NaN only at explicitly missing values."""
    times = increasing_grid(times, 'observation times', minimum_size=1)
    if np.iscomplexobj(values):
        raise ValueError('observation values must be real-valued.')
    values = np.asarray(values, dtype=float)
    if values.shape != times.shape:
        raise ValueError('observation values must be one-dimensional and match times.')
    if mask is None:
        observed = np.ones(times.size, dtype=bool)
    else:
        observed = np.asarray(mask)
        if observed.shape != times.shape or observed.dtype.kind != 'b':
            raise ValueError('observation mask must be a boolean vector matching times.')
    if np.any(np.isinf(values)) or not np.all(np.isfinite(values[observed])):
        raise ValueError('observed values must be finite; NaN is permitted only under a False mask.')
    return times, values.copy(), observed.copy()
