"""Small response-geometry checks shared by active BM models.

Keep these checks independent of the archived estimation pipeline.
"""
import numpy as np


def finite_vector(values, name):
    if np.iscomplexobj(values): raise ValueError(f'{name} must be real.')
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError(f'{name} must be a nonempty finite vector.')
    return values


def increasing_grid(values, name):
    values = finite_vector(values, name)
    if len(values) < 2 or np.any(np.diff(values) <= 0):
        raise ValueError(f'{name} must be strictly increasing with at least two points.')
    return values


def positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)
