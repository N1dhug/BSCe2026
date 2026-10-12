"""BM3-specific checks; shared input/fit checks remain those of BM1."""
import numpy as np


def configuration(restoring_strength=10.0, state_bounds=(-1.0, 2.0), tail_tolerance=1e-4):
    strength = float(restoring_strength)
    bounds = np.asarray(state_bounds, dtype=float)
    tolerance = float(tail_tolerance)
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError('BM3 restoring_strength must be finite and positive.')
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not bounds[0] < 0 < 1 < bounds[1]:
        raise ValueError('BM3 numerical state bounds must extend below 0 and above 1.')
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('BM3 tail_tolerance must be in (0,1).')
    return strength, tuple(map(float, bounds)), tolerance


def finite_state(x):
    x = np.asarray(x, dtype=float)
    if not x.size or not np.isfinite(x).all():
        raise ValueError('BM3 requires finite states, including states outside [0,1].')
    return x
