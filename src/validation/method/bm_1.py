"""Input checks used by BM1-only numerical/control methods. No solver calls."""

import numpy as np


def _validate_dynamic_controls(t_eval, breakpoints, max_step, cfl):
    t_eval = np.asarray(t_eval, dtype=float)
    if (
        t_eval.ndim != 1
        or t_eval.size < 2
        or not np.all(np.isfinite(t_eval))
        or np.any(np.diff(t_eval) <= 0.0)
    ):
        raise ValueError('t_eval must be a finite strictly increasing one-dimensional grid.')
    breakpoints = np.asarray([] if breakpoints is None else breakpoints, dtype=float).reshape(-1)
    if (
        not np.all(np.isfinite(breakpoints))
        or np.any(np.diff(breakpoints) <= 0.0)
        or np.any(breakpoints <= t_eval[0])
        or np.any(breakpoints >= t_eval[-1])
    ):
        raise ValueError('breakpoints must be strictly increasing and inside the time interval.')
    max_step = float(max_step)
    if np.isnan(max_step) or max_step <= 0.0:
        raise ValueError('max_step must be strictly positive.')
    cfl = float(cfl)
    if not np.isfinite(cfl) or not 0.0 < cfl < 1.0:
        raise ValueError('cfl must lie strictly between zero and one.')
    return (t_eval, breakpoints, max_step, cfl)


def _validate_interval_edges(interval_edges, interval):
    edges = np.asarray(interval_edges, dtype=float).reshape(-1)
    interval = np.asarray(interval, dtype=float).reshape(-1)
    if (
        interval.size != 2
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
        or edges[0] != interval[0]
        or edges[-1] != interval[1]
    ):
        raise ValueError(
            'interval_edges must be finite, strictly increasing and span the complete model interval.',
        )
    return edges.copy()


def _validate_prices(prices, n_intervals, name='prices'):
    prices = np.asarray(prices, dtype=float).reshape(-1)
    if (
        prices.size != n_intervals
        or not np.all(np.isfinite(prices))
        or np.any(prices < 0.0)
        or np.any(prices > 1.0)
    ):
        raise ValueError(f'{name} must contain one finite value in [0, 1] per price interval.')
    return prices


def _positive_integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be an integer.')
    original = value
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f'{name} must be an integer.') from exc
    try:
        same_value = float(value) == float(original)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f'{name} must be an integer.') from exc
    if not same_value or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return value


def _price_vector(values, n_intervals, name):
    """Own a validated public price vector, e.g. a recorded external signal."""
    return _validate_prices(values, n_intervals, name=name).copy()
