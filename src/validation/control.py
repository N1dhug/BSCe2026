"""Input contracts for the BM price-control layer; no model or solver imports."""

import numpy as np



def validate_interval_edges(interval_edges, interval):
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


def validate_prices(prices, n_intervals, name='prices'):
    prices = np.asarray(prices, dtype=float).reshape(-1)
    if (
        prices.size != n_intervals
        or not np.all(np.isfinite(prices))
        or np.any(prices < 0.0)
        or np.any(prices > 1.0)
    ):
        raise ValueError(f'{name} must contain one finite value in [0, 1] per price interval.')
    return prices


def positive_integer(value, name, minimum):
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
