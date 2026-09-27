"""BM1 demand moments and breakpoint-aware stochastic tracking objectives."""

import numpy as np
from dataclasses import dataclass

_VARIANCE_TOL = 1e-12

@dataclass(frozen=True)
class StochasticTrackingResult:
    """Integrated stochastic tracking objective and its time-local terms."""
    total: float
    bias: float
    variance: float
    expected_demand: np.ndarray
    second_moment: np.ndarray
    demand_variance: np.ndarray
    total_integrand: np.ndarray
    bias_integrand: np.ndarray
    variance_integrand: np.ndarray

def midpoint_quadrature(interval, *, breakpoints=None, max_step):
    """Build midpoint quadrature intervals that never cross a breakpoint.

    Returns edges, midpoints, widths and solver_t_eval. The latter contains
    the model-interval endpoints plus every quadrature midpoint so stochastic
    solvers can return the states/densities needed by the quadrature while
    still spanning the full model horizon.
    """
    interval = np.asarray(interval, dtype=float).reshape(-1)
    if interval.size != 2 or not np.all(np.isfinite(interval)) or interval[1] <= interval[0]:
        raise ValueError('interval must contain two finite values with end > start.')
    breakpoints = np.asarray([] if breakpoints is None else breakpoints, dtype=float).reshape(-1)
    if not np.all(np.isfinite(breakpoints)) or np.any(np.diff(breakpoints) <= 0.0) or np.any(breakpoints <= interval[0]) or np.any(breakpoints >= interval[1]):
        raise ValueError('breakpoints must be strictly increasing and lie inside the interval.')
    max_step = float(max_step)
    if not np.isfinite(max_step) or max_step <= 0.0:
        raise ValueError('max_step must be finite and strictly positive.')
    knots = np.concatenate(([interval[0]], breakpoints, [interval[1]]))
    pieces = []
    for index, (left, right) in enumerate(zip(knots[:-1], knots[1:])):
        width = right - left
        n_substeps = max(1, int(np.ceil(np.nextafter(width / max_step, -np.inf))))
        segment = np.linspace(left, right, n_substeps + 1, dtype=float)
        segment[0], segment[-1] = (left, right)
        pieces.append(segment if index == 0 else segment[1:])
    edges = np.concatenate(pieces)
    widths = np.diff(edges)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    solver_t_eval = np.concatenate(([interval[0]], midpoints, [interval[1]]))
    if np.max(widths) > max_step + 1e-14:
        raise RuntimeError('midpoint quadrature failed to respect max_step.')
    if breakpoints.size:
        for breakpoint in breakpoints:
            if not np.any(edges == breakpoint):
                raise RuntimeError('midpoint quadrature failed to place a breakpoint on an edge.')
            crossing = (edges[:-1] < breakpoint) & (edges[1:] > breakpoint)
            if np.any(crossing):
                raise RuntimeError('a quadrature interval crosses a breakpoint.')
    return (edges, midpoints, widths, solver_t_eval)

def sample_demand_moments(demand_paths):
    """Return E[D], E[D^2] and Var(D) from Monte Carlo demand samples.

    The population second moment/variance convention is used so the discrete
    identity E[(D-A)^2] = (E[D]-A)^2 + Var(D) holds exactly for the sample
    empirical distribution.
    """
    demand_paths = np.asarray(demand_paths, dtype=float)
    if demand_paths.ndim != 2 or demand_paths.shape[0] < 1 or demand_paths.shape[1] < 1 or (not np.all(np.isfinite(demand_paths))):
        raise ValueError('demand_paths must be a finite non-empty 2D array with shape (n_paths, n_times).')
    expected = np.mean(demand_paths, axis=0)
    second = np.mean(np.square(demand_paths), axis=0)
    variance = second - np.square(expected)
    if np.min(variance) < -_VARIANCE_TOL:
        raise RuntimeError('Monte Carlo demand variance became materially negative.')
    variance = np.maximum(variance, 0.0)
    return (expected, second, variance)

def discrete_demand_moments(demand_values, cell_mass):
    """Return demand moments from values on a discrete probability grid."""
    demand_values = np.asarray(demand_values, dtype=float)
    cell_mass = np.asarray(cell_mass, dtype=float)
    if demand_values.ndim != 2 or cell_mass.shape != demand_values.shape or demand_values.size == 0 or (not np.all(np.isfinite(demand_values))) or (not np.all(np.isfinite(cell_mass))):
        raise ValueError('demand_values and cell_mass must be finite equal-shaped 2D arrays.')
    if np.min(cell_mass) < -1e-12:
        raise ValueError('cell_mass must be non-negative.')
    mass = np.sum(cell_mass, axis=1)
    if not np.allclose(mass, 1.0, atol=1e-09, rtol=0.0):
        raise ValueError('each cell_mass row must sum to one.')
    expected = np.sum(demand_values * cell_mass, axis=1)
    second = np.sum(np.square(demand_values) * cell_mass, axis=1)
    variance = second - np.square(expected)
    if np.min(variance) < -_VARIANCE_TOL:
        raise RuntimeError('discrete demand variance became materially negative.')
    variance = np.maximum(variance, 0.0)
    return (expected, second, variance)

def stochastic_tracking_objective(widths, expected_demand, second_moment, target):
    """Integrate E[(D-A)^2] with supplied midpoint quadrature weights."""
    widths = np.asarray(widths, dtype=float).reshape(-1)
    expected_demand = np.asarray(expected_demand, dtype=float).reshape(-1)
    second_moment = np.asarray(second_moment, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    n = widths.size
    if n == 0 or any((array.size != n for array in (expected_demand, second_moment, target))):
        raise ValueError('widths, expected_demand, second_moment and target must have equal non-zero length.')
    if not np.all(np.isfinite(widths)) or not np.all(np.isfinite(expected_demand)) or (not np.all(np.isfinite(second_moment))) or (not np.all(np.isfinite(target))) or np.any(widths <= 0.0):
        raise ValueError('objective inputs must be finite and widths positive.')
    variance = second_moment - np.square(expected_demand)
    if np.min(variance) < -_VARIANCE_TOL:
        raise ValueError('second_moment is inconsistent with expected_demand.')
    variance = np.maximum(variance, 0.0)
    bias_integrand = np.square(expected_demand - target)
    variance_integrand = variance
    total_integrand = bias_integrand + variance_integrand
    bias = float(np.dot(widths, bias_integrand))
    variance_cost = float(np.dot(widths, variance_integrand))
    total = float(np.dot(widths, total_integrand))
    return StochasticTrackingResult(total=total, bias=bias, variance=variance_cost,
        expected_demand=expected_demand.copy(), second_moment=second_moment.copy(), demand_variance=variance.copy(),
        total_integrand=total_integrand.copy(), bias_integrand=bias_integrand.copy(),
        variance_integrand=variance_integrand.copy())

def monte_carlo_tracking_objective(widths, demand_paths, target):
    """Return MC stochastic tracking cost and pathwise standard error."""
    demand_paths = np.asarray(demand_paths, dtype=float)
    widths = np.asarray(widths, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    if demand_paths.ndim != 2 or demand_paths.shape[0] < 2 or demand_paths.shape[1] != widths.size or (target.size != widths.size) or (not np.all(np.isfinite(demand_paths))):
        raise ValueError('demand_paths must have shape (n_paths >= 2, len(widths)) and target must match len(widths).')
    expected, second, _ = sample_demand_moments(demand_paths)
    objective = stochastic_tracking_objective(widths, expected, second, target)
    squared_error = np.square(demand_paths - target[None, :])
    direct_integrand = np.mean(squared_error, axis=0)
    direct_total = float(np.dot(widths, direct_integrand))
    path_objectives = squared_error @ widths
    standard_error = float(np.std(path_objectives, ddof=1) / np.sqrt(demand_paths.shape[0]))
    maximum_decomposition_error = float(np.max(np.abs(direct_integrand - objective.total_integrand)))
    return MonteCarloTrackingResult(objective=objective, path_objectives=path_objectives,
        standard_error=standard_error, direct_total=direct_total,
        maximum_decomposition_error=maximum_decomposition_error)

def demand_grid(model, parameters, times, x):
    """Return D(x_i,t_k) on a common state grid."""
    times = np.asarray(times, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    if times.size == 0 or x.size == 0:
        raise ValueError('times and x must be non-empty.')
    values = np.empty((times.size, x.size), dtype=float)
    for index, time in enumerate(times):
        _, _, _, _, _, demand = model.eval(float(time), x, parameters[0], parameters[1], parameters[2], parameters[4],
            parameters[5])
        values[index] = np.asarray(demand, dtype=float)
    return values

def deterministic_midpoint_objective(model, init_cond, parameters, *, objective_max_step, breakpoints=None,
    solver_max_step=np.inf, rtol=1e-06, atol=1e-08):
    """Evaluate deterministic tracking with the same jump-safe quadrature as H.

    This is an opt-in BM-only alternative to the historical F1 sampled-Simpson
    objective. Supply every known jump in U, B or A through breakpoints. The
    legacy shared tracking_error function and model.optimize are unchanged.
    """
    _, midpoints, widths, output_t = midpoint_quadrature(model.T, breakpoints=breakpoints, max_step=objective_max_step)
    result = model.deterministic_solver(init_cond, parameters, t_eval=output_t, breakpoints=breakpoints, rtol=rtol,
        atol=atol, max_step=solver_max_step)
    demand = np.asarray(result[7], dtype=float)[1:-1]
    target = np.asarray(model.A(midpoints), dtype=float)
    return stochastic_tracking_objective(widths, demand, demand ** 2, target)

@dataclass(frozen=True)
class MonteCarloTrackingResult:
    """Monte Carlo objective estimate with pathwise sampling uncertainty."""
    objective: StochasticTrackingResult
    path_objectives: np.ndarray
    standard_error: float
    direct_total: float
    maximum_decomposition_error: float
