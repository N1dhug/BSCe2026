"""BM1 stochastic price-optimization utilities.

The primary optimization backend is the deterministic pure finite-volume
Fokker-Planck reference.  Candidate prices are piecewise constant on supplied
intervals and are bounded to [0, 1].

The module intentionally contains no Monte Carlo optimizer: MC is reserved for
final validation with common random numbers.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.method.bm_1_stochastic_objective import (
    StochasticTrackingResult,
    discrete_demand_moments,
    midpoint_quadrature,
    stochastic_tracking_objective,
)
from src.tools.input_generator.input_builder import input_builder


@dataclass(frozen=True)
class PriceObjectiveEvaluation:
    """One deterministic stochastic-objective evaluation for a price vector."""

    prices: np.ndarray
    objective: StochasticTrackingResult
    cell_mass: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class MultistartOptimizationResult:
    """Collection of bounded Powell solves and the best successful result."""

    best: object
    results: tuple
    starts: np.ndarray


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
            "interval_edges must be finite, strictly increasing and span "
            "the complete model interval."
        )
    return edges


def _validate_prices(prices, n_intervals):
    prices = np.asarray(prices, dtype=float).reshape(-1)
    if (
        prices.size != n_intervals
        or not np.all(np.isfinite(prices))
        or np.any(prices < 0.0)
        or np.any(prices > 1.0)
    ):
        raise ValueError(
            "prices must contain one finite value in [0, 1] per price interval."
        )
    return prices


def demand_grid(model, parameters, times, x):
    """Return D(x_i,t_k) on a common state grid."""
    times = np.asarray(times, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    if times.size == 0 or x.size == 0:
        raise ValueError("times and x must be non-empty.")

    values = np.empty((times.size, x.size), dtype=float)
    for index, time in enumerate(times):
        _, _, _, _, _, demand = model.eval(
            float(time),
            x,
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[4],
            parameters[5],
        )
        values[index] = np.asarray(demand, dtype=float)
    return values


class FVStochasticPriceEvaluator:
    """Evaluate stochastic BM1 tracking cost with pure finite volume."""

    def __init__(
        self,
        model,
        parameter_template,
        interval_edges,
        sigma_x,
        initial_mass,
        *,
        objective_max_step,
        solver_max_step,
    ):
        self.model = model
        self.parameter_template = list(parameter_template)
        if len(self.parameter_template) != 6:
            raise ValueError("parameter_template must contain [U, f, g, C, k, Xi].")

        self.interval_edges = _validate_interval_edges(
            interval_edges,
            model.T,
        )
        self.breakpoints = self.interval_edges[1:-1].copy()
        self.n_intervals = self.interval_edges.size - 1

        self.sigma_x = float(sigma_x)
        if not np.isfinite(self.sigma_x) or self.sigma_x < 0.0:
            raise ValueError("sigma_x must be finite and non-negative.")

        self.initial_mass = np.asarray(initial_mass, dtype=float).reshape(-1)
        if (
            self.initial_mass.size < 3
            or not np.all(np.isfinite(self.initial_mass))
            or np.min(self.initial_mass) < 0.0
            or not np.isclose(
                np.sum(self.initial_mass),
                1.0,
                atol=1.0e-10,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "initial_mass must be a finite non-negative probability vector."
            )

        self.edges, self.x, self.dx = model.fokker_planc_reference.grid(
            self.initial_mass.size,
            bounds=(0.0, 1.0),
        )
        (
            self.objective_edges,
            self.midpoints,
            self.widths,
            self.solver_t_eval,
        ) = midpoint_quadrature(
            model.T,
            breakpoints=self.breakpoints,
            max_step=objective_max_step,
        )

        self.solver_max_step = float(solver_max_step)
        if (
            np.isnan(self.solver_max_step)
            or self.solver_max_step <= 0.0
        ):
            raise ValueError("solver_max_step must be strictly positive.")

        self.target = np.asarray(model.A(self.midpoints), dtype=float).reshape(-1)
        if self.target.size != self.midpoints.size or not np.all(np.isfinite(self.target)):
            raise ValueError("model target A(t) must return finite midpoint values.")

    def validate_prices(self, prices):
        return _validate_prices(prices, self.n_intervals)

    def parameters_for_prices(self, prices):
        prices = self.validate_prices(prices)
        price = input_builder.price_intervals(
            (self.interval_edges, prices)
        )
        parameters = list(self.parameter_template)
        parameters[0] = price
        return parameters

    def evaluate(self, prices):
        prices = self.validate_prices(prices)
        parameters = self.parameters_for_prices(prices)

        result = self.model.fokker_planc_reference_solver(
            self.initial_mass,
            parameters,
            self.sigma_x,
            t_eval=self.solver_t_eval,
            breakpoints=self.breakpoints,
            max_step=self.solver_max_step,
        )
        (
            t,
            x,
            _,
            cell_mass,
            _,
            _,
            _,
            _,
            _,
            diagnostics,
        ) = result
        np.testing.assert_array_equal(t, self.solver_t_eval)
        np.testing.assert_allclose(x, self.x, atol=0.0, rtol=0.0)

        midpoint_mass = cell_mass[1:-1]
        demand_values = demand_grid(
            self.model,
            parameters,
            self.midpoints,
            self.x,
        )
        expected, second, _ = discrete_demand_moments(
            demand_values,
            midpoint_mass,
        )
        objective = stochastic_tracking_objective(
            self.widths,
            expected,
            second,
            self.target,
        )
        return PriceObjectiveEvaluation(
            prices=prices.copy(),
            objective=objective,
            cell_mass=cell_mass.copy(),
            diagnostics=dict(diagnostics),
        )

    def __call__(self, prices):
        return self.evaluate(prices).objective.total


def multistart_powell(
    objective,
    starts,
    *,
    lower=0.0,
    upper=1.0,
    options=None,
):
    """Minimize a deterministic price objective from several feasible starts."""
    starts = np.asarray(starts, dtype=float)
    if starts.ndim == 1:
        starts = starts[None, :]
    if (
        starts.ndim != 2
        or starts.shape[0] < 1
        or starts.shape[1] < 1
        or not np.all(np.isfinite(starts))
    ):
        raise ValueError("starts must be a finite non-empty 2D array.")

    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("optimization bounds must be finite with lower < upper.")
    if np.any(starts < lower) or np.any(starts > upper):
        raise ValueError("every optimization start must lie inside the bounds.")

    # Preserve order while removing duplicate starts.
    unique = []
    for start in starts:
        if not any(np.array_equal(start, previous) for previous in unique):
            unique.append(start.copy())
    starts = np.asarray(unique, dtype=float)

    settings = {
        "xtol": 1.0e-5,
        "ftol": 1.0e-9,
        "maxiter": 100,
    }
    if options is not None:
        settings.update(options)

    bounds = [(lower, upper)] * starts.shape[1]
    results = []
    for start in starts:
        result = minimize(
            lambda candidate: float(objective(candidate)),
            start,
            method="Powell",
            bounds=bounds,
            options=settings,
        )
        results.append(result)

    successful = [
        result
        for result in results
        if bool(result.success) and np.isfinite(result.fun)
    ]
    if not successful:
        messages = "; ".join(str(result.message) for result in results)
        raise RuntimeError(
            "Every bounded Powell start failed. Messages: " + messages
        )

    best = min(successful, key=lambda result: float(result.fun))
    return MultistartOptimizationResult(
        best=best,
        results=tuple(results),
        starts=starts.copy(),
    )


def coarse_price_surface(evaluator, values):
    """Evaluate a two-price objective on a Cartesian coarse grid."""
    if evaluator.n_intervals != 2:
        raise ValueError("coarse_price_surface currently requires exactly two intervals.")
    values = np.asarray(values, dtype=float).reshape(-1)
    if (
        values.size < 2
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError("values must be strictly increasing and lie in [0, 1].")

    objective = np.empty((values.size, values.size), dtype=float)
    for i, first in enumerate(values):
        for j, second in enumerate(values):
            objective[i, j] = evaluator([first, second])
    return objective


def best_surface_starts(values, objective, n_best):
    """Return the n_best unique price vectors from a coarse 2D surface."""
    values = np.asarray(values, dtype=float).reshape(-1)
    objective = np.asarray(objective, dtype=float)
    n_best = int(n_best)
    if (
        objective.shape != (values.size, values.size)
        or not np.all(np.isfinite(objective))
        or n_best < 1
    ):
        raise ValueError("invalid coarse surface or n_best.")

    order = np.argsort(objective, axis=None)
    starts = []
    for flat_index in order[: min(n_best, order.size)]:
        i, j = np.unravel_index(flat_index, objective.shape)
        starts.append([values[i], values[j]])
    return np.asarray(starts, dtype=float)
