"""BM1 stochastic price-optimization utilities.

The primary optimization backend is the deterministic pure finite-volume
Fokker-Planck reference.  Candidate prices are piecewise constant on supplied
intervals and are bounded to [0, 1].

The module intentionally contains no Monte Carlo optimizer: MC is reserved for
final validation with common random numbers.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from src.method.bm_1_beta_mixture import (
    _projection_matrices,
    evolve_projected_density as _evolve_projected_density,
)

from src.method.bm_1_stochastic_objective import (
    StochasticTrackingResult,
    discrete_demand_moments,
    demand_grid,
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
    objective_evaluations: int = 0
    cache_hits: int = 0
    elapsed_seconds: float = 0.0


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
    return edges.copy()


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
        breakpoints=None,
    ):
        self.model = model
        self.parameter_template = list(parameter_template)
        if len(self.parameter_template) != 6:
            raise ValueError("parameter_template must contain [U, f, g, C, k, Xi].")

        self.interval_edges = _validate_interval_edges(
            interval_edges,
            model.T,
        )
        extra_breakpoints = np.asarray(
            [] if breakpoints is None else breakpoints, dtype=float
        )
        if (extra_breakpoints.ndim != 1
                or not np.all(np.isfinite(extra_breakpoints))
                or np.any(extra_breakpoints <= model.T[0])
                or np.any(extra_breakpoints >= model.T[1])):
            raise ValueError("breakpoints must be finite points inside the model interval.")
        self.breakpoints = np.unique(np.concatenate(
            (self.interval_edges[1:-1], extra_breakpoints)
        ))
        self.n_intervals = self.interval_edges.size - 1

        self.sigma_x = float(sigma_x)
        if not np.isfinite(self.sigma_x) or self.sigma_x < 0.0:
            raise ValueError("sigma_x must be finite and non-negative.")

        self.initial_mass = np.asarray(initial_mass, dtype=float).reshape(-1).copy()
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
    progress=None,
    cache=True,
):
    """Minimize a deterministic price objective from several feasible starts.

    Exact duplicate price vectors may reuse scalar values within this call.
    No prices are rounded. Set cache=False for a stateful/non-deterministic
    objective (such objectives are not the intended reference use). Progress
    is an optional callback and never changes stopping tolerances.
    """
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
    memo = {}
    objective_evaluations = 0
    cache_hits = 0
    started = perf_counter()

    def evaluate(candidate):
        nonlocal objective_evaluations, cache_hits
        key = tuple(float(value) for value in candidate)
        if cache and key in memo:
            cache_hits += 1
            return memo[key]
        value = float(objective(candidate))
        if not np.isfinite(value):
            raise RuntimeError("The deterministic price objective returned a non-finite value.")
        objective_evaluations += 1
        if cache:
            memo[key] = value
        if progress is not None and objective_evaluations % 50 == 0:
            progress(f"  {objective_evaluations} objective evaluations; "
                     f"{perf_counter() - started:.1f} s elapsed")
        return value

    for index, start in enumerate(starts, start=1):
        if progress is not None:
            progress(f"Powell start {index}/{len(starts)}: {start}")
        result = minimize(
            evaluate,
            start,
            method="Powell",
            bounds=bounds,
            options=settings,
        )
        results.append(result)
        if progress is not None:
            progress(f"  finished: success={result.success}, J={result.fun:.8e}")

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
        objective_evaluations=objective_evaluations,
        cache_hits=cache_hits,
        elapsed_seconds=perf_counter() - started,
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


class DynamicMixturePriceEvaluator:
    """Evaluate arbitrary price vectors with the F2-F dynamic mixture."""

    def __init__(
        self,
        support,
        degree,
        *,
        cfl=0.40,
    ):
        self.support = support
        self.degree = degree
        self.cfl = float(cfl)

        raw_basis = support.model.fokker_planc.beta_basis(
            support.x,
            self.degree,
        )
        self.degree = int(degree)  # beta_basis has validated the integer degree.
        self.basis = support.model.fokker_planc._normalise_basis_discretely(
            raw_basis,
            support.dx,
        )
        # Basis and weighted Hessian do not depend on candidate prices.
        self.projection_data = _projection_matrices(self.basis, support.dx)
        initial_density = support.initial_mass / support.dx
        (
            self.initial_coefficients,
            _,
        ) = support.model.fokker_planc.project_density(
            initial_density,
            self.basis,
            support.dx,
        )

    @property
    def n_intervals(self):
        return self.support.n_intervals

    def evaluate(self, prices):
        prices = self.support.validate_prices(prices)
        parameters = self.support.parameters_for_prices(prices)

        def drift(time, state):
            return self.support.model._drift(time, state, parameters)

        def diffusion(time, state):
            del time
            return self.support.model.diffusion_coefficient(
                state,
                self.support.sigma_x,
            )

        evolved = _evolve_projected_density(
            method=self.support.model.fokker_planc,
            t_eval=self.support.solver_t_eval,
            breakpoints=self.support.breakpoints,
            x=self.support.x,
            dx=self.support.dx,
            basis=self.basis,
            initial_coefficients=self.initial_coefficients,
            drift=drift,
            diffusion=diffusion,
            max_step=self.support.solver_max_step,
            cfl=self.cfl,
            projection_data=self.projection_data,
        )

        cell_mass = evolved["density"] * self.support.dx
        midpoint_mass = cell_mass[1:-1]
        demand_values = demand_grid(
            self.support.model,
            parameters,
            self.support.midpoints,
            self.support.x,
        )
        expected, second, _ = discrete_demand_moments(
            demand_values,
            midpoint_mass,
        )
        objective = stochastic_tracking_objective(
            self.support.widths,
            expected,
            second,
            self.support.target,
        )
        return PriceObjectiveEvaluation(
            prices=prices.copy(),
            objective=objective,
            cell_mass=cell_mass.copy(),
            diagnostics=dict(evolved["diagnostics"]),
        )

    def __call__(self, prices):
        return self.evaluate(prices).objective.total
