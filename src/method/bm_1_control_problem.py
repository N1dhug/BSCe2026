"""Reusable F3 control-problem contract for stochastic BM1 experiments.

The contract deliberately separates model/physical inputs from numerical
resolution.  A fixed external price path (for example an observed spot-price
history) may be attached to a problem, but candidate prices remain explicit
arguments to the evaluator/optimizer.
"""

from dataclasses import dataclass

import numpy as np

from src.method.bm_1_fast_fv import FastFVStochasticPriceEvaluator
from src.method.bm_1_stochastic_optimization import FVStochasticPriceEvaluator
from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder


def _positive_integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer.")
    original = value
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    try:
        same_value = float(value) == float(original)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if not same_value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _price_vector(values, n_intervals, name):
    values = np.asarray(values, dtype=float).reshape(-1)
    if (
        values.size != n_intervals
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(
            f"{name} must contain one finite value in [0, 1] per price interval."
        )
    return values.copy()


@dataclass(frozen=True)
class BM1Parameters:
    """Known BM1 parameters used by one F3 control problem.

    Parameter estimation is intentionally outside this class.  Later F4 work
    can construct the same object from estimated values without changing the
    F3 simulation/optimization interface.
    """

    state_response: object
    price_response: object
    C: float
    k: float
    Xi: float
    sigma_x: float

    def __post_init__(self):
        if not callable(self.state_response):
            raise TypeError("state_response must be callable.")
        if not callable(self.price_response):
            raise TypeError("price_response must be callable.")

        C = float(self.C)
        k = float(self.k)
        Xi = float(self.Xi)
        sigma_x = float(self.sigma_x)
        if not np.isfinite(C) or C <= 0.0:
            raise ValueError("C must be finite and strictly positive.")
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError("k must be finite and strictly positive.")
        if not np.isfinite(Xi) or not 0.0 <= Xi <= 1.0:
            raise ValueError("Xi must be finite and lie in [0, 1].")
        if not np.isfinite(sigma_x) or sigma_x < 0.0:
            raise ValueError("sigma_x must be finite and non-negative.")

        object.__setattr__(self, "C", C)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "Xi", Xi)
        object.__setattr__(self, "sigma_x", sigma_x)

    def parameter_template(self, price_edges):
        """Return [U, f, g, C, k, Xi] with a neutral placeholder U.

        Stochastic price evaluators replace the placeholder with each supplied
        candidate price vector.  Keeping the placeholder neutral makes it
        explicit that an observed price path is not silently used as a
        decision variable.
        """
        edges = np.asarray(price_edges, dtype=float).reshape(-1)
        neutral = input_builder.price_intervals(
            (edges, np.full(edges.size - 1, 0.5, dtype=float))
        )
        return [
            neutral,
            self.state_response,
            self.price_response,
            self.C,
            self.k,
            self.Xi,
        ]


@dataclass(frozen=True)
class BM1ControlNumerics:
    """Numerical resolution for one BM1 F3 calculation."""

    x_steps: int = 201
    objective_max_step: float = 1.0 / 96.0
    solver_max_step: float = 1.0 / 96.0
    model_t_steps: int = 2

    def __post_init__(self):
        x_steps = _positive_integer(self.x_steps, "x_steps", 3)
        model_t_steps = _positive_integer(self.model_t_steps, "model_t_steps", 2)
        objective_max_step = float(self.objective_max_step)
        solver_max_step = float(self.solver_max_step)

        if (
            not np.isfinite(objective_max_step)
            or objective_max_step <= 0.0
        ):
            raise ValueError(
                "objective_max_step must be finite and strictly positive."
            )
        if np.isnan(solver_max_step) or solver_max_step <= 0.0:
            raise ValueError("solver_max_step must be strictly positive.")

        object.__setattr__(self, "x_steps", x_steps)
        object.__setattr__(self, "model_t_steps", model_t_steps)
        object.__setattr__(self, "objective_max_step", objective_max_step)
        object.__setattr__(self, "solver_max_step", solver_max_step)


@dataclass(frozen=True)
class BM1ControlProblem:
    """Common F3 input contract for fixed-price simulation and optimization.

    observed_prices is optional and immutable.  It represents a known
    external price history to evaluate, not a price vector that the optimizer
    may overwrite.  Optimization code instead receives explicit candidate
    vectors through the evaluator returned by build_evaluator.

    initial_mass_builder receives FV cell edges and must return one
    normalized probability mass per cell.  This keeps the physical initial
    law independent of the chosen spatial resolution.
    """

    baseline: object
    target: object
    horizon: object
    price_edges: object
    parameters: BM1Parameters
    initial_mass_builder: object
    breakpoints: object = None
    observed_prices: object = None

    def __post_init__(self):
        if not callable(self.baseline):
            raise TypeError("baseline must be callable.")
        if not callable(self.target):
            raise TypeError("target must be callable.")
        if not isinstance(self.parameters, BM1Parameters):
            raise TypeError("parameters must be a BM1Parameters instance.")
        if not callable(self.initial_mass_builder):
            raise TypeError("initial_mass_builder must be callable.")

        horizon = np.asarray(self.horizon, dtype=float).reshape(-1)
        if (
            horizon.size != 2
            or not np.all(np.isfinite(horizon))
            or horizon[1] <= horizon[0]
        ):
            raise ValueError(
                "horizon must contain two finite values with end > start."
            )

        price_edges = np.asarray(self.price_edges, dtype=float).reshape(-1)
        if (
            price_edges.size < 2
            or not np.all(np.isfinite(price_edges))
            or np.any(np.diff(price_edges) <= 0.0)
            or price_edges[0] != horizon[0]
            or price_edges[-1] != horizon[1]
        ):
            raise ValueError(
                "price_edges must be finite, strictly increasing and span horizon."
            )

        breakpoints = np.asarray(
            [] if self.breakpoints is None else self.breakpoints,
            dtype=float,
        ).reshape(-1)
        if (
            not np.all(np.isfinite(breakpoints))
            or np.any(breakpoints <= horizon[0])
            or np.any(breakpoints >= horizon[1])
        ):
            raise ValueError(
                "breakpoints must be finite points strictly inside horizon."
            )
        breakpoints = np.unique(breakpoints)

        observed_prices = None
        if self.observed_prices is not None:
            observed_prices = _price_vector(
                self.observed_prices,
                price_edges.size - 1,
                "observed_prices",
            )

        price_edges = price_edges.copy()
        breakpoints = breakpoints.copy()
        price_edges.setflags(write=False)
        breakpoints.setflags(write=False)
        if observed_prices is not None:
            observed_prices.setflags(write=False)

        object.__setattr__(self, "horizon", tuple(float(v) for v in horizon))
        object.__setattr__(self, "price_edges", price_edges)
        object.__setattr__(self, "breakpoints", breakpoints)
        object.__setattr__(self, "observed_prices", observed_prices)

    @property
    def n_intervals(self):
        return self.price_edges.size - 1

    def build_model(self, numerics):
        if not isinstance(numerics, BM1ControlNumerics):
            raise TypeError("numerics must be a BM1ControlNumerics instance.")
        return bm_1(
            self.baseline,
            self.target,
            self.horizon,
            numerics.model_t_steps,
        )

    def initial_mass(self, model, x_steps):
        edges, _, _ = model.fokker_planc_reference.grid(
            _positive_integer(x_steps, "x_steps", 3),
            bounds=(0.0, 1.0),
        )
        mass = np.asarray(
            self.initial_mass_builder(edges.copy()),
            dtype=float,
        ).reshape(-1)
        if (
            mass.size != edges.size - 1
            or not np.all(np.isfinite(mass))
            or np.min(mass) < 0.0
            or not np.isclose(np.sum(mass), 1.0, atol=1.0e-10, rtol=0.0)
        ):
            raise ValueError(
                "initial_mass_builder must return one finite non-negative "
                "probability mass per FV cell, summing to one."
            )
        return mass.copy()

    def build_evaluator(
        self,
        numerics,
        *,
        fast=True,
        batch_size=128,
        vectorized=True,
    ):
        """Build an FV price evaluator without changing the problem inputs."""
        if not isinstance(numerics, BM1ControlNumerics):
            raise TypeError("numerics must be a BM1ControlNumerics instance.")
        model = self.build_model(numerics)
        initial_mass = self.initial_mass(model, numerics.x_steps)
        common = dict(
            model=model,
            parameter_template=self.parameters.parameter_template(self.price_edges),
            interval_edges=self.price_edges,
            sigma_x=self.parameters.sigma_x,
            initial_mass=initial_mass,
            objective_max_step=numerics.objective_max_step,
            solver_max_step=numerics.solver_max_step,
            breakpoints=self.breakpoints,
        )
        if fast:
            return FastFVStochasticPriceEvaluator(
                **common,
                batch_size=batch_size,
                vectorized=vectorized,
            )
        return FVStochasticPriceEvaluator(**common)

    def evaluate_fixed(
        self,
        prices,
        numerics,
        *,
        fast=True,
        batch_size=128,
        vectorized=True,
    ):
        """Evaluate a supplied fixed price path; no optimization is performed."""
        prices = _price_vector(prices, self.n_intervals, "prices")
        evaluator = self.build_evaluator(
            numerics,
            fast=fast,
            batch_size=batch_size,
            vectorized=vectorized,
        )
        return evaluator.evaluate(prices)

    def evaluate_observed(
        self,
        numerics,
        *,
        fast=True,
        batch_size=128,
        vectorized=True,
    ):
        """Evaluate the attached observed/external price history unchanged."""
        if self.observed_prices is None:
            raise ValueError("No observed_prices are attached to this control problem.")
        return self.evaluate_fixed(
            self.observed_prices,
            numerics,
            fast=fast,
            batch_size=batch_size,
            vectorized=vectorized,
        )
