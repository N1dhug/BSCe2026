"""BM1 price control: problem definition -> forward evaluator -> optimization.

Model parameters and initial conditions/laws are supplied, not fitted here.
Deterministic, FV and mixture-backed evaluators share one control objective;
numerical time-steppers remain in method/solver.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg.lapack import dgtsv
from scipy.optimize import minimize
from scipy.special import betainc

from src.method import beta_mixture as mixture
from src.control import objective as tracking
from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder
from src.validation import control as checks
from src.validation.model.bm_1 import bm_1_validation
from src.validation.solver.finite_volume import finite_volume_1d_validation


# Problem inputs: physical parameters, numerical resolution and fixed external prices.


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
            raise TypeError('state_response must be callable.')
        if not callable(self.price_response):
            raise TypeError('price_response must be callable.')
        C = float(self.C)
        k = float(self.k)
        Xi = float(self.Xi)
        sigma_x = float(self.sigma_x)
        if not np.isfinite(C) or C <= 0.0:
            raise ValueError('C must be finite and strictly positive.')
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError('k must be finite and strictly positive.')
        if not np.isfinite(Xi) or not 0.0 <= Xi <= 1.0:
            raise ValueError('Xi must be finite and lie in [0, 1].')
        if not np.isfinite(sigma_x) or sigma_x < 0.0:
            raise ValueError('sigma_x must be finite and non-negative.')
        object.__setattr__(self, 'C', C)
        object.__setattr__(self, 'k', k)
        object.__setattr__(self, 'Xi', Xi)
        object.__setattr__(self, 'sigma_x', sigma_x)

    def parameter_template(self, price_edges):
        """Return [U, f, g, C, k, Xi] with a neutral placeholder U.

        Stochastic price evaluators replace the placeholder with each supplied
        candidate price vector.  Keeping the placeholder neutral makes it
        explicit that an observed price path is not silently used as a
        decision variable.
        """
        edges = np.asarray(price_edges, dtype=float).reshape(-1)
        neutral = input_builder.price_intervals((edges, np.full(edges.size - 1, 0.5, dtype=float)))
        return [neutral, self.state_response, self.price_response, self.C, self.k, self.Xi]


@dataclass(frozen=True)
class BM1ControlNumerics:
    """Numerical resolution for one BM1 F3 calculation."""
    x_steps: int = 201
    objective_max_step: float = 1.0 / 96.0
    solver_max_step: float = 1.0 / 96.0
    model_t_steps: int = 2
    deterministic_rtol: float = 1e-09
    deterministic_atol: float = 1e-11

    def __post_init__(self):
        x_steps = checks.positive_integer(self.x_steps, 'x_steps', 3)
        model_t_steps = checks.positive_integer(self.model_t_steps, 'model_t_steps', 2)
        objective_max_step = float(self.objective_max_step)
        solver_max_step = float(self.solver_max_step)
        deterministic_rtol = float(self.deterministic_rtol)
        deterministic_atol = float(self.deterministic_atol)
        if not np.isfinite(objective_max_step) or objective_max_step <= 0.0:
            raise ValueError('objective_max_step must be finite and strictly positive.')
        if np.isnan(solver_max_step) or solver_max_step <= 0.0:
            raise ValueError('solver_max_step must be strictly positive.')
        if not np.isfinite(deterministic_rtol) or deterministic_rtol <= 0.0:
            raise ValueError('deterministic_rtol must be finite and strictly positive.')
        if not np.isfinite(deterministic_atol) or deterministic_atol <= 0.0:
            raise ValueError('deterministic_atol must be finite and strictly positive.')
        object.__setattr__(self, 'x_steps', x_steps)
        object.__setattr__(self, 'model_t_steps', model_t_steps)
        object.__setattr__(self, 'objective_max_step', objective_max_step)
        object.__setattr__(self, 'solver_max_step', solver_max_step)
        object.__setattr__(self, 'deterministic_rtol', deterministic_rtol)
        object.__setattr__(self, 'deterministic_atol', deterministic_atol)


@dataclass(frozen=True)
class BM1ControlProblem:
    """Common F3 input contract for fixed-price simulation and optimization.

    observed_prices is optional and immutable.  It represents a known
    external price history to evaluate, not a price vector that the optimizer
    may overwrite.  Optimization code instead receives explicit candidate
    vectors through the evaluator returned by build_evaluator.

    initial_mass_builder receives FV cell edges and must return one
    normalized probability mass per cell. This keeps the stochastic initial
    law independent of the chosen spatial resolution. deterministic_initial_state
    is an explicit point-state counterpart used only for deterministic control
    comparisons.
    """
    baseline: object
    target: object
    horizon: object
    price_edges: object
    parameters: BM1Parameters
    initial_mass_builder: object
    deterministic_initial_state: object = None
    breakpoints: object = None
    observed_prices: object = None
    terminal_reference: object = None
    terminal_weight: float = 0.0
    price_smoothing_weight: float = 0.0

    def __post_init__(self):
        if not callable(self.baseline):
            raise TypeError('baseline must be callable.')
        if not callable(self.target):
            raise TypeError('target must be callable.')
        if not isinstance(self.parameters, BM1Parameters):
            raise TypeError('parameters must be a BM1Parameters instance.')
        if not callable(self.initial_mass_builder):
            raise TypeError('initial_mass_builder must be callable.')
        horizon = np.asarray(self.horizon, dtype=float).reshape(-1)
        if horizon.size != 2 or not np.all(np.isfinite(horizon)) or horizon[1] <= horizon[0]:
            raise ValueError('horizon must contain two finite values with end > start.')
        price_edges = checks.validate_interval_edges(self.price_edges, horizon)
        breakpoints = np.asarray([] if self.breakpoints is None else self.breakpoints, dtype=float).reshape(
            -1,
        )
        if (
            not np.all(np.isfinite(breakpoints))
            or np.any(breakpoints <= horizon[0])
            or np.any(breakpoints >= horizon[1])
        ):
            raise ValueError('breakpoints must be finite points strictly inside horizon.')
        breakpoints = np.unique(breakpoints)
        deterministic_initial_state = None
        if self.deterministic_initial_state is not None:
            state = np.asarray(self.deterministic_initial_state, dtype=float)
            if state.size != 1:
                raise ValueError('deterministic_initial_state must be a scalar.')
            deterministic_initial_state = float(state.reshape(-1)[0])
            if not np.isfinite(deterministic_initial_state) or not 0.0 <= deterministic_initial_state <= 1.0:
                raise ValueError('deterministic_initial_state must be finite and lie in [0, 1].')
        observed_prices = None
        if self.observed_prices is not None:
            observed_prices = checks.validate_prices(
                self.observed_prices, price_edges.size - 1, name='observed_prices',
            ).copy()
        terminal_reference, terminal_weight, smoothing_weight = tracking.validate_control_penalty_settings(
            self.terminal_reference,
            self.terminal_weight,
            self.price_smoothing_weight,
        )
        breakpoints = breakpoints.copy()
        price_edges.setflags(write=False)
        breakpoints.setflags(write=False)
        if observed_prices is not None:
            observed_prices.setflags(write=False)
        object.__setattr__(self, 'horizon', tuple((float(v) for v in horizon)))
        object.__setattr__(self, 'price_edges', price_edges)
        object.__setattr__(self, 'breakpoints', breakpoints)
        object.__setattr__(self, 'observed_prices', observed_prices)
        object.__setattr__(self, 'deterministic_initial_state', deterministic_initial_state)
        object.__setattr__(self, 'terminal_reference', terminal_reference)
        object.__setattr__(self, 'terminal_weight', terminal_weight)
        object.__setattr__(self, 'price_smoothing_weight', smoothing_weight)

    @property
    def n_intervals(self):
        return self.price_edges.size - 1

    def build_model(self, numerics):
        """Construct BM1 with this baseline, target and time horizon."""
        if not isinstance(numerics, BM1ControlNumerics):
            raise TypeError('numerics must be a BM1ControlNumerics instance.')
        return bm_1(self.baseline, self.target, self.horizon, numerics.model_t_steps)

    def initial_mass(self, model, x_steps):
        """Discretize the same physical initial law on the requested FV grid."""
        edges, _, _ = model.fokker_planc_reference.grid(
            checks.positive_integer(x_steps, 'x_steps', 3),
            bounds=(0.0, 1.0),
        )
        mass = np.asarray(self.initial_mass_builder(edges.copy()), dtype=float).reshape(-1)
        if (
            mass.size != edges.size - 1
            or not np.all(np.isfinite(mass))
            or np.min(mass) < 0.0
            or not np.isclose(np.sum(mass), 1.0, atol=1e-10, rtol=0.0)
        ):
            raise ValueError(
                'initial_mass_builder must return one finite non-negative probability mass per FV cell, summing to one.',
            )
        return mass.copy()

    def build_deterministic_evaluator(self, numerics):
        """Build the deterministic counterpart of the same F3 control problem."""
        if not isinstance(numerics, BM1ControlNumerics):
            raise TypeError('numerics must be a BM1ControlNumerics instance.')
        if self.deterministic_initial_state is None:
            raise ValueError('deterministic_initial_state is required for deterministic control evaluation.')
        return DeterministicPriceEvaluator(
            model=self.build_model(numerics),
            parameter_template=self.parameters.parameter_template(self.price_edges),
            interval_edges=self.price_edges,
            initial_state=self.deterministic_initial_state,
            objective_max_step=numerics.objective_max_step,
            solver_max_step=numerics.solver_max_step,
            breakpoints=self.breakpoints,
            terminal_reference=self.terminal_reference,
            terminal_weight=self.terminal_weight,
            price_smoothing_weight=self.price_smoothing_weight,
            rtol=numerics.deterministic_rtol,
            atol=numerics.deterministic_atol,
        )

    def build_evaluator(self, numerics, *, fast=True, batch_size=128, vectorized=True):
        """Build an FV price evaluator without changing the problem inputs."""
        if not isinstance(numerics, BM1ControlNumerics):
            raise TypeError('numerics must be a BM1ControlNumerics instance.')
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
            terminal_reference=self.terminal_reference,
            terminal_weight=self.terminal_weight,
            price_smoothing_weight=self.price_smoothing_weight,
        )
        if fast:
            return FastFVStochasticPriceEvaluator(**common, batch_size=batch_size, vectorized=vectorized)
        return FVStochasticPriceEvaluator(**common)

    def evaluate_fixed(self, prices, numerics, *, fast=True, batch_size=128, vectorized=True):
        """Evaluate a supplied fixed price path; no optimization is performed."""
        prices = checks.validate_prices(prices, self.n_intervals).copy()
        evaluator = self.build_evaluator(numerics, fast=fast, batch_size=batch_size, vectorized=vectorized)
        return evaluator.evaluate(prices)

    def evaluate_observed(self, numerics, *, fast=True, batch_size=128, vectorized=True):
        """Evaluate the attached observed/external price history unchanged."""
        if self.observed_prices is None:
            raise ValueError('No observed_prices are attached to this control problem.')
        return self.evaluate_fixed(
            self.observed_prices,
            numerics,
            fast=fast,
            batch_size=batch_size,
            vectorized=vectorized,
        )


# Results returned by forward evaluations and price searches.


@dataclass(frozen=True)
class DeterministicPriceEvaluation:
    """Deterministic F3 cost and state trajectory for one held price plan."""
    prices: np.ndarray
    objective: tracking.StochasticTrackingResult
    state: np.ndarray
    demand: np.ndarray
    penalties: tracking.ControlPenaltyResult

    @property
    def total(self):
        return self.objective.total + self.penalties.total


@dataclass(frozen=True)
class PriceObjectiveEvaluation:
    """Tracking cost, F3-C penalties, density history and diagnostics for one price plan."""
    prices: np.ndarray
    objective: tracking.StochasticTrackingResult
    cell_mass: np.ndarray
    diagnostics: dict
    penalties: tracking.ControlPenaltyResult

    @property
    def total(self):
        return self.objective.total + self.penalties.total


@dataclass(frozen=True)
class FVCostEvaluation:
    """Cost-only evaluation: no full density history is retained."""
    objective: tracking.StochasticTrackingResult
    diagnostics: dict
    penalties: tracking.ControlPenaltyResult

    @property
    def total(self):
        return self.objective.total + self.penalties.total


@dataclass(frozen=True)
class MonteCarloCandidateResult:
    """MC tracking result plus pathwise F3-C penalties for one price candidate."""
    prices: np.ndarray
    objective: object
    diagnostics: dict
    penalties: tracking.ControlPenaltyResult
    path_objectives: np.ndarray
    standard_error: float

    @property
    def total(self):
        return self.objective.objective.total + self.penalties.total


@dataclass(frozen=True)
class MultistartOptimizationResult:
    """Collection of bounded Powell solves and the best successful result."""
    best: object
    results: tuple
    starts: np.ndarray
    objective_evaluations: int = 0
    cache_hits: int = 0
    elapsed_seconds: float = 0.0


# Initial law: the same Beta distribution for FV masses and sampled paths.


def beta_initial_cell_mass(edges, alpha=20.0, beta=20.0):
    """Return exact Beta(alpha,beta) probability in each supplied cell."""
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError('edges must be a one-dimensional array with at least two values.')
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError('edges must be finite and strictly increasing.')
    if edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError('Beta initial-law edges must span [0, 1].')
    alpha = float(alpha)
    beta = float(beta)
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0.0 or (beta <= 0.0):
        raise ValueError('Beta shape parameters must be finite and strictly positive.')
    cdf = betainc(alpha, beta, edges)
    mass = np.diff(cdf)
    if np.any(mass < 0.0):
        raise RuntimeError('Beta CDF produced negative cell probability.')
    if not np.isclose(np.sum(mass), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError('Beta cell probabilities do not sum to one.')
    return mass


def sample_beta_initial_states(n_paths, seed, alpha=20.0, beta=20.0):
    """Draw X0 from Beta(alpha, beta); use the same law as the FV cell masses."""
    n_paths = checks.positive_integer(n_paths, 'n_paths', 1)
    alpha, beta = (float(alpha), float(beta))
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0.0 or (beta <= 0.0):
        raise ValueError('Beta shape parameters must be finite and strictly positive.')
    return np.random.default_rng(seed).beta(alpha, beta, size=n_paths)


# Forward evaluators: deterministic ODE, independent/fast FV and projected mixture.


class _PriceEvaluatorBase:
    """Common fixed inputs for deterministic, FV and mixture-backed price evaluation."""

    def __init__(
        self,
        model,
        parameter_template,
        interval_edges,
        *,
        objective_max_step,
        solver_max_step,
        breakpoints=None,
        terminal_reference=None,
        terminal_weight=0.0,
        price_smoothing_weight=0.0,
    ):
        self.model = model
        self.parameter_template = list(parameter_template)
        if len(self.parameter_template) != 6:
            raise ValueError('parameter_template must contain [U, f, g, C, k, Xi].')
        self.interval_edges = checks.validate_interval_edges(interval_edges, model.T)
        extra_breakpoints = np.asarray([] if breakpoints is None else breakpoints, dtype=float).reshape(-1)
        if (
            not np.all(np.isfinite(extra_breakpoints))
            or np.any(extra_breakpoints <= model.T[0])
            or np.any(extra_breakpoints >= model.T[1])
        ):
            raise ValueError('breakpoints must be finite points inside the model interval.')
        self.breakpoints = np.unique(np.concatenate((self.interval_edges[1:-1], extra_breakpoints)))
        self.n_intervals = self.interval_edges.size - 1
        self.objective_edges, self.midpoints, self.widths, self.solver_t_eval = tracking.midpoint_quadrature(
            model.T,
            breakpoints=self.breakpoints,
            max_step=objective_max_step,
        )
        self.solver_max_step = float(solver_max_step)
        if np.isnan(self.solver_max_step) or self.solver_max_step <= 0.0:
            raise ValueError('solver_max_step must be strictly positive.')
        self.terminal_reference, self.terminal_weight, self.price_smoothing_weight = (
            tracking.validate_control_penalty_settings(
                terminal_reference,
                terminal_weight,
                price_smoothing_weight,
            )
        )
        self.target = np.asarray(model.A(self.midpoints), dtype=float).reshape(-1)
        if self.target.size != self.midpoints.size or not np.all(np.isfinite(self.target)):
            raise ValueError('model target A(t) must return finite midpoint values.')

    def validate_prices(self, prices):
        return checks.validate_prices(prices, self.n_intervals)

    def parameters_for_prices(self, prices):
        """Validate a public candidate and construct its held-price input."""
        return self._parameters_for_validated_prices(self.validate_prices(prices))

    def _parameters_for_validated_prices(self, prices):
        price = input_builder.price_intervals((self.interval_edges, prices))
        parameters = list(self.parameter_template)
        parameters[0] = price
        return parameters


class DeterministicPriceEvaluator(_PriceEvaluatorBase):
    """Evaluate the deterministic counterpart of an F3 BM1 price-control problem."""

    def __init__(
        self,
        model,
        parameter_template,
        interval_edges,
        initial_state,
        *,
        objective_max_step,
        solver_max_step,
        breakpoints=None,
        terminal_reference=None,
        terminal_weight=0.0,
        price_smoothing_weight=0.0,
        rtol=1e-09,
        atol=1e-11,
    ):
        super().__init__(
            model,
            parameter_template,
            interval_edges,
            objective_max_step=objective_max_step,
            solver_max_step=solver_max_step,
            breakpoints=breakpoints,
            terminal_reference=terminal_reference,
            terminal_weight=terminal_weight,
            price_smoothing_weight=price_smoothing_weight,
        )
        state = np.asarray(initial_state, dtype=float)
        if state.size != 1:
            raise ValueError('initial_state must be a scalar.')
        self.initial_state = float(state.reshape(-1)[0])
        if not np.isfinite(self.initial_state) or not 0.0 <= self.initial_state <= 1.0:
            raise ValueError('initial_state must be finite and lie in [0, 1].')
        self.rtol = float(rtol)
        self.atol = float(atol)
        if not np.isfinite(self.rtol) or self.rtol <= 0.0 or not np.isfinite(self.atol) or self.atol <= 0.0:
            raise ValueError('rtol and atol must be finite and strictly positive.')

    def evaluate(self, prices):
        """Solve the ODE once and evaluate the same F3 tracking/penalty definition."""
        prices = self.validate_prices(prices)
        parameters = self._parameters_for_validated_prices(prices)
        result = self.model.deterministic_solver(
            [self.initial_state],
            parameters,
            t_eval=self.solver_t_eval,
            breakpoints=self.breakpoints,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.solver_max_step,
        )
        state = np.asarray(result[1], dtype=float)
        demand = np.asarray(result[7], dtype=float)
        midpoint_demand = demand[1:-1]
        objective = tracking.stochastic_tracking_objective(
            self.widths,
            midpoint_demand,
            np.square(midpoint_demand),
            self.target,
        )
        penalties = tracking.control_penalties(
            np.array([state[-1]], dtype=float),
            np.array([1.0], dtype=float),
            prices,
            terminal_reference=self.terminal_reference,
            terminal_weight=self.terminal_weight,
            smoothing_weight=self.price_smoothing_weight,
        )
        return DeterministicPriceEvaluation(
            prices=prices.copy(),
            objective=objective,
            state=state.copy(),
            demand=demand.copy(),
            penalties=penalties,
        )

    def __call__(self, prices):
        return self.evaluate(prices).total


class FVStochasticPriceEvaluator(_PriceEvaluatorBase):
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
        terminal_reference=None,
        terminal_weight=0.0,
        price_smoothing_weight=0.0,
    ):
        """Validate stochastic inputs after the common price-control setup."""
        super().__init__(
            model,
            parameter_template,
            interval_edges,
            objective_max_step=objective_max_step,
            solver_max_step=solver_max_step,
            breakpoints=breakpoints,
            terminal_reference=terminal_reference,
            terminal_weight=terminal_weight,
            price_smoothing_weight=price_smoothing_weight,
        )
        self.sigma_x = float(sigma_x)
        if not np.isfinite(self.sigma_x) or self.sigma_x < 0.0:
            raise ValueError('sigma_x must be finite and non-negative.')
        self.initial_mass = np.asarray(initial_mass, dtype=float).reshape(-1).copy()
        if (
            self.initial_mass.size < 3
            or not np.all(np.isfinite(self.initial_mass))
            or np.min(self.initial_mass) < 0.0
            or not np.isclose(np.sum(self.initial_mass), 1.0, atol=1e-10, rtol=0.0)
        ):
            raise ValueError('initial_mass must be a finite non-negative probability vector.')
        self.edges, self.x, self.dx = model.fokker_planc_reference.grid(
            self.initial_mass.size,
            bounds=(0.0, 1.0),
        )

    def _penalties(self, prices, terminal_mass):
        return tracking.control_penalties(
            self.x,
            terminal_mass,
            prices,
            terminal_reference=self.terminal_reference,
            terminal_weight=self.terminal_weight,
            smoothing_weight=self.price_smoothing_weight,
        )


    def evaluate(self, prices):
        """Propagate a fixed candidate and compute its bias/variance tracking contributions."""
        prices = self.validate_prices(prices)
        parameters = self._parameters_for_validated_prices(prices)
        result = self.model.fokker_planc_reference_solver(
            self.initial_mass,
            parameters,
            self.sigma_x,
            t_eval=self.solver_t_eval,
            breakpoints=self.breakpoints,
            max_step=self.solver_max_step,
        )
        t, x, _, cell_mass, _, _, _, _, _, diagnostics = result
        midpoint_mass = cell_mass[1:-1]
        demand_values = tracking.demand_grid(self.model, parameters, self.midpoints, self.x)
        expected, second, _ = tracking.discrete_demand_moments(demand_values, midpoint_mass)
        # Integrate expected tracking cost from the two demand moments.
        objective = tracking.stochastic_tracking_objective(self.widths, expected, second, self.target)
        penalties = self._penalties(prices, cell_mass[-1])
        return PriceObjectiveEvaluation(
            prices=prices.copy(),
            objective=objective,
            cell_mass=cell_mass.copy(),
            diagnostics=dict(diagnostics),
            penalties=penalties,
        )

    def __call__(self, prices):
        """Scalar objective interface for a numerical optimizer."""
        return self.evaluate(prices).total


class FastFVStochasticPriceEvaluator(FVStochasticPriceEvaluator):
    """Drop-in opt-in evaluator; construct a new instance if grids change.

    The mathematical model/input callbacks and the discretization are fixed
    for the life of an evaluator, just as its target quadrature values are.
    Candidate prices vary between calls. Coefficients are *not* cached across
    calls and are evaluated at every original left time (including varying B).
    """

    def __init__(self, *args, batch_size=128, vectorized=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = finite_volume_1d_validation.integer(batch_size, name='batch_size', minimum=1)
        if not isinstance(vectorized, (bool, np.bool_)):
            raise TypeError('vectorized must be a boolean.')
        self.vectorized = bool(vectorized)
        self.integration_t, self.output_indices = self.model.fokker_planc_reference.build_time_grid(
            self.solver_t_eval,
            self.breakpoints,
            self.solver_max_step,
        )
        self._steps = np.diff(self.integration_t)
        self._output_at = np.full(self.integration_t.size, -1, dtype=int)
        self._output_at[self.output_indices] = np.arange(self.output_indices.size)

    def _drift_block(self, times, parameters):
        faces = self.edges[1:-1]
        if self.vectorized:
            values = self.model._drift(times[:, None], faces[None, :], parameters)
            return finite_volume_1d_validation.field(values, (times.size, faces.size), 'drift')
        return np.asarray(
            [
                finite_volume_1d_validation.field(
                    self.model._drift(float(t), faces, parameters),
                    faces.shape,
                    'drift',
                )
                 for t in times
            ],
        )

    def _demand_block(self, times, parameters):
        u, f, g, _, k, xi = parameters
        if self.vectorized:
            values = self.model.eval(times[:, None], self.x[None, :], u, f, g, k, xi)[-1]
            return finite_volume_1d_validation.field(values, (times.size, self.x.size), 'demand')
        return np.asarray(
            [
                finite_volume_1d_validation.field(
                    self.model.eval(float(t), self.x, u, f, g, k, xi)[-1],
                    self.x.shape,
                    'demand',
                )
                 for t in times
            ],
        )

    def _run(self, prices, keep_history):
        """Execute the same FV scheme using tridiagonal LAPACK solves.

        keep_history=False stores only the running cost and diagnostics. Mass
        conservation, nonnegativity and solver failure checks remain active.
        """
        prices = self.validate_prices(prices)
        _, parameters, sigma = bm_1_validation.stochastic_inputs(
            [0.5],
            self._parameters_for_validated_prices(prices),
            self.sigma_x,
        )
        # Fixed state grid and diffusion; candidate prices affect the drift.
        mass = finite_volume_1d_validation.initial_mass(self.initial_mass)
        diffusion = finite_volume_1d_validation.field(
            self.model.diffusion_coefficient(self.x, sigma),
            self.x.shape,
            'diffusion',
        )
        diffusion_term = 0.5 * np.square(diffusion)
        diffusion_rates = diffusion_term / (self.dx * self.dx)
        if not np.all(np.isfinite(diffusion_rates)):
            raise ValueError('Diffusion rates must be finite.')
        n_output = self.solver_t_eval.size
        history = np.empty((n_output, mass.size)) if keep_history else None
        if keep_history:
            history[0] = mass
        expected = np.full(self.midpoints.size, np.nan)
        second = np.full_like(expected, np.nan)
        min_mass = float(np.min(mass))
        max_mass_error = abs(float(np.sum(mass)) - 1.0)
        max_column_sum = 0.0
        min_offdiag = np.inf
        # Assemble conservative tridiagonal generators in vectorized blocks.
        for start in range(0, self._steps.size, self.batch_size):
            stop = min(start + self.batch_size, self._steps.size)
            face_drift = self._drift_block(self.integration_t[start:stop], parameters)
            right = np.zeros((stop - start, mass.size))
            left = np.zeros_like(right)
            right[:, :-1] = np.maximum(face_drift, 0.0) / self.dx + diffusion_rates[:-1]
            left[:, 1:] = np.maximum(-face_drift, 0.0) / self.dx + diffusion_rates[1:]
            diagonal = -(left + right)
            lower = right[:, :-1]
            upper = left[:, 1:]
            column_sum = diagonal.copy()
            column_sum[:, :-1] += lower
            column_sum[:, 1:] += upper
            max_column_sum = max(max_column_sum, float(np.max(np.abs(column_sum))))
            min_offdiag = min(min_offdiag, float(np.min(lower)), float(np.min(upper)))
            if not np.all(np.isfinite(diagonal)):
                raise ValueError('Generator rates must be finite.')
            output_ids = self._output_at[start + 1:stop + 1]
            moment_ids = output_ids[(output_ids > 0) & (output_ids < n_output - 1)] - 1
            demand = self._demand_block(self.midpoints[moment_ids], parameters) if moment_ids.size else None
            demand_row = 0
            # Backward Euler: (I - h Q) m_new = m_old. Never clip failed masses.
            for local, step in enumerate(range(start, stop)):
                h = self._steps[step]
                dl = -h * lower[local]
                d = 1.0 - h * diagonal[local]
                du = -h * upper[local]
                if not (np.all(np.isfinite(dl)) and np.all(np.isfinite(d)) and np.all(np.isfinite(du))):
                    raise ValueError('Implicit step matrix must be finite.')
                _, _, _, solved, info = dgtsv(
                    dl,
                    d,
                    du,
                    mass[:, None],
                    overwrite_dl=1,
                    overwrite_d=1,
                    overwrite_du=1,
                    overwrite_b=1,
                )
                if info != 0:
                    raise RuntimeError(f'LAPACK tridiagonal solve failed (info={info}).')
                mass = finite_volume_1d_validation.evolved_mass(solved[:, 0])
                min_mass = min(min_mass, float(np.min(mass)))
                max_mass_error = max(max_mass_error, abs(float(np.sum(mass)) - 1.0))
                if max_mass_error > 1e-09:
                    raise RuntimeError('FV probability mass drift exceeds tolerance; no repair applied.')
                index = output_ids[local]
                if index >= 0:
                    if keep_history:
                        history[index] = mass
                    if 0 < index < n_output - 1:
                        values = demand[demand_row]
                        expected[index - 1] = np.sum(values * mass)
                        second[index - 1] = np.sum(np.square(values) * mass)
                        demand_row += 1
        # Integrate expected tracking cost, then add the F3-C terminal/price terms.
        objective = tracking.stochastic_tracking_objective(self.widths, expected, second, self.target)
        penalties = self._penalties(prices, mass)
        diagnostics = {
            'integration_t': self.integration_t.copy(),
            'output_indices': self.output_indices.copy(),
            'n_steps': self._steps.size,
            'maximum_abs_generator_column_sum': max_column_sum,
            'minimum_off_diagonal_rate': min_offdiag,
            'minimum_cell_mass': min_mass,
            'maximum_abs_mass_error': max_mass_error,
            'used_clipping': False,
            'used_renormalization': False,
            'backend': 'lapack-blocked-fv',
            'batch_size': self.batch_size,
            'saved_density_history': bool(keep_history),
        }
        if keep_history:
            return PriceObjectiveEvaluation(prices.copy(), objective, history, diagnostics, penalties)
        return FVCostEvaluation(objective, diagnostics, penalties)

    def evaluate(self, prices):
        """Return full state history for plots/reference comparisons."""
        return self._run(prices, keep_history=True)

    def evaluate_cost(self, prices):
        """Return objective and diagnostics without saving a density history."""
        return self._run(prices, keep_history=False)

    def __call__(self, prices):
        """Return full F3 control cost without storing a full trajectory."""
        return self.evaluate_cost(prices).total


class DynamicMixturePriceEvaluator:
    """Evaluate arbitrary price vectors with the F2-F dynamic mixture."""

    def __init__(self, support, degree, *, cfl=0.4):
        self.support = support
        self.degree = degree
        self.cfl = float(cfl)
        raw_basis = support.model.fokker_planc.beta_basis(support.x, self.degree)
        self.degree = int(degree)
        self.basis = support.model.fokker_planc._normalise_basis_discretely(raw_basis, support.dx)
        self.projection_data = mixture._projection_matrices(self.basis, support.dx)
        initial_density = support.initial_mass / support.dx
        self.initial_coefficients, _ = support.model.fokker_planc.project_density(
            initial_density,
            self.basis,
            support.dx,
        )

    @property
    def n_intervals(self):
        return self.support.n_intervals

    def evaluate(self, prices):
        """Propagate the mixture approximation, then use the same demand-moment objective."""
        prices = self.support.validate_prices(prices)
        parameters = self.support._parameters_for_validated_prices(prices)

        def drift(time, state):
            return self.support.model._drift(time, state, parameters)

        def diffusion(time, state):
            del time
            return self.support.model.diffusion_coefficient(state, self.support.sigma_x)
        evolved = mixture.evolve_projected_density(
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
        cell_mass = evolved['density'] * self.support.dx
        midpoint_mass = cell_mass[1:-1]
        demand_values = tracking.demand_grid(
            self.support.model,
            parameters,
            self.support.midpoints,
            self.support.x,
        )
        expected, second, _ = tracking.discrete_demand_moments(demand_values, midpoint_mass)
        objective = tracking.stochastic_tracking_objective(
            self.support.widths,
            expected,
            second,
            self.support.target,
        )
        penalties = self.support._penalties(prices, cell_mass[-1])
        return PriceObjectiveEvaluation(
            prices=prices.copy(),
            objective=objective,
            cell_mass=cell_mass.copy(),
            diagnostics=dict(evolved['diagnostics']),
            penalties=penalties,
        )

    def __call__(self, prices):
        return self.evaluate(prices).total


# Bounded price search and common-random-number assessment.


def multistart_powell(objective, starts, *, lower=0.0, upper=1.0, options=None, progress=None, cache=True):
    """Minimize a deterministic price objective from several feasible starts.

    Exact duplicate price vectors may reuse scalar values within this call.
    No prices are rounded. Set cache=False for a stateful/non-deterministic
    objective (such objectives are not the intended reference use). Progress
    is an optional callback and never changes stopping tolerances.
    """
    starts = np.asarray(starts, dtype=float)
    if starts.ndim == 1:
        starts = starts[None, :]
    if starts.ndim != 2 or starts.shape[0] < 1 or starts.shape[1] < 1 or (not np.all(np.isfinite(starts))):
        raise ValueError('starts must be a finite non-empty 2D array.')
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError('optimization bounds must be finite with lower < upper.')
    if np.any(starts < lower) or np.any(starts > upper):
        raise ValueError('every optimization start must lie inside the bounds.')
    unique = []
    for start in starts:
        if not any((np.array_equal(start, previous) for previous in unique)):
            unique.append(start.copy())
    starts = np.asarray(unique, dtype=float)
    settings = {'xtol': 1e-05, 'ftol': 1e-09, 'maxiter': 100}
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
        key = tuple((float(value) for value in candidate))
        if cache and key in memo:
            cache_hits += 1
            return memo[key]
        value = float(objective(candidate))
        if not np.isfinite(value):
            raise RuntimeError('The deterministic price objective returned a non-finite value.')
        objective_evaluations += 1
        if cache:
            memo[key] = value
        if progress is not None and objective_evaluations % 50 == 0:
            progress(
                f'  {objective_evaluations} objective evaluations; {perf_counter() - started:.1f} s elapsed',
            )
        return value
    for index, start in enumerate(starts, start=1):
        if progress is not None:
            progress(f'Powell start {index}/{len(starts)}: {start}')
        result = minimize(evaluate, start, method='Powell', bounds=bounds, options=settings)
        results.append(result)
        if progress is not None:
            progress(f'  finished: success={result.success}, J={result.fun:.8e}')
    successful = [result for result in results if bool(result.success) and np.isfinite(result.fun)]
    if not successful:
        messages = '; '.join((str(result.message) for result in results))
        raise RuntimeError('Every bounded Powell start failed. Messages: ' + messages)
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
        raise ValueError('coarse_price_surface currently requires exactly two intervals.')
    values = np.asarray(values, dtype=float).reshape(-1)
    if (
        values.size < 2
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError('values must be strictly increasing and lie in [0, 1].')
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
    if objective.shape != (values.size, values.size) or not np.all(np.isfinite(objective)) or n_best < 1:
        raise ValueError('invalid coarse surface or n_best.')
    order = np.argsort(objective, axis=None)
    starts = []
    for flat_index in order[:min(n_best, order.size)]:
        i, j = np.unravel_index(flat_index, objective.shape)
        starts.append([values[i], values[j]])
    return np.asarray(starts, dtype=float)


def evaluate_mc_candidates_common_random_numbers(
    evaluator,
    candidates,
    *,
    n_paths,
    initial_seed,
    brownian_seed,
    initial_sampler=sample_beta_initial_states,
):
    """Evaluate prices with shared initial states and Wiener increments.

    The default initial law is Beta(20,20). Supply initial_sampler for a different
    physical initial law; it must match the law used by the FV evaluator.
    """
    model = evaluator.model
    initial_states = initial_sampler(n_paths, initial_seed)
    integration_t, _ = model.euler_maruyama.build_time_grid(
        evaluator.solver_t_eval,
        breakpoints=evaluator.breakpoints,
        max_step=evaluator.solver_max_step,
    )
    dW = model.euler_maruyama.brownian_increments(integration_t, n_paths=n_paths, seed=brownian_seed)
    results = {}
    for name, prices in candidates.items():
        parameters = evaluator.parameters_for_prices(prices)
        output = model.euler_maruyama_solver(
            initial_states,
            parameters,
            evaluator.sigma_x,
            n_paths=n_paths,
            boundary='none',
            t_eval=evaluator.solver_t_eval,
            breakpoints=evaluator.breakpoints,
            max_step=evaluator.solver_max_step,
            brownian_increments=dW,
            return_diagnostics=True,
        )
        _, states, _, _, _, _, _, demand, diagnostics = output
        if diagnostics['total_violations']:
            raise RuntimeError(f'MC validation for {name} produced raw boundary violations.')
        states = np.asarray(states, dtype=float)
        midpoint_demand = np.asarray(demand, dtype=float)[:, 1:-1]
        objective = tracking.monte_carlo_tracking_objective(
            evaluator.widths,
            midpoint_demand,
            evaluator.target,
        )
        prices = np.asarray(prices, dtype=float)
        terminal_states = states[:, -1]
        path_mass = np.full(terminal_states.size, 1.0 / terminal_states.size, dtype=float)
        penalties = tracking.control_penalties(
            terminal_states,
            path_mass,
            prices,
            terminal_reference=evaluator.terminal_reference,
            terminal_weight=evaluator.terminal_weight,
            smoothing_weight=evaluator.price_smoothing_weight,
        )
        path_objectives = objective.path_objectives.copy()
        if evaluator.terminal_reference is not None:
            path_objectives += evaluator.terminal_weight * np.square(
                terminal_states - evaluator.terminal_reference,
            )
        path_objectives += penalties.smoothing
        standard_error = float(np.std(path_objectives, ddof=1) / np.sqrt(path_objectives.size))
        results[name] = MonteCarloCandidateResult(
            prices=prices.copy(),
            objective=objective,
            diagnostics=dict(diagnostics),
            penalties=penalties,
            path_objectives=path_objectives,
            standard_error=standard_error,
        )
    return results


def paired_mc_improvement(reference, candidate):
    """Return mean and SE of paired pathwise full-control cost differences."""
    difference = np.asarray(reference.path_objectives, dtype=float) - np.asarray(
        candidate.path_objectives,
        dtype=float,
    )
    return (float(np.mean(difference)), float(np.std(difference, ddof=1) / np.sqrt(difference.size)))
