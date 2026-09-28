"""Controlled price experiments (not fitting): two-price, multi-price and hourly targets.

    Settings define the physical benchmark. Builders create targets/evaluators;
    run_* functions optimize and reevaluate. Reporting and plotting belong to exe.
    """

import platform
from dataclasses import dataclass, replace
from statistics import median
from time import perf_counter

import numpy as np
import scipy
from scipy.interpolate import PchipInterpolator

from src.control import bm_1 as control
from src.control.objective import monte_carlo_tracking_objective
from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder
from src.validation.bm_1.cases import (
    INITIAL_ALPHA,
    INITIAL_BETA,
    SIGMA_X,
    linear_price_response,
    linear_state_response,
)
from src.validation.bm_1.mixture import DEFAULT_CFL


T = (0.0, 1.0)
INTERVAL_EDGES = np.array([0.0, 0.5, 1.0], dtype=float)
KNOWN_DETERMINISTIC_PRICES = np.array([0.72, 0.28], dtype=float)
INITIAL_PRICES = np.array([0.55, 0.45], dtype=float)
BASELINE = 0.3
X0_DETERMINISTIC = 0.5
F3_C_TERMINAL_REFERENCE = 0.5
F3_C_TERMINAL_WEIGHTS = (0.0, 1e-3, 1e-2, 1e-1, 0.5)
F3_C_SMOOTHING_WEIGHTS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
F3_D_LIGHT_TERMINAL_WEIGHT = 0.03
F3_D_LIGHT_SMOOTHING_WEIGHT = 0.003
F3_D_MC_PATHS = 20000
F3_D_MC_INITIAL_SEED = 2071
F3_D_MC_BROWNIAN_SEED = 2072
F3_E_SCENARIO_ORDER = ('deterministic', 'initial_only', 'process_only', 'full')
F3_F_MC_PATHS = 20000
F3_F_MC_INITIAL_SEED = 2091
F3_F_MC_BROWNIAN_SEED = 2092
C = 1.0
K = 2.0
XI = 1.0
FV_X_STEPS = 201
MIXTURE_OPT_X_STEPS = 81
FINAL_MIXTURE_X_STEPS = 201
MIXTURE_OPT_DEGREE = 30
FINAL_MIXTURE_DEGREES = (30, 38)
OBJECTIVE_MAX_STEP = 1.0 / 96.0
SOLVER_MAX_STEP = 1.0 / 96.0
TARGET_T_STEPS = 1537
COARSE_VALUES = np.linspace(0.0, 1.0, 5)
N_COARSE_STARTS = 4
MC_PATHS = 50000
MC_INITIAL_SEED = 2033
MC_BROWNIAN_SEED = 2034
INTERVAL_COUNTS = (2, 4, 6, 12, 24)
REFERENCE_PRICES_24 = np.array(
    [
        0.78,
        0.74,
        0.68,
        0.58,
        0.46,
        0.34,
        0.24,
        0.3,
        0.44,
        0.62,
        0.78,
        0.84,
        0.74,
        0.6,
        0.46,
        0.32,
        0.24,
        0.3,
        0.46,
        0.66,
        0.82,
        0.72,
        0.56,
        0.42,
    ],
    dtype=float,
)
COMPLEX_OBJECTIVE_MAX_STEP = 1.0 / 96.0
COMPLEX_SOLVER_MAX_STEP = 1.0 / 96.0
COMPLEX_MC_PATHS = 20000
COMPLEX_MC_INITIAL_SEED = 2051
COMPLEX_MC_BROWNIAN_SEED = 2052
MULTI_INTERVAL_COUNTS = (2, 4, 6, 24)
MULTI_OBJECTIVE_MAX_STEP = 1.0 / 96.0
MULTI_SOLVER_MAX_STEP = 1.0 / 96.0
MULTI_MC_PATHS = 20000
MULTI_MC_INITIAL_SEED = 2041
MULTI_MC_BROWNIAN_SEED = 2042
DETERMINISTIC_REFERENCE_PRICES = np.array([0.72, 0.28], dtype=float)
NEUTRAL_PRICES = np.array([0.5, 0.5], dtype=float)
HOURLY_EDGES = np.linspace(T[0], T[1], 25, dtype=float)


def _baseline():
    """Constant normalized baseline shared by the controlled price experiments."""
    return input_builder.energy((np.array(T), np.array([BASELINE, BASELINE])))


def _parameter_template():
    """Parameter order [U,f,g,C,k,Xi] for the two-interval benchmark."""
    placeholder = input_builder.price_intervals((INTERVAL_EDGES, INITIAL_PRICES))
    return [placeholder, linear_state_response, linear_price_response, C, K, XI]


def _parameters_for_prices(prices):
    parameters = _parameter_template()
    parameters[0] = input_builder.price_intervals((INTERVAL_EDGES, np.asarray(prices, dtype=float)))
    return parameters


def build_deterministic_target():
    """Return the F1-D target demand as a breakpoint-aware callable."""
    baseline = _baseline()
    generator = bm_1(baseline, baseline, T, TARGET_T_STEPS)
    target_t = np.linspace(T[0], T[1], TARGET_T_STEPS)
    result = generator.deterministic_solver(
        [X0_DETERMINISTIC],
        _parameters_for_prices(KNOWN_DETERMINISTIC_PRICES),
        t_eval=target_t,
        breakpoints=INTERVAL_EDGES[1:-1],
        rtol=1e-10,
        atol=1e-12,
        max_step=1.0 / (TARGET_T_STEPS - 1),
    )
    target_demand = np.asarray(result[7], dtype=float)
    breakpoint = INTERVAL_EDGES[1]
    left_mask = target_t < breakpoint
    right_mask = target_t >= breakpoint
    left = PchipInterpolator(target_t[left_mask], target_demand[left_mask], extrapolate=True)
    right = PchipInterpolator(target_t[right_mask], target_demand[right_mask], extrapolate=True)

    def target(query):
        array = np.asarray(query, dtype=float)
        values = np.where(array < breakpoint, left(array), right(array))
        return float(values) if array.ndim == 0 else values
    return (target, target_t, target_demand)


def build_stochastic_model(target, t_steps=2):
    baseline = _baseline()
    return bm_1(baseline, target, T, t_steps)


def _initial_mass(model, x_steps):
    """Exact Beta(20,20) probabilities on the evaluator state grid."""
    edges, _, _ = model.fokker_planc_reference.grid(x_steps)
    return control.beta_initial_cell_mass(edges, alpha=INITIAL_ALPHA, beta=INITIAL_BETA)


def _build_fv_evaluator(model, *, x_steps, objective_max_step, solver_max_step):
    initial_mass = _initial_mass(model, x_steps)
    return control.FVStochasticPriceEvaluator(
        model,
        _parameter_template(),
        INTERVAL_EDGES,
        SIGMA_X,
        initial_mass,
        objective_max_step=objective_max_step,
        solver_max_step=solver_max_step,
    )


@dataclass(frozen=True)
class PriceCase:
    """One interval count, its optimized candidate and the common-reference evaluation."""
    n_intervals: int
    edges: np.ndarray
    evaluator: control.FVStochasticPriceEvaluator
    optimization: object
    evaluation: object


def _edges(n_intervals):
    n_intervals = int(n_intervals)
    if n_intervals < 1 or 24 % n_intervals != 0:
        raise ValueError('n_intervals must be a positive divisor of 24.')
    return np.linspace(T[0], T[1], n_intervals + 1, dtype=float)


def complex_parameter_template(edges):
    placeholder = input_builder.price_intervals((edges, np.full(edges.size - 1, 0.5, dtype=float)))
    return [placeholder, linear_state_response, linear_price_response, C, K, XI]


def complex_parameters_for_prices(edges, prices):
    parameters = complex_parameter_template(edges)
    parameters[0] = input_builder.price_intervals((edges, np.asarray(prices, dtype=float)))
    return parameters


def _coarsen_reference_prices(n_intervals):
    """Average the 24 hourly reference prices inside each candidate interval."""
    n_intervals = int(n_intervals)
    if n_intervals < 1 or 24 % n_intervals != 0:
        raise ValueError('n_intervals must divide 24.')
    block = 24 // n_intervals
    return REFERENCE_PRICES_24.reshape(n_intervals, block).mean(axis=1)


def _sample_piecewise_prices(old_edges, old_prices, new_edges):
    """Sample an existing held-price signal at the new interval midpoints."""
    old_edges = np.asarray(old_edges, dtype=float)
    old_prices = np.asarray(old_prices, dtype=float)
    new_edges = np.asarray(new_edges, dtype=float)
    midpoints = 0.5 * (new_edges[:-1] + new_edges[1:])
    if old_prices.size + 1 != old_edges.size:
        raise ValueError('old_edges and old_prices have incompatible lengths.')
    indices = np.searchsorted(old_edges[1:], midpoints, side='right')
    indices = np.minimum(indices, old_prices.size - 1)
    return old_prices[indices].copy()


def build_complex_deterministic_target():
    """Generate a jump-aware target from the 24-hour reference price signal."""
    baseline = _baseline()
    generator = bm_1(baseline, baseline, T, TARGET_T_STEPS)
    target_t = np.linspace(T[0], T[1], TARGET_T_STEPS, dtype=float)
    result = generator.deterministic_solver(
        [X0_DETERMINISTIC],
        complex_parameters_for_prices(HOURLY_EDGES, REFERENCE_PRICES_24),
        t_eval=target_t,
        breakpoints=HOURLY_EDGES[1:-1],
        rtol=1e-10,
        atol=1e-12,
        max_step=1.0 / (TARGET_T_STEPS - 1),
    )
    target_demand = np.asarray(result[7], dtype=float)
    interpolators = []
    for hour in range(24):
        left = HOURLY_EDGES[hour]
        right = HOURLY_EDGES[hour + 1]
        if hour < 23:
            mask = (target_t >= left) & (target_t < right)
        else:
            mask = (target_t >= left) & (target_t <= right)
        local_t = target_t[mask]
        local_demand = target_demand[mask]
        if local_t.size < 3:
            raise RuntimeError('Target grid is too coarse inside an hourly interval.')
        interpolators.append(PchipInterpolator(local_t, local_demand, extrapolate=True))

    def target(query):
        array = np.asarray(query, dtype=float)
        flat = array.reshape(-1)
        indices = np.searchsorted(HOURLY_EDGES[1:], flat, side='right')
        indices = np.minimum(indices, 23)
        values = np.empty_like(flat, dtype=float)
        for hour in range(24):
            mask = indices == hour
            if np.any(mask):
                values[mask] = interpolators[hour](flat[mask])
        values = values.reshape(array.shape)
        return float(values) if array.ndim == 0 else values
    return (target, target_t, target_demand)


def _build_evaluator(model, n_intervals, *, fast_fv=False):
    edges = _edges(n_intervals)
    spatial_edges, _, _ = model.fokker_planc_reference.grid(FV_X_STEPS)
    initial_mass = control.beta_initial_cell_mass(spatial_edges, alpha=INITIAL_ALPHA, beta=INITIAL_BETA)
    evaluator_class = control.FastFVStochasticPriceEvaluator if fast_fv else control.FVStochasticPriceEvaluator
    return evaluator_class(
        model,
        complex_parameter_template(edges),
        edges,
        SIGMA_X,
        initial_mass,
        objective_max_step=COMPLEX_OBJECTIVE_MAX_STEP,
        solver_max_step=COMPLEX_SOLVER_MAX_STEP,
        breakpoints=HOURLY_EDGES[1:-1],
    )


def _unique_rows(*groups):
    rows = []
    for group in groups:
        array = np.asarray(group, dtype=float)
        if array.ndim == 1:
            array = array[None, :]
        for row in array:
            if not any((np.array_equal(row, previous) for previous in rows)):
                rows.append(row.copy())
    return np.asarray(rows, dtype=float)


def _starts_for_case(n_intervals, previous_case=None):
    reference = _coarsen_reference_prices(n_intervals)
    neutral = np.full(n_intervals, 0.5, dtype=float)
    if previous_case is None:
        warm = reference
    else:
        warm = _sample_piecewise_prices(
            previous_case.edges,
            previous_case.optimization.best.x,
            _edges(n_intervals),
        )
    if n_intervals >= 12:
        return _unique_rows(warm, reference)
    phase = np.linspace(0.0, 2.0 * np.pi, n_intervals, endpoint=False)
    perturb_up = np.clip(warm + 0.05 * np.sin(phase), 0.0, 1.0)
    perturb_down = np.clip(warm - 0.05 * np.sin(phase), 0.0, 1.0)
    return _unique_rows(warm, reference, neutral, perturb_up, perturb_down)


def _verify_common_numerical_partition(evaluators):
    """The same physical held signal must give the same cost for every N."""
    test_prices = np.array([0.72, 0.28], dtype=float)
    values = {}
    for count, evaluator in evaluators.items():
        prices = _lift_half_day_prices(test_prices, count)
        values[count] = evaluator.evaluate(prices).objective.total
    reference = values[min(values)]
    for count, value in values.items():
        if not np.isclose(value, reference, atol=5e-13, rtol=0.0):
            raise RuntimeError(
                f'Common-policy check failed: N={count}, J={value:.16e}, reference={reference:.16e}.',
            )
    return values


def _paired_improvement(reference, candidate):
    difference = np.asarray(reference.path_objectives, dtype=float) - np.asarray(
        candidate.path_objectives,
        dtype=float,
    )
    mean = float(np.mean(difference))
    se = float(np.std(difference, ddof=1) / np.sqrt(difference.size))
    return (mean, se)


def _optimize_price_cases(evaluators, interval_counts, starts_for_case, progress):
    """Optimize price grids in order, warm-starting each from the previous case."""
    cases = {}
    previous = None
    total = len(interval_counts)
    for index, count in enumerate(interval_counts, start=1):
        evaluator = evaluators[count]
        if progress is not None:
            progress('')
            progress(f'[{index}/{total}] Optimizing {count} price intervals ({count - 1} price changes)')
        optimization = control.multistart_powell(
            evaluator,
            starts_for_case(count, previous),
            progress=progress,
            options={'xtol': 5e-05, 'ftol': 1e-09, 'maxiter': 100},
        )
        case = PriceCase(
            n_intervals=count,
            edges=evaluator.interval_edges.copy(),
            evaluator=evaluator,
            optimization=optimization,
            evaluation=evaluator.evaluate(optimization.best.x),
        )
        cases[count] = case
        previous = case
    return cases


def run_complex_target_experiment(progress=print, validate_mc=True, *, fast_fv=False):
    target, target_t, target_demand = build_complex_deterministic_target()
    model = build_stochastic_model(target)
    evaluators = {count: _build_evaluator(model, count, fast_fv=fast_fv) for count in INTERVAL_COUNTS}
    common_policy_values = _verify_common_numerical_partition(evaluators)
    if progress is not None:
        progress('Common numerical-partition check:')
        for count in INTERVAL_COUNTS:
            progress(f'  N={count}: J={common_policy_values[count]:.10e}')
    cases = _optimize_price_cases(evaluators, INTERVAL_COUNTS, _starts_for_case, progress)
    mc = _evaluate_cases_mc(
        model,
        cases,
        HOURLY_EDGES[1:-1],
        n_paths=COMPLEX_MC_PATHS,
        initial_seed=COMPLEX_MC_INITIAL_SEED,
        brownian_seed=COMPLEX_MC_BROWNIAN_SEED,
        max_step=COMPLEX_SOLVER_MAX_STEP,
    ) if validate_mc else None
    return {
        'model': model,
        'target_t': target_t,
        'target_demand': target_demand,
        'common_policy_values': common_policy_values,
        'cases': cases,
        'mc': mc,
    }


def multi_edges(n_intervals):
    n_intervals = int(n_intervals)
    if n_intervals < 1:
        raise ValueError('n_intervals must be positive.')
    return np.linspace(T[0], T[1], n_intervals + 1, dtype=float)


def _lift_half_day_prices(two_prices, n_intervals):
    """Represent one first-half and one second-half price on an even grid."""
    two_prices = np.asarray(two_prices, dtype=float).reshape(-1)
    n_intervals = int(n_intervals)
    if two_prices.size != 2:
        raise ValueError('two_prices must contain exactly two values.')
    if n_intervals < 2 or n_intervals % 2:
        raise ValueError('n_intervals must be an even integer >= 2.')
    return np.concatenate(
        (
            np.full(n_intervals // 2, two_prices[0], dtype=float),
            np.full(n_intervals // 2, two_prices[1], dtype=float),
        ),
    )


def multi_build_evaluator(model, n_intervals, common_breakpoints, *, fast_fv=False):
    edges = multi_edges(n_intervals)
    spatial_edges, _, _ = model.fokker_planc_reference.grid(FV_X_STEPS)
    initial_mass = control.beta_initial_cell_mass(spatial_edges, alpha=INITIAL_ALPHA, beta=INITIAL_BETA)
    evaluator_class = control.FastFVStochasticPriceEvaluator if fast_fv else control.FVStochasticPriceEvaluator
    return evaluator_class(
        model,
        complex_parameter_template(edges),
        edges,
        SIGMA_X,
        initial_mass,
        objective_max_step=MULTI_OBJECTIVE_MAX_STEP,
        solver_max_step=MULTI_SOLVER_MAX_STEP,
        breakpoints=common_breakpoints,
    )


def multi_starts_for_case(n_intervals, previous_case=None):
    deterministic = _lift_half_day_prices(DETERMINISTIC_REFERENCE_PRICES, n_intervals)
    neutral = np.full(n_intervals, 0.5, dtype=float)
    if previous_case is None:
        warm = deterministic
    else:
        warm = _sample_piecewise_prices(
            previous_case.edges,
            previous_case.optimization.best.x,
            multi_edges(n_intervals),
        )
    phase = np.linspace(0.0, np.pi, n_intervals)
    up_down = np.clip(warm + 0.04 * np.cos(phase), 0.0, 1.0)
    down_up = np.clip(warm - 0.04 * np.cos(phase), 0.0, 1.0)
    if n_intervals >= 24:
        return _unique_rows(warm, deterministic)
    return _unique_rows(warm, deterministic, neutral, up_down, down_up)


def run_multi_price_experiment(progress=print, validate_mc=True, *, fast_fv=False):
    target, _, _ = build_deterministic_target()
    model = build_stochastic_model(target)
    common_breakpoints = _union_breakpoints()
    evaluators = {
        count: multi_build_evaluator(model, count, common_breakpoints, fast_fv=fast_fv)
         for count in MULTI_INTERVAL_COUNTS
    }
    nested_values = _verify_common_numerical_partition(evaluators)
    if progress is not None:
        progress('Nested two-price policy check:')
        for count in MULTI_INTERVAL_COUNTS:
            progress(f'  N={count}: J={nested_values[count]:.10e}')
    cases = _optimize_price_cases(evaluators, MULTI_INTERVAL_COUNTS, multi_starts_for_case, progress)
    mc = _evaluate_cases_mc(
        model,
        cases,
        common_breakpoints,
        n_paths=MULTI_MC_PATHS,
        initial_seed=MULTI_MC_INITIAL_SEED,
        brownian_seed=MULTI_MC_BROWNIAN_SEED,
        max_step=MULTI_SOLVER_MAX_STEP,
    ) if validate_mc else None
    return {
        'model': model,
        'common_breakpoints': common_breakpoints,
        'nested_values': nested_values,
        'cases': cases,
        'mc': mc,
    }


def _initial_mass_builder(edges):
    return control.beta_initial_cell_mass(edges, alpha=INITIAL_ALPHA, beta=INITIAL_BETA)


def _point_initial_mass_builder(edges):
    return control.point_initial_cell_mass(edges, X0_DETERMINISTIC)


def build_f3_a_problem():
    """Return the F3-A problem using the established two-price reference."""
    target, _, _ = build_deterministic_target()
    baseline = build_stochastic_model(target).B
    parameters = control.BM1Parameters(
        state_response=linear_state_response,
        price_response=linear_price_response,
        C=C,
        k=K,
        Xi=XI,
        sigma_x=SIGMA_X,
    )
    problem = control.BM1ControlProblem(
        baseline=baseline,
        target=target,
        horizon=T,
        price_edges=INTERVAL_EDGES,
        parameters=parameters,
        initial_mass_builder=_initial_mass_builder,
        deterministic_initial_state=X0_DETERMINISTIC,
        observed_prices=KNOWN_DETERMINISTIC_PRICES,
    )
    numerics = control.BM1ControlNumerics(
        x_steps=201,
        objective_max_step=1.0 / 96.0,
        solver_max_step=1.0 / 96.0,
    )
    return (problem, numerics)


def _reference_starts(problem):
    """Common starts for the two-price F3 reference and sensitivity problems."""
    return np.vstack(
        (problem.observed_prices, INITIAL_PRICES, np.full(problem.n_intervals, 0.5, dtype=float)),
    )


def run_reference(*, optimize=False, progress=print):
    problem, numerics = build_f3_a_problem()
    fixed = problem.evaluate_observed(numerics, fast=True)
    result = {'problem': problem, 'numerics': numerics, 'fixed': fixed, 'optimization': None}
    if optimize:
        evaluator = problem.build_evaluator(numerics, fast=True)
        result['optimization'] = control.multistart_powell(
            evaluator,
            _reference_starts(problem),
            progress=progress,
            options={'xtol': 2e-05, 'ftol': 1e-09, 'maxiter': 100},
        )
    return result


@dataclass(frozen=True)
class F3CSensitivityCase:
    """One independently optimized F3-C weight pair and its common-FV reevaluation."""
    terminal_weight: float
    smoothing_weight: float
    optimization: object
    evaluation: object


def build_f3_c_problem(
    *,
    terminal_reference=F3_C_TERMINAL_REFERENCE,
    terminal_weight=0.0,
    smoothing_weight=0.0,
):
    """Extend the F3-A problem with terminal-state and price-jump penalties."""
    problem, numerics = build_f3_a_problem()
    problem = replace(
        problem,
        terminal_reference=terminal_reference,
        terminal_weight=terminal_weight,
        price_smoothing_weight=smoothing_weight,
    )
    return problem, numerics


def _f3_c_weights(values, name):
    values = np.asarray(values, dtype=float).reshape(-1)
    if (
        values.size < 1
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(np.diff(values) <= 0.0)
    ):
        raise ValueError(f'{name} must be a finite, non-negative, strictly increasing sequence.')
    return tuple(float(value) for value in values)


def _f3_c_weight_pairs(terminal_weights, smoothing_weights, *, include_grid=False):
    """Return unique axis sweeps, and optionally the full Cartesian weight grid."""
    terminal_weights = _f3_c_weights(terminal_weights, 'terminal_weights')
    smoothing_weights = _f3_c_weights(smoothing_weights, 'smoothing_weights')
    pairs = [(0.0, 0.0)]
    pairs.extend((weight, 0.0) for weight in terminal_weights if weight > 0.0)
    pairs.extend((0.0, weight) for weight in smoothing_weights if weight > 0.0)
    if include_grid:
        pairs.extend(
            (terminal, smoothing)
            for terminal in terminal_weights
            for smoothing in smoothing_weights
            if terminal > 0.0 and smoothing > 0.0
        )
    return tuple(pairs)


def _optimize_f3_c_problem(problem, numerics, progress):
    evaluator = problem.build_evaluator(numerics, fast=True)
    optimization = control.multistart_powell(
        evaluator,
        _reference_starts(problem),
        progress=progress,
        options={'xtol': 2e-05, 'ftol': 1e-09, 'maxiter': 100},
    )
    return evaluator, optimization, evaluator.evaluate(optimization.best.x)


def run_f3_c(
    *,
    terminal_reference=F3_C_TERMINAL_REFERENCE,
    terminal_weight=0.0,
    smoothing_weight=0.0,
    progress=print,
):
    """Optimize the F3-C augmented objective and reevaluate the returned price vector."""
    problem, numerics = build_f3_c_problem(
        terminal_reference=terminal_reference,
        terminal_weight=terminal_weight,
        smoothing_weight=smoothing_weight,
    )
    evaluator, optimization, optimized = _optimize_f3_c_problem(problem, numerics, progress)
    fixed = evaluator.evaluate(problem.observed_prices)
    return {
        'problem': problem,
        'numerics': numerics,
        'evaluator': evaluator,
        'fixed': fixed,
        'optimization': optimization,
        'optimized': optimized,
    }


def run_f3_c_sensitivity(
    *,
    terminal_reference=F3_C_TERMINAL_REFERENCE,
    terminal_weights=F3_C_TERMINAL_WEIGHTS,
    smoothing_weights=F3_C_SMOOTHING_WEIGHTS,
    include_grid=False,
    progress=print,
):
    """Optimize common F3-C problems while changing only the objective weights.

    Cross-weight comparisons should use the raw tracking cost, terminal MSE,
    price roughness and returned prices. The weighted total changes definition
    with the weights and is therefore not a common performance metric.
    """
    pairs = _f3_c_weight_pairs(
        terminal_weights,
        smoothing_weights,
        include_grid=include_grid,
    )
    base_problem, numerics = build_f3_a_problem()
    cases = []
    for index, (terminal_weight, smoothing_weight) in enumerate(pairs, start=1):
        if progress is not None:
            progress(
                f'[{index}/{len(pairs)}] lambda_T={terminal_weight:.6g}, '
                f'lambda_U={smoothing_weight:.6g}',
            )
        problem = replace(
            base_problem,
            terminal_reference=terminal_reference,
            terminal_weight=terminal_weight,
            price_smoothing_weight=smoothing_weight,
        )
        _, optimization, evaluation = _optimize_f3_c_problem(problem, numerics, progress)
        cases.append(
            F3CSensitivityCase(
                terminal_weight=terminal_weight,
                smoothing_weight=smoothing_weight,
                optimization=optimization,
                evaluation=evaluation,
            ),
        )
    return {
        'terminal_reference': float(terminal_reference),
        'terminal_weights': _f3_c_weights(terminal_weights, 'terminal_weights'),
        'smoothing_weights': _f3_c_weights(smoothing_weights, 'smoothing_weights'),
        'include_grid': bool(include_grid),
        'cases': tuple(cases),
    }


def _run_f3_d_case(
    *,
    terminal_weight,
    smoothing_weight,
    validate_mc,
    mc_paths,
    progress,
):
    """Optimize deterministic/stochastic policies, then cross-evaluate both."""
    problem, numerics = build_f3_c_problem(
        terminal_weight=terminal_weight,
        smoothing_weight=smoothing_weight,
    )
    deterministic = problem.build_deterministic_evaluator(numerics)
    stochastic = problem.build_evaluator(numerics, fast=True)
    starts = _reference_starts(problem)

    if progress is not None:
        progress(
            f'deterministic optimization: lambda_T={terminal_weight:.6g}, '
            f'lambda_U={smoothing_weight:.6g}',
        )
    deterministic_optimization = control.multistart_powell(
        deterministic,
        starts,
        progress=progress,
        options={'xtol': 2e-05, 'ftol': 1e-09, 'maxiter': 100},
    )
    if progress is not None:
        progress('stochastic optimization')
    stochastic_optimization = control.multistart_powell(
        stochastic,
        starts,
        progress=progress,
        options={'xtol': 2e-05, 'ftol': 1e-09, 'maxiter': 100},
    )

    policies = {
        'deterministic_policy': np.asarray(deterministic_optimization.best.x, dtype=float).copy(),
        'stochastic_policy': np.asarray(stochastic_optimization.best.x, dtype=float).copy(),
    }
    deterministic_evaluations = {
        name: deterministic.evaluate(prices)
        for name, prices in policies.items()
    }
    stochastic_evaluations = {
        name: stochastic.evaluate(prices)
        for name, prices in policies.items()
    }

    deterministic_regret = (
        deterministic_evaluations['stochastic_policy'].total
        - deterministic_evaluations['deterministic_policy'].total
    )
    stochastic_value = (
        stochastic_evaluations['deterministic_policy'].total
        - stochastic_evaluations['stochastic_policy'].total
    )
    policy_difference = policies['stochastic_policy'] - policies['deterministic_policy']

    mc = None
    paired_mc = None
    if validate_mc:
        mc = control.evaluate_mc_candidates_common_random_numbers(
            stochastic,
            policies,
            n_paths=int(mc_paths),
            initial_seed=F3_D_MC_INITIAL_SEED,
            brownian_seed=F3_D_MC_BROWNIAN_SEED,
        )
        improvement, standard_error = control.paired_mc_improvement(
            mc['deterministic_policy'],
            mc['stochastic_policy'],
        )
        paired_mc = {
            'improvement': float(improvement),
            'standard_error': float(standard_error),
            'ratio': float(improvement / standard_error) if standard_error > 0.0 else float('inf'),
        }

    return {
        'problem': problem,
        'numerics': numerics,
        'deterministic_evaluator': deterministic,
        'stochastic_evaluator': stochastic,
        'deterministic_optimization': deterministic_optimization,
        'stochastic_optimization': stochastic_optimization,
        'policies': policies,
        'deterministic_evaluations': deterministic_evaluations,
        'stochastic_evaluations': stochastic_evaluations,
        'deterministic_regret': float(deterministic_regret),
        'stochastic_value': float(stochastic_value),
        'policy_linf': float(np.max(np.abs(policy_difference))),
        'policy_l2': float(np.linalg.norm(policy_difference)),
        'mc': mc,
        'paired_mc': paired_mc,
    }


def _validated_mc_paths(mc_paths, validate_mc):
    """Return a validated path count, or zero when an MC check is disabled."""
    if not validate_mc:
        return 0
    if isinstance(mc_paths, (bool, np.bool_)):
        raise TypeError('mc_paths must be an integer.')
    try:
        validated = int(mc_paths)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError('mc_paths must be an integer.') from exc
    if float(validated) != float(mc_paths) or validated < 2:
        raise ValueError('mc_paths must be an integer >= 2.')
    return validated


def run_f3_d(
    *,
    validate_mc=True,
    mc_paths=F3_D_MC_PATHS,
    progress=print,
):
    """Compare deterministic and stochastic F3 decisions under common objectives."""
    validated_mc_paths = _validated_mc_paths(mc_paths, validate_mc)
    cases = {
        'unregularized': _run_f3_d_case(
            terminal_weight=0.0,
            smoothing_weight=0.0,
            validate_mc=validate_mc,
            mc_paths=validated_mc_paths,
            progress=progress,
        ),
        'light_regularization': _run_f3_d_case(
            terminal_weight=F3_D_LIGHT_TERMINAL_WEIGHT,
            smoothing_weight=F3_D_LIGHT_SMOOTHING_WEIGHT,
            validate_mc=validate_mc,
            mc_paths=validated_mc_paths,
            progress=progress,
        ),
    }
    return {'cases': cases, 'mc_paths': validated_mc_paths}


def build_f3_e_scenarios():
    """Build the four F3-E models that isolate the two uncertainty sources."""
    base_problem, numerics = build_f3_a_problem()
    zero_noise = replace(base_problem.parameters, sigma_x=0.0)

    problems = {
        'deterministic': replace(
            base_problem,
            parameters=zero_noise,
            initial_mass_builder=_point_initial_mass_builder,
        ),
        'initial_only': replace(
            base_problem,
            parameters=zero_noise,
            initial_mass_builder=_initial_mass_builder,
        ),
        'process_only': replace(
            base_problem,
            initial_mass_builder=_point_initial_mass_builder,
        ),
        'full': base_problem,
    }
    evaluators = {
        'deterministic': problems['deterministic'].build_deterministic_evaluator(numerics),
        'initial_only': problems['initial_only'].build_evaluator(numerics, fast=True),
        'process_only': problems['process_only'].build_evaluator(numerics, fast=True),
        'full': problems['full'].build_evaluator(numerics, fast=True),
    }
    return {
        'numerics': numerics,
        'problems': problems,
        'evaluators': evaluators,
    }


def run_f3_e(progress=print):
    """Optimize and cross-evaluate the four initial/process uncertainty cases."""
    setup = build_f3_e_scenarios()
    problems = setup['problems']
    evaluators = setup['evaluators']
    starts = _reference_starts(problems['full'])

    optimizations = {}
    policies = {}
    for index, name in enumerate(F3_E_SCENARIO_ORDER, start=1):
        if progress is not None:
            progress(f'[{index}/{len(F3_E_SCENARIO_ORDER)}] optimizing {name}')
        optimization = control.multistart_powell(
            evaluators[name],
            starts,
            progress=progress,
            options={'xtol': 2e-05, 'ftol': 1e-09, 'maxiter': 100},
        )
        optimizations[name] = optimization
        policies[name] = np.asarray(optimization.best.x, dtype=float).copy()

    evaluations = {
        model_name: {
            policy_name: evaluators[model_name].evaluate(policies[policy_name])
            for policy_name in F3_E_SCENARIO_ORDER
        }
        for model_name in F3_E_SCENARIO_ORDER
    }
    regrets = {
        model_name: {
            policy_name: float(
                evaluations[model_name][policy_name].total
                - evaluations[model_name][model_name].total
            )
            for policy_name in F3_E_SCENARIO_ORDER
        }
        for model_name in F3_E_SCENARIO_ORDER
    }

    deterministic_policy = policies['deterministic']
    policy_distances = {}
    for name in F3_E_SCENARIO_ORDER:
        difference = policies[name] - deterministic_policy
        policy_distances[name] = {
            'linf': float(np.max(np.abs(difference))),
            'l2': float(np.linalg.norm(difference)),
        }

    return {
        'numerics': setup['numerics'],
        'problems': problems,
        'evaluators': evaluators,
        'scenario_order': F3_E_SCENARIO_ORDER,
        'optimizations': optimizations,
        'policies': policies,
        'evaluations': evaluations,
        'regrets': regrets,
        'policy_distances': policy_distances,
    }


def build_f3_f_setup():
    """Build matched FV/mixture design evaluators plus common 201-cell references."""
    problem, reference_numerics = build_f3_a_problem()
    design_numerics = replace(reference_numerics, x_steps=MIXTURE_OPT_X_STEPS)

    fv_reference = problem.build_evaluator(reference_numerics, fast=True)
    fv_design = problem.build_evaluator(design_numerics, fast=True)
    mixture_design = problem.build_mixture_evaluator(
        design_numerics,
        MIXTURE_OPT_DEGREE,
        cfl=DEFAULT_CFL,
    )
    mixture_reference = {
        degree: problem.build_mixture_evaluator(
            reference_numerics,
            degree,
            cfl=DEFAULT_CFL,
        )
        for degree in FINAL_MIXTURE_DEGREES
    }
    return {
        'problem': problem,
        'reference_numerics': reference_numerics,
        'design_numerics': design_numerics,
        'fv_reference': fv_reference,
        'fv_design': fv_design,
        'mixture_design': mixture_design,
        'mixture_reference': mixture_reference,
    }


def run_f3_f(
    *,
    validate_mc=True,
    mc_paths=F3_F_MC_PATHS,
    progress=print,
):
    """Compare FV-grid and dynamic-mixture effects on the stochastic control decision."""
    validated_mc_paths = _validated_mc_paths(mc_paths, validate_mc)
    setup = build_f3_f_setup()
    problem = setup['problem']
    fv_reference = setup['fv_reference']
    fv_design = setup['fv_design']
    mixture_design = setup['mixture_design']
    starts = _reference_starts(problem)

    if progress is not None:
        progress('[1/3] optimizing the 201-cell FV reference policy')
    fv_reference_optimization = control.multistart_powell(
        fv_reference,
        starts,
        progress=progress,
        options={'xtol': 2e-05, 'ftol': 1e-09, 'maxiter': 100},
    )

    if progress is not None:
        progress('[2/3] optimizing the 81-cell FV design-grid policy')
    fv_design_optimization = control.multistart_powell(
        fv_design,
        starts,
        progress=progress,
        options={'xtol': 2e-05, 'ftol': 1e-09, 'maxiter': 100},
    )

    if progress is not None:
        progress('[3/3] optimizing the 81-cell degree-30 dynamic-mixture policy')
    mixture_optimization = control.multistart_powell(
        mixture_design,
        starts,
        progress=progress,
        options={'xtol': 0.001, 'ftol': 1e-05, 'maxiter': 30},
    )

    policies = {
        'fv_reference_policy': np.asarray(fv_reference_optimization.best.x, dtype=float).copy(),
        'fv_design_policy': np.asarray(fv_design_optimization.best.x, dtype=float).copy(),
        'mixture_policy': np.asarray(mixture_optimization.best.x, dtype=float).copy(),
    }

    design_evaluations = {
        'fv_81': {
            name: fv_design.evaluate(prices)
            for name, prices in policies.items()
        },
        'mixture_30_81': {
            name: mixture_design.evaluate(prices)
            for name, prices in policies.items()
        },
    }
    reference_evaluations = {
        'fv_201': {
            name: fv_reference.evaluate(prices)
            for name, prices in policies.items()
        },
    }
    for degree, evaluator in setup['mixture_reference'].items():
        reference_evaluations[f'mixture_{degree}_201'] = {
            name: evaluator.evaluate(prices)
            for name, prices in policies.items()
        }

    fv_grid_regret = (
        reference_evaluations['fv_201']['fv_design_policy'].total
        - reference_evaluations['fv_201']['fv_reference_policy'].total
    )
    fv_reference_regret_of_mixture = (
        reference_evaluations['fv_201']['mixture_policy'].total
        - reference_evaluations['fv_201']['fv_reference_policy'].total
    )
    fv_design_regret_of_mixture = (
        design_evaluations['fv_81']['mixture_policy'].total
        - design_evaluations['fv_81']['fv_design_policy'].total
    )
    mixture_design_regret_of_fv = (
        design_evaluations['mixture_30_81']['fv_design_policy'].total
        - design_evaluations['mixture_30_81']['mixture_policy'].total
    )

    reference_differences = {
        method: {
            policy: float(values[policy].total - values['fv_reference_policy'].total)
            for policy in ('fv_design_policy', 'mixture_policy')
        }
        for method, values in reference_evaluations.items()
    }

    def policy_distance(left, right):
        difference = policies[left] - policies[right]
        return {
            'linf': float(np.max(np.abs(difference))),
            'l2': float(np.linalg.norm(difference)),
        }

    policy_distances = {
        'fv_design_to_reference': policy_distance('fv_design_policy', 'fv_reference_policy'),
        'mixture_to_fv_design': policy_distance('mixture_policy', 'fv_design_policy'),
        'mixture_to_reference': policy_distance('mixture_policy', 'fv_reference_policy'),
    }

    mc = None
    paired_mc = None
    if validate_mc:
        mc = control.evaluate_mc_candidates_common_random_numbers(
            fv_reference,
            policies,
            n_paths=validated_mc_paths,
            initial_seed=F3_F_MC_INITIAL_SEED,
            brownian_seed=F3_F_MC_BROWNIAN_SEED,
        )
        paired_mc = {}
        for name in ('fv_design_policy', 'mixture_policy'):
            difference, standard_error = control.paired_mc_improvement(
                mc['fv_reference_policy'],
                mc[name],
            )
            paired_mc[name] = {
                'reference_minus_candidate': float(difference),
                'standard_error': float(standard_error),
                'ratio': float(difference / standard_error) if standard_error > 0.0 else float('inf'),
            }

    return {
        'problem': problem,
        'reference_numerics': setup['reference_numerics'],
        'design_numerics': setup['design_numerics'],
        'fv_reference': fv_reference,
        'fv_design': fv_design,
        'mixture_design': mixture_design,
        'mixture_reference': setup['mixture_reference'],
        'starts': starts.copy(),
        'policies': policies,
        'fv_reference_optimization': fv_reference_optimization,
        'fv_design_optimization': fv_design_optimization,
        'mixture_optimization': mixture_optimization,
        'design_evaluations': design_evaluations,
        'reference_evaluations': reference_evaluations,
        'fv_grid_regret': float(fv_grid_regret),
        'fv_reference_regret_of_mixture': float(fv_reference_regret_of_mixture),
        'fv_design_regret_of_mixture': float(fv_design_regret_of_mixture),
        'mixture_design_regret_of_fv': float(mixture_design_regret_of_fv),
        'reference_differences': reference_differences,
        'policy_distances': policy_distances,
        'mc': mc,
        'paired_mc': paired_mc,
        'mc_paths': validated_mc_paths,
    }


def build_pair(model, n_cells):
    edges, _, _ = model.fokker_planc_reference.grid(n_cells)
    args = (
        model,
        complex_parameter_template(HOURLY_EDGES),
        HOURLY_EDGES,
        SIGMA_X,
        control.beta_initial_cell_mass(edges),
    )
    controls = dict(objective_max_step=1 / 96, solver_max_step=1 / 96, breakpoints=HOURLY_EDGES[1:-1])
    return (
        control.FVStochasticPriceEvaluator(*args, **controls),
        control.FastFVStochasticPriceEvaluator(*args, **controls),
    )


def run_benchmark(n_cells=(201, 1001), repeats=7, optimize=False, progress=print):
    if repeats < 1:
        raise ValueError('repeats must be positive.')
    target, _, _ = build_complex_deterministic_target()
    model = build_stochastic_model(target)
    records = []
    for n in n_cells:
        slow, fast = build_pair(model, n)
        a, b = (slow.evaluate(REFERENCE_PRICES_24), fast.evaluate(REFERENCE_PRICES_24))
        mass_difference = float(np.max(np.abs(a.cell_mass - b.cell_mass)))
        np.testing.assert_allclose(a.cell_mass, b.cell_mass, atol=2e-12, rtol=2e-11)
        np.testing.assert_allclose(a.objective.total, b.objective.total, atol=2e-12, rtol=2e-11)
        timings = {'reference': [], 'fast': []}
        for repeat in range(repeats):
            order = [('reference', slow), ('fast', fast)]
            if repeat % 2:
                order.reverse()
            for name, evaluator in order:
                started = perf_counter()
                evaluator(REFERENCE_PRICES_24)
                timings[name].append(perf_counter() - started)
        old, new = (median(timings['reference']), median(timings['fast']))
        item = dict(
            cells=n,
            reference_seconds=old,
            fast_seconds=new,
            speedup=old / new,
            maximum_cell_mass_difference=mass_difference,
            bitwise_identical_mass=bool(np.array_equal(a.cell_mass, b.cell_mass)),
            objective_difference=abs(a.objective.total - b.objective.total),
            timings=timings,
        )
        records.append(item)
        if progress:
            progress(
                (
                    f'{n} cells: {old:.6f} -> {new:.6f} s; {old / new:.2f}x; max mass difference='
                    f'{mass_difference:.3e}'
                ),
            )
    output = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        repeats=repeats,
        price_intervals=24,
        solver_max_step=1 / 96,
        objective_max_step=1 / 96,
        fixed_price=records,
    )
    if optimize:
        slow, fast = build_pair(model, 201)
        coarse = np.repeat(REFERENCE_PRICES_24.reshape(12, 2).mean(axis=1), 2)
        starts = np.array([coarse, REFERENCE_PRICES_24])
        results = {}
        for name, evaluator in [('reference', slow), ('fast', fast)]:
            if progress:
                progress(f'Full 24-variable Powell comparison: {name}')
            started = perf_counter()
            fit = control.multistart_powell(
                evaluator,
                starts,
                options=dict(xtol=5e-05, ftol=1e-09, maxiter=100),
                progress=progress,
            )
            results[name] = dict(
                seconds=perf_counter() - started,
                prices=fit.best.x.tolist(),
                objective=float(fit.best.fun),
                evaluations=fit.objective_evaluations,
                cache_hits=fit.cache_hits,
                successes=sum((bool(r.success) for r in fit.results)),
            )
        x0, x1 = (results['reference']['prices'], results['fast']['prices'])
        results['price_max_difference'] = float(np.max(np.abs(np.array(x0) - x1)))
        results['reference_cost_difference'] = abs(slow(x0) - slow(x1))
        results['speedup'] = results['reference']['seconds'] / results['fast']['seconds']
        output['optimization'] = results
        if progress:
            progress(
                (
                    f"Full optimization speedup={results['speedup']:.2f}x; price max difference="
                    f"{results['price_max_difference']:.3e}; reference cost difference="
                    f"{results['reference_cost_difference']:.3e}"
                ),
            )
    return output


def run_optimization_reference(
    *,
    fv_x_steps=FV_X_STEPS,
    mixture_opt_x_steps=MIXTURE_OPT_X_STEPS,
    final_mixture_x_steps=FINAL_MIXTURE_X_STEPS,
    objective_max_step=OBJECTIVE_MAX_STEP,
    solver_max_step=SOLVER_MAX_STEP,
    coarse_values=COARSE_VALUES,
    n_coarse_starts=N_COARSE_STARTS,
    optimize_mixture=True,
    validate_mc=True,
    mc_paths=MC_PATHS,
    progress=None,
):
    """Run the complete F2-I reference workflow."""

    def report(message):
        if progress is not None:
            progress(message)
    report('[1/6] Building deterministic target and FV objective surface')
    target, target_t, target_demand = build_deterministic_target()
    model = build_stochastic_model(target)
    fv = _build_fv_evaluator(
        model,
        x_steps=fv_x_steps,
        objective_max_step=objective_max_step,
        solver_max_step=solver_max_step,
    )
    coarse_values = np.asarray(coarse_values, dtype=float)
    surface = control.coarse_price_surface(fv, coarse_values)
    surface_starts = control.best_surface_starts(coarse_values, surface, n_coarse_starts)
    fv_starts = _unique_rows(surface_starts, KNOWN_DETERMINISTIC_PRICES, INITIAL_PRICES)
    report('[2/6] Pure-FV multistart Powell optimization')
    fv_optimization = control.multistart_powell(
        fv,
        fv_starts,
        progress=progress,
        options={'xtol': 2e-05, 'ftol': 1e-09, 'maxiter': 100},
    )
    fv_prices = np.asarray(fv_optimization.best.x, dtype=float)
    report('[3/6] Dynamic-mixture optimizer-model comparison')
    if optimize_mixture:
        mixture_support = _build_fv_evaluator(
            model,
            x_steps=mixture_opt_x_steps,
            objective_max_step=objective_max_step,
            solver_max_step=solver_max_step,
        )
        mixture_optimizer = control.DynamicMixturePriceEvaluator(
            mixture_support,
            MIXTURE_OPT_DEGREE,
            cfl=DEFAULT_CFL,
        )
        mixture_starts = _unique_rows(fv_prices, KNOWN_DETERMINISTIC_PRICES)
        mixture_optimization = control.multistart_powell(
            mixture_optimizer,
            mixture_starts,
            progress=progress,
            options={'xtol': 0.001, 'ftol': 1e-05, 'maxiter': 30},
        )
        mixture_prices = np.asarray(mixture_optimization.best.x, dtype=float)
    else:
        mixture_optimizer = None
        mixture_optimization = None
        mixture_prices = fv_prices.copy()
    candidates = {
        'deterministic_reference': KNOWN_DETERMINISTIC_PRICES.copy(),
        'fv_optimum': fv_prices.copy(),
        'mixture_optimum': mixture_prices.copy(),
    }
    report('[4/6] Reevaluating candidates with pure FV')
    fv_evaluations = {name: fv.evaluate(prices) for name, prices in candidates.items()}
    report('[5/6] High-resolution mixture reevaluations')
    final_mixture = {}
    support = _build_fv_evaluator(
        model,
        x_steps=final_mixture_x_steps,
        objective_max_step=objective_max_step,
        solver_max_step=solver_max_step,
    )
    for degree in FINAL_MIXTURE_DEGREES:
        evaluator = control.DynamicMixturePriceEvaluator(support, degree, cfl=DEFAULT_CFL)
        final_mixture[degree] = {name: evaluator.evaluate(prices) for name, prices in candidates.items()}
    report('[6/6] Paired Monte Carlo validation')
    mc = None
    paired = None
    if validate_mc:
        mc = control.evaluate_mc_candidates_common_random_numbers(
            fv,
            candidates,
            n_paths=mc_paths,
            initial_seed=MC_INITIAL_SEED,
            brownian_seed=MC_BROWNIAN_SEED,
        )
        reference = mc['deterministic_reference']
        paired = {
            name: control.paired_mc_improvement(reference, item)
             for name, item in mc.items() if name != 'deterministic_reference'
        }
    return {
        'target_t': target_t,
        'target_demand_dense': target_demand,
        'model': model,
        'fv_evaluator': fv,
        'coarse_values': coarse_values,
        'coarse_surface': surface,
        'fv_starts': fv_starts,
        'fv_optimization': fv_optimization,
        'mixture_optimizer': mixture_optimizer,
        'mixture_optimization': mixture_optimization,
        'candidates': candidates,
        'fv_evaluations': fv_evaluations,
        'final_mixture': final_mixture,
        'mc': mc,
        'paired_mc_improvement': paired,
    }


def _union_breakpoints(interval_counts=MULTI_INTERVAL_COUNTS):
    points = []
    for count in interval_counts:
        points.extend(multi_edges(count)[1:-1])
    return np.unique(np.asarray(points, dtype=float))


def _evaluate_cases_mc(model, cases, breakpoints, *, n_paths, initial_seed, brownian_seed, max_step):
    """Compare different price grids using identical initial states and Wiener increments."""
    reference = cases[min(cases)].evaluator
    for case in cases.values():
        evaluator = case.evaluator
        if not all(
            (np.array_equal(getattr(evaluator, name), getattr(reference, name)) for name in (
                'solver_t_eval',
                'widths',
                'target',
            )),
        ):
            raise ValueError('Paired MC comparisons require the same output grid and objective.')
    initial_states = control.sample_beta_initial_states(
        n_paths,
        initial_seed,
        alpha=INITIAL_ALPHA,
        beta=INITIAL_BETA,
    )
    integration_t, _ = model.euler_maruyama.build_time_grid(
        reference.solver_t_eval,
        breakpoints=breakpoints,
        max_step=max_step,
    )
    increments = model.euler_maruyama.brownian_increments(
        integration_t,
        n_paths=n_paths,
        seed=brownian_seed,
    )
    results = {}
    for count, case in cases.items():
        parameters = case.evaluator.parameters_for_prices(case.optimization.best.x)
        output = model.euler_maruyama_solver(
            initial_states,
            parameters,
            SIGMA_X,
            n_paths=n_paths,
            boundary='none',
            t_eval=reference.solver_t_eval,
            breakpoints=breakpoints,
            max_step=max_step,
            brownian_increments=increments,
            return_diagnostics=True,
        )
        demand, diagnostics = (output[7], output[8])
        if diagnostics['total_violations']:
            raise RuntimeError(f'N={count} MC comparison produced raw boundary violations.')
        results[count] = monte_carlo_tracking_objective(
            reference.widths,
            np.asarray(demand, dtype=float)[:, 1:-1],
            reference.target,
        )
    return results
