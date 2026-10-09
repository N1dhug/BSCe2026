"""F1 analytic benchmarks and feasibility experiments; API regressions live in tests."""

import numpy as np

from src.diagnostics.bm_1 import (
    linear_response_required_price,
    linear_response_demand_envelope,
    constant_demand_capacity_certificate,
    constant_demand_implied_state,
)
from src.model.bm_1 import bm_1
from src.method.objective import tracking_error
from src.tools.input_generator.input_builder import input_builder
from src.validation.bm_1.cases import linear_price_response, linear_state_response


T = (0.0, 1.0)
PRICE_STEP_TIME = 0.5
PRICE_BEFORE = 0.75
PRICE_AFTER = 0.25
X0 = 0.5
BASELINE = 0.3
C = 1.0
K = 2.0
XI = 1.0
STEPS_PER_DAY = (24, 48, 96, 192)
IVP_RTOL = 1e-09
IVP_ATOL = 1e-11
IVP_OUTPUT_POINTS = 200
KNOWN_PRICES = np.array([0.72, 0.28], dtype=float)
INITIAL_PRICES = np.array([0.55, 0.45], dtype=float)
RTOL = 1e-09
ATOL = 1e-11
FINE_RTOL = 1e-10
FINE_ATOL = 1e-12
REQUIRED_PRICE = -0.05
PRICE_STARTS = (np.array([0.0]), np.array([0.05]), np.array([0.25]), np.array([0.5]), np.array([0.95]))
CAPACITY_TARGET = 0.82
PRICE_INTERVAL_EDGES = np.array([T[0], PRICE_STEP_TIME, T[1]], dtype=float)
PRICE_INTERVAL_VALUES = np.array([PRICE_BEFORE, PRICE_AFTER], dtype=float)
INTERVAL_EDGES = np.array([T[0], 0.5, T[1]], dtype=float)
OBJECTIVE_T = np.linspace(T[0], T[1], 193)
FEASIBILITY_OBJECTIVE_T = np.linspace(T[0], T[1], 385)
FINE_T = np.linspace(T[0], T[1], 1537)
PRICE_EDGES = np.array(T, dtype=float)
CAPACITY_EDGES = np.linspace(T[0], T[1], 5)
INTERNAL_PRICE_BREAKPOINTS = PRICE_INTERVAL_EDGES[1:-1]
CAPACITY_STARTS = tuple((np.full(CAPACITY_EDGES.size - 1, value) for value in (0.05, 0.25, 0.5, 0.75))) + (
    np.array([0.05, 0.95, 0.05, 0.95]),
    np.array([0.95, 0.05, 0.95, 0.05]),
)


def build_inputs():
    """Build the constant baseline and right-continuous price step."""
    inputs = input_builder()
    B = inputs.energy((np.array(T, dtype=float), np.array([BASELINE, BASELINE], dtype=float)))
    U = inputs.price_intervals((PRICE_INTERVAL_EDGES, PRICE_INTERVAL_VALUES))
    return (B, U)


def exact_constant_price_state(t, initial_state, initial_time, price):
    """Return the exact bm_1 state while price and baseline are constant."""
    times = np.asarray(t, dtype=float)
    equilibrium = 1.0 - price
    if initial_state < equilibrium:
        demand_scale = 1.0 - BASELINE
    else:
        demand_scale = BASELINE
    decay_rate = K * XI * demand_scale / C
    initial_offset = K * (initial_state - equilibrium)
    elapsed = times - initial_time
    return equilibrium + np.arcsinh(np.sinh(initial_offset) * np.exp(-decay_rate * elapsed)) / K


def exact_state(t):
    """Return the exact state for the two price intervals."""
    times = np.asarray(t, dtype=float)
    state = np.empty_like(times)
    before_step = times <= PRICE_STEP_TIME
    state[before_step] = exact_constant_price_state(times[before_step], X0, T[0], PRICE_BEFORE)
    state_at_step = float(
        exact_constant_price_state(np.array([PRICE_STEP_TIME]), X0, T[0], PRICE_BEFORE)[0],
    )
    state[~before_step] = exact_constant_price_state(
        times[~before_step],
        state_at_step,
        PRICE_STEP_TIME,
        PRICE_AFTER,
    )
    return state


def energy_residual(t, state, baseline, demand):
    """Return the cumulative Euler energy-balance residual."""
    increments = np.diff(t) * (demand[:-1] - baseline[:-1])
    accumulated_energy = np.concatenate(([0.0], np.cumsum(increments)))
    return C * (state - state[0]) - accumulated_energy


def run_euler(steps_per_day, B, U):
    """Run the existing Euler-Maruyama interface as uncorrected Euler."""
    model = bm_1(B, B, T, steps_per_day + 1)
    parameters = [U, linear_state_response, linear_price_response, C, K, XI]
    result = model.euler_maruyama_solver([X0], parameters, sigma_x=0.0, n_paths=1, seed=0, boundary='none')
    t, state, price, baseline, _, _, delta, demand = result
    return {
        'steps_per_day': steps_per_day,
        'dt': 1.0 / steps_per_day,
        't': t,
        'state': state,
        'price': price,
        'baseline': baseline,
        'delta': delta,
        'demand': demand,
        'error': float(np.max(np.abs(state - exact_state(t)))),
        'energy_error': float(np.max(np.abs(energy_residual(t, state, baseline, demand)))),
    }


def observed_orders(euler_results):
    """Compute the observed order between consecutive halved time steps."""
    orders = [np.nan]
    for coarse, fine in zip(euler_results[:-1], euler_results[1:]):
        orders.append(np.log(coarse['error'] / fine['error']) / np.log(2.0))
    return orders


def build_baseline():
    """Return the constant normalized baseline used by the benchmark."""
    return input_builder.energy((np.array(T, dtype=float), np.array([BASELINE, BASELINE], dtype=float)))


def build_parameters(prices):
    """Build bm_1 parameters for one price value per interval."""
    price = input_builder.price_intervals((INTERVAL_EDGES, prices))
    return [price, linear_state_response, linear_price_response, C, K, XI]


def solve_prices(model, prices):
    """Solve bm_1 for a candidate price vector on the objective grid."""
    return model.deterministic_solver(
        [X0],
        build_parameters(prices),
        t_eval=OBJECTIVE_T,
        breakpoints=INTERVAL_EDGES[1:-1],
        rtol=RTOL,
        atol=ATOL,
    )


def as_scalar_if_needed(query, values):
    """Preserve scalar calls while supporting arbitrary NumPy query shapes."""
    return float(values) if np.asarray(query).ndim == 0 else values


def feasibility_build_parameters(price):
    """Return the standard deterministic bm_1 parameter vector."""
    return [price, linear_state_response, linear_price_response, C, K, XI]


def price_bound_target_state(t):
    """Return the exact state generated by the extended price -0.05.

    The formula is used only to construct and diagnose the target.  It does
    not relax the optimizer's admissible price bounds.
    """
    query = np.asarray(t, dtype=float)
    equilibrium = 1.0 - REQUIRED_PRICE
    rate = K * XI * (1.0 - BASELINE) / C
    state = equilibrium + np.arcsinh(np.sinh(K * (X0 - equilibrium)) * np.exp(-rate * (query - T[0]))) / K
    return as_scalar_if_needed(query, state)


def price_bound_target(t):
    """Return the demand target associated with the extended negative price."""
    query = np.asarray(t, dtype=float)
    state = np.asarray(price_bound_target_state(query), dtype=float)
    delta = np.tanh(K * (1.0 - state - REQUIRED_PRICE))
    demand = BASELINE + XI * (1.0 - BASELINE) * delta
    return as_scalar_if_needed(query, demand)


def capacity_target(t):
    """Return the constant capacity-stress demand target."""
    query = np.asarray(t, dtype=float)
    demand = np.full_like(query, CAPACITY_TARGET, dtype=float)
    return as_scalar_if_needed(query, demand)


def feasibility_solve_prices(model, interval_edges, prices, t_eval, rtol, atol, max_step=np.inf):
    """Solve bm_1 for an admissible held-price vector."""
    price = input_builder.price_intervals((interval_edges, prices))
    return model.deterministic_solver(
        [X0],
        feasibility_build_parameters(price),
        t_eval=t_eval,
        breakpoints=interval_edges[1:-1],
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )


def optimize_from_starts(model, interval_edges, starts):
    """Run identical bounded optimizations from several starting prices."""
    placeholder = input_builder.price_intervals((np.array(T, dtype=float), np.array([0.5], dtype=float)))
    parameters = feasibility_build_parameters(placeholder)
    results = []
    for initial_prices in starts:
        results.append(
            model.optimize(
                initial_prices,
                [X0],
                parameters,
                interval_edges=interval_edges,
                t_eval=FEASIBILITY_OBJECTIVE_T,
                rtol=RTOL,
                atol=ATOL,
                options={'ftol': 1e-14, 'gtol': 1e-08, 'maxiter': 300},
            ),
        )
    return results


def best_result(results):
    """Return the successful result with the smallest objective."""
    successful = [result for result in results if result.success]
    if not successful:
        raise RuntimeError('Every F1-E optimization start failed.')
    return min(successful, key=lambda result: result.fun)


def run_deterministic_reference():
    """Compare the default/segmented ODE solution and Euler grids to exact X(t).

    This computes diagnostic values; API and invalid-input regressions live in tests.
    """
    B, U = build_inputs()
    parameters = [U, linear_state_response, linear_price_response, C, K, XI]
    model = bm_1(B, B, T, STEPS_PER_DAY[-1] + 1)
    legacy_ivp_result = model.deterministic_solver([X0], parameters)
    legacy_t, legacy_state = (legacy_ivp_result[0], legacy_ivp_result[1])
    legacy_difference = np.abs(legacy_state - exact_state(legacy_t))
    legacy_error = float(np.max(legacy_difference))
    breakpoint_ivp_result = model.deterministic_solver(
        [X0],
        parameters,
        breakpoints=INTERNAL_PRICE_BREAKPOINTS,
    )
    breakpoint_t = breakpoint_ivp_result[0]
    breakpoint_state = breakpoint_ivp_result[1]
    breakpoint_difference = np.abs(breakpoint_state - exact_state(breakpoint_t))
    breakpoint_error = float(np.max(breakpoint_difference))
    improvement = legacy_error / breakpoint_error if breakpoint_error > 0.0 else np.inf
    ivp_t_eval = np.linspace(T[0], T[1], IVP_OUTPUT_POINTS)
    ivp_result = model.deterministic_solver(
        [X0],
        parameters,
        t_eval=ivp_t_eval,
        breakpoints=INTERNAL_PRICE_BREAKPOINTS,
        rtol=IVP_RTOL,
        atol=IVP_ATOL,
    )
    ivp_t, ivp_state = (ivp_result[0], ivp_result[1])
    ivp_difference = np.abs(ivp_state - exact_state(ivp_t))
    ivp_error = float(np.max(ivp_difference))
    euler_results = [run_euler(steps, B, U) for steps in STEPS_PER_DAY]
    orders = observed_orders(euler_results)
    return {
        'baseline': B,
        'price': U,
        'default': legacy_ivp_result,
        'accurate': ivp_result,
        'default_error': legacy_error,
        'segmented_error': breakpoint_error,
        'accurate_error': ivp_error,
        'same_tolerance_improvement': improvement,
        'euler': euler_results,
        'orders': orders,
    }


def run_feasibility_reference():
    """Evaluate price/capacity impossibility certificates and bounded fits."""
    baseline = build_baseline()
    target_state = np.asarray(price_bound_target_state(FEASIBILITY_OBJECTIVE_T), dtype=float)
    target_demand = np.asarray(price_bound_target(FEASIBILITY_OBJECTIVE_T), dtype=float)
    required_price = linear_response_required_price(target_state, target_demand, BASELINE, XI, K)
    _, maximum_demand = linear_response_demand_envelope(target_state, BASELINE, XI, K)
    envelope_gap = target_demand - maximum_demand
    price_model = bm_1(baseline, price_bound_target, T, FEASIBILITY_OBJECTIVE_T.size)
    price_results = optimize_from_starts(price_model, PRICE_EDGES, PRICE_STARTS)
    price_best = best_result(price_results)
    price_solution = feasibility_solve_prices(
        price_model,
        PRICE_EDGES,
        price_best.x,
        FEASIBILITY_OBJECTIVE_T,
        RTOL,
        ATOL,
    )
    price_objective = tracking_error(FEASIBILITY_OBJECTIVE_T, price_solution[7], target_demand)
    fine_solution = feasibility_solve_prices(
        price_model,
        PRICE_EDGES,
        price_best.x,
        FINE_T,
        FINE_RTOL,
        FINE_ATOL,
        max_step=(T[1] - T[0]) / (FINE_T.size - 1),
    )
    fine_objective = tracking_error(
        FINE_T,
        fine_solution[7],
        np.asarray(price_bound_target(FINE_T), dtype=float),
    )
    initial_solution = feasibility_solve_prices(
        price_model,
        PRICE_EDGES,
        np.array([0.5]),
        FEASIBILITY_OBJECTIVE_T,
        RTOL,
        ATOL,
    )
    initial_objective = tracking_error(FEASIBILITY_OBJECTIVE_T, initial_solution[7], target_demand)
    capacity_values = np.asarray(capacity_target(FEASIBILITY_OBJECTIVE_T), dtype=float)
    capacity_required_state = np.asarray(
        constant_demand_implied_state(
            FEASIBILITY_OBJECTIVE_T,
            CAPACITY_TARGET,
            BASELINE,
            C,
            X0,
            initial_time=T[0],
        ),
        dtype=float,
    )
    capacity_certificate = constant_demand_capacity_certificate(CAPACITY_TARGET, BASELINE, C, X0, T)
    capacity_model = bm_1(baseline, capacity_target, T, FEASIBILITY_OBJECTIVE_T.size)
    capacity_results = optimize_from_starts(capacity_model, CAPACITY_EDGES, CAPACITY_STARTS)
    capacity_best = best_result(capacity_results)
    capacity_solution = feasibility_solve_prices(
        capacity_model,
        CAPACITY_EDGES,
        capacity_best.x,
        FEASIBILITY_OBJECTIVE_T,
        RTOL,
        ATOL,
    )
    capacity_objective = tracking_error(FEASIBILITY_OBJECTIVE_T, capacity_solution[7], capacity_values)
    capacity_fine_solution = feasibility_solve_prices(
        capacity_model,
        CAPACITY_EDGES,
        capacity_best.x,
        FINE_T,
        FINE_RTOL,
        FINE_ATOL,
        max_step=(T[1] - T[0]) / (FINE_T.size - 1),
    )
    capacity_fine_objective = tracking_error(
        FINE_T,
        capacity_fine_solution[7],
        np.asarray(capacity_target(FINE_T), dtype=float),
    )
    return {
        't': FEASIBILITY_OBJECTIVE_T,
        'target_state': target_state,
        'target_demand': target_demand,
        'required_price': required_price,
        'envelope_gap': envelope_gap,
        'price_best': price_best,
        'price_solution': price_solution,
        'price_objective': price_objective,
        'price_fine_objective': fine_objective,
        'price_initial_objective': initial_objective,
        'capacity_values': capacity_values,
        'capacity_required_state': capacity_required_state,
        'capacity_certificate': capacity_certificate,
        'capacity_best': capacity_best,
        'capacity_solution': capacity_solution,
        'capacity_objective': capacity_objective,
        'capacity_fine_objective': capacity_fine_objective,
    }
