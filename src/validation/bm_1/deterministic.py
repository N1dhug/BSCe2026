"""F1 analytic benchmarks, feasibility experiments and consistency checks."""

import numpy as np
from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder
from src.validation.bm_1.cases import (linear_price_response, linear_state_response)

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
DEFAULT_BREAKPOINT_TOL = 0.0001
CONTROLLED_IVP_TOL = 1e-07
IMPROVEMENT_FACTOR = 100.0
IVP_RTOL = 1e-09
IVP_ATOL = 1e-11
IVP_OUTPUT_POINTS = 200
ENERGY_TOL = 1e-12
STATE_TOL = 1e-12

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
    state_at_step = float(exact_constant_price_state(np.array([PRICE_STEP_TIME]), X0, T[0], PRICE_BEFORE)[0])
    state[~before_step] = exact_constant_price_state(times[~before_step], state_at_step, PRICE_STEP_TIME, PRICE_AFTER)
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
    return {'steps_per_day': steps_per_day, 'dt': 1.0 / steps_per_day, 't': t, 'state': state, 'price': price,
        'baseline': baseline, 'delta': delta, 'demand': demand, 'error': float(np.max(np.abs(state - exact_state(t)))),
        'energy_error': float(np.max(np.abs(energy_residual(t, state, baseline, demand))))}

def observed_orders(euler_results):
    """Compute the observed order between consecutive halved time steps."""
    orders = [np.nan]
    for coarse, fine in zip(euler_results[:-1], euler_results[1:]):
        orders.append(np.log(coarse['error'] / fine['error']) / np.log(2.0))
    return orders

def boundary_drift_checks():
    """Check that the continuous benchmark drift points into [0, 1]."""
    prices = np.linspace(0.0, 1.0, 101)
    lower_drift = []
    upper_drift = []
    inputs = input_builder()
    B = inputs.energy((np.array(T, dtype=float), np.array([BASELINE, BASELINE], dtype=float)))
    for price in prices:
        U = inputs.price_intervals((np.array(T, dtype=float), np.array([price], dtype=float)))
        model = bm_1(B, B, T, 2)
        parameters = [U, linear_state_response, linear_price_response, C, K, XI]
        lower_drift.append(float(model._drift(T[0], np.array([0.0]), parameters)[0]))
        upper_drift.append(float(model._drift(T[0], np.array([1.0]), parameters)[0]))
    return min(lower_drift) >= -STATE_TOL and max(upper_drift) <= STATE_TOL

def sanity_checks(euler_results):
    """Run simple model checks that are independent of solver accuracy."""
    equilibrium = constant_price_run(0.5)
    no_flexibility = constant_price_run(0.0, xi=0.0)
    low_price = constant_price_run(0.25)
    high_price = constant_price_run(0.75)
    equilibrium_state = equilibrium[1]
    no_flexibility_state = no_flexibility[1]
    low_price_state = low_price[1]
    high_price_state = high_price[1]
    all_states = np.concatenate([result['state'] for result in euler_results])
    max_energy_error = max((result['energy_error'] for result in euler_results))
    return {'equilibrium remains fixed': np.max(np.abs(equilibrium_state - X0)) <= STATE_TOL,
        'Xi = 0 gives D = B and fixed X': np.max(np.abs(no_flexibility_state - X0)) <= STATE_TOL and np.max(np.abs(no_flexibility[7] - no_flexibility[3])) <= ENERGY_TOL,
        'low price increases X toward 0.75': low_price_state[-1] > X0 and np.all(np.diff(low_price_state) >= -STATE_TOL) and (low_price_state[-1] < 0.75),
        'high price decreases X toward 0.25': high_price_state[-1] < X0 and np.all(np.diff(high_price_state) <= STATE_TOL) and (high_price_state[-1] > 0.25),
        'continuous drift is inward at both boundaries': boundary_drift_checks(),
        'Euler states stay in [0, 1] without correction': np.min(all_states) >= -STATE_TOL and np.max(all_states) <= 1.0 + STATE_TOL,
        'Euler energy identity': max_energy_error <= ENERGY_TOL}

def price_parameterization_checks(U, output_t):
    """Check the F1-C one-price-per-interval representation."""
    legacy_U = input_builder.price((PRICE_INTERVAL_EDGES,
        np.array([PRICE_BEFORE, PRICE_AFTER, PRICE_AFTER], dtype=float)))
    step_left = np.nextafter(PRICE_STEP_TIME, T[0])
    step_right = np.nextafter(PRICE_STEP_TIME, T[1])
    probe_t = np.array([T[0], step_left, PRICE_STEP_TIME, step_right, T[1]], dtype=float)
    expected_price = np.array([PRICE_BEFORE, PRICE_BEFORE, PRICE_AFTER, PRICE_AFTER, PRICE_AFTER], dtype=float)
    comparison_t = np.linspace(T[0], T[1], 1001)
    return {'N price intervals use exactly N decision values': PRICE_INTERVAL_EDGES.size == PRICE_INTERVAL_VALUES.size + 1,
        'interval price is right-continuous through the terminal edge': np.array_equal(np.asarray(U(probe_t), dtype=float), expected_price),
        'interval price matches the legacy benchmark signal': np.array_equal(np.asarray(U(comparison_t), dtype=float), np.asarray(legacy_U(comparison_t), dtype=float)),
        'price decisions are independent of the output grid': PRICE_INTERVAL_VALUES.size != output_t.size and np.asarray(U(output_t), dtype=float).shape == output_t.shape}

def deterministic_result_is_consistent(result, expected_t, model, parameters):
    """Check the deterministic solver's unchanged eight-array interface."""
    if len(result) != 8:
        return False
    arrays = [np.asarray(values, dtype=float) for values in result]
    if any((values.shape != expected_t.shape for values in arrays)):
        return False
    if any((not np.all(np.isfinite(values)) for values in arrays)):
        return False
    if not np.array_equal(arrays[0], expected_t):
        return False
    recomputed = model.eval(arrays[0], arrays[1], parameters[0], parameters[1], parameters[2], parameters[4],
        parameters[5])
    return all((np.allclose(values, expected, rtol=1e-12, atol=1e-12) for values,
        expected in zip(arrays[2:], recomputed)))
KNOWN_PRICES = np.array([0.72, 0.28], dtype=float)
INITIAL_PRICES = np.array([0.55, 0.45], dtype=float)
RTOL = 1e-09
ATOL = 1e-11
TARGET_TOL = 1e-16
OBJECTIVE_TOL = 1e-12
PRICE_TOL = 1e-05

def build_baseline():
    """Return the constant normalized baseline used by the benchmark."""
    return input_builder.energy((np.array(T, dtype=float), np.array([BASELINE, BASELINE], dtype=float)))

def build_parameters(prices):
    """Build bm_1 parameters for one price value per interval."""
    price = input_builder.price_intervals((INTERVAL_EDGES, prices))
    return [price, linear_state_response, linear_price_response, C, K, XI]

def solve_prices(model, prices):
    """Solve bm_1 for a candidate price vector on the objective grid."""
    return model.deterministic_solver([X0], build_parameters(prices), t_eval=OBJECTIVE_T,
        breakpoints=INTERVAL_EDGES[1:-1], rtol=RTOL, atol=ATOL)

def raises_value_error(action):
    """Return True when an input-validation probe raises ValueError."""
    try:
        action()
    except ValueError:
        return True
    return False
FINE_RTOL = 1e-10
FINE_ATOL = 1e-12
REQUIRED_PRICE = -0.05
PRICE_STARTS = (np.array([0.0]), np.array([0.05]), np.array([0.25]), np.array([0.5]), np.array([0.95]))
CAPACITY_TARGET = 0.82

def as_scalar_if_needed(query, values):
    """Preserve scalar calls while supporting arbitrary NumPy query shapes."""
    return float(values) if np.asarray(query).ndim == 0 else values

def feasibility_build_baseline():
    """Return the constant normalized baseline used by the benchmark."""
    return input_builder.energy((np.array(T, dtype=float), np.array([BASELINE, BASELINE], dtype=float)))

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
    return model.deterministic_solver([X0], feasibility_build_parameters(price), t_eval=t_eval,
        breakpoints=interval_edges[1:-1], rtol=rtol, atol=atol, max_step=max_step)

def optimize_from_starts(model, interval_edges, starts):
    """Run identical bounded optimizations from several starting prices."""
    placeholder = input_builder.price_intervals((np.array(T, dtype=float), np.array([0.5], dtype=float)))
    parameters = feasibility_build_parameters(placeholder)
    results = []
    for initial_prices in starts:
        results.append(model.optimize(initial_prices, [X0], parameters, interval_edges=interval_edges,
            t_eval=FEASIBILITY_OBJECTIVE_T, rtol=RTOL, atol=ATOL,
            options={'ftol': 1e-14, 'gtol': 1e-08, 'maxiter': 300}))
    return results

def best_result(results):
    """Return the successful result with the smallest objective."""
    successful = [result for result in results if result.success]
    if not successful:
        raise RuntimeError('Every F1-E optimization start failed.')
    return min(successful, key=lambda result: result.fun)
PRICE_INTERVAL_EDGES = np.array([T[0], PRICE_STEP_TIME, T[1]], dtype=float)
PRICE_INTERVAL_VALUES = np.array([PRICE_BEFORE, PRICE_AFTER], dtype=float)

def constant_price_run(price, initial_state=X0, xi=XI, steps_per_day=192):
    """Run a small deterministic Euler case used by the sanity checks."""
    inputs = input_builder()
    B = inputs.energy((np.array(T, dtype=float), np.array([BASELINE, BASELINE], dtype=float)))
    U = inputs.price_intervals((np.array(T, dtype=float), np.array([price], dtype=float)))
    model = bm_1(B, B, T, steps_per_day + 1)
    parameters = [U, linear_state_response, linear_price_response, C, K, xi]
    return model.euler_maruyama_solver([initial_state], parameters, sigma_x=0.0, n_paths=1, seed=0, boundary='none')
INTERVAL_EDGES = np.array([T[0], 0.5, T[1]], dtype=float)
OBJECTIVE_T = np.linspace(T[0], T[1], 193)
FEASIBILITY_OBJECTIVE_T = np.linspace(T[0], T[1], 385)
FINE_T = np.linspace(T[0], T[1], 1537)
PRICE_EDGES = np.array(T, dtype=float)
CAPACITY_EDGES = np.linspace(T[0], T[1], 5)
INTERNAL_PRICE_BREAKPOINTS = PRICE_INTERVAL_EDGES[1:-1]
CAPACITY_STARTS = tuple((np.full(CAPACITY_EDGES.size - 1, value) for value in (0.05, 0.25, 0.5, 0.75))) + (np.array([0.05, 0.95, 0.05, 0.95]),
    np.array([0.95, 0.05, 0.95, 0.05]))
