"""BM1 executable: input selection, reporting and plots."""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.diagnostics.bm_1 import (constant_demand_capacity_certificate, constant_demand_implied_state,
    linear_response_demand_envelope, linear_response_required_price)
from src.method.bm_1_control import beta_initial_cell_mass
from src.method.euler_maruyama import euler_maruyama
from src.method.objective import tracking_error
from src.model.bm_1 import bm_1
from src.validation.bm_1.cases import (MAX_STEP, PRICE_BREAKPOINTS, PRICE_DROP_HOUR, PRICE_DROP_TIME, SIGMA_X, T_EVAL,
    X_STEPS, build_reference_case, X0 as cases_X0, linear_price_response, linear_state_response)
from src.validation.bm_1.control import run_benchmark as control_run_benchmark
from src.validation.bm_1.deterministic import (ATOL, BASELINE, C, CAPACITY_EDGES, CAPACITY_STARTS, CAPACITY_TARGET,
    CONTROLLED_IVP_TOL, DEFAULT_BREAKPOINT_TOL, FINE_ATOL, FINE_RTOL, FINE_T, IMPROVEMENT_FACTOR,
    INTERNAL_PRICE_BREAKPOINTS, IVP_ATOL, IVP_OUTPUT_POINTS, IVP_RTOL, K, FEASIBILITY_OBJECTIVE_T as OBJECTIVE_T,
    PRICE_AFTER, PRICE_EDGES, PRICE_STARTS, PRICE_STEP_TIME, REQUIRED_PRICE, RTOL, STEPS_PER_DAY, T, X0, XI,
    best_result, feasibility_build_baseline as build_baseline, build_inputs, capacity_target,
    deterministic_result_is_consistent, exact_state, observed_orders, optimize_from_starts, price_bound_target,
    price_bound_target_state, price_parameterization_checks, run_euler, sanity_checks,
    feasibility_solve_prices as solve_prices)
from src.validation.bm_1.mixture import (_NEGATIVE_PREDICTION_TOL, run_conditioning_study, run_dynamic_mixture_study,
    run_objective_reference, run_projection_study)
from src.validation.bm_1.stochastic import (EM_REF_N_PATHS as N_PATHS, SEED, _require_no_boundary_violations,
    bm1_self_convergence, exact_logistic_benchmark, run_comparison)

def save_figure(legacy_ivp_result, ivp_result, finest_euler, B, U):
    """Save the benchmark state, inputs, equilibria, and demand plot."""
    legacy_t, legacy_state = (legacy_ivp_result[0], legacy_ivp_result[1])
    t_ivp, state_ivp = (ivp_result[0], ivp_result[1])
    plot_t = np.linspace(T[0], T[1], 1001)
    plot_price = np.asarray(U(plot_t), dtype=float)
    plot_equilibrium = 1.0 - plot_price
    plot_exact_state = exact_state(plot_t)
    model = bm_1(B, B, T, len(plot_t))
    _, plot_baseline, _, _, _, plot_demand = model.eval(plot_t, plot_exact_state, U, linear_state_response,
        linear_price_response, K, XI)
    hours = 24.0 * plot_t
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(hours, plot_exact_state, linewidth=2.5, label='Exact X(t)')
    axes[0].plot(24.0 * legacy_t, legacy_state, linestyle='--', label='Legacy solve_ivp')
    axes[0].plot(24.0 * t_ivp, state_ivp, linestyle='-.', label='Breakpoint-aware solve_ivp (tight tolerances)')
    axes[0].plot(24.0 * finest_euler['t'], finest_euler['state'], linestyle=':', marker='.', markersize=3,
        label=f"Euler, dt = {24.0 * finest_euler['dt'] * 60.0:.1f} min")
    axes[0].set_ylabel('State X')
    axes[0].set_title('bm_1 deterministic F1 reference')
    axes[0].grid(True)
    axes[0].legend(loc='best')
    axes[1].step(hours, plot_price, where='post', label='Price U(t)')
    axes[1].step(hours, plot_equilibrium, where='post', label='Equilibrium X*(t) = 1 - U(t)')
    axes[1].set_ylabel('Price / state')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True)
    axes[1].legend(loc='best')
    axes[2].plot(hours, plot_demand, label='Exact demand D(t)')
    axes[2].plot(hours, plot_baseline, linestyle='--', label='Baseline B(t)')
    axes[2].set_xlabel('Time [hours]')
    axes[2].set_ylabel('Normalized power')
    axes[2].grid(True)
    axes[2].legend(loc='best')
    for axis in axes:
        axis.axvline(24.0 * PRICE_STEP_TIME, color='black', alpha=0.4, linestyle='--')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'bm_1' / 'runs' / 'graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'bm_1_deterministic_reference.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return graph_path

def run_deterministic():
    """Run the full deterministic F1 reference experiment."""
    B, U = build_inputs()
    parameters = [U, linear_state_response, linear_price_response, C, K, XI]
    model = bm_1(B, B, T, STEPS_PER_DAY[-1] + 1)
    legacy_ivp_result = model.deterministic_solver([X0], parameters)
    legacy_t, legacy_state = (legacy_ivp_result[0], legacy_ivp_result[1])
    legacy_difference = np.abs(legacy_state - exact_state(legacy_t))
    legacy_error = float(np.max(legacy_difference))
    legacy_error_time = float(legacy_t[np.argmax(legacy_difference)])
    breakpoint_ivp_result = model.deterministic_solver([X0], parameters, breakpoints=INTERNAL_PRICE_BREAKPOINTS)
    breakpoint_t = breakpoint_ivp_result[0]
    breakpoint_state = breakpoint_ivp_result[1]
    breakpoint_difference = np.abs(breakpoint_state - exact_state(breakpoint_t))
    breakpoint_error = float(np.max(breakpoint_difference))
    breakpoint_error_time = float(breakpoint_t[np.argmax(breakpoint_difference)])
    improvement = legacy_error / breakpoint_error if breakpoint_error > 0.0 else np.inf
    ivp_t_eval = np.linspace(T[0], T[1], IVP_OUTPUT_POINTS)
    ivp_result = model.deterministic_solver([X0], parameters, t_eval=ivp_t_eval,
        breakpoints=INTERNAL_PRICE_BREAKPOINTS, rtol=IVP_RTOL, atol=IVP_ATOL)
    ivp_t, ivp_state = (ivp_result[0], ivp_result[1])
    ivp_difference = np.abs(ivp_state - exact_state(ivp_t))
    ivp_error = float(np.max(ivp_difference))
    ivp_error_time = float(ivp_t[np.argmax(ivp_difference)])
    breakpoint_indices = np.flatnonzero(np.isclose(breakpoint_t, PRICE_STEP_TIME, rtol=0.0, atol=1e-14))
    right_continuous_output = breakpoint_indices.size == 1 and np.isclose(breakpoint_ivp_result[2][breakpoint_indices[0]],
        PRICE_AFTER, rtol=0.0, atol=1e-14)
    euler_results = [run_euler(steps, B, U) for steps in STEPS_PER_DAY]
    orders = observed_orders(euler_results)
    checks = sanity_checks(euler_results)
    checks.update(price_parameterization_checks(U, ivp_t_eval))
    checks.update({'legacy deterministic API keeps the model grid': deterministic_result_is_consistent(legacy_ivp_result, model.t, model, parameters),
        'breakpoint call keeps the default output grid': deterministic_result_is_consistent(breakpoint_ivp_result, model.t, model, parameters),
        'custom deterministic output grid is honored': deterministic_result_is_consistent(ivp_result, ivp_t_eval, model, parameters),
        'default-tolerance breakpoint solve_ivp reaches target accuracy': breakpoint_error <= DEFAULT_BREAKPOINT_TOL,
        'breakpoint-aware solve_ivp reaches target accuracy': ivp_error <= CONTROLLED_IVP_TOL,
        'breakpoint handling improves the legacy error': improvement >= IMPROVEMENT_FACTOR,
        'breakpoint output uses the right-continuous price': right_continuous_output})
    graph_path = save_figure(legacy_ivp_result, ivp_result, euler_results[-1], B, U)
    checks['reference figure is saved'] = graph_path.is_file() and graph_path.stat().st_size > 0
    print('bm_1 deterministic F1 verification')
    print('==================================')
    print()
    print('Exact vs solve_ivp')
    print(f'legacy max error          = {legacy_error:.6e}')
    print(f'legacy largest error at t = {legacy_error_time:.6f} days')
    print(f'breakpoint max error      = {breakpoint_error:.6e}')
    print(f'breakpoint error at t     = {breakpoint_error_time:.6f} days')
    print(f'same-tolerance improvement = {improvement:.3e}')
    print(f'high-accuracy max error   = {ivp_error:.6e}')
    print(f'high-accuracy error at t  = {ivp_error_time:.6f} days')
    if breakpoint_error <= DEFAULT_BREAKPOINT_TOL:
        print(f'PASS default breakpoint error <= {DEFAULT_BREAKPOINT_TOL:.1e}')
    else:
        print(f'FAIL default breakpoint error > {DEFAULT_BREAKPOINT_TOL:.1e}')
    if ivp_error <= CONTROLLED_IVP_TOL:
        print(f'PASS controlled error <= {CONTROLLED_IVP_TOL:.1e}')
    else:
        print(f'FAIL controlled error > {CONTROLLED_IVP_TOL:.1e}')
    print()
    print('Euler convergence')
    print('dt [days]    dt [min]    max X error    observed order')
    for result, order in zip(euler_results, orders):
        order_text = '-' if np.isnan(order) else f'{order:.6f}'
        minutes = 24.0 * 60.0 * result['dt']
        print(f"{result['dt']:<12.8f} {minutes:<11.3f} {result['error']:<14.6e} {order_text}")
    finest = euler_results[-1]
    print()
    print('Finest Euler grid')
    print(f"max |energy residual| = {finest['energy_error']:.6e}")
    print(f"state range           = [{np.min(finest['state']):.6f}, {np.max(finest['state']):.6f}]")
    print(f"demand range          = [{np.min(finest['demand']):.6f}, {np.max(finest['demand']):.6f}]")
    print()
    print('Sanity checks')
    for name, passed in checks.items():
        print(f"{('PASS' if passed else 'FAIL')} {name}")
    print()
    print(f'Saved graph to {graph_path}')
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print()
        print('One or more F1-B checks failed; inspect these before continuing:')
        for name in failed:
            print(f'- {name}')
        raise SystemExit(1)

def feasibility_save_figure(price_result, price_state, price_demand, required_price, capacity_state, capacity_demand,
    capacity_required_state):
    """Save target, feasibility-certificate, and best-achieved trajectories."""
    hours = 24.0 * OBJECTIVE_T
    price_target = np.asarray(price_bound_target(OBJECTIVE_T), dtype=float)
    price_target_state = np.asarray(price_bound_target_state(OBJECTIVE_T), dtype=float)
    capacity_values = np.asarray(capacity_target(OBJECTIVE_T), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes[0, 0].plot(hours, price_target, linewidth=2.5, label='Target A(t)')
    axes[0, 0].plot(hours, price_demand, linestyle='--', label='Best admissible demand')
    axes[0, 0].set_title('Price-bound certificate')
    axes[0, 0].set_ylabel('Normalized demand')
    axes[0, 0].grid(True)
    axes[0, 0].legend(loc='best')
    axes[0, 1].plot(hours, required_price, linewidth=2.5, label='Required held price')
    axes[0, 1].step(hours, np.full_like(OBJECTIVE_T, price_result.x[0]), where='post', linestyle='--',
        label='Best admissible price')
    axes[0, 1].axhspan(0.0, 1.0, color='green', alpha=0.08)
    axes[0, 1].axhline(0.0, color='black', linewidth=1.0)
    axes[0, 1].set_ylim(-0.1, 0.25)
    axes[0, 1].set_title('Required price violates U >= 0')
    axes[0, 1].set_ylabel('Normalized price')
    axes[0, 1].grid(True)
    axes[0, 1].legend(loc='best')
    axes[1, 0].plot(hours, price_target_state, linewidth=2.5, label='Target-implied state')
    axes[1, 0].plot(hours, price_state, linestyle='--', label='Best admissible state')
    axes[1, 0].axhspan(0.0, 1.0, color='green', alpha=0.08)
    axes[1, 0].set_title('Price target respects state capacity')
    axes[1, 0].set_xlabel('Time [hours]')
    axes[1, 0].set_ylabel('State X')
    axes[1, 0].grid(True)
    axes[1, 0].legend(loc='best')
    axes[1, 1].plot(hours, capacity_required_state, linewidth=2.5, label='Target-implied state')
    axes[1, 1].plot(hours, capacity_state, linestyle='--', label='Best admissible state')
    axes[1, 1].axhspan(0.0, 1.0, color='green', alpha=0.08)
    axes[1, 1].axhline(1.0, color='black', linewidth=1.0)
    axes[1,
        1].set_title(f'Capacity target A(t) = {CAPACITY_TARGET:.2f}, J = {tracking_error(OBJECTIVE_T, capacity_demand, capacity_values):.3e}')
    axes[1, 1].set_xlabel('Time [hours]')
    axes[1, 1].set_ylabel('State X')
    axes[1, 1].grid(True)
    axes[1, 1].legend(loc='best')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'bm_1' / 'runs' / 'graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'bm_1_optimization_feasibility_reference.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return graph_path

def run_feasibility():
    """Run analytical feasibility diagnostics before bounded optimization."""
    baseline = build_baseline()
    target_state = np.asarray(price_bound_target_state(OBJECTIVE_T), dtype=float)
    target_demand = np.asarray(price_bound_target(OBJECTIVE_T), dtype=float)
    required_price = linear_response_required_price(target_state, target_demand, BASELINE, XI, K)
    _, maximum_demand = linear_response_demand_envelope(target_state, BASELINE, XI, K)
    envelope_gap = target_demand - maximum_demand
    price_model = bm_1(baseline, price_bound_target, T, OBJECTIVE_T.size)
    price_results = optimize_from_starts(price_model, PRICE_EDGES, PRICE_STARTS)
    price_best = best_result(price_results)
    price_solution = solve_prices(price_model, PRICE_EDGES, price_best.x, OBJECTIVE_T, RTOL, ATOL)
    price_objective = tracking_error(OBJECTIVE_T, price_solution[7], target_demand)
    fine_solution = solve_prices(price_model, PRICE_EDGES, price_best.x, FINE_T, FINE_RTOL, FINE_ATOL,
        max_step=(T[1] - T[0]) / (FINE_T.size - 1))
    fine_objective = tracking_error(FINE_T, fine_solution[7], np.asarray(price_bound_target(FINE_T), dtype=float))
    initial_solution = solve_prices(price_model, PRICE_EDGES, np.array([0.5]), OBJECTIVE_T, RTOL, ATOL)
    initial_objective = tracking_error(OBJECTIVE_T, initial_solution[7], target_demand)
    price_vectors = np.vstack([result.x for result in price_results])
    price_objectives = np.array([result.fun for result in price_results], dtype=float)
    capacity_values = np.asarray(capacity_target(OBJECTIVE_T), dtype=float)
    capacity_required_state = np.asarray(constant_demand_implied_state(OBJECTIVE_T, CAPACITY_TARGET, BASELINE, C, X0, initial_time=T[0]),
        dtype=float)
    capacity_certificate = constant_demand_capacity_certificate(CAPACITY_TARGET, BASELINE, C, X0, T)
    capacity_model = bm_1(baseline, capacity_target, T, OBJECTIVE_T.size)
    capacity_results = optimize_from_starts(capacity_model, CAPACITY_EDGES, CAPACITY_STARTS)
    capacity_best = best_result(capacity_results)
    capacity_solution = solve_prices(capacity_model, CAPACITY_EDGES, capacity_best.x, OBJECTIVE_T, RTOL, ATOL)
    capacity_objective = tracking_error(OBJECTIVE_T, capacity_solution[7], capacity_values)
    capacity_fine_solution = solve_prices(capacity_model, CAPACITY_EDGES, capacity_best.x, FINE_T, FINE_RTOL,
        FINE_ATOL, max_step=(T[1] - T[0]) / (FINE_T.size - 1))
    capacity_fine_objective = tracking_error(FINE_T, capacity_fine_solution[7],
        np.asarray(capacity_target(FINE_T), dtype=float))
    capacity_objectives = np.array([result.fun for result in capacity_results], dtype=float)
    graph_path = feasibility_save_figure(price_best, price_solution[1], price_solution[7], required_price,
        capacity_solution[1], capacity_solution[7], capacity_required_state)
    checks = {'price-bound target demand remains normalized': np.min(target_demand) >= 0.0 and np.max(target_demand) <= 1.0,
        'price-bound target state remains in [0, 1]': np.min(target_state) >= 0.0 and np.max(target_state) <= 1.0,
        'inverse diagnostic recovers the held required price': np.allclose(required_price, REQUIRED_PRICE, rtol=0.0, atol=1e-10),
        'required price satisfies the one-interval hold restriction': np.ptp(required_price) <= 1e-10,
        'required held price violates the admissible lower bound': np.max(required_price) < 0.0,
        'target exceeds the admissible demand envelope': np.min(envelope_gap) > 0.0,
        'all price-bound multistarts converge': all((result.success for result in price_results)),
        'price-bound multistarts agree on the boundary solution': np.max(np.ptp(price_vectors, axis=0)) <= 1e-08 and np.ptp(price_objectives) <= 1e-10,
        'optimized price is active at the lower bound': price_best.x[0] <= 1e-08,
        'price-bound objective is positive and reproducible': 0.0005 < price_objective < 0.002 and np.isclose(price_best.fun, price_objective, rtol=1e-07, atol=1e-12),
        'fine-grid recomputation agrees': np.isclose(fine_objective, price_objective, rtol=1e-06, atol=1e-12),
        'bounded optimization improves the initial price': price_objective < initial_objective / 100.0,
        'best price-bound state remains in [0, 1]': np.min(price_solution[1]) >= -1e-12 and np.max(price_solution[1]) <= 1.0 + 1e-12,
        'capacity target violates the upper state bound': capacity_certificate.violated_boundary == 'upper' and capacity_certificate.required_final_state > 1.0,
        'capacity violation occurs inside the model horizon': capacity_certificate.violation_time is not None and T[0] < capacity_certificate.violation_time < T[1],
        'capacity certificate has positive excess energy': capacity_certificate.excess_energy > 0.0 and capacity_certificate.objective_lower_bound > 0.0,
        'all capacity-case multistarts converge': all((result.success for result in capacity_results)),
        'capacity-case multistarts agree on the objective': np.ptp(capacity_objectives) <= 1e-09,
        'reported and recomputed capacity objectives agree': np.isclose(capacity_best.fun, capacity_objective, rtol=1e-07, atol=1e-12),
        'capacity fine-grid recomputation agrees': np.isclose(capacity_fine_objective, capacity_objective, rtol=0.001, atol=1e-10),
        'capacity objective respects the analytical lower bound': capacity_objective >= capacity_certificate.objective_lower_bound - 1e-10,
        'best capacity-case state remains in [0, 1]': np.min(capacity_solution[1]) >= -1e-12 and np.max(capacity_solution[1]) <= 1.0 + 1e-12,
        'reference figure is saved': graph_path.is_file() and graph_path.stat().st_size > 0}
    print('bm_1 deterministic optimization F1-E feasibility verification')
    print('============================================================')
    print()
    print('Price-bound certificate')
    print(f'required held price       = {REQUIRED_PRICE:.6f}')
    print(f'target state range        = [{np.min(target_state):.6f}, {np.max(target_state):.6f}]')
    print(f'target demand range       = [{np.min(target_demand):.6f}, {np.max(target_demand):.6f}]')
    print(f'minimum envelope gap      = {np.min(envelope_gap):.6e}')
    print(f'optimized admissible price = {price_best.x}')
    print(f'initial objective (U=.5)  = {initial_objective:.6e}')
    print(f'optimized objective       = {price_objective:.6e}')
    print(f'fine-grid objective       = {fine_objective:.6e}')
    print()
    print('Capacity certificate')
    print(f'constant target demand    = {CAPACITY_TARGET:.6f}')
    print(f'required final state      = {capacity_certificate.required_final_state:.6f}')
    print(f'capacity violation time   = {24.0 * capacity_certificate.violation_time:.6f} hours')
    print(f'target excess energy      = {capacity_certificate.excess_energy:.6e}')
    print(f'objective lower bound     = {capacity_certificate.objective_lower_bound:.6e}')
    print(f'optimized prices          = {capacity_best.x}')
    print(f'optimized objective       = {capacity_objective:.6e}')
    print(f'fine-grid objective       = {capacity_fine_objective:.6e}')
    print()
    print('Checks')
    for name, passed in checks.items():
        print(f"{('PASS' if passed else 'FAIL')} {name}")
    print()
    print(f'Saved graph to {graph_path}')
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print()
        print('One or more F1-E checks failed:')
        for name in failed:
            print(f'- {name}')
        raise SystemExit(1)

def _print_exact_results(step_sizes, errors, local_orders, fitted_order):
    print('Exact logistic-diffusion benchmark')
    print('----------------------------------')
    print('step [min]    RMS strong error    local order')
    for index, (step, error) in enumerate(zip(step_sizes, errors)):
        order = '-' if index == 0 else f'{local_orders[index - 1]:.4f}'
        print(f'{24.0 * 60.0 * step:10.4f}    {error:16.8e}    {order:>11}')
    print(f'Log-log fitted strong order: {fitted_order:.4f}')
    print('Expected Euler-Maruyama strong order: 0.5')
    print()

def _print_bm1_results(step_sizes, errors, local_orders, fitted_order):
    print('BM1 coupled self-convergence')
    print('----------------------------')
    print('coarse step [min]    RMS(X^h - X^(h/2))    local order')
    for index, (step, error) in enumerate(zip(step_sizes, errors)):
        order = '-' if index == 0 else f'{local_orders[index - 1]:.4f}'
        print(f'{24.0 * 60.0 * step:17.4f}    {error:20.8e}    {order:>11}')
    print(f'Log-log fitted self-convergence order: {fitted_order:.4f}')
    print('This is a self-convergence estimate, not an error against an exact BM1 solution.')
    print()

def _plot_results(exact_step_sizes, exact_errors, bm_step_sizes, bm_errors):
    fig, (ax_exact, ax_bm1) = plt.subplots(nrows=2, figsize=(9, 9))
    exact_minutes = 24.0 * 60.0 * exact_step_sizes
    exact_guide = exact_errors[0] * np.sqrt(exact_step_sizes / exact_step_sizes[0])
    ax_exact.loglog(exact_minutes, exact_errors, 'o-', label='EM RMS strong error')
    ax_exact.loglog(exact_minutes, exact_guide, '--', label='$C h^{1/2}$')
    ax_exact.set_xlabel('Step size [min]')
    ax_exact.set_ylabel('Terminal RMS error')
    ax_exact.set_title('Exact logistic-diffusion benchmark')
    ax_exact.grid(True)
    ax_exact.legend(loc='best')
    bm_minutes = 24.0 * 60.0 * bm_step_sizes
    bm_guide = bm_errors[0] * np.sqrt(bm_step_sizes / bm_step_sizes[0])
    ax_bm1.loglog(bm_minutes, bm_errors, 'o-', label='RMS$(X_T^h-X_T^{h/2})$')
    ax_bm1.loglog(bm_minutes, bm_guide, '--', label='$C h^{1/2}$')
    ax_bm1.set_xlabel('Coarse step size [min]')
    ax_bm1.set_ylabel('Coupled terminal difference')
    ax_bm1.set_title('BM1 coupled self-convergence')
    ax_bm1.grid(True)
    ax_bm1.legend(loc='best')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'bm_1' / 'runs' / 'graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'bm_1_em_convergence.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return graph_path

def run_em_convergence():
    """Run the reference experiment and save its diagnostic figure."""
    method = euler_maruyama()
    exact_step_sizes, exact_errors, exact_local_orders, exact_fitted_order, exact_diagnostics = exact_logistic_benchmark(method)
    bm_step_sizes, bm_errors, bm_local_orders, bm_fitted_order, bm_diagnostics = bm1_self_convergence(method)
    _require_no_boundary_violations('Exact benchmark', exact_diagnostics)
    _require_no_boundary_violations('BM1 self-convergence', bm_diagnostics)
    if not 0.35 <= exact_fitted_order <= 0.7:
        raise RuntimeError('The exact benchmark did not exhibit the expected EM strong-order regime.')
    if not np.all(np.diff(bm_errors) < 0.0):
        raise RuntimeError('BM1 successive coupled errors did not decrease under refinement.')
    _print_exact_results(exact_step_sizes, exact_errors, exact_local_orders, exact_fitted_order)
    _print_bm1_results(bm_step_sizes, bm_errors, bm_local_orders, bm_fitted_order)
    graph_path = _plot_results(exact_step_sizes, exact_errors, bm_step_sizes, bm_errors)
    print(f'Saved graph to {graph_path}')

def run_em():
    """Run the reference experiment and save its diagnostic figure."""
    model, parameters = build_reference_case()
    integration_t, output_indices = model.euler_maruyama.build_time_grid(T_EVAL, breakpoints=PRICE_BREAKPOINTS,
        max_step=MAX_STEP)
    dW = model.euler_maruyama.brownian_increments(integration_t, n_paths=N_PATHS, seed=SEED)
    result = model.euler_maruyama_solver([cases_X0], parameters, SIGMA_X, n_paths=N_PATHS, boundary='none',
        t_eval=T_EVAL, breakpoints=PRICE_BREAKPOINTS, max_step=MAX_STEP, brownian_increments=dW,
        return_diagnostics=True)
    t, state, price, baseline, _, _, _, demand, diagnostics = result
    state_mean = np.mean(state, axis=0)
    state_var = np.var(state, axis=0, ddof=1)
    state_se = np.sqrt(state_var / N_PATHS)
    demand_mean = np.mean(demand, axis=0)
    demand_var = np.var(demand, axis=0, ddof=1)
    demand_se = np.sqrt(demand_var / N_PATHS)
    print('F2-B controlled Euler-Maruyama reference')
    print('----------------------------------------')
    print(f'Output points:                 {len(t)}')
    print(f"Internal EM steps:             {diagnostics['n_steps']}")
    print(f"Maximum internal step [h]:     {24.0 * np.max(np.diff(diagnostics['integration_t'])):.6f}")
    print(f"Price breakpoint included:     {bool(np.any(diagnostics['integration_t'] == PRICE_DROP_TIME))}")
    print(f"Raw proposal minimum:          {diagnostics['minimum_raw_proposal']:.6f}")
    print(f"Raw proposal maximum:          {diagnostics['maximum_raw_proposal']:.6f}")
    print(f"Lower-bound violations:        {diagnostics['lower_violations']}")
    print(f"Upper-bound violations:        {diagnostics['upper_violations']}")
    print(f"Paths with any violation:      {diagnostics['violating_paths']}")
    print(f"Boundary corrections applied:  {diagnostics['boundary_corrections']}")
    print()
    print(f'Terminal E[X]:                 {state_mean[-1]:.6f}')
    print(f'Terminal SE(E[X]):             {state_se[-1]:.3e}')
    print(f'Terminal E[D]:                 {demand_mean[-1]:.6f}')
    print(f'Terminal SE(E[D]):             {demand_se[-1]:.3e}')
    if diagnostics['total_violations']:
        raise RuntimeError('The uncorrected EM reference left [0, 1]. Reduce max_step or noise before using this run as a bounded-state reference.')
    hours = 24.0 * t
    fig, (ax_state, ax_paths, ax_demand) = plt.subplots(nrows=3, figsize=(11, 11))
    ax_state.step(hours, price, where='post', label='Price U(t)')
    ax_state.plot(hours, state_mean, linewidth=2.0, label='Monte Carlo E[X(t)]')
    ax_state.fill_between(hours, state_mean - 1.96 * state_se, state_mean + 1.96 * state_se, alpha=0.2,
        label='95% MC interval for E[X]')
    ax_state.axvline(PRICE_DROP_HOUR, linestyle='--', label='Price drop')
    ax_state.set_ylabel('Price / SoC')
    ax_state.set_title('Controlled Euler-Maruyama mean and sampling uncertainty')
    ax_state.grid(True)
    ax_state.legend(loc='best')
    for path in state[:30]:
        ax_paths.plot(hours, path, alpha=0.2)
    ax_paths.plot(hours, state_mean, linewidth=2.5, label='E[X(t)]')
    ax_paths.axvline(PRICE_DROP_HOUR, linestyle='--')
    ax_paths.set_ylabel('State of charge')
    ax_paths.set_title('30 uncorrected Euler-Maruyama sample paths')
    ax_paths.set_ylim(0.0, 1.0)
    ax_paths.grid(True)
    ax_paths.legend(loc='best')
    ax_demand.plot(hours, demand_mean, linewidth=2.0, label='Monte Carlo E[D(t)]')
    ax_demand.fill_between(hours, demand_mean - 1.96 * demand_se, demand_mean + 1.96 * demand_se, alpha=0.2,
        label='95% MC interval for E[D]')
    ax_demand.plot(hours, baseline, linestyle='--', label='Baseline B(t)')
    ax_demand.axvline(PRICE_DROP_HOUR, linestyle='--')
    ax_demand.set_xlabel('Time [h]')
    ax_demand.set_ylabel('Normalized demand')
    ax_demand.set_title('Expected demand and Monte Carlo sampling uncertainty')
    ax_demand.grid(True)
    ax_demand.legend(loc='best')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'bm_1' / 'runs' / 'graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'bm_1_em_reference.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    np.testing.assert_array_equal(t, T_EVAL)
    np.testing.assert_array_equal(diagnostics['integration_t'][output_indices], T_EVAL)
    print(f'Saved graph to {graph_path}')

def run_fv():
    """Run the reference experiment and save its diagnostic figure."""
    model, parameters = build_reference_case()
    x_edges, _, _ = model.fokker_planc_reference.grid(X_STEPS)
    initial_mass = beta_initial_cell_mass(x_edges)
    t, x, density, cell_mass, mean, variance, price, baseline, expected_demand, diagnostics = model.fokker_planc_reference_solver(initial_mass,
        parameters, SIGMA_X, t_eval=T_EVAL, breakpoints=PRICE_BREAKPOINTS, max_step=MAX_STEP)
    mass = np.sum(cell_mass, axis=1)
    std = np.sqrt(variance)
    hours = 24.0 * t
    print('F2-C pure finite-volume Fokker-Planck reference')
    print('-----------------------------------------------')
    print(f'Spatial cells:                  {X_STEPS}')
    print(f'Output points:                  {len(t)}')
    print(f"Internal implicit-Euler steps:  {diagnostics['n_steps']}")
    print(f"Maximum internal step [h]:      {24.0 * np.max(np.diff(diagnostics['integration_t'])):.6f}")
    print(f"Price breakpoint included:      {bool(np.any(diagnostics['integration_t'] == PRICE_DROP_TIME))}")
    print(f'Maximum |mass - 1|:             {np.max(np.abs(mass - 1.0)):.3e}')
    print(f'Minimum cell probability:       {np.min(cell_mass):.3e}')
    print(f'Minimum density value:          {np.min(density):.3e}')
    print(f"Max generator column-sum error: {diagnostics['maximum_abs_generator_column_sum']:.3e}")
    print(f"Minimum off-diagonal rate:      {diagnostics['minimum_off_diagonal_rate']:.3e}")
    print(f"Clipping used:                  {diagnostics['used_clipping']}")
    print(f"Renormalization used:           {diagnostics['used_renormalization']}")
    print()
    print(f'Initial E[X]:                   {mean[0]:.6f}')
    print(f'Terminal E[X]:                  {mean[-1]:.6f}')
    print(f'Terminal std[X]:                {std[-1]:.6f}')
    print(f'Terminal E[D]:                  {expected_demand[-1]:.6f}')
    if diagnostics['used_clipping'] or diagnostics['used_renormalization']:
        raise RuntimeError('The F2-C reference must not use post-correction.')
    if diagnostics['minimum_cell_mass'] < -1e-12:
        raise RuntimeError('The F2-C reference produced a negative cell probability.')
    if diagnostics['maximum_abs_mass_error'] > 1e-10:
        raise RuntimeError('The F2-C reference failed to conserve total probability.')
    fig, (ax_pdf, ax_state, ax_terminal) = plt.subplots(nrows=3, figsize=(11, 11),
        gridspec_kw={'height_ratios': [2.0, 1.0, 1.0]})
    normalized_density = density / np.max(density)
    mesh = ax_pdf.pcolormesh(hours, x, normalized_density.T, shading='auto', vmin=0.0, vmax=1.0)
    ax_pdf.plot(hours, mean, linewidth=2.0, label='Finite-volume E[X(t)]')
    ax_pdf.axvline(PRICE_DROP_HOUR, linestyle='--', label='Price drop')
    ax_pdf.set_ylabel('State of charge x')
    ax_pdf.set_title('Pure finite-volume Fokker-Planck density')
    ax_pdf.set_ylim(0.0, 1.0)
    ax_pdf.legend(loc='best')
    fig.colorbar(mesh, ax=ax_pdf, label='Normalized probability density')
    ax_state.step(hours, price, where='post', label='Price U(t)')
    ax_state.plot(hours, mean, linewidth=2.0, label='E[X(t)]')
    ax_state.fill_between(hours, np.clip(mean - std, 0.0, 1.0), np.clip(mean + std, 0.0, 1.0), alpha=0.2,
        label='E[X] +/- std')
    ax_state.axvline(PRICE_DROP_HOUR, linestyle='--')
    ax_state.set_ylabel('Price / SoC')
    ax_state.set_ylim(0.0, 1.0)
    ax_state.set_title('Price and finite-volume state moments')
    ax_state.grid(True)
    ax_state.legend(loc='best')
    ax_terminal.plot(x, density[0], label='Initial density')
    ax_terminal.plot(x, density[-1], linewidth=2.0, label='Terminal density')
    ax_terminal.axvline(mean[-1], linestyle='--', label='Terminal E[X]')
    ax_terminal.set_xlabel('State of charge x')
    ax_terminal.set_ylabel('Probability density')
    ax_terminal.set_title('Initial and terminal finite-volume density')
    ax_terminal.set_xlim(0.0, 1.0)
    ax_terminal.grid(True)
    ax_terminal.legend(loc='best')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'bm_1' / 'runs' / 'graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'bm_1_fp_reference.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved graph to {graph_path}')

def run_mc_fv():
    result = run_comparison()
    print('F2-D Monte Carlo versus finite-volume reference')
    print('-----------------------------------------------')
    print(f"Monte Carlo paths:                {result['n_paths']}")
    print(f"Spatial cells:                    {len(result['x'])}")
    print(f"EM raw boundary violations:       {result['em_diagnostics']['total_violations']}")
    print(f"FV maximum |mass - 1|:            {result['fp_diagnostics']['maximum_abs_mass_error']:.3e}")
    print()
    print(f"Maximum |E_MC[X]-E_FV[X]|:        {np.max(np.abs(result['mc_mean'] - result['fp_mean'])):.6e}")
    print(f"Terminal |E_MC[X]-E_FV[X]|:       {abs(result['mc_mean'][-1] - result['fp_mean'][-1]):.6e}")
    print(f"Maximum |Var_MC-Var_FV|:           {np.max(np.abs(result['mc_variance'] - result['fp_variance'])):.6e}")
    print(f"Maximum |E_MC[D]-E_FV[D]|:        {np.max(np.abs(result['mc_expected_demand'] - result['fp_expected_demand'])):.6e}")
    print(f"Terminal cell-mass L1 distance:    {result['l1'][-1]:.6e}")
    print(f"Terminal CDF sup distance:         {result['cdf_sup'][-1]:.6e}")
    print(f"Terminal Wasserstein-1 distance:   {result['wasserstein_1'][-1]:.6e}")
    hours = 24.0 * result['t']
    fig, axes = plt.subplots(4, 1, figsize=(11, 14))
    ax_mean, ax_demand, ax_distance, ax_terminal = axes
    ax_mean.step(hours, result['price'], where='post', label='Price U(t)')
    ax_mean.plot(hours, result['mc_mean'], label='MC E[X]')
    ax_mean.plot(hours, result['fp_mean'], linestyle='--', label='FV E[X]')
    ax_mean.fill_between(hours, result['mc_mean'] - 1.96 * result['mc_mean_se'],
        result['mc_mean'] + 1.96 * result['mc_mean_se'], alpha=0.2, label='95% MC interval for E[X]')
    ax_mean.axvline(PRICE_DROP_HOUR, linestyle=':')
    ax_mean.set_ylabel('Price / state')
    ax_mean.grid(True)
    ax_mean.legend(loc='best')
    ax_demand.plot(hours, result['mc_expected_demand'], label='MC E[D]')
    ax_demand.plot(hours, result['fp_expected_demand'], linestyle='--', label='FV E[D]')
    ax_demand.fill_between(hours, result['mc_expected_demand'] - 1.96 * result['mc_demand_se'],
        result['mc_expected_demand'] + 1.96 * result['mc_demand_se'], alpha=0.2, label='95% MC interval for E[D]')
    ax_demand.plot(hours, result['baseline'], linestyle=':', label='B(t)')
    ax_demand.set_ylabel('Demand')
    ax_demand.grid(True)
    ax_demand.legend(loc='best')
    ax_distance.plot(hours, result['l1'], label='Cell-mass L1')
    ax_distance.plot(hours, result['cdf_sup'], label='CDF sup')
    ax_distance.plot(hours, result['wasserstein_1'], label='Wasserstein-1')
    ax_distance.set_ylabel('Distribution distance')
    ax_distance.grid(True)
    ax_distance.legend(loc='best')
    mc_terminal_density = result['mc_cell_mass'][-1] / result['dx']
    ax_terminal.plot(result['x'], mc_terminal_density, label='MC histogram density')
    ax_terminal.plot(result['x'], result['fp_density'][-1], linestyle='--', label='FV density')
    ax_terminal.set_xlabel('State of charge x')
    ax_terminal.set_ylabel('Probability density')
    ax_terminal.grid(True)
    ax_terminal.legend(loc='best')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'bm_1' / 'runs' / 'graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'bm_1_mc_vs_fp_reference.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved graph to {graph_path}')

def run_projection():
    result = run_projection_study()
    print('F2-E static Beta-mixture projection study')
    print('-----------------------------------------')
    print(f"Spatial cells: {len(result['x'])}")
    print(f"Snapshots:     {len(result['t'])}")
    print()
    print('degree    max L2       max L1       max W1       max |dE[X]|   max |dVar|    max |dE[D]|')
    for degree in result['degrees']:
        item = result['projection'][degree]
        print(f"{degree:6d}  {np.max(item['l2']):.3e}  {np.max(item['l1']):.3e}  {np.max(item['wasserstein_1']):.3e}  {np.max(np.abs(item['mean'] - result['fp_mean'])):.3e}  {np.max(np.abs(item['variance'] - result['fp_variance'])):.3e}  {np.max(np.abs(item['expected_demand'] - result['fp_expected_demand'])):.3e}")
    print()
    print('Initial-law variance representation')
    print('-----------------------------------')
    print(f"FV initial variance: {result['fp_variance'][0]:.8e}")
    for degree in result['degrees']:
        item = result['projection'][degree]
        print(f"M={degree:2d}: floor={item['variance_floor'][0]:.8e}, projected={item['variance'][0]:.8e}, L1={item['l1'][0]:.8e}")
    for degree in result['degrees']:
        item = result['projection'][degree]
        if item['maximum_coefficient_sum_error'] > 1e-09:
            raise RuntimeError('Beta-mixture coefficients failed unit-sum constraint.')
        if item['minimum_coefficient'] < -1e-09:
            raise RuntimeError('Beta-mixture projection produced a materially negative coefficient.')
        if item['maximum_mass_error'] > 1e-09:
            raise RuntimeError('Projected Beta-mixture density lost probability mass.')
    hours = 24.0 * result['t']
    fig, axes = plt.subplots(4, 1, figsize=(11, 15))
    ax_degree, ax_time, ax_variance, ax_terminal = axes
    degrees = np.asarray(result['degrees'], dtype=float)
    max_l2 = np.array([np.max(result['projection'][degree]['l2']) for degree in result['degrees']])
    terminal_l2 = np.array([result['projection'][degree]['l2'][-1] for degree in result['degrees']])
    ax_degree.plot(degrees, max_l2, marker='o', label='Maximum density L2')
    ax_degree.plot(degrees, terminal_l2, marker='o', label='Terminal density L2')
    ax_degree.set_xlabel('Beta-mixture degree M')
    ax_degree.set_ylabel('Density L2 error')
    ax_degree.set_title('Pure representation error versus Beta-mixture degree')
    ax_degree.grid(True)
    ax_degree.legend(loc='best')
    for degree in result['degrees']:
        ax_time.plot(hours, result['projection'][degree]['l1'], label=f'M={degree}')
    ax_time.axvline(PRICE_DROP_HOUR, linestyle='--')
    ax_time.set_xlabel('Time [h]')
    ax_time.set_ylabel('Cell-mass L1 error')
    ax_time.set_title('Static projection error over the BM1 reference trajectory')
    ax_time.grid(True)
    ax_time.legend(loc='best')
    ax_variance.plot(degrees, [result['projection'][degree]['variance_floor'][0] for degree in result['degrees']],
        marker='o', label='Theoretical variance floor at mean 0.5')
    ax_variance.axhline(result['fp_variance'][0], linestyle='--', label='FV initial variance')
    ax_variance.set_xlabel('Beta-mixture degree M')
    ax_variance.set_ylabel('Variance')
    ax_variance.set_title('Why low-degree mixtures cannot represent the narrow initial law')
    ax_variance.grid(True)
    ax_variance.legend(loc='best')
    ax_terminal.plot(result['x'], result['fp_density'][-1], linewidth=2.5, label='Pure FV')
    for degree in result['degrees']:
        if degree in (10, 20, 38):
            ax_terminal.plot(result['x'], result['projection'][degree]['density'][-1], label=f'Projected M={degree}')
    ax_terminal.set_xlabel('State of charge x')
    ax_terminal.set_ylabel('Probability density')
    ax_terminal.set_title('Terminal pure-FV density and static Beta projections')
    ax_terminal.grid(True)
    ax_terminal.legend(loc='best')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'bm_1' / 'runs' / 'graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'bm_1_beta_projection_reference.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved graph to {graph_path}')

def run_mixture():
    result = run_dynamic_mixture_study()
    print('F2-F dynamic Beta-mixture evolution study')
    print('-----------------------------------------')
    print(f"Spatial cells: {len(result['x'])}")
    print(f"Snapshots:     {len(result['t'])}")
    print(f"CFL:           {result['cfl']:.3f}")
    print()
    print('degree   cond(H)      steps   max total L2   max static L2   max dyn-gap L2   max |dE[X]|   max |dE[D]|')
    for degree in result['degrees']:
        item = result['dynamic'][degree]
        static_item = result['projection'][degree]
        diagnostics = item['diagnostics']
        print(f"{degree:6d}  {diagnostics['hessian_condition_number']:.3e}  {diagnostics['n_steps']:6d}  {np.max(item['total_error']['l2']):.3e}      {np.max(static_item['l2']):.3e}       {np.max(item['dynamic_gap']['l2']):.3e}       {np.max(np.abs(item['mean'] - result['fp_mean'])):.3e}      {np.max(np.abs(item['expected_demand'] - result['fp_expected_demand'])):.3e}")
    print()
    print('Numerical diagnostics')
    print('---------------------')
    for degree in result['degrees']:
        diagnostics = result['dynamic'][degree]['diagnostics']
        print(f"M={degree:2d}: raw min={diagnostics['minimum_raw_prediction']:.3e}, raw mass err={diagnostics['maximum_abs_raw_prediction_mass_error']:.3e}, proj mass err={diagnostics['maximum_abs_projected_mass_error']:.3e}, min c={diagnostics['minimum_coefficient']:.3e}, max boundary c={diagnostics['maximum_boundary_coefficient']:.3e}, max step={24.0 * 60.0 * diagnostics['maximum_internal_step']:.3f} min")
        if diagnostics['used_clipping'] or diagnostics['used_renormalization']:
            raise RuntimeError('F2-F must not use clipping or renormalization.')
        if diagnostics['minimum_raw_prediction'] < -_NEGATIVE_PREDICTION_TOL:
            raise RuntimeError('F2-F produced a materially negative raw prediction.')
        if diagnostics['maximum_abs_raw_prediction_mass_error'] > 1e-09:
            raise RuntimeError('F2-F raw Euler prediction failed probability conservation.')
        if diagnostics['maximum_abs_projected_mass_error'] > 1e-09:
            raise RuntimeError('F2-F projected mixture failed probability conservation.')
        if diagnostics['minimum_coefficient'] < -1e-09:
            raise RuntimeError('F2-F produced a materially negative mixture coefficient.')
        if diagnostics['maximum_abs_coefficient_sum_error'] > 1e-09:
            raise RuntimeError('F2-F mixture coefficients failed the simplex sum constraint.')
    hours = 24.0 * result['t']
    fig, axes = plt.subplots(4, 1, figsize=(11, 15))
    ax_total, ax_gap, ax_mean, ax_terminal = axes
    for degree in result['degrees']:
        ax_total.plot(hours, result['dynamic'][degree]['total_error']['l2'], label=f'Dynamic M={degree}')
        ax_total.plot(hours, result['projection'][degree]['l2'], linestyle='--', label=f'Static best M={degree}')
    ax_total.axvline(PRICE_DROP_HOUR, linestyle=':')
    ax_total.set_ylabel('Density L2 error')
    ax_total.set_title('Dynamic mixture error versus static representation limit')
    ax_total.grid(True)
    ax_total.legend(loc='best', ncol=2)
    for degree in result['degrees']:
        ax_gap.plot(hours, result['dynamic'][degree]['dynamic_gap']['l2'], label=f'M={degree}')
    ax_gap.axvline(PRICE_DROP_HOUR, linestyle=':')
    ax_gap.set_ylabel('Density L2 distance')
    ax_gap.set_title('Additional dynamic gap to the best same-degree projection')
    ax_gap.grid(True)
    ax_gap.legend(loc='best')
    ax_mean.plot(hours, result['fp_mean'], linewidth=2.5, label='Pure FV E[X]')
    for degree in result['degrees']:
        ax_mean.plot(hours, result['dynamic'][degree]['mean'], label=f'Dynamic M={degree}')
    ax_mean.axvline(PRICE_DROP_HOUR, linestyle=':')
    ax_mean.set_ylabel('Expected state')
    ax_mean.set_title('Expected state under dynamic mixture evolution')
    ax_mean.grid(True)
    ax_mean.legend(loc='best')
    ax_terminal.plot(result['x'], result['fp_density'][-1], linewidth=2.5, label='Pure FV')
    for degree in result['degrees']:
        ax_terminal.plot(result['x'], result['dynamic'][degree]['density'][-1], label=f'Dynamic M={degree}')
    ax_terminal.set_xlabel('State of charge x')
    ax_terminal.set_ylabel('Probability density')
    ax_terminal.set_title('Terminal dynamic Beta mixtures')
    ax_terminal.grid(True)
    ax_terminal.legend(loc='best')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'bm_1' / 'runs' / 'graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'bm_1_beta_dynamic_reference.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved graph to {graph_path}')

def run_conditioning():
    result = run_conditioning_study()
    print('F2-G Beta-mixture conditioning and sensitivity')
    print('----------------------------------------------')
    print(f"Spatial cells:       {len(result['x'])}")
    print(f"Snapshots:           {len(result['t'])}")
    print(f"Directions/epsilon:  {result['n_directions']}")
    print(f"Active threshold:    {result['active_tol']:.1e}")
    print()
    print('degree   cond(A)      cond(H)      max local static   max local dynamic   free static   free dynamic')
    for degree in result['degrees']:
        item = result['conditioning'][degree]
        print(f"{degree:6d}  {item['matrix_condition_number']:.3e}  {item['hessian_condition_number']:.3e}  {np.max(item['static_local_condition_numbers']):.3e}        {np.max(item['dynamic_local_condition_numbers']):.3e}         {np.max(item['static_free_sizes']):6d}        {np.max(item['dynamic_free_sizes']):6d}")
    print()
    print('Dependence on feasible QP start')
    print('-------------------------------')
    for degree in result['degrees']:
        item = result['conditioning'][degree]['alternative_start']
        print(f"M={degree:2d}: failures={item['failures']}, max ||dc||_1={item['maximum_coefficient_l1']:.3e}, max density L2={item['maximum_density_l2']:.3e}, max |dmean|={item['maximum_mean_difference']:.3e}, max |dvar|={item['maximum_variance_difference']:.3e}")
    print()
    print('Positive mass-preserving target perturbations')
    print('---------------------------------------------')
    for degree in result['degrees']:
        print(f'M={degree}:')
        for epsilon in result['epsilons']:
            item = result['conditioning'][degree]['perturbations'][epsilon]
            print(f"  eps={epsilon:.0e}: failures={item['failures']}, input L2={item['maximum_input_l2']:.3e}, ||dc||_1={item['maximum_coefficient_l1']:.3e}, density L2={item['maximum_projected_density_l2']:.3e}, density amp={item['maximum_density_amplification']:.3e}, coeff amp={item['maximum_coefficient_amplification']:.3e}, |dmean|={item['maximum_mean_difference']:.3e}, |dvar|={item['maximum_variance_difference']:.3e}")
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    ax_condition, ax_coeff, ax_density = axes
    degrees = np.asarray(result['degrees'], dtype=float)
    ax_condition.semilogy(degrees,
        [result['conditioning'][degree]['hessian_condition_number'] for degree in result['degrees']], marker='o',
        label='Full cond(H)')
    ax_condition.semilogy(degrees,
        [np.max(result['conditioning'][degree]['dynamic_local_condition_numbers']) for degree in result['degrees']],
        marker='o', label='Max active-tangent cond')
    ax_condition.set_xlabel('Beta-mixture degree M')
    ax_condition.set_ylabel('Condition number')
    ax_condition.set_title('Full versus locally relevant conditioning')
    ax_condition.grid(True)
    ax_condition.legend(loc='best')
    for degree in result['degrees']:
        ax_coeff.loglog(result['epsilons'],
            [result['conditioning'][degree]['perturbations'][epsilon]['maximum_coefficient_l1'] for epsilon in result['epsilons']],
            marker='o', label=f'M={degree}')
    ax_coeff.set_xlabel('Log-density perturbation epsilon')
    ax_coeff.set_ylabel('Maximum coefficient L1 change')
    ax_coeff.set_title('Coefficient sensitivity')
    ax_coeff.grid(True)
    ax_coeff.legend(loc='best')
    for degree in result['degrees']:
        ax_density.loglog(result['epsilons'],
            [result['conditioning'][degree]['perturbations'][epsilon]['maximum_projected_density_l2'] for epsilon in result['epsilons']],
            marker='o', label=f'M={degree}')
    ax_density.set_xlabel('Log-density perturbation epsilon')
    ax_density.set_ylabel('Maximum projected density L2 change')
    ax_density.set_title('Density sensitivity')
    ax_density.grid(True)
    ax_density.legend(loc='best')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'bm_1' / 'runs' / 'graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'bm_1_beta_conditioning_reference.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved graph to {graph_path}')

def _relative_difference(left, right):
    scale = max(abs(float(right)), np.finfo(float).tiny)
    return abs(float(left) - float(right)) / scale

def run_objective():
    result = run_objective_reference()
    print('F2-H stochastic BM1 tracking objective')
    print('-------------------------------------')
    print(f"Monte Carlo paths:       {result['n_paths']}")
    print(f"FV cells:                {result['x_steps']}")
    print(f"Objective intervals:     {len(result['widths'])}")
    print(f"Maximum objective step:  {24.0 * 60.0 * np.max(result['widths']):.3f} min")
    print(f"Price breakpoint on edge: {bool(np.any(result['edges'] == PRICE_BREAKPOINTS[0]))}")
    print(f"EM raw violations:       {result['em_diagnostics']['total_violations']}")
    print()
    mc = result['mc']
    fv = result['fv']
    print('Objective decomposition')
    print('-----------------------')
    print(f'MC:  J={mc.objective.total:.8e}, bias={mc.objective.bias:.8e}, variance={mc.objective.variance:.8e}, SE(J)={mc.standard_error:.3e}')
    print(f'FV:  J={fv.total:.8e}, bias={fv.bias:.8e}, variance={fv.variance:.8e}')
    for degree in result['degrees']:
        item = result['mixture'][degree]
        print(f'M={degree:2d}: J={item.total:.8e}, bias={item.bias:.8e}, variance={item.variance:.8e}')
    print()
    print('Cross-method differences')
    print('------------------------')
    mc_fv_gap = abs(mc.objective.total - fv.total)
    mc_fv_se = mc_fv_gap / mc.standard_error if mc.standard_error > 0.0 else np.inf
    relative_mc_fv = _relative_difference(mc.objective.total, fv.total)
    print(f'|J_MC-J_FV| = {mc_fv_gap:.6e} ({mc_fv_se:.3f} MC standard errors, relative {relative_mc_fv:.3%})')
    for degree in result['degrees']:
        item = result['mixture'][degree]
        relative_mix_fv = _relative_difference(item.total, fv.total)
        print(f'M={degree:2d}: |J_mix-J_FV|={abs(item.total - fv.total):.6e}, relative={relative_mix_fv:.3%}')
    print()
    print('Algebra diagnostics')
    print('-------------------')
    print(f'MC direct-vs-decomposed J error: {abs(mc.direct_total - mc.objective.total):.3e}')
    print(f'MC max pointwise decomposition error: {mc.maximum_decomposition_error:.3e}')
    if mc.maximum_decomposition_error > 1e-12:
        raise RuntimeError('MC bias-variance decomposition failed.')
    if abs(mc.direct_total - mc.objective.total) > 1e-12:
        raise RuntimeError('MC direct and decomposed stochastic objectives disagree.')
    hours = 24.0 * result['midpoints']
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    ax_total, ax_parts, ax_cumulative = axes
    ax_total.step(hours, result['price'], where='mid', label='Price U(t)')
    ax_total.plot(hours, mc.objective.total_integrand, label='MC E[(D-A)^2]')
    ax_total.plot(hours, fv.total_integrand, linestyle='--', label='FV E[(D-A)^2]')
    best_degree = max(result['degrees'])
    ax_total.plot(hours, result['mixture'][best_degree].total_integrand, linestyle=':',
        label=f'Dynamic mixture M={best_degree}')
    ax_total.axvline(PRICE_DROP_HOUR, linestyle='--')
    ax_total.set_ylabel('Price / cost density')
    ax_total.set_title('Stochastic tracking-cost integrand')
    ax_total.grid(True)
    ax_total.legend(loc='best')
    ax_parts.plot(hours, fv.bias_integrand, label='FV bias-squared contribution')
    ax_parts.plot(hours, fv.variance_integrand, label='FV demand-variance contribution')
    ax_parts.axvline(PRICE_DROP_HOUR, linestyle='--')
    ax_parts.set_ylabel('Cost density')
    ax_parts.set_title('Why the stochastic objective differs from mean tracking')
    ax_parts.grid(True)
    ax_parts.legend(loc='best')
    ax_cumulative.plot(24.0 * result['edges'][1:], np.cumsum(result['widths'] * mc.objective.total_integrand),
        label='MC cumulative J')
    ax_cumulative.plot(24.0 * result['edges'][1:], np.cumsum(result['widths'] * fv.total_integrand), linestyle='--',
        label='FV cumulative J')
    ax_cumulative.plot(24.0 * result['edges'][1:],
        np.cumsum(result['widths'] * result['mixture'][best_degree].total_integrand), linestyle=':',
        label=f'Mixture M={best_degree} cumulative J')
    ax_cumulative.axvline(PRICE_DROP_HOUR, linestyle='--')
    ax_cumulative.set_xlabel('Time [h]')
    ax_cumulative.set_ylabel('Accumulated objective')
    ax_cumulative.set_title('Accumulation of the stochastic objective')
    ax_cumulative.grid(True)
    ax_cumulative.legend(loc='best')
    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / 'files' / 'output' / 'bm_1' / 'runs' / 'graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / 'bm_1_stochastic_objective_reference.png'
    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved graph to {graph_path}')

def run_benchmark():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, nargs='+', default=[201, 1001])
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[4] / 'files/output/bm_1/runs/fast_fv_benchmark.json')
    args = parser.parse_args()
    result = control_run_benchmark(args.cells, args.repeats, args.optimize, lambda s: print(s, flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Saved numerical benchmark: {args.output}')


def main():
    import sys
    commands = {'deterministic': 'run_deterministic', 'feasibility': 'run_feasibility',
        'em-convergence': 'run_em_convergence', 'em': 'run_em', 'fv': 'run_fv', 'mc-fv': 'run_mc_fv',
        'projection': 'run_projection', 'mixture': 'run_mixture', 'conditioning': 'run_conditioning',
        'objective': 'run_objective', 'benchmark': 'run_benchmark'}
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Cases: ' + ', '.join(commands))
        print('Run one case, then pass its options after the case name.')
        return
    case = sys.argv.pop(1)
    if case not in commands: raise SystemExit('Unknown case: ' + case)
    globals()[commands[case]]()


if __name__ == '__main__':
    main()
