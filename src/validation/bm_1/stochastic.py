"""F2 stochastic reference and convergence calculations, without plotting or I/O."""

import numpy as np

from src.diagnostics.bm_1 import (
    cell_mass_cdf_sup,
    cell_mass_l1,
    cell_mass_wasserstein_1,
    empirical_cell_mass,
)
from src.diagnostics.stochastic import (
    loglog_convergence_order,
    rms_strong_error,
    successive_convergence_orders,
)
from src.control.bm_1 import beta_initial_cell_mass, sample_beta_initial_states
from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder
from src.validation.bm_1.cases import (
    MAX_STEP,
    PRICE_BREAKPOINTS,
    T_EVAL,
    X_STEPS,
    build_reference_case,
    SIGMA_X as cases_SIGMA_X,
)


T = (0.0, 1.0)
X0 = 0.5
SIGMA_X = 0.5
N_PATHS = 10000
EXACT_SEED = 2026
BM1_SEED = 2027
EXACT_STEPS = np.array([24, 48, 96, 192], dtype=int)
BM1_STEPS = np.array([96, 192, 384, 768], dtype=int)
EM_REF_N_PATHS = 20000
SEED = 2026
DEFAULT_N_PATHS = 50000
INITIAL_SEED = 2028
BROWNIAN_SEED = 2029


def _logistic(values):
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def _coarsened_increments(method, fine_increments, fine_steps, target_steps):
    factor = int(fine_steps // target_steps)
    if fine_steps % target_steps != 0:
        raise ValueError('target_steps must divide the finest step count.')
    if factor == 1:
        return fine_increments
    return method.coarsen_brownian_increments(fine_increments, factor=factor)


def exact_logistic_benchmark(method):
    """Return coupled EM errors against an exact logistic-diffusion solution."""
    finest_steps = int(EXACT_STEPS[-1])
    finest_t = np.linspace(T[0], T[1], finest_steps + 1)
    fine_dW = method.brownian_increments(finest_t, n_paths=N_PATHS, seed=EXACT_SEED)
    logit_x0 = np.log(X0 / (1.0 - X0))
    exact_terminal = _logistic(logit_x0 + SIGMA_X * np.sum(fine_dW, axis=1))

    def drift(time, state):
        del time
        state = np.asarray(state, dtype=float)
        return 0.5 * SIGMA_X ** 2 * state * (1.0 - state) * (1.0 - 2.0 * state)

    def diffusion(time, state):
        del time
        state = np.asarray(state, dtype=float)
        return SIGMA_X * state * (1.0 - state)
    errors = []
    diagnostics = []
    for n_steps in EXACT_STEPS:
        t = np.linspace(T[0], T[1], int(n_steps) + 1)
        dW = _coarsened_increments(method, fine_dW, finest_steps, int(n_steps))
        paths, level_diagnostics = method.solve_paths(
            t,
            X0,
            drift,
            diffusion,
            n_paths=N_PATHS,
            boundary='none',
            brownian_increments=dW,
            return_diagnostics=True,
        )
        errors.append(rms_strong_error(paths[:, -1], exact_terminal))
        diagnostics.append(level_diagnostics)
    step_sizes = (T[1] - T[0]) / EXACT_STEPS.astype(float)
    errors = np.asarray(errors, dtype=float)
    local_orders = successive_convergence_orders(errors)
    fitted_order = loglog_convergence_order(step_sizes, errors)
    return (step_sizes, errors, local_orders, fitted_order, diagnostics)


def bm1_self_convergence(method):
    """Return coupled successive-level terminal errors for the BM1 SDE."""
    inputs = input_builder()
    B = inputs.energy((np.array(T), np.array([0.3, 0.3])))
    U = inputs.price_intervals((np.array(T), np.array([0.25])))

    def linear_response(x):
        return 1.0 - 2.0 * np.asarray(x, dtype=float)
    parameters = [U, linear_response, linear_response, 1.0, 2.0, 1.0]
    model = bm_1(B, B, T, 2)

    def drift(time, state):
        return model._drift(time, state, parameters)

    def diffusion(time, state):
        del time
        return model.diffusion_coefficient(state, SIGMA_X)
    finest_steps = int(BM1_STEPS[-1])
    finest_t = np.linspace(T[0], T[1], finest_steps + 1)
    fine_dW = method.brownian_increments(finest_t, n_paths=N_PATHS, seed=BM1_SEED)
    terminal_states = []
    diagnostics = []
    for n_steps in BM1_STEPS:
        t = np.linspace(T[0], T[1], int(n_steps) + 1)
        dW = _coarsened_increments(method, fine_dW, finest_steps, int(n_steps))
        paths, level_diagnostics = method.solve_paths(
            t,
            X0,
            drift,
            diffusion,
            n_paths=N_PATHS,
            boundary='none',
            brownian_increments=dW,
            return_diagnostics=True,
        )
        terminal_states.append(paths[:, -1].copy())
        diagnostics.append(level_diagnostics)
    self_errors = np.asarray(
        [
            rms_strong_error(terminal_states[i], terminal_states[i + 1])
             for i in range(len(terminal_states) - 1)
        ],
        dtype=float,
    )
    coarse_step_sizes = (T[1] - T[0]) / BM1_STEPS[:-1].astype(float)
    local_orders = successive_convergence_orders(self_errors)
    fitted_order = loglog_convergence_order(coarse_step_sizes, self_errors)
    return (coarse_step_sizes, self_errors, local_orders, fitted_order, diagnostics)


def _require_no_boundary_violations(name, diagnostics):
    total = sum((item['total_violations'] for item in diagnostics))
    if total:
        raise RuntimeError(
            (
                f'{name} produced {total}'
                f' raw boundary violations. The convergence comparison must not hide these with reflection or clipping.'
            ),
        )


def run_comparison(
    n_paths=DEFAULT_N_PATHS,
    x_steps=X_STEPS,
    max_step=MAX_STEP,
    initial_seed=INITIAL_SEED,
    brownian_seed=BROWNIAN_SEED,
):
    """Run MC and pure FV from the same Beta initial law and BM1 inputs."""
    model, parameters = build_reference_case()
    edges, _, dx = model.fokker_planc_reference.grid(x_steps)
    initial_states = sample_beta_initial_states(n_paths, initial_seed)
    integration_t, _ = model.euler_maruyama.build_time_grid(
        T_EVAL,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=max_step,
    )
    dW = model.euler_maruyama.brownian_increments(integration_t, n_paths=n_paths, seed=brownian_seed)
    em_result = model.euler_maruyama_solver(
        initial_states,
        parameters,
        cases_SIGMA_X,
        n_paths=n_paths,
        boundary='none',
        t_eval=T_EVAL,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=max_step,
        brownian_increments=dW,
        return_diagnostics=True,
    )
    t, state, price, baseline, _, _, _, demand, em_diagnostics = em_result
    if em_diagnostics['total_violations']:
        raise RuntimeError(
            'F2-D Monte Carlo produced raw proposals outside [0,1]; the comparison does not apply reflection or clipping.',
        )
    initial_mass = beta_initial_cell_mass(edges)
    fp_result = model.fokker_planc_reference_solver(
        initial_mass,
        parameters,
        cases_SIGMA_X,
        t_eval=T_EVAL,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=max_step,
    )
    fp_t, x, density, cell_mass, fp_mean, fp_variance, _, _, fp_expected_demand, fp_diagnostics = fp_result
    np.testing.assert_array_equal(t, fp_t)
    mc_mean = np.mean(state, axis=0)
    mc_variance = np.var(state, axis=0, ddof=1)
    mc_mean_se = np.sqrt(mc_variance / n_paths)
    mc_demand = np.mean(demand, axis=0)
    mc_demand_se = np.std(demand, axis=0, ddof=1) / np.sqrt(n_paths)
    mc_cell_mass = np.empty_like(cell_mass)
    l1 = np.empty(len(t))
    cdf_sup = np.empty(len(t))
    wasserstein = np.empty(len(t))
    for index in range(len(t)):
        mc_cell_mass[index] = empirical_cell_mass(state[:, index], edges)
        l1[index] = cell_mass_l1(mc_cell_mass[index], cell_mass[index])
        cdf_sup[index] = cell_mass_cdf_sup(mc_cell_mass[index], cell_mass[index])
        wasserstein[index] = cell_mass_wasserstein_1(mc_cell_mass[index], cell_mass[index], dx)
    return {
        't': t,
        'x': x,
        'edges': edges,
        'price': price,
        'baseline': baseline,
        'mc_state': state,
        'mc_mean': mc_mean,
        'mc_variance': mc_variance,
        'mc_mean_se': mc_mean_se,
        'mc_expected_demand': mc_demand,
        'mc_demand_se': mc_demand_se,
        'mc_cell_mass': mc_cell_mass,
        'fp_density': density,
        'fp_cell_mass': cell_mass,
        'fp_mean': fp_mean,
        'fp_variance': fp_variance,
        'fp_expected_demand': fp_expected_demand,
        'l1': l1,
        'cdf_sup': cdf_sup,
        'wasserstein_1': wasserstein,
        'em_diagnostics': em_diagnostics,
        'fp_diagnostics': fp_diagnostics,
        'dx': dx,
        'n_paths': n_paths,
    }
