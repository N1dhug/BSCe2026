"""F2 projection, dynamic-mixture, conditioning and objective comparison studies."""

import numpy as np
from scipy.linalg import null_space

from src.validation.control import positive_integer

from src.diagnostics.bm_1 import (
    beta_mixture_variance_floor,
    cell_mass_cdf_sup,
    cell_mass_l1,
    cell_mass_wasserstein_1,
    density_l2,
)
from src.control.bm_1 import beta_initial_cell_mass, sample_beta_initial_states
from src.method.beta_mixture import (
    evolve_projected_density as _evolve_projected_density,
    _project_with_warm_start,
    _projection_matrices,
)
from src.control import objective as tracking
from src.validation.bm_1.cases import (
    MAX_STEP,
    PRICE_BREAKPOINTS,
    SIGMA_X,
    T,
    T_EVAL,
    X_STEPS,
    build_reference_case,
)


DEFAULT_DEGREES = (5, 10, 20, 30, 38)
DYNAMIC_DEFAULT_DEGREES = (20, 30, 38)
DEFAULT_CFL = 0.4
_NEGATIVE_PREDICTION_TOL = 1e-10
DEFAULT_EPSILONS = (1e-10, 1e-08, 1e-06)
DEFAULT_DIRECTIONS = 4
DEFAULT_SEED = 2030
DEFAULT_ACTIVE_TOL = 1e-08
DEFAULT_N_PATHS = 50000
OBJECTIVE_DEFAULT_DEGREES = (20, 30, 38)
OBJECTIVE_MAX_STEP = MAX_STEP
INITIAL_SEED = 2031
BROWNIAN_SEED = 2032


def _expected_demand(model, parameters, t, x, cell_mass):
    values = np.empty(len(t), dtype=float)
    for index, time in enumerate(t):
        Ut = parameters[0](time)
        Bt = model.B(time)
        fXt = parameters[1](x)
        gUt = parameters[2](Ut)
        delta = model.l(fXt, gUt, parameters[4])
        demand = model.D(Bt, parameters[5], delta)
        values[index] = np.sum(np.asarray(demand, dtype=float) * cell_mass[index])
    return values


def _distribution_errors(left_density, right_density, dx):
    left_mass = np.asarray(left_density, dtype=float) * dx
    right_mass = np.asarray(right_density, dtype=float) * dx
    n_times = left_density.shape[0]
    l2 = np.empty(n_times, dtype=float)
    l1 = np.empty(n_times, dtype=float)
    cdf_sup = np.empty(n_times, dtype=float)
    wasserstein = np.empty(n_times, dtype=float)
    for index in range(n_times):
        l2[index] = density_l2(left_density[index], right_density[index], dx)
        l1[index] = cell_mass_l1(left_mass[index], right_mass[index])
        cdf_sup[index] = cell_mass_cdf_sup(left_mass[index], right_mass[index])
        wasserstein[index] = cell_mass_wasserstein_1(left_mass[index], right_mass[index], dx)
    return {'l2': l2, 'l1': l1, 'cdf_sup': cdf_sup, 'wasserstein_1': wasserstein}


def _positive_mass_preserving_perturbation(density, dx, epsilon, direction):
    """Apply a tiny multiplicative log-density perturbation.

    p_eps is proportional to p * exp(epsilon*r), where r is centered with
    respect to p and scaled to max |r| = 1.  The construction preserves
    non-negativity and then enforces unit mass exactly up to roundoff.
    """
    density = np.asarray(density, dtype=float)
    direction = np.asarray(direction, dtype=float)
    dx = float(dx)
    epsilon = float(epsilon)
    if density.ndim != 1 or direction.shape != density.shape:
        raise ValueError('density and direction must be one-dimensional with equal shape.')
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError('density must be finite and non-negative.')
    if not np.all(np.isfinite(direction)):
        raise ValueError('direction must be finite.')
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError('dx must be finite and strictly positive.')
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError('epsilon must be finite and non-negative.')
    mass = float(np.sum(density) * dx)
    if not np.isclose(mass, 1.0, atol=1e-09, rtol=0.0):
        raise ValueError('density must have unit discrete mass.')
    centered = direction - float(np.sum(density * direction) * dx)
    scale = float(np.max(np.abs(centered)))
    if scale == 0.0:
        raise ValueError('direction must contain non-constant variation.')
    centered /= scale
    perturbed = density * np.exp(epsilon * centered)
    perturbed_mass = float(np.sum(perturbed) * dx)
    if not np.isfinite(perturbed_mass) or perturbed_mass <= 0.0:
        raise RuntimeError('perturbed density lost positive finite mass.')
    return perturbed / perturbed_mass


def _density_moments(density, x, dx):
    mass = np.asarray(density, dtype=float) * dx
    mean = float(np.sum(mass * x))
    variance = float(np.sum(mass * np.square(x - mean)))
    return (mean, variance)


def _distribution_objective(widths, target, demand_values, cell_mass):
    expected, second, variance = tracking.discrete_demand_moments(demand_values, cell_mass)
    result = tracking.stochastic_tracking_objective(widths, expected, second, target)
    np.testing.assert_allclose(result.demand_variance, variance, atol=1e-14, rtol=0.0)
    return result


def run_projection_study(degrees=DEFAULT_DEGREES, *, x_steps=X_STEPS, max_step=MAX_STEP, t_eval=T_EVAL):
    """Project pure-FV snapshots onto independent degree-M Beta mixtures."""
    degrees = tuple(positive_integer(value, 'degree', 0) for value in degrees)
    if not degrees or any((value < 0 for value in degrees)):
        raise ValueError('degrees must contain non-negative integers.')
    t_eval = np.asarray(t_eval, dtype=float)
    model, parameters = build_reference_case()
    edges, x, dx = model.fokker_planc_reference.grid(x_steps)
    initial_mass = beta_initial_cell_mass(edges)
    (
        t,
        fp_x,
        fp_density,
        fp_cell_mass,
        fp_mean,
        fp_variance,
        price,
        baseline,
        fp_expected_demand,
        fp_diagnostics,
    ) = model.fokker_planc_reference_solver(
        initial_mass,
        parameters,
        SIGMA_X,
        t_eval=t_eval,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=max_step,
    )
    np.testing.assert_allclose(fp_x, x, atol=0.0, rtol=0.0)
    results = {}
    for degree in degrees:
        raw_basis = model.fokker_planc.beta_basis(x, degree)
        basis = model.fokker_planc._normalise_basis_discretely(raw_basis, dx)
        coefficients = np.empty((len(t), degree + 1), dtype=float)
        projected_density = np.empty_like(fp_density)
        for index in range(len(t)):
            coefficients[index], projected_density[index] = model.fokker_planc.project_density(
                fp_density[index],
                basis,
                dx,
            )
        projected_mass = projected_density * dx
        projected_mean = np.sum(projected_mass * x[None, :], axis=1)
        projected_variance = np.sum(
            projected_mass * np.square(x[None, :] - projected_mean[:, None]),
            axis=1,
        )
        projected_expected_demand = _expected_demand(model, parameters, t, x, projected_mass)
        l2 = np.empty(len(t), dtype=float)
        l1 = np.empty(len(t), dtype=float)
        cdf_sup = np.empty(len(t), dtype=float)
        wasserstein = np.empty(len(t), dtype=float)
        for index in range(len(t)):
            l2[index] = density_l2(projected_density[index], fp_density[index], dx)
            l1[index] = cell_mass_l1(projected_mass[index], fp_cell_mass[index])
            cdf_sup[index] = cell_mass_cdf_sup(projected_mass[index], fp_cell_mass[index])
            wasserstein[index] = cell_mass_wasserstein_1(projected_mass[index], fp_cell_mass[index], dx)
        variance_floor = beta_mixture_variance_floor(fp_mean, degree)
        results[degree] = {
            'basis': basis,
            'coefficients': coefficients,
            'density': projected_density,
            'cell_mass': projected_mass,
            'mean': projected_mean,
            'variance': projected_variance,
            'expected_demand': projected_expected_demand,
            'l2': l2,
            'l1': l1,
            'cdf_sup': cdf_sup,
            'wasserstein_1': wasserstein,
            'variance_floor': variance_floor,
            'variance_floor_gap': np.maximum(variance_floor - fp_variance, 0.0),
            'maximum_coefficient_sum_error': float(np.max(np.abs(np.sum(coefficients, axis=1) - 1.0))),
            'minimum_coefficient': float(np.min(coefficients)),
            'maximum_mass_error': float(np.max(np.abs(np.sum(projected_mass, axis=1) - 1.0))),
        }
    return {
        't': t,
        'x': x,
        'edges': edges,
        'dx': dx,
        'price': price,
        'baseline': baseline,
        'fp_density': fp_density,
        'fp_cell_mass': fp_cell_mass,
        'fp_mean': fp_mean,
        'fp_variance': fp_variance,
        'fp_expected_demand': fp_expected_demand,
        'fp_diagnostics': fp_diagnostics,
        'degrees': degrees,
        'projection': results,
    }


def run_dynamic_mixture_study(
    degrees=DYNAMIC_DEFAULT_DEGREES,
    *,
    x_steps=X_STEPS,
    max_step=MAX_STEP,
    t_eval=T_EVAL,
    cfl=DEFAULT_CFL,
):
    """Compare dynamic Beta mixtures with pure FV and static best projections."""
    degrees = tuple(positive_integer(value, 'degree', 0) for value in degrees)
    if not degrees or any((value < 0 for value in degrees)):
        raise ValueError('degrees must contain non-negative integers.')
    t_eval = np.asarray(t_eval, dtype=float)
    static = run_projection_study(degrees=degrees, x_steps=x_steps, max_step=max_step, t_eval=t_eval)
    model, parameters = build_reference_case()
    x = static['x']
    dx = static['dx']

    def drift(time, state):
        return model._drift(time, state, parameters)

    def diffusion(time, state):
        del time
        return model.diffusion_coefficient(state, SIGMA_X)
    dynamic = {}
    for degree in degrees:
        static_item = static['projection'][degree]
        basis = static_item['basis']
        initial_coefficients = static_item['coefficients'][0]
        evolved = _evolve_projected_density(
            method=model.fokker_planc,
            t_eval=t_eval,
            breakpoints=PRICE_BREAKPOINTS,
            x=x,
            dx=dx,
            basis=basis,
            initial_coefficients=initial_coefficients,
            drift=drift,
            diffusion=diffusion,
            max_step=max_step,
            cfl=cfl,
        )
        density = evolved['density']
        cell_mass = density * dx
        mean = np.sum(cell_mass * x[None, :], axis=1)
        variance = np.sum(cell_mass * np.square(x[None, :] - mean[:, None]), axis=1)
        expected_demand = _expected_demand(model, parameters, static['t'], x, cell_mass)
        total_error = _distribution_errors(density, static['fp_density'], dx)
        dynamic_gap = _distribution_errors(density, static_item['density'], dx)
        dynamic[degree] = {
            'density': density,
            'cell_mass': cell_mass,
            'coefficients': evolved['coefficients'],
            'mean': mean,
            'variance': variance,
            'expected_demand': expected_demand,
            'total_error': total_error,
            'dynamic_gap': dynamic_gap,
            'static_error': {
                'l2': static_item['l2'].copy(),
                'l1': static_item['l1'].copy(),
                'cdf_sup': static_item['cdf_sup'].copy(),
                'wasserstein_1': static_item['wasserstein_1'].copy(),
            },
            'diagnostics': evolved['diagnostics'],
        }
    result = dict(static)
    result['dynamic'] = dynamic
    result['cfl'] = float(cfl)
    return result


def _simplex_tangent_condition(hessian, coefficients, active_tol=DEFAULT_ACTIVE_TOL):
    """Condition number of local QP curvature along feasible free directions.

    Coefficients larger than active_tol define the local free set.  The simplex
    equality removes the all-ones direction.  The returned condition number is
    therefore computed for Z.T @ H_FF @ Z, where columns of Z span
    {d : 1.T d = 0} on the free set.
    """
    hessian = np.asarray(hessian, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    active_tol = float(active_tol)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1] or hessian.shape[0] != coefficients.size:
        raise ValueError('hessian and coefficients have incompatible shapes.')
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(coefficients)):
        raise ValueError('hessian and coefficients must be finite.')
    if not np.isfinite(active_tol) or active_tol < 0.0:
        raise ValueError('active_tol must be finite and non-negative.')
    free = np.flatnonzero(coefficients > active_tol)
    if free.size <= 1:
        return (1.0, int(free.size), 0)
    reduced_hessian = hessian[np.ix_(free, free)]
    tangent = null_space(np.ones((1, free.size), dtype=float))
    if tangent.shape[1] == 0:
        return (1.0, int(free.size), 0)
    curvature = tangent.T @ reduced_hessian @ tangent
    condition = float(np.linalg.cond(curvature))
    return (condition, int(free.size), int(tangent.shape[1]))


def run_conditioning_study(
    degrees=(20, 30, 38),
    *,
    epsilons=DEFAULT_EPSILONS,
    n_directions=DEFAULT_DIRECTIONS,
    seed=DEFAULT_SEED,
    active_tol=DEFAULT_ACTIVE_TOL,
    x_steps=X_STEPS,
    max_step=MAX_STEP,
    t_eval=T_EVAL,
    cfl=DEFAULT_CFL,
):
    """Return conditioning and projection-sensitivity diagnostics."""
    degrees = tuple(positive_integer(value, 'degree', 0) for value in degrees)
    epsilons = tuple((float(value) for value in epsilons))
    n_directions = positive_integer(n_directions, 'n_directions', 1)
    if not degrees or any((value < 0 for value in degrees)):
        raise ValueError('degrees must contain non-negative integers.')
    if not epsilons or any((not np.isfinite(value) or value < 0.0 for value in epsilons)):
        raise ValueError('epsilons must contain finite non-negative values.')
    if n_directions < 1:
        raise ValueError('n_directions must be at least one.')
    base = run_dynamic_mixture_study(
        degrees=degrees,
        x_steps=x_steps,
        max_step=max_step,
        t_eval=np.asarray(t_eval, dtype=float),
        cfl=cfl,
    )
    x = base['x']
    dx = base['dx']
    rng = np.random.default_rng(seed)
    conditioning = {}
    for degree in degrees:
        static_item = base['projection'][degree]
        dynamic_item = base['dynamic'][degree]
        basis = static_item['basis']
        matrix, hessian, hessian_condition = _projection_matrices(basis, dx)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        matrix_condition = float(singular_values[0] / singular_values[-1])
        static_local_conditions = []
        static_free_sizes = []
        dynamic_local_conditions = []
        dynamic_free_sizes = []
        for static_c, dynamic_c in zip(static_item['coefficients'], dynamic_item['coefficients']):
            value, free_size, _ = _simplex_tangent_condition(hessian, static_c, active_tol)
            static_local_conditions.append(value)
            static_free_sizes.append(free_size)
            value, free_size, _ = _simplex_tangent_condition(hessian, dynamic_c, active_tol)
            dynamic_local_conditions.append(value)
            dynamic_free_sizes.append(free_size)
        uniform = np.full(degree + 1, 1.0 / (degree + 1.0))
        alternative_start = {
            'failures': 0,
            'maximum_coefficient_l1': 0.0,
            'maximum_density_l2': 0.0,
            'maximum_mean_difference': 0.0,
            'maximum_variance_difference': 0.0,
        }
        for index, target in enumerate(base['fp_density']):
            try:
                coefficients, density = _project_with_warm_start(
                    target,
                    basis,
                    matrix,
                    hessian,
                    dx,
                    uniform,
                )
            except RuntimeError:
                alternative_start['failures'] += 1
                continue
            baseline_coefficients = static_item['coefficients'][index]
            baseline_density = static_item['density'][index]
            alternative_start['maximum_coefficient_l1'] = max(
                alternative_start['maximum_coefficient_l1'],
                float(np.sum(np.abs(coefficients - baseline_coefficients))),
            )
            alternative_start['maximum_density_l2'] = max(
                alternative_start['maximum_density_l2'],
                density_l2(density, baseline_density, dx),
            )
            mean, variance = _density_moments(density, x, dx)
            baseline_mean, baseline_variance = _density_moments(baseline_density, x, dx)
            alternative_start['maximum_mean_difference'] = max(
                alternative_start['maximum_mean_difference'],
                abs(mean - baseline_mean),
            )
            alternative_start['maximum_variance_difference'] = max(
                alternative_start['maximum_variance_difference'],
                abs(variance - baseline_variance),
            )
        perturbations = {}
        for epsilon in epsilons:
            summary = {
                'failures': 0,
                'maximum_input_l2': 0.0,
                'maximum_coefficient_l1': 0.0,
                'maximum_projected_density_l2': 0.0,
                'maximum_density_amplification': 0.0,
                'maximum_coefficient_amplification': 0.0,
                'maximum_mean_difference': 0.0,
                'maximum_variance_difference': 0.0,
            }
            for index, target in enumerate(base['fp_density']):
                baseline_coefficients = static_item['coefficients'][index]
                baseline_density = static_item['density'][index]
                baseline_mean, baseline_variance = _density_moments(baseline_density, x, dx)
                for _ in range(n_directions):
                    direction = rng.standard_normal(target.size)
                    perturbed_target = _positive_mass_preserving_perturbation(
                        target,
                        dx,
                        epsilon,
                        direction,
                    )
                    input_l2 = density_l2(perturbed_target, target, dx)
                    try:
                        coefficients, density = _project_with_warm_start(
                            perturbed_target,
                            basis,
                            matrix,
                            hessian,
                            dx,
                            baseline_coefficients,
                        )
                    except RuntimeError:
                        summary['failures'] += 1
                        continue
                    coefficient_l1 = float(np.sum(np.abs(coefficients - baseline_coefficients)))
                    projected_l2 = density_l2(density, baseline_density, dx)
                    scale = max(input_l2, np.finfo(float).tiny)
                    summary['maximum_input_l2'] = max(summary['maximum_input_l2'], input_l2)
                    summary['maximum_coefficient_l1'] = max(
                        summary['maximum_coefficient_l1'],
                        coefficient_l1,
                    )
                    summary['maximum_projected_density_l2'] = max(
                        summary['maximum_projected_density_l2'],
                        projected_l2,
                    )
                    summary['maximum_density_amplification'] = max(
                        summary['maximum_density_amplification'],
                        projected_l2 / scale,
                    )
                    summary['maximum_coefficient_amplification'] = max(
                        summary['maximum_coefficient_amplification'],
                        coefficient_l1 / scale,
                    )
                    mean, variance = _density_moments(density, x, dx)
                    summary['maximum_mean_difference'] = max(
                        summary['maximum_mean_difference'],
                        abs(mean - baseline_mean),
                    )
                    summary['maximum_variance_difference'] = max(
                        summary['maximum_variance_difference'],
                        abs(variance - baseline_variance),
                    )
            perturbations[epsilon] = summary
        conditioning[degree] = {
            'matrix_condition_number': matrix_condition,
            'hessian_condition_number': hessian_condition,
            'minimum_singular_value': float(singular_values[-1]),
            'maximum_singular_value': float(singular_values[0]),
            'static_local_condition_numbers': np.asarray(static_local_conditions, dtype=float),
            'static_free_sizes': np.asarray(static_free_sizes, dtype=int),
            'dynamic_local_condition_numbers': np.asarray(dynamic_local_conditions, dtype=float),
            'dynamic_free_sizes': np.asarray(dynamic_free_sizes, dtype=int),
            'alternative_start': alternative_start,
            'perturbations': perturbations,
        }
    result = dict(base)
    result['conditioning'] = conditioning
    result['epsilons'] = epsilons
    result['n_directions'] = n_directions
    result['active_tol'] = float(active_tol)
    return result


def run_objective_reference(
    *,
    n_paths=DEFAULT_N_PATHS,
    degrees=OBJECTIVE_DEFAULT_DEGREES,
    x_steps=X_STEPS,
    objective_max_step=OBJECTIVE_MAX_STEP,
    solver_max_step=MAX_STEP,
    cfl=DEFAULT_CFL,
    initial_seed=INITIAL_SEED,
    brownian_seed=BROWNIAN_SEED,
):
    """Evaluate the same stochastic BM1 objective with MC, FV and mixtures."""
    degrees = tuple(positive_integer(value, 'degree', 0) for value in degrees)
    if not degrees or any((value < 0 for value in degrees)):
        raise ValueError('degrees must contain non-negative integers.')
    edges, midpoints, widths, solver_t_eval = tracking.midpoint_quadrature(
        T,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=objective_max_step,
    )
    model, parameters = build_reference_case()
    target = np.asarray(model.A(midpoints), dtype=float)
    initial_states = sample_beta_initial_states(n_paths, initial_seed)
    integration_t, _ = model.euler_maruyama.build_time_grid(
        solver_t_eval,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=solver_max_step,
    )
    brownian_increments = model.euler_maruyama.brownian_increments(
        integration_t,
        n_paths=n_paths,
        seed=brownian_seed,
    )
    em_result = model.euler_maruyama_solver(
        initial_states,
        parameters,
        SIGMA_X,
        n_paths=n_paths,
        boundary='none',
        t_eval=solver_t_eval,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=solver_max_step,
        brownian_increments=brownian_increments,
        return_diagnostics=True,
    )
    em_t, _, _, _, _, _, _, em_demand, em_diagnostics = em_result
    np.testing.assert_array_equal(em_t, solver_t_eval)
    if em_diagnostics['total_violations']:
        raise RuntimeError(
            'F2-H Monte Carlo produced raw boundary violations; the stochastic objective reference does not hide them.',
        )
    midpoint_demand_paths = np.asarray(em_demand, dtype=float)[:, 1:-1]
    mc = tracking.monte_carlo_tracking_objective(widths, midpoint_demand_paths, target)
    density_result = run_dynamic_mixture_study(
        degrees=degrees,
        x_steps=x_steps,
        max_step=solver_max_step,
        t_eval=solver_t_eval,
        cfl=cfl,
    )
    np.testing.assert_array_equal(density_result['t'], solver_t_eval)
    x = density_result['x']
    demand_values = tracking.demand_grid(model, parameters, midpoints, x)
    fv = _distribution_objective(widths, target, demand_values, density_result['fp_cell_mass'][1:-1])
    mixture = {}
    for degree in degrees:
        mixture[degree] = _distribution_objective(
            widths,
            target,
            demand_values,
            density_result['dynamic'][degree]['cell_mass'][1:-1],
        )
    return {
        'edges': edges,
        'midpoints': midpoints,
        'widths': widths,
        'solver_t_eval': solver_t_eval,
        'target': target,
        'price': np.asarray(parameters[0](midpoints), dtype=float),
        'mc': mc,
        'fv': fv,
        'mixture': mixture,
        'degrees': degrees,
        'n_paths': int(n_paths),
        'x_steps': int(x_steps),
        'em_diagnostics': em_diagnostics,
        'density_result': density_result,
    }
