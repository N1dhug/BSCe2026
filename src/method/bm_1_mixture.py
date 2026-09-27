"""BM-only conservative Beta-mixture evolution and constrained projection."""

import numpy as np
from src.solver.active_set import active_set
from src.validation.method.bm_1 import _validate_dynamic_controls

_NEGATIVE_PREDICTION_TOL = 1e-10
_MASS_TOL = 1e-09

def _projection_matrices(basis, dx):
    """Return weighted least-squares matrix, Hessian and its condition number."""
    basis = np.asarray(basis, dtype=float)
    dx = float(dx)
    scale = np.sqrt(dx)
    matrix = basis.T * scale
    hessian = matrix.T @ matrix
    condition = float(np.linalg.cond(hessian))
    return (matrix, hessian, condition)

def _project_with_warm_start(density, basis, matrix, hessian, dx, x0):
    """Project one density onto the simplex-constrained basis using a warm start."""
    density = np.asarray(density, dtype=float)
    target = density * np.sqrt(dx)
    linear = -(matrix.T @ target)
    n_coefficients = basis.shape[0]
    coefficients = active_set.solve(hessian, linear, A_eq=np.ones((1, n_coefficients), dtype=float),
        b_eq=np.array([1.0], dtype=float), lb=np.zeros(n_coefficients, dtype=float), x0=np.asarray(x0, dtype=float))
    projected = coefficients @ basis
    return (coefficients, projected)

def _positive_explicit_step_limit(dx, drift_values, diffusion_values, cfl):
    """Return a positivity-oriented CFL limit for the conservative grid update.

    The outgoing rate from an interior cell is the sum of its upwind advective
    rates and the two half-diffusion rates.  Requiring h * outgoing_rate <= cfl
    is stricter than taking the minimum of separate advection/diffusion limits
    and avoids relying on post-step clipping.
    """
    drift_values = np.asarray(drift_values, dtype=float)
    diffusion_values = np.asarray(diffusion_values, dtype=float)
    face_drift = 0.5 * (drift_values[:-1] + drift_values[1:])
    diffusion_term = 0.5 * np.square(diffusion_values)
    outgoing = np.zeros_like(drift_values)
    outgoing[:-1] += np.maximum(face_drift, 0.0) / dx
    outgoing[1:] += np.maximum(-face_drift, 0.0) / dx
    outgoing[:-1] += diffusion_term[:-1] / (dx * dx)
    outgoing[1:] += diffusion_term[1:] / (dx * dx)
    maximum = float(np.max(outgoing))
    return np.inf if maximum == 0.0 else float(cfl / maximum)

def evolve_projected_density(*, method, t_eval, breakpoints, x, dx, basis, initial_coefficients, drift, diffusion,
    max_step, cfl, projection_data=None):
    """Propagate one fixed Beta basis through explicit Euler + QP projection."""
    t_eval, breakpoints, max_step, cfl = _validate_dynamic_controls(t_eval, breakpoints, max_step, cfl)
    x = np.asarray(x, dtype=float)
    basis = np.asarray(basis, dtype=float)
    coefficients = np.asarray(initial_coefficients, dtype=float).copy()
    if x.ndim != 1 or x.size < 2 or (not np.all(np.isfinite(x))):
        raise ValueError('x must be a finite one-dimensional grid.')
    dx = float(dx)
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError('dx must be finite and positive.')
    if not np.allclose(np.diff(x), dx, rtol=1e-10, atol=1e-14):
        raise ValueError('x must be uniformly spaced with spacing dx.')
    if basis.ndim != 2 or basis.shape[1] != x.size or basis.shape[0] != coefficients.size or (not np.all(np.isfinite(basis))):
        raise ValueError('basis and coefficients must match the state grid.')
    if coefficients.ndim != 1 or not np.all(np.isfinite(coefficients)) or np.min(coefficients) < -_MASS_TOL or (abs(float(np.sum(coefficients)) - 1.0) > _MASS_TOL):
        raise ValueError('initial_coefficients must satisfy the probability simplex.')
    if np.min(basis) < 0.0 or not np.allclose(np.sum(basis, axis=1) * dx, 1.0, rtol=0.0, atol=_MASS_TOL):
        raise ValueError('each basis component must have non-negative unit mass.')
    density = coefficients @ basis
    if projection_data is None:
        projection_data = _projection_matrices(basis, dx)
    matrix, hessian, condition = projection_data
    if matrix.shape != basis.T.shape or hessian.shape != (basis.shape[0],) * 2:
        raise ValueError('projection_data does not match the supplied basis.')
    output_density = np.full((len(t_eval), len(x)), np.nan, dtype=float)
    output_coefficients = np.full((len(t_eval), basis.shape[0]), np.nan, dtype=float)
    output_density[0] = density
    output_coefficients[0] = coefficients
    knots = np.unique(np.concatenate((t_eval, breakpoints)))
    output_lookup = {float(value): index for index, value in enumerate(t_eval)}
    n_steps = 0
    minimum_raw_prediction = float(np.min(density))
    maximum_raw_prediction_mass_error = abs(float(np.sum(density) * dx) - 1.0)
    maximum_projection_mass_error = maximum_raw_prediction_mass_error
    maximum_coefficient_sum_error = abs(float(np.sum(coefficients)) - 1.0)
    minimum_coefficient = float(np.min(coefficients))
    maximum_boundary_coefficient = float(max(coefficients[0], coefficients[-1]))
    maximum_internal_step = 0.0
    minimum_internal_step = np.inf
    for left, right in zip(knots[:-1], knots[1:]):
        current = float(left)
        target = float(right)
        while current < target - 1e-15:
            drift_values = np.asarray(drift(current, x), dtype=float)
            diffusion_values = np.asarray(diffusion(current, x), dtype=float)
            if drift_values.shape != x.shape or diffusion_values.shape != x.shape:
                raise ValueError('drift and diffusion must return arrays matching x.')
            if not np.all(np.isfinite(drift_values)) or not np.all(np.isfinite(diffusion_values)):
                raise ValueError('drift and diffusion must return finite values.')
            stable_h = _positive_explicit_step_limit(dx, drift_values, diffusion_values, cfl)
            remaining = target - current
            h = min(remaining, max_step, stable_h)
            if not np.isfinite(h) or h <= 0.0:
                raise RuntimeError('Dynamic mixture produced a non-positive internal step.')
            rhs = method._flux_rhs(x, density, drift_values, diffusion_values)
            prediction = density + h * rhs
            raw_minimum = float(np.min(prediction))
            raw_mass_error = abs(float(np.sum(prediction) * dx) - 1.0)
            minimum_raw_prediction = min(minimum_raw_prediction, raw_minimum)
            maximum_raw_prediction_mass_error = max(maximum_raw_prediction_mass_error, raw_mass_error)
            if raw_minimum < -_NEGATIVE_PREDICTION_TOL:
                raise RuntimeError('Raw explicit-Euler mixture prediction became materially negative; reduce cfl/max_step rather than clipping it.')
            if raw_mass_error > _MASS_TOL:
                raise RuntimeError('Raw mixture prediction failed probability conservation.')
            coefficients, density = _project_with_warm_start(prediction, basis, matrix, hessian, dx, coefficients)
            projected_mass_error = abs(float(np.sum(density) * dx) - 1.0)
            coefficient_sum_error = abs(float(np.sum(coefficients)) - 1.0)
            maximum_projection_mass_error = max(maximum_projection_mass_error, projected_mass_error)
            maximum_coefficient_sum_error = max(maximum_coefficient_sum_error, coefficient_sum_error)
            minimum_coefficient = min(minimum_coefficient, float(np.min(coefficients)))
            maximum_boundary_coefficient = max(maximum_boundary_coefficient, float(coefficients[0]),
                float(coefficients[-1]))
            if projected_mass_error > _MASS_TOL or coefficient_sum_error > _MASS_TOL or minimum_coefficient < -_MASS_TOL:
                raise RuntimeError('Projected mixture failed probability constraints.')
            n_steps += 1
            maximum_internal_step = max(maximum_internal_step, h)
            minimum_internal_step = min(minimum_internal_step, h)
            current += h
            if target - current <= 1e-14 * max(1.0, abs(target)):
                current = target
        if target in output_lookup:
            index = output_lookup[target]
            output_density[index] = density
            output_coefficients[index] = coefficients
    if not np.all(np.isfinite(output_density)) or not np.all(np.isfinite(output_coefficients)):
        raise RuntimeError('Dynamic mixture failed to populate finite output histories.')
    return {'density': output_density, 'coefficients': output_coefficients,
        'diagnostics': {'n_steps': n_steps, 'hessian_condition_number': condition, 'minimum_raw_prediction': minimum_raw_prediction, 'maximum_abs_raw_prediction_mass_error': maximum_raw_prediction_mass_error, 'maximum_abs_projected_mass_error': maximum_projection_mass_error, 'maximum_abs_coefficient_sum_error': maximum_coefficient_sum_error, 'minimum_coefficient': minimum_coefficient, 'maximum_boundary_coefficient': maximum_boundary_coefficient, 'maximum_internal_step': maximum_internal_step, 'minimum_internal_step': minimum_internal_step, 'used_clipping': False, 'used_renormalization': False}}
