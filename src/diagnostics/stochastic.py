"""Reusable stochastic-convergence diagnostics."""
import numpy as np

def _finite_array(values, name):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f'{name} must not be empty.')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain finite values.')
    return array

def rms_strong_error(approximation, reference):
    """Return the empirical root-mean-square pathwise error."""
    approximation = _finite_array(approximation, 'approximation')
    reference = _finite_array(reference, 'reference')
    if approximation.shape != reference.shape:
        raise ValueError('approximation and reference must have the same shape.')
    return float(np.sqrt(np.mean(np.square(approximation - reference))))

def successive_convergence_orders(errors, refinement_factor=2.0):
    """Estimate local orders from successive errors under uniform refinement."""
    errors = _finite_array(errors, 'errors').reshape(-1)
    if errors.size < 2:
        raise ValueError('errors must contain at least two values.')
    if np.any(errors <= 0.0):
        raise ValueError('errors must be strictly positive.')
    refinement_factor = float(refinement_factor)
    if not np.isfinite(refinement_factor) or refinement_factor <= 1.0:
        raise ValueError('refinement_factor must be finite and greater than one.')
    return np.log(errors[:-1] / errors[1:]) / np.log(refinement_factor)

def loglog_convergence_order(step_sizes, errors):
    """Fit log(error) = constant + p log(h) and return the slope p."""
    step_sizes = _finite_array(step_sizes, 'step_sizes').reshape(-1)
    errors = _finite_array(errors, 'errors').reshape(-1)
    if step_sizes.size != errors.size or step_sizes.size < 2:
        raise ValueError('step_sizes and errors must have the same length of at least two.')
    if np.any(step_sizes <= 0.0) or np.any(errors <= 0.0):
        raise ValueError('step_sizes and errors must be strictly positive.')
    if np.any(np.diff(step_sizes) >= 0.0):
        raise ValueError('step_sizes must be strictly decreasing.')
    return float(np.polyfit(np.log(step_sizes), np.log(errors), 1)[0])
