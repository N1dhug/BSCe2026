"""G6a: conditional linear-noise reference on the existing G4 trajectory.

Constant diffusion amplitude permits exact nonnegative one-parameter LS after
one unit-amplitude propagation. This is not a full SDE distribution solver.
"""
from time import perf_counter
import numpy as np
from scipy.stats import norm
from src.method.bm_1_deterministic_fit import BudgetStop, _parameters
from src.method.bm_1_deterministic_response import BM1ResponseInputs, is_linear


def demand_state_derivative(state, price_response, baseline, k, xi):
    """Linear f; right branch at delta=0 (one-sided, generally a kink)."""
    delta = np.tanh(.5*k*(1-2*np.asarray(state)+price_response))
    gamma = np.where(delta >= 0, 1-np.asarray(baseline), baseline)
    return -k*xi*gamma*(1-delta**2)


def unit_variance(inputs, parameters, *, price_response=None, reset_hour=0, deadline=None):
    """P'=2(D_x/C)P+[x(1-x)]^2, P(reset)=0, unit sigma per sqrt(hour).

    Reuse G4 RK4 states. A positive exponential midpoint step solves locally
    frozen scalar coefficients, segments every price-hour edge, and never
    clips state or variance. Warm-up state is retained, warm-up noise discarded.
    Only the legacy point operator is supported: interval covariance is distinct.
    """
    if not isinstance(inputs, BM1ResponseInputs) or not is_linear(inputs.state_response):
        raise ValueError('G6a requires G4 response inputs with linear f.')
    if inputs.contract != 'hourly-point':
        raise ValueError('Interval variance requires a separate covariance operator.')
    if isinstance(reset_hour, bool) or int(reset_hour) != reset_hour or not 0 <= reset_hour < inputs.n_hours:
        raise ValueError('Reset must be an integer physical hour inside the trajectory.')
    p = _parameters(parameters)
    curve = inputs.price_response if price_response is None else price_response
    forward = inputs.solve(p, price_response=curve, keep_fine=True, deadline=deadline)
    x = forward['fine_state']; h = inputs.h
    midpoint = .5*(x[:-1]+x[1:])
    g = np.repeat(curve(inputs.prices), inputs.steps)
    dx_mid = demand_state_derivative(midpoint, g, inputs.b_mid, p[1], p[2])
    rates = 2*dx_mid/p[0]
    source = (midpoint*(1-midpoint))**2
    variance = np.zeros_like(x)
    first = int(reset_hour)*inputs.steps
    for j in range(first, len(x)-1):
        if j % (24*inputs.steps) == 0 and deadline is not None and perf_counter() >= deadline:
            raise BudgetStop('G6a wall-time limit')
        rate = rates[j]
        factor = np.expm1(rate*h)/rate if rate != 0 else h
        variance[j+1] = np.exp(rate*h)*variance[j]+source[j]*factor
    if not np.isfinite(variance).all() or np.any(variance < 0):
        raise ValueError('Invalid propagated variance; no clipping.')
    hourly = variance[::inputs.steps]
    b = np.interp(np.arange(inputs.n_hours), inputs.time, inputs.baseline)
    derivative = demand_state_derivative(forward['state'][:-1], curve(inputs.prices), b, p[1], p[2])
    point_variance = derivative**2*hourly[:-1]
    midpoint_delta = 1-2*midpoint+g
    return {'state':forward['state'], 'demand':forward['demand'],
            'unit_state_variance':hourly, 'unit_demand_variance':point_variance,
            'unit_demand_std':np.sqrt(point_variance),
            'response_sign_changes':int(np.count_nonzero(np.diff(np.sign(midpoint_delta[first:])))),
            'reset_hour':int(reset_hour), 'steps_per_hour':inputs.steps}


def fit_constant_sigma(unit_std, observed_std):
    """Global LS minimizer sigma>=0 of mean((sigma*w-s)^2), no ODE search.

    Nonfinite empirical bins (e.g. insufficient measured history) are excluded.
    A zero model sensitivity is reported as unidentifiable, never a fitted zero.
    """
    w, s = np.broadcast_arrays(np.asarray(unit_std, float), np.asarray(observed_std, float))
    mask = np.isfinite(s)
    if not np.isfinite(w).all() or np.any(w < 0) or np.any(s[mask] < 0) or not mask.any():
        raise ValueError('Finite nonnegative model and measured dispersions are required.')
    denominator = float(w[mask]@w[mask])
    if denominator <= np.finfo(float).tiny:
        return {'status':'unidentifiable', 'sigma_per_sqrt_hour':None, 'samples':int(mask.sum())}
    sigma = max(0., float(w[mask]@s[mask])/denominator)
    residual = sigma*w[mask]-s[mask]
    return {'status':'complete', 'sigma_per_sqrt_hour':sigma,
            'sigma_per_sqrt_day':np.sqrt(24)*sigma, 'samples':int(mask.sum()),
            'sigma_rmse':float(np.sqrt(np.mean(residual**2))),
            'sigma_mae':float(np.mean(abs(residual))), 'sigma_bias':float(np.mean(residual)),
            'unit_sensitivity_squared':denominator}


def interval_metrics(observed, center, std, *, alpha=.05):
    """Nominal Gaussian bands without clipping; proper interval score."""
    y, mean, sd = np.broadcast_arrays(np.asarray(observed,float), np.asarray(center,float), np.asarray(std,float))
    if not 0 < alpha < 1 or not np.isfinite(mean).all() or not np.isfinite(sd).all() or np.any(sd < 0):
        raise ValueError('Invalid interval inputs.')
    mask = np.isfinite(y)
    if not mask.any(): raise ValueError('No measured interval targets.')
    y, mean, sd = y[mask], mean[mask], sd[mask]
    z = norm.ppf(1-alpha/2); lower, upper = mean-z*sd, mean+z*sd
    score = upper-lower + (2/alpha)*np.maximum(lower-y,0) + (2/alpha)*np.maximum(y-upper,0)
    return {'samples':int(mask.sum()), 'nominal_coverage':1-alpha,
            'coverage':float(np.mean((y >= lower)&(y <= upper))),
            'mean_width':float(np.mean(upper-lower)), 'mean_interval_score':float(np.mean(score)),
            'point_mse':float(np.mean((mean-y)**2))}
