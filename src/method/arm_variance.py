"""ARM variance propagation and nonnegative least-squares calibration.

Shared computational methods; no executable or plotting dependencies.
"""
import numpy as np

Z_SCORE = 1.959963984540054

def variance_kernel(theta_F, theta_R, steps_per_hour, horizon):
    """Exact unit process variance at each observed hour under ARM Euler-Maruyama.

    The impulse response sums dt*(e_R^T A^i e_R)^2. This is equivalent to
    wm_4.process_variance_kernel(...).sum(axis=1) at hourly sample indices,
    without building a (k*s+1)^2 kernel matrix.
    """
    steps_per_hour, horizon = int(steps_per_hour), int(horizon)
    if steps_per_hour < 2 or not 1 <= horizon <= 24:
        raise ValueError("steps_per_hour must be >=2 and horizon in [1,24].")
    dt = 1./(24*steps_per_hour)
    f, r, value = 0., 1., 0.
    kernel = np.empty(horizon, dtype=float)
    for hour in range(horizon):
        kernel[hour] = value
        if hour == horizon-1: break
        for _ in range(steps_per_hour):
            value += dt*r*r
            f, r = f+dt*r, r+dt*(theta_F*f+theta_R*r)
    if not np.isfinite(kernel).all() or np.min(kernel) < -1e-12:
        raise ValueError("Nonfinite or negative Euler-Maruyama variance kernel.")
    return np.maximum(kernel, 0.)

def fit_variances(errors, kernel):
    """Analytic nonnegative LS for E[error^2 | K]=tau^2+q*K.

    Checks the unconstrained nonnegative solution and both active faces.
    Equivalent to NNLS with columns [1,K], avoiding millions of solver calls.
    """
    errors = np.asarray(errors, dtype=float).reshape(-1)
    K = np.asarray(kernel, dtype=float).reshape(-1)
    if errors.size != K.size or not errors.size or not np.isfinite(errors).all():
        raise ValueError("Error and kernel must contain matching finite observations.")
    if not np.isfinite(K).all() or np.min(K) < 0:
        raise ValueError("Variance kernel must be finite and nonnegative.")
    y = errors*errors
    mean_y = float(np.mean(y))
    if np.ptp(K) < 1e-12: return mean_y, 0., False
    mean_K = float(np.mean(K))
    mean_K2 = float(np.mean(K*K))
    mean_Ky = float(np.mean(K*y))
    covariance_K = mean_K2-mean_K**2
    if covariance_K <= 0: return mean_y, 0., False
    q_ols = (mean_Ky-mean_K*mean_y)/covariance_K
    tau_ols = mean_y-q_ols*mean_K
    # All candidate faces of the two-variable nonnegative quadratic program.
    candidates = [(mean_y, 0.), (0., max(0., mean_Ky/mean_K2))]
    if tau_ols >= 0. and q_ols >= 0.: candidates.append((tau_ols, q_ols))
    tau2, q = min(candidates, key=lambda x: float(np.mean((y-x[0]-x[1]*K)**2)))
    return float(tau2), float(q), True
