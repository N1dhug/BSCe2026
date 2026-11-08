"""Historical-only affine ARM variance fit for the Chapter 4 example plots.

The deterministic joint WM4 fit supplies one common drift and N independently
fitted historical initial states. Estimate sigma^2 and tau^2 on those same N
trajectories by nonnegative least squares of squared hourly demand residuals.
Propagate the Euler-Maruyama kernel on the fine plotting grid. Oracle target
observations do not enter the diffusion fit.
"""
import numpy as np

from src.exe.run_model.wm_4.fit_moving_horizon_diffusion import (
    Z_SCORE, fit_variances, variance_kernel,
)


def fine_variance_kernel(theta_F, theta_R, steps_per_hour, horizon):
    """Unit sigma^2 rate variance at every fine-grid point, including last endpoint."""
    sph, horizon = int(steps_per_hour), int(horizon)
    if sph < 2 or horizon < 1: raise ValueError("Invalid variance-kernel grid.")
    dt = 1./(24*sph)
    result = np.zeros(horizon*sph+1, dtype=float)
    f, r = 0., 1.
    for j in range(1, len(result)):
        result[j] = result[j-1]+dt*r*r
        f, r = f+dt*r, r+dt*(theta_F*f+theta_R*r)
    if not np.isfinite(result).all(): raise ValueError("Nonfinite stochastic variance kernel.")
    return result


def fit_historical_uncertainty(fit, inputs, steps_per_hour):
    """Return noise parameters and Gaussian demand standard deviations.

    Condition on the historically fitted initial states. All estimation uses
    historical hourly observation residuals, never target/oracle observations.
    """
    if len(inputs) != len(fit["deterministic_states"]) or not inputs:
        raise ValueError("Historical inputs and fitted trajectories must match.")
    sph = int(steps_per_hour)
    indices = [np.flatnonzero(np.asarray(item["observation_mask"], dtype=bool))
               for item in inputs]
    k = len(indices[0])
    if k < 1 or any(len(idx) != k or not np.array_equal(idx, np.arange(k)*sph)
                    for idx in indices):
        raise ValueError("Expected matching hourly observations in every historical interval.")
    r = np.asarray(fit["r"], dtype=float)
    if r.shape != (3,) or not np.isfinite(r).all():
        raise ValueError("Historical drift coefficients must be finite.")
    residuals = []
    for item, state, idx in zip(inputs, fit["deterministic_states"], indices):
        actual, predicted = np.asarray(item["I"], dtype=float), np.asarray(state["I"], dtype=float)
        if actual.shape != predicted.shape or not np.isfinite(predicted).all():
            raise ValueError("Invalid historical deterministic fitted demand.")
        residuals.append(actual[idx]-predicted[idx])
    errors = np.concatenate(residuals)
    kernel_hourly = variance_kernel(r[0], r[2], sph, k)
    tau2, sigma2, identifiable = fit_variances(errors, np.tile(kernel_hourly, len(inputs)))
    kernel_fine = fine_variance_kernel(r[0], r[2], sph, k)
    if not np.allclose(kernel_fine[np.arange(k)*sph], kernel_hourly, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Fine and hourly variance kernels disagree.")
    sd_fine = np.sqrt(np.maximum(tau2+sigma2*kernel_fine, 0.))
    sd_hourly = sd_fine[np.arange(k)*sph]
    covered = float(np.mean(np.abs(errors) <= Z_SCORE*np.tile(sd_hourly, len(inputs))))
    return {
        "tau":float(np.sqrt(tau2)), "sigma":float(np.sqrt(sigma2)),
        "tau2":float(tau2), "sigma2":float(sigma2),
        "identifiable":bool(identifiable),
        "kernel_hourly":kernel_hourly, "kernel_fine":kernel_fine,
        "sd_hourly":sd_hourly, "sd_fine":sd_fine,
        "historical_coverage_95":covered,
        "historical_observations":int(len(errors)),
        "historical_residual_rmse":float(np.sqrt(np.mean(errors**2))),
    }


def demand_interval(mean, uncertainty):
    """Pointwise 95% marginal Gaussian demand interval on the fine grid."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(uncertainty["sd_fine"], dtype=float)
    if mean.shape != std.shape: raise ValueError("Demand and uncertainty have different grids.")
    return mean-Z_SCORE*std, mean+Z_SCORE*std
