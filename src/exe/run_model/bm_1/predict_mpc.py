"""BM1 MPC: fit five shifted 24h windows and forecast up to eight hours.

The pooled logistic theta and each historical X0 are refitted at EVERY
hourly origin, using five previous matching weekday/hour windows.
Historical demand residuals determine the nonnegative sigma_x² estimate.
Only the latest observed consumption I(origin) is assimilated. Future I is
never used by fitting, diffusion calibration or state initialization.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.exe.run_model.bm_1.predict_daily import held_signal
from src.model.bm_1 import bm_1
from src.params.bm_1 import lsq_params, historical_diffusion_params
from src.solver.finite_volume import finite_volume_1d_cached
from src.tools.baseline import statistical_output
from src.tools.forecast_metrics import predictive_intervals, weighted_quantiles
from src.validation.exe.run_model.bm_1.predict_mpc import bm_1_mpc_validation as checks


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "bm_1" / "prediction" / "mpc"


def posterior_mass(states, x_grid, prediction, observed, observation_sd, bandwidth):
    """Smooth five empirical X0 points, then assimilate I(origin) via Bayes rule.

    The KDE bandwidth and observation SD are explicit approximations, not
    additional state-process diffusion. Posterior normalization is exact.
    """
    states = np.asarray(states, dtype=float)
    smooth = max(float(bandwidth), float(np.std(states)) / 2.0)
    prior = np.exp(-0.5 * ((x_grid[:, None] - states[None, :]) / smooth) ** 2).mean(axis=1)
    prior = checks.posterior(prior, len(x_grid))
    z = (prediction - observed) / observation_sd
    log_prob = np.log(np.maximum(prior, 1e-300)) - 0.5 * z * z
    weights = np.exp(log_prob - np.max(log_prob))
    return checks.posterior(weights, len(x_grid))


def predict_origin(frame, origin, output_model, *, X0=0.5, weeks=5, horizon=8, x_steps=101,
    max_iterations=3, max_nfev=120, state_maxiter=60, substeps=2,
    fp_max_step=0.25, prior_bandwidth=0.025, C=None, Xi=None):
    """Return 1..horizon predictive demand distributions for one MPC origin.

    frame must be normalized to BM1's [0,1] I and U before calling.
    output_model must be trained using only observations preceding the origin.
    """
    origin = checks.origin(origin)
    checks.options(horizon, x_steps, fp_max_step, prior_bandwidth, substeps, weeks)
    training_origins = [origin - pd.Timedelta(weeks=k) for k in range(weeks, 0, -1)]
    training = [checks.window(frame, t, output_model, hours=24) for t in training_origins]
    model = bm_1(held_signal(training[0]["B"]), lambda t: 0.5, (0.0, 24.0), 25)
    fitted = lsq_params(model).fit_many(training, X0, C=C, Xi=Xi, max_iterations=max_iterations,
        max_nfev=max_nfev, state_maxiter=state_maxiter, substeps=substeps)
    diffusion = historical_diffusion_params().fit_many(training, fitted, substeps=substeps)
    sigma = float(diffusion["sigma_x"])

    future = checks.window(frame, origin, output_model, hours=horizon + 1, observations=False)
    observed = checks.observation(frame, origin)
    x_grid = (np.arange(x_steps, dtype=float) + 0.5) / x_steps
    f, g = fitted["f"], fitted["g"]
    delta0 = np.tanh(0.5 * fitted["k"] * (f(x_grid) + g(future["U"][0])))
    output0 = future["B"][0] + fitted["Xi"] * delta0 * np.where(delta0 >= 0.0, 1.0 - future["B"][0], future["B"][0])
    masses0 = posterior_mass(fitted["initial_states"], x_grid, output0, observed,
        diffusion["observation_sd"], prior_bandwidth)

    supports, masses = [], []
    if sigma > 0.0:
        price = held_signal(future["U"])
        model = bm_1(held_signal(future["B"]), lambda t: 0.5, (0.0, float(horizon)), horizon + 1)
        model.fokker_planc_reference = finite_volume_1d_cached()
        parameters = [price, f, g, fitted["C"], fitted["k"], fitted["Xi"]]
        t, x, density, cell_mass, ex, vx, u, b, expected_d, diagnostics = model.fokker_planc_reference_solver(
            masses0, parameters, sigma, t_eval=np.arange(horizon + 1, dtype=float), max_step=fp_max_step,
        )
        for lead in range(1, horizon + 1):
            delta = np.tanh(0.5 * fitted["k"] * (f(x) + g(future["U"][lead])))
            D = future["B"][lead] + fitted["Xi"] * delta * np.where(delta >= 0, 1.0 - future["B"][lead], future["B"][lead])
            supports.append(D)
            masses.append(cell_mass[lead])
    else:
        trajectories = []
        for x0 in x_grid:
            _, demand = lsq_params._predict(future, float(x0), fitted, substeps)
            trajectories.append(demand)
        predictions = np.asarray(trajectories)
        for lead in range(1, horizon + 1):
            supports.append(predictions[:, lead])
            masses.append(masses0)

    intervals = predictive_intervals(supports, masses)
    rows = []
    for lead, (D, weights) in enumerate(zip(supports, masses), start=1):
        rows.append({
            "origin": origin, "origin_day": origin.normalize(), "origin_hour": origin.hour,
            "target_time": origin + pd.Timedelta(hours=lead), "lead_hours": lead,
            "I_BM1": float(np.dot(D, weights)),
            "lower_68": float(intervals[68][0][lead - 1]), "upper_68": float(intervals[68][1][lead - 1]),
            "lower_95": float(intervals[95][0][lead - 1]), "upper_95": float(intervals[95][1][lead - 1]),
            "sigma_x": sigma, "diffusion_q": float(diffusion["q"]),
            "observation_sd": float(diffusion["observation_sd"]),
            "historical_fit_rmse": float(diffusion["historical_rmse"]),
            "historical_diffusion_variance_rmse": float(diffusion["variance_rmse"]),
            "prior_state_mean": float(np.mean(fitted["initial_states"])),
            "posterior_state_mean": float(np.dot(x_grid, masses0)),
            "posterior_state_std": float(np.sqrt(np.dot(np.square(x_grid - np.dot(x_grid, masses0)), masses0))),
            "pooled_training_mse": float(fitted["mse"]), "fit_iterations": int(fitted["iterations"]),
        })
    return rows, {"fit": fitted, "diffusion": diffusion, "posterior": masses0,
        "training_origins": training_origins, "origin_observed_I": observed}


def predict_day(frame, day, *, X0=0.5, horizon=8, x_steps=101, max_iterations=3,
    max_nfev=120, state_maxiter=60, substeps=2, fp_max_step=0.25, prior_bandwidth=0.025,
    C=None, Xi=None, skip_errors=False):
    """Apply the same 24h five-week historical fitting at all 24 hourly origins."""
    day = checks.origin(pd.Timestamp(day).normalize())
    frame = checks.frame(frame)
    baseline = statistical_output().fit(frame, day)
    forecasts, origins, skipped = [], [], []
    for hour in range(24):
        origin = day + pd.Timedelta(hours=hour)
        try:
            predicted, diagnostics = predict_origin(frame, origin, baseline, X0=X0,
                horizon=horizon, x_steps=x_steps, max_iterations=max_iterations,
                max_nfev=max_nfev, state_maxiter=state_maxiter, substeps=substeps,
                fp_max_step=fp_max_step, prior_bandwidth=prior_bandwidth, C=C, Xi=Xi)
            forecasts.extend(predicted)
            origins.append({"origin": origin, **diagnostics["diffusion"]})
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            if not skip_errors: raise
            skipped.append((origin, str(exc)))
    return pd.DataFrame(forecasts), pd.DataFrame(origins), skipped
