from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start, fit_expected_output, fitted_initial_condition, model_output
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY
from src.exe.run_model.wm_4.predict_daily import deviation_metrics
from src.model.wm_4 import wm_4
from src.params.wm_4 import constant_diffusion_qp, optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_mpc"
HORIZONS = (1, 2, 3, 4, 6, 8)
SIGMA_FIT_MODES = ("mpc_replay", "mpc_replay_24d_24h", "mpc_replay_24d_6h",
    "mpc_replay_30d_6h", "mpc_replay_28d_6h", "historical_p0", "zero_p0")
ROLLING_SIGMA_HORIZONS = {"mpc_replay_24d_24h": 24,
    "mpc_replay_24d_6h": 6, "mpc_replay_30d_6h": 6,
    "mpc_replay_28d_6h": 6}
ROLLING_SIGMA_DAYS = {"mpc_replay_24d_24h": 24,
    "mpc_replay_24d_6h": 24, "mpc_replay_30d_6h": 30,
    "mpc_replay_28d_6h": 28}
SIGMA_DAYS = 24
SIGMA_HOURS = 24
R0_VARIANCE_MODES = ("oracle_reconstruction", "baseline_residual", "zero")
F0_VARIANCE_MODES = ("mpc_reconstruction", "oracle_spread")


def _window_input(frame, start, steps_per_hour, output_model, hours=DAY_HOURS):
    start = pd.Timestamp(start)
    steps_per_hour, hours = int(steps_per_hour), int(hours)
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2.")
    if hours < 2 or hours > DAY_HOURS:
        raise ValueError("WM4 fitting window must be between 2 and 24 hours.")

    end = start + pd.Timedelta(hours=hours)
    n_steps = hours * steps_per_hour + 1
    elapsed = np.linspace(0.0, float(hours), n_steps)
    fine_index = start + pd.to_timedelta(elapsed, unit="h")

    source_index = pd.date_range(start, end, freq="h")
    source = frame.reindex(source_index)[["I", "U", "Temp"]].copy()
    if source.iloc[:-1].isna().any().any(): raise ValueError(f"Incomplete I/U/temperature data between {start} and {end}.")
    source["I"] = source["I"].interpolate(method="time", limit_direction="both").ffill().bfill()
    source["U"] = source["U"].ffill().bfill()
    source["Temp"] = source["Temp"].interpolate(method="time", limit_direction="both").ffill().bfill()

    hour_bin = np.minimum(np.floor(elapsed + 1e-12).astype(int), hours)
    I = np.interp(elapsed, np.arange(hours + 1, dtype=float), source["I"].to_numpy(dtype=float))
    U = source["U"].to_numpy(dtype=float)[hour_bin]
    Temp = np.interp(elapsed, np.arange(hours + 1, dtype=float), source["Temp"].to_numpy(dtype=float))

    observation_mask = np.zeros(n_steps, dtype=bool)
    observation_mask[np.arange(hours, dtype=int) * steps_per_hour] = True
    transition_mask = np.ones(n_steps - 1, dtype=bool)
    data = {
        "I": I,
        "U": U,
        "PE": U.copy(),
        "Temp": Temp,
        "Time": elapsed,
        "O": output_model.predict(fine_index, Temp),
        "observation_mask": observation_mask,
        "transition_mask": transition_mask,
    }
    return data, fine_index


def _fit_shifted_history(frame, origin, baseline_weeks, steps_per_hour, output_model,
    fit_hours=DAY_HOURS, **fit_kwargs):
    origin, baseline_weeks = pd.Timestamp(origin), int(baseline_weeks)
    if baseline_weeks < 1: raise ValueError("baseline_weeks must be at least 1.")

    training_origins = [origin - pd.Timedelta(weeks=k) for k in range(1, baseline_weeks + 1)]
    datasets, indices, seeds = [], [], []

    for historical_origin in training_origins:
        data, fine_index = _window_input(frame, historical_origin, steps_per_hour,
            output_model, hours=fit_hours)
        datasets.append(data)
        indices.append(fine_index)
        seeds.append(fitted_initial_condition(data))

    model = wm_4((0.0, 1.0), len(indices[0]), model_output(datasets[0], indices[0]))
    fit = optimized_qp_params(model).fit_many(
        datasets,
        np.vstack(seeds),
        fit_initial_state=True,
        verbose=False,
        **fit_kwargs,
    )
    fit["training_origins"] = training_origins
    fit["prediction_origin"] = origin
    fit["fit_hours"] = int(fit_hours)
    return fit, model, datasets


def _mpc_initial_covariance(fit, datasets, r0_variance_mode="oracle_reconstruction",
    *, f0_variance_mode="mpc_reconstruction", state_archive=None):
    """Estimate errors between causal MPC initialization and retrospective oracles.

    For a historical origin, an archived error compares that origin's genuine
    MPC initial state with the oracle fitted subsequently using the SAME fixed
    parameter set. Five historical oracles bootstrap missing archived records.
    Archived reconstruction errors are only read after the historical 24-hour
    trajectory is complete. Correlated F/R errors yield a full 2x2 P0.
    """
    if r0_variance_mode not in R0_VARIANCE_MODES:
        raise ValueError(f"r0_variance_mode must be one of {R0_VARIANCE_MODES}.")
    if f0_variance_mode not in F0_VARIANCE_MODES:
        raise ValueError(f"f0_variance_mode must be one of {F0_VARIANCE_MODES}.")
    states = np.asarray(fit["init_cond"], dtype=float)
    if states.shape != (len(datasets), 2) or len(datasets) < 2 or not np.all(np.isfinite(states)):
        raise ValueError("WM4 covariance needs at least two finite historical [F0,R0] oracle states.")
    for data in datasets:
        if "I" not in data or "O" not in data:
            raise ValueError("Reconstruction uncertainty needs I and O in every historical window.")
    observed_R = np.array([
        float(np.asarray(data["I"], dtype=float)[0] - np.asarray(data["O"], dtype=float)[0])
        for data in datasets], dtype=float)
    if not np.all(np.isfinite(observed_R)):
        raise ValueError("Historical observed rate estimates must be finite.")
    oracle_mean_F = float(np.mean(states[:, 0]))
    bootstrap_errors = np.column_stack([
        oracle_mean_F - states[:, 0], observed_R - states[:, 1]])
    errors = bootstrap_errors.copy()
    kinds = ["oracle_bootstrap"] * len(datasets)
    training_origins = fit.get("training_origins")
    archive = {} if state_archive is None else state_archive
    if training_origins is not None and len(training_origins) != len(datasets):
        raise ValueError("Historical training origins must align with oracle states.")
    prediction_origin = pd.Timestamp(fit["prediction_origin"]) if "prediction_origin" in fit else None
    if f0_variance_mode == "mpc_reconstruction" and training_origins is not None:
        for i, origin in enumerate(training_origins):
            origin = pd.Timestamp(origin)
            if origin not in archive: continue
            if prediction_origin is not None and origin + pd.Timedelta(
                hours=int(fit.get("fit_hours", DAY_HOURS))) > prediction_origin:
                continue
            record = np.asarray(archive[origin], dtype=float).reshape(-1)
            if record.shape != (2,) or not np.all(np.isfinite(record)):
                raise ValueError(f"Invalid archived F/R oracle reconstruction errors at {origin}.")
            errors[i] = record
            kinds[i] = "mpc"
    reconstruction_bias = np.mean(errors, axis=0)
    reconstruction_covariance = np.cov(errors, rowvar=False, ddof=1)
    reconstruction_covariance = 0.5 * (reconstruction_covariance + reconstruction_covariance.T)
    if f0_variance_mode == "mpc_reconstruction":
        F_variance = float(max(reconstruction_covariance[0, 0], 0.0))
        F_bias_correction = float(reconstruction_bias[0])
    else:
        F_variance = float(np.var(states[:, 0], ddof=1))
        F_bias_correction = 0.0
    R_variance = (float(max(reconstruction_covariance[1, 1], 0.0))
        if r0_variance_mode == "oracle_reconstruction"
        else float(np.var(observed_R, ddof=1)) if r0_variance_mode == "baseline_residual" else 0.0)
    R_bias_correction = float(reconstruction_bias[1]) if r0_variance_mode == "oracle_reconstruction" else 0.0
    FR_covariance = (float(reconstruction_covariance[0, 1])
        if f0_variance_mode == "mpc_reconstruction" and r0_variance_mode == "oracle_reconstruction" else 0.0)
    P0 = np.array([[F_variance, FR_covariance], [FR_covariance, R_variance]], dtype=float)
    if np.linalg.eigvalsh(P0)[0] < -1e-10:
        raise ValueError("Empirical initial reconstruction covariance must be positive semidefinite.")
    return {
        "covariance": P0, "F_variance": F_variance, "R_variance": R_variance,
        "FR_covariance": FR_covariance,
        "F_bias_correction": F_bias_correction, "R_bias_correction": R_bias_correction,
        "F_reconstruction_bias": float(reconstruction_bias[0]),
        "F_reconstruction_variance": float(max(reconstruction_covariance[0, 0], 0.0)),
        "F_reconstruction_rmse": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
        "R_reconstruction_bias": float(reconstruction_bias[1]),
        "R_reconstruction_variance": float(max(reconstruction_covariance[1, 1], 0.0)),
        "R_reconstruction_rmse": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
        "FR_reconstruction_covariance": float(reconstruction_covariance[0, 1]),
        "reconstruction_errors": errors.copy(),
        "R_reconstruction_errors": errors[:, 1].copy(),
        "reconstruction_samples": int(len(errors)),
        "R_reconstruction_samples": int(len(errors)),
        "F_calibration_mpc_count": kinds.count("mpc"),
        "F_calibration_oracle_count": kinds.count("oracle_bootstrap"),
        "F_calibration_origin_types": kinds,
        "baseline_residual_variance": float(np.var(observed_R, ddof=1)),
        "baseline_residual_mean": float(np.mean(observed_R)),
        "baseline_residual_samples": int(len(observed_R)),
        "r0_variance_mode": r0_variance_mode, "f0_variance_mode": f0_variance_mode,
    }


def _archive_reconstruction_origin(state_archive, origin, state):
    """Store post-hoc oracle mismatch for use by future weekly origins only."""
    if state_archive is None: return
    origin = pd.Timestamp(origin)
    if pd.Timestamp(state["origin"]) != origin: raise ValueError("Archive origin mismatch.")
    errors = np.array([state["F_reconstruction_error_origin"],
        state["R_reconstruction_error_origin"]], dtype=float)
    if not np.all(np.isfinite(errors)): raise ValueError("Oracle reconstruction errors must be finite.")
    state_archive[origin] = errors


def _archive_mpc_origin(archive, origin, rows, horizon=8):
    """Archive completed forward MPC errors; never infer errors from oracle states."""
    if archive is None: return
    origin = pd.Timestamp(origin)
    if not isinstance(rows, (list, tuple)) or len(rows) < horizon: return
    by_lead = {int(row["lead_hours"]): row for row in rows}
    if not all(h in by_lead for h in range(1, horizon + 1)): return
    values = []
    for h in range(1, horizon + 1):
        row = by_lead[h]
        if pd.Timestamp(row["target_time"]) != origin + pd.Timedelta(hours=h): return
        values.append([float(row["forecast_error"]), float(row["initial_variance"]),
            float(row["process_kernel_unit"])])
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values[:, 1:] < 0.0): return
    previous = archive.get(origin)
    if previous is None or np.asarray(previous).shape[0] < len(values):
        archive[origin] = values


def _archive_mpc_calibration(archive, origin, model, state, prepared, fit, diffusion,
    steps_per_hour, horizon=SIGMA_HOURS):
    """Archive a complete forward MPC trajectory and its origin-specific P0/J/K."""
    if archive is None: return
    horizon, steps_per_hour = int(horizon), int(steps_per_hour)
    if horizon < 1 or horizon > SIGMA_HOURS or horizon * steps_per_hour >= model.t_steps:
        raise ValueError("Invalid calibration archive horizon.")
    stochastic = _stochastic_forecast_variance(model, state, fit, diffusion, horizon, steps_per_hour)
    records = []
    for h in range(1, horizon + 1):
        k = h * steps_per_hour
        records.append({"lead_hours": h, "target_time": origin + pd.Timedelta(hours=h),
            "forecast_error": float(prepared["I"][k] - state["I"][k]),
            "initial_variance": float(stochastic["initial_variance"][h - 1]),
            "process_kernel_unit": float(stochastic["process_kernel_unit"][h - 1])})
    _archive_mpc_origin(archive, origin, records, horizon)


def _archive_mpc_24h(archive, origin, model, state, prepared, fit, diffusion, steps_per_hour):
    """Compatibility wrapper for the 24-hour rolling estimator."""
    return _archive_mpc_calibration(archive, origin, model, state, prepared,
        fit, diffusion, steps_per_hour, horizon=SIGMA_HOURS)


def _rolling_sigma_training(archive, origin, days=SIGMA_DAYS, horizon=SIGMA_HOURS):
    """Extract completed historical MPC errors at the same hour of the last 24 days."""
    origin, horizon = pd.Timestamp(origin), int(horizon)
    if not 1 <= horizon <= SIGMA_HOURS: raise ValueError("Invalid rolling sigma horizon.")
    selected, stamps = [], []
    for k in range(1, int(days) + 1):
        historical = origin - pd.Timedelta(days=k)
        if historical + pd.Timedelta(hours=horizon) > origin: continue
        record = archive.get(historical)
        if record is None: continue
        values = np.asarray(record, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < horizon:
            continue
        values = values[:horizon]
        if not np.all(np.isfinite(values)) or np.any(values[:, 1:] < 0): continue
        selected.append(values)
        stamps.append(historical)
    return {"values": np.asarray(selected, dtype=float), "origins": stamps}


def _fit_sigma_rolling_qp(model, rolling, horizon=SIGMA_HOURS):
    """Convex QP on 6 or 24 lead-specific sample variances across historical MPC paths."""
    samples, horizon = np.asarray(rolling["values"], dtype=float), int(horizon)
    if samples.ndim != 3 or samples.shape[1:] != (horizon, 3) or len(samples) < 2:
        raise ValueError(f"Rolling sigma QP requires two or more complete {horizon}h MPC trajectories.")
    observed = np.var(samples[:, :, 0], axis=0, ddof=1)
    initial = np.mean(samples[:, :, 1], axis=0)
    response = np.mean(samples[:, :, 2], axis=0)
    fit = constant_diffusion_qp(model).fit(response[:, None], observed,
        initial_variance=initial, fit_mask=np.ones(horizon, dtype=bool))
    return fit, observed, initial, response


def _fit_stochastic_variance(model, fit, datasets, steps_per_hour, sigma_fit_mode="mpc_replay",
    r0_variance_mode="oracle_reconstruction", mpc_archive=None, max_horizon=8,
    *, f0_variance_mode="mpc_reconstruction", state_archive=None, rolling_days=None):
    """Fit q=sigma^2 from causal historical MPC errors, with oracle bootstrap.

    Each of the five aligned historical origins contributes a forward MPC
    trajectory if already archived. Unavailable early-history origins use
    oracle-centred historical residuals as bootstrap only. The initial
    covariance for this fit and for the final forecast is always the same
    oracle-reconstruction covariance. The current trajectory's future
    observations are never used to estimate its own initial state.
    """
    if sigma_fit_mode not in SIGMA_FIT_MODES:
        raise ValueError(f"sigma_fit_mode must be one of {SIGMA_FIT_MODES}.")
    rolling_days = int(ROLLING_SIGMA_DAYS.get(sigma_fit_mode, SIGMA_DAYS)
        if rolling_days is None else rolling_days)
    if rolling_days < 2: raise ValueError("rolling_days must be at least 2.")
    initial = _mpc_initial_covariance(fit, datasets, r0_variance_mode,
        f0_variance_mode=f0_variance_mode, state_archive=state_archive)
    P0 = initial["covariance"]
    fit_hours = (model.t_steps - 1) // int(steps_per_hour)
    if fit_hours < 2 or fit_hours * int(steps_per_hour) + 1 != model.t_steps:
        raise ValueError("WM4 sigma calibration needs a whole-hour fitting grid.")
    hourly_index = np.arange(fit_hours, dtype=int) * int(steps_per_hour)
    states = np.asarray(fit["init_cond"], dtype=float)
    if len(fit["deterministic_states"]) != len(datasets):
        raise ValueError("Historical WM4 deterministic states must align with fitting datasets.")

    x_hat = np.mean(states, axis=0)
    J = np.asarray(model.sensitivity({"r": fit["r"]})["input"], dtype=float)[hourly_index]
    propagated_initial_variance = np.maximum(np.einsum("ni,ij,nj->n", J, P0, J), 0.0)
    fitted_residuals = np.vstack([
        np.asarray(data["I"], dtype=float)[hourly_index] - np.asarray(state["I"], dtype=float)[hourly_index]
        for data, state in zip(datasets, fit["deterministic_states"])
    ])
    # Oracle-centred bootstrap: equivalent to all oracle trajectories
    # reinitialized at the shared historical F/R mean, retaining their
    # observed residuals. This is NOT a forward MPC prediction.
    centered_residuals = fitted_residuals + (states - x_hat) @ J.T
    fitted_variance = np.var(fitted_residuals, axis=0, ddof=1)
    centered_variance = np.var(centered_residuals, axis=0, ddof=1)
    kernel = model.process_variance_kernel({"r": fit["r"]}, hourly_index)

    calibration_mpc_count, calibration_oracle_count = 0, 0
    calibration_origin_types = []
    rolling_horizon = ROLLING_SIGMA_HORIZONS.get(sigma_fit_mode)
    rolling = (_rolling_sigma_training({} if mpc_archive is None else mpc_archive,
        fit["prediction_origin"], days=rolling_days, horizon=rolling_horizon)
        if rolling_horizon is not None else None)
    rolling_count = len(rolling["origins"]) if rolling is not None else 0
    rolling_active = rolling_count >= 2
    if rolling_active:
        diffusion, observed_variance, initial_variance, response = _fit_sigma_rolling_qp(
            model, rolling, horizon=rolling_horizon)
        calibration_errors = rolling["values"][:, :, 0]
        calibration_mpc_count = rolling_count
        calibration_origin_types = ["daily_mpc"] * rolling_count
    elif sigma_fit_mode == "mpc_replay" or rolling_horizon is not None:
        # Before two complete historical rolling paths: bootstrap with five-week 8h fit.
        horizon = int(max_horizon)
        if horizon < 1 or horizon > 8: raise ValueError("max_horizon must be in [1, 8].")
        indices = np.arange(1, horizon + 1, dtype=int) * int(steps_per_hour)
        if np.max(indices) >= model.t_steps: raise ValueError("Historical calibration horizon exceeds the WM4 time grid.")
        bootstrap_kernel = model.process_variance_kernel({"r": fit["r"]}, indices)
        bootstrap_response = bootstrap_kernel @ np.ones(model.t_steps, dtype=float)
        bootstrap_J = np.asarray(model.sensitivity({"r": fit["r"]})["input"], dtype=float)[indices]
        bootstrap_initial = np.maximum(np.einsum("ni,ij,nj->n", bootstrap_J, P0, bootstrap_J), 0.0)
        archive = {} if mpc_archive is None else mpc_archive
        training_origins = fit.get("training_origins")
        if training_origins is None or len(training_origins) != len(datasets):
            raise ValueError("MPC replay needs the exact aligned historical training origins.")
        errors, initial_samples, kernel_samples = [], [], []
        for i, historical_origin in enumerate(training_origins):
            historical_origin = pd.Timestamp(historical_origin)
            archived = archive.get(historical_origin)
            if archived is not None and rolling_horizon is not None:
                # The rolling sigma archive intentionally stores only its
                # 6h (or 24h) MPC path. During early bootstrap, a 6h path
                # cannot fill an 8h weekly sample: use the oracle bootstrap.
                values = np.asarray(archived, dtype=float)
                if values.ndim == 2 and values.shape[0] < horizon: archived = None
            if archived is not None:
                archived = np.asarray(archived, dtype=float)
                if archived.ndim != 2 or archived.shape[1] != 3 or archived.shape[0] < horizon:
                    raise ValueError(f"Incomplete MPC calibration record at {historical_origin}.")
                if not np.all(np.isfinite(archived[:horizon])) or np.any(archived[:horizon, 1:] < 0.0):
                    raise ValueError(f"Invalid MPC calibration record at {historical_origin}.")
                errors.append(archived[:horizon, 0])
                initial_samples.append(archived[:horizon, 1])
                kernel_samples.append(archived[:horizon, 2])
                calibration_mpc_count += 1
                calibration_origin_types.append("mpc")
            else:
                errors.append(centered_residuals[i, 1:horizon + 1])
                initial_samples.append(bootstrap_initial.copy())
                kernel_samples.append(bootstrap_response.copy())
                calibration_oracle_count += 1
                calibration_origin_types.append("oracle_bootstrap")
        calibration_errors = np.vstack(errors)
        initial_samples = np.vstack(initial_samples)
        kernel_samples = np.vstack(kernel_samples)
        observed_variance = np.var(calibration_errors, axis=0, ddof=1)
        initial_variance = np.mean(initial_samples, axis=0)
        response = np.mean(kernel_samples, axis=0)
        # One column of unit-q responses is sufficient for the existing QP.
        calibration_kernel = response[:, None]
        fit_mask = np.ones(horizon, dtype=bool)
        diffusion = constant_diffusion_qp(model).fit(
            calibration_kernel, observed_variance,
            initial_variance=initial_variance, fit_mask=fit_mask)
    else:
        observed_variance = centered_variance if sigma_fit_mode == "historical_p0" else fitted_variance
        initial_variance = propagated_initial_variance if sigma_fit_mode == "historical_p0" else np.zeros(fit_hours)
        fit_mask = np.arange(fit_hours, dtype=int) > 0
        diffusion = constant_diffusion_qp(model).fit(
            kernel, observed_variance, initial_variance=initial_variance, fit_mask=fit_mask)
        calibration_errors = centered_residuals if sigma_fit_mode == "historical_p0" else fitted_residuals
        response = kernel @ np.ones(model.t_steps, dtype=float)
        calibration_origin_types = ["oracle_fit"] * len(datasets)

    return {
        **diffusion, **initial,
        "sigma_fit_mode": sigma_fit_mode,
        "mpc_initial_covariance": P0.copy(),
        "training_residual_variance": observed_variance,
        "training_residuals": calibration_errors,
        "training_fit_residual_variance": fitted_variance,
        "training_centered_residual_variance": centered_variance,
        "training_initial_variance": initial_variance,
        "training_initial_covariance": np.cov(states, rowvar=False, ddof=1),
        "training_initial_state_mean": x_hat,
        "calibration_kernel_unit": response,
        "calibration_mpc_count": calibration_mpc_count,
        "calibration_oracle_count": calibration_oracle_count,
        "calibration_origin_types": calibration_origin_types,
        "calibration_horizon": int(len(observed_variance)),
        "sigma_rolling_days": int(rolling_days) if rolling_horizon is not None else 0,
        "sigma_rolling_count": int(rolling_count),
        "sigma_rolling_complete": bool(rolling_active and rolling_count == int(rolling_days)),
        "sigma_rolling_bootstrap": bool(rolling_horizon is not None and not rolling_active),
    }


def _stochastic_forecast_variance(model, state, fit, diffusion, max_horizon, steps_per_hour):
    lead_steps = np.arange(1, int(max_horizon) + 1, dtype=int) * int(steps_per_hour)
    sensitivity = model.sensitivity({"r": fit["r"]})["input"][lead_steps]
    P0 = np.asarray(diffusion["mpc_initial_covariance"], dtype=float)
    initial_F_component = P0[0, 0] * sensitivity[:, 0] ** 2
    initial_R_component = P0[1, 1] * sensitivity[:, 1] ** 2
    initial_FR_component = 2.0 * P0[0, 1] * sensitivity[:, 0] * sensitivity[:, 1]
    initial_variance = np.maximum(initial_F_component + initial_R_component + initial_FR_component, 0.0)
    kernel = model.process_variance_kernel(state, lead_steps)
    process_kernel_unit = kernel @ np.ones(model.t_steps, dtype=float)
    process_variance = process_kernel_unit * float(diffusion["q"])
    total_variance = np.maximum(initial_variance + process_variance, 0.0)
    return {
        "F_variance": float(P0[0, 0]), "R_variance": float(P0[1, 1]),
        "FR_covariance": float(P0[0, 1]),
        "initial_F_component": initial_F_component, "initial_R_component": initial_R_component,
        "initial_FR_component": initial_FR_component,
        "initial_variance": initial_variance,
        "process_variance": process_variance,
        "process_kernel_unit": process_kernel_unit,
        "variance": total_variance,
        "std": np.sqrt(total_variance),
    }


def _mpc_origin(frame, origin, baseline_weeks, steps_per_hour, output_model, max_horizon, *, sigma_fit_mode="mpc_replay", r0_variance_mode="oracle_reconstruction",
    f0_variance_mode="mpc_reconstruction", mpc_archive=None, state_archive=None,
    sigma_rolling_days=None, fit_hours=DAY_HOURS, **fit_kwargs):
    origin = pd.Timestamp(origin)
    if sigma_fit_mode == "mpc_replay_28d_6h" and (max_horizon != 6 or fit_hours != 12):
        raise ValueError("mpc_replay_28d_6h requires 12-hour parameter/baseline windows and 6-hour MPC forecasts.")
    fit, fit_model, datasets = _fit_shifted_history(
        frame, origin, baseline_weeks, steps_per_hour, output_model,
        fit_hours=fit_hours, **fit_kwargs,
    )

    data, fine_index = _window_input(frame, origin, steps_per_hour,
        output_model, hours=fit_hours)
    if max_horizon >= fit_hours:
        raise ValueError("MPC prediction horizon must be shorter than the fitted model window.")
    model = wm_4((0.0, 1.0), len(fine_index), model_output(data, fine_index))
    helper = optimized_qp_params(model)
    prepared = helper._prepare_data(data)

    historical = np.asarray(fit["x_hat"], dtype=float)
    R_observed = float(prepared["I"][0] - prepared["O"][0])
    diffusion = _fit_stochastic_variance(fit_model, fit, datasets, steps_per_hour,
        sigma_fit_mode=sigma_fit_mode, r0_variance_mode=r0_variance_mode,
        mpc_archive=mpc_archive, max_horizon=max_horizon,
        f0_variance_mode=f0_variance_mode, state_archive=state_archive,
        rolling_days=sigma_rolling_days)
    # WM4's flexibility state must remain in [-1, 1] after bias correction.
    F_initialized = float(np.clip(historical[0] - diffusion["F_bias_correction"], -1.0, 1.0))
    F_bias_applied = float(historical[0] - F_initialized)
    R_initialized = float(R_observed - diffusion["R_bias_correction"])
    init = np.array([F_initialized, R_initialized], dtype=float)
    state = helper._state_from_parameters(prepared, init, fit["r"], full=True)
    stochastic = _stochastic_forecast_variance(
        model, state, fit, diffusion, max_horizon, steps_per_hour)
    # Retrospective oracle reconstruction is for future historical calibration
    # only; the CURRENT forecast mean, q, and P0 have already been fixed.
    oracle = helper.optimize_prediction_initial_state(data, fit["r"],
        r_bounds=(-np.inf, np.inf))["init_cond"]
    oracle = np.asarray(oracle, dtype=float)
    if oracle.shape != (2,) or not np.all(np.isfinite(oracle)):
        raise ValueError("Retrospective current-origin WM4 oracle state must be finite [F0,R0].")

    rolling_horizon = ROLLING_SIGMA_HORIZONS.get(sigma_fit_mode)
    if rolling_horizon is not None:
        # The rolling archive must include the REAL horizon endpoint.
        # Otherwise _window_input may interpolate missing future observations.
        endpoint = origin + pd.Timedelta(hours=rolling_horizon)
        if endpoint in frame.index and np.isfinite(float(frame.loc[endpoint, "I"])):
            _archive_mpc_calibration(mpc_archive, origin, model, state, prepared,
                fit, diffusion, steps_per_hour, horizon=rolling_horizon)

    rows = []
    for lead in range(1, int(max_horizon) + 1):
        step = lead * int(steps_per_hour)
        target_time = origin + pd.Timedelta(hours=lead)
        variance = float(stochastic["variance"][lead - 1])
        std = float(stochastic["std"][lead - 1])
        error = float(prepared["I"][step] - state["I"][step])
        z = error / std if std > 1e-12 else np.nan
        rows.append({
            "origin": origin,
            "origin_day": origin.normalize(),
            "origin_hour": origin.hour,
            "lead_hours": lead,
            "target_time": target_time,
            "I_observed": float(prepared["I"][step]),
            "I_mpc": float(state["I"][step]),
            "O": float(state["O"][step]),
            "F_predicted": float(state["F"][step]),
            "R_predicted": float(state["R"][step]),
            "F_historical": float(historical[0]),
            "F_initialized_origin": F_initialized,
            "F_oracle_origin": float(oracle[0]),
            "F_reconstruction_error_origin": float(historical[0] - oracle[0]),
            "F_corrected_error_origin": float(F_initialized - oracle[0]),
            "R_oracle_origin": float(oracle[1]),
            "R_reconstruction_error_origin": float(R_observed - oracle[1]),
            "f0_variance_mode": f0_variance_mode,
            "F_bias_correction": F_bias_applied,
            "F_bias_correction_estimated": float(diffusion["F_bias_correction"]),
            "F_reconstruction_bias": float(diffusion["F_reconstruction_bias"]),
            "F_reconstruction_variance": float(diffusion["F_reconstruction_variance"]),
            "F_reconstruction_rmse": float(diffusion["F_reconstruction_rmse"]),
            "FR_reconstruction_covariance": float(diffusion["FR_reconstruction_covariance"]),
            "initial_FR_covariance": float(stochastic["FR_covariance"]),
            "initial_FR_component_variance": float(stochastic["initial_FR_component"][lead - 1]),
            "F_calibration_mpc_count": int(diffusion["F_calibration_mpc_count"]),
            "F_calibration_oracle_count": int(diffusion["F_calibration_oracle_count"]),
            "R_observed_origin": R_observed,
            "R_initialized_origin": R_initialized,
            "R_reconstruction_bias": float(diffusion["R_reconstruction_bias"]),
            "R_reconstruction_variance": float(diffusion["R_reconstruction_variance"]),
            "R_reconstruction_rmse": float(diffusion["R_reconstruction_rmse"]),
            "R_reconstruction_samples": int(diffusion["R_reconstruction_samples"]),
            "R_bias_correction": float(diffusion["R_bias_correction"]),
            "r_F": float(fit["r"][0]),
            "r_U": float(fit["r"][1]),
            "r_R": float(fit["r"][2]),
            "training_F_mean": float(np.mean(np.asarray(fit["init_cond"], dtype=float)[:, 0])),
            "training_F_std": float(np.std(np.asarray(fit["init_cond"], dtype=float)[:, 0])),
            "training_R_mean": float(np.mean(np.asarray(fit["init_cond"], dtype=float)[:, 1])),
            "training_R_std": float(np.std(np.asarray(fit["init_cond"], dtype=float)[:, 1])),
            "state_iterations": int(fit["state_iterations"]),
            "qp_loss": float(fit["qp_loss"]),
            "diffusion_q": float(diffusion["q"]),
            "diffusion_sigma": float(diffusion["sigma"]),
            "sigma_fit_mode": sigma_fit_mode,
            "parameter_fit_hours": int(fit_hours),
            "baseline_window_hours": int(fit_hours),
            "calibration_mpc_count": int(diffusion["calibration_mpc_count"]),
            "calibration_oracle_count": int(diffusion["calibration_oracle_count"]),
            "calibration_horizon": int(diffusion["calibration_horizon"]),
            "sigma_rolling_days": int(diffusion["sigma_rolling_days"]),
            "sigma_rolling_count": int(diffusion["sigma_rolling_count"]),
            "sigma_rolling_complete": bool(diffusion["sigma_rolling_complete"]),
            "sigma_rolling_bootstrap": bool(diffusion["sigma_rolling_bootstrap"]),
            "r0_variance_mode": r0_variance_mode,
            "initial_R_variance": float(stochastic["R_variance"]),
            "initial_F_component_variance": float(stochastic["initial_F_component"][lead - 1]),
            "initial_R_component_variance": float(stochastic["initial_R_component"][lead - 1]),
            "baseline_origin_residual_variance": float(diffusion["baseline_residual_variance"]),
            "training_initial_variance_mean": float(np.mean(diffusion["training_initial_variance"])),
            "training_centered_variance_mean": float(np.mean(diffusion["training_centered_residual_variance"])),
            "initial_F_variance": float(stochastic["F_variance"]),
            "initial_variance": float(stochastic["initial_variance"][lead - 1]),
            "process_variance": float(stochastic["process_variance"][lead - 1]),
            "process_kernel_unit": float(stochastic["process_kernel_unit"][lead - 1]),
            "predicted_variance": variance,
            "predicted_std": std,
            "forecast_error": error,
            "z_score": z,
            "within_1sigma": bool(abs(error) <= std),
            "within_95": bool(abs(error) <= 1.96 * std),
            "lower_68": float(state["I"][step] - std),
            "upper_68": float(state["I"][step] + std),
            "lower_95": float(state["I"][step] - 1.96 * std),
            "upper_95": float(state["I"][step] + 1.96 * std),
        })
    state_row = {
        "origin": origin,
        "origin_day": origin.normalize(),
        "origin_hour": origin.hour,
        "F_historical": float(historical[0]),
        "F_initialized": F_initialized,
        "F_oracle_origin": float(oracle[0]),
        "F_reconstruction_error_origin": float(historical[0] - oracle[0]),
        "F_corrected_error_origin": float(F_initialized - oracle[0]),
        "R_oracle_origin": float(oracle[1]),
        "R_reconstruction_error_origin": float(R_observed - oracle[1]),
        "f0_variance_mode": f0_variance_mode,
        "F_bias_correction": F_bias_applied,
        "F_bias_correction_estimated": float(diffusion["F_bias_correction"]),
        "F_reconstruction_bias": float(diffusion["F_reconstruction_bias"]),
        "F_reconstruction_variance": float(diffusion["F_reconstruction_variance"]),
        "F_reconstruction_rmse": float(diffusion["F_reconstruction_rmse"]),
        "FR_reconstruction_covariance": float(diffusion["FR_reconstruction_covariance"]),
        "F_calibration_mpc_count": int(diffusion["F_calibration_mpc_count"]),
        "F_calibration_oracle_count": int(diffusion["F_calibration_oracle_count"]),
        "R_historical": float(historical[1]),
        "R_observed": R_observed,
        "R_initialized": R_initialized,
        "R_reconstruction_bias": float(diffusion["R_reconstruction_bias"]),
        "R_reconstruction_variance": float(diffusion["R_reconstruction_variance"]),
        "R_reconstruction_rmse": float(diffusion["R_reconstruction_rmse"]),
        "R_reconstruction_samples": int(diffusion["R_reconstruction_samples"]),
        "R_bias_correction": float(diffusion["R_bias_correction"]),
        "I_observed": float(prepared["I"][0]),
        "O": float(prepared["O"][0]),
        "r_F": float(fit["r"][0]),
        "r_U": float(fit["r"][1]),
        "r_R": float(fit["r"][2]),
        "training_F_std": float(np.std(np.asarray(fit["init_cond"], dtype=float)[:, 0])),
        "training_R_std": float(np.std(np.asarray(fit["init_cond"], dtype=float)[:, 1])),
        "state_iterations": int(fit["state_iterations"]),
        "qp_loss": float(fit["qp_loss"]),
        "diffusion_q": float(diffusion["q"]),
        "diffusion_sigma": float(diffusion["sigma"]),
        "sigma_fit_mode": sigma_fit_mode,
        "calibration_mpc_count": int(diffusion["calibration_mpc_count"]),
        "calibration_oracle_count": int(diffusion["calibration_oracle_count"]),
        "calibration_horizon": int(diffusion["calibration_horizon"]),
        "sigma_rolling_days": int(diffusion["sigma_rolling_days"]),
        "sigma_rolling_count": int(diffusion["sigma_rolling_count"]),
        "sigma_rolling_complete": bool(diffusion["sigma_rolling_complete"]),
        "sigma_rolling_bootstrap": bool(diffusion["sigma_rolling_bootstrap"]),
        "r0_variance_mode": r0_variance_mode,
        "mpc_P0_FF": float(stochastic["F_variance"]),
        "mpc_P0_FR": float(stochastic["FR_covariance"]),
        "mpc_P0_RR": float(stochastic["R_variance"]),
        "baseline_origin_residual_variance": float(diffusion["baseline_residual_variance"]),
        "baseline_origin_residual_mean": float(diffusion["baseline_residual_mean"]),
        "baseline_origin_residual_samples": int(diffusion["baseline_residual_samples"]),
        "R_reconstruction_errors": ";".join(f"{v:.9g}" for v in diffusion["R_reconstruction_errors"]),
        "F_reconstruction_errors": ";".join(f"{v:.9g}" for v in diffusion["reconstruction_errors"][:, 0]),
        "F_calibration_origin_types": ";".join(diffusion["F_calibration_origin_types"]),
        "parameter_fit_hours": int(fit_hours),
        "baseline_window_hours": int(fit_hours),
        "training_P0_FF": float(diffusion["training_initial_covariance"][0, 0]),
        "training_P0_FR": float(diffusion["training_initial_covariance"][0, 1]),
        "training_P0_RR": float(diffusion["training_initial_covariance"][1, 1]),
        "training_initial_variance_mean": float(np.mean(diffusion["training_initial_variance"])),
        "training_centered_variance_mean": float(np.mean(diffusion["training_centered_residual_variance"])),
        "training_fit_residual_variance_mean": float(np.mean(diffusion["training_fit_residual_variance"])),
        "initial_F_variance": float(stochastic["F_variance"]),
        "initial_R_variance": float(stochastic["R_variance"]),
        "training_diffusion_variance_rmse": float(diffusion["variance_rmse"]),
        "calibration_origin_types": ";".join(diffusion["calibration_origin_types"]),
        "training_diffusion_std_rmse": float(diffusion["std_rmse"]),
    }
    return rows, state_row


def _stochastic_metrics(part):
    error = part["forecast_error"].to_numpy(dtype=float)
    variance = part["predicted_variance"].to_numpy(dtype=float)
    std = np.sqrt(np.maximum(variance, 0.0))
    valid = np.isfinite(part["z_score"].to_numpy(dtype=float))
    z = part.loc[valid, "z_score"].to_numpy(dtype=float)
    error_std = float(np.std(error, ddof=1)) if error.size > 1 else 0.0
    rms_std = float(np.sqrt(np.mean(variance)))
    z_mean = float(np.mean(z)) if z.size else np.nan
    z_std = float(np.std(z, ddof=1)) if z.size > 1 else np.nan
    if z.size > 2 and z_std > 0.0:
        centered = (z - z_mean) / z_std
        z_skew = float(np.mean(centered ** 3))
        z_excess_kurtosis = float(np.mean(centered ** 4) - 3.0)
    else:
        z_skew, z_excess_kurtosis = np.nan, np.nan
    return {
        "empirical_error_std": error_std,
        "predicted_rms_std": rms_std,
        "std_to_rmse_ratio": rms_std / float(np.sqrt(np.mean(error ** 2))) if np.any(np.isfinite(error)) and np.mean(error ** 2) > 0.0 else np.nan,
        "coverage_68_pct": 100.0 * float(np.mean(part["within_1sigma"].to_numpy(dtype=bool))),
        "coverage_95_pct": 100.0 * float(np.mean(part["within_95"].to_numpy(dtype=bool))),
        "z_mean": z_mean,
        "z_std": z_std,
        "z_skew": z_skew,
        "z_excess_kurtosis": z_excess_kurtosis,
        "mean_sigma": float(part["diffusion_sigma"].mean()),
        "mean_initial_F_std": float(np.sqrt(np.maximum(part["initial_F_variance"].to_numpy(dtype=float), 0.0)).mean()),
        "mean_initial_R_std": float(np.sqrt(np.maximum(part["initial_R_variance"].to_numpy(dtype=float), 0.0)).mean()),
    }


def summary_by_lead(forecasts):
    rows = []
    for lead, part in forecasts.groupby("lead_hours", sort=True):
        observed = part["I_observed"].to_numpy(dtype=float)
        mpc = deviation_metrics(observed, part["I_mpc"].to_numpy(dtype=float))
        baseline = deviation_metrics(observed, part["O"].to_numpy(dtype=float))
        rows.append({
            "lead_hours": int(lead),
            "origins": int(len(part)),
            "days": int(part["origin_day"].nunique()),
            **{f"mpc_{key}": value for key, value in mpc.items()},
            **{f"O_{key}": value for key, value in baseline.items()},
            "mpc_gain_vs_O_pct": 100.0 * (1.0 - mpc["rmse"] / baseline["rmse"]) if baseline["rmse"] > 0.0 else np.nan,
            **_stochastic_metrics(part),
        })
    return pd.DataFrame(rows)


def cumulative_summary(forecasts, horizons=HORIZONS):
    rows = []
    for horizon in horizons:
        part = forecasts[forecasts["lead_hours"] <= int(horizon)]
        if part.empty: continue
        observed = part["I_observed"].to_numpy(dtype=float)
        mpc = deviation_metrics(observed, part["I_mpc"].to_numpy(dtype=float))
        baseline = deviation_metrics(observed, part["O"].to_numpy(dtype=float))
        rows.append({
            "horizon_hours": int(horizon),
            "origins": int(part["origin"].nunique()),
            "forecasts": int(len(part)),
            "days": int(part["origin_day"].nunique()),
            **{f"mpc_{key}": value for key, value in mpc.items()},
            **{f"O_{key}": value for key, value in baseline.items()},
            "mpc_gain_vs_O_pct": 100.0 * (1.0 - mpc["rmse"] / baseline["rmse"]) if baseline["rmse"] > 0.0 else np.nan,
            **_stochastic_metrics(part),
        })
    return pd.DataFrame(rows)


def summary_by_origin_hour(forecasts, max_horizon):
    rows = []
    part = forecasts[forecasts["lead_hours"] <= int(max_horizon)]
    for hour, group in part.groupby("origin_hour", sort=True):
        observed = group["I_observed"].to_numpy(dtype=float)
        mpc = deviation_metrics(observed, group["I_mpc"].to_numpy(dtype=float))
        baseline = deviation_metrics(observed, group["O"].to_numpy(dtype=float))
        rows.append({
            "origin_hour": int(hour),
            "days": int(group["origin_day"].nunique()),
            **{f"mpc_{key}": value for key, value in mpc.items()},
            **{f"O_{key}": value for key, value in baseline.items()},
            "mpc_gain_vs_O_pct": 100.0 * (1.0 - mpc["rmse"] / baseline["rmse"]) if baseline["rmse"] > 0.0 else np.nan,
        })
    return pd.DataFrame(rows)


def save_plot(cumulative, lead, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

    h = cumulative["horizon_hours"].to_numpy(dtype=int)
    axes[0].plot(h, cumulative["mpc_rmse"], marker="o", label="WM4 MPC")
    axes[0].plot(h, cumulative["O_rmse"], marker="o", label="O(t)")
    axes[0].set_ylabel("Cumulative RMSE")
    axes[0].set_xlabel("Prediction horizon from MPC update (hours)")
    axes[0].set_xticks(h)
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    lead_h = lead["lead_hours"].to_numpy(dtype=int)
    axes[1].plot(lead_h, lead["mpc_rmse"], marker="o", label="WM4 MPC")
    axes[1].plot(lead_h, lead["O_rmse"], marker="o", label="O(t)")
    axes[1].set_ylabel("Exact-lead RMSE")
    axes[1].set_xlabel("Lead time (hours)")
    axes[1].set_xticks(lead_h)
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("WM4 receding-horizon MPC forecast vs baseline O(t)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_stochastic_plot(lead, path):
    lead_h = lead["lead_hours"].to_numpy(dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(lead_h, lead["mpc_rmse"], marker="o", label="Empirical MPC RMSE")
    axes[0].plot(lead_h, lead["predicted_rms_std"], marker="o", label="Predicted stochastic RMS std")
    axes[0].set_ylabel("Error / predicted standard deviation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(lead_h, lead["coverage_68_pct"], marker="o", label="Observed ±1σ coverage")
    axes[1].plot(lead_h, lead["coverage_95_pct"], marker="o", label="Observed 95% coverage")
    axes[1].axhline(68.27, linestyle="--", linewidth=0.8, label="Gaussian 68.27%")
    axes[1].axhline(95.0, linestyle="--", linewidth=0.8, label="Gaussian 95%")
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_xlabel("Lead time (hours)")
    axes[1].set_xticks(lead_h)
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("WM4 MPC stochastic variance calibration")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(cumulative, lead, skipped):
    print("cumulative RMSE from each hourly MPC update:")
    print(f"{'hours':>7}{'MPC':>12}{'O(t)':>12}{'gain':>12}{'origins':>12}")
    for _, row in cumulative.iterrows():
        print(
            f"{int(row['horizon_hours']):>7d}"
            f"{row['mpc_rmse']:>12.8f}"
            f"{row['O_rmse']:>12.8f}"
            f"{row['mpc_gain_vs_O_pct']:>11.2f}%"
            f"{int(row['origins']):>12d}"
        )

    print()
    print("RMSE by exact lead time:")
    print(f"{'lead':>7}{'MPC':>12}{'O(t)':>12}{'gain':>12}")
    for _, row in lead.iterrows():
        print(
            f"{int(row['lead_hours']):>7d}"
            f"{row['mpc_rmse']:>12.8f}"
            f"{row['O_rmse']:>12.8f}"
            f"{row['mpc_gain_vs_O_pct']:>11.2f}%"
        )

    print()
    print("stochastic calibration by exact lead:")
    print(f"{'lead':>7}{'RMSE':>12}{'pred std':>12}{'ratio':>10}{'z mean':>10}{'z std':>10}{'cov68':>10}{'cov95':>10}")
    for _, row in lead.iterrows():
        print(
            f"{int(row['lead_hours']):>7d}"
            f"{row['mpc_rmse']:>12.8f}"
            f"{row['predicted_rms_std']:>12.8f}"
            f"{row['std_to_rmse_ratio']:>10.3f}"
            f"{row['z_mean']:>10.3f}"
            f"{row['z_std']:>10.3f}"
            f"{row['coverage_68_pct']:>9.2f}%"
            f"{row['coverage_95_pct']:>9.2f}%"
        )

    if skipped:
        print()
        print("skipped MPC origins:")
        for origin, reason in skipped: print(f"{origin}: {reason}")


def run_all_days(frame, start=START_DAY, end=END_DAY, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, max_horizon=None,
    fail_fast=False, mpc_archive=None, state_archive=None, fit_hours=None, **fit_kwargs):
    start, end = day_start(start), day_start(end)
    is_six_hour = fit_kwargs.get("sigma_fit_mode") == "mpc_replay_28d_6h"
    max_horizon = (6 if is_six_hour else 8) if max_horizon is None else int(max_horizon)
    fit_hours = (12 if is_six_hour else DAY_HOURS) if fit_hours is None else int(fit_hours)
    if is_six_hour and (max_horizon != 6 or fit_hours != 12):
        raise ValueError("mpc_replay_28d_6h requires 6-hour MPC and 12-hour fitting windows.")
    if fit_hours < 2 or fit_hours > DAY_HOURS:
        raise ValueError("fit_hours must be in [2, 24].")
    if not 1 <= max_horizon < fit_hours:
        raise ValueError("MPC horizon must be between 1 and fit_hours-1.")
    if end < start: raise ValueError("end must be on or after start.")
    forecast_rows, state_rows, skipped = [], [], []
    mpc_archive = {} if mpc_archive is None else mpc_archive
    state_archive = {} if state_archive is None else state_archive

    for day in pd.date_range(start, end, freq="D"):
        output_model, _ = fit_expected_output(frame, day)
        fitted = 0
        for hour in range(DAY_HOURS):
            origin = day + pd.Timedelta(hours=hour)
            try:
                rows, state = _mpc_origin(
                    frame, origin, baseline_weeks, steps_per_hour,
                    output_model, max_horizon, mpc_archive=mpc_archive,
                    state_archive=state_archive, fit_hours=fit_hours, **fit_kwargs,
                )
                if fit_kwargs.get("sigma_fit_mode", "mpc_replay") not in ROLLING_SIGMA_HORIZONS:
                    _archive_mpc_origin(mpc_archive, origin, rows, max_horizon)
                # Rolling modes archive only completed 6h or 24h MPC
                # calibration paths inside _mpc_origin. Do not overwrite
                # their six-hour records with eight-hour evaluation outputs.
                _archive_reconstruction_origin(state_archive, origin, state)
                forecast_rows.extend(rows)
                state_rows.append(state)
                fitted += 1
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                if fail_fast: raise
                skipped.append((origin, str(exc)))
        print(f"{day:%Y-%m-%d} | MPC origins fitted={fitted}/{DAY_HOURS} | skipped={DAY_HOURS - fitted}")

    if not forecast_rows: raise RuntimeError("No MPC origins in the requested range could be fitted.")
    return pd.DataFrame(forecast_rows), pd.DataFrame(state_rows), skipped


def main():
    parser = ArgumentParser(description="Evaluate WM4 receding-horizon MPC forecasting. At every hourly update, refit a shifted 24-hour WM4 model on corresponding historical weekdays, initialize F from the historical fitted state and R from the oracle-bias-corrected current residual I-O, then compare the next 1-8 forecast hours with baseline O(t).")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--max-horizon", type=int, default=None,
        help="MPC forecast horizon: 6h for 28-day/6h mode, otherwise 8h.")
    parser.add_argument("--fit-hours", type=int, default=None,
        help="Historical parameter and O(t) trajectory fitting window: 12h for 28-day/6h, otherwise 24h.")
    parser.add_argument("--f0-variance-mode", choices=F0_VARIANCE_MODES, default="mpc_reconstruction",
        help="Calibrate F0 bias, variance and F/R covariance from causal historical MPC-versus-oracle errors (default), or legacy oracle_spread.")
    parser.add_argument("--r0-variance-mode", choices=R0_VARIANCE_MODES, default="oracle_reconstruction",
        help="Initialize R0 uncertainty from historical (I-O_hat)-R_oracle reconstruction errors (default), baseline_residual or zero for ablations.")
    parser.add_argument("--sigma-fit-mode", choices=SIGMA_FIT_MODES, default="mpc_replay",
        help="mpc_replay: previous five matching weekdays, 8h (default). "
             "mpc_replay_24d_24h: last 24 days, 24h MPC errors. "
             "mpc_replay_24d_6h: last 24 days, six-hour MPC errors. "
             "mpc_replay_30d_6h: last 30 days, six-hour MPC errors. "
             "mpc_replay_28d_6h: last 28 days, 6-hour receding forecasts, 12-hour parameter/baseline window.")
    parser.add_argument("--sigma-rolling-days", type=int, default=None,
        help="Override rolling history length; default is 24 or 30 according to sigma-fit-mode.")
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    preset = args.sigma_fit_mode == "mpc_replay_28d_6h"
    args.max_horizon = (6 if preset else 8) if args.max_horizon is None else args.max_horizon
    args.fit_hours = (12 if preset else DAY_HOURS) if args.fit_hours is None else args.fit_hours
    if not 1 <= args.max_horizon <= 8: raise ValueError("max-horizon must lie in [1, 8].")
    if not 2 <= args.fit_hours <= 24 or args.max_horizon >= args.fit_hours:
        raise ValueError("Require 2 <= fit-hours <= 24 and max-horizon < fit-hours.")
    if preset and (args.max_horizon != 6 or args.fit_hours != 12):
        raise ValueError("28-day/6h mode requires --max-horizon 6 and --fit-hours 12.")
    if args.sigma_rolling_days is not None and args.sigma_rolling_days < 2:
        raise ValueError("sigma-rolling-days must be >=2.")

    frame, _ = aligned_wm4_data("MED")
    forecasts, states, skipped = run_all_days(
        frame,
        args.start_day,
        args.end_day,
        args.baseline_weeks,
        args.steps_per_hour,
        max_horizon=args.max_horizon,
        fit_hours=args.fit_hours,
        fail_fast=args.fail_fast,
        sigma_fit_mode=args.sigma_fit_mode,
        sigma_rolling_days=args.sigma_rolling_days,
        r0_variance_mode=args.r0_variance_mode,
        f0_variance_mode=args.f0_variance_mode,
        max_state_iter=args.max_state_iter,
        state_tol=args.state_tol,
    )

    lead = summary_by_lead(forecasts)
    horizons = tuple(h for h in HORIZONS if h <= args.max_horizon)
    cumulative = cumulative_summary(forecasts, horizons)
    origin_hour = summary_by_origin_hour(forecasts, args.max_horizon)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output / "mpc_forecasts.csv", index=False)
    states.to_csv(output / "mpc_states.csv", index=False)
    lead.to_csv(output / "lead_time_summary.csv", index=False)
    cumulative.to_csv(output / "cumulative_horizon_summary.csv", index=False)
    origin_hour.to_csv(output / "origin_hour_summary.csv", index=False)
    lead[[
        "lead_hours", "origins", "days", "mpc_bias", "mpc_rmse",
        "empirical_error_std", "predicted_rms_std", "std_to_rmse_ratio",
        "coverage_68_pct", "coverage_95_pct", "z_mean", "z_std",
        "z_skew", "z_excess_kurtosis", "mean_sigma", "mean_initial_F_std", "mean_initial_R_std",
    ]].to_csv(output / "stochastic_calibration_by_lead.csv", index=False)
    if skipped:
        pd.DataFrame([{"origin": origin, "reason": reason} for origin, reason in skipped]).to_csv(output / "skipped_origins.csv", index=False)

    save_plot(cumulative, lead, output / "mpc_vs_output_rmse.png")
    save_stochastic_plot(lead, output / "stochastic_variance_calibration.png")
    print()
    print(f"days={forecasts['origin_day'].nunique()}, MPC origins={forecasts['origin'].nunique()}, forecasts={len(forecasts)}")
    print_summary(cumulative, lead, skipped)
    print(f"fitting window={args.fit_hours}h; baseline O evaluated over {args.fit_hours}h; MPC horizon={args.max_horizon}h")
    print(f"sigma fit mode={args.sigma_fit_mode}; initial F0 variance mode={args.f0_variance_mode}; "
        f"initial R0 variance mode={args.r0_variance_mode} "
        "(oracle_reconstruction uses observed-minus-oracle initial-rate error)")
    print(f"output={output}")


if __name__ == "__main__": main()
