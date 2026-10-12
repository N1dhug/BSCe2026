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
SIGMA_FIT_MODES = ("mpc_replay", "historical_p0", "zero_p0")
R0_VARIANCE_MODES = ("baseline_residual", "zero")


def _window_input(frame, start, steps_per_hour, output_model):
    start = pd.Timestamp(start)
    steps_per_hour = int(steps_per_hour)
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2.")

    end = start + pd.Timedelta(hours=DAY_HOURS)
    n_steps = DAY_HOURS * steps_per_hour + 1
    elapsed = np.linspace(0.0, float(DAY_HOURS), n_steps)
    fine_index = start + pd.to_timedelta(elapsed, unit="h")

    source_index = pd.date_range(start, end, freq="h")
    source = frame.reindex(source_index)[["I", "U", "Temp"]].copy()
    if source.iloc[:-1].isna().any().any(): raise ValueError(f"Incomplete I/U/temperature data between {start} and {end}.")
    source["I"] = source["I"].interpolate(method="time", limit_direction="both").ffill().bfill()
    source["U"] = source["U"].ffill().bfill()
    source["Temp"] = source["Temp"].interpolate(method="time", limit_direction="both").ffill().bfill()

    hour_bin = np.minimum(np.floor(elapsed + 1e-12).astype(int), DAY_HOURS)
    I = np.interp(elapsed, np.arange(DAY_HOURS + 1, dtype=float), source["I"].to_numpy(dtype=float))
    U = source["U"].to_numpy(dtype=float)[hour_bin]
    Temp = np.interp(elapsed, np.arange(DAY_HOURS + 1, dtype=float), source["Temp"].to_numpy(dtype=float))

    observation_mask = np.zeros(n_steps, dtype=bool)
    observation_mask[np.arange(DAY_HOURS, dtype=int) * steps_per_hour] = True
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


def _fit_shifted_history(frame, origin, baseline_weeks, steps_per_hour, output_model, **fit_kwargs):
    origin, baseline_weeks = pd.Timestamp(origin), int(baseline_weeks)
    if baseline_weeks < 1: raise ValueError("baseline_weeks must be at least 1.")

    training_origins = [origin - pd.Timedelta(weeks=k) for k in range(1, baseline_weeks + 1)]
    datasets, indices, seeds = [], [], []

    for historical_origin in training_origins:
        data, fine_index = _window_input(frame, historical_origin, steps_per_hour, output_model)
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
    return fit, model, datasets


def _mpc_initial_covariance(fit, datasets, r0_variance_mode="baseline_residual"):
    """Use historical F0 spread and uncertainty in the baseline-derived R0.

    At an MPC origin R0=I0-O_hat0. Since the unobserved baseline is not
    directly measurable, cross-week variance of observed I0-O_hat0 provides a
    historical *proxy* for Var(O0 error), not an identified baseline-error
    variance. The proxy also contains actual historical flexibility variation.
    Only data preceding the prediction origin enters these estimates.
    """
    if r0_variance_mode not in R0_VARIANCE_MODES:
        raise ValueError(f"r0_variance_mode must be one of {R0_VARIANCE_MODES}.")
    states = np.asarray(fit["init_cond"], dtype=float)
    if states.shape != (len(datasets), 2) or len(datasets) < 2 or not np.all(np.isfinite(states)):
        raise ValueError("WM4 MPC covariance needs at least two finite fitted historical [F0,R0] states.")
    for data in datasets:
        if "I" not in data or "O" not in data: raise ValueError("Historical baseline uncertainty requires I and O for each training window.")
    baseline_residuals = np.array([
        float(np.asarray(data["I"], dtype=float)[0] - np.asarray(data["O"], dtype=float)[0])
        for data in datasets], dtype=float)
    if not np.all(np.isfinite(baseline_residuals)):
        raise ValueError("Historical baseline-origin residuals must be finite.")
    F_variance = float(np.var(states[:, 0], ddof=1))
    baseline_residual_variance = float(np.var(baseline_residuals, ddof=1))
    R_variance = baseline_residual_variance if r0_variance_mode == "baseline_residual" else 0.0
    P0 = np.diag([max(F_variance, 0.0), max(R_variance, 0.0)])
    return {
        "covariance": P0, "F_variance": float(P0[0, 0]),
        "R_variance": float(P0[1, 1]), "baseline_residual_variance": baseline_residual_variance,
        "baseline_residual_mean": float(np.mean(baseline_residuals)),
        "baseline_residual_samples": int(len(baseline_residuals)),
        "r0_variance_mode": r0_variance_mode,
    }


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
    archive[origin] = values


def _fit_stochastic_variance(model, fit, datasets, steps_per_hour, sigma_fit_mode="mpc_replay",
    r0_variance_mode="baseline_residual", mpc_archive=None, max_horizon=8):
    """Fit q=sigma^2 from causal historical MPC errors, with oracle bootstrap.

    Each of the five aligned historical origins contributes a forward MPC
    trajectory if already archived. Unavailable early-history origins use
    the existing oracle-centred historical residual (bootstrap only). No
    current or future observation is used to estimate the historical MPC
    initial state. As five weeks of MPC records accumulate, all bootstrap
    rows naturally disappear from the rolling calibration.
    """
    if sigma_fit_mode not in SIGMA_FIT_MODES:
        raise ValueError(f"sigma_fit_mode must be one of {SIGMA_FIT_MODES}.")
    initial = _mpc_initial_covariance(fit, datasets, r0_variance_mode)
    P0 = initial["covariance"]
    hourly_index = np.arange(DAY_HOURS, dtype=int) * int(steps_per_hour)
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
    if sigma_fit_mode == "mpc_replay":
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
        initial_variance = propagated_initial_variance if sigma_fit_mode == "historical_p0" else np.zeros(DAY_HOURS)
        fit_mask = np.arange(DAY_HOURS, dtype=int) > 0
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
    }


def _stochastic_forecast_variance(model, state, fit, diffusion, max_horizon, steps_per_hour):
    lead_steps = np.arange(1, int(max_horizon) + 1, dtype=int) * int(steps_per_hour)
    sensitivity = model.sensitivity({"r": fit["r"]})["input"][lead_steps]
    P0 = np.asarray(diffusion["mpc_initial_covariance"], dtype=float)
    initial_F_component = P0[0, 0] * sensitivity[:, 0] ** 2
    initial_R_component = P0[1, 1] * sensitivity[:, 1] ** 2
    initial_variance = np.maximum(initial_F_component + initial_R_component, 0.0)
    kernel = model.process_variance_kernel(state, lead_steps)
    process_kernel_unit = kernel @ np.ones(model.t_steps, dtype=float)
    process_variance = process_kernel_unit * float(diffusion["q"])
    total_variance = np.maximum(initial_variance + process_variance, 0.0)
    return {
        "F_variance": float(P0[0, 0]), "R_variance": float(P0[1, 1]),
        "initial_F_component": initial_F_component, "initial_R_component": initial_R_component,
        "initial_variance": initial_variance,
        "process_variance": process_variance,
        "process_kernel_unit": process_kernel_unit,
        "variance": total_variance,
        "std": np.sqrt(total_variance),
    }


def _mpc_origin(frame, origin, baseline_weeks, steps_per_hour, output_model, max_horizon, *, sigma_fit_mode="mpc_replay", r0_variance_mode="baseline_residual", mpc_archive=None, **fit_kwargs):
    origin = pd.Timestamp(origin)
    fit, fit_model, datasets = _fit_shifted_history(
        frame, origin, baseline_weeks, steps_per_hour, output_model, **fit_kwargs,
    )

    data, fine_index = _window_input(frame, origin, steps_per_hour, output_model)
    model = wm_4((0.0, 1.0), len(fine_index), model_output(data, fine_index))
    helper = optimized_qp_params(model)
    prepared = helper._prepare_data(data)

    historical = np.asarray(fit["x_hat"], dtype=float)
    R_observed = float(prepared["I"][0] - prepared["O"][0])
    init = np.array([historical[0], R_observed], dtype=float)
    state = helper._state_from_parameters(prepared, init, fit["r"], full=True)
    diffusion = _fit_stochastic_variance(fit_model, fit, datasets, steps_per_hour,
        sigma_fit_mode=sigma_fit_mode, r0_variance_mode=r0_variance_mode,
        mpc_archive=mpc_archive, max_horizon=max_horizon)
    stochastic = _stochastic_forecast_variance(
        model, state, fit, diffusion, max_horizon, steps_per_hour,
    )

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
            "R_observed_origin": R_observed,
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
            "calibration_mpc_count": int(diffusion["calibration_mpc_count"]),
            "calibration_oracle_count": int(diffusion["calibration_oracle_count"]),
            "calibration_horizon": int(diffusion["calibration_horizon"]),
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
        "R_historical": float(historical[1]),
        "R_observed": R_observed,
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
        "r0_variance_mode": r0_variance_mode,
        "mpc_P0_FF": float(stochastic["F_variance"]),
        "mpc_P0_FR": 0.0,
        "mpc_P0_RR": float(stochastic["R_variance"]),
        "baseline_origin_residual_variance": float(diffusion["baseline_residual_variance"]),
        "baseline_origin_residual_mean": float(diffusion["baseline_residual_mean"]),
        "baseline_origin_residual_samples": int(diffusion["baseline_residual_samples"]),
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


def run_all_days(frame, start=START_DAY, end=END_DAY, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, max_horizon=8, fail_fast=False, mpc_archive=None, **fit_kwargs):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")
    forecast_rows, state_rows, skipped = [], [], []
    mpc_archive = {} if mpc_archive is None else mpc_archive

    for day in pd.date_range(start, end, freq="D"):
        output_model, _ = fit_expected_output(frame, day)
        fitted = 0
        for hour in range(DAY_HOURS):
            origin = day + pd.Timedelta(hours=hour)
            try:
                rows, state = _mpc_origin(
                    frame, origin, baseline_weeks, steps_per_hour,
                    output_model, max_horizon, mpc_archive=mpc_archive, **fit_kwargs,
                )
                _archive_mpc_origin(mpc_archive, origin, rows, max_horizon)
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
    parser = ArgumentParser(description="Evaluate WM4 receding-horizon MPC forecasting. At every hourly update, refit a shifted 24-hour WM4 model on corresponding historical weekdays, initialize F from the historical fitted state and R from the current observed residual I-O, then compare the next 1-8 forecast hours with baseline O(t).")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--r0-variance-mode", choices=R0_VARIANCE_MODES, default="baseline_residual",
        help="Initialize R0 variance from historical I-O baseline residual spread (default), or zero for an ablation.")
    parser.add_argument("--sigma-fit-mode", choices=SIGMA_FIT_MODES, default="mpc_replay",
        help="Fit sigma from saved causal MPC forecasts with oracle bootstrap (default); historical_p0 and zero_p0 retain legacy fits.")
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    if args.max_horizon < 1 or args.max_horizon > 8: raise ValueError("max-horizon must lie in [1, 8].")

    frame, _ = aligned_wm4_data("MED")
    forecasts, states, skipped = run_all_days(
        frame,
        args.start_day,
        args.end_day,
        args.baseline_weeks,
        args.steps_per_hour,
        max_horizon=args.max_horizon,
        fail_fast=args.fail_fast,
        sigma_fit_mode=args.sigma_fit_mode,
        r0_variance_mode=args.r0_variance_mode,
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
    print(f"sigma fit mode={args.sigma_fit_mode}; initial R0 variance mode={args.r0_variance_mode} " 
        "(baseline_residual uses historical I-O spread as a proxy for O estimation error)")
    print(f"output={output}")


if __name__ == "__main__": main()
