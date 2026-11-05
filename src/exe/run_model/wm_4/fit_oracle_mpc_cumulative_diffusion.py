from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import DAY_HOURS, day_start
from src.exe.run_model.wm_4.fit_oracle_mpc_diffusion import OUTPUT as ORACLE_OUTPUT, Z_95


INPUT = ORACLE_OUTPUT / "oracle_24h_by_origin_and_lead.csv"
OUTPUT = ORACLE_OUTPUT / "cumulative_24h"
Z_68 = 1.0
NOMINAL_68_PCT = 68.26894921370859  # Two-sided Gaussian +/-1 sigma
NOMINAL_95_PCT = 95.0
REQUIRED = {"origin", "lead_hours", "I_observed", "I_oracle", "oracle_error", "unit_process_variance"}


def load_predictions(path, start_day=None, end_day=None):
    """Read previously fitted oracle trajectories and fixed WM4 process kernels; fit nothing here."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing cached oracle trajectories: {path}\n"
            "First run: python -m src.exe.run_model.wm_4.fit_oracle_mpc_diffusion"
        )
    frame = pd.read_csv(path, parse_dates=["origin"])
    missing = sorted(REQUIRED.difference(frame.columns))
    if missing: raise ValueError("Oracle prediction file is missing: " + ", ".join(missing))
    if frame["origin"].isna().any(): raise ValueError("Oracle origin timestamps must be valid.")
    if start_day is not None: frame = frame.loc[frame["origin"] >= day_start(start_day)]
    if end_day is not None: frame = frame.loc[frame["origin"] < day_start(end_day) + pd.Timedelta(days=1)]
    if frame.empty: raise ValueError("No oracle trajectories for the requested period.")
    if frame.duplicated(["origin", "lead_hours"]).any(): raise ValueError("Duplicate oracle origin / lead-hour rows.")
    if not np.isfinite(frame[["I_observed", "I_oracle", "oracle_error", "unit_process_variance"]].to_numpy(dtype=float)).all():
        raise ValueError("Observed/oracle values, residuals and WM4 unit-variance kernels must be finite.")
    if (frame["unit_process_variance"] < -1e-12).any(): raise ValueError("WM4 unit-process variance cannot be negative.")
    if not np.allclose(frame["oracle_error"], frame["I_observed"] - frame["I_oracle"], atol=1e-8, rtol=1e-6):
        raise ValueError("Saved oracle_error does not equal I_observed - I_oracle.")
    frame = frame.copy()
    frame["unit_process_variance"] = frame["unit_process_variance"].clip(lower=0.0)
    return frame.sort_values(["origin", "lead_hours"]).reset_index(drop=True)


def fit_nonnegative_variances(target, process_cumulative):
    """Exact nonnegative two-parameter least-squares solution for q_0, q_p."""
    y, b = np.asarray(target, dtype=float), np.asarray(process_cumulative, dtype=float)
    h = np.arange(1, y.size + 1, dtype=float)
    if y.size != DAY_HOURS or b.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(b).all():
        raise ValueError("Expected 24 finite cumulative error and kernel samples.")
    if np.any(b < 0.0) or np.all(b == 0.0): raise ValueError("WM4 process kernel is zero or invalid.")
    A = np.column_stack((h, b))
    if np.linalg.svd(A / np.linalg.norm(A, axis=0), compute_uv=False)[-1] < 1e-10:
        raise ValueError("Observation and process variances cannot be separately identified on this kernel.")
    candidates = [np.zeros(2)]
    for j in (0, 1):
        v = np.zeros(2)
        v[j] = max(0.0, float(A[:, j] @ y) / float(A[:, j] @ A[:, j]))
        candidates.append(v)
    interior = np.linalg.lstsq(A, y, rcond=None)[0]
    if np.all(interior >= -1e-12): candidates.append(np.maximum(interior, 0.0))
    q = min(candidates, key=lambda v: float(np.sum((A @ v - y) ** 2)))
    return float(q[0]), float(q[1])


def fit_cumulative_origin(group, q_0=None, q_p=None):
    """Fit q_0=sigma_0**2 and q_p=sigma_p**2 for each oracle 24-hour trajectory.

    Observation errors epsilon_h iid N(0,1) enter I_h=O_h+R_h+sigma_0*epsilon_h.
    Independent Wiener noise enters dR=drift*dt+sigma_p*dW.
    With known oracle x0, Var(I_h)=q_0+q_p*K_h and expected cumulative
    squared error at horizon H is H*q_0+q_p*sum(K[:H]).
    Match realized cumulative squared errors with nonnegative two-variable QP.
    This is an in-sample, one-path moment proxy, not empirical ensemble variance.
    """
    group = group.sort_values("lead_hours").copy()
    hours = group["lead_hours"].to_numpy(dtype=int)
    if len(hours) != DAY_HOURS or not np.array_equal(hours, np.arange(1, DAY_HOURS + 1)):
        raise ValueError(f"Origin {group['origin'].iloc[0]} must have exactly the 24 leads 1..24.")

    error = group["oracle_error"].to_numpy(dtype=float)
    K = group["unit_process_variance"].to_numpy(dtype=float)
    if not np.isfinite(error).all() or not np.isfinite(K).all() or np.any(K < 0.0):
        raise ValueError("Oracle residuals / WM4 variance kernels must be finite and nonnegative.")
    observed_cumulative = np.cumsum(error ** 2)
    unit_cumulative = np.cumsum(K)
    denominator = float(unit_cumulative @ unit_cumulative)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("The cumulative WM4 process-variance kernel is zero or invalid.")

    if (q_0 is None) != (q_p is None):
        raise ValueError("q_0 and q_p must both be supplied or both estimated.")
    if q_0 is None:
        q_0, q_p = fit_nonnegative_variances(observed_cumulative, unit_cumulative)
        fit_mode = "in_sample_oracle"
    else:
        q_0, q_p = float(q_0), float(q_p)
        if not np.isfinite(q_0) or not np.isfinite(q_p) or min(q_0, q_p) < 0.0:
            raise ValueError("Historical q_0 and q_p must be finite and nonnegative.")
        fit_mode = "historical"
    sigma_0, sigma_p = float(np.sqrt(q_0)), float(np.sqrt(q_p))
    observation_variance = np.full(DAY_HOURS, q_0)
    process_variance = q_p * K
    total_variance = observation_variance + process_variance
    process_std = np.sqrt(np.maximum(process_variance, 0.0))
    total_std = np.sqrt(np.maximum(total_variance, 0.0))
    fitted_cumulative_process = q_p * unit_cumulative
    fitted_cumulative_observation = q_0 * hours
    fitted_cumulative = fitted_cumulative_observation + fitted_cumulative_process
    group["error_squared"] = error ** 2
    group["cumulative_error_squared"] = observed_cumulative
    group["cumulative_unit_variance"] = unit_cumulative
    group["fitted_cumulative_process_variance"] = fitted_cumulative_process
    group["fitted_cumulative_observation_variance"] = fitted_cumulative_observation
    group["fitted_cumulative_total_variance"] = fitted_cumulative
    group["cumulative_variance_residual"] = fitted_cumulative - observed_cumulative
    group["cumulative_error_mse"] = observed_cumulative / hours
    group["cumulative_error_rmse"] = np.sqrt(group["cumulative_error_mse"])
    group["cumulative_model_mean_variance"] = fitted_cumulative / hours
    group["cumulative_model_rms_std"] = np.sqrt(group["cumulative_model_mean_variance"])
    group["observation_variance"] = observation_variance
    group["process_variance"] = process_variance
    group["total_variance"] = total_variance
    group["process_std"] = process_std
    group["total_std"] = total_std
    group["lower_68"] = group["I_oracle"] - Z_68 * total_std
    group["upper_68"] = group["I_oracle"] + Z_68 * total_std
    group["covered_68"] = np.abs(error) <= Z_68 * total_std
    group["lower_95"] = group["I_oracle"] - Z_95 * total_std
    group["upper_95"] = group["I_oracle"] + Z_95 * total_std
    group["covered_95"] = np.abs(error) <= Z_95 * total_std
    group["fit_mode"] = fit_mode
    group["q_0"], group["q_p"] = q_0, q_p
    group["sigma_0"], group["sigma_p"] = sigma_0, sigma_p
    group["q"], group["sigma"] = q_p, sigma_p  # Backward-compatible process-only aliases

    fit = {
        "origin": group["origin"].iloc[0],
        "origin_day": group["origin"].iloc[0].normalize(),
        "fit_mode": fit_mode,
        "q_0": q_0, "q_p": q_p, "sigma_0": sigma_0, "sigma_p": sigma_p,
        "q": q_p, "sigma": sigma_p,
        "horizon_hours": DAY_HOURS,
        "cumulative_curve_mse": float(np.mean((fitted_cumulative - observed_cumulative) ** 2)),
        "observed_24h_error_squared_sum": float(observed_cumulative[-1]),
        "fitted_24h_variance_sum": float(fitted_cumulative[-1]),
        "observed_24h_rmse": float(np.sqrt(observed_cumulative[-1] / DAY_HOURS)),
        "model_24h_rms_std": float(np.sqrt(fitted_cumulative[-1] / DAY_HOURS)),
        "mean_oracle_residual": float(np.mean(error)),
        "total_68_coverage_pct": 100.0 * float(np.mean(group["covered_68"])),
        "total_95_coverage_pct": 100.0 * float(np.mean(group["covered_95"])),
        "process_68_coverage_pct": 100.0 * float(np.mean(group["covered_68"])),  # Legacy name: total coverage
        "process_95_coverage_pct": 100.0 * float(np.mean(group["covered_95"])),  # Legacy name: total coverage
    }
    for key in ("r_F", "r_U", "r_R", "F0_oracle", "R0_oracle"):
        if key in group.columns: fit[key] = float(group[key].iloc[0])
    return fit, group


def fit_all_origins(forecasts):
    """Fit an independent diffusion for each hourly MPC origin without refitting model parameters."""
    fits, trajectories = [], []
    for _, group in forecasts.groupby("origin", sort=True):
        fit, result = fit_cumulative_origin(group)
        fits.append(fit); trajectories.append(result)
    if not fits: raise ValueError("No complete oracle MPC origins were provided.")
    origins = pd.DataFrame(fits)
    hourly = pd.concat(trajectories, ignore_index=True)
    return origins, hourly


def fit_historical_variances(groups):
    """Fit q0 and qp jointly on COMPLETE, already-observed historical oracle horizons.

    For each historical origin j and horizon H, minimize
      [sum_{h<=H} error[j,h]**2 - H*q0 - q_p*sum_{h<=H}K[j,h]]**2.
    This is a two-variable convex QP with nonnegative constraints.
    """
    if not groups: raise ValueError("Historical diffusion estimation requires prior completed horizons.")
    design, targets = [], []
    for group in groups:
        group = group.sort_values("lead_hours")
        lead = group["lead_hours"].to_numpy(dtype=int)
        if len(lead) != DAY_HOURS or not np.array_equal(lead, np.arange(1, DAY_HOURS + 1)):
            raise ValueError("Historical origins must contain every hour 1..24 exactly once.")
        residual = group["oracle_error"].to_numpy(dtype=float)
        kernel = group["unit_process_variance"].to_numpy(dtype=float)
        if not np.isfinite(residual).all() or not np.isfinite(kernel).all() or np.any(kernel < 0.0):
            raise ValueError("Historical residuals and kernels must be valid.")
        design.append(np.column_stack((lead.astype(float), np.cumsum(kernel))))
        targets.append(np.cumsum(residual ** 2))
    A, y = np.concatenate(design), np.concatenate(targets)
    norm = np.linalg.norm(A, axis=0)
    if np.any(norm <= 0.0) or np.linalg.svd(A / norm, compute_uv=False)[-1] < 1e-10:
        raise ValueError("Historical observation and process variances are not separately identifiable.")

    feasible = [np.zeros(2, dtype=float)]
    for i in (0, 1):
        q = np.zeros(2, dtype=float)
        q[i] = max(0.0, float(A[:, i] @ y) / float(A[:, i] @ A[:, i]))
        feasible.append(q)
    interior = np.linalg.lstsq(A, y, rcond=None)[0]
    if np.all(interior >= -1e-12): feasible.append(np.maximum(interior, 0.0))
    fitted = min(feasible, key=lambda q: float(np.sum((A @ q - y) ** 2)))
    return float(fitted[0]), float(fitted[1])


def fit_all_origins_historical(forecasts, lookback_days=24, min_training_days=5,
    start_day=None, end_day=None):
    """Use only prior fully observed same-clock-hour oracle horizons to fit sigma.

    Training origin j must finish strictly before the calendar prediction day,
    so no target-day observations or residuals affect q0/qp. The target's already fitted oracle x0 is
    retained as an explicit retrospective exception.
    """
    lookback_days, min_training_days = int(lookback_days), int(min_training_days)
    if lookback_days < 1 or min_training_days < 2:
        raise ValueError("lookback_days >= 1 and min_training_days >= 2 are required.")
    if min_training_days > lookback_days:
        raise ValueError("min_training_days cannot exceed lookback_days when matching calendar days.")
    all_groups = {}
    for stamp, group in forecasts.groupby("origin", sort=True):
        stamp = pd.Timestamp(stamp)
        lead = group["lead_hours"].to_numpy(dtype=int)
        if not np.array_equal(np.sort(lead), np.arange(1, DAY_HOURS + 1)):
            raise ValueError(f"Incomplete 24-hour oracle trajectory at {stamp}.")
        all_groups[stamp] = group.sort_values("lead_hours")
    if not all_groups: raise ValueError("No oracle trajectories were supplied.")
    by_clock = {}
    for stamp in all_groups:
        by_clock.setdefault((stamp.hour, stamp.minute, stamp.second), []).append(stamp)
    first = day_start(start_day) if start_day is not None else None
    last = day_start(end_day) + pd.Timedelta(days=1) if end_day is not None else None
    if first is not None and last is not None and first >= last:
        raise ValueError("end_day must be on or after start_day.")

    origin_fits, forecast_rows = [], []
    missing_history = 0
    for target, group in all_groups.items():
        # Include the previous day's 23:00 origin for a calendar-day
        # 00:00..23:00 validation trajectory when --start-day is set.
        if first is not None and target < first - pd.Timedelta(hours=1): continue
        if last is not None and target >= last: continue
        prediction_day = (target + pd.Timedelta(hours=1)).normalize()
        lower = target - pd.Timedelta(days=lookback_days)
        # Strictly before the prediction calendar day, not merely before this
        # hour within that day. This also excludes a previous-day 24h path
        # that contains observations from today's forecast window.
        same_hour = by_clock.get((target.hour, target.minute, target.second), ())
        eligible = [origin for origin in same_hour if
            lower <= origin <= target - pd.Timedelta(hours=DAY_HOURS) and
            origin + pd.Timedelta(hours=DAY_HOURS) < prediction_day]
        if len(eligible) < min_training_days:
            missing_history += 1
            continue
        history = [all_groups[t] for t in eligible]
        q0, qp = fit_historical_variances(history)
        fit, evaluated = fit_cumulative_origin(group, q_0=q0, q_p=qp)
        # The training window's last observed instant is <= target origin.
        last_training_observation = max(eligible) + pd.Timedelta(hours=DAY_HOURS)
        if last_training_observation >= prediction_day:
            raise AssertionError("Historical diffusion calibration used observations from the prediction day.")
        if last_training_observation > target:
            raise AssertionError("Historical diffusion calibration used future data.")
        fit["prediction_day"] = prediction_day
        fit["training_origins"] = len(eligible)
        fit["training_first_origin"] = min(eligible)
        fit["training_last_origin"] = max(eligible)
        fit["training_last_observation"] = last_training_observation
        fit["lookback_days"] = lookback_days
        fit["min_training_days"] = min_training_days
        evaluated = evaluated.copy()
        evaluated["prediction_day"] = prediction_day
        evaluated["training_origins"] = len(eligible)
        evaluated["training_last_observation"] = last_training_observation
        origin_fits.append(fit); forecast_rows.append(evaluated)

    if not origin_fits:
        raise ValueError(f"No validation origins with at least {min_training_days} completed historical days were found.")
    origins = pd.DataFrame(origin_fits).sort_values("origin").reset_index(drop=True)
    hourly = pd.concat(forecast_rows, ignore_index=True)
    return origins, hourly, missing_history


def horizon_summary(hourly):
    """Aggregate empirical residual variance and nominal interval coverage at each 1..24h lead.

    Compare empirical residual variance across origins to the sum of mean
    observation and Wiener process variance. Intervals use total variance.
    Each origin fits both sigmas on these same errors (in-sample diagnostic).
    """
    rows = []
    for hour, group in hourly.groupby("lead_hours", sort=True):
        residual = group["oracle_error"].to_numpy(dtype=float)
        n = residual.size
        observed_cum = group["cumulative_error_squared"].to_numpy(dtype=float)
        model_cum = group["fitted_cumulative_total_variance"].to_numpy(dtype=float)
        covered_68 = group["covered_68"].to_numpy(dtype=bool)
        covered_95 = group["covered_95"].to_numpy(dtype=bool)
        empirical_var = float(np.var(residual, ddof=1)) if n > 1 else np.nan
        observation_var = float(np.mean(group["observation_variance"]))
        process_var = float(np.mean(group["process_variance"]))
        predicted_var = float(np.mean(group["total_variance"]))
        coverage_68 = 100.0 * float(np.mean(covered_68))
        coverage_95 = 100.0 * float(np.mean(covered_95))
        rows.append({
            "lead_hours": int(hour), "origins": int(n),
            "fit_mode": str(group["fit_mode"].iloc[0]) if "fit_mode" in group else "unknown",
            "hourly_oracle_bias": float(np.mean(residual)),
            "hourly_oracle_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "hourly_oracle_mse": float(np.mean(residual ** 2)),
            "empirical_oracle_variance": empirical_var,
            "empirical_oracle_std": float(np.sqrt(empirical_var)),
            "mean_predicted_observation_variance": observation_var,
            "mean_predicted_process_variance": process_var,
            "mean_predicted_total_variance": predicted_var,
            "mean_predicted_process_std": float(np.sqrt(process_var)),
            "mean_predicted_total_std": float(np.sqrt(predicted_var)),
            "predicted_to_empirical_variance_ratio": (
                predicted_var / empirical_var if empirical_var > 0.0 else np.nan
            ),
            "mean_cumulative_squared_error": float(np.mean(observed_cum)),
            "mean_cumulative_model_variance": float(np.mean(model_cum)),
            "cumulative_error_rmse": float(np.sqrt(np.mean(observed_cum) / hour)),
            "cumulative_model_rms_std": float(np.sqrt(np.mean(model_cum) / hour)),
            "mean_cumulative_difference": float(np.mean(model_cum - observed_cum)),
            "covered_68_origins": int(np.count_nonzero(covered_68)),
            "covered_95_origins": int(np.count_nonzero(covered_95)),
            "nominal_68_coverage_pct": NOMINAL_68_PCT,
            "nominal_95_coverage_pct": NOMINAL_95_PCT,
            "total_68_coverage_pct": coverage_68,
            "total_95_coverage_pct": coverage_95,
            "process_68_coverage_pct": coverage_68,
            "process_95_coverage_pct": coverage_95,
            "coverage_68_error_pp": coverage_68 - NOMINAL_68_PCT,
            "coverage_95_error_pp": coverage_95 - NOMINAL_95_PCT,
            "coverage_68_ratio": coverage_68 / NOMINAL_68_PCT,
            "coverage_95_ratio": coverage_95 / NOMINAL_95_PCT,
        })
    return pd.DataFrame(rows).sort_values("lead_hours").reset_index(drop=True)


def save_calibration_plot(summary, output):
    """Per-lead variance, 68%/95% interval coverage and 1:1 coverage ratios."""
    h = summary["lead_hours"].to_numpy(dtype=int)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(h, summary["empirical_oracle_variance"], marker="o",
        label="Empirical oracle-residual variance across MPC origins")
    axes[0].plot(h, summary["mean_predicted_total_variance"], marker="o",
        label="Mean ARM fitted total variance")
    axes[0].plot(h, summary["mean_predicted_observation_variance"], linestyle=":",
        label="Observation variance sigma_0^2")
    axes[0].plot(h, summary["mean_predicted_process_variance"], linestyle="--",
        label="Propagated process variance sigma_p^2 K_h")
    axes[0].set_ylabel("Marginal variance")
    axes[0].legend(ncol=2); axes[0].grid(True, alpha=0.25)

    axes[1].plot(h, summary["total_68_coverage_pct"], marker="o", label="Observed +/-1 total sigma coverage")
    axes[1].plot(h, summary["total_95_coverage_pct"], marker="o", label="Observed 95% total sigma coverage")
    axes[1].axhline(NOMINAL_68_PCT, linestyle="--", label="Gaussian 68.27% nominal")
    axes[1].axhline(NOMINAL_95_PCT, linestyle="--", label="95% nominal")
    axes[1].set_ylabel("Empirical coverage (%)")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].legend(ncol=2); axes[1].grid(True, alpha=0.25)

    axes[2].plot(h, summary["coverage_68_ratio"], marker="o", label="68.27% empirical / nominal")
    axes[2].plot(h, summary["coverage_95_ratio"], marker="o", label="95% empirical / nominal")
    axes[2].axhline(1.0, linestyle="--", label="Perfect calibration (1:1)")
    axes[2].set_ylabel("Coverage calibration ratio")
    axes[2].set_xlabel("Hours since hourly MPC origin")
    axes[2].legend(); axes[2].grid(True, alpha=0.25)
    axes[2].set_xticks(np.arange(1, DAY_HOURS + 1, 2))
    axes[2].set_xlim(1, DAY_HOURS)

    mode = ("historical diffusion / oracle state" if
        "fit_mode" in summary and summary["fit_mode"].eq("historical").all() else "in-sample")
    fig.suptitle(f"ARM (Affine Revised Model) observation + Wiener process variance and 68% / 95% coverage ({mode})")
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def save_plot(origins, summary, output):
    fig, axes = plt.subplots(3, 1, figsize=(15, 13))
    axes[0].scatter(origins["origin"], origins["sigma_0"], s=7, alpha=0.75, label="Observation sigma_0")
    axes[0].scatter(origins["origin"], origins["sigma_p"], s=7, alpha=0.75, label="Process sigma_p")
    axes[0].axhline(float(np.mean(origins["sigma_0"])), linestyle=":", label="Mean sigma_0")
    axes[0].axhline(float(np.mean(origins["sigma_p"])), linestyle="--", label="Mean sigma_p")
    axes[0].set_ylabel("Uncertainty sigma")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)

    h = summary["lead_hours"].to_numpy(dtype=int)
    axes[1].plot(h, summary["mean_cumulative_squared_error"], marker="o", label="Observed cumulative squared oracle error")
    axes[1].plot(h, summary["mean_cumulative_model_variance"], marker="o", label="Fitted cumulative observation + process variance")
    axes[1].set_ylabel("Cumulative squared error / variance")
    axes[1].legend(); axes[1].grid(True, alpha=0.25)

    axes[2].plot(h, summary["cumulative_error_rmse"], marker="o", label="Cumulative oracle RMSE")
    axes[2].plot(h, summary["cumulative_model_rms_std"], marker="o", label="Cumulative model total RMS std")
    axes[2].plot(h, summary["hourly_oracle_rmse"], linestyle=":", label="Exact-hour oracle RMSE")
    axes[2].set_xlabel("Hours since MPC origin")
    axes[2].set_ylabel("Normalized input")
    axes[2].legend(); axes[2].grid(True, alpha=0.25)
    for ax in axes[1:]: ax.set_xticks(np.arange(1, DAY_HOURS + 1, 2))

    mode = ("historical" if "fit_mode" in origins and
        origins["fit_mode"].eq("historical").all() else "in-sample")
    fig.suptitle(f"ARM (Affine Revised Model) observation + process uncertainty: 24h cumulative oracle errors ({mode})")
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)



QUINTILE_NAMES = ("Lowest", "Low", "Neutral", "High", "Highest")


def day_error_quintiles(hourly, mode="calendar-day", start_day=None, end_day=None):
    """Rank days by MODEL-predicted 24h RMS std, never by observed RMSE.

    calendar-day: select the 23:00 origin on the PREVIOUS day; its leads
    1..24 target exactly 00:00..23:00 of the next calendar day.
    midnight: 00:00-origin forecast, leads 1..24 (01:00..next 00:00).
    all-origins: pool forecasts by their origin day, not target day.
    With historical fitting, sigma_0 and sigma_p are estimated from completed
    preceding days only. The target day's oracle state nevertheless uses future
    observations; hence this remains an oracle-conditioned retrospective test.
    """
    if mode not in ("calendar-day", "midnight", "all-origins"):
        raise ValueError("day quintile mode must be calendar-day, midnight or all-origins.")
    required = {"origin", "lead_hours", "oracle_error", "total_variance"}
    missing = sorted(required.difference(hourly.columns))
    if missing: raise ValueError(f"Hourly oracle predictions lack: {', '.join(missing)}")
    df = hourly.copy()
    df["origin"] = pd.to_datetime(df["origin"])
    if df[["oracle_error", "total_variance"]].isna().any().any():
        raise ValueError("Oracle residual and fitted variance must not contain missing values.")
    if (df["total_variance"] < 0).any():
        raise ValueError("Fitted marginal variance must be nonnegative.")
    if mode == "calendar-day":
        df = df.loc[df["origin"].dt.hour.eq(23) & df["origin"].dt.minute.eq(0)]
    elif mode == "midnight":
        df = df.loc[df["origin"].dt.hour.eq(0) & df["origin"].dt.minute.eq(0)]
    if df.empty: raise ValueError(f"No MPC origins available for day quintile mode '{mode}'.")

    # Ensure all included trajectories are complete, exactly 24 hourly leads.
    expected = list(range(1, DAY_HOURS + 1))
    for origin, group in df.groupby("origin", sort=False):
        if sorted(group["lead_hours"].astype(int).tolist()) != expected:
            raise ValueError(f"MPC origin {origin} lacks a complete 24-hour trajectory.")

    df["day"] = (df["origin"] + pd.Timedelta(hours=1)).dt.normalize() if mode == "calendar-day" else df["origin"].dt.normalize()
    if start_day is not None: df = df.loc[df["day"] >= day_start(start_day)]
    if end_day is not None: df = df.loc[df["day"] <= day_start(end_day)]
    if df.empty: raise ValueError("No complete prediction calendar days remain in the requested date range.")
    daily = df.groupby("day", sort=True).agg(
        origins=("origin", "nunique"),
        evaluated_hours=("lead_hours", "size"),
        observed_mse=("oracle_error", lambda e: float(np.mean(e.to_numpy(dtype=float) ** 2))),
        predicted_mean_variance=("total_variance", "mean"),
    ).reset_index()
    if "fit_mode" in df.columns:
        modes = df.groupby("day")["fit_mode"].first()
        daily["fit_mode"] = daily["day"].map(modes)
    daily["predicted_24h_rms_std"] = np.sqrt(daily["predicted_mean_variance"])
    daily["actual_24h_rmse"] = np.sqrt(daily["observed_mse"])
    if len(daily) < 5:
        raise ValueError("At least five distinct calendar days with complete 24-hour MPC forecasts are required.")

    # Rank then partition by count: five well-defined quintiles even with ties.
    daily = daily.sort_values(["predicted_24h_rms_std", "day"], kind="mergesort").reset_index(drop=True)
    daily["quintile"] = (np.arange(len(daily), dtype=int) * 5 // len(daily)) + 1
    daily["quintile_name"] = daily["quintile"].map(dict(enumerate(QUINTILE_NAMES, start=1)))
    daily["predicted_to_actual_ratio"] = np.divide(
        daily["predicted_24h_rms_std"], daily["actual_24h_rmse"],
        out=np.full(len(daily), np.nan),
        where=daily["actual_24h_rmse"].to_numpy(dtype=float) > 0.0,
    )

    rows = []
    for idx, group in daily.groupby("quintile", sort=True):
        predicted = group["predicted_24h_rms_std"].to_numpy(dtype=float)
        actual = group["actual_24h_rmse"].to_numpy(dtype=float)
        predicted_rms = float(np.sqrt(np.mean(predicted ** 2)))
        actual_rms = float(np.sqrt(np.mean(actual ** 2)))
        rows.append({
            "quintile": int(idx), "quintile_name": QUINTILE_NAMES[int(idx) - 1],
            "days": int(len(group)), "origins": int(group["origins"].sum()),
            "min_predicted_day_rms_std": float(predicted.min()),
            "max_predicted_day_rms_std": float(predicted.max()),
            "predicted_24h_rms_std": predicted_rms,
            "actual_24h_rmse": actual_rms,
            "actual_minus_predicted": actual_rms - predicted_rms,
            "actual_to_predicted_ratio": actual_rms / predicted_rms if predicted_rms > 0.0 else np.nan,
        })
    return daily, pd.DataFrame(rows)


def save_day_error_quintiles(daily, quintiles, output, mode="calendar-day"):
    """Compare mean expected vs realized 24h error by *predicted* difficulty quintile."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    x = np.arange(5, dtype=float)
    width = 0.38
    label = [f"{name}\n(n={n})" for name, n in zip(quintiles["quintile_name"], quintiles["days"])]
    axes[0].bar(x - width / 2, quintiles["predicted_24h_rms_std"], width,
        label="Predicted 24h RMS std (total variance)")
    axes[0].bar(x + width / 2, quintiles["actual_24h_rmse"], width,
        label="Observed 24h oracle RMSE")
    axes[0].set_xticks(x, label)
    axes[0].set_ylabel("Normalized input")
    axes[0].legend(); axes[0].grid(True, axis="y", alpha=0.25)

    for i, group in daily.groupby("quintile", sort=True):
        axes[1].scatter(group["predicted_24h_rms_std"], group["actual_24h_rmse"],
            s=22, alpha=0.65, label=QUINTILE_NAMES[int(i) - 1])
    extent = float(max(daily["predicted_24h_rms_std"].max(), daily["actual_24h_rmse"].max()))
    axes[1].plot([0.0, extent], [0.0, extent], linestyle="--", label="Perfect 1:1 error calibration")
    axes[1].set_xlim(left=0.0); axes[1].set_ylim(bottom=0.0)
    axes[1].set_xlabel("Predicted daily 24h RMS std")
    axes[1].set_ylabel("Actual daily 24h oracle RMSE")
    axes[1].legend(ncol=2); axes[1].grid(True, alpha=0.25)

    context = ("calendar days (previous 23:00-origin forecast)" if mode == "calendar-day" else
        "midnight-origin 24h forecasts" if mode == "midnight" else "all MPC origins pooled by origin day")
    protocol = ("historical diffusion, oracle state" if
        "fit_mode" in daily.columns and daily["fit_mode"].eq("historical").all()
        else "in-sample diffusion and oracle state")
    fig.suptitle(f"ARM (Affine Revised Model) predicted-error quintiles vs observed error ({context}; {protocol})")
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    parser = ArgumentParser(description="Fit observation sigma_0 and process sigma_p at each MPC origin to cumulative 24h squared oracle errors, using saved deterministic parameters and oracle states.")
    parser.add_argument("--input", default=str(INPUT), help="Saved oracle_24h_by_origin_and_lead.csv.")
    parser.add_argument("--start-day", default=None)
    parser.add_argument("--end-day", default=None)
    parser.add_argument("--fit-mode", choices=("historical", "in-sample"), default="historical",
        help="Default: fit observation and process variances using ONLY earlier completed days.")
    parser.add_argument("--lookback-days", type=int, default=24,
        help="Number of previous calendar days eligible for diffusion fitting (default 24).")
    parser.add_argument("--min-training-days", type=int, default=5,
        help="Minimum preceding completed same-hour 24h horizons to calibrate diffusion.")
    parser.add_argument("--day-quintile-mode", choices=("calendar-day", "midnight", "all-origins"),
        default="calendar-day", help="Default: exact 00-23 calendar-day targets via previous 23:00 MPC origin; or midnight-origin forecasts / all origins pooled by start day.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if args.start_day and args.end_day and day_start(args.end_day) < day_start(args.start_day):
        parser.error("--end-day must be on/after --start-day.")

    if args.fit_mode == "historical":
        # Retain all prior data for calibration even if --start-day filters evaluation.
        forecasts = load_predictions(args.input, end_day=args.end_day)
        origins, hourly, skipped = fit_all_origins_historical(forecasts,
            lookback_days=args.lookback_days, min_training_days=args.min_training_days,
            start_day=args.start_day, end_day=args.end_day)
        print(f"Historical calibration: {len(origins)} evaluated origins; "
            f"{skipped} skipped for insufficient completed preceding horizons.")
    else:
        forecasts = load_predictions(args.input, args.start_day, args.end_day)
        origins, hourly = fit_all_origins(forecasts)
        print("WARNING: in-sample mode fits sigma_0/sigma_p to the evaluation origin's own errors.")
    horizons = horizon_summary(hourly)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)

    origins.to_csv(output / "cumulative_diffusion_by_mpc_origin.csv", index=False)
    hourly.to_csv(output / "cumulative_diffusion_by_origin_and_lead.csv", index=False)
    horizons.to_csv(output / "cumulative_diffusion_by_horizon.csv", index=False)
    horizons.to_csv(output / "variance_coverage_by_horizon.csv", index=False)
    save_plot(origins, horizons, output / "cumulative_24h_diffusion.png")
    save_calibration_plot(horizons, output / "variance_coverage_by_horizon.png")
    try:
        daily, quintiles = day_error_quintiles(hourly, mode=args.day_quintile_mode,
            start_day=args.start_day, end_day=args.end_day)
    except ValueError as exc:
        print(f"Day error quintile plot skipped: {exc}")
    else:
        daily.to_csv(output / "day_error_quintile_assignments.csv", index=False)
        quintiles.to_csv(output / "day_error_quintile_summary.csv", index=False)
        save_day_error_quintiles(daily, quintiles, output / "day_error_quintiles.png",
            mode=args.day_quintile_mode)
        print("Daily predicted-error quintiles (sorted by predicted 24h RMS std):")
        print(quintiles[["quintile_name", "days", "predicted_24h_rms_std",
            "actual_24h_rmse", "actual_to_predicted_ratio"]].to_string(index=False))
        print(f"Saved daily quintile diagnostics to {output / 'day_error_quintiles.png'}")


    print(f"origins={len(origins)}, fitted horizon={DAY_HOURS} hours")
    print(f"observation sigma_0 mean={origins['sigma_0'].mean():.9f}, median={origins['sigma_0'].median():.9f}")
    print(f"process sigma_p mean={origins['sigma_p'].mean():.9f}, median={origins['sigma_p'].median():.9f}")
    print(f"mean final observed RMSE={origins['observed_24h_rmse'].mean():.9f}")
    print(f"mean final predicted RMS std={origins['model_24h_rms_std'].mean():.9f}")
    print("Nonnegative q_0=sigma_0^2 and q_p=sigma_p^2 jointly fitted per origin to 1..24h cumulative squared errors.")
    print("No WM4 drift coefficients, robust baseline or oracle initial states are refitted.")
    print("Cumulative squared errors are one-path proxies for expected variances, NOT empirical ensemble variances.")
    print("Empirical variance and Gaussian 68.27% / 95% interval coverage by lead:")
    print(f"{'lead':>6} {'n':>6} {'var(obs)':>12} {'var(total)':>12} {'cov68':>10} {'cov95':>10} {'ratio68':>10} {'ratio95':>10}")
    for row in horizons.itertuples(index=False):
        print(f"{row.lead_hours:6d} {row.origins:6d} {row.empirical_oracle_variance:12.7f} "
            f"{row.mean_predicted_total_variance:12.7f} {row.total_68_coverage_pct:9.2f}% "
            f"{row.total_95_coverage_pct:9.2f}% {row.coverage_68_ratio:10.3f} "
            f"{row.coverage_95_ratio:10.3f}")
    print("Coverage ratios are observed / nominal: 1.0 indicates ideal calibration.")
    print("Intervals use sqrt(sigma_0^2 + sigma_p^2*K_h).")
    if args.fit_mode == "historical":
        print("Diffusion q_0/q_p for each validation origin uses only completed prior-day data.")
        print("Only the oracle initial state on the validation day uses future target observations.")
    else:
        print("WARNING: diffusion estimates are in-sample, using the same evaluation errors.")
    print("Overlapping hourly origins are dependent; these coverages are NOT independent holdout estimates.")
    print(f"output={output}")


if __name__ == "__main__": main()
