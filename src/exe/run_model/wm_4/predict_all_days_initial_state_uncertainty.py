from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY, fit_history_optimized
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup
from src.exe.run_model.wm_4.predict_initial_state_uncertainty import _quantile_bounds
from src.params.wm_4 import optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_initial_state_uncertainty"


def day_result(frame, start, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, quantile_bounds=(0.025, 0.975), max_state_iter=50, state_tol=1e-8, oracle_comparison=False):
    start = day_start(start)
    lower_q, upper_q = _quantile_bounds(quantile_bounds)
    output_model, output_fit, fit, _ = fit_history_optimized(
        frame, start, baseline_weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )
    _, actual, U, Time, model, _ = prediction_setup(frame, start, output_model, steps_per_hour)

    x_hat = np.asarray(fit["x_hat"], dtype=float)
    x_cov = np.asarray(fit["x_cov"], dtype=float)
    x_std = np.sqrt(np.maximum(np.diag(x_cov), 0.0))
    state = model.deterministic_solver(x_hat, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})
    sensitivity = model.sensitivity(fit["r"])
    J = np.asarray(sensitivity["input"], dtype=float)

    variance = np.einsum("ni,ij,nj->n", J, x_cov, J)
    input_std = np.sqrt(np.maximum(variance, 0.0))
    lower = state["I"] + norm.ppf(lower_q) * input_std
    upper = state["I"] + norm.ppf(upper_q) * input_std
    width = upper - lower

    hourly_index = np.arange(DAY_HOURS) * int(steps_per_hour)
    observed = actual["I"].to_numpy(dtype=float)
    predicted = state["I"][hourly_index]
    baseline = state["O"][hourly_index]
    lower_h, upper_h = lower[hourly_index], upper[hourly_index]
    width_h, std_h = width[hourly_index], input_std[hourly_index]
    J_h = J[hourly_index]
    inside = (observed >= lower_h) & (observed <= upper_h)
    residual = predicted - observed
    metrics = deviation_metrics(observed, predicted)
    r = np.asarray(fit["r"], dtype=float)

    daily = {
        "day": start, "weekday": start.day_name(),
        "lower_quantile": lower_q, "upper_quantile": upper_q,
        "nominal_coverage_pct": 100.0 * (upper_q - lower_q),
        "coverage_pct": 100.0 * float(np.mean(inside)),
        "covered_hours": int(np.count_nonzero(inside)), "observations": int(len(inside)),
        "mean_interval_width": float(np.mean(width_h)), "median_interval_width": float(np.median(width_h)),
        "min_interval_width": float(np.min(width_h)), "max_interval_width": float(np.max(width_h)),
        "F0_hat": x_hat[0], "R0_hat": x_hat[1],
        "F0_std": x_std[0], "R0_std": x_std[1],
        "cov_FF": x_cov[0, 0], "cov_FR": x_cov[0, 1], "cov_RR": x_cov[1, 1],
        "x_samples": int(fit["x_samples"]),
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "qp_loss": float(fit["qp_loss"]), "state_iterations": int(fit["state_iterations"]),
        "wm4_mse": metrics["mse"], "wm4_rmse": metrics["rmse"], "wm4_mae": metrics["mae"], "wm4_bias": metrics["bias"],
        "O_fit_mse": float(output_fit["mse"]), "O_fit_samples": int(output_fit["samples"]),
    }
    hourly = pd.DataFrame({
        "day": start, "time": actual.index, "forecast_hour": np.arange(DAY_HOURS, dtype=int),
        "I_observed": observed, "I_wm4": predicted, "O": baseline,
        "lower": lower_h, "upper": upper_h, "interval_std": std_h, "interval_width": width_h,
        "inside_interval": inside, "error": residual, "abs_error": np.abs(residual), "squared_error": residual ** 2,
        "dI_dF0": J_h[:, 0], "dI_dR0": J_h[:, 1],
    })
    if oracle_comparison:
        # Oracle is fitted using the full target-day input: diagnostic only, never a deployable forecast.
        prediction_data, _, _, _ = model_input(frame, start, steps_per_hour, 0, output_model)
        oracle = optimized_qp_params(model).optimize_prediction_initial_state(prediction_data, fit["r"])
        x_star = np.asarray(oracle["init_cond"], dtype=float)
        oracle_state = model.deterministic_solver(x_star, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})
        oracle_predicted = oracle_state["I"][hourly_index]
        # The distributional spread and r, U, O remain unchanged; only its centre changes.
        oracle_lower_h = oracle_predicted + norm.ppf(lower_q) * std_h
        oracle_upper_h = oracle_predicted + norm.ppf(upper_q) * std_h
        oracle_inside = (observed >= oracle_lower_h) & (observed <= oracle_upper_h)
        oracle_metrics = deviation_metrics(observed, oracle_predicted)
        delta = x_star - x_hat
        marginal = np.abs(delta) <= norm.ppf(0.975) * x_std
        daily.update({
            "oracle_coverage_pct": 100.0 * float(np.mean(oracle_inside)),
            "oracle_covered_hours": int(np.count_nonzero(oracle_inside)),
            "coverage_lift_pct_points": 100.0 * float(np.mean(oracle_inside) - np.mean(inside)),
            "oracle_F0_star": x_star[0], "oracle_R0_star": x_star[1],
            "oracle_F0_minus_hat": delta[0], "oracle_R0_minus_hat": delta[1],
            "oracle_F0_within_prior_marginal_95": bool(marginal[0]),
            "oracle_R0_within_prior_marginal_95": bool(marginal[1]),
            "oracle_state_fit_mse": float(oracle["loss"]),
            "oracle_wm4_rmse": oracle_metrics["rmse"],
            "oracle_wm4_mae": oracle_metrics["mae"],
            "oracle_wm4_bias": oracle_metrics["bias"],
        })
        hourly["oracle_I_wm4"] = oracle_predicted
        hourly["oracle_lower"] = oracle_lower_h
        hourly["oracle_upper"] = oracle_upper_h
        hourly["oracle_inside_interval"] = oracle_inside
        hourly["oracle_error"] = oracle_predicted - observed
        hourly["oracle_squared_error"] = (oracle_predicted - observed) ** 2
    return daily, hourly


def horizon_summary(hourly):
    rows = []
    for hour, part in hourly.groupby("forecast_hour", sort=True):
        error = part["error"].to_numpy(dtype=float)
        rows.append({
            "forecast_hour": int(hour), "days": int(len(part)),
            "coverage_pct": 100.0 * float(part["inside_interval"].mean()),
            "mean_interval_width": float(part["interval_width"].mean()),
            "median_interval_width": float(part["interval_width"].median()),
            "max_interval_width": float(part["interval_width"].max()),
            "wm4_rmse": float(np.sqrt(np.mean(error ** 2))),
            "wm4_mae": float(np.mean(np.abs(error))),
            "wm4_bias": float(np.mean(error)),
            "mean_abs_dI_dF0": float(np.mean(np.abs(part["dI_dF0"]))),
            "mean_abs_dI_dR0": float(np.mean(np.abs(part["dI_dR0"]))),
        })
    summary = pd.DataFrame(rows)
    if "oracle_inside_interval" in hourly.columns:
        grouped = hourly.groupby("forecast_hour", sort=True)
        summary["oracle_coverage_pct"] = grouped["oracle_inside_interval"].mean().to_numpy() * 100.0
        summary["oracle_wm4_rmse"] = np.sqrt(grouped["oracle_squared_error"].mean().to_numpy())
        summary["oracle_coverage_lift_pct_points"] = summary["oracle_coverage_pct"] - summary["coverage_pct"]
    return summary


def weekday_summary(daily):
    rows = []
    for weekday, part in daily.groupby("weekday", sort=False):
        rows.append({
            "weekday": weekday, "days": int(len(part)),
            "mean_coverage_pct": float(part["coverage_pct"].mean()),
            "median_coverage_pct": float(part["coverage_pct"].median()),
            "mean_interval_width": float(part["mean_interval_width"].mean()),
            "mean_wm4_rmse": float(part["wm4_rmse"].mean()),
        })
    summary = pd.DataFrame(rows)
    if "oracle_coverage_pct" in daily.columns:
        oracle_by_weekday = daily.groupby("weekday", sort=False)
        summary["oracle_mean_coverage_pct"] = oracle_by_weekday["oracle_coverage_pct"].mean().to_numpy()
        summary["oracle_coverage_lift_pct_points"] = summary["oracle_mean_coverage_pct"] - summary["mean_coverage_pct"]
        summary["oracle_mean_wm4_rmse"] = oracle_by_weekday["oracle_wm4_rmse"].mean().to_numpy()
    return summary


def overall_summary(daily, hourly, skipped, quantile_bounds):
    lower_q, upper_q = _quantile_bounds(quantile_bounds)
    error = hourly["error"].to_numpy(dtype=float)
    result = [{
        "start_day": daily["day"].min(), "end_day": daily["day"].max(),
        "fitted_days": int(len(daily)), "skipped_days": int(len(skipped)),
        "hourly_observations": int(len(hourly)),
        "lower_quantile": lower_q, "upper_quantile": upper_q,
        "nominal_coverage_pct": 100.0 * (upper_q - lower_q),
        "overall_coverage_pct": 100.0 * float(hourly["inside_interval"].mean()),
        "mean_daily_coverage_pct": float(daily["coverage_pct"].mean()),
        "median_daily_coverage_pct": float(daily["coverage_pct"].median()),
        "mean_interval_width": float(hourly["interval_width"].mean()),
        "median_interval_width": float(hourly["interval_width"].median()),
        "max_interval_width": float(hourly["interval_width"].max()),
        "wm4_rmse": float(np.sqrt(np.mean(error ** 2))),
        "wm4_mae": float(np.mean(np.abs(error))),
        "wm4_bias": float(np.mean(error)),
        "mean_F0_std": float(daily["F0_std"].mean()),
        "mean_R0_std": float(daily["R0_std"].mean()),
    }]
    if "oracle_inside_interval" in hourly.columns:
        result[0].update({
            "oracle_overall_coverage_pct": 100.0 * float(hourly["oracle_inside_interval"].mean()),
            "oracle_mean_daily_coverage_pct": float(daily["oracle_coverage_pct"].mean()),
            "oracle_coverage_lift_pct_points": float(daily["coverage_lift_pct_points"].mean()),
            "oracle_wm4_rmse": float(np.sqrt(hourly["oracle_squared_error"].mean())),
            "oracle_wm4_mae": float(hourly["oracle_error"].abs().mean()),
            "oracle_wm4_bias": float(hourly["oracle_error"].mean()),
            "oracle_F0_mean": float(daily["oracle_F0_star"].mean()),
            "oracle_F0_std": float(daily["oracle_F0_star"].std(ddof=1)),
            "oracle_R0_mean": float(daily["oracle_R0_star"].mean()),
            "oracle_R0_std": float(daily["oracle_R0_star"].std(ddof=1)),
            "oracle_F0_in_prior_marginal_95_pct": 100.0 * float(daily["oracle_F0_within_prior_marginal_95"].mean()),
            "oracle_R0_in_prior_marginal_95_pct": 100.0 * float(daily["oracle_R0_within_prior_marginal_95"].mean()),
        })
    return pd.DataFrame(result)


def run_all_days(frame, start=START_DAY, end=END_DAY, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, quantile_bounds=(0.025, 0.975), max_state_iter=50, state_tol=1e-8, fail_fast=False, oracle_comparison=False):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")
    daily_rows, hourly_frames, skipped = [], [], []

    for day in pd.date_range(start, end, freq="D"):
        try:
            daily, hourly = day_result(
                frame, day, baseline_weeks=baseline_weeks, steps_per_hour=steps_per_hour,
                quantile_bounds=quantile_bounds,
                max_state_iter=max_state_iter, state_tol=state_tol, oracle_comparison=oracle_comparison,
            )
            daily_rows.append(daily); hourly_frames.append(hourly)
            oracle_text = f" | oracle_coverage={daily['oracle_coverage_pct']:6.2f}% | oracle_RMSE={daily['oracle_wm4_rmse']:.8f}" if oracle_comparison else ""
            print(
                f"{day:%Y-%m-%d} | coverage={daily['coverage_pct']:6.2f}% "
                f"| mean_width={daily['mean_interval_width']:.6f} "
                f"| max_width={daily['max_interval_width']:.6f} "
                f"| RMSE={daily['wm4_rmse']:.8f}{oracle_text}"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc))); print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not daily_rows: raise RuntimeError("No days in the requested range could be fitted.")
    daily = pd.DataFrame(daily_rows).sort_values("day").reset_index(drop=True)
    hourly = pd.concat(hourly_frames, ignore_index=True).sort_values(["day", "forecast_hour"]).reset_index(drop=True)
    return daily, hourly, skipped


def print_summary(summary, horizon, skipped, output):
    row = summary.iloc[0]
    print()
    print("initial-state uncertainty summary")
    print(f"days fitted={int(row['fitted_days'])}, skipped={int(row['skipped_days'])}, hourly observations={int(row['hourly_observations'])}")
    print(f"nominal interval coverage={row['nominal_coverage_pct']:.2f}%")
    print(f"overall empirical coverage={row['overall_coverage_pct']:.2f}%")
    print(f"mean daily coverage={row['mean_daily_coverage_pct']:.2f}%")
    print(f"mean interval width={row['mean_interval_width']:.8f}, max interval width={row['max_interval_width']:.8f}")
    print(f"aggregate WM4 RMSE={row['wm4_rmse']:.10f}, MAE={row['wm4_mae']:.10f}, bias={row['wm4_bias']:.10f}")
    oracle_enabled = "oracle_overall_coverage_pct" in summary.columns
    if oracle_enabled:
        print()
        print("oracle-centred comparison (same covariance and interval width; full prediction-day I used to fit x*)")
        print(f"oracle empirical coverage={row['oracle_overall_coverage_pct']:.2f}% (lift={row['oracle_coverage_lift_pct_points']:+.2f} percentage points)")
        print(f"oracle WM4 RMSE={row['oracle_wm4_rmse']:.10f}, MAE={row['oracle_wm4_mae']:.10f}, bias={row['oracle_wm4_bias']:.10f}")
        print(f"oracle F0 inside historical prior marginal 95%={row['oracle_F0_in_prior_marginal_95_pct']:.2f}%")
        print(f"oracle R0 inside historical prior marginal 95%={row['oracle_R0_in_prior_marginal_95_pct']:.2f}%")
        print("Oracle coverage is an in-sample diagnostic, not out-of-sample forecast calibration.")
    print()
    if oracle_enabled: print(f"{'hour':>6}{'coverage':>14}{'oracle cov.':>14}{'mean width':>16}{'RMSE':>14}{'oracle RMSE':>14}")
    else: print(f"{'hour':>6}{'coverage':>14}{'mean width':>16}{'RMSE':>14}")
    for hour in (0, 3, 6, 12, 18, 23):
        part = horizon.loc[horizon["forecast_hour"] == hour]
        if part.empty: continue
        item = part.iloc[0]
        if oracle_enabled:
            print(f"{hour:>6d}{item['coverage_pct']:>13.2f}%{item['oracle_coverage_pct']:>13.2f}%{item['mean_interval_width']:>16.8f}{item['wm4_rmse']:>14.8f}{item['oracle_wm4_rmse']:>14.8f}")
        else: print(f"{hour:>6d}{item['coverage_pct']:>13.2f}%{item['mean_interval_width']:>16.8f}{item['wm4_rmse']:>14.8f}")
    if skipped:
        print(); print("skipped days:")
        for day, reason in skipped: print(f"{day:%Y-%m-%d}: {reason}")
    print(); print(f"output={output}")


def main():
    parser = ArgumentParser(description="Evaluate WM4 initial-state uncertainty intervals across all usable prediction days.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--quantile-bounds", type=float, nargs=2, default=(0.025, 0.975), metavar=("LOWER", "UPPER"))
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--oracle-comparison", action="store_true", help="Also fit x* to the full prediction-day I and compare coverage using the identical historical initial-state covariance (diagnostic/leaky).")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    quantile_bounds = tuple(args.quantile_bounds)
    frame, _ = aligned_wm4_data("MED")
    daily, hourly, skipped = run_all_days(
        frame, start=args.start_day, end=args.end_day,
        baseline_weeks=args.baseline_weeks, steps_per_hour=args.steps_per_hour,
        quantile_bounds=quantile_bounds,
        max_state_iter=args.max_state_iter, state_tol=args.state_tol, fail_fast=args.fail_fast,
        oracle_comparison=args.oracle_comparison,
    )
    horizon = horizon_summary(hourly)
    weekdays = weekday_summary(daily)
    summary = overall_summary(daily, hourly, skipped, quantile_bounds)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_metrics.csv", index=False)
    hourly.to_csv(output / "hourly_predictions.csv", index=False)
    horizon.to_csv(output / "horizon_summary.csv", index=False)
    weekdays.to_csv(output / "weekday_summary.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    if args.oracle_comparison:
        daily[[
            "day", "weekday", "F0_hat", "R0_hat", "F0_std", "R0_std", "cov_FR",
            "oracle_F0_star", "oracle_R0_star", "oracle_F0_minus_hat", "oracle_R0_minus_hat",
            "oracle_F0_within_prior_marginal_95", "oracle_R0_within_prior_marginal_95",
        ]].to_csv(output / "oracle_initial_states.csv", index=False)
    if skipped: pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)
    print_summary(summary, horizon, skipped, output)


if __name__ == "__main__": main()
