from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.moving_horizon_common import HOURS, baseline_before, group_frame, metrics, window_data
from src.params.wm_4 import joint_qp_params
from src.validation.exe.run_model.wm_4.predict_all_moving_horizon import predict_all_moving_horizon_validation as validation


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "predict_all_moving_horizon"
TRAINING_WEEKDAYS = 5
QUINTILES = (0, 20, 40, 60, 80, 100)


def historical_weekdays(day, count=TRAINING_WEEKDAYS):
    return [pd.Timestamp(day) - pd.Timedelta(weeks=k) for k in range(count, 0, -1)]


def oracle_f0(model, forecast, observed, theta, historical_r0, steps_per_hour):
    """Optimize only F0 using the target day's observations; R0 is historical."""
    idx = np.arange(HOURS, dtype=int) * steps_per_hour
    zero = model._solve_deterministic(
        [0.0, historical_r0], forecast["U"], forecast["PE"],
        forecast["Time"], forecast["O"], theta)
    sensitivity = model.sensitivity({"r": theta})["input"][idx, 0]
    observed, base, sensitivity = validation.oracle(
        observed, np.asarray(zero["I"])[idx], sensitivity, historical_r0)
    f0 = float(np.clip(sensitivity @ (observed-base) / (sensitivity @ sensitivity), -1.0, 1.0))
    state = model._solve_deterministic(
        [f0, historical_r0], forecast["U"], forecast["PE"],
        forecast["Time"], forecast["O"], theta)
    return f0, state


def evaluate_day(frame, day, baseline, steps_per_hour):
    history = historical_weekdays(day)
    datasets = [window_data(frame, earlier, baseline, steps_per_hour)[1] for earlier in history]
    model, forecast, _, observed = window_data(frame, day, baseline, steps_per_hour)
    started = perf_counter()
    fit = joint_qp_params(model).fit_many(datasets)
    fit_seconds = perf_counter()-started

    # R0 is the last observed pre-forecast deviation, not a target-day oracle fit.
    previous = pd.Timestamp(day) - pd.Timedelta(hours=1)
    previous_data = frame.reindex(pd.DatetimeIndex([previous]))[["I", "Temp"]]
    if previous_data.isna().any().any():
        raise ValueError(f"Missing last historical observation at {previous}.")
    prev_B = float(baseline.predict(pd.DatetimeIndex([previous]), previous_data["Temp"].to_numpy())[0])
    historical_r0 = float(previous_data["I"].iloc[0] - prev_B)

    theta = np.asarray(fit["r"], dtype=float)
    f0, state = oracle_f0(model, forecast, observed, theta, historical_r0, steps_per_hour)
    idx = np.arange(HOURS, dtype=int) * steps_per_hour
    predicted = np.asarray(state["I"], dtype=float)[idx]
    reference = np.asarray(forecast["O"], dtype=float)[idx]
    score, baseline_score = metrics(observed, predicted), metrics(observed, reference)
    rate = np.asarray(state["R"], dtype=float)[idx]
    restoring, price, damping = map(float, theta)  # WM4 stores [theta_F, theta_U, theta_R].
    zeta = float(-damping/(2.0*np.sqrt(-restoring)))
    row = {
        "day": pd.Timestamp(day), "weekday": pd.Timestamp(day).day_name(),
        "training_days": ",".join(f"{d:%Y-%m-%d}" for d in history),
        "theta_F": restoring, "theta_U": price, "theta_R": damping,
        "damping_ratio": zeta, "damping_branch": "under" if zeta < 1.0 else "over" if zeta > 1.0 else "critical",
        "oracle_F0": f0, "historical_R0": historical_r0,
        "training_F0_mean": float(np.mean(fit["init_cond"][:, 0])),
        "training_F0_std": float(np.std(fit["init_cond"][:, 0])),
        "fit_qp_loss": float(fit["qp_loss"]),
        "fit_free_run_rmse": float(np.sqrt(fit["deterministic_loss"])),
        "fit_seconds": fit_seconds,
        "design_rank": int(fit["design_rank"]), "design_columns": len(fit["beta"]),
        "rmse": score["rmse"], "mse": score["mse"], "mae": score["mae"], "bias": score["bias"],
        "baseline_rmse": baseline_score["rmse"], "baseline_mse": baseline_score["mse"],
    }
    times = pd.date_range(pd.Timestamp(day), periods=HOURS, freq="h")
    hourly = pd.DataFrame({
        "day": pd.Timestamp(day), "hour": np.arange(HOURS, dtype=int), "time": times,
        "observed": observed, "prediction": predicted, "baseline": reference,
        "error": predicted-observed, "baseline_error": reference-observed,
        "F": np.asarray(state["F"])[idx], "R": rate, "U": np.asarray(state["U"])[idx],
    })
    return row, hourly


def horizon_analysis(hourly):
    matrix = hourly.pivot(index="day", columns="hour", values="error").sort_index()
    if matrix.isna().any().any() or matrix.shape[1] != HOURS:
        raise RuntimeError("Every evaluated day must contain all 24 forecast hours.")
    errors = matrix.to_numpy(dtype=float)
    cumulative = np.sqrt(np.cumsum(errors**2, axis=1) / np.arange(1, HOURS+1)[None, :])
    absolute = np.abs(errors)
    table = pd.DataFrame({
        "hour": np.arange(HOURS, dtype=int), "days": len(errors),
        "aggregate_rmse": np.sqrt(np.mean(errors**2, axis=0)),
        "aggregate_cumulative_rmse": np.sqrt(np.cumsum(np.mean(errors**2, axis=0)) / np.arange(1, HOURS+1)),
        "mae": np.mean(absolute, axis=0), "bias": np.mean(errors, axis=0),
    })
    for q in QUINTILES:
        table[f"absolute_error_q{q:02d}"] = np.percentile(absolute, q, axis=0)
        table[f"cumulative_rmse_q{q:02d}"] = np.percentile(cumulative, q, axis=0)
    return table


def save_rmse_plot(horizon, output):
    x = horizon["hour"].to_numpy(dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(x, horizon["aggregate_rmse"], lw=2.2, label="Pooled RMSE at each forecast hour")
    axes[0].fill_between(x, horizon["absolute_error_q20"].to_numpy(), horizon["absolute_error_q80"].to_numpy(),
                         alpha=0.16, label="20th-80th percentile of daily absolute error")
    for q in (20, 40, 60, 80):
        axes[0].plot(x, horizon[f"absolute_error_q{q:02d}"], lw=1, ls="--",
                     label=f"Absolute-error Q{q}")
    axes[0].set_title("Hourly prediction error across all evaluated days")
    axes[0].set_ylabel("Normalized demand error"); axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].plot(x, horizon["aggregate_cumulative_rmse"], lw=2.2, label="Pooled RMSE through forecast hour")
    axes[1].fill_between(x, horizon["cumulative_rmse_q20"].to_numpy(), horizon["cumulative_rmse_q80"].to_numpy(),
                         alpha=0.16, label="20th-80th percentile of per-day cumulative RMSE")
    for q in (20, 40, 60, 80):
        axes[1].plot(x, horizon[f"cumulative_rmse_q{q:02d}"], lw=1, ls="--",
                     label=f"Cumulative-RMSE Q{q}")
    axes[1].set_title("RMSE accumulated through each forecast hour")
    axes[1].set_ylabel("Normalized demand RMSE")
    axes[1].set_xlabel("Forecast hour (0-23; 24 hourly samples)")
    axes[1].grid(alpha=0.25); axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = output / "rmse_horizon_quintiles.png"
    fig.savefig(path, dpi=190, bbox_inches="tight"); plt.close(fig)
    return path


def save_daily_rmse_plot(daily, output):
    values = daily["rmse"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(values, bins=min(30, max(5, int(np.ceil(np.sqrt(len(values)))))), alpha=0.65)
    for q in (20, 40, 60, 80):
        value = float(np.percentile(values, q))
        ax.axvline(value, ls="--", lw=1.3, label=f"Q{q} = {value:.4f}")
    ax.set_title("Distribution of 24-hour WM4 oracle-F0 forecast RMSE")
    ax.set_xlabel("Per-day RMSE (normalized demand)"); ax.set_ylabel("Days")
    ax.grid(alpha=0.2); ax.legend()
    fig.tight_layout()
    path = output / "daily_rmse_quintiles.png"
    fig.savefig(path, dpi=190, bbox_inches="tight"); plt.close(fig)
    return path


def print_analysis(daily, hourly, skipped, start, end, output):
    errors = hourly["error"].to_numpy(dtype=float)
    reference_errors = hourly["baseline_error"].to_numpy(dtype=float)
    print("\nWM4 5-matching-weekday / 24-hour oracle-F0 moving-horizon study")
    print(f"requested interval: {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
    print(f"successful days: {len(daily)} | skipped days: {len(skipped)} | hourly scores: {len(errors)}")
    print(f"pooled WM4 RMSE: {np.sqrt(np.mean(errors**2)):.8f} | baseline RMSE: {np.sqrt(np.mean(reference_errors**2)):.8f}")
    print(f"pooled bias (prediction - observation): {np.mean(errors):.8f}")
    print("\n24-hour per-day RMSE quintile boundaries:")
    print(pd.Series(np.percentile(daily["rmse"], QUINTILES), index=[f"Q{q}" for q in QUINTILES]).to_string())
    print("\nFitted theta distribution across forecast days (training windows only):")
    names = ["theta_F", "theta_R", "theta_U"]
    stats = daily[names].describe(percentiles=[.2, .4, .5, .6, .8]).T
    print(stats[["count", "mean", "std", "min", "20%", "40%", "50%", "60%", "80%", "max"]].to_string(float_format=lambda v: f"{v:.8f}"))
    print("\nTheta correlations:")
    print(daily[names].corr().to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nStability and parameter identification:")
    print(f"theta_F<0: {(daily['theta_F']<0).mean()*100:.1f}% | theta_R<0: {(daily['theta_R']<0).mean()*100:.1f}% | theta_U<=0: {(daily['theta_U']<=0).mean()*100:.1f}%")
    print(f"near-zero price coefficient (|theta_U|<1e-6): {(daily['theta_U'].abs()<1e-6).mean()*100:.1f}%")
    print(f"rank-deficient joint designs: {(daily['design_rank']<daily['design_columns']).sum()}/{len(daily)}")
    print(f"damping branches: {daily['damping_branch'].value_counts().to_dict()}")
    print(f"mean damping ratio: {daily['damping_ratio'].mean():.6f}")
    print(f"mean QP derivative loss: {daily['fit_qp_loss'].mean():.8f}")
    print(f"median QP runtime: {daily['fit_seconds'].median():.3f}s")
    print("\nBy weekday:")
    by_weekday = daily.groupby("weekday").agg(
        days=("day", "size"), aggregate_mse=("mse", "mean"),
        theta_F=("theta_F", "mean"), theta_R=("theta_R", "mean"), theta_U=("theta_U", "mean"))
    by_weekday["rmse"] = np.sqrt(by_weekday.pop("aggregate_mse"))
    print(by_weekday.to_string(float_format=lambda v: f"{v:.6f}"))
    print(f"\noutputs: {output}")
    print("NOTE: F0 is an ORACLE optimized from the 24 target observations; this is not deployable forecast performance.")
    print("R0 uses the last pre-forecast observed deviation; the 2022 full-period normalization is retrospective.")


def run(args):
    validation.arguments(args)
    frame = group_frame(args.group)
    earliest, latest = frame.index.min().normalize(), frame.index.max().normalize()
    latest = min(latest, pd.Timestamp("2022-10-31"))
    start = max(pd.Timestamp(args.start_day) if args.start_day else earliest, pd.Timestamp("2022-01-01"))
    end = min(pd.Timestamp(args.end_day) if args.end_day else latest, latest)
    if end < start: raise ValueError("No days remain inside the available Jan-End dataset.")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    daily_rows, hourly_frames, skipped, baselines = [], [], [], {}
    days = pd.date_range(start, end, freq="D")
    for count, day in enumerate(days, 1):
        try:
            # Five complete, strictly preceding occurrences of the target weekday.
            history = historical_weekdays(day, args.training_weekdays)
            if history[0] < earliest:
                raise ValueError("Five previous matching weekdays are not yet available (warm-up).")
            baseline = baseline_before(frame, day, baselines)
            row, hourly = evaluate_day(frame, day, baseline, args.steps_per_hour)
            daily_rows.append(row); hourly_frames.append(hourly)
            if args.max_days and len(daily_rows) >= args.max_days: break
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if args.fail_fast: raise
            skipped.append({"day": day, "reason": str(exc)})
        if count % args.progress_every == 0:
            print(f"[WM4] {day:%Y-%m-%d}: {len(daily_rows)} evaluated, {len(skipped)} skipped", flush=True)
    pd.DataFrame(skipped, columns=["day", "reason"]).to_csv(output / "skipped_days.csv", index=False)
    if not daily_rows: raise RuntimeError("No full 24-hour predictions completed; consult skipped_days.csv.")
    daily = pd.DataFrame(daily_rows).sort_values("day")
    hourly = pd.concat(hourly_frames, ignore_index=True).sort_values(["day", "hour"])
    horizon = horizon_analysis(hourly)
    daily.to_csv(output / "daily_metrics.csv", index=False)
    hourly.to_csv(output / "hourly_predictions.csv", index=False)
    horizon.to_csv(output / "horizon_rmse_quintiles.csv", index=False)
    save_rmse_plot(horizon, output)
    save_daily_rmse_plot(daily, output)
    print_analysis(daily, hourly, skipped, start, end, output)
    return daily, hourly, horizon


def main():
    parser = ArgumentParser(description="Complete Jan-End WM4 five-corresponding-weekday 24h moving-horizon oracle-F0 analysis.")
    parser.add_argument("--group", choices=("MED", "ALL"), default="MED")
    parser.add_argument("--start-day", default=None)
    parser.add_argument("--end-day", default=None)
    parser.add_argument("--training-weekdays", type=int, default=TRAINING_WEEKDAYS)
    parser.add_argument("--steps-per-hour", type=int, default=4)
    parser.add_argument("--max-days", type=int, default=0, help="Limit successful predictions for a smoke run; 0=all.")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--fail-fast", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__": main()
