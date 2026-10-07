from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY, fit_history_optimized
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup
from src.params.wm_4 import optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_i0_initial_state"
Z95 = 1.959963984540054


def output_at(output_model, frame, timestamp):
    timestamp = pd.Timestamp(timestamp)
    row = frame.reindex(pd.DatetimeIndex([timestamp]))
    if row["Temp"].isna().any(): raise ValueError(f"Missing temperature at {timestamp}.")
    return float(output_model.predict(pd.DatetimeIndex([timestamp]), row["Temp"].to_numpy(dtype=float))[0])


def i0_distribution(frame, training_days):
    index = pd.DatetimeIndex([day_start(day) for day in training_days])
    values = frame.reindex(index)["I"].to_numpy(dtype=float)
    if values.size < 2 or not np.all(np.isfinite(values)): raise ValueError("Historical I0 distribution requires at least two finite corresponding-weekday I0 samples.")
    return values, float(np.mean(values)), float(np.std(values, ddof=1))


def one_hour_map(model, r, U_minus_1, steps_per_hour):
    r = np.asarray(r, dtype=float)
    A = np.array([
        [1.0, model.dt],
        [model.dt * r[0], 1.0 + model.dt * r[2]],
    ], dtype=float)
    b = np.array([0.0, model.dt * r[1] * float(U_minus_1)], dtype=float)
    M, c = np.eye(2, dtype=float), np.zeros(2, dtype=float)
    for _ in range(int(steps_per_hour)):
        c = A @ c + b
        M = A @ M
    return M, c


def i0_initial_state(frame, start, output_model, model, fit, steps_per_hour):
    start = day_start(start)
    previous = start - pd.Timedelta(hours=1)
    row = frame.reindex(pd.DatetimeIndex([previous]))
    if row[["I", "U", "Temp"]].isna().any().any(): raise ValueError(f"Missing I/U/temperature at t=-1 ({previous}).")

    samples, I0_mean, I0_std = i0_distribution(frame, fit["training_days"])
    I_minus_1 = float(row["I"].iloc[0])
    U_minus_1 = float(row["U"].iloc[0])
    O_minus_1 = output_at(output_model, frame, previous)
    O0 = output_at(output_model, frame, start)
    R_minus_1 = I_minus_1 - O_minus_1
    R0 = I0_mean - O0

    M, c = one_hour_map(model, fit["r"], U_minus_1, steps_per_hour)
    denominator = float(M[1, 0])
    if abs(denominator) <= 1e-10: raise ValueError("The one-hour WM4 transition is singular for recovering F_-1 from estimated I0.")

    F_minus_1 = float((R0 - M[1, 1] * R_minus_1 - c[1]) / denominator)
    F0 = float(M[0, 0] * F_minus_1 + M[0, 1] * R_minus_1 + c[0])
    if not -1.0 <= F0 <= 1.0: raise ValueError(f"I0-inferred F0={F0:.6f} lies outside [-1,1].")

    jacobian = np.array([M[0, 0] / denominator, 1.0], dtype=float)
    P0 = (I0_std ** 2) * np.outer(jacobian, jacobian)
    init = np.array([F0, R0], dtype=float)

    return {
        "init_cond": init,
        "P0": P0,
        "std": np.sqrt(np.maximum(np.diag(P0), 0.0)),
        "jacobian": jacobian,
        "I0_samples": samples,
        "I0_mean": I0_mean,
        "I0_std": I0_std,
        "I_minus_1": I_minus_1,
        "O_minus_1": O_minus_1,
        "R_minus_1": R_minus_1,
        "U_minus_1": U_minus_1,
        "F_minus_1": F_minus_1,
        "O0": O0,
        "transition": M,
        "offset": c,
    }


def prediction_state(model, init, U, Time, r):
    return model.deterministic_solver(np.asarray(init, dtype=float), {"U": U, "PE": U, "Time": Time, "r": r})


def day_result(frame, start, baseline_weeks, steps_per_hour, **fit_kwargs):
    start = day_start(start)
    output_model, output_fit, fit, _ = fit_history_optimized(
        frame, start, baseline_weeks, steps_per_hour, **fit_kwargs,
    )
    _, actual, U, Time, model, _ = prediction_setup(frame, start, output_model, steps_per_hour)
    hourly_index = np.arange(DAY_HOURS, dtype=int) * int(steps_per_hour)

    i0_fit = i0_initial_state(frame, start, output_model, model, fit, steps_per_hour)
    i0_init = i0_fit["init_cond"]
    i0_state = prediction_state(model, i0_init, U, Time, fit["r"])

    historical_init = np.asarray(fit["x_hat"], dtype=float)
    historical_P0 = np.asarray(fit["x_cov"], dtype=float)
    historical_state = prediction_state(model, historical_init, U, Time, fit["r"])

    prediction_data, _, _, _ = model_input(frame, start, steps_per_hour, 0, output_model)
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(prediction_data, fit["r"])
    oracle_init = np.asarray(oracle["init_cond"], dtype=float)
    oracle_state = oracle["deterministic_state"]

    observed = actual["I"].to_numpy(dtype=float)
    i0_predicted = np.asarray(i0_state["I"], dtype=float)[hourly_index]
    historical_predicted = np.asarray(historical_state["I"], dtype=float)[hourly_index]
    oracle_predicted = np.asarray(oracle_state["I"], dtype=float)[hourly_index]
    baseline = np.asarray(i0_state["O"], dtype=float)[hourly_index]

    i0_metrics = deviation_metrics(observed, i0_predicted)
    historical_metrics = deviation_metrics(observed, historical_predicted)
    oracle_metrics = deviation_metrics(observed, oracle_predicted)
    output_metrics = deviation_metrics(observed, baseline)

    i0_error = i0_init - oracle_init
    historical_error = historical_init - oracle_init
    r = np.asarray(fit["r"], dtype=float)
    i0_std = i0_fit["std"]
    historical_std = np.sqrt(np.maximum(np.diag(historical_P0), 0.0))
    actual_I0 = float(observed[0])

    daily = {
        "day": start, "weekday": start.day_name(),
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "I_minus_1": i0_fit["I_minus_1"], "O_minus_1": i0_fit["O_minus_1"],
        "R_minus_1": i0_fit["R_minus_1"], "U_minus_1": i0_fit["U_minus_1"],
        "F_minus_1_inferred": i0_fit["F_minus_1"],
        "I0_estimated_mean": i0_fit["I0_mean"], "I0_estimated_std": i0_fit["I0_std"],
        "I0_actual": actual_I0, "I0_error": i0_fit["I0_mean"] - actual_I0,
        "I0_z_error": (i0_fit["I0_mean"] - actual_I0) / i0_fit["I0_std"] if i0_fit["I0_std"] > 0.0 else np.nan,
        "I0_sample_min": float(np.min(i0_fit["I0_samples"])),
        "I0_sample_max": float(np.max(i0_fit["I0_samples"])),
        "i0_F0": i0_init[0], "i0_R0": i0_init[1],
        "i0_F0_std": i0_std[0], "i0_R0_std": i0_std[1],
        "i0_P0_FF": i0_fit["P0"][0, 0], "i0_P0_FR": i0_fit["P0"][0, 1], "i0_P0_RR": i0_fit["P0"][1, 1],
        "i0_dF0_dI0": i0_fit["jacobian"][0], "i0_dR0_dI0": i0_fit["jacobian"][1],
        "historical_F0": historical_init[0], "historical_R0": historical_init[1],
        "historical_F0_std": historical_std[0], "historical_R0_std": historical_std[1],
        "oracle_F0": oracle_init[0], "oracle_R0": oracle_init[1], "oracle_fit_mse": float(oracle["loss"]),
        "i0_F0_error": i0_error[0], "i0_R0_error": i0_error[1],
        "historical_F0_error": historical_error[0], "historical_R0_error": historical_error[1],
        "i0_state_error_norm": float(np.linalg.norm(i0_error)),
        "historical_state_error_norm": float(np.linalg.norm(historical_error)),
        "i0_mse": i0_metrics["mse"], "i0_rmse": i0_metrics["rmse"], "i0_mae": i0_metrics["mae"], "i0_bias": i0_metrics["bias"],
        "historical_mse": historical_metrics["mse"], "historical_rmse": historical_metrics["rmse"], "historical_mae": historical_metrics["mae"], "historical_bias": historical_metrics["bias"],
        "oracle_mse": oracle_metrics["mse"], "oracle_rmse": oracle_metrics["rmse"], "oracle_mae": oracle_metrics["mae"], "oracle_bias": oracle_metrics["bias"],
        "O_mse": output_metrics["mse"], "O_rmse": output_metrics["rmse"], "O_mae": output_metrics["mae"], "O_bias": output_metrics["bias"],
        "O_fit_mse": output_fit["mse"], "state_iterations": int(fit["state_iterations"]),
    }

    hourly = pd.DataFrame({
        "day": start,
        "hour": np.arange(DAY_HOURS, dtype=int),
        "I_observed": observed,
        "I_i0": i0_predicted,
        "I_historical": historical_predicted,
        "I_oracle": oracle_predicted,
        "O": baseline,
        "U": np.asarray(i0_state["U"], dtype=float)[hourly_index],
        "F_i0": np.asarray(i0_state["F"], dtype=float)[hourly_index],
        "R_i0": np.asarray(i0_state["R"], dtype=float)[hourly_index],
        "F_historical": np.asarray(historical_state["F"], dtype=float)[hourly_index],
        "R_historical": np.asarray(historical_state["R"], dtype=float)[hourly_index],
        "F_oracle": np.asarray(oracle_state["F"], dtype=float)[hourly_index],
        "R_oracle": np.asarray(oracle_state["R"], dtype=float)[hourly_index],
    }, index=actual.index)
    hourly.index.name = "time"

    samples = pd.DataFrame({
        "prediction_day": start,
        "training_day": fit["training_days"],
        "I0_sample": i0_fit["I0_samples"],
        "optimized_F0": np.asarray(fit["init_cond"], dtype=float)[:, 0],
        "optimized_R0": np.asarray(fit["init_cond"], dtype=float)[:, 1],
    })
    return daily, hourly, samples


def horizon_metrics(hourly):
    rows = []
    for hour, data in hourly.groupby("hour", sort=True):
        observed = data["I_observed"].to_numpy(dtype=float)
        row = {"hour": int(hour), "observations": int(len(data))}
        for name, column in (("i0", "I_i0"), ("historical", "I_historical"), ("oracle", "I_oracle")):
            metrics = deviation_metrics(observed, data[column].to_numpy(dtype=float))
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def save_summary_plot(daily, path):
    day = pd.to_datetime(daily["day"])
    fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True)

    axes[0].plot(day, daily["i0_mse"], label="I0-inferred state")
    axes[0].plot(day, daily["historical_mse"], label="Historical direct state")
    axes[0].plot(day, daily["oracle_mse"], label="Oracle state")
    axes[0].set_ylabel("Daily MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(day, daily["i0_F0"], label="I0-inferred F0")
    axes[1].fill_between(day, daily["i0_F0"] - Z95 * daily["i0_F0_std"], daily["i0_F0"] + Z95 * daily["i0_F0_std"], alpha=0.15, label="I0-derived 95% state interval")
    axes[1].plot(day, daily["historical_F0"], label="Historical direct F0")
    axes[1].plot(day, daily["oracle_F0"], label="Oracle F0")
    axes[1].set_ylabel("F0")
    axes[1].legend(ncol=2)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(day, daily["i0_R0"], label="I0-inferred R0")
    axes[2].fill_between(day, daily["i0_R0"] - Z95 * daily["i0_R0_std"], daily["i0_R0"] + Z95 * daily["i0_R0_std"], alpha=0.15, label="I0-derived 95% state interval")
    axes[2].plot(day, daily["historical_R0"], label="Historical direct R0")
    axes[2].plot(day, daily["oracle_R0"], label="Oracle R0")
    axes[2].set_ylabel("R0")
    axes[2].legend(ncol=2)
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(day, daily["I0_estimated_mean"], label="Estimated I0 mean")
    axes[3].fill_between(day, daily["I0_estimated_mean"] - Z95 * daily["I0_estimated_std"], daily["I0_estimated_mean"] + Z95 * daily["I0_estimated_std"], alpha=0.18, label="Estimated I0 95% interval")
    axes[3].plot(day, daily["I0_actual"], marker=".", linestyle="none", label="Actual I0")
    axes[3].set_ylabel("Initial input I0")
    axes[3].set_xlabel("Prediction day")
    axes[3].legend()
    axes[3].grid(True, alpha=0.25)

    fig.suptitle("WM4 I0-inferred initial state vs historical direct state and oracle")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_state_error_plot(daily, path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    day = pd.to_datetime(daily["day"])

    axes[0].plot(day, np.abs(daily["i0_F0_error"]), label="|I0-inferred F0 - oracle F0|")
    axes[0].plot(day, np.abs(daily["historical_F0_error"]), label="|Historical F0 - oracle F0|")
    axes[0].set_ylabel("Absolute F0 error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(day, np.abs(daily["i0_R0_error"]), label="|I0-inferred R0 - oracle R0|")
    axes[1].plot(day, np.abs(daily["historical_R0_error"]), label="|Historical R0 - oracle R0|")
    axes[1].set_ylabel("Absolute R0 error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(day, daily["i0_state_error_norm"], label="I0-inferred state distance")
    axes[2].plot(day, daily["historical_state_error_norm"], label="Historical state distance")
    axes[2].set_ylabel("Euclidean distance to oracle")
    axes[2].set_xlabel("Prediction day")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("WM4 initial-state centre error relative to target-day oracle")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(daily, hourly, skipped):
    observed = hourly["I_observed"].to_numpy(dtype=float)
    print(f"days fitted={len(daily)}, skipped={len(skipped)}, hourly observations={len(hourly)}")
    print()
    print(f"{'method':<18}{'MSE':>14}{'RMSE':>14}{'MAE':>14}{'bias':>14}")
    for name, column in (("I0 inferred", "I_i0"), ("historical", "I_historical"), ("oracle", "I_oracle"), ("O(t)", "O")):
        metrics = deviation_metrics(observed, hourly[column].to_numpy(dtype=float))
        print(f"{name:<18}{metrics['mse']:>14.10f}{metrics['rmse']:>14.10f}{metrics['mae']:>14.10f}{metrics['bias']:>14.10f}")

    print()
    print(f"mean |I0 estimate - actual|={np.mean(np.abs(daily['I0_error'])):.8f}")
    print(f"mean I0 estimate std={daily['I0_estimated_std'].mean():.8f}")
    print(f"mean I0-derived F0 std={daily['i0_F0_std'].mean():.8f}")
    print(f"mean I0-derived R0 std={daily['i0_R0_std'].mean():.8f}")
    print(f"mean state distance to oracle: I0-inferred={daily['i0_state_error_norm'].mean():.8f}, historical={daily['historical_state_error_norm'].mean():.8f}")
    print(f"mean |F0-oracle|: I0-inferred={np.mean(np.abs(daily['i0_F0_error'])):.8f}, historical={np.mean(np.abs(daily['historical_F0_error'])):.8f}")
    print(f"mean |R0-oracle|: I0-inferred={np.mean(np.abs(daily['i0_R0_error'])):.8f}, historical={np.mean(np.abs(daily['historical_R0_error'])):.8f}")
    if skipped:
        print()
        print("skipped days:")
        for day, reason in skipped: print(f"{day:%Y-%m-%d}: {reason}")


def run_all_days(frame, start=START_DAY, end=END_DAY, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, fail_fast=False, **fit_kwargs):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")
    if int(baseline_weeks) < 2: raise ValueError("baseline_weeks must be at least 2 to estimate the I0 Normal distribution.")

    daily_rows, hourly_frames, sample_frames, skipped = [], [], [], []
    for day in pd.date_range(start, end, freq="D"):
        try:
            daily, hourly, samples = day_result(frame, day, baseline_weeks, steps_per_hour, **fit_kwargs)
            daily_rows.append(daily)
            hourly_frames.append(hourly)
            sample_frames.append(samples)
            print(
                f"{day:%Y-%m-%d} | I0={daily['I0_estimated_mean']:.6f} +/- {daily['I0_estimated_std']:.6f} "
                f"| x0_I0=[{daily['i0_F0']:.6f}, {daily['i0_R0']:.6f}] "
                f"| x0_hist=[{daily['historical_F0']:.6f}, {daily['historical_R0']:.6f}] "
                f"| x0_oracle=[{daily['oracle_F0']:.6f}, {daily['oracle_R0']:.6f}] "
                f"| MSE I0={daily['i0_mse']:.8f} hist={daily['historical_mse']:.8f} oracle={daily['oracle_mse']:.8f}"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc)))
            print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not daily_rows: raise RuntimeError("No days in the requested range could be fitted.")
    return (
        pd.DataFrame(daily_rows).sort_values("day").reset_index(drop=True),
        pd.concat(hourly_frames).sort_index(),
        pd.concat(sample_frames, ignore_index=True),
        skipped,
    )


def main():
    parser = ArgumentParser(description="Evaluate causal WM4 initialization by estimating target-day I0 from previous corresponding weekdays, recovering F_-1 from the last observed I_-1, propagating to x0, and comparing with historical-direct and oracle states.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS, help="Previous corresponding weekdays used both for the WM4 fit and the Normal I0 estimate; default 5.")
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    daily, hourly, samples, skipped = run_all_days(
        frame, args.start_day, args.end_day, args.baseline_weeks, args.steps_per_hour,
        fail_fast=args.fail_fast, max_state_iter=args.max_state_iter, state_tol=args.state_tol,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_comparison.csv", index=False)
    hourly.to_csv(output / "hourly_predictions.csv")
    samples.to_csv(output / "historical_i0_samples.csv", index=False)
    horizon_metrics(hourly).to_csv(output / "horizon_metrics.csv", index=False)
    if skipped: pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)
    save_summary_plot(daily, output / "i0_initial_state_comparison.png")
    save_state_error_plot(daily, output / "state_error_vs_oracle.png")
    print()
    print_summary(daily, hourly, skipped)
    print(f"output={output}")


if __name__ == "__main__": main()
