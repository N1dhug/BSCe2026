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
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_shared_initial_state"


def _state_design(helper, data, r):
    base = helper._state_from_parameters(data, np.array([0.0, 0.0]), r)
    f_unit = helper._state_from_parameters(data, np.array([1.0, 0.0]), r)
    r_unit = helper._state_from_parameters(data, np.array([0.0, 1.0]), r)
    obs = np.asarray(data["observation_mask"], dtype=bool)
    X = np.column_stack((
        (f_unit["I"] - base["I"])[obs],
        (r_unit["I"] - base["I"])[obs],
    ))
    y = np.asarray(data["I"], dtype=float)[obs] - np.asarray(base["I"], dtype=float)[obs]
    return X, y, obs


def fit_shared_initial_state(frame, output_model, model, fit, steps_per_hour, f_bounds=(-1.0, 1.0), r_bounds=(-1.0, 1.0)):
    helper = optimized_qp_params(model)
    datasets, designs, targets = [], [], []

    for day in fit["training_days"]:
        raw, _, _, _ = model_input(frame, day, steps_per_hour, 0, output_model)
        data = helper._prepare_data(raw)
        X, y, _ = _state_design(helper, data, fit["r"])
        datasets.append(data)
        designs.append(X)
        targets.append(y)

    X = np.vstack(designs)
    y = np.concatenate(targets)
    lb = np.array([float(f_bounds[0]), float(r_bounds[0])], dtype=float)
    ub = np.array([float(f_bounds[1]), float(r_bounds[1])], dtype=float)
    shared = model._least_squares_qp(X, y, lb=lb, ub=ub, ridge=model._FIT_RIDGE)

    historical = np.asarray(fit["x_hat"], dtype=float)
    shared_losses, historical_losses = [], []
    shared_states, historical_states = [], []
    for data in datasets:
        obs = np.asarray(data["observation_mask"], dtype=bool)
        shared_state = helper._state_from_parameters(data, shared, fit["r"], full=True)
        historical_state = helper._state_from_parameters(data, historical, fit["r"], full=True)
        shared_residual = np.asarray(shared_state["I"], dtype=float)[obs] - np.asarray(data["I"], dtype=float)[obs]
        historical_residual = np.asarray(historical_state["I"], dtype=float)[obs] - np.asarray(data["I"], dtype=float)[obs]
        shared_losses.append(float(np.mean(shared_residual ** 2)))
        historical_losses.append(float(np.mean(historical_residual ** 2)))
        shared_states.append(shared_state)
        historical_states.append(historical_state)

    return {
        "init_cond": np.asarray(shared, dtype=float),
        "training_mse": float(np.mean(shared_losses)),
        "per_day_training_mse": np.asarray(shared_losses, dtype=float),
        "historical_training_mse": float(np.mean(historical_losses)),
        "historical_per_day_training_mse": np.asarray(historical_losses, dtype=float),
        "individual_training_mse": float(np.mean(np.asarray(fit["per_trajectory_loss"], dtype=float))),
        "shared_states": shared_states,
        "historical_states": historical_states,
        "datasets": datasets,
    }


def _predict(model, init, U, Time, r):
    return model.deterministic_solver(
        np.asarray(init, dtype=float),
        {"U": U, "PE": U, "Time": Time, "r": r},
    )


def day_result(frame, start, baseline_weeks, steps_per_hour, **fit_kwargs):
    start = day_start(start)
    output_model, output_fit, fit, fit_model = fit_history_optimized(
        frame, start, baseline_weeks, steps_per_hour, **fit_kwargs,
    )

    shared_fit = fit_shared_initial_state(
        frame, output_model, fit_model, fit, steps_per_hour,
    )
    shared_init = np.asarray(shared_fit["init_cond"], dtype=float)
    historical_init = np.asarray(fit["x_hat"], dtype=float)

    _, actual, U, Time, model, _ = prediction_setup(
        frame, start, output_model, steps_per_hour,
    )
    shared_state = _predict(model, shared_init, U, Time, fit["r"])
    historical_state = _predict(model, historical_init, U, Time, fit["r"])

    prediction_data, _, _, _ = model_input(
        frame, start, steps_per_hour, 0, output_model,
    )
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(
        prediction_data, fit["r"],
    )
    oracle_init = np.asarray(oracle["init_cond"], dtype=float)
    oracle_state = oracle["deterministic_state"]

    hourly_index = np.arange(DAY_HOURS, dtype=int) * int(steps_per_hour)
    observed = actual["I"].to_numpy(dtype=float)
    shared_I = np.asarray(shared_state["I"], dtype=float)[hourly_index]
    historical_I = np.asarray(historical_state["I"], dtype=float)[hourly_index]
    oracle_I = np.asarray(oracle_state["I"], dtype=float)[hourly_index]
    baseline = np.asarray(shared_state["O"], dtype=float)[hourly_index]

    shared_metrics = deviation_metrics(observed, shared_I)
    historical_metrics = deviation_metrics(observed, historical_I)
    oracle_metrics = deviation_metrics(observed, oracle_I)
    output_metrics = deviation_metrics(observed, baseline)

    shared_error = shared_init - oracle_init
    historical_error = historical_init - oracle_init
    r = np.asarray(fit["r"], dtype=float)

    daily = {
        "day": start,
        "weekday": start.day_name(),
        "r_F": r[0],
        "r_U": r[1],
        "r_R": r[2],
        "shared_F0": shared_init[0],
        "shared_R0": shared_init[1],
        "historical_F0": historical_init[0],
        "historical_R0": historical_init[1],
        "oracle_F0": oracle_init[0],
        "oracle_R0": oracle_init[1],
        "shared_F0_error": shared_error[0],
        "shared_R0_error": shared_error[1],
        "historical_F0_error": historical_error[0],
        "historical_R0_error": historical_error[1],
        "shared_state_error_norm": float(np.linalg.norm(shared_error)),
        "historical_state_error_norm": float(np.linalg.norm(historical_error)),
        "shared_mse": shared_metrics["mse"],
        "shared_rmse": shared_metrics["rmse"],
        "shared_mae": shared_metrics["mae"],
        "shared_bias": shared_metrics["bias"],
        "historical_mse": historical_metrics["mse"],
        "historical_rmse": historical_metrics["rmse"],
        "historical_mae": historical_metrics["mae"],
        "historical_bias": historical_metrics["bias"],
        "oracle_mse": oracle_metrics["mse"],
        "oracle_rmse": oracle_metrics["rmse"],
        "oracle_mae": oracle_metrics["mae"],
        "oracle_bias": oracle_metrics["bias"],
        "O_mse": output_metrics["mse"],
        "O_rmse": output_metrics["rmse"],
        "O_mae": output_metrics["mae"],
        "O_bias": output_metrics["bias"],
        "shared_training_mse": shared_fit["training_mse"],
        "historical_training_mse": shared_fit["historical_training_mse"],
        "individual_training_mse": shared_fit["individual_training_mse"],
        "oracle_fit_mse": float(oracle["loss"]),
        "O_fit_mse": float(output_fit["mse"]),
        "state_iterations": int(fit["state_iterations"]),
    }

    hourly = pd.DataFrame({
        "day": start,
        "hour": np.arange(DAY_HOURS, dtype=int),
        "I_observed": observed,
        "I_shared": shared_I,
        "I_historical": historical_I,
        "I_oracle": oracle_I,
        "O": baseline,
        "U": np.asarray(shared_state["U"], dtype=float)[hourly_index],
        "F_shared": np.asarray(shared_state["F"], dtype=float)[hourly_index],
        "R_shared": np.asarray(shared_state["R"], dtype=float)[hourly_index],
        "F_historical": np.asarray(historical_state["F"], dtype=float)[hourly_index],
        "R_historical": np.asarray(historical_state["R"], dtype=float)[hourly_index],
        "F_oracle": np.asarray(oracle_state["F"], dtype=float)[hourly_index],
        "R_oracle": np.asarray(oracle_state["R"], dtype=float)[hourly_index],
    }, index=actual.index)
    hourly.index.name = "time"

    training = pd.DataFrame({
        "prediction_day": start,
        "training_day": fit["training_days"],
        "optimized_F0": np.asarray(fit["init_cond"], dtype=float)[:, 0],
        "optimized_R0": np.asarray(fit["init_cond"], dtype=float)[:, 1],
        "individual_training_mse": np.asarray(fit["per_trajectory_loss"], dtype=float),
        "shared_training_mse": shared_fit["per_day_training_mse"],
        "historical_training_mse": shared_fit["historical_per_day_training_mse"],
        "shared_F0": shared_init[0],
        "shared_R0": shared_init[1],
        "historical_F0": historical_init[0],
        "historical_R0": historical_init[1],
    })
    return daily, hourly, training


def horizon_summary(hourly):
    rows = []
    for hour, part in hourly.groupby("hour", sort=True):
        observed = part["I_observed"].to_numpy(dtype=float)
        row = {"hour": int(hour), "days": int(len(part))}
        for name, column in (
            ("shared", "I_shared"),
            ("historical", "I_historical"),
            ("oracle", "I_oracle"),
        ):
            metrics = deviation_metrics(observed, part[column].to_numpy(dtype=float))
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def weekday_summary(daily):
    rows = []
    for weekday, part in daily.groupby("weekday", sort=False):
        rows.append({
            "weekday": weekday,
            "days": int(len(part)),
            "shared_mse": float(part["shared_mse"].mean()),
            "historical_mse": float(part["historical_mse"].mean()),
            "oracle_mse": float(part["oracle_mse"].mean()),
            "shared_state_error_norm": float(part["shared_state_error_norm"].mean()),
            "historical_state_error_norm": float(part["historical_state_error_norm"].mean()),
            "shared_better_pct": 100.0 * float(np.mean(part["shared_mse"] < part["historical_mse"])),
        })
    return pd.DataFrame(rows)


def save_summary_plot(daily, path):
    day = pd.to_datetime(daily["day"])
    fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True)

    axes[0].plot(day, daily["shared_mse"], label="Shared-state MSE")
    axes[0].plot(day, daily["historical_mse"], label="Historical-direct MSE")
    axes[0].plot(day, daily["oracle_mse"], label="Oracle MSE")
    axes[0].set_ylabel("Daily MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(day, daily["shared_F0"], label="Shared F0")
    axes[1].plot(day, daily["historical_F0"], label="Historical-direct F0")
    axes[1].plot(day, daily["oracle_F0"], label="Oracle F0")
    axes[1].set_ylabel("F0")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(day, daily["shared_R0"], label="Shared R0")
    axes[2].plot(day, daily["historical_R0"], label="Historical-direct R0")
    axes[2].plot(day, daily["oracle_R0"], label="Oracle R0")
    axes[2].set_ylabel("R0")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(day, daily["shared_state_error_norm"], label="Shared distance to oracle")
    axes[3].plot(day, daily["historical_state_error_norm"], label="Historical-direct distance to oracle")
    axes[3].set_ylabel("State distance")
    axes[3].set_xlabel("Prediction day")
    axes[3].legend()
    axes[3].grid(True, alpha=0.25)

    fig.suptitle("WM4 shared historical initial state vs historical-direct state and oracle")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_training_plot(daily, path):
    day = pd.to_datetime(daily["day"])
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

    axes[0].plot(day, daily["shared_training_mse"], label="One shared state")
    axes[0].plot(day, daily["historical_training_mse"], label="Historical-direct state")
    axes[0].plot(day, daily["individual_training_mse"], label="Individual optimized states")
    axes[0].set_ylabel("Historical training MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    gain = daily["historical_mse"] - daily["shared_mse"]
    axes[1].plot(day, gain, label="Prediction MSE gain: historical - shared")
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_ylabel("MSE gain")
    axes[1].set_xlabel("Prediction day")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("WM4 shared-state historical fit and out-of-sample prediction gain")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(daily, hourly, skipped):
    observed = hourly["I_observed"].to_numpy(dtype=float)
    print(f"days fitted={len(daily)}, skipped={len(skipped)}, hourly observations={len(hourly)}")
    print()
    print(f"{'method':<20}{'MSE':>14}{'RMSE':>14}{'MAE':>14}{'bias':>14}")
    for name, column in (
        ("shared state", "I_shared"),
        ("historical direct", "I_historical"),
        ("oracle", "I_oracle"),
        ("O(t)", "O"),
    ):
        metrics = deviation_metrics(observed, hourly[column].to_numpy(dtype=float))
        print(f"{name:<20}{metrics['mse']:>14.10f}{metrics['rmse']:>14.10f}{metrics['mae']:>14.10f}{metrics['bias']:>14.10f}")

    print()
    print(f"shared state beats historical direct on {100.0 * np.mean(daily['shared_mse'] < daily['historical_mse']):.2f}% of fitted days")
    print(f"mean daily MSE: shared={daily['shared_mse'].mean():.10f}, historical={daily['historical_mse'].mean():.10f}, oracle={daily['oracle_mse'].mean():.10f}")
    print(f"mean state distance to oracle: shared={daily['shared_state_error_norm'].mean():.8f}, historical={daily['historical_state_error_norm'].mean():.8f}")
    print(f"mean |F0-oracle|: shared={np.mean(np.abs(daily['shared_F0_error'])):.8f}, historical={np.mean(np.abs(daily['historical_F0_error'])):.8f}")
    print(f"mean |R0-oracle|: shared={np.mean(np.abs(daily['shared_R0_error'])):.8f}, historical={np.mean(np.abs(daily['historical_R0_error'])):.8f}")
    print(f"mean historical fit MSE: shared={daily['shared_training_mse'].mean():.10f}, historical={daily['historical_training_mse'].mean():.10f}, individual-oracle-per-day={daily['individual_training_mse'].mean():.10f}")

    if skipped:
        print()
        print("skipped days:")
        for day, reason in skipped: print(f"{day:%Y-%m-%d}: {reason}")


def run_all_days(frame, start=START_DAY, end=END_DAY, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, fail_fast=False, **fit_kwargs):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")
    daily_rows, hourly_frames, training_frames, skipped = [], [], [], []

    for day in pd.date_range(start, end, freq="D"):
        try:
            daily, hourly, training = day_result(
                frame, day, baseline_weeks, steps_per_hour, **fit_kwargs,
            )
            daily_rows.append(daily)
            hourly_frames.append(hourly)
            training_frames.append(training)
            print(
                f"{day:%Y-%m-%d} | shared=[{daily['shared_F0']:.6f}, {daily['shared_R0']:.6f}] "
                f"| hist=[{daily['historical_F0']:.6f}, {daily['historical_R0']:.6f}] "
                f"| oracle=[{daily['oracle_F0']:.6f}, {daily['oracle_R0']:.6f}] "
                f"| MSE shared={daily['shared_mse']:.8f} hist={daily['historical_mse']:.8f} oracle={daily['oracle_mse']:.8f}"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc)))
            print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not daily_rows: raise RuntimeError("No days in the requested range could be fitted.")
    return (
        pd.DataFrame(daily_rows).sort_values("day").reset_index(drop=True),
        pd.concat(hourly_frames).sort_index(),
        pd.concat(training_frames, ignore_index=True),
        skipped,
    )


def main():
    parser = ArgumentParser(description="Fit one shared WM4 [F0,R0] over the corresponding historical weekdays after fitting r, then compare its all-days prediction performance with the direct historical state estimate and target-day oracle.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    daily, hourly, training, skipped = run_all_days(
        frame, args.start_day, args.end_day, args.baseline_weeks, args.steps_per_hour,
        fail_fast=args.fail_fast,
        max_state_iter=args.max_state_iter,
        state_tol=args.state_tol,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_comparison.csv", index=False)
    hourly.to_csv(output / "hourly_predictions.csv")
    training.to_csv(output / "historical_training_comparison.csv", index=False)
    horizon_summary(hourly).to_csv(output / "horizon_summary.csv", index=False)
    weekday_summary(daily).to_csv(output / "weekday_summary.csv", index=False)
    if skipped:
        pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)

    save_summary_plot(daily, output / "shared_initial_state_comparison.png")
    save_training_plot(daily, output / "shared_state_training_and_gain.png")
    print()
    print_summary(daily, hourly, skipped)
    print(f"output={output}")


if __name__ == "__main__": main()
