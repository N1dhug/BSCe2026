from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import DAY_HOURS, aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_all_days_adaptive_traceback import DEFAULT_TRACEBACK_WEEKS, normalized_tracebacks
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY, fit_history_optimized, historical_optimized_state
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup
from src.params.wm_4 import optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "fixed_tracebacks"
STATE_MODES = ("historical_median", "oracle")


def evaluate_state(day, weeks, mode, init, state, observed, hourly_index, fit, output_fit):
    predicted = np.asarray(state["I"], dtype=float)[hourly_index]
    baseline = np.asarray(state["O"], dtype=float)[hourly_index]
    wm4 = deviation_metrics(observed, predicted)
    output = deviation_metrics(observed, baseline)
    r = np.asarray(fit["r"], dtype=float)

    daily = {
        "day": day, "weekday": day.day_name(), "traceback_weeks": int(weeks), "prediction_state_mode": mode,
        "historical_mse": float(fit["deterministic_loss"]), "qp_loss": float(fit["qp_loss"]),
        "state_iterations": int(fit["state_iterations"]),
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "F_0": float(init[0]), "R_0": float(init[1]),
        "wm4_mse": wm4["mse"], "wm4_rmse": wm4["rmse"], "wm4_mae": wm4["mae"], "wm4_bias": wm4["bias"],
        "O_mse": output["mse"], "O_rmse": output["rmse"], "O_mae": output["mae"], "O_bias": output["bias"],
        "O_fit_mse": float(output_fit["mse"]),
    }
    hourly = pd.DataFrame({
        "day": day, "traceback_weeks": int(weeks), "prediction_state_mode": mode,
        "I_observed": observed, "I_wm4": predicted, "O": baseline,
        "U": np.asarray(state["U"], dtype=float)[hourly_index],
        "F": np.asarray(state["F"], dtype=float)[hourly_index],
        "R": np.asarray(state["R"], dtype=float)[hourly_index],
    })
    return daily, hourly


def fit_candidate(frame, day, weeks, steps_per_hour, max_state_iter=50, state_tol=1e-8):
    output_model, output_fit, fit, _ = fit_history_optimized(
        frame, day, weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )
    _, actual, U, Time, model, _ = prediction_setup(frame, day, output_model, steps_per_hour)
    prediction_data, _, _, _ = model_input(frame, day, steps_per_hour, 0, output_model)
    hourly_index = np.arange(DAY_HOURS, dtype=int) * int(steps_per_hour)
    observed = actual["I"].to_numpy(dtype=float)

    estimated_init = historical_optimized_state(fit)
    estimated_state = model.deterministic_solver(estimated_init, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})

    oracle = optimized_qp_params(model).optimize_prediction_initial_state(prediction_data, fit["r"])
    oracle_init = np.asarray(oracle["init_cond"], dtype=float)
    oracle_state = oracle["deterministic_state"]

    daily_rows, hourly_frames = [], []
    for mode, init, state in (
        ("historical_median", estimated_init, estimated_state),
        ("oracle", oracle_init, oracle_state),
    ):
        daily, hourly = evaluate_state(day, weeks, mode, init, state, observed, hourly_index, fit, output_fit)
        daily["prediction_state_fit_mse"] = np.nan if mode == "historical_median" else float(oracle["loss"])
        hourly["time"] = actual.index
        daily_rows.append(daily); hourly_frames.append(hourly)

    training_states = np.asarray(fit["init_cond"], dtype=float)
    training = pd.DataFrame({
        "prediction_day": day, "traceback_weeks": int(weeks),
        "training_day": fit["training_days"],
        "F_0": training_states[:, 0], "R_0": training_states[:, 1],
    })
    return daily_rows, pd.concat(hourly_frames, ignore_index=True), training


def common_days(daily, tracebacks):
    required = len(tracebacks) * len(STATE_MODES)
    counts = daily.groupby("day").size()
    complete = counts[counts == required].index
    return pd.DatetimeIndex(complete)


def aggregate_summary(daily, hourly, tracebacks):
    days = common_days(daily, tracebacks)
    if len(days) < 1: raise RuntimeError("No common prediction days have every traceback and both state modes available.")

    daily["common_day"] = daily["day"].isin(days)
    hourly["common_day"] = hourly["day"].isin(days)
    common_hourly = hourly[hourly["common_day"]].copy()

    rows = []
    for mode in STATE_MODES:
        for weeks in tracebacks:
            group = common_hourly[
                (common_hourly["prediction_state_mode"] == mode)
                & (common_hourly["traceback_weeks"] == int(weeks))
            ]
            if len(group) != len(days) * DAY_HOURS:
                raise RuntimeError(f"Common-day aggregation expected {len(days) * DAY_HOURS} observations for {mode}, {weeks} weeks; found {len(group)}.")
            metrics = deviation_metrics(group["I_observed"], group["I_wm4"])
            daily_group = daily[
                daily["common_day"]
                & (daily["prediction_state_mode"] == mode)
                & (daily["traceback_weeks"] == int(weeks))
            ]
            rows.append({
                "prediction_state_mode": mode, "traceback_weeks": int(weeks),
                "days": int(len(days)), "hourly_observations": int(len(group)),
                "mse": metrics["mse"], "rmse": metrics["rmse"], "mae": metrics["mae"], "bias": metrics["bias"],
                "mean_daily_mse": float(daily_group["wm4_mse"].mean()),
                "median_daily_mse": float(daily_group["wm4_mse"].median()),
                "mean_historical_mse": float(daily_group["historical_mse"].mean()),
            })

    summary = pd.DataFrame(rows)
    for mode in STATE_MODES:
        mask = summary["prediction_state_mode"] == mode
        best = summary.loc[mask, "mse"].idxmin()
        summary.loc[mask, "best_mse_for_mode"] = False
        summary.loc[best, "best_mse_for_mode"] = True
    return summary, days


def run_all_days(frame, start=START_DAY, end=END_DAY, tracebacks=DEFAULT_TRACEBACK_WEEKS, steps_per_hour=60, fail_fast=False, max_state_iter=50, state_tol=1e-8):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")
    tracebacks = normalized_tracebacks(tracebacks)

    daily_rows, hourly_frames, training_frames, failures = [], [], [], []
    for day in pd.date_range(start, end, freq="D"):
        for weeks in tracebacks:
            try:
                daily, hourly, training = fit_candidate(
                    frame, day, weeks, steps_per_hour,
                    max_state_iter=max_state_iter, state_tol=state_tol,
                )
                daily_rows.extend(daily); hourly_frames.append(hourly); training_frames.append(training)
                estimated = next(row for row in daily if row["prediction_state_mode"] == "historical_median")
                oracle = next(row for row in daily if row["prediction_state_mode"] == "oracle")
                print(
                    f"{day:%Y-%m-%d} | {weeks} weeks | "
                    f"estimated MSE={estimated['wm4_mse']:.8f} | oracle MSE={oracle['wm4_mse']:.8f}"
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                if fail_fast: raise
                failures.append({"day": day, "traceback_weeks": int(weeks), "reason": str(exc)})
                print(f"{day:%Y-%m-%d} | {weeks} weeks | failed | {exc}")

    if not daily_rows: raise RuntimeError("No traceback fits succeeded.")
    daily = pd.DataFrame(daily_rows).sort_values(["day", "traceback_weeks", "prediction_state_mode"]).reset_index(drop=True)
    hourly = pd.concat(hourly_frames, ignore_index=True).sort_values(["time", "traceback_weeks", "prediction_state_mode"]).reset_index(drop=True)
    training = pd.concat(training_frames, ignore_index=True)
    failures = pd.DataFrame(failures)
    summary, days = aggregate_summary(daily, hourly, tracebacks)
    return daily, hourly, training, summary, days, failures


def save_plot(summary, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for mode in STATE_MODES:
        values = summary[summary["prediction_state_mode"] == mode].sort_values("traceback_weeks")
        label = "Initial-state estimate" if mode == "historical_median" else "Oracle state"
        axes[0].plot(values["traceback_weeks"], values["mse"], marker="o", label=label)
        axes[1].plot(values["traceback_weeks"], values["rmse"], marker="o", label=label)
        axes[2].plot(values["traceback_weeks"], values["mae"], marker="o", label=label)

    axes[0].set_ylabel("MSE"); axes[1].set_ylabel("RMSE"); axes[2].set_ylabel("MAE")
    axes[2].set_xlabel("Corresponding historical weekdays used for parameter fit")
    for ax in axes:
        ax.grid(True, alpha=0.25); ax.legend()
    fig.suptitle("ARM (Affine Revised Model) fixed traceback comparison on identical prediction days")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def print_summary(summary, common, failures):
    print()
    print(f"common prediction days={len(common)}, candidate failures={len(failures)}, hourly observations per week/mode={len(common) * DAY_HOURS}")
    print("All metrics below use the identical common set of prediction days.")
    for mode in STATE_MODES:
        label = "initial-state estimate" if mode == "historical_median" else "oracle state"
        print(); print(label)
        values = summary[summary["prediction_state_mode"] == mode].sort_values("traceback_weeks")
        for _, row in values.iterrows():
            marker = "  <-- best MSE" if bool(row["best_mse_for_mode"]) else ""
            print(
                f"  {int(row['traceback_weeks'])} weeks | "
                f"MSE={row['mse']:.10f} | RMSE={row['rmse']:.10f} | "
                f"MAE={row['mae']:.10f} | bias={row['bias']:.10f}{marker}"
            )


def main():
    parser = ArgumentParser(description="Compare fixed WM4 parameter traceback lengths on one common set of prediction days using both historical initial-state estimation and oracle initial states.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--traceback-weeks", type=int, nargs="+", default=DEFAULT_TRACEBACK_WEEKS, help="Fixed traceback lengths to compare. Default: 3 4 5 6 7 8.")
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    tracebacks = normalized_tracebacks(args.traceback_weeks)
    frame, _ = aligned_wm4_data("MED")
    daily, hourly, training, summary, common, failures = run_all_days(
        frame, args.start_day, args.end_day, tracebacks, args.steps_per_hour,
        fail_fast=args.fail_fast, max_state_iter=args.max_state_iter, state_tol=args.state_tol,
    )

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_fixed_tracebacks.csv", index=False)
    hourly.to_csv(output / "hourly_fixed_tracebacks.csv", index=False)
    training.to_csv(output / "optimized_training_states.csv", index=False)
    summary.to_csv(output / "common_day_summary.csv", index=False)
    pd.DataFrame({"day": common}).to_csv(output / "common_prediction_days.csv", index=False)
    if not failures.empty: failures.to_csv(output / "failed_candidates.csv", index=False)
    save_plot(summary, output / "fixed_traceback_comparison.png")
    print_summary(summary, common, failures)


if __name__ == "__main__": main()
