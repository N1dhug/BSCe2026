from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_3.fit_weekly import aligned_data
from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, day_start, fit_expected_output, fitted_initial_condition, model_input, model_output
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup, save_plot
from src.model.wm_4 import wm_4
from src.params.wm_4 import optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_optimized"
START_DAY = "2022-02-28"
END_DAY = "2022-08-22"


def fit_history_optimized(frame, predict_start, parameter_weeks, steps_per_hour, fit_diffusion=True, **kwargs):
    predict_start, parameter_weeks = day_start(predict_start), int(parameter_weeks)
    if parameter_weeks < 1: raise ValueError("parameter_weeks must be at least 1.")
    output_model, output_fit = fit_expected_output(frame, predict_start)
    fit_days = [predict_start - pd.Timedelta(weeks=k) for k in range(1, parameter_weeks + 1)]
    datasets, indices, seeds = [], [], []
    model_T = None

    for day in fit_days:
        data, fine_index, _, day_T = model_input(frame, day, steps_per_hour, 0, output_model)
        if model_T is None: model_T = day_T
        elif tuple(day_T) != tuple(model_T): raise ValueError("Historical wm_4 fitting days must use the same model horizon.")
        datasets.append(data); indices.append(fine_index); seeds.append(fitted_initial_condition(data))

    model = wm_4(model_T, len(indices[0]), model_output(datasets[0], indices[0]))
    fit = optimized_qp_params(model).fit_many(
        datasets, np.vstack(seeds), fit_initial_state=True,
        fit_diffusion=fit_diffusion, verbose=False, **kwargs,
    )
    fit["training_days"] = fit_days
    fit["fit_trajectories"] = len(datasets)
    return output_model, output_fit, fit, model


def historical_optimized_state(fit):
    states = np.asarray(fit["init_cond"], dtype=float)
    if states.ndim != 2 or states.shape[1] != 2: raise ValueError("Optimized historical fit must contain one [F0,R0] row per training day.")
    init = np.median(states, axis=0)
    init[0] = np.clip(init[0], 0.0, 1.0)
    return init


def day_result(frame, start, baseline_weeks, steps_per_hour, deterministic=False, save_daily_plots=False, **fit_kwargs):
    output_model, output_fit, fit, _ = fit_history_optimized(
        frame, start, baseline_weeks, steps_per_hour,
        fit_diffusion=not deterministic, **fit_kwargs,
    )
    end, actual, U, Time, model, fine_index = prediction_setup(frame, start, output_model, steps_per_hour)
    init = historical_optimized_state(fit)
    state = model.deterministic_solver(init, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})

    hourly_index = np.arange(DAY_HOURS) * steps_per_hour
    observed = actual["I"].to_numpy(dtype=float)
    predicted = state["I"][hourly_index]
    baseline = state["O"][hourly_index]
    wm4_metrics, output_metrics = deviation_metrics(observed, predicted), deviation_metrics(observed, baseline)
    r = np.asarray(fit["r"], dtype=float)
    training_states = np.asarray(fit["init_cond"], dtype=float)

    daily = {
        "day": start, "weekday": start.day_name(), "fit_method": fit["fit_method"],
        "r_0": r[0], "r_F": r[1], "r_U": r[2], "r_R": r[3],
        "F_0": init[0], "R_0": init[1],
        "training_F0_mean": float(np.mean(training_states[:, 0])),
        "training_F0_std": float(np.std(training_states[:, 0])),
        "training_F0_min": float(np.min(training_states[:, 0])),
        "training_F0_max": float(np.max(training_states[:, 0])),
        "training_R0_mean": float(np.mean(training_states[:, 1])),
        "training_R0_std": float(np.std(training_states[:, 1])),
        "state_iterations": int(fit["state_iterations"]),
        "input_fit_loss": float(fit["input_fit_loss"]),
        "sigma_0": np.nan if deterministic else fit["sigma_0"],
        "sigma_I": np.nan if deterministic else fit["sigma_I"],
        "O_fit_mse": output_fit["mse"], "O_fit_samples": output_fit["samples"],
        "wm4_mse": wm4_metrics["mse"], "wm4_rmse": wm4_metrics["rmse"],
        "wm4_mae": wm4_metrics["mae"], "wm4_bias": wm4_metrics["bias"],
        "O_mse": output_metrics["mse"], "O_rmse": output_metrics["rmse"],
        "O_mae": output_metrics["mae"], "O_bias": output_metrics["bias"],
    }
    hourly = pd.DataFrame({
        "day": start, "I_observed": observed, "I_wm4": predicted, "O": baseline,
        "U": state["U"][hourly_index], "F": state["F"][hourly_index], "R": state["R"][hourly_index],
    }, index=actual.index)
    hourly.index.name = "time"

    training = pd.DataFrame({
        "prediction_day": start,
        "training_day": fit["training_days"],
        "F_0": training_states[:, 0],
        "R_0": training_states[:, 1],
    })

    if save_daily_plots:
        path = OUTPUT / "daily" / f"daily_prediction_{start:%Y-%m-%d}.png"
        save_plot(start, end, actual, fine_index, state, path, baseline_weeks)

    return daily, hourly, training


def save_performance_plot(hourly, daily, path):
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=False)
    axes[0].plot(hourly.index, hourly["I_observed"], label="Observed I")
    axes[0].plot(hourly.index, hourly["I_wm4"], label="WM4 optimized-input")
    axes[0].plot(hourly.index, hourly["O"], label="Weather-aware O", alpha=0.8)
    axes[0].set_ylabel("Normalized input"); axes[0].legend(); axes[0].grid(True, alpha=0.25)

    axes[1].plot(daily["day"], daily["wm4_mse"], label="WM4 daily MSE")
    axes[1].plot(daily["day"], daily["O_mse"], label="O daily MSE")
    axes[1].set_ylabel("Daily MSE"); axes[1].legend(); axes[1].grid(True, alpha=0.25)

    axes[2].plot(daily["day"], daily["training_F0_mean"], label="Historical optimized F0 mean")
    axes[2].fill_between(daily["day"], daily["training_F0_mean"] - daily["training_F0_std"], daily["training_F0_mean"] + daily["training_F0_std"], alpha=0.2)
    axes[2].plot(daily["day"], daily["F_0"], label="Prediction F0 = historical median")
    axes[2].set_ylabel("F0"); axes[2].set_xlabel("Day"); axes[2].legend(); axes[2].grid(True, alpha=0.25)

    fig.suptitle("wm_4 optimized-state input-fit out-of-sample performance")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def print_summary(hourly, daily, skipped):
    wm4 = deviation_metrics(hourly["I_observed"], hourly["I_wm4"])
    output = deviation_metrics(hourly["I_observed"], hourly["O"])
    print(f"days fitted={len(daily)}, skipped={len(skipped)}, hourly observations={len(hourly)}")
    print()
    print(f"{'':<8}{'MSE':>14}{'RMSE':>14}{'MAE':>14}{'bias':>14}")
    print(f"{'WM4':<8}{wm4['mse']:>14.10f}{wm4['rmse']:>14.10f}{wm4['mae']:>14.10f}{wm4['bias']:>14.10f}")
    print(f"{'O(t)':<8}{output['mse']:>14.10f}{output['rmse']:>14.10f}{output['mae']:>14.10f}{output['bias']:>14.10f}")
    print(f"mean optimized historical F0 std={daily['training_F0_std'].mean():.8f}")
    print(f"mean alternating state iterations={daily['state_iterations'].mean():.2f}")
    if skipped:
        print(); print("skipped days:")
        for day, reason in skipped: print(f"{day:%Y-%m-%d}: {reason}")


def run_all_days(frame, start=START_DAY, end=END_DAY, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, deterministic=False, save_daily_plots=False, fail_fast=False, **fit_kwargs):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")
    daily_rows, hourly_frames, training_frames, skipped = [], [], [], []

    for day in pd.date_range(start, end, freq="D"):
        try:
            daily, hourly, training = day_result(
                frame, day, baseline_weeks, steps_per_hour,
                deterministic=deterministic, save_daily_plots=save_daily_plots, **fit_kwargs,
            )
            daily_rows.append(daily); hourly_frames.append(hourly); training_frames.append(training)
            print(f"{day:%Y-%m-%d} | r=[{daily['r_0']:.6f}, {daily['r_F']:.6f}, {daily['r_U']:.6f}, {daily['r_R']:.6f}] | initial_cond=[{daily['F_0']:.6f}, {daily['R_0']:.6f}] | state_iters={daily['state_iterations']} | WM4 MSE={daily['wm4_mse']:.8f} | O MSE={daily['O_mse']:.8f}")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc))); print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not daily_rows: raise RuntimeError("No days in the requested range could be fitted.")
    return (
        pd.DataFrame(daily_rows).sort_values("day").reset_index(drop=True),
        pd.concat(hourly_frames).sort_index(),
        pd.concat(training_frames, ignore_index=True),
        skipped,
    )


def main():
    parser = ArgumentParser(description="Evaluate wm_4 predictions using parameters and historical initial states fitted directly to observed input I.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--deterministic", action="store_true", help="Skip diffusion fitting.")
    parser.add_argument("--save-daily-plots", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    args = parser.parse_args()

    frame, _ = aligned_data("MED")
    daily, hourly, training, skipped = run_all_days(
        frame, args.start_day, args.end_day, args.baseline_weeks, args.steps_per_hour,
        deterministic=args.deterministic, save_daily_plots=args.save_daily_plots, fail_fast=args.fail_fast,
        max_state_iter=args.max_state_iter, state_tol=args.state_tol,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT / "daily_metrics.csv", index=False)
    hourly.to_csv(OUTPUT / "hourly_predictions.csv")
    training.to_csv(OUTPUT / "optimized_training_states.csv", index=False)
    if skipped: pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(OUTPUT / "skipped_days.csv", index=False)
    save_performance_plot(hourly, daily, OUTPUT / "all_days_performance.png")
    print(); print_summary(hourly, daily, skipped)


if __name__ == "__main__": main()
