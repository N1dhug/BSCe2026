from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_3.fit_weekly import aligned_data
from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, day_start
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, fit_history, historical_prediction_state, optimized_prediction_state, prediction_setup, save_plot


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days"
START_DAY = "2022-02-28"
END_DAY = "2022-08-22"


def day_result(frame, start, baseline_weeks, warmup_hours, steps_per_hour, deterministic=False, optimize_prediction_state=False, save_daily_plots=False, **fit_kwargs):
    output_model, output_fit, fit, _, _, _ = fit_history(
        frame, start, baseline_weeks, warmup_hours, steps_per_hour,
        fit_diffusion=not deterministic, **fit_kwargs,
    )
    end, actual, U, Time, model, fine_index = prediction_setup(frame, start, output_model, steps_per_hour)

    if optimize_prediction_state:
        init, state, fine_index, _ = optimized_prediction_state(frame, start, model, fit["r"], steps_per_hour, output_model)
    else:
        init = historical_prediction_state(fit, warmup_hours, steps_per_hour)
        state = model.deterministic_solver(init, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})

    hourly_index = np.arange(DAY_HOURS) * steps_per_hour
    observed = actual["I"].to_numpy(dtype=float)
    predicted = state["I"][hourly_index]
    baseline = state["O"][hourly_index]
    wm4_metrics, output_metrics = deviation_metrics(observed, predicted), deviation_metrics(observed, baseline)
    r = np.asarray(fit["r"], dtype=float)

    daily = {
        "day": start, "weekday": start.day_name(), "optimized": bool(optimize_prediction_state),
        "r_0": r[0], "r_F": r[1], "r_U": r[2], "r_R": r[3],
        "F_0": init[0], "R_0": init[1],
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

    if save_daily_plots:
        path = OUTPUT / "daily" / f"daily_prediction_{start:%Y-%m-%d}.png"
        save_plot(start, end, actual, fine_index, state, path, baseline_weeks)

    return daily, hourly


def save_performance_plot(hourly, daily, path):
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=False)
    axes[0].plot(hourly.index, hourly["I_observed"], label="Observed I")
    axes[0].plot(hourly.index, hourly["I_wm4"], label="WM4")
    axes[0].plot(hourly.index, hourly["O"], label="Weather-aware O", alpha=0.8)
    axes[0].set_ylabel("Normalized input")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(daily["day"], daily["wm4_mse"], label="WM4 daily MSE")
    axes[1].plot(daily["day"], daily["O_mse"], label="O daily MSE")
    axes[1].set_ylabel("Daily MSE")
    axes[1].set_xlabel("Day")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("wm_4 out-of-sample daily performance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(hourly, fitted_days, skipped):
    wm4 = deviation_metrics(hourly["I_observed"], hourly["I_wm4"])
    output = deviation_metrics(hourly["I_observed"], hourly["O"])
    print(f"days fitted={fitted_days}, skipped={len(skipped)}, hourly observations={len(hourly)}")
    print()
    print(f"{'':<8}{'MSE':>14}{'RMSE':>14}{'MAE':>14}{'bias':>14}")
    print(f"{'WM4':<8}{wm4['mse']:>14.10f}{wm4['rmse']:>14.10f}{wm4['mae']:>14.10f}{wm4['bias']:>14.10f}")
    print(f"{'O(t)':<8}{output['mse']:>14.10f}{output['rmse']:>14.10f}{output['mae']:>14.10f}{output['bias']:>14.10f}")
    if skipped:
        print()
        print("skipped days:")
        for day, reason in skipped: print(f"{day:%Y-%m-%d}: {reason}")


def run_all_days(frame, start=START_DAY, end=END_DAY, baseline_weeks=BASELINE_WEEKS, warmup_hours=24, steps_per_hour=60, deterministic=False, optimize_prediction_state=False, save_daily_plots=False, fail_fast=False, **fit_kwargs):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")

    daily_rows, hourly_frames, skipped = [], [], []
    for day in pd.date_range(start, end, freq="D"):
        try:
            daily, hourly = day_result(
                frame, day, baseline_weeks, warmup_hours, steps_per_hour,
                deterministic=deterministic, optimize_prediction_state=optimize_prediction_state,
                save_daily_plots=save_daily_plots, **fit_kwargs,
            )
            daily_rows.append(daily); hourly_frames.append(hourly)
            print(f"{day:%Y-%m-%d} | r=[{daily['r_0']:.6f}, {daily['r_F']:.6f}, {daily['r_U']:.6f}, {daily['r_R']:.6f}] | initial_cond=[{daily['F_0']:.6f}, {daily['R_0']:.6f}] | WM4 MSE={daily['wm4_mse']:.8f} | O MSE={daily['O_mse']:.8f}")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc)))
            print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not daily_rows: raise RuntimeError("No days in the requested range could be fitted.")
    return pd.DataFrame(daily_rows).sort_values("day").reset_index(drop=True), pd.concat(hourly_frames).sort_index(), skipped


def main():
    parser = ArgumentParser(description="Evaluate weather-aware, jointly fitted wm_4 predictions across every usable day in a date range.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS, help="Previous matching weekdays fitted jointly for WM4 parameters.")
    parser.add_argument("--warmup-hours", type=int, default=24)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--deterministic", action="store_true", help="Skip diffusion fitting.")
    parser.add_argument("--optimize-prediction-state", action="store_true", help="Diagnostic only: fit each prediction-day initial state from its observed I.")
    parser.add_argument("--save-daily-plots", action="store_true", help="Also save one prediction plot per fitted day.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately instead of skipping a day that cannot be fitted.")
    args = parser.parse_args()

    frame, _ = aligned_data("MED")
    daily, hourly, skipped = run_all_days(
        frame, args.start_day, args.end_day, args.baseline_weeks, args.warmup_hours, args.steps_per_hour,
        deterministic=args.deterministic, optimize_prediction_state=args.optimize_prediction_state,
        save_daily_plots=args.save_daily_plots, fail_fast=args.fail_fast,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT / "daily_metrics.csv", index=False)
    hourly.to_csv(OUTPUT / "hourly_predictions.csv")
    if skipped: pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(OUTPUT / "skipped_days.csv", index=False)
    save_performance_plot(hourly, daily, OUTPUT / "all_days_performance.png")
    print()
    print_summary(hourly, len(daily), skipped)


if __name__ == "__main__": main()
