from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.brute_force_f0 import fit_history_brute_force
from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup, save_plot
from src.params.wm_4 import optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_brute_force"
TOP_END_OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_brute_force_top_end"
START_DAY = "2022-02-28"
END_DAY = "2022-08-22"


def historical_brute_force_state(fit):
    states = np.asarray(fit["init_cond"], dtype=float)
    if states.ndim != 2 or states.shape[1] != 2: raise ValueError("Brute-force historical fit must contain one [F0,R0] row per training day.")
    return np.median(states, axis=0).astype(float)


def brute_force_profiles(fit, prediction_day):
    training_days, profiles = list(fit["training_days"]), fit["f0_profile"]
    if len(profiles) != len(training_days): raise ValueError("Brute-force F0 profiles must contain one profile per training day.")
    frames = []
    for training_day, profile in zip(training_days, profiles):
        frame = pd.DataFrame(profile)
        frame.insert(0, "training_day", training_day)
        frame.insert(0, "prediction_day", prediction_day)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def day_result(frame, start, baseline_weeks, steps_per_hour, deterministic=False, optimize_initial_state=False, save_daily_plots=False, output_root=OUTPUT, **fit_kwargs):
    output_model, output_fit, fit, _ = fit_history_brute_force(
        frame, start, baseline_weeks, steps_per_hour,
        deterministic=deterministic, **fit_kwargs,
    )
    end, actual, U, Time, model, fine_index = prediction_setup(frame, start, output_model, steps_per_hour)
    if optimize_initial_state:
        prediction_data, fine_index, _, _ = model_input(frame, start, steps_per_hour, 0, output_model)
        oracle = optimized_qp_params(model).optimize_prediction_initial_state(prediction_data, fit["r"])
        init, state, init_fit_mse = oracle["init_cond"], oracle["deterministic_state"], oracle["loss"]
    else:
        init = historical_brute_force_state(fit)
        state = model.deterministic_solver(init, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})
        init_fit_mse = np.nan

    hourly_index = np.arange(DAY_HOURS) * steps_per_hour
    observed = actual["I"].to_numpy(dtype=float)
    predicted = state["I"][hourly_index]
    baseline = state["O"][hourly_index]
    wm4_metrics, output_metrics = deviation_metrics(observed, predicted), deviation_metrics(observed, baseline)
    r = np.asarray(fit["r"], dtype=float)
    training_states = np.asarray(fit["init_cond"], dtype=float)
    if training_states.ndim != 2 or training_states.shape[1] != 2: raise ValueError("Brute-force historical fit must contain one [F0,R0] row per training day.")
    f0 = training_states[:, 0]
    feasible = np.asarray(fit["brute_force_feasible_candidates"], dtype=float).reshape(-1)
    if feasible.size != len(training_states): raise ValueError("Brute-force feasible-candidate counts must contain one value per training day.")
    profile = brute_force_profiles(fit, start)

    daily = {
        "day": start, "weekday": start.day_name(), "fit_method": fit["fit_method"],
        "optimized_prediction_state": bool(optimize_initial_state), "prediction_state_fit_mse": init_fit_mse,
        "F_0": init[0], "R_0": init[1],
        "best_F0": float(np.median(f0)),
        "training_F0_mean": float(np.mean(f0)), "training_F0_std": float(np.std(f0)),
        "training_F0_median": float(np.median(f0)), "training_F0_min": float(np.min(f0)), "training_F0_max": float(np.max(f0)),
        "r_F": r[0], "r_U": r[1], "r_R": r[2], "r_0": r[3],
        "training_R0_mean": float(np.mean(training_states[:, 1])),
        "training_R0_std": float(np.std(training_states[:, 1])),
        "training_R0_median": float(np.median(training_states[:, 1])),
        "qp_loss": float(fit["qp_loss"]),
        "deterministic_fit_loss": float(fit["deterministic_loss"]),
        "brute_force_candidates": int(fit["brute_force_candidates"]),
        "brute_force_feasible_candidates": float(np.mean(feasible)),
        "brute_force_feasible_candidates_min": int(np.min(feasible)),
        "brute_force_feasible_candidates_max": int(np.max(feasible)),
        "state_iterations": int(fit["state_iterations"]),
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
        "feasible_F0_candidates": feasible.astype(int),
    })

    if save_daily_plots:
        path = Path(output_root) / "daily" / f"daily_prediction_{start:%Y-%m-%d}.png"
        save_plot(start, end, actual, fine_index, state, path, baseline_weeks)

    return daily, hourly, training, profile


def save_performance_plot(hourly, daily, path):
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=False)
    axes[0].plot(hourly.index, hourly["I_observed"], label="Observed I")
    axes[0].plot(hourly.index, hourly["I_wm4"], label="WM4 brute-force F0")
    axes[0].plot(hourly.index, hourly["O"], label="Weather-aware O", alpha=0.8)
    axes[0].set_ylabel("Normalized input"); axes[0].legend(); axes[0].grid(True, alpha=0.25)

    axes[1].plot(daily["day"], daily["wm4_mse"], label="WM4 daily MSE")
    axes[1].plot(daily["day"], daily["O_mse"], label="O daily MSE")
    axes[1].set_ylabel("Daily MSE"); axes[1].legend(); axes[1].grid(True, alpha=0.25)

    axes[2].plot(daily["day"], daily["training_F0_median"], label="Historical training F0 median")
    axes[2].plot(daily["day"], daily["F_0"], label="Prediction-day F0")
    axes[2].axhline(0.0, linestyle="--", label="Neutral F0=0")
    axes[2].set_ylabel("F0"); axes[2].set_xlabel("Day"); axes[2].legend(); axes[2].grid(True, alpha=0.25)

    fig.suptitle("wm_4 brute-force per-training-day F0 performance")
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
    print(f"historical training F0 median mean={daily['training_F0_median'].mean():.8f}, std={daily['training_F0_median'].std():.8f}")
    print(f"historical training F0 mean mean={daily['training_F0_mean'].mean():.8f}")
    if daily["optimized_prediction_state"].any():
        print(f"oracle prediction F0 mean={daily['F_0'].mean():.8f}, std={daily['F_0'].std():.8f}")
        print(f"oracle prediction-state MSE mean={daily['prediction_state_fit_mse'].mean():.10f}")
    print(f"mean feasible candidates per training day={daily['brute_force_feasible_candidates'].mean():.2f}/{daily['brute_force_candidates'].mean():.2f}")
    if skipped:
        print(); print("skipped days:")
        for day, reason in skipped: print(f"{day:%Y-%m-%d}: {reason}")


def run_all_days(frame, start=START_DAY, end=END_DAY, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, deterministic=False, optimize_initial_state=False, save_daily_plots=False, fail_fast=False, output_root=OUTPUT, **fit_kwargs):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")
    daily_rows, hourly_frames, training_frames, profile_frames, skipped = [], [], [], [], []

    for day in pd.date_range(start, end, freq="D"):
        try:
            daily, hourly, training, profile = day_result(
                frame, day, baseline_weeks, steps_per_hour,
                deterministic=deterministic, optimize_initial_state=optimize_initial_state,
                save_daily_plots=save_daily_plots,
                output_root=output_root, **fit_kwargs,
            )
            daily_rows.append(daily); hourly_frames.append(hourly)
            training_frames.append(training); profile_frames.append(profile)
            print(
                f"{day:%Y-%m-%d} | training_F0_median={daily['training_F0_median']:.6f} | "
                f"r=[{daily['r_F']:.6f}, {daily['r_U']:.6f}, {daily['r_R']:.6f}, {daily['r_0']:.6f}] | "
                f"initial_cond=[{daily['F_0']:.6f}, {daily['R_0']:.6f}] | QP={daily['qp_loss']:.8g} | "
                f"WM4 MSE={daily['wm4_mse']:.8f} | O MSE={daily['O_mse']:.8f}"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc))); print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not daily_rows: raise RuntimeError("No days in the requested range could be fitted.")
    return (
        pd.DataFrame(daily_rows).sort_values("day").reset_index(drop=True),
        pd.concat(hourly_frames).sort_index(),
        pd.concat(training_frames, ignore_index=True),
        pd.concat(profile_frames, ignore_index=True),
        skipped,
    )


def main():
    parser = ArgumentParser(description="Evaluate WM4 predictions across a date range using per-training-day brute-force F0 fits.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--f0-min", type=float, default=-1.0)
    parser.add_argument("--f0-max", type=float, default=1.0)
    parser.add_argument("--f0-steps", type=int, default=81)
    parser.add_argument("--max-state-iter", type=int, default=25)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--deterministic", action="store_true", help="Skip diffusion fitting.")
    parser.add_argument("--optimize-initial-state", action="store_true", help="Oracle diagnostic: fit prediction-day [F0,R0] from that day's observed I for top-end fitter comparison.")
    parser.add_argument("--save-daily-plots", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    output_root = TOP_END_OUTPUT if args.optimize_initial_state else OUTPUT
    daily, hourly, training, profiles, skipped = run_all_days(
        frame, args.start_day, args.end_day, args.baseline_weeks, args.steps_per_hour,
        deterministic=args.deterministic, optimize_initial_state=args.optimize_initial_state,
        save_daily_plots=args.save_daily_plots, fail_fast=args.fail_fast, output_root=output_root,
        f0_min=args.f0_min, f0_max=args.f0_max, f0_steps=args.f0_steps,
        max_state_iter=args.max_state_iter, state_tol=args.state_tol, verbose=args.verbose,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_root / "daily_metrics.csv", index=False)
    hourly.to_csv(output_root / "hourly_predictions.csv")
    training.to_csv(output_root / "brute_force_training_states.csv", index=False)
    profiles.to_csv(output_root / "f0_profiles.csv", index=False)
    if skipped: pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output_root / "skipped_days.csv", index=False)
    save_performance_plot(hourly, daily, output_root / "all_days_performance.png")
    print(); print_summary(hourly, daily, skipped)


if __name__ == "__main__": main()
