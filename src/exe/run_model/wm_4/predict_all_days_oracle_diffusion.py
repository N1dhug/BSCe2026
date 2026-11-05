from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.exe.run_model.wm_4.analyze_observation_uncertainty import _stack_days, uncertainty_frame
from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY, fit_history_optimized
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup
from src.params.wm_4 import constant_diffusion_qp, optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_oracle_diffusion"
Z_95 = float(norm.ppf(0.975))


def fit_day(frame, day, baseline_weeks, steps_per_hour, max_state_iter=50, state_tol=1e-8):
    day, baseline_weeks, steps_per_hour = day_start(day), int(baseline_weeks), int(steps_per_hour)
    output_model, output_fit, fit, _ = fit_history_optimized(
        frame, day, baseline_weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )
    _, actual, U, Time, model, _ = prediction_setup(frame, day, output_model, steps_per_hour)

    prediction_data, fine_index, _, _ = model_input(frame, day, steps_per_hour, 0, output_model)
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(prediction_data, fit["r"])
    init = np.asarray(oracle["init_cond"], dtype=float)
    state = oracle["deterministic_state"]

    empirical = uncertainty_frame(_stack_days(frame, fit["training_days"]))
    observed_variance = empirical["var_I"].to_numpy(dtype=float)
    hourly_index = np.arange(DAY_HOURS, dtype=int) * steps_per_hour
    process_kernel = model.process_variance_kernel({"r": fit["r"]}, hourly_index)
    diffusion = constant_diffusion_qp(model).fit(
        process_kernel, observed_variance,
        initial_variance=np.zeros(DAY_HOURS, dtype=float),
        fit_mask=np.ones(DAY_HOURS, dtype=bool),
    )

    process_variance = np.asarray(diffusion["process_variance"], dtype=float)
    process_std = np.sqrt(np.maximum(process_variance, 0.0))
    observed = actual["I"].to_numpy(dtype=float)
    predicted = np.asarray(state["I"], dtype=float)[hourly_index]
    lower, upper = predicted - Z_95 * process_std, predicted + Z_95 * process_std
    covered = (observed >= lower) & (observed <= upper)
    metrics = deviation_metrics(observed, predicted)
    r = np.asarray(fit["r"], dtype=float)

    post_initial = covered[1:] if covered.size > 1 else covered
    daily = {
        "day": day, "weekday": day.day_name(),
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "sigma": float(diffusion["sigma"]), "diffusion_q": float(diffusion["q"]),
        "diffusion_variance_rmse": float(diffusion["variance_rmse"]),
        "oracle_F0": init[0], "oracle_R0": init[1], "oracle_fit_mse": float(oracle["loss"]),
        "coverage_95": float(np.mean(covered)),
        "coverage_95_hours_1_23": float(np.mean(post_initial)),
        "full_day_covered_95": bool(np.all(covered)),
        "hours_1_23_all_covered_95": bool(np.all(post_initial)),
        "hour0_covered_95": bool(covered[0]),
        "mean_process_std": float(np.mean(process_std)),
        "wm4_mse": metrics["mse"], "wm4_rmse": metrics["rmse"],
        "wm4_mae": metrics["mae"], "wm4_bias": metrics["bias"],
        "O_fit_mse": float(output_fit["mse"]), "state_iterations": int(fit["state_iterations"]),
    }
    hourly = pd.DataFrame({
        "day": day, "I_observed": observed, "I_oracle": predicted,
        "ci95_lower": lower, "ci95_upper": upper, "covered_95": covered,
        "process_std": process_std, "process_variance": process_variance,
        "O": np.asarray(state["O"], dtype=float)[hourly_index],
        "U": np.asarray(state["U"], dtype=float)[hourly_index],
        "F": np.asarray(state["F"], dtype=float)[hourly_index],
        "R": np.asarray(state["R"], dtype=float)[hourly_index],
    }, index=actual.index)
    hourly.index.name = "time"
    return daily, hourly


def run_all_days(frame, start=START_DAY, end=END_DAY, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, fail_fast=False, max_state_iter=50, state_tol=1e-8):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")
    daily_rows, hourly_frames, skipped = [], [], []

    for day in pd.date_range(start, end, freq="D"):
        try:
            daily, hourly = fit_day(
                frame, day, baseline_weeks, steps_per_hour,
                max_state_iter=max_state_iter, state_tol=state_tol,
            )
            daily_rows.append(daily); hourly_frames.append(hourly)
            print(
                f"{day:%Y-%m-%d} | sigma={daily['sigma']:.6f} | "
                f"oracle MSE={daily['wm4_mse']:.8f} | "
                f"coverage95={100.0 * daily['coverage_95']:.2f}% | "
                f"hours1-23={100.0 * daily['coverage_95_hours_1_23']:.2f}%"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc))); print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not daily_rows: raise RuntimeError("No days in the requested range could be fitted.")
    return pd.DataFrame(daily_rows).sort_values("day").reset_index(drop=True), pd.concat(hourly_frames).sort_index(), skipped


def save_plot(hourly, daily, path):
    fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=False)

    axes[0].plot(hourly.index, hourly["I_observed"], label="Observed I", linewidth=0.9)
    axes[0].plot(hourly.index, hourly["I_oracle"], label="Oracle-state ARM mean", linewidth=0.9)
    axes[0].fill_between(hourly.index, hourly["ci95_lower"], hourly["ci95_upper"], alpha=0.2, label="Process-only 95% interval")
    axes[0].set_ylabel("Normalized input"); axes[0].legend(); axes[0].grid(True, alpha=0.25)

    axes[1].plot(daily["day"], 100.0 * daily["coverage_95"], label="24-hour coverage")
    axes[1].plot(daily["day"], 100.0 * daily["coverage_95_hours_1_23"], label="Hours 1-23 coverage")
    axes[1].axhline(95.0, linestyle="--", label="95% target")
    axes[1].set_ylim(0.0, 100.0); axes[1].set_ylabel("Coverage (%)")
    axes[1].legend(); axes[1].grid(True, alpha=0.25)

    axes[2].plot(daily["day"], daily["sigma"], label="Fitted constant diffusion sigma")
    axes[2].set_xlabel("Prediction day"); axes[2].set_ylabel("sigma")
    axes[2].legend(); axes[2].grid(True, alpha=0.25)

    fig.suptitle("ARM (Affine Revised Model) oracle initial state + constant process diffusion")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def print_summary(hourly, daily, skipped):
    covered = hourly["covered_95"].to_numpy(dtype=bool)
    post_initial_mask = hourly.groupby("day", sort=False).cumcount().to_numpy() > 0
    post_initial = hourly.loc[post_initial_mask, "covered_95"].to_numpy(dtype=bool)
    full_days = daily["full_day_covered_95"].to_numpy(dtype=bool)
    post_full_days = daily["hours_1_23_all_covered_95"].to_numpy(dtype=bool)
    hour0 = daily["hour0_covered_95"].to_numpy(dtype=bool)

    print()
    print(f"days fitted={len(daily)}, skipped={len(skipped)}, hourly observations={len(hourly)}")
    print(f"oracle-state + sigma 95% input coverage={100.0 * np.mean(covered):.2f}% ({np.count_nonzero(covered)}/{covered.size})")
    print(f"oracle-state + sigma coverage, hours 1-23={100.0 * np.mean(post_initial):.2f}% ({np.count_nonzero(post_initial)}/{post_initial.size})")
    print(f"hour-0 coverage={100.0 * np.mean(hour0):.2f}% ({np.count_nonzero(hour0)}/{hour0.size})")
    print(f"days with all 24 observations covered={100.0 * np.mean(full_days):.2f}% ({np.count_nonzero(full_days)}/{full_days.size})")
    print(f"days with all hours 1-23 covered={100.0 * np.mean(post_full_days):.2f}% ({np.count_nonzero(post_full_days)}/{post_full_days.size})")
    print()
    print(f"mean oracle MSE={daily['wm4_mse'].mean():.10f}")
    print(f"mean sigma={daily['sigma'].mean():.8f}")
    print(f"mean process std={daily['mean_process_std'].mean():.8f}")

    if skipped:
        print(); print("skipped days:")
        for day, reason in skipped: print(f"{day:%Y-%m-%d}: {reason}")


def main():
    parser = ArgumentParser(description="Evaluate all WM4 prediction days with the oracle initial state and constant process diffusion only.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    daily, hourly, skipped = run_all_days(
        frame, args.start_day, args.end_day, args.baseline_weeks, args.steps_per_hour,
        fail_fast=args.fail_fast, max_state_iter=args.max_state_iter, state_tol=args.state_tol,
    )

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_oracle_diffusion.csv", index=False)
    hourly.to_csv(output / "hourly_oracle_diffusion.csv")
    if skipped: pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)
    save_plot(hourly, daily, output / "all_days_oracle_diffusion.png")
    print_summary(hourly, daily, skipped)


if __name__ == "__main__": main()
