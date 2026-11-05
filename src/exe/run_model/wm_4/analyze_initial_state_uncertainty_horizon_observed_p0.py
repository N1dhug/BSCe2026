from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.exe.run_model.wm_4.analyze_initial_state_uncertainty_horizon import calibration_window, process_covariance
from src.exe.run_model.wm_4.analyze_observation_uncertainty import _stack_days, uncertainty_frame
from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY, fit_history_optimized
from src.exe.run_model.wm_4.predict_daily import prediction_setup
from src.params.wm_4 import constant_diffusion_qp, optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "uncertainty" / "initial_state_horizon_observed_p0"
CONFIDENCE = 0.95


def fit_day(frame, day, weeks=BASELINE_WEEKS, steps_per_hour=60, confidence=CONFIDENCE, max_state_iter=50, state_tol=1e-8):
    day, weeks, steps_per_hour = day_start(day), int(weeks), int(steps_per_hour)
    output_model, _, fit, _ = fit_history_optimized(
        frame, day, weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )
    _, actual, U, Time, model, _ = prediction_setup(frame, day, output_model, steps_per_hour)
    hourly_index = np.arange(DAY_HOURS, dtype=int) * steps_per_hour

    x_hat = np.asarray(fit["x_hat"], dtype=float)
    P0 = np.asarray(fit["x_cov"], dtype=float)
    estimated_state = model.deterministic_solver(x_hat, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})

    prediction_data, _, _, _ = model_input(frame, day, steps_per_hour, 0, output_model)
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(prediction_data, fit["r"])
    oracle_state = oracle["deterministic_state"]

    empirical = uncertainty_frame(_stack_days(frame, fit["training_days"]))
    observed_variance = empirical["var_I"].to_numpy(dtype=float)

    sensitivity = model.sensitivity(fit["r"])
    J_h = np.asarray(sensitivity["input"], dtype=float)[hourly_index]
    initial_variance = np.einsum("ni,ij,nj->n", J_h, P0, J_h)
    initial_variance = np.maximum(initial_variance, 0.0)

    process_kernel = model.process_variance_kernel({"r": fit["r"]}, hourly_index)

    # Historical P0 is part of the initial distribution for both centres.
    # Fit q only to variance not already explained by propagated P0.
    diffusion = constant_diffusion_qp(model).fit(
        process_kernel, observed_variance,
        initial_variance=initial_variance,
        fit_mask=np.ones(DAY_HOURS, dtype=bool),
    )

    process_cov = process_covariance(model, fit["r"], diffusion["q"])
    process_variance = np.maximum(process_cov[hourly_index, 1, 1], 0.0)
    joint_variance = np.maximum(initial_variance + process_variance, 0.0)
    initial_std = np.sqrt(initial_variance)
    process_std = np.sqrt(process_variance)
    joint_std = np.sqrt(joint_variance)

    observed = actual["I"].to_numpy(dtype=float)
    estimated_I = np.asarray(estimated_state["I"], dtype=float)[hourly_index]
    oracle_I = np.asarray(oracle_state["I"], dtype=float)[hourly_index]

    z = float(norm.ppf(0.5 + 0.5 * confidence))
    initial_lower, initial_upper = estimated_I - z * initial_std, estimated_I + z * initial_std
    joint_lower, joint_upper = estimated_I - z * joint_std, estimated_I + z * joint_std
    oracle_joint_lower, oracle_joint_upper = oracle_I - z * joint_std, oracle_I + z * joint_std

    observed_initial = (observed >= initial_lower) & (observed <= initial_upper)
    observed_joint = (observed >= joint_lower) & (observed <= joint_upper)
    oracle_I_initial = (oracle_I >= initial_lower) & (oracle_I <= initial_upper)
    observed_oracle_joint = (observed >= oracle_joint_lower) & (observed <= oracle_joint_upper)

    hourly = pd.DataFrame({
        "day": day, "forecast_hour": np.arange(DAY_HOURS, dtype=int), "time": actual.index,
        "I_observed": observed, "I_estimated": estimated_I, "I_oracle": oracle_I,
        "observed_variance": observed_variance,
        "initial_variance": initial_variance,
        "process_variance": process_variance,
        "joint_variance": joint_variance,
        "initial_std_I": initial_std,
        "process_std_I": process_std,
        "joint_std_I": joint_std,
        "initial_ci95_lower": initial_lower, "initial_ci95_upper": initial_upper,
        "joint_ci95_lower": joint_lower, "joint_ci95_upper": joint_upper,
        "oracle_joint_ci95_lower": oracle_joint_lower, "oracle_joint_ci95_upper": oracle_joint_upper,
        "observed_in_initial_95": observed_initial,
        "observed_in_joint_95": observed_joint,
        "oracle_I_in_initial_95": oracle_I_initial,
        "observed_in_oracle_joint_95": observed_oracle_joint,
        "estimated_error": estimated_I - observed,
        "oracle_error": oracle_I - observed,
    })

    daily = {
        "day": day, "weekday": day.day_name(), "weeks": weeks,
        "sigma": float(diffusion["sigma"]), "q": float(diffusion["q"]),
        "variance_rmse": float(diffusion["variance_rmse"]),
        "F0_hat": x_hat[0], "R0_hat": x_hat[1],
        "F0_std": float(np.sqrt(max(P0[0, 0], 0.0))),
        "R0_std": float(np.sqrt(max(P0[1, 1], 0.0))),
        "cov_F0_R0": float(P0[0, 1]),
        "oracle_F0": float(oracle["init_cond"][0]),
        "oracle_R0": float(oracle["init_cond"][1]),
        "oracle_fit_mse": float(oracle["loss"]),
        "observed_initial_coverage_pct": 100.0 * float(np.mean(observed_initial)),
        "observed_joint_coverage_pct": 100.0 * float(np.mean(observed_joint)),
        "oracle_I_initial_coverage_pct": 100.0 * float(np.mean(oracle_I_initial)),
        "oracle_joint_coverage_pct": 100.0 * float(np.mean(observed_oracle_joint)),
        "mean_initial_std_I": float(np.mean(initial_std)),
        "mean_process_std_I": float(np.mean(process_std)),
        "mean_joint_std_I": float(np.mean(joint_std)),
    }
    return daily, hourly


def horizon_summary(hourly, confidence=CONFIDENCE, tolerance_pct_points=5.0):
    nominal = 100.0 * float(confidence)
    rows = []
    for hour, part in hourly.groupby("forecast_hour", sort=True):
        row = {
            "forecast_hour": int(hour), "days": int(len(part)), "nominal_coverage_pct": nominal,
            "observed_initial_coverage_pct": 100.0 * float(part["observed_in_initial_95"].mean()),
            "observed_joint_coverage_pct": 100.0 * float(part["observed_in_joint_95"].mean()),
            "oracle_I_initial_coverage_pct": 100.0 * float(part["oracle_I_in_initial_95"].mean()),
            "oracle_joint_coverage_pct": 100.0 * float(part["observed_in_oracle_joint_95"].mean()),
            "mean_observed_variance": float(part["observed_variance"].mean()),
            "mean_initial_variance": float(part["initial_variance"].mean()),
            "mean_process_variance": float(part["process_variance"].mean()),
            "mean_joint_variance": float(part["joint_variance"].mean()),
            "mean_initial_std_I": float(part["initial_std_I"].mean()),
            "mean_process_std_I": float(part["process_std_I"].mean()),
            "mean_joint_std_I": float(part["joint_std_I"].mean()),
            "estimated_rmse": float(np.sqrt(np.mean(part["estimated_error"].to_numpy(dtype=float) ** 2))),
            "oracle_rmse": float(np.sqrt(np.mean(part["oracle_error"].to_numpy(dtype=float) ** 2))),
        }
        for name in ("observed_joint", "oracle_I_initial", "oracle_joint"):
            row[f"{name}_within_tolerance"] = bool(
                abs(row[f"{name}_coverage_pct"] - nominal) <= float(tolerance_pct_points)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(horizon):
    mapping = {
        "observed_joint": "observed_joint_within_tolerance",
        "oracle_I_initial": "oracle_I_initial_within_tolerance",
        "oracle_joint": "oracle_joint_within_tolerance",
    }
    return pd.DataFrame([
        {"metric": metric, **calibration_window(horizon, column)}
        for metric, column in mapping.items()
    ])


def save_plot(horizon, path, confidence=CONFIDENCE, tolerance_pct_points=5.0):
    nominal = 100.0 * float(confidence)
    hours = horizon["forecast_hour"].to_numpy(dtype=int)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(hours, horizon["observed_joint_coverage_pct"], marker="o", label="Estimated state + fitted P0 + diffusion")
    axes[0].plot(hours, horizon["oracle_I_initial_coverage_pct"], marker="o", label="Oracle I under initial-state distribution")
    axes[0].plot(hours, horizon["oracle_joint_coverage_pct"], marker="o", label="Oracle centre + same P0 + diffusion")
    axes[0].axhline(nominal, linestyle="--", label=f"{nominal:.0f}% target")
    axes[0].fill_between(hours, nominal - tolerance_pct_points, nominal + tolerance_pct_points, alpha=0.1, label=f"±{tolerance_pct_points:g} pp")
    axes[0].set_ylabel("Coverage (%)"); axes[0].set_ylim(0.0, 102.0)
    axes[0].legend(); axes[0].grid(True, alpha=0.25)

    axes[1].plot(hours, horizon["mean_initial_std_I"], marker="o", label="Initial-state std")
    axes[1].plot(hours, horizon["mean_process_std_I"], marker="o", label="Process std fitted after P0")
    axes[1].plot(hours, horizon["mean_joint_std_I"], marker="o", label="Joint std")
    axes[1].set_ylabel("Mean std(I)"); axes[1].legend(); axes[1].grid(True, alpha=0.25)

    axes[2].plot(hours, horizon["mean_observed_variance"], marker="o", label="Observed cross-day variance")
    axes[2].plot(hours, horizon["mean_initial_variance"], marker="o", label="Propagated P0 variance")
    axes[2].plot(hours, horizon["mean_process_variance"], marker="o", label="Fitted process variance")
    axes[2].plot(hours, horizon["mean_joint_variance"], marker="o", label="Joint fitted variance")
    axes[2].set_xlabel("Forecast horizon (hours)"); axes[2].set_ylabel("Variance")
    axes[2].set_xticks(np.arange(0, DAY_HOURS, 2)); axes[2].legend(); axes[2].grid(True, alpha=0.25)

    fig.suptitle("ARM (Affine Revised Model) uncertainty decomposition with observed initial-state covariance")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def print_summary(daily, horizon, skipped, tolerance_pct_points):
    print()
    print(f"days fitted={len(daily)}, skipped={len(skipped)}, nominal coverage=95.00%, tolerance=±{tolerance_pct_points:.2f} percentage points")
    print(f"mean sigma fitted after propagated P0={daily['sigma'].mean():.8f}")
    print("Estimated-centre and oracle-centre intervals use the same P0, diffusion, and joint interval width.")
    print()

    for label, column in (
        ("observed joint", "observed_joint_within_tolerance"),
        ("oracle I under initial-state distribution", "oracle_I_initial_within_tolerance"),
        ("oracle-centred joint P0+diffusion", "oracle_joint_within_tolerance"),
    ):
        window = calibration_window(horizon, column)
        run = "none" if window["longest_run_start"] is None else f"{window['longest_run_start']}-{window['longest_run_end']} h"
        print(
            f"{label}: within tolerance at {window['hours_within_tolerance']}/{DAY_HOURS} horizons | "
            f"from t=0: {window['hours_from_start']} h | longest run: {run} ({window['longest_run_hours']} h)"
        )

    print()
    print(f"{'hour':>5}{'joint obs':>12}{'oracle I':>12}{'oracle joint':>14}{'init std':>12}{'proc std':>12}{'joint std':>12}")
    for _, row in horizon.iterrows():
        print(
            f"{int(row['forecast_hour']):>5d}"
            f"{row['observed_joint_coverage_pct']:>11.2f}%"
            f"{row['oracle_I_initial_coverage_pct']:>11.2f}%"
            f"{row['oracle_joint_coverage_pct']:>13.2f}%"
            f"{row['mean_initial_std_I']:>12.6f}"
            f"{row['mean_process_std_I']:>12.6f}"
            f"{row['mean_joint_std_I']:>12.6f}"
        )


def main():
    parser = ArgumentParser(description="Analyse WM4 uncertainty using observed historical P0 for both estimated-centre and oracle-centre predictions, with one shared constant diffusion fit.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--coverage-tolerance", type=float, default=5.0)
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    daily_rows, hourly_frames, skipped = [], [], []
    for day in pd.date_range(day_start(args.start_day), day_start(args.end_day), freq="D"):
        try:
            daily, hourly = fit_day(
                frame, day, weeks=args.weeks, steps_per_hour=args.steps_per_hour,
                max_state_iter=args.max_state_iter, state_tol=args.state_tol,
            )
            daily_rows.append(daily); hourly_frames.append(hourly)
            print(
                f"{day:%Y-%m-%d} | "
                f"sigma={daily['sigma']:.6f} | "
                f"estimated joint={daily['observed_joint_coverage_pct']:.2f}% | "
                f"oracle joint={daily['oracle_joint_coverage_pct']:.2f}%"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if args.fail_fast: raise
            skipped.append((day, str(exc))); print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not daily_rows: raise RuntimeError("No days in the requested range could be analysed.")
    daily = pd.DataFrame(daily_rows).sort_values("day").reset_index(drop=True)
    hourly = pd.concat(hourly_frames, ignore_index=True).sort_values(["day", "forecast_hour"]).reset_index(drop=True)
    horizon = horizon_summary(hourly, tolerance_pct_points=args.coverage_tolerance)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_observed_p0.csv", index=False)
    hourly.to_csv(output / "hourly_observed_p0.csv", index=False)
    horizon.to_csv(output / "horizon_observed_p0.csv", index=False)
    calibration_table(horizon).to_csv(output / "calibration_windows.csv", index=False)
    if skipped: pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)
    save_plot(horizon, output / "observed_p0_horizon.png", tolerance_pct_points=args.coverage_tolerance)
    print_summary(daily, horizon, skipped, args.coverage_tolerance)
    print(); print(f"output={output}")


if __name__ == "__main__": main()
