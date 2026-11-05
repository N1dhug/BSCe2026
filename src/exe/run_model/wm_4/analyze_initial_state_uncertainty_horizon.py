from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from src.exe.run_model.wm_4.analyze_observation_uncertainty import _stack_days, uncertainty_frame
from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY, fit_history_optimized
from src.exe.run_model.wm_4.predict_daily import prediction_setup
from src.params.wm_4 import constant_diffusion_qp, optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "uncertainty" / "initial_state_horizon"
CONFIDENCE = 0.95


def ellipse_coverage(mean, covariance, target, confidence=CONFIDENCE):
    mean, target = np.asarray(mean, dtype=float).reshape(2), np.asarray(target, dtype=float).reshape(2)
    covariance = np.asarray(covariance, dtype=float).reshape(2, 2)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    positive = eigenvalues > 1e-10 * scale
    rank = int(np.count_nonzero(positive))
    delta = target - mean

    if rank == 0:
        distance2 = 0.0 if np.linalg.norm(delta) <= 1e-10 else np.inf
        threshold = 0.0
        inside = bool(np.isfinite(distance2))
    else:
        projected = eigenvectors.T @ delta
        null_residual = projected[~positive]
        supported = bool(np.linalg.norm(null_residual) <= 1e-8 * max(1.0, np.linalg.norm(delta)))
        distance2 = float(np.sum(projected[positive] ** 2 / eigenvalues[positive]))
        threshold = float(chi2.ppf(confidence, rank))
        inside = bool(supported and distance2 <= threshold)
    return inside, distance2, threshold, rank


def process_covariance(model, r, q):
    transition = np.asarray(model.sensitivity(r)["transition"], dtype=float)
    covariance = np.zeros((model.t_steps, 2, 2), dtype=float)
    process = np.array([[0.0, 0.0], [0.0, float(q) * model.dt]], dtype=float)
    for k in range(model.t_steps - 1):
        covariance[k + 1] = transition @ covariance[k] @ transition.T + process
        covariance[k + 1] = 0.5 * (covariance[k + 1] + covariance[k + 1].T)
    return covariance


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
    x_oracle = np.asarray(oracle["init_cond"], dtype=float)
    oracle_state = oracle["deterministic_state"]

    empirical = uncertainty_frame(_stack_days(frame, fit["training_days"]))
    observed_variance = empirical["var_I"].to_numpy(dtype=float)
    kernel = model.process_variance_kernel({"r": fit["r"]}, hourly_index)
    diffusion = constant_diffusion_qp(model).fit(
        kernel, observed_variance,
        initial_variance=np.zeros(DAY_HOURS, dtype=float),
        fit_mask=np.ones(DAY_HOURS, dtype=bool),
    )

    sensitivity = model.sensitivity(fit["r"])
    Phi = np.asarray(sensitivity["state"], dtype=float)
    initial_covariance = np.einsum("nij,jk,nlk->nil", Phi, P0, Phi)
    process_cov = process_covariance(model, fit["r"], diffusion["q"])
    joint_covariance = initial_covariance + process_cov

    initial_cov_h = initial_covariance[hourly_index]
    process_cov_h = process_cov[hourly_index]
    joint_cov_h = joint_covariance[hourly_index]

    observed = actual["I"].to_numpy(dtype=float)
    estimated_I = np.asarray(estimated_state["I"], dtype=float)[hourly_index]
    oracle_I = np.asarray(oracle_state["I"], dtype=float)[hourly_index]
    estimated_x = np.column_stack((
        np.asarray(estimated_state["F"], dtype=float)[hourly_index],
        np.asarray(estimated_state["R"], dtype=float)[hourly_index],
    ))
    oracle_x = np.column_stack((
        np.asarray(oracle_state["F"], dtype=float)[hourly_index],
        np.asarray(oracle_state["R"], dtype=float)[hourly_index],
    ))

    z = float(norm.ppf(0.5 + 0.5 * confidence))
    initial_std = np.sqrt(np.maximum(initial_cov_h[:, 1, 1], 0.0))
    process_std = np.sqrt(np.maximum(process_cov_h[:, 1, 1], 0.0))
    joint_std = np.sqrt(np.maximum(joint_cov_h[:, 1, 1], 0.0))

    initial_lower, initial_upper = estimated_I - z * initial_std, estimated_I + z * initial_std
    joint_lower, joint_upper = estimated_I - z * joint_std, estimated_I + z * joint_std
    oracle_process_lower, oracle_process_upper = oracle_I - z * process_std, oracle_I + z * process_std

    observed_initial = (observed >= initial_lower) & (observed <= initial_upper)
    observed_joint = (observed >= joint_lower) & (observed <= joint_upper)
    oracle_I_initial = (oracle_I >= initial_lower) & (oracle_I <= initial_upper)
    oracle_I_joint = (oracle_I >= joint_lower) & (oracle_I <= joint_upper)
    observed_oracle_process = (observed >= oracle_process_lower) & (observed <= oracle_process_upper)

    oracle_state_initial, oracle_state_joint = [], []
    initial_distance2, joint_distance2 = [], []
    F_initial, R_initial = [], []
    for h in range(DAY_HOURS):
        init_inside, init_d2, _, _ = ellipse_coverage(estimated_x[h], initial_cov_h[h], oracle_x[h], confidence)
        joint_inside, joint_d2, _, _ = ellipse_coverage(estimated_x[h], joint_cov_h[h], oracle_x[h], confidence)
        oracle_state_initial.append(init_inside); oracle_state_joint.append(joint_inside)
        initial_distance2.append(init_d2); joint_distance2.append(joint_d2)

        marginal_std = np.sqrt(np.maximum(np.diag(initial_cov_h[h]), 0.0))
        delta = np.abs(oracle_x[h] - estimated_x[h])
        F_initial.append(bool(delta[0] <= z * marginal_std[0]))
        R_initial.append(bool(delta[1] <= z * marginal_std[1]))

    hourly = pd.DataFrame({
        "day": day, "forecast_hour": np.arange(DAY_HOURS, dtype=int), "time": actual.index,
        "I_observed": observed, "I_estimated": estimated_I, "I_oracle": oracle_I,
        "initial_std_I": initial_std, "process_std_I": process_std, "joint_std_I": joint_std,
        "initial_ci95_lower": initial_lower, "initial_ci95_upper": initial_upper,
        "joint_ci95_lower": joint_lower, "joint_ci95_upper": joint_upper,
        "oracle_process_ci95_lower": oracle_process_lower, "oracle_process_ci95_upper": oracle_process_upper,
        "observed_in_initial_95": observed_initial, "observed_in_joint_95": observed_joint,
        "oracle_I_in_initial_95": oracle_I_initial, "oracle_I_in_joint_95": oracle_I_joint,
        "observed_in_oracle_process_95": observed_oracle_process,
        "oracle_state_in_initial_ellipse_95": np.asarray(oracle_state_initial, dtype=bool),
        "oracle_state_in_joint_ellipse_95": np.asarray(oracle_state_joint, dtype=bool),
        "oracle_F_in_initial_marginal_95": np.asarray(F_initial, dtype=bool),
        "oracle_R_in_initial_marginal_95": np.asarray(R_initial, dtype=bool),
        "oracle_state_initial_mahalanobis2": np.asarray(initial_distance2, dtype=float),
        "oracle_state_joint_mahalanobis2": np.asarray(joint_distance2, dtype=float),
        "estimated_error": estimated_I - observed, "oracle_error": oracle_I - observed,
    })

    daily = {
        "day": day, "weekday": day.day_name(), "weeks": weeks,
        "sigma": float(diffusion["sigma"]), "q": float(diffusion["q"]),
        "F0_hat": x_hat[0], "R0_hat": x_hat[1],
        "F0_std": float(np.sqrt(max(P0[0, 0], 0.0))), "R0_std": float(np.sqrt(max(P0[1, 1], 0.0))),
        "oracle_F0": x_oracle[0], "oracle_R0": x_oracle[1],
        "oracle_fit_mse": float(oracle["loss"]),
        "observed_joint_coverage_pct": 100.0 * float(np.mean(observed_joint)),
        "oracle_I_initial_coverage_pct": 100.0 * float(np.mean(oracle_I_initial)),
        "oracle_state_initial_coverage_pct": 100.0 * float(np.mean(oracle_state_initial)),
        "oracle_process_coverage_pct": 100.0 * float(np.mean(observed_oracle_process)),
        "mean_initial_std_I": float(np.mean(initial_std)),
        "mean_process_std_I": float(np.mean(process_std)),
        "mean_joint_std_I": float(np.mean(joint_std)),
    }
    return daily, hourly


def horizon_summary(hourly, confidence=CONFIDENCE, tolerance_pct_points=5.0):
    nominal = 100.0 * float(confidence)
    rows = []
    for hour, part in hourly.groupby("forecast_hour", sort=True):
        estimated_error = part["estimated_error"].to_numpy(dtype=float)
        oracle_error = part["oracle_error"].to_numpy(dtype=float)
        row = {
            "forecast_hour": int(hour), "days": int(len(part)), "nominal_coverage_pct": nominal,
            "observed_initial_coverage_pct": 100.0 * float(part["observed_in_initial_95"].mean()),
            "observed_joint_coverage_pct": 100.0 * float(part["observed_in_joint_95"].mean()),
            "oracle_I_initial_coverage_pct": 100.0 * float(part["oracle_I_in_initial_95"].mean()),
            "oracle_I_joint_coverage_pct": 100.0 * float(part["oracle_I_in_joint_95"].mean()),
            "oracle_state_initial_ellipse_coverage_pct": 100.0 * float(part["oracle_state_in_initial_ellipse_95"].mean()),
            "oracle_state_joint_ellipse_coverage_pct": 100.0 * float(part["oracle_state_in_joint_ellipse_95"].mean()),
            "oracle_F_initial_marginal_coverage_pct": 100.0 * float(part["oracle_F_in_initial_marginal_95"].mean()),
            "oracle_R_initial_marginal_coverage_pct": 100.0 * float(part["oracle_R_in_initial_marginal_95"].mean()),
            "oracle_process_coverage_pct": 100.0 * float(part["observed_in_oracle_process_95"].mean()),
            "mean_initial_std_I": float(part["initial_std_I"].mean()),
            "mean_process_std_I": float(part["process_std_I"].mean()),
            "mean_joint_std_I": float(part["joint_std_I"].mean()),
            "estimated_rmse": float(np.sqrt(np.mean(estimated_error ** 2))),
            "oracle_rmse": float(np.sqrt(np.mean(oracle_error ** 2))),
        }
        for name in ("observed_joint", "oracle_I_initial", "oracle_state_initial_ellipse", "oracle_process"):
            coverage = row[f"{name}_coverage_pct"]
            row[f"{name}_within_tolerance"] = bool(abs(coverage - nominal) <= float(tolerance_pct_points))
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_window(horizon, column):
    mask = horizon[column].to_numpy(dtype=bool)
    hours = horizon["forecast_hour"].to_numpy(dtype=int)
    from_start = 0
    for value in mask:
        if not value: break
        from_start += 1

    best_start, best_end, best_length = None, None, 0
    run_start = None
    for i, value in enumerate(mask):
        if value and run_start is None: run_start = i
        if run_start is not None and (not value or i == len(mask) - 1):
            run_end = i if value and i == len(mask) - 1 else i - 1
            length = run_end - run_start + 1
            if length > best_length:
                best_start, best_end, best_length = int(hours[run_start]), int(hours[run_end]), int(length)
            run_start = None
    return {
        "hours_from_start": int(from_start),
        "longest_run_hours": int(best_length),
        "longest_run_start": best_start,
        "longest_run_end": best_end,
        "hours_within_tolerance": int(np.count_nonzero(mask)),
    }


def calibration_table(horizon):
    labels = {
        "observed_joint": "observed_joint_within_tolerance",
        "oracle_I_initial": "oracle_I_initial_within_tolerance",
        "oracle_state_initial_ellipse": "oracle_state_initial_ellipse_within_tolerance",
        "oracle_process": "oracle_process_within_tolerance",
    }
    rows = []
    for metric, column in labels.items():
        rows.append({"metric": metric, **calibration_window(horizon, column)})
    return pd.DataFrame(rows)


def save_plot(horizon, path, confidence=CONFIDENCE, tolerance_pct_points=5.0):
    nominal = 100.0 * float(confidence)
    hours = horizon["forecast_hour"].to_numpy(dtype=int)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(hours, horizon["observed_joint_coverage_pct"], marker="o", label="Observed I: initial + diffusion")
    axes[0].plot(hours, horizon["oracle_I_initial_coverage_pct"], marker="o", label="Oracle I: initial-state uncertainty only")
    axes[0].plot(hours, horizon["oracle_state_initial_ellipse_coverage_pct"], marker="o", label="Oracle [F,R]: initial-state 95% ellipse")
    axes[0].plot(hours, horizon["oracle_process_coverage_pct"], marker="o", label="Observed I: oracle centre + diffusion")
    axes[0].axhline(nominal, linestyle="--", label=f"{nominal:.0f}% target")
    axes[0].fill_between(hours, nominal - tolerance_pct_points, nominal + tolerance_pct_points, alpha=0.1, label=f"±{tolerance_pct_points:g} pp")
    axes[0].set_ylabel("Coverage (%)"); axes[0].set_ylim(0.0, 102.0)
    axes[0].legend(ncol=2); axes[0].grid(True, alpha=0.25)

    axes[1].plot(hours, horizon["mean_initial_std_I"], marker="o", label="Initial-state contribution")
    axes[1].plot(hours, horizon["mean_process_std_I"], marker="o", label="Diffusion contribution")
    axes[1].plot(hours, horizon["mean_joint_std_I"], marker="o", label="Joint")
    axes[1].set_ylabel("Mean std(I)"); axes[1].legend(); axes[1].grid(True, alpha=0.25)

    axes[2].plot(hours, horizon["estimated_rmse"], marker="o", label="Estimated-state mean RMSE")
    axes[2].plot(hours, horizon["oracle_rmse"], marker="o", label="Oracle-state mean RMSE")
    axes[2].set_xlabel("Forecast horizon (hours)"); axes[2].set_ylabel("RMSE")
    axes[2].set_xticks(np.arange(0, DAY_HOURS, 2)); axes[2].legend(); axes[2].grid(True, alpha=0.25)

    fig.suptitle("ARM (Affine Revised Model) initial-state uncertainty calibration by forecast horizon")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def print_summary(daily, horizon, skipped, tolerance_pct_points):
    nominal = 100.0 * CONFIDENCE
    metrics = {
        "observed joint": "observed_joint_within_tolerance",
        "oracle I under initial-state distribution": "oracle_I_initial_within_tolerance",
        "oracle [F,R] under initial-state ellipse": "oracle_state_initial_ellipse_within_tolerance",
        "oracle-centred process": "oracle_process_within_tolerance",
    }
    print()
    print(f"days fitted={len(daily)}, skipped={len(skipped)}, nominal coverage={nominal:.2f}%, tolerance=±{tolerance_pct_points:.2f} percentage points")
    print(f"mean sigma={daily['sigma'].mean():.8f}")
    print()
    for label, column in metrics.items():
        window = calibration_window(horizon, column)
        run = "none" if window["longest_run_start"] is None else f"{window['longest_run_start']}-{window['longest_run_end']} h"
        print(
            f"{label}: within tolerance at {window['hours_within_tolerance']}/{DAY_HOURS} horizons | "
            f"from t=0: {window['hours_from_start']} h | longest run: {run} ({window['longest_run_hours']} h)"
        )

    print()
    print(f"{'hour':>5}{'joint obs':>12}{'oracle I':>12}{'oracle state':>14}{'oracle+sigma':>14}{'init std':>12}{'proc std':>12}")
    for _, row in horizon.iterrows():
        print(
            f"{int(row['forecast_hour']):>5d}"
            f"{row['observed_joint_coverage_pct']:>11.2f}%"
            f"{row['oracle_I_initial_coverage_pct']:>11.2f}%"
            f"{row['oracle_state_initial_ellipse_coverage_pct']:>13.2f}%"
            f"{row['oracle_process_coverage_pct']:>13.2f}%"
            f"{row['mean_initial_std_I']:>12.6f}"
            f"{row['mean_process_std_I']:>12.6f}"
        )


def main():
    parser = ArgumentParser(description="Analyse how WM4 initial-state uncertainty remains calibrated as the prediction horizon increases.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--coverage-tolerance", type=float, default=5.0, help="Allowed absolute coverage error from 95 percent, in percentage points.")
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
                f"{day:%Y-%m-%d} | sigma={daily['sigma']:.6f} | "
                f"joint={daily['observed_joint_coverage_pct']:.2f}% | "
                f"oracle-I/init={daily['oracle_I_initial_coverage_pct']:.2f}% | "
                f"oracle-state/init={daily['oracle_state_initial_coverage_pct']:.2f}%"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if args.fail_fast: raise
            skipped.append((day, str(exc))); print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not daily_rows: raise RuntimeError("No days in the requested range could be analysed.")
    daily = pd.DataFrame(daily_rows).sort_values("day").reset_index(drop=True)
    hourly = pd.concat(hourly_frames, ignore_index=True).sort_values(["day", "forecast_hour"]).reset_index(drop=True)
    horizon = horizon_summary(hourly, tolerance_pct_points=args.coverage_tolerance)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_initial_state_horizon.csv", index=False)
    hourly.to_csv(output / "hourly_initial_state_horizon.csv", index=False)
    horizon.to_csv(output / "horizon_coverage_summary.csv", index=False)
    calibration_table(horizon).to_csv(output / "calibration_windows.csv", index=False)
    if skipped: pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)
    save_plot(horizon, output / "initial_state_horizon_coverage.png", tolerance_pct_points=args.coverage_tolerance)
    print_summary(daily, horizon, skipped, args.coverage_tolerance)
    print(); print(f"output={output}")


if __name__ == "__main__": main()
