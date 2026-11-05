from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from src.exe.run_model.wm_4.analyze_observation_uncertainty import _stack_days, uncertainty_frame
from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY, fit_history_optimized
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup
from src.exe.run_model.wm_4.fit_constant_diffusion_qp import historical_demand_residuals
from src.params.wm_4 import constant_diffusion_qp, optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_uncertainty"
Z_95 = float(norm.ppf(0.975))


def state_interval_coverage(mean, covariance, oracle, confidence=0.95):
    mean = np.asarray(mean, dtype=float).reshape(2)
    covariance = np.asarray(covariance, dtype=float).reshape(2, 2)
    oracle = np.asarray(oracle, dtype=float).reshape(2)
    covariance = 0.5 * (covariance + covariance.T)

    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z = float(norm.ppf(0.5 + 0.5 * confidence))
    lower, upper = mean - z * std, mean + z * std
    marginal = (oracle >= lower) & (oracle <= upper)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tol = 1e-10 * scale
    positive = eigenvalues > tol
    rank = int(np.count_nonzero(positive))
    delta = oracle - mean

    if rank == 0:
        mahalanobis2 = 0.0 if np.linalg.norm(delta) <= 1e-10 else np.inf
        threshold = 0.0
        joint = bool(np.isfinite(mahalanobis2))
    else:
        projected = eigenvectors.T @ delta
        null_residual = projected[~positive]
        supported = bool(np.linalg.norm(null_residual) <= 1e-8 * max(1.0, np.linalg.norm(delta)))
        mahalanobis2 = float(np.sum((projected[positive] ** 2) / eigenvalues[positive]))
        threshold = float(chi2.ppf(confidence, df=rank))
        joint = bool(supported and mahalanobis2 <= threshold)

    return {
        "lower": lower, "upper": upper, "std": std,
        "F_covered": bool(marginal[0]), "R_covered": bool(marginal[1]),
        "marginal_box_covered": bool(np.all(marginal)),
        "joint_ellipse_covered": joint,
        "mahalanobis2": mahalanobis2, "chi2_threshold": threshold, "cov_rank": rank,
    }


def fit_day(frame, day, baseline_weeks, steps_per_hour, max_state_iter=50, state_tol=1e-8):
    day = day_start(day)
    output_model, output_fit, fit, _ = fit_history_optimized(
        frame, day, baseline_weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )
    _, actual, U, Time, model, fine_index = prediction_setup(frame, day, output_model, steps_per_hour)

    state_estimate = optimized_qp_params.estimate_prediction_initial_state(fit["init_cond"])
    init_mean = np.asarray(state_estimate["x_hat"], dtype=float)
    P0 = np.asarray(state_estimate["covariance"], dtype=float)
    state = model.deterministic_solver(init_mean, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})

    prediction_data, _, _, _ = model_input(frame, day, steps_per_hour, 0, output_model)
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(prediction_data, fit["r"])
    oracle_init = np.asarray(oracle["init_cond"], dtype=float)
    oracle_coverage = state_interval_coverage(init_mean, P0, oracle_init)

    empirical = uncertainty_frame(_stack_days(frame, fit["training_days"]))
    observed_variance = empirical["var_I"].to_numpy(dtype=float)

    hourly_index = np.arange(DAY_HOURS, dtype=int) * int(steps_per_hour)
    process_kernel = model.process_variance_kernel({"r": fit["r"]}, hourly_index)
    residuals = historical_demand_residuals(frame, fit["training_days"], fit["deterministic_states"], steps_per_hour)
    diffusion_fit = constant_diffusion_qp(model).fit(
        process_kernel, observed_variance,
        initial_variance=np.zeros(DAY_HOURS, dtype=float),
        fit_mask=np.ones(DAY_HOURS, dtype=bool), residuals=residuals,
    )

    sensitivity = model.sensitivity(fit["r"])["input"][hourly_index]
    initial_variance = np.einsum("ni,ij,nj->n", sensitivity, P0, sensitivity)
    initial_variance = np.maximum(initial_variance, 0.0)
    process_variance = np.asarray(diffusion_fit["process_variance"], dtype=float)
    observation_variance = np.asarray(diffusion_fit["observation_variance"], dtype=float)
    joint_variance = np.maximum(initial_variance + process_variance + observation_variance, 0.0)
    joint_std = np.sqrt(joint_variance)

    observed = actual["I"].to_numpy(dtype=float)
    predicted = np.asarray(state["I"], dtype=float)[hourly_index]
    lower = predicted - Z_95 * joint_std
    upper = predicted + Z_95 * joint_std
    covered = (observed >= lower) & (observed <= upper)

    process_std = np.sqrt(np.maximum(process_variance, 0.0))
    process_lower = predicted - Z_95 * process_std
    process_upper = predicted + Z_95 * process_std
    process_covered = (observed >= process_lower) & (observed <= process_upper)

    metrics = deviation_metrics(observed, predicted)
    r = np.asarray(fit["r"], dtype=float)

    daily = {
        "day": day, "weekday": day.day_name(),
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "sigma": float(diffusion_fit["sigma"]), "diffusion_q": float(diffusion_fit["q"]),
        "observation_std": float(diffusion_fit["observation_sigma"]), "observation_q": float(diffusion_fit["observation_q"]),
        "diffusion_variance_rmse": float(diffusion_fit["variance_rmse"]),
        "F0_mean": init_mean[0], "R0_mean": init_mean[1],
        "F0_std": oracle_coverage["std"][0], "R0_std": oracle_coverage["std"][1],
        "oracle_F0": oracle_init[0], "oracle_R0": oracle_init[1],
        "F0_ci95_lower": oracle_coverage["lower"][0], "F0_ci95_upper": oracle_coverage["upper"][0],
        "R0_ci95_lower": oracle_coverage["lower"][1], "R0_ci95_upper": oracle_coverage["upper"][1],
        "oracle_F0_in_95": oracle_coverage["F_covered"],
        "oracle_R0_in_95": oracle_coverage["R_covered"],
        "oracle_state_in_marginal_95_box": oracle_coverage["marginal_box_covered"],
        "oracle_state_in_joint_95_ellipse": oracle_coverage["joint_ellipse_covered"],
        "oracle_state_mahalanobis2": oracle_coverage["mahalanobis2"],
        "oracle_state_chi2_95": oracle_coverage["chi2_threshold"],
        "oracle_state_cov_rank": oracle_coverage["cov_rank"],
        "coverage_95": float(np.mean(covered)),
        "process_only_coverage_95": float(np.mean(process_covered)),
        "full_day_covered_95": bool(np.all(covered)),
        "mean_initial_std": float(np.mean(np.sqrt(initial_variance))),
        "mean_process_std": float(np.mean(process_std)),
        "mean_joint_std": float(np.mean(joint_std)),
        "wm4_mse": metrics["mse"], "wm4_rmse": metrics["rmse"],
        "wm4_mae": metrics["mae"], "wm4_bias": metrics["bias"],
        "oracle_fit_mse": float(oracle["loss"]),
        "O_fit_mse": float(output_fit["mse"]),
        "state_iterations": int(fit["state_iterations"]),
    }

    hourly = pd.DataFrame({
        "day": day,
        "I_observed": observed, "I_mean": predicted,
        "ci95_lower": lower, "ci95_upper": upper, "covered_95": covered,
        "joint_std": joint_std,
        "initial_variance": initial_variance,
        "process_variance": process_variance,
        "observation_variance": observation_variance,
        "joint_variance": joint_variance,
        "process_only_ci95_lower": process_lower,
        "process_only_ci95_upper": process_upper,
        "process_only_covered_95": process_covered,
        "O": np.asarray(state["O"], dtype=float)[hourly_index],
        "U": np.asarray(state["U"], dtype=float)[hourly_index],
        "F": np.asarray(state["F"], dtype=float)[hourly_index],
        "R": np.asarray(state["R"], dtype=float)[hourly_index],
    }, index=actual.index)
    hourly.index.name = "time"

    training_states = np.asarray(fit["init_cond"], dtype=float)
    training = pd.DataFrame({
        "prediction_day": day,
        "training_day": fit["training_days"],
        "F0": training_states[:, 0],
        "R0": training_states[:, 1],
    })
    return daily, hourly, training


def run_all_days(frame, start=START_DAY, end=END_DAY, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, fail_fast=False, max_state_iter=50, state_tol=1e-8):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")

    daily_rows, hourly_frames, training_frames, skipped = [], [], [], []
    for day in pd.date_range(start, end, freq="D"):
        try:
            daily, hourly, training = fit_day(
                frame, day, baseline_weeks, steps_per_hour,
                max_state_iter=max_state_iter, state_tol=state_tol,
            )
            daily_rows.append(daily); hourly_frames.append(hourly); training_frames.append(training)
            print(
                f"{day:%Y-%m-%d} | sigma={daily['sigma']:.6f} | "
                f"coverage95={100.0 * daily['coverage_95']:.2f}% | "
                f"oracle joint95={daily['oracle_state_in_joint_95_ellipse']} | "
                f"oracle marginal95={daily['oracle_state_in_marginal_95_box']}"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc)))
            print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not daily_rows: raise RuntimeError("No days in the requested range could be fitted.")
    daily = pd.DataFrame(daily_rows).sort_values("day").reset_index(drop=True)
    hourly = pd.concat(hourly_frames).sort_index()
    training = pd.concat(training_frames, ignore_index=True)
    return daily, hourly, training, skipped


def save_plot(hourly, daily, path):
    fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=False)

    axes[0].plot(hourly.index, hourly["I_observed"], label="Observed I", linewidth=0.9)
    axes[0].plot(hourly.index, hourly["I_mean"], label="WM4 mean prediction", linewidth=0.9)
    axes[0].fill_between(hourly.index, hourly["ci95_lower"], hourly["ci95_upper"], alpha=0.2, label="Joint 95% interval")
    axes[0].set_ylabel("Normalized input")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)

    axes[1].plot(daily["day"], 100.0 * daily["coverage_95"], label="Daily hourly coverage")
    axes[1].axhline(95.0, linestyle="--", label="95% target")
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].legend(); axes[1].grid(True, alpha=0.25)

    axes[2].step(daily["day"], daily["oracle_state_in_joint_95_ellipse"].astype(int), where="mid", label="Oracle state in joint 95% ellipse")
    axes[2].step(daily["day"], daily["oracle_state_in_marginal_95_box"].astype(int), where="mid", label="Oracle state in marginal 95% box", alpha=0.8)
    axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(["outside", "inside"])
    axes[2].set_xlabel("Prediction day"); axes[2].set_ylabel("Oracle state")
    axes[2].legend(); axes[2].grid(True, alpha=0.25)

    fig.suptitle("WM4 all-days uncertainty: historical initial-state covariance + constant process diffusion")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def print_summary(hourly, daily, skipped):
    covered = hourly["covered_95"].to_numpy(dtype=bool)
    process_covered = hourly["process_only_covered_95"].to_numpy(dtype=bool)
    oracle_joint = daily["oracle_state_in_joint_95_ellipse"].to_numpy(dtype=bool)
    oracle_box = daily["oracle_state_in_marginal_95_box"].to_numpy(dtype=bool)
    oracle_F = daily["oracle_F0_in_95"].to_numpy(dtype=bool)
    oracle_R = daily["oracle_R0_in_95"].to_numpy(dtype=bool)
    full_days = daily["full_day_covered_95"].to_numpy(dtype=bool)

    print()
    print(f"days fitted={len(daily)}, skipped={len(skipped)}, hourly observations={len(hourly)}")
    print(f"joint 95% input coverage={100.0 * np.mean(covered):.2f}% ({np.count_nonzero(covered)}/{covered.size})")
    print(f"process-only 95% input coverage={100.0 * np.mean(process_covered):.2f}% ({np.count_nonzero(process_covered)}/{process_covered.size})")
    print(f"days with all 24 observations inside joint 95% interval={100.0 * np.mean(full_days):.2f}% ({np.count_nonzero(full_days)}/{full_days.size})")
    print()
    print(f"oracle state inside joint 95% ellipse={100.0 * np.mean(oracle_joint):.2f}% ({np.count_nonzero(oracle_joint)}/{oracle_joint.size})")
    print(f"oracle state inside both marginal 95% intervals={100.0 * np.mean(oracle_box):.2f}% ({np.count_nonzero(oracle_box)}/{oracle_box.size})")
    print(f"oracle F0 inside marginal 95% interval={100.0 * np.mean(oracle_F):.2f}% ({np.count_nonzero(oracle_F)}/{oracle_F.size})")
    print(f"oracle R0 inside marginal 95% interval={100.0 * np.mean(oracle_R):.2f}% ({np.count_nonzero(oracle_R)}/{oracle_R.size})")
    print()
    print(f"mean sigma={daily['sigma'].mean():.8f}")
    print(f"mean initial-state propagated std={daily['mean_initial_std'].mean():.8f}")
    print(f"mean process std={daily['mean_process_std'].mean():.8f}")
    print(f"mean joint std={daily['mean_joint_std'].mean():.8f}")

    if skipped:
        print(); print("skipped days:")
        for day, reason in skipped: print(f"{day:%Y-%m-%d}: {reason}")


def main():
    parser = ArgumentParser(description="Evaluate all-day WM4 predictions with constant process diffusion and historical initial-state uncertainty.")
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
    daily, hourly, training, skipped = run_all_days(
        frame, args.start_day, args.end_day, args.baseline_weeks, args.steps_per_hour,
        fail_fast=args.fail_fast, max_state_iter=args.max_state_iter, state_tol=args.state_tol,
    )

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_uncertainty.csv", index=False)
    hourly.to_csv(output / "hourly_uncertainty.csv")
    training.to_csv(output / "optimized_training_states.csv", index=False)
    if skipped:
        pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)
    save_plot(hourly, daily, output / "all_days_joint_uncertainty.png")
    print_summary(hourly, daily, skipped)


if __name__ == "__main__": main()
