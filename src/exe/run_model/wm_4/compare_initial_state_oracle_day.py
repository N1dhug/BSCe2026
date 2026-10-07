from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.exe.run_model.wm_4.analyze_initial_state_uncertainty_horizon import process_covariance
from src.exe.run_model.wm_4.analyze_observation_uncertainty import _stack_days, uncertainty_frame
from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_all_days_optimized import fit_history_optimized
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup
from src.params.wm_4 import constant_diffusion_qp, optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "uncertainty" / "initial_state_oracle_day"
CONFIDENCE = 0.95


def fit_target_day(frame, day, weeks=BASELINE_WEEKS, steps_per_hour=60, confidence=CONFIDENCE, max_state_iter=50, state_tol=1e-8):
    day, weeks, steps_per_hour = day_start(day), int(weeks), int(steps_per_hour)
    if weeks < 2: raise ValueError("weeks must be at least 2.")
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2.")

    output_model, output_fit, fit, _ = fit_history_optimized(
        frame, day, weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )
    _, actual, U, Time, model, fine_index = prediction_setup(frame, day, output_model, steps_per_hour)
    hourly_index = np.arange(DAY_HOURS, dtype=int) * steps_per_hour

    x_hat = np.asarray(fit["x_hat"], dtype=float)
    P0 = np.asarray(fit["x_cov"], dtype=float)
    historical_state = model.deterministic_solver(
        x_hat, {"U": U, "PE": U, "Time": Time, "r": fit["r"]},
    )

    prediction_data, _, _, _ = model_input(frame, day, steps_per_hour, 0, output_model)
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(prediction_data, fit["r"])
    x_oracle = np.asarray(oracle["init_cond"], dtype=float)
    oracle_state = oracle["deterministic_state"]

    empirical = uncertainty_frame(_stack_days(frame, fit["training_days"]))
    observed_variance = empirical["var_I"].to_numpy(dtype=float)

    sensitivity = model.sensitivity(fit["r"])
    J_fine = np.asarray(sensitivity["input"], dtype=float)
    initial_variance_fine = np.einsum("ni,ij,nj->n", J_fine, P0, J_fine)
    initial_variance_fine = np.maximum(initial_variance_fine, 0.0)
    initial_variance_hourly = initial_variance_fine[hourly_index]

    process_kernel = model.process_variance_kernel({"r": fit["r"]}, hourly_index)
    diffusion = constant_diffusion_qp(model).fit(
        process_kernel, observed_variance,
        initial_variance=initial_variance_hourly,
        fit_mask=np.ones(DAY_HOURS, dtype=bool),
    )

    process_cov = process_covariance(model, fit["r"], diffusion["q"])
    process_variance_fine = np.maximum(process_cov[:, 1, 1], 0.0)
    joint_variance_fine = np.maximum(initial_variance_fine + process_variance_fine, 0.0)
    joint_std_fine = np.sqrt(joint_variance_fine)

    z = float(norm.ppf(0.5 + 0.5 * confidence))
    historical_I = np.asarray(historical_state["I"], dtype=float)
    oracle_I = np.asarray(oracle_state["I"], dtype=float)
    historical_lower, historical_upper = historical_I - z * joint_std_fine, historical_I + z * joint_std_fine
    oracle_lower, oracle_upper = oracle_I - z * joint_std_fine, oracle_I + z * joint_std_fine

    r = np.asarray(fit["r"], dtype=float)
    historical_rF_F = r[0] * np.asarray(historical_state["F"], dtype=float)
    historical_rR_R = r[2] * np.asarray(historical_state["R"], dtype=float)
    oracle_rF_F = r[0] * np.asarray(oracle_state["F"], dtype=float)
    oracle_rR_R = r[2] * np.asarray(oracle_state["R"], dtype=float)

    observed = actual["I"].to_numpy(dtype=float)
    historical_hourly = historical_I[hourly_index]
    oracle_hourly = oracle_I[hourly_index]
    historical_metrics = deviation_metrics(observed, historical_hourly)
    oracle_metrics = deviation_metrics(observed, oracle_hourly)

    hourly = pd.DataFrame({
        "time": actual.index,
        "I_observed": observed,
        "I_historical": historical_hourly,
        "I_oracle": oracle_hourly,
        "historical_ci95_lower": historical_lower[hourly_index],
        "historical_ci95_upper": historical_upper[hourly_index],
        "oracle_ci95_lower": oracle_lower[hourly_index],
        "oracle_ci95_upper": oracle_upper[hourly_index],
        "joint_std_I": joint_std_fine[hourly_index],
        "initial_std_I": np.sqrt(initial_variance_fine[hourly_index]),
        "process_std_I": np.sqrt(process_variance_fine[hourly_index]),
        "historical_F": np.asarray(historical_state["F"], dtype=float)[hourly_index],
        "historical_R": np.asarray(historical_state["R"], dtype=float)[hourly_index],
        "oracle_F": np.asarray(oracle_state["F"], dtype=float)[hourly_index],
        "oracle_R": np.asarray(oracle_state["R"], dtype=float)[hourly_index],
        "historical_rF_F": historical_rF_F[hourly_index],
        "historical_rR_R": historical_rR_R[hourly_index],
        "oracle_rF_F": oracle_rF_F[hourly_index],
        "oracle_rR_R": oracle_rR_R[hourly_index],
    })

    return {
        "day": day,
        "weeks": weeks,
        "fine_index": fine_index,
        "actual": actual,
        "historical_state": historical_state,
        "oracle_state": oracle_state,
        "historical_I": historical_I,
        "oracle_I": oracle_I,
        "historical_lower": historical_lower,
        "historical_upper": historical_upper,
        "oracle_lower": oracle_lower,
        "oracle_upper": oracle_upper,
        "joint_std_fine": joint_std_fine,
        "initial_variance_fine": initial_variance_fine,
        "process_variance_fine": process_variance_fine,
        "historical_rF_F": historical_rF_F,
        "historical_rR_R": historical_rR_R,
        "oracle_rF_F": oracle_rF_F,
        "oracle_rR_R": oracle_rR_R,
        "x_hat": x_hat,
        "x_oracle": x_oracle,
        "P0": P0,
        "r": r,
        "sigma": float(diffusion["sigma"]),
        "q": float(diffusion["q"]),
        "variance_rmse": float(diffusion["variance_rmse"]),
        "historical_metrics": historical_metrics,
        "oracle_metrics": oracle_metrics,
        "oracle_fit_mse": float(oracle["loss"]),
        "training_days": list(fit["training_days"]),
        "output_fit": output_fit,
        "hourly": hourly,
    }


def save_plot(result, path):
    time = result["fine_index"]
    actual = result["actual"]
    historical = result["historical_state"]
    oracle = result["oracle_state"]

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    axes[0].plot(actual.index, actual["I"], marker="o", linestyle="none", label="Observed I")
    axes[0].plot(time, result["historical_I"], label="Historical-state centre")
    axes[0].fill_between(time, result["historical_lower"], result["historical_upper"], alpha=0.18, label="Historical 95% joint interval")
    axes[0].plot(time, result["oracle_I"], label="Oracle-state centre")
    axes[0].fill_between(time, result["oracle_lower"], result["oracle_upper"], alpha=0.12, label="Oracle 95% joint interval")
    axes[0].plot(time, historical["O"], linestyle="--", alpha=0.75, label="Expected output O")
    axes[0].set_ylabel("Normalized input")
    axes[0].legend(ncol=2)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time, historical["F"], label="Historical F")
    axes[1].plot(time, oracle["F"], label="Oracle F")
    axes[1].plot(time, historical["R"], label="Historical R")
    axes[1].plot(time, oracle["R"], label="Oracle R")
    axes[1].set_ylabel("State / rate")
    axes[1].legend(ncol=2)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time, result["historical_rF_F"], label=r"Historical $r_FF$")
    axes[2].plot(time, result["oracle_rF_F"], label=r"Oracle $r_FF$")
    axes[2].axhline(0.0, linewidth=0.8)
    axes[2].set_ylabel(r"$r_F F$")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(time, result["historical_rR_R"], label=r"Historical $r_RR$")
    axes[3].plot(time, result["oracle_rR_R"], label=r"Oracle $r_RR$")
    axes[3].axhline(0.0, linewidth=0.8)
    axes[3].set_ylabel(r"$r_R R$")
    axes[3].set_xlabel("Physical time")
    axes[3].legend()
    axes[3].grid(True, alpha=0.25)

    r = result["r"]
    fig.suptitle(
        f"WM4 initial-state comparison | {result['day']:%Y-%m-%d} | "
        f"r=[{r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}], sigma={result['sigma']:.4f}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_result(result, plot_path, csv_path):
    hist = result["historical_metrics"]
    oracle = result["oracle_metrics"]
    P0 = result["P0"]
    r = result["r"]

    print(f"prediction day={result['day']:%Y-%m-%d}, historical weekdays={result['weeks']}")
    print("training days=" + ", ".join(day.strftime("%Y-%m-%d") for day in result["training_days"]))
    print(f"r=[r_F={r[0]:.8f}, r_U={r[1]:.8f}, r_R={r[2]:.8f}]")
    print(f"historical x0=[F0={result['x_hat'][0]:.8f}, R0={result['x_hat'][1]:.8f}]")
    print(f"oracle x0=[F0={result['x_oracle'][0]:.8f}, R0={result['x_oracle'][1]:.8f}]")
    print(
        "P0=[[{:.8f}, {:.8f}], [{:.8f}, {:.8f}]]".format(
            P0[0, 0], P0[0, 1], P0[1, 0], P0[1, 1],
        )
    )
    print(f"sigma={result['sigma']:.8f}, q={result['q']:.8f}, variance_rmse={result['variance_rmse']:.8f}")
    print("Historical and oracle predictions use the same P0, diffusion, and uncertainty width.")
    print()
    print(f"{'':<12}{'MSE':>14}{'RMSE':>14}{'MAE':>14}{'bias':>14}")
    print(f"{'historical':<12}{hist['mse']:>14.10f}{hist['rmse']:>14.10f}{hist['mae']:>14.10f}{hist['bias']:>14.10f}")
    print(f"{'oracle':<12}{oracle['mse']:>14.10f}{oracle['rmse']:>14.10f}{oracle['mae']:>14.10f}{oracle['bias']:>14.10f}")
    print()
    print(f"hourly={csv_path}")
    print(f"plot={plot_path}")


def main():
    parser = ArgumentParser(description="Compare historical and oracle WM4 initial-state centres for one target day using the same observed P0 and fitted diffusion.")
    parser.add_argument("--day", default="2022-08-22")
    parser.add_argument("--weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    result = fit_target_day(
        frame, args.day, args.weeks, args.steps_per_hour,
        max_state_iter=args.max_state_iter, state_tol=args.state_tol,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{result['day']:%Y-%m-%d}_{int(args.weeks)}_weekdays"
    csv_path = output / f"{stem}_hourly.csv"
    plot_path = output / f"{stem}.png"
    result["hourly"].to_csv(csv_path, index=False)
    save_plot(result, plot_path)
    print_result(result, plot_path, csv_path)


if __name__ == "__main__": main()
