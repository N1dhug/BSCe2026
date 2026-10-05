from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.exe.run_model.wm_4.analyze_observation_uncertainty import DEFAULT_WEEKS, _stack_days, corresponding_days, uncertainty_frame
from src.exe.run_model.wm_4.fit_daily import DAY_HOURS, aligned_wm4_data, day_start
from src.exe.run_model.wm_4.predict_all_days_optimized import fit_history_optimized, historical_optimized_state
from src.exe.run_model.wm_4.predict_daily import prediction_setup
from src.params.wm_4 import time_diffusion_qp


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "uncertainty" / "time_diffusion_qp"


def fit_prediction_day(frame, prediction_day, weeks=DEFAULT_WEEKS, steps_per_hour=60, max_state_iter=50, state_tol=1e-8):
    prediction_day, weeks, steps_per_hour = day_start(prediction_day), int(weeks), int(steps_per_hour)
    if weeks < 2: raise ValueError("weeks must be at least 2.")
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2.")

    history_days = corresponding_days(prediction_day, weeks)
    empirical = uncertainty_frame(_stack_days(frame, history_days))
    hours = empirical["hour"].to_numpy(dtype=float)
    observed_variance = empirical["var_I"].to_numpy(dtype=float)
    observed_std = empirical["s_I"].to_numpy(dtype=float)

    output_model, _, fit, fit_model = fit_history_optimized(
        frame, prediction_day, weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )

    _, actual, U, Time, prediction_model, fine_index = prediction_setup(frame, prediction_day, output_model, steps_per_hour)
    init = historical_optimized_state(fit)
    prediction_state = prediction_model.deterministic_solver(init, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})

    hourly_index = np.arange(DAY_HOURS, dtype=int) * steps_per_hour
    diffusion_time = np.asarray(prediction_state["Time"], dtype=float)
    process_kernel = prediction_model.process_variance_kernel({"r": fit["r"]}, hourly_index)
    fit_mask = np.arange(DAY_HOURS, dtype=int) > 0
    qp = time_diffusion_qp(prediction_model).fit(
        process_kernel, diffusion_time, observed_variance, fit_mask=fit_mask,
    )

    fine_diffusion = time_diffusion_qp(prediction_model).evaluate(diffusion_time, qp["beta"])
    hourly_variance = qp["propagated_variance"]
    hourly_std = qp["propagated_std"]

    unit_q = np.ones(diffusion_time.size, dtype=float)
    constant_kernel = process_kernel @ unit_q
    constant_design = constant_kernel[fit_mask, None]
    constant_beta = prediction_model._least_squares_qp(
        constant_design, observed_variance[fit_mask],
        lb=np.array([0.0], dtype=float), ridge=prediction_model._FIT_RIDGE,
    )
    constant_q = float(constant_beta[0])
    constant_propagated_variance = np.maximum(constant_kernel * constant_q, 0.0)
    constant_propagated_std = np.sqrt(constant_propagated_variance)
    variance_rmse_constant = float(np.sqrt(np.mean((constant_propagated_variance[fit_mask] - observed_variance[fit_mask]) ** 2)))
    std_rmse_constant = float(np.sqrt(np.mean((constant_propagated_std[fit_mask] - observed_std[fit_mask]) ** 2)))
    var_scale = float(np.var(observed_variance[fit_mask]))
    r2 = np.nan if var_scale <= 0.0 else 1.0 - qp["variance_mse"] / var_scale

    hourly = empirical[["hour", "var_I", "s_I"]].copy()
    hourly["fitted_propagated_variance"] = hourly_variance
    hourly["fitted_propagated_std"] = hourly_std
    hourly["constant_propagated_variance"] = constant_propagated_variance
    hourly["constant_propagated_std"] = constant_propagated_std

    return {
        "prediction_day": prediction_day, "history_days": history_days,
        "r": np.asarray(fit["r"], dtype=float), "init": np.asarray(init, dtype=float),
        "beta": qp["beta"], "variance_baseline": qp["variance_baseline"],
        "variance_amplitude_1": qp["variance_amplitude_1"], "phase_1_h": qp["phase_1_h"],
        "variance_amplitude_2": qp["variance_amplitude_2"], "phase_2_h": qp["phase_2_h"],
        "variance_rmse": qp["variance_rmse"], "std_rmse": qp["std_rmse"], "std_mae": qp["std_mae"], "r2": r2,
        "constant_q": constant_q,
        "constant_variance_rmse": variance_rmse_constant, "constant_std_rmse": std_rmse_constant,
        "hourly": hourly, "prediction_state": prediction_state,
        "fine_index": fine_index, "fine_variance": fine_diffusion["variance"], "fine_g": fine_diffusion["std"],
        "actual": actual,
    }


def save_plot(result, path):
    hourly = result["hourly"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(hourly["hour"], hourly["var_I"], marker="o", label="Observed 5-week hourly variance $s_I^2$")
    axes[0].plot(hourly["hour"], hourly["fitted_propagated_variance"], marker="o", label="WM4 propagated variance from double-harmonic $q(t)$")
    axes[0].plot(hourly["hour"], hourly["constant_propagated_variance"], linestyle="--", label="WM4 propagated variance from constant $q$")
    axes[0].set_ylabel("Variance")
    axes[0].grid(True, alpha=0.25); axes[0].legend()

    axes[1].plot(hourly["hour"], hourly["s_I"], marker="o", label="Observed 5-week $s_I$")
    axes[1].plot(hourly["hour"], hourly["fitted_propagated_std"], marker="o", label="WM4 propagated std from double-harmonic $g(t)$")
    axes[1].plot(hourly["hour"], hourly["constant_propagated_std"], linestyle="--", label="WM4 propagated std from constant $g$")
    axes[1].set_xlabel("Hour of day"); axes[1].set_ylabel("Standard deviation / diffusion")
    axes[1].set_xticks(np.arange(0, DAY_HOURS, 2))
    axes[1].grid(True, alpha=0.25); axes[1].legend()

    fig.suptitle(f"WM4 local double-harmonic time diffusion QP | prediction {result['prediction_day']:%Y-%m-%d}")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def print_result(result, plot_path, csv_path):
    r, beta = result["r"], result["beta"]
    print(f"prediction day={result['prediction_day']:%Y-%m-%d}")
    print("corresponding weekdays=" + ", ".join(day.strftime("%Y-%m-%d") for day in result["history_days"]))
    print(f"deterministic r=[{r[0]:.8f}, {r[1]:.8f}, {r[2]:.8f}]")
    print()
    print("local convex time-diffusion QP")
    print("q(t)=g(t)^2=beta_0+beta_c1*cos(wt)+beta_s1*sin(wt)+beta_c2*cos(2wt)+beta_s2*sin(2wt)")
    print("w=2*pi/24 h; q(t)>=0 imposed over the full prediction-day numerical grid")
    print("The QP fits WM4-propagated hourly Var(I), not q(t) itself, to the empirical hourly variance.")
    print(f"beta_0={beta[0]:.10f}")
    print(f"beta_c1={beta[1]:.10f}, beta_s1={beta[2]:.10f}")
    print(f"beta_c2={beta[3]:.10f}, beta_s2={beta[4]:.10f}")
    print(f"first harmonic variance amplitude={result['variance_amplitude_1']:.10f}, phase={result['phase_1_h']:.6f} h")
    print(f"second harmonic variance amplitude={result['variance_amplitude_2']:.10f}, phase={result['phase_2_h']:.6f} h")
    print(f"variance RMSE={result['variance_rmse']:.10f}, R2={result['r2']:.8f}")
    print(f"std RMSE after sqrt={result['std_rmse']:.10f}, MAE={result['std_mae']:.10f}")
    print(f"constant process variance q={result['constant_q']:.10f}, variance RMSE={result['constant_variance_rmse']:.10f}")
    print(f"constant diffusion g={np.sqrt(max(result['constant_q'], 0.0)):.10f}, std RMSE={result['constant_std_rmse']:.10f}")
    print()
    print("Hours 1-23 are fitted to the empirical hourly variances; hour 0 is excluded because P0=0 gives zero process variance initially.")
    print("The resulting additive time-only Ito diffusion has zero mean, so E[I_stochastic]=I_deterministic for the same initial mean.")
    print(f"hourly={csv_path}")
    print(f"plot={plot_path}")


def main():
    parser = ArgumentParser(description="Fit one prediction day's strictly time-dependent double-harmonic WM4 diffusion variance by convex QP.")
    parser.add_argument("--day", default="2022-08-22")
    parser.add_argument("--weeks", type=int, default=DEFAULT_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    result = fit_prediction_day(
        frame, args.day, args.weeks, args.steps_per_hour,
        max_state_iter=args.max_state_iter, state_tol=args.state_tol,
    )
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    stem = f"{result['prediction_day']:%Y-%m-%d}_{int(args.weeks)}_weekdays"
    csv_path = output / f"{stem}_hourly.csv"
    plot_path = output / f"{stem}.png"
    result["hourly"].to_csv(csv_path, index=False)
    save_plot(result, plot_path)
    print_result(result, plot_path, csv_path)


if __name__ == "__main__": main()
