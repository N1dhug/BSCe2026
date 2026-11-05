from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.exe.run_model.wm_4.analyze_observation_uncertainty import DEFAULT_WEEKS, _stack_days, corresponding_days, uncertainty_frame
from src.exe.run_model.wm_4.fit_daily import DAY_HOURS, aligned_wm4_data, day_start, selected_day
from src.exe.run_model.wm_4.predict_all_days_optimized import fit_history_optimized, historical_optimized_state
from src.exe.run_model.wm_4.predict_daily import prediction_setup
from src.params.wm_4 import constant_diffusion_qp


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "uncertainty" / "constant_diffusion_qp"


def fit_prediction_day(frame, prediction_day, weeks=DEFAULT_WEEKS, steps_per_hour=60, max_state_iter=50, state_tol=1e-8):
    prediction_day, weeks, steps_per_hour = day_start(prediction_day), int(weeks), int(steps_per_hour)
    if weeks < 2: raise ValueError("weeks must be at least 2.")
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2.")

    history_days = corresponding_days(prediction_day, weeks)
    empirical = uncertainty_frame(_stack_days(frame, history_days))
    observed_variance = empirical["var_I"].to_numpy(dtype=float)

    output_model, _, fit, _ = fit_history_optimized(
        frame, prediction_day, weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )

    _, actual, U, Time, prediction_model, fine_index = prediction_setup(frame, prediction_day, output_model, steps_per_hour)
    init = historical_optimized_state(fit)
    prediction_state = prediction_model.deterministic_solver(init, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})

    hourly_index = np.arange(DAY_HOURS, dtype=int) * steps_per_hour
    process_kernel = prediction_model.process_variance_kernel({"r": fit["r"]}, hourly_index)

    # Historical demand errors separate the constant observation floor from
    # the Brownian variance accumulated through the R-state dynamics.
    residual_days = []
    for day, historical_state in zip(fit["training_days"], fit["deterministic_states"]):
        _, measured = selected_day(frame, day)
        prediction = np.asarray(historical_state["I"], dtype=float)[hourly_index]
        residual_days.append(measured["I"].to_numpy(dtype=float) - prediction)
    demand_residuals = np.vstack(residual_days)

    P0 = np.zeros((2, 2), dtype=float)
    initial_variance = np.zeros(DAY_HOURS, dtype=float)

    fit_mask = np.ones(DAY_HOURS, dtype=bool)
    qp = constant_diffusion_qp(prediction_model).fit(
        process_kernel, observed_variance,
        initial_variance=initial_variance, fit_mask=fit_mask, residuals=demand_residuals,
    )
    fine_diffusion = constant_diffusion_qp.evaluate(prediction_model.t_steps, qp["q"], qp["observation_q"])

    observed_variance = qp["observed_variance"]
    var_scale = float(np.var(observed_variance[fit_mask]))
    r2 = np.nan if var_scale <= 0.0 else 1.0 - qp["variance_mse"] / var_scale

    hourly = empirical[["hour", "var_I", "s_I"]].copy()
    hourly["demand_residual_variance"] = observed_variance
    hourly["demand_residual_std"] = np.sqrt(observed_variance)
    hourly["observation_variance"] = qp["observation_variance"]
    hourly["initial_variance"] = initial_variance
    hourly["fitted_process_variance"] = qp["process_variance"]
    hourly["fitted_propagated_variance"] = qp["propagated_variance"]
    hourly["fitted_propagated_std"] = qp["propagated_std"]

    return {
        "prediction_day": prediction_day, "history_days": history_days,
        "r": np.asarray(fit["r"], dtype=float), "init": np.asarray(init, dtype=float),
        "q": qp["q"], "sigma": qp["sigma"],
        "observation_q": qp["observation_q"], "observation_sigma": qp["observation_sigma"],
        "demand_residuals": demand_residuals,
        "variance_rmse": qp["variance_rmse"], "std_rmse": qp["std_rmse"], "std_mae": qp["std_mae"], "r2": r2,
        "P0": P0, "initial_variance": initial_variance,
        "hourly": hourly, "prediction_state": prediction_state,
        "training_days": list(fit["training_days"]),
        "training_states": list(fit["deterministic_states"]),
        "fine_index": fine_index, "fine_variance": fine_diffusion["variance"], "fine_g": fine_diffusion["std"],
        "fine_observation_std": fine_diffusion["observation_std"],
        "actual": actual,
    }


def save_plot(result, path):
    hourly = result["hourly"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(hourly["hour"], hourly["demand_residual_variance"], marker="o", label="Observed demand prediction MSE by hour")
    axes[0].axhline(result["observation_q"], linestyle=":", label="Constant demand observation variance")
    axes[0].plot(hourly["hour"], hourly["fitted_propagated_variance"], marker="o", label="WM4 propagated variance from constant $q$")
    axes[0].set_ylabel("Variance")
    axes[0].grid(True, alpha=0.25); axes[0].legend()

    axes[1].plot(hourly["hour"], hourly["demand_residual_std"], marker="o", label="Observed demand residual RMS")
    axes[1].plot(hourly["hour"], hourly["fitted_propagated_std"], marker="o", label="WM4 propagated std")
    axes[1].axhline(result["observation_sigma"], linestyle=":", label=f"Observation error std={result['observation_sigma']:.4f}")
    axes[1].set_xlabel("Hour of day"); axes[1].set_ylabel("Standard deviation / diffusion")
    axes[1].set_xticks(np.arange(0, DAY_HOURS, 2))
    axes[1].grid(True, alpha=0.25); axes[1].legend()

    fig.suptitle(f"WM4 local constant diffusion QP | prediction {result['prediction_day']:%Y-%m-%d}")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def print_result(result, plot_path, csv_path):
    r = result["r"]
    print(f"prediction day={result['prediction_day']:%Y-%m-%d}")
    print("corresponding weekdays=" + ", ".join(day.strftime("%Y-%m-%d") for day in result["history_days"]))
    print(f"deterministic r=[{r[0]:.8f}, {r[1]:.8f}, {r[2]:.8f}]")
    print()
    print("local convex constant-diffusion QP")
    print("g(t)=sigma, q=sigma^2")
    print("The QP fits mean squared demand residuals to constant observation variance plus accumulated process variance.")
    print(f"q={result['q']:.10f}")
    print(f"sigma={result['sigma']:.10f}")
    print(f"observation variance={result['observation_q']:.10f}, observation std={result['observation_sigma']:.10f}")
    print(f"variance RMSE={result['variance_rmse']:.10f}, R2={result['r2']:.8f}")
    print(f"std RMSE after sqrt={result['std_rmse']:.10f}, MAE={result['std_mae']:.10f}")
    print()
    print("P0=[[0,0],[0,0]]: the constant diffusion fit is conditional on a known/oracle initial state.")
    print("All 24 hourly residual moments are included; hour 0 identifies the observation error floor when P0=0.")
    print("Historical fitted states may absorb observation error; verify on held-out demand residuals.")
    print("The additive constant Ito diffusion has zero mean, so E[I_stochastic]=I_deterministic for the same initial mean.")
    print(f"hourly={csv_path}")
    print(f"plot={plot_path}")


def main():
    parser = ArgumentParser(description="Fit one prediction day's constant WM4 diffusion by convex QP.")
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
