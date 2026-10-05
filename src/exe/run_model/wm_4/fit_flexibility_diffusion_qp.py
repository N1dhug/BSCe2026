from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_observation_uncertainty import DEFAULT_WEEKS, _stack_days, corresponding_days, uncertainty_frame
from src.exe.run_model.wm_4.fit_daily import DAY_HOURS, aligned_wm4_data, day_start
from src.exe.run_model.wm_4.predict_all_days_optimized import fit_history_optimized, historical_optimized_state
from src.exe.run_model.wm_4.predict_daily import prediction_setup
from src.params.wm_4 import flexibility_diffusion_qp


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "uncertainty" / "flexibility_diffusion_qp"


def fit_prediction_day(frame, prediction_day, weeks=DEFAULT_WEEKS, steps_per_hour=60, max_state_iter=50, state_tol=1e-8):
    prediction_day, weeks, steps_per_hour = day_start(prediction_day), int(weeks), int(steps_per_hour)
    if weeks < 2: raise ValueError("weeks must be at least 2.")
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2.")

    history_days = corresponding_days(prediction_day, weeks)
    empirical = uncertainty_frame(_stack_days(frame, history_days))
    observed_std = empirical["s_I"].to_numpy(dtype=float)

    output_model, _, fit, fit_model = fit_history_optimized(
        frame, prediction_day, weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )
    hourly_index = np.arange(DAY_HOURS, dtype=int) * steps_per_hour
    historical_states = fit["deterministic_states"]
    historical_F = np.vstack([np.asarray(state["F"], dtype=float)[hourly_index] for state in historical_states])
    mean_F = np.mean(historical_F, axis=0)

    qp = flexibility_diffusion_qp(fit_model).fit(mean_F, observed_std)
    sigma_0, sigma_F = qp["sigma_0"], qp["sigma_F"]

    _, actual, U, Time, prediction_model, fine_index = prediction_setup(frame, prediction_day, output_model, steps_per_hour)
    init = historical_optimized_state(fit)
    prediction_state = prediction_model.deterministic_solver(init, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})
    prediction_F = np.asarray(prediction_state["F"], dtype=float)
    if np.any(prediction_F < -1.0) or np.any(prediction_F > 1.0): raise ValueError("Prediction-day deterministic F left [-1, 1], so the constrained H4 diffusion guarantee no longer applies.")
    prediction_g = sigma_0 + sigma_F * prediction_F
    prediction_g_hourly = prediction_g[hourly_index]

    constant_sigma = float(np.clip(np.mean(observed_std), 0.0, 1.0))
    constant_residual = constant_sigma - observed_std
    constant_rmse = float(np.sqrt(np.mean(constant_residual ** 2)))
    variance = float(np.var(observed_std))
    r2 = np.nan if variance <= 0.0 else 1.0 - qp["mse"] / variance

    hourly = pd.DataFrame({
        "hour": np.arange(DAY_HOURS, dtype=int),
        "observed_s_I": observed_std,
        "historical_mean_F": mean_F,
        "fitted_g_H4": qp["fitted_std"],
        "prediction_F": prediction_F[hourly_index],
        "prediction_g_H4": prediction_g_hourly,
    })
    return {
        "prediction_day": prediction_day, "history_days": history_days,
        "r": np.asarray(fit["r"], dtype=float), "init": np.asarray(init, dtype=float),
        "sigma_0": sigma_0, "sigma_F": sigma_F,
        "qp_mse": qp["mse"], "qp_rmse": qp["rmse"], "qp_mae": qp["mae"], "r2": r2,
        "constant_sigma": constant_sigma, "constant_rmse": constant_rmse,
        "hourly": hourly, "prediction_state": prediction_state,
        "fine_index": fine_index, "actual": actual,
    }


def save_plot(result, path):
    hourly = result["hourly"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(hourly["hour"], hourly["observed_s_I"], marker="o", label="Observed 5-week $s_I$")
    axes[0].plot(hourly["hour"], hourly["fitted_g_H4"], marker="o", label="QP fit $g=\\sigma_0+\\sigma_F\\bar F$")
    axes[0].plot(hourly["hour"], hourly["prediction_g_H4"], marker="o", label="Prediction-day $g(F)$")
    axes[0].axhline(result["constant_sigma"], linestyle="--", label="Constant-only fit")
    axes[0].set_ylabel("Standard deviation / diffusion")
    axes[0].grid(True, alpha=0.25); axes[0].legend()

    axes[1].plot(hourly["hour"], hourly["historical_mean_F"], marker="o", label="Historical mean fitted F")
    axes[1].plot(hourly["hour"], hourly["prediction_F"], marker="o", label="Prediction-day deterministic F")
    axes[1].set_xlabel("Hour of day"); axes[1].set_ylabel("Flexibility F")
    axes[1].set_xticks(np.arange(0, DAY_HOURS, 2))
    axes[1].grid(True, alpha=0.25); axes[1].legend()

    fig.suptitle(f"WM4 local flexibility diffusion QP | prediction {result['prediction_day']:%Y-%m-%d}")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def print_result(result, plot_path, csv_path):
    r = result["r"]
    print(f"prediction day={result['prediction_day']:%Y-%m-%d}")
    print("corresponding weekdays=" + ", ".join(day.strftime("%Y-%m-%d") for day in result["history_days"]))
    print(f"deterministic r=[{r[0]:.8f}, {r[1]:.8f}, {r[2]:.8f}]")
    print()
    print("H4 local convex QP")
    print("g(F) = sigma_0 + sigma_F * F")
    print("constraints: 0 <= sigma_F <= sigma_0 <= 1")
    print(f"sigma_0={result['sigma_0']:.8f}")
    print(f"sigma_F={result['sigma_F']:.8f}")
    print(f"sigma_F/sigma_0={(result['sigma_F'] / result['sigma_0']) if result['sigma_0'] > 0.0 else np.nan:.8f}")
    print(f"QP RMSE={result['qp_rmse']:.8f}, MAE={result['qp_mae']:.8f}, R2={result['r2']:.8f}")
    print(f"constant-only sigma={result['constant_sigma']:.8f}, RMSE={result['constant_rmse']:.8f}")
    print()
    print("The QP target is the empirical hourly s_I from the corresponding weekdays; O is not used in s_I.")
    print("The additive Ito diffusion has zero mean, so E[I_stochastic] follows the deterministic WM4 trajectory for the same initial mean.")
    print(f"hourly={csv_path}")
    print(f"plot={plot_path}")


def main():
    parser = ArgumentParser(description="Fit one prediction day's WM4 flexibility diffusion g(F)=sigma_0+sigma_F*F by convex QP from previous corresponding weekdays.")
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
