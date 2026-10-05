from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start
from src.exe.run_model.wm_4.predict_all_days_optimized import fit_history_optimized
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "initial_state_uncertainty"


def _quantile_bounds(bounds):
    values = np.asarray(bounds, dtype=float).reshape(-1)
    if values.size != 2 or not np.all(np.isfinite(values)): raise ValueError("quantile_bounds must contain exactly two finite probabilities.")
    lower, upper = map(float, values)
    if not 0.0 < lower < upper < 1.0: raise ValueError("quantile_bounds must satisfy 0 < lower < upper < 1.")
    return lower, upper


def predict_day(date="2022-08-22", quantile_bounds=(0.025, 0.975), baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, max_state_iter=50, state_tol=1e-8, output_root=OUTPUT):
    start = day_start(date)
    lower_q, upper_q = _quantile_bounds(quantile_bounds)
    frame, _ = aligned_wm4_data("MED")
    output_model, output_fit, fit, _ = fit_history_optimized(
        frame, start, baseline_weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )
    end, actual, U, Time, model, fine_index = prediction_setup(frame, start, output_model, steps_per_hour)

    if "x_hat" not in fit or "x_cov" not in fit: raise RuntimeError("Optimized WM4 fit did not return prediction-state uncertainty.")
    x_hat = np.asarray(fit["x_hat"], dtype=float)
    x_cov = np.asarray(fit["x_cov"], dtype=float)
    state = model.deterministic_solver(x_hat, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})

    sensitivity = model.sensitivity(fit["r"])
    J = np.asarray(sensitivity["input"], dtype=float)
    variance = np.einsum("ni,ij,nj->n", J, x_cov, J)
    std = np.sqrt(np.maximum(variance, 0.0))
    lower = state["I"] + norm.ppf(lower_q) * std
    upper = state["I"] + norm.ppf(upper_q) * std
    width = upper - lower

    hourly = np.arange(DAY_HOURS) * int(steps_per_hour)
    observed = actual["I"].to_numpy(dtype=float)
    predicted = state["I"][hourly]
    lower_hourly, upper_hourly = lower[hourly], upper[hourly]
    inside = (observed >= lower_hourly) & (observed <= upper_hourly)
    metrics = deviation_metrics(observed, predicted)
    coverage = float(np.mean(inside))
    mean_width = float(np.mean(width[hourly]))
    max_width = float(np.max(width[hourly]))

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"initial_state_uncertainty_{start:%Y-%m-%d}.png"

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    axes[0].fill_between(fine_index, lower, upper, alpha=0.2, label=f"{100.0 * (upper_q - lower_q):.1f}% initial-state interval")
    axes[0].plot(fine_index, state["I"], label="WM4 mean prediction")
    axes[0].plot(fine_index, state["O"], label="Expected output O", alpha=0.8)
    axes[0].scatter(actual.index, observed, s=24, label="Actual user input I", zorder=3)
    axes[0].set_ylabel("Normalized input"); axes[0].legend(); axes[0].grid(True, alpha=0.25)

    axes[1].plot(fine_index, J[:, 0], label="dI/dF0")
    axes[1].plot(fine_index, J[:, 1], label="dI/dR0")
    axes[1].set_ylabel("Initial-state sensitivity"); axes[1].legend(); axes[1].grid(True, alpha=0.25)

    axes[2].plot(fine_index, width, label="Quantile interval width")
    axes[2].set_ylabel("Interval width"); axes[2].set_xlabel("Physical time"); axes[2].legend(); axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"wm_4 initial-state uncertainty | {start:%Y-%m-%d} | coverage={100.0 * coverage:.1f}%")
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)

    return {
        "date": start, "end": end, "deterministic": bool(deterministic),
        "quantile_bounds": (lower_q, upper_q), "x_hat": x_hat, "x_cov": x_cov,
        "x_std": np.sqrt(np.maximum(np.diag(x_cov), 0.0)), "fit": fit,
        "state": state, "sensitivity": sensitivity, "lower": lower, "upper": upper, "width": width,
        "observed": observed, "inside": inside, "coverage": coverage,
        "mean_hourly_width": mean_width, "max_hourly_width": max_width,
        "metrics": metrics, "O_fit": output_fit, "plot": path,
    }


def main():
    parser = ArgumentParser(description="Predict one WM4 day from the optimized historical initial-state distribution and propagate its uncertainty by model sensitivity.")
    parser.add_argument("--date", default="2022-08-22")
    parser.add_argument("--quantile-bounds", type=float, nargs=2, default=(0.025, 0.975), metavar=("LOWER", "UPPER"))
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    args = parser.parse_args()

    result = predict_day(
        date=args.date, quantile_bounds=tuple(args.quantile_bounds),
        baseline_weeks=args.baseline_weeks, steps_per_hour=args.steps_per_hour,
        max_state_iter=args.max_state_iter, state_tol=args.state_tol,
    )
    r = np.asarray(result["fit"]["r"], dtype=float)
    print(f"prediction day={result['date']:%Y-%m-%d}")
    print(f"r=[r_F={r[0]:.8f}, r_U={r[1]:.8f}, r_R={r[2]:.8f}]")
    print(f"x_hat=[F0={result['x_hat'][0]:.8f}, R0={result['x_hat'][1]:.8f}]")
    print("x_cov=")
    print(np.array2string(result["x_cov"], precision=8))
    print(f"quantile bounds={result['quantile_bounds']}, empirical hourly coverage={100.0 * result['coverage']:.2f}%")
    print(f"mean hourly interval width={result['mean_hourly_width']:.8f}, max hourly interval width={result['max_hourly_width']:.8f}")
    print(f"WM4 RMSE={result['metrics']['rmse']:.10f}, MAE={result['metrics']['mae']:.10f}, bias={result['metrics']['bias']:.10f}")
    print(f"plot={result['plot']}")


if __name__ == "__main__": main()
