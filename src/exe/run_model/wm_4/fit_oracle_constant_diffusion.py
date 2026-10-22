from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import DAY_HOURS, day_start
from src.exe.run_model.wm_4.predict_all_days_optimized import TOP_END_OUTPUT


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "uncertainty" / "oracle_constant_diffusion"
Z_95 = 1.959963984540054


def load_oracle_day(root, day):
    """Read the already-fitted oracle state, structural r, robust O and observed I. Never refit them."""
    root, day = Path(root), day_start(day)
    daily_path, hourly_path = root / "daily_metrics.csv", root / "hourly_predictions.csv"
    for path in (daily_path, hourly_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing saved oracle predictions: {path}\n"
                f"Generate them with: python -m src.exe.run_model.wm_4.predict_all_days_optimized "
                f"--start-day {day:%Y-%m-%d} --end-day {day:%Y-%m-%d} --optimize-initial-state"
            )

    daily = pd.read_csv(daily_path, parse_dates=["day"])
    hourly = pd.read_csv(hourly_path, parse_dates=["day", "time"])
    daily_required = {"day", "prediction_state_mode", "r_F", "r_U", "r_R", "F_0", "R_0"}
    hourly_required = {"day", "time", "I_observed", "I_wm4", "O", "R"}
    for name, data, required in (("daily_metrics.csv", daily, daily_required), ("hourly_predictions.csv", hourly, hourly_required)):
        missing = sorted(required.difference(data.columns))
        if missing: raise ValueError(f"{name} is missing columns: {', '.join(missing)}.")

    daily = daily.loc[daily["day"].dt.normalize() == day]
    hourly = hourly.loc[hourly["day"].dt.normalize() == day].sort_values("time").reset_index(drop=True)
    if len(daily) != 1: raise ValueError(f"Expected exactly one saved daily fit for {day:%Y-%m-%d}, found {len(daily)}.")
    if len(hourly) != DAY_HOURS: raise ValueError(f"Expected {DAY_HOURS} hourly oracle samples for {day:%Y-%m-%d}, found {len(hourly)}.")
    row = daily.iloc[0]
    if str(row["prediction_state_mode"]).lower() != "oracle":
        raise ValueError("Saved predictions are not oracle-state results. Re-run with --optimize-initial-state.")

    expected = pd.date_range(day, periods=DAY_HOURS, freq="h")
    if not pd.DatetimeIndex(hourly["time"]).equals(expected):
        raise ValueError("Saved samples must cover the requested calendar day at hours 00:00 through 23:00.")
    columns = ["I_observed", "I_wm4", "O", "R"]
    parameters = np.asarray([row["r_F"], row["r_U"], row["r_R"], row["F_0"], row["R_0"]], dtype=float)
    if not np.isfinite(parameters).all() or not np.isfinite(hourly[columns].to_numpy(dtype=float)).all():
        raise ValueError("Saved oracle parameters, observed input and robust baseline must all be finite.")
    if not np.allclose(hourly["I_wm4"], hourly["O"] + hourly["R"], rtol=1e-6, atol=1e-7):
        raise ValueError("Saved oracle I does not match O + R; verify the source prediction files.")
    if not np.isclose(float(row["R_0"]), float(hourly.loc[0, "R"]), rtol=1e-5, atol=1e-7):
        raise ValueError("Saved oracle initial rate does not match the hourly trajectory.")

    return row, hourly


def unit_process_covariance(r, steps_per_hour):
    """Euler-Maruyama covariance of hourly I=O+R for q=sigma**2=1 and a known initial state.

    WM4 uses normalized days: dt=1/(24*steps_per_hour); noise enters the R equation.
    Hour zero has zero process variance, and is not informative about sigma.
    """
    r = np.asarray(r, dtype=float).reshape(3)
    steps_per_hour = int(steps_per_hour)
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2.")
    dt = 1.0 / (DAY_HOURS * steps_per_hour)
    A = np.array([[1.0, dt], [dt * r[0], 1.0 + dt * r[2]]], dtype=float)
    sample_indices = np.arange(DAY_HOURS, dtype=int) * steps_per_hour
    n = int(sample_indices[-1])
    impulse = np.empty(n, dtype=float)
    response = np.array([0.0, 1.0], dtype=float)
    for lag in range(n):
        impulse[lag] = response[1]
        response = A @ response

    # Each row maps independent fine-grid standard normal increments to the sampled R.
    H = np.zeros((DAY_HOURS, n), dtype=float)
    for hour, index in enumerate(sample_indices):
        if index: H[hour, :index] = np.sqrt(dt) * impulse[:index][::-1]
    return H @ H.T


def fit_oracle_diffusion(daily, hourly, steps_per_hour=60):
    """Estimate only constant q via the Gaussian likelihood of the 23 correlated oracle errors."""
    r = np.asarray([daily["r_F"], daily["r_U"], daily["r_R"]], dtype=float)
    observed = hourly["I_observed"].to_numpy(dtype=float)
    oracle = hourly["I_wm4"].to_numpy(dtype=float)
    baseline = hourly["O"].to_numpy(dtype=float)
    residual = observed - oracle

    unit_covariance = unit_process_covariance(r, steps_per_hour)
    # For fixed oracle mean and P0=0, Cov(error[1:])=q*C. MLE is e.T C^-1 e / n.
    # The hour-zero residual cannot be explained by a process starting at midnight.
    L = np.linalg.cholesky(unit_covariance[1:, 1:])
    whitened = np.linalg.solve(L, residual[1:])
    q = max(0.0, float(whitened @ whitened) / (DAY_HOURS - 1))
    sigma = float(np.sqrt(q))

    variance = np.maximum(q * np.diag(unit_covariance), 0.0)
    std = np.sqrt(variance)
    lower, upper = oracle - Z_95 * std, oracle + Z_95 * std
    coverage = (observed >= lower) & (observed <= upper)
    fitted = hourly[["time"]].copy()
    fitted["hour"] = np.arange(DAY_HOURS, dtype=int)
    fitted["I_observed"] = observed
    fitted["I_oracle"] = oracle
    fitted["O_robust"] = baseline
    fitted["oracle_residual"] = residual
    fitted["unit_process_variance"] = np.diag(unit_covariance)
    fitted["process_variance"] = variance
    fitted["process_std"] = std
    fitted["ci95_lower"] = lower
    fitted["ci95_upper"] = upper
    fitted["covered_95"] = coverage

    summary = {
        "day": f"{day_start(daily['day']):%Y-%m-%d}",
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "oracle_F0": float(daily["F_0"]), "oracle_R0": float(daily["R_0"]),
        "q": q, "sigma": sigma, "sample_hours": DAY_HOURS,
        "diffusion_fit_hours": DAY_HOURS - 1,
        "oracle_mse": float(np.mean(residual ** 2)),
        "baseline_mse": float(np.mean((observed - baseline) ** 2)),
        "hour0_residual": float(residual[0]),
        "coverage_95_hours_1_23": float(np.mean(coverage[1:])),
    }
    return summary, fitted


def save_plot(result, hourly, path):
    hour = hourly["hour"].to_numpy(dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(hour, hourly["I_observed"], "o-", label="Observed input I")
    axes[0].plot(hour, hourly["I_oracle"], label="Fixed WM4 oracle mean I=O+R")
    axes[0].plot(hour, hourly["O_robust"], linestyle="--", label="Fixed robust baseline O(t)")
    axes[0].fill_between(hour, hourly["ci95_lower"], hourly["ci95_upper"], alpha=0.2, label="Fitted process-only 95% interval")
    axes[0].set_ylabel("Normalized input")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)

    axes[1].plot(hour, hourly["oracle_residual"], "o-", label="Observed - oracle")
    axes[1].fill_between(hour, -Z_95 * hourly["process_std"], Z_95 * hourly["process_std"], alpha=0.2, label="Process-only 95% range")
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_xlabel("Hour of day"); axes[1].set_ylabel("Oracle residual")
    axes[1].set_xticks(np.arange(0, DAY_HOURS, 2))
    axes[1].legend(); axes[1].grid(True, alpha=0.25)

    fig.suptitle(f"WM4 fixed oracle + robust O | {result['day']} | sigma={result['sigma']:.6g}")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    parser = ArgumentParser(description="Estimate only constant WM4 diffusion sigma from a saved 24-hour robust-baseline oracle prediction (no parameter or state refitting).")
    parser.add_argument("--day", default="2022-08-22", help="Calendar day already saved with --optimize-initial-state.")
    parser.add_argument("--input", default=str(TOP_END_OUTPUT), help="Saved all_days_optimized_top_end directory.")
    parser.add_argument("--steps-per-hour", type=int, default=60, help="Use the same solver resolution as the saved oracle fit.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    daily, observed = load_oracle_day(args.input, args.day)
    summary, hourly = fit_oracle_diffusion(daily, observed, args.steps_per_hour)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    stem = f"{summary['day']}_oracle_constant_diffusion"
    csv_path, plot_path, summary_path = output / f"{stem}_hourly.csv", output / f"{stem}.png", output / f"{stem}_summary.csv"
    hourly.to_csv(csv_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    save_plot(summary, hourly, plot_path)

    print(f"day={summary['day']}, fixed r=[{summary['r_F']:.8f}, {summary['r_U']:.8f}, {summary['r_R']:.8f}]")
    print(f"fixed oracle [F0,R0]=[{summary['oracle_F0']:.8f}, {summary['oracle_R0']:.8f}]")
    print(f"constant diffusion q=sigma^2={summary['q']:.10f}, sigma={summary['sigma']:.10f}")
    print(f"oracle MSE={summary['oracle_mse']:.10f}, robust O MSE={summary['baseline_mse']:.10f}")
    print(f"hour-0 residual={summary['hour0_residual']:.8f} (cannot be explained by process noise)")
    print(f"in-sample 95% coverage, hours 1-23={100.0 * summary['coverage_95_hours_1_23']:.2f}%")
    print("The 23 hourly errors are one correlated realization; the likelihood accounts for their covariance.")
    print("This is an oracle-conditioned, in-sample process-diffusion estimate, not a holdout calibration.")
    print(f"hourly={csv_path}\nsummary={summary_path}\nplot={plot_path}")


if __name__ == "__main__": main()
