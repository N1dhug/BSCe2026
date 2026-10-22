from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import DAY_HOURS, aligned_wm4_data, day_start, fit_expected_output, model_output
from src.exe.run_model.wm_4.predict_all_days_mpc import OUTPUT as MPC_OUTPUT, _window_input
from src.model.wm_4 import wm_4
from src.params.wm_4 import optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = MPC_OUTPUT / "mpc_states.csv"
OUTPUT = ROOT / "files" / "output" / "wm_4" / "uncertainty" / "oracle_mpc_24h_diffusion"
Z_95 = 1.959963984540054


def load_origins(path, start_day=None, end_day=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved MPC parameters not found: {path}\n"
            "Run python -m src.exe.run_model.wm_4.predict_all_days_mpc first."
        )
    source = pd.read_csv(path, parse_dates=["origin"])
    required = {"origin", "r_F", "r_U", "r_R"}
    missing = sorted(required.difference(source.columns))
    if missing: raise ValueError(f"Missing saved MPC columns: {', '.join(missing)}.")
    if source["origin"].isna().any(): raise ValueError("Saved MPC origins contain missing timestamps.")
    if source["origin"].duplicated().any(): raise ValueError("Saved MPC origins must be unique.")
    source = source.sort_values("origin")
    if start_day is not None: source = source.loc[source["origin"] >= day_start(start_day)]
    if end_day is not None: source = source.loc[source["origin"] < day_start(end_day) + pd.Timedelta(days=1)]
    if source.empty: raise ValueError("No saved MPC origins fall in the requested date range.")
    if not np.isfinite(source[["r_F", "r_U", "r_R"]].to_numpy(dtype=float)).all():
        raise ValueError("Saved deterministic WM4 coefficients must be finite.")
    return source


def unit_process_covariance(r, steps_per_hour=60):
    """24x24 process covariance of [I(1h),...,I(24h)] for unit q=sigma**2 and P0=0.

    WM4 time is normalized to days, so dt = 1/(24*steps_per_hour).
    For i>=j, Cov(R_i,R_j)=e_R.T A**(i-j) Cov(x_j) e_R.
    """
    steps_per_hour = int(steps_per_hour)
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2.")
    r = np.asarray(r, dtype=float).reshape(3)
    dt = 1.0 / (DAY_HOURS * steps_per_hour)
    A = np.array([[1.0, dt], [dt * r[0], 1.0 + dt * r[2]]], dtype=float)
    Q = np.array([[0.0, 0.0], [0.0, dt]], dtype=float)
    P = np.zeros((2, 2), dtype=float)
    hourly = []
    for step in range(1, DAY_HOURS * steps_per_hour + 1):
        P = A @ P @ A.T + Q
        if step % steps_per_hour == 0: hourly.append(P.copy())

    A_hour = np.linalg.matrix_power(A, steps_per_hour)
    powers = [np.eye(2, dtype=float)]
    for _ in range(1, DAY_HOURS): powers.append(A_hour @ powers[-1])
    covariance = np.zeros((DAY_HOURS, DAY_HOURS), dtype=float)
    for j, Pj in enumerate(hourly):
        for i in range(j, DAY_HOURS):
            covariance[i, j] = (powers[i - j] @ Pj)[1, 1]
            covariance[j, i] = covariance[i, j]
    return covariance


def fit_origin(frame, origin, r, output_model, steps_per_hour):
    """Keep saved MPC r and the robust O fixed; optimize only oracle x0 and diffusion q."""
    origin, steps_per_hour = pd.Timestamp(origin), int(steps_per_hour)
    last_time = origin + pd.Timedelta(hours=DAY_HOURS)
    end_I = frame.reindex([last_time])["I"].to_numpy(dtype=float)
    if end_I.size != 1 or not np.isfinite(end_I[0]):
        raise ValueError(f"Missing genuine observed input at the 24-hour endpoint {last_time}.")

    data, fine_index = _window_input(frame, origin, steps_per_hour, output_model)
    samples = np.arange(DAY_HOURS + 1, dtype=int) * steps_per_hour
    data["observation_mask"][samples] = True
    model = wm_4((0.0, 1.0), len(fine_index), model_output(data, fine_index))
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(data, r)
    state = oracle["deterministic_state"]

    actual = np.asarray(data["I"], dtype=float)[samples]
    predicted = np.asarray(state["I"], dtype=float)[samples]
    baseline = np.asarray(state["O"], dtype=float)[samples]
    error = actual - predicted
    unit_covariance = unit_process_covariance(r, steps_per_hour)

    # A known/oracle initial state has P0=0. Time zero carries no process variance.
    # Whiten hours 1..24 to correctly account for the correlated process errors.
    L = np.linalg.cholesky(unit_covariance)
    whitened = np.linalg.solve(L, error[1:])
    chi2 = float(whitened @ whitened)
    q = max(0.0, chi2 / DAY_HOURS)
    sigma = float(np.sqrt(q))
    # Descriptive small-sample diagnostic; oracle x0 was fitted from these data,
    # so this is not an unbiased or out-of-sample uncertainty estimate.
    q_dof = max(0.0, chi2 / (DAY_HOURS - 2))

    process_variance = np.maximum(q * np.diag(unit_covariance), 0.0)
    process_std = np.sqrt(process_variance)
    forecasts = pd.DataFrame({
        "origin": origin,
        "origin_day": origin.normalize(),
        "origin_hour": origin.hour,
        "lead_hours": np.arange(1, DAY_HOURS + 1, dtype=int),
        "target_time": fine_index[samples[1:]],
        "I_observed": actual[1:],
        "I_oracle": predicted[1:],
        "O": baseline[1:],
        "oracle_error": error[1:],
        "unit_process_variance": np.diag(unit_covariance),
        "process_variance": process_variance,
        "process_std": process_std,
        "lower_95": predicted[1:] - Z_95 * process_std,
        "upper_95": predicted[1:] + Z_95 * process_std,
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "F0_oracle": oracle["init_cond"][0],
        "R0_oracle": oracle["init_cond"][1],
        "q": q, "sigma": sigma,
    })
    forecasts["covered_95"] = (forecasts["I_observed"] >= forecasts["lower_95"]) & (forecasts["I_observed"] <= forecasts["upper_95"])
    metrics = {
        "origin": origin,
        "origin_day": origin.normalize(),
        "origin_hour": origin.hour,
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "F0_oracle": float(oracle["init_cond"][0]),
        "R0_oracle": float(oracle["init_cond"][1]),
        "q": q, "sigma": sigma, "q_dof_diagnostic": q_dof,
        "weighted_residual_sum": chi2,
        "fit_hours": DAY_HOURS,
        "hour_0_residual": float(error[0]),
        "oracle_mse_hours_1_24": float(np.mean(error[1:] ** 2)),
        "baseline_mse_hours_1_24": float(np.mean((actual[1:] - baseline[1:]) ** 2)),
        "coverage_95_hours_1_24": float(forecasts["covered_95"].mean()),
    }
    return metrics, forecasts


def run(frame, origins, steps_per_hour=60, fail_fast=False):
    """One fixed 24-hour oracle/diffusion problem at every saved hourly MPC origin."""
    rows, trajectories, skipped = [], [], []
    cached_day, output_model = None, None

    for item in origins.itertuples(index=False):
        origin = pd.Timestamp(item.origin)
        day = origin.normalize()
        try:
            # MPC originally fitted its robust baseline once per origin day.
            # Its coefficients are not serialized in mpc_states.csv, so recover
            # the same historical-only baseline fit once per day here.
            if cached_day != day:
                output_model, _ = fit_expected_output(frame, day)
                cached_day = day
            r = np.array([item.r_F, item.r_U, item.r_R], dtype=float)
            result, forecasts = fit_origin(frame, origin, r, output_model, steps_per_hour)
            rows.append(result); trajectories.append(forecasts)
            print(f"{origin:%Y-%m-%d %H:%M} | oracle x0=({result['F0_oracle']:.5f},{result['R0_oracle']:.5f}) | sigma={result['sigma']:.7f}")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append({"origin": origin, "reason": str(exc)})
            print(f"{origin:%Y-%m-%d %H:%M} | skipped | {exc}")

    if not rows: raise RuntimeError("No saved MPC origin has a complete observed future 24-hour horizon.")
    return pd.DataFrame(rows), pd.concat(trajectories, ignore_index=True), pd.DataFrame(skipped)


def summarise(origins, forecasts):
    # Each origin contains 24 whitened residuals. Pooling their squared
    # whitened errors fits one additional common q across all origins.
    pooled_q = float(origins["weighted_residual_sum"].sum() / (DAY_HOURS * len(origins)))
    pooled_sigma = float(np.sqrt(max(0.0, pooled_q)))
    forecasts = forecasts.copy()
    forecasts["pooled_process_std"] = np.sqrt(np.maximum(pooled_q * forecasts["unit_process_variance"].to_numpy(dtype=float), 0.0))
    forecasts["pooled_covered_95"] = np.abs(forecasts["oracle_error"]) <= Z_95 * forecasts["pooled_process_std"]

    rows = []
    for lead, part in forecasts.groupby("lead_hours", sort=True):
        residual = part["oracle_error"].to_numpy(dtype=float)
        rows.append({
            "lead_hours": int(lead),
            "origins": int(len(part)),
            "oracle_bias": float(np.mean(residual)),
            "oracle_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "baseline_rmse": float(np.sqrt(np.mean((part["I_observed"] - part["O"]) ** 2))),
            "mean_origin_fitted_std": float(np.sqrt(np.mean(part["process_variance"]))),
            "pooled_predicted_rms_std": float(np.sqrt(np.mean(part["pooled_process_std"] ** 2))),
            "origin_fitted_coverage_95_pct": 100.0 * float(part["covered_95"].mean()),
            "pooled_coverage_95_pct": 100.0 * float(part["pooled_covered_95"].mean()),
        })
    lead_summary = pd.DataFrame(rows)
    global_summary = pd.DataFrame([{
        "origins": len(origins),
        "forecast_rows": len(forecasts),
        "horizon_hours": DAY_HOURS,
        "mean_origin_sigma": float(origins["sigma"].mean()),
        "median_origin_sigma": float(origins["sigma"].median()),
        "pooled_q": pooled_q,
        "pooled_sigma": pooled_sigma,
        "oracle_mse": float(np.mean(forecasts["oracle_error"] ** 2)),
        "baseline_mse": float(np.mean((forecasts["I_observed"] - forecasts["O"]) ** 2)),
        "pooled_coverage_95_pct": 100.0 * float(forecasts["pooled_covered_95"].mean()),
    }])
    return forecasts, lead_summary, global_summary


def save_plot(origins, summary, path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].plot(origins["origin"], origins["sigma"], label="24h oracle-conditioned sigma by MPC origin")
    axes[0].set_ylabel("Diffusion sigma")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)

    h = summary["lead_hours"].to_numpy(dtype=int)
    axes[1].plot(h, summary["oracle_rmse"], marker="o", label="Oracle residual RMSE")
    axes[1].plot(h, summary["pooled_predicted_rms_std"], marker="o", label="Pooled process RMS std")
    axes[1].plot(h, summary["baseline_rmse"], linestyle="--", label="Robust O(t) RMSE")
    axes[1].set_xlabel("Hours since hourly MPC origin")
    axes[1].set_ylabel("Normalized input")
    axes[1].set_xticks(np.arange(1, DAY_HOURS + 1, 2))
    axes[1].legend(); axes[1].grid(True, alpha=0.25)

    fig.suptitle("WM4 oracle-conditioned constant diffusion | 24h at every MPC origin")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    parser = ArgumentParser(description="Fit constant WM4 diffusion on each rolling 24h oracle horizon using saved hourly MPC coefficients, without re-optimizing structural r.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Previously saved mpc_states.csv containing hourly-origin r_F, r_U, r_R.")
    parser.add_argument("--start-day", default=None, help="First origin day, inclusive; default all saved.")
    parser.add_argument("--end-day", default=None, help="Last origin day, inclusive; default all saved.")
    parser.add_argument("--steps-per-hour", type=int, default=60, help="Match the resolution of the saved MPC parameter fit.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if args.start_day and args.end_day and day_start(args.end_day) < day_start(args.start_day):
        parser.error("--end-day must not precede --start-day.")

    origins = load_origins(args.input, args.start_day, args.end_day)
    frame, _ = aligned_wm4_data("MED")
    fitted, forecasts, skipped = run(frame, origins, args.steps_per_hour, args.fail_fast)
    forecasts, lead, global_summary = summarise(fitted, forecasts)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    fitted.to_csv(output / "diffusion_by_mpc_origin.csv", index=False)
    forecasts.to_csv(output / "oracle_24h_by_origin_and_lead.csv", index=False)
    lead.to_csv(output / "diffusion_by_lead.csv", index=False)
    global_summary.to_csv(output / "diffusion_summary.csv", index=False)
    if not skipped.empty: skipped.to_csv(output / "skipped_origins.csv", index=False)
    save_plot(fitted, lead, output / "oracle_mpc_24h_diffusion.png")

    row = global_summary.iloc[0]
    print(f"origins={row['origins']}, horizon=24 h, skipped={len(skipped)}")
    print(f"pooled sigma={row['pooled_sigma']:.10f}, mean origin sigma={row['mean_origin_sigma']:.10f}")
    print(f"oracle MSE={row['oracle_mse']:.10f}, robust O MSE={row['baseline_mse']:.10f}")
    print(f"pooled 95% coverage={row['pooled_coverage_95_pct']:.2f}% (in sample)")
    print("Fixed r from saved MPC states; F0/R0 are re-estimated as an oracle using all 25 observed hours.")
    print("The previously fitted robust O(t) is reconstructed once per day using the unchanged historical-only method.")
    print("Diffusion is conditional on each fitted oracle state: calibration is in sample, not a forecast validation.")
    print(f"output={output}")


if __name__ == "__main__": main()
