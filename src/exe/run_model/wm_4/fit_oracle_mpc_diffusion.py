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
    """Fix historical r and robust O, fit oracle x0, then collect residuals and unit variance."""
    origin, steps_per_hour = pd.Timestamp(origin), int(steps_per_hour)
    endpoint = origin + pd.Timedelta(hours=DAY_HOURS)
    observed_end = frame.reindex([endpoint])["I"].to_numpy(dtype=float)
    if observed_end.size != 1 or not np.isfinite(observed_end[0]):
        raise ValueError(f"Missing observed input at the 24-hour endpoint {endpoint}.")

    data, fine_index = _window_input(frame, origin, steps_per_hour, output_model)
    samples = np.arange(DAY_HOURS + 1, dtype=int) * steps_per_hour
    data["observation_mask"][samples] = True
    model = wm_4((0.0, 1.0), len(fine_index), model_output(data, fine_index))
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(data, r)
    state = oracle["deterministic_state"]

    observed = np.asarray(data["I"], dtype=float)[samples]
    predicted = np.asarray(state["I"], dtype=float)[samples]
    baseline = np.asarray(state["O"], dtype=float)[samples]
    residual = observed - predicted
    unit_variance = np.maximum(np.diag(unit_process_covariance(r, steps_per_hour)), 0.0)

    forecasts = pd.DataFrame({
        "origin": origin,
        "origin_day": origin.normalize(),
        "origin_hour": origin.hour,
        "lead_hours": np.arange(1, DAY_HOURS + 1, dtype=int),
        "target_time": fine_index[samples[1:]],
        "I_observed": observed[1:],
        "I_oracle": predicted[1:],
        "O": baseline[1:],
        "oracle_error": residual[1:],
        "unit_process_variance": unit_variance,
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "F0_oracle": float(oracle["init_cond"][0]),
        "R0_oracle": float(oracle["init_cond"][1]),
    })
    metrics = {
        "origin": origin, "origin_day": origin.normalize(), "origin_hour": origin.hour,
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "F0_oracle": float(oracle["init_cond"][0]),
        "R0_oracle": float(oracle["init_cond"][1]),
        "fit_hours": DAY_HOURS, "hour_0_residual": float(residual[0]),
        "oracle_mse_hours_1_24": float(np.mean(residual[1:] ** 2)),
        "baseline_mse_hours_1_24": float(np.mean((observed[1:] - baseline[1:]) ** 2)),
    }
    return metrics, forecasts

def run(frame, origins, steps_per_hour=60, fail_fast=False):
    """Reconstruct every oracle trajectory using existing MPC parameters; never fit diffusion here."""
    rows, trajectories, skipped = [], [], []
    cached_day, output_model = None, None
    for item in origins.itertuples(index=False):
        origin = pd.Timestamp(item.origin)
        day = origin.normalize()
        try:
            # The historical-only robust O was fitted once per origin day in the MPC.
            # Its coefficients were not saved, so repeat that unchanged fit once per day.
            if cached_day != day:
                output_model, _ = fit_expected_output(frame, day)
                cached_day = day
            r = np.array([item.r_F, item.r_U, item.r_R], dtype=float)
            result, forecasts = fit_origin(frame, origin, r, output_model, steps_per_hour)
            rows.append(result); trajectories.append(forecasts)
            print(f"{origin:%Y-%m-%d %H:%M} | fixed r | oracle x0=({result['F0_oracle']:.5f},{result['R0_oracle']:.5f})")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append({"origin": origin, "reason": str(exc)})
            print(f"{origin:%Y-%m-%d %H:%M} | skipped | {exc}")

    if not rows: raise RuntimeError("No saved MPC origin has a complete observed future 24-hour horizon.")
    return pd.DataFrame(rows), pd.concat(trajectories, ignore_index=True), pd.DataFrame(skipped)

def fit_variance(forecasts, min_origins=2):
    """Direct convex QP: min_{q>=0} sum_h [var_k(e_{k,h}) - q * mean_k(K_{k,h})]**2.

    Each hourly MPC origin has fixed structural parameters and its own oracle
    initial state. Variances are computed across origins, not within one
    trajectory. One q=sigma**2 is fitted jointly for all 24 lead times.
    """
    if int(forecasts["origin"].nunique()) < int(min_origins):
        raise ValueError(f"At least {min_origins} distinct complete MPC origins are required to estimate empirical residual variance.")

    rows = []
    for lead, part in forecasts.groupby("lead_hours", sort=True):
        error = part["oracle_error"].to_numpy(dtype=float)
        kernel = part["unit_process_variance"].to_numpy(dtype=float)
        if error.size < 2 or not np.isfinite(error).all() or not np.isfinite(kernel).all():
            raise ValueError(f"Invalid oracle residuals or unit variance at lead {lead}.")
        if np.any(kernel < 0.0): raise ValueError(f"Negative model process variance at lead {lead}.")
        rows.append({
            "lead_hours": int(lead),
            "origins": int(error.size),
            "oracle_bias": float(np.mean(error)),
            "oracle_rmse": float(np.sqrt(np.mean(error ** 2))),
            "empirical_oracle_variance": float(np.var(error, ddof=1)),
            "empirical_oracle_std": float(np.std(error, ddof=1)),
            "mean_unit_process_variance": float(np.mean(kernel)),
            "baseline_rmse": float(np.sqrt(np.mean(
                (part["I_observed"].to_numpy(dtype=float) - part["O"].to_numpy(dtype=float)) ** 2))),
        })
    lead = pd.DataFrame(rows).sort_values("lead_hours").reset_index(drop=True)
    if lead["lead_hours"].tolist() != list(range(1, DAY_HOURS + 1)):
        raise ValueError("Direct diffusion fitting requires all 24 forecast lead times.")
    k = lead["mean_unit_process_variance"].to_numpy(dtype=float)
    v = lead["empirical_oracle_variance"].to_numpy(dtype=float)
    denom = float(k @ k)
    if denom <= 0.0 or not np.isfinite(denom): raise ValueError("No positive WM4 process-variance response is available.")
    q = max(0.0, float(k @ v) / denom)
    return q, lead


def rolling_variance_fits(forecasts, rolling_days=14, min_origins=24):
    """Optional descriptive, backward-looking cross-origin fits for a sigma time series."""
    rolling_days, min_origins = int(rolling_days), int(min_origins)
    if rolling_days < 1 or min_origins < 2: raise ValueError("rolling_days >= 1 and min_origins >= 2 are required.")
    dates = pd.DatetimeIndex(forecasts["origin_day"].drop_duplicates().sort_values())
    rows = []
    for day in dates:
        start, stop = day - pd.Timedelta(days=rolling_days - 1), day + pd.Timedelta(days=1)
        part = forecasts.loc[(forecasts["origin"] >= start) & (forecasts["origin"] < stop)]
        n = int(part["origin"].nunique())
        q = np.nan if n < min_origins else fit_variance(part)[0]
        rows.append({
            "day": day, "origins": n,
            "rolling_days": rolling_days,
            "q": q, "sigma": float(np.sqrt(q)) if np.isfinite(q) else np.nan,
        })
    return pd.DataFrame(rows)


def summarise(origins, forecasts, rolling_days=14, min_rolling_origins=24):
    q, lead = fit_variance(forecasts)
    sigma = float(np.sqrt(q))
    lead["fitted_process_variance"] = q * lead["mean_unit_process_variance"]
    lead["fitted_process_std"] = np.sqrt(np.maximum(lead["fitted_process_variance"], 0.0))
    lead["variance_residual"] = lead["fitted_process_variance"] - lead["empirical_oracle_variance"]
    lead["q_by_lead"] = np.divide(
        lead["empirical_oracle_variance"], lead["mean_unit_process_variance"],
        out=np.full(len(lead), np.nan), where=lead["mean_unit_process_variance"] > 0.0,
    )
    lead["sigma_by_lead"] = np.sqrt(np.maximum(lead["q_by_lead"], 0.0))

    forecasts = forecasts.copy()
    forecasts["q"] = q
    forecasts["sigma"] = sigma
    forecasts["process_variance"] = q * forecasts["unit_process_variance"]
    forecasts["process_std"] = np.sqrt(np.maximum(forecasts["process_variance"], 0.0))
    forecasts["lower_95"] = forecasts["I_oracle"] - Z_95 * forecasts["process_std"]
    forecasts["upper_95"] = forecasts["I_oracle"] + Z_95 * forecasts["process_std"]
    forecasts["covered_95"] = (forecasts["I_observed"] >= forecasts["lower_95"]) & (forecasts["I_observed"] <= forecasts["upper_95"])
    lead["coverage_95_pct"] = 100.0 * forecasts.groupby("lead_hours", sort=True)["covered_95"].mean().to_numpy(dtype=float)

    rolling = rolling_variance_fits(forecasts, rolling_days, min_rolling_origins)
    origins = origins.copy()
    origins["global_q"] = q
    origins["global_sigma"] = sigma
    origins = origins.merge(rolling[["day", "q", "sigma"]], left_on="origin_day", right_on="day", how="left")
    origins = origins.rename(columns={"q": "rolling_q", "sigma": "rolling_sigma"}).drop(columns=["day"])

    var_error = lead["variance_residual"].to_numpy(dtype=float)
    target = lead["empirical_oracle_variance"].to_numpy(dtype=float)
    target_var = float(np.mean((target - np.mean(target)) ** 2))
    global_summary = pd.DataFrame([{
        "origins": int(origins.shape[0]), "forecast_rows": int(len(forecasts)),
        "horizon_hours": DAY_HOURS, "fit_objective": "empirical_oracle_variance",
        "q": q, "sigma": sigma,
        "variance_mse": float(np.mean(var_error ** 2)),
        "variance_rmse": float(np.sqrt(np.mean(var_error ** 2))),
        "variance_r2": float(1.0 - np.mean(var_error ** 2) / target_var) if target_var > 0.0 else np.nan,
        "oracle_mse": float(np.mean(forecasts["oracle_error"] ** 2)),
        "baseline_mse": float(np.mean((forecasts["I_observed"] - forecasts["O"]) ** 2)),
        "coverage_95_pct": 100.0 * float(forecasts["covered_95"].mean()),
        "rolling_days": int(rolling_days), "min_rolling_origins": int(min_rolling_origins),
    }])
    return origins, forecasts, lead, global_summary, rolling

def save_plot(rolling, lead, global_summary, path):
    fit = global_summary.iloc[0]
    fig, axes = plt.subplots(3, 1, figsize=(15, 14), sharex=False)

    axes[0].plot(rolling["day"], rolling["sigma"], label="Rolling cross-origin variance-fit sigma")
    axes[0].axhline(fit["sigma"], linestyle="--", label=f"Global variance-fit sigma={fit['sigma']:.5f}")
    axes[0].set_ylabel("Diffusion sigma")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)

    h = lead["lead_hours"].to_numpy(dtype=int)
    axes[1].plot(h, lead["empirical_oracle_variance"], marker="o", label="Oracle residual variance across MPC origins")
    axes[1].plot(h, lead["fitted_process_variance"], marker="o", label="WM4 propagated variance (q * mean K)")
    axes[1].set_ylabel("Input variance")
    axes[1].set_xticks(np.arange(1, DAY_HOURS + 1, 2))
    axes[1].legend(); axes[1].grid(True, alpha=0.25)

    axes[2].plot(h, lead["empirical_oracle_std"], marker="o", label="Empirical oracle residual std")
    axes[2].plot(h, lead["fitted_process_std"], marker="o", label="Variance-matched WM4 process std")
    axes[2].plot(h, lead["oracle_rmse"], linestyle=":", label="Oracle residual RMSE")
    axes[2].plot(h, lead["baseline_rmse"], linestyle="--", label="Robust O(t) RMSE")
    axes[2].set_xlabel("Hours since hourly MPC origin")
    axes[2].set_ylabel("Normalized input")
    axes[2].set_xticks(np.arange(1, DAY_HOURS + 1, 2))
    axes[2].legend(); axes[2].grid(True, alpha=0.25)

    fig.suptitle("WM4 oracle-conditioned constant diffusion | direct empirical variance fit over 24h")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)

def main():
    parser = ArgumentParser(description="Fit one constant WM4 diffusion directly to the empirical oracle-residual variance across saved MPC origins at all 24 leads; r remains fixed.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Existing MPC hourly-origin parameters from mpc_states.csv.")
    parser.add_argument("--start-day", default=None, help="First origin day, inclusive; default all saved.")
    parser.add_argument("--end-day", default=None, help="Last origin day, inclusive; default all saved.")
    parser.add_argument("--steps-per-hour", type=int, default=60, help="Same solver resolution as the saved MPC fitting.")
    parser.add_argument("--rolling-days", type=int, default=14, help="Backward-looking window in days for descriptive rolling cross-origin sigma.")
    parser.add_argument("--min-rolling-origins", type=int, default=24, help="Minimum origins for each rolling variance fit.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if args.start_day and args.end_day and day_start(args.end_day) < day_start(args.start_day):
        parser.error("--end-day must not precede --start-day.")

    origins = load_origins(args.input, args.start_day, args.end_day)
    frame, _ = aligned_wm4_data("MED")
    fitted, forecasts, skipped = run(frame, origins, args.steps_per_hour, args.fail_fast)
    fitted, forecasts, lead, summary, rolling = summarise(
        fitted, forecasts, args.rolling_days, args.min_rolling_origins,
    )

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    fitted.to_csv(output / "diffusion_by_mpc_origin.csv", index=False)
    forecasts.to_csv(output / "oracle_24h_by_origin_and_lead.csv", index=False)
    lead.to_csv(output / "diffusion_by_lead.csv", index=False)
    rolling.to_csv(output / "diffusion_rolling.csv", index=False)
    summary.to_csv(output / "diffusion_summary.csv", index=False)
    if not skipped.empty: skipped.to_csv(output / "skipped_origins.csv", index=False)
    save_plot(rolling, lead, summary, output / "oracle_mpc_24h_diffusion.png")

    row = summary.iloc[0]
    print(f"origins={row['origins']}, horizon=24 h, skipped={len(skipped)}")
    print(f"direct variance-fit q={row['q']:.10f}, sigma={row['sigma']:.10f}")
    print(f"variance RMSE={row['variance_rmse']:.10f}, R2={row['variance_r2']:.7f}")
    print(f"oracle MSE={row['oracle_mse']:.10f}, robust O MSE={row['baseline_mse']:.10f}")
    print(f"95% process-only coverage={row['coverage_95_pct']:.2f}% (in sample)")
    print("One q is fitted to the 24 empirical oracle-residual variances across origins, not to individual paths.")
    print("Fixed structural r read from existing MPC states; oracle x0 fitted independently at each origin.")
    print("Overlapping horizons and in-sample oracle states make this diagnostic, not calibrated forecast uncertainty.")
    print(f"output={output}")

if __name__ == "__main__": main()
