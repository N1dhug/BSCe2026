from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import kstest

from src.exe.run_model.wm_4.sweep_weekday_horizon import OUTPUT as SWEEP_OUTPUT, COLUMNS, read_table


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "weekday_horizon_diffusion"
BASELINES = ("weekly_average", "weekly_temperature")
N_VALUES = tuple(range(2, 15))
K_VALUES = tuple(range(1, 25))
START_DAY = "2022-04-09"
END_DAY = "2022-10-31"


def same_period(frame, start_day=START_DAY, end_day=END_DAY, baselines=BASELINES, backlogs=N_VALUES, horizons=K_VALUES):
    """Use one identical calendar-day cohort for both baselines and every N,k cell."""
    start, end = pd.Timestamp(start_day).normalize(), pd.Timestamp(end_day).normalize()
    if end < start: raise ValueError("end-day must be on or after start-day.")
    data = frame.loc[frame["day"].between(start, end)].copy()
    if data.empty: raise ValueError(f"No saved sweep predictions in {start:%Y-%m-%d} to {end:%Y-%m-%d}.")
    required = {(baseline, n, k) for baseline in baselines for n in backlogs for k in horizons}
    days = pd.DatetimeIndex([
        day for day, part in data.groupby("day", sort=True)
        if required.issubset(set(zip(part["baseline"], part["training_weekdays"], part["horizon_hours"])))
    ])
    if days.empty:
        raise ValueError("No fully matched days across all requested baselines, backlogs and horizons in the selected period. Run the complete weekday sweep or narrow the hyperparameter ranges.")
    return data.loc[data["day"].isin(days)].copy(), days



@lru_cache(maxsize=50000)
def unit_process_variance(theta_F, theta_R, steps_per_hour, hour):
    """Exact variance of R at hourly sample with unit constant Euler-Maruyama q."""
    steps_per_hour, hour = int(steps_per_hour), int(hour)
    if steps_per_hour < 2 or hour < 0: raise ValueError("Invalid solver resolution or hour.")
    dt = 1.0 / (24.0 * steps_per_hour)
    A = np.array([[1.0, dt], [dt*theta_F, 1.0+dt*theta_R]], dtype=float)
    response = np.array([0.0, 1.0])
    variance = 0.0
    for _ in range(hour*steps_per_hour):
        variance += dt * response[1]**2
        response = A @ response
    return float(variance)


def load_sweep(source, baselines=BASELINES):
    """Load the actual sweep tables, reconstructing oracle forecasts without refitting theta."""
    source = Path(source)
    fit = read_table(source/COLUMNS["fit"], ["day", "baseline", "training_weekdays"])
    scores = read_table(source/COLUMNS["score"], ["day", "baseline", "training_weekdays", "horizon_hours"])
    components = read_table(source/COLUMNS["component"], ["day", "baseline", "training_weekdays", "hour"])
    if fit.empty or scores.empty or components.empty:
        raise FileNotFoundError("Run sweep_weekday_horizon.py first: parameter_estimates.csv, oracle_horizon_scores.csv and forecast_components.csv are required.")
    config_path = source/"configuration.json"
    if not config_path.is_file(): raise FileNotFoundError(f"Missing sweep configuration: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    steps_per_hour = int(config["steps_per_hour"])
    keys = ["day", "baseline", "training_weekdays"]
    selected = set(baselines)
    fit = fit[fit["baseline"].isin(selected)].copy()
    scores = scores[scores["baseline"].isin(selected)].copy()
    components = components[components["baseline"].isin(selected)].copy()
    theta = fit[keys+["theta_F", "theta_R"]].copy()
    score_cols = keys+["horizon_hours", "oracle_F0", "oracle_R0"]
    merged = components.merge(scores[score_cols], on=keys, how="inner", validate="many_to_many")
    merged = merged.loc[merged["hour"] < merged["horizon_hours"]].copy()
    merged = merged.merge(theta, on=keys, how="inner", validate="many_to_one")
    if merged.empty: raise ValueError("No eligible sweep predictions after joining saved fit, scores and components.")
    merged["predicted"] = (merged["prediction_at_initial_zero"]
                           + merged["sensitivity_F0"]*merged["oracle_F0"]
                           + merged["sensitivity_R0"]*merged["oracle_R0"])
    merged["error"] = merged["observed"]-merged["predicted"]
    merged["unit_process_variance"] = [
        unit_process_variance(float(rf), float(rr), steps_per_hour, int(h))
        for rf, rr, h in zip(merged["theta_F"], merged["theta_R"], merged["hour"])
    ]
    numeric = ["observed", "predicted", "error", "unit_process_variance"]
    if not np.isfinite(merged[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite saved sweep forecasts or covariance response.")
    merged["day"] = pd.to_datetime(merged["day"]).dt.normalize()
    merged["training_weekdays"] = merged["training_weekdays"].astype(int)
    merged["horizon_hours"] = merged["horizon_hours"].astype(int)
    return merged, config


def fit_variance(train):
    """min_{tau2,q>=0} sum (error**2-tau2-q*K)**2, with both variances nonnegative."""
    if train.empty: return None
    K = train["unit_process_variance"].to_numpy(dtype=float)
    error = train["error"].to_numpy(dtype=float)
    if np.ptp(K) < 1e-12:
        # At horizon 1 only hour zero is observed; q is unidentifiable.
        return float(np.mean(error**2)), 0.0, False
    A = np.column_stack((np.ones(len(train)), K))
    tau2, q = nnls(A, error**2)[0]
    return float(tau2), float(q), True


def distribution_similarity(error, variance):
    error = np.asarray(error, dtype=float)
    variance = np.asarray(variance, dtype=float)
    valid = np.isfinite(error) & np.isfinite(variance) & (variance > 1e-14)
    if np.count_nonzero(valid) < 3: return np.nan
    return float(1.0-kstest(error[valid]/np.sqrt(variance[valid]), "norm").statistic)


def summarize_grid(frame, min_days=1):
    """In-sample approximation: calibrate and score diffusion on identical saved residuals."""
    records, fitted_rows = [], []
    for (baseline, N, k), part in frame.groupby(["baseline", "training_weekdays", "horizon_hours"], sort=True):
        n_days = int(part["day"].nunique())
        if n_days < int(min_days): continue
        fitted = fit_variance(part)
        if fitted is None: continue
        tau2, q, identifiable = fitted
        error = part["error"].to_numpy(dtype=float)
        kernel = part["unit_process_variance"].to_numpy(dtype=float)
        variance = np.maximum(tau2 + q*kernel, 0.0)
        observed_rmse = float(np.sqrt(np.mean(error**2)))
        reported_rmse = float(np.sqrt(np.mean(variance)))
        similarity = distribution_similarity(error, variance)
        records.append({
            "baseline":baseline, "training_weekdays":int(N), "horizon_hours":int(k),
            "days":n_days, "forecasts":len(part),
            "observation_variance":tau2, "observation_sigma":float(np.sqrt(tau2)),
            "process_variance_rate":q, "process_sigma":float(np.sqrt(q)),
            "process_identifiable":bool(identifiable),
            "reported_rmse":reported_rmse, "observed_rmse":observed_rmse,
            "rmse_error":reported_rmse-observed_rmse,
            "rmse_absolute_error":abs(reported_rmse-observed_rmse),
            "distribution_similarity":similarity, "ks_distance":1.0-similarity if np.isfinite(similarity) else np.nan,
            "coverage_95":float(np.mean(np.abs(error)<=1.959963984540054*np.sqrt(variance))),
            "demand_bias":float(np.mean(error)),
        })
        details = part[["day", "baseline", "training_weekdays", "horizon_hours", "hour", "observed", "predicted", "error", "unit_process_variance"]].copy()
        details["observation_variance"] = tau2
        details["process_variance"] = q*kernel
        details["estimated_variance"] = variance
        details["estimated_std"] = np.sqrt(variance)
        fitted_rows.append(details)
    return pd.DataFrame(records), pd.concat(fitted_rows, ignore_index=True) if fitted_rows else pd.DataFrame()


def plot_grid(summary, output, baselines=BASELINES, N_values=N_VALUES, k_values=K_VALUES, start_day=START_DAY, end_day=END_DAY):
    fig, axes = plt.subplots(2, 2, figsize=(19, 10), sharex=True, sharey=True, constrained_layout=True)
    for row, (metric, label) in enumerate((
            ("reported_rmse", "Predicted demand RMSE (normalized demand)"),
            ("distribution_similarity", "Similarity to Gaussian error law (1 - KS)"))):
        grids = []
        for name in baselines:
            filtered = summary[summary["baseline"] == name]
            pivot = filtered.pivot(index="training_weekdays", columns="horizon_hours", values=metric)
            pivot = pivot.reindex(index=N_values, columns=k_values)
            grids.append(pivot.to_numpy(dtype=float))
        finite = np.concatenate([g[np.isfinite(g)] for g in grids])
        vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        if metric == "distribution_similarity": vmin, vmax = 0.0, 1.0
        if vmin >= vmax: vmax = vmin+1e-10
        for col, (name, grid) in enumerate(zip(baselines, grids)):
            ax = axes[row, col]
            im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto",
                           interpolation="nearest", vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(f"{'Reported RMSE' if row == 0 else 'Distribution similarity'} | {name.replace('_', ' ')}")
            ax.set_xticks(np.arange(len(k_values)), labels=k_values, fontsize=8)
            ax.set_yticks(np.arange(len(N_values)), labels=N_values)
            if row == 1: ax.set_xlabel("Prediction horizon k (hours scored)")
            if col == 0: ax.set_ylabel("Backlog N (preceding matching weekdays)")
        fig.colorbar(im, ax=axes[row, :], label=label, shrink=0.94, pad=0.012)
    start, end = pd.Timestamp(start_day), pd.Timestamp(end_day)
    date_label = f"{start.month}/{start.day}-{end.month}/{end.day}/{end.year}"
    fig.suptitle(f"ARM constant demand error + Brownian diffusion | {date_label} | in-sample")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args):
    baselines = tuple(args.baselines)
    if len(baselines) != 2: raise ValueError("The reference 2x2 heatmap requires exactly two baseline methods.")
    source = Path(args.source) if args.source else SWEEP_OUTPUT/args.group.lower()
    forecasts, config = load_sweep(source, baselines)
    forecasts = forecasts.loc[forecasts["training_weekdays"].between(args.min_weekdays, args.max_weekdays)
                              & forecasts["horizon_hours"].between(args.min_horizon, args.max_horizon)].copy()
    if forecasts.empty: raise RuntimeError("No saved sweep forecasts in requested N and k ranges.")
    backlogs = tuple(range(args.min_weekdays, args.max_weekdays+1))
    horizons = tuple(range(args.min_horizon, args.max_horizon+1))
    forecasts, matched_days = same_period(forecasts, args.start_day, args.end_day, baselines, backlogs, horizons)
    summary, fitted = summarize_grid(forecasts, args.min_days)
    if summary.empty: raise RuntimeError("No N,k cells contained sufficient in-sample demand residuals.")
    output = Path(args.output) if args.output else OUTPUT/args.group.lower()
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output/"diffusion_grid.csv", index=False)
    fitted.to_csv(output/"diffusion_fitted_forecasts.csv", index=False)
    plot_grid(summary, output/"diffusion_heatmaps_2x2.png", baselines, backlogs, horizons,
              args.start_day, args.end_day)
    print(f"source={source} | group={args.group} | steps_per_hour={config['steps_per_hour']}")
    print(f"period={args.start_day} to {args.end_day} | matched days={len(matched_days)}")
    print(f"saved forecast rows={len(forecasts)} | in-sample fitted cells={len(summary)}")
    print("theta_F, theta_R were loaded from the original WM4 sweep and never refitted.")
    print("Diffusion parameters fitted and assessed on the SAME demand residuals; no train/test split.")
    print("NOTE: the saved sweep uses target-horizon oracle [F0,R0]; its residuals are in-sample with respect to state, not causal forecasts.")
    print("NOTE: k=1 contains zero accumulated process variance and cannot identify sigma; q=0 is a convention there.")
    print(f"heatmap={output/'diffusion_heatmaps_2x2.png'}")
    return summary, fitted


def main():
    parser = ArgumentParser(description="Estimate fixed-theta WM4 demand observation and Brownian variances on the existing weekday/horizon sweep, then plot 2x2 heatmaps.")
    parser.add_argument("--group", choices=("ALL", "MED"), default="ALL")
    parser.add_argument("--source", default=None, help="Existing sweep output directory with saved fit/scores/components CSVs.")
    parser.add_argument("--baselines", nargs=2, choices=("statistical_output",)+BASELINES, default=list(BASELINES))
    parser.add_argument("--start-day", default=START_DAY, help="First scored calendar day (inclusive).")
    parser.add_argument("--end-day", default=END_DAY, help="Last scored calendar day (inclusive).")
    parser.add_argument("--min-weekdays", type=int, default=2)
    parser.add_argument("--max-weekdays", type=int, default=14)
    parser.add_argument("--min-horizon", type=int, default=1)
    parser.add_argument("--max-horizon", type=int, default=24)
    parser.add_argument("--min-days", type=int, default=1, help="Minimum number of fitted days per N,k cell.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.min_weekdays < 2 or args.max_weekdays < args.min_weekdays: parser.error("Invalid backlog N range.")
    if args.min_days < 1: parser.error("--min-days must be >= 1.")
    if args.min_horizon < 1 or args.max_horizon > 24 or args.max_horizon < args.min_horizon: parser.error("Invalid horizon k range.")
    return run(args)


if __name__ == "__main__": main()
