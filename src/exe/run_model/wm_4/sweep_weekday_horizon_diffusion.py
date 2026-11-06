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
    """Match target calendar days across both baseline methods, every N and every k."""
    start, end = pd.Timestamp(start_day).normalize(), pd.Timestamp(end_day).normalize()
    if end < start: raise ValueError("end-day must be on or after start-day.")
    data = frame.loc[frame["day"].between(start, end)].copy()
    if data.empty: raise ValueError(f"No saved sweep configurations in {start:%Y-%m-%d} to {end:%Y-%m-%d}.")
    required = {(baseline, n, k) for baseline in baselines for n in backlogs for k in horizons}
    days = pd.DatetimeIndex([
        day for day, part in data.groupby("day", sort=True)
        if required.issubset(set(zip(part["baseline"], part["training_weekdays"], part["horizon_hours"])))
    ])
    if days.empty: raise ValueError("No fully matched target days for the requested baselines, backlogs and horizons.")
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
    """Read saved drift coefficients and their historical fitted initial states, not oracle states."""
    source = Path(source)
    fit = read_table(source/COLUMNS["fit"], ["day", "baseline", "training_weekdays"])
    states = read_table(source/COLUMNS["state"], ["day", "baseline", "training_weekdays", "historical_day"])
    scores = read_table(source/COLUMNS["score"], ["day", "baseline", "training_weekdays", "horizon_hours"])
    if fit.empty or states.empty or scores.empty:
        raise FileNotFoundError("First run sweep_weekday_horizon.py: parameter_estimates.csv, historical_initial_states.csv and oracle_horizon_scores.csv are required.")
    config_path = source/"configuration.json"
    if not config_path.is_file(): raise FileNotFoundError(f"Missing sweep configuration: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = set(baselines)
    fit = fit[fit["baseline"].isin(selected)].copy()
    states = states[states["baseline"].isin(selected)].copy()
    scores = scores[scores["baseline"].isin(selected)].copy()
    return fit, states, scores, config


def historical_trajectories(frame, fits, states, steps_per_hour):
    """Rebuild historical full trajectories using theta and [F0,R0] from the original fit.

    The target day is NEVER simulated or used to fit diffusion. Its date indexes
    the independent historical fit to which these in-sample trajectories belong.
    """
    from src.exe.run_model.wm_4.moving_horizon_common import HOURS, window_data
    from src.exe.run_model.wm_4.sweep_weekday_horizon import baseline_from_history
    from src.tools.impute_hourly import impute_hourly_days

    keys = ["day", "baseline", "training_weekdays"]
    rows = []
    sample = np.arange(HOURS, dtype=int)*int(steps_per_hour)
    states = states.copy()
    states["historical_day"] = pd.to_datetime(states["historical_day"])
    for fit in fits.itertuples(index=False):
        day = pd.Timestamp(fit.day)
        baseline_name, N = str(fit.baseline), int(fit.training_weekdays)
        selection = states.loc[(states["day"] == day) &
                               (states["baseline"] == baseline_name) &
                               (states["training_weekdays"] == N)].sort_values("historical_day")
        if len(selection) != N:
            raise ValueError(f"Expected {N} saved historical states for {day} {baseline_name}, found {len(selection)}.")
        history = pd.DatetimeIndex(selection["historical_day"])
        if hasattr(fit, "training_days") and isinstance(fit.training_days, str):
            expected = tuple(pd.to_datetime(fit.training_days.split(";")))
            if tuple(history) != expected:
                raise ValueError(f"Historical initial states do not match parameter-fit training days: {day} {baseline_name}.")
        training, _ = impute_hourly_days(frame, days=history, columns=("I", "U", "Temp"), require_days=True)
        baseline = baseline_from_history(training, history, baseline_name)
        theta = np.array([fit.theta_F, fit.theta_U, fit.theta_R], dtype=float)
        for row in selection.itertuples(index=False):
            historical_day = pd.Timestamp(row.historical_day)
            model, data, _, observed = window_data(training, historical_day, baseline, steps_per_hour)
            initial = np.array([row.historical_F0, row.historical_R0], dtype=float)
            trajectory = model._solve_deterministic(initial, data["U"], data["PE"],
                                                    data["Time"], data["O"], theta)
            predicted = np.asarray(trajectory["I"], dtype=float)[sample]
            kernel = model.process_variance_kernel({"r":theta}, sample).sum(axis=1)
            if not np.isfinite(predicted).all() or not np.isfinite(kernel).all():
                raise ValueError(f"Nonfinite historical fitted trajectory: {day} {historical_day}.")
            rows.extend({
                "day":day, "historical_day":historical_day, "baseline":baseline_name,
                "training_weekdays":N, "hour":int(h), "observed":float(obs),
                "predicted":float(pred), "error":float(obs-pred),
                "unit_process_variance":float(K),
            } for h, obs, pred, K in zip(range(HOURS), observed, predicted, kernel))
    return pd.DataFrame(rows)


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


def summarize_grid(frame, min_days=1, horizons=K_VALUES):
    """Fit variance on each full in-sample historical collection; assess cumulative hourly profiles."""
    fitted_parts, evaluations = [], []
    for (day, baseline, N), group in frame.groupby(["day", "baseline", "training_weekdays"], sort=True):
        estimated = fit_variance(group)
        if estimated is None: continue
        tau2, q, identifiable = estimated
        part = group.copy()
        part["observation_variance"] = tau2
        part["process_variance_rate"] = q
        part["process_variance"] = q*part["unit_process_variance"]
        part["estimated_variance"] = np.maximum(tau2+part["process_variance"], 0.0)
        part["estimated_std"] = np.sqrt(part["estimated_variance"])
        part["process_identifiable"] = bool(identifiable)
        fitted_parts.append(part)
    if not fitted_parts: return pd.DataFrame(), pd.DataFrame()
    fitted = pd.concat(fitted_parts, ignore_index=True)
    for (baseline, N), group in fitted.groupby(["baseline", "training_weekdays"], sort=True):
        n_days = int(group["day"].nunique())
        if n_days < min_days: continue
        for k in horizons:
            part = group.loc[group["hour"] < k]
            if part.empty: continue
            error = part["error"].to_numpy(dtype=float)
            variance = part["estimated_variance"].to_numpy(dtype=float)
            observed_rmse = float(np.sqrt(np.mean(error**2)))
            reported_rmse = float(np.sqrt(np.mean(variance)))
            hourly = part.groupby("hour", sort=True).agg(
                error_mse=("error", lambda e: float(np.mean(np.asarray(e, dtype=float)**2))),
                expected_mse=("estimated_variance", "mean"),
            )
            # Unlike the pooled RMSE, the hourly-profile difference is not forced
            # to zero by the intercept in a two-parameter variance regression.
            hourly_rmse_difference = float(np.mean(np.abs(
                np.sqrt(hourly["error_mse"].to_numpy()) -
                np.sqrt(hourly["expected_mse"].to_numpy()))))
            similarity = distribution_similarity(error, variance)
            evaluations.append({
                "baseline":baseline, "training_weekdays":int(N), "horizon_hours":int(k),
                "days":n_days, "historical_trajectories":len(part[["day", "historical_day"]].drop_duplicates()),
                "observations":len(part),
                "observation_variance_mean":float(part["observation_variance"].mean()),
                "process_variance_rate_mean":float(part["process_variance_rate"].mean()),
                "process_identifiable_pct":float(100*part.groupby("day")["process_identifiable"].first().mean()),
                "reported_rmse":reported_rmse, "observed_rmse":observed_rmse,
                "rmse_error":reported_rmse-observed_rmse,
                "rmse_pooled_absolute_error":abs(reported_rmse-observed_rmse),
                "rmse_absolute_error":hourly_rmse_difference,
                "distribution_similarity":similarity,
                "ks_distance":1.0-similarity if np.isfinite(similarity) else np.nan,
                "coverage_95":float(np.mean(np.abs(error)<=1.959963984540054*np.sqrt(variance))),
                "demand_bias":float(np.mean(error)),
            })
    return pd.DataFrame(evaluations), fitted


def plot_grid(summary, output, baselines=BASELINES, N_values=N_VALUES, k_values=K_VALUES, start_day=START_DAY, end_day=END_DAY):
    fig, axes = plt.subplots(2, 2, figsize=(19, 10), sharex=True, sharey=True, constrained_layout=True)
    for row, (metric, label) in enumerate((
            ("rmse_absolute_error", "Mean hourly RMSE difference (normalized demand; 0 is ideal)"),
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
        else: vmin = 0.0
        if vmin >= vmax: vmax = vmin+1e-10
        for col, (name, grid) in enumerate(zip(baselines, grids)):
            ax = axes[row, col]
            im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto",
                           interpolation="nearest", vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(f"{'Mean hourly RMSE difference' if row == 0 else 'Distribution similarity'} | {name.replace('_', ' ')}")
            ax.set_xticks(np.arange(len(k_values)), labels=k_values, fontsize=8)
            ax.set_yticks(np.arange(len(N_values)), labels=N_values)
            if row == 1: ax.set_xlabel("Historical evaluation horizon k (hours scored)")
            if col == 0: ax.set_ylabel("Backlog N (preceding matching weekdays)")
        fig.colorbar(im, ax=axes[row, :], label=label, shrink=0.94, pad=0.012)
    start, end = pd.Timestamp(start_day), pd.Timestamp(end_day)
    date_label = f"{start.month}/{start.day}-{end.month}/{end.day}/{end.year}"
    fig.suptitle(f"ARM constant demand error + Brownian diffusion | {date_label} | in-sample trajectories")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args):
    from src.exe.run_model.wm_4.moving_horizon_common import group_frame

    baselines = tuple(args.baselines)
    if len(baselines) != 2: raise ValueError("The reference 2x2 heatmap requires exactly two baseline methods.")
    source = Path(args.source) if args.source else SWEEP_OUTPUT/args.group.lower()
    fits, states, scores, config = load_sweep(source, baselines)
    backlogs = tuple(range(args.min_weekdays, args.max_weekdays+1))
    horizons = tuple(range(args.min_horizon, args.max_horizon+1))
    scores = scores.loc[scores["training_weekdays"].isin(backlogs) & scores["horizon_hours"].isin(horizons)]
    _, matched_days = same_period(scores, args.start_day, args.end_day, baselines, backlogs, horizons)
    fits = fits.loc[fits["day"].isin(matched_days) & fits["training_weekdays"].isin(backlogs)]
    if fits.empty: raise RuntimeError("No complete saved historical parameter fits available.")
    print(f"[ARM diffusion] Replaying {len(fits)} historical fitted parameter sets.", flush=True)
    fitted_trajectories = historical_trajectories(group_frame(args.group), fits, states, int(config["steps_per_hour"]))
    summary, fitted = summarize_grid(fitted_trajectories, args.min_days, horizons)
    if summary.empty: raise RuntimeError("No matched in-sample trajectory residuals available for fitting.")
    output = Path(args.output) if args.output else OUTPUT/args.group.lower()
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output/"diffusion_grid.csv", index=False)
    fitted.to_csv(output/"diffusion_fitted_trajectories.csv", index=False)
    plot_grid(summary, output/"diffusion_heatmaps_2x2.png", baselines, backlogs, horizons,
              args.start_day, args.end_day)
    print(f"source={source} | group={args.group} | steps_per_hour={config['steps_per_hour']}")
    print(f"period={args.start_day} to {args.end_day} | matched target fit days={len(matched_days)}")
    print(f"historical trajectory samples={len(fitted)} | evaluated N,k cells={len(summary)}")
    print("In-sample: theta and each historical [F0,R0] are reused from the existing drift QP.")
    print("Diffusion is calibrated once on each FULL 24h historical fit; no target-day oracle states are used.")
    print("RMSE heatmaps show mean absolute HOURLY RMSE discrepancy; pooled RMSE discrepancy is also saved.")
    print(f"heatmap={output/'diffusion_heatmaps_2x2.png'}")
    return summary, fitted


def main():
    parser = ArgumentParser(description="Fit ARM constant observation and Brownian variances to saved in-sample historical fitted-state trajectories, then plot RMSE and KS heatmaps.")
    parser.add_argument("--group", choices=("ALL", "MED"), default="ALL")
    parser.add_argument("--source", default=None, help="Existing sweep output directory with saved fit/states/scores CSVs.")
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
