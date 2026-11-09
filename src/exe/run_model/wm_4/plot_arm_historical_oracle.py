"""Fit ARM on N matching historical k-hour intervals and plot an oracle target.

Run from the repository root:
    python -m src.exe.run_model.wm_4.plot_arm_historical_oracle \
        --origin "2022-05-05 00:00" --n 7 --k 24 --steps-per-hour 60

Both baselines use exactly the same historical timestamps, but independently
estimated baselines, ARM drift coefficients, and stochastic variance parameters.
The target oracle uses target observations to optimize [F0,R0]; it is NOT a
causal forecast. All noise parameters are estimated on historical data only.
"""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.exe.run_model.wm_4.moving_horizon_common import group_frame, metrics
from src.exe.run_model.wm_4.predict_all_moving_horizon import historical_weekdays
from src.exe.run_model.wm_4.sweep_weekday_horizon import (
    baseline_from_history, historical_calendar_days, interval_window,
)
from src.exe.run_model.wm_4.sweep_weekday_horizon_diffusion import fit_variance
from src.params.wm_4 import joint_qp_params
from src.tools.impute_hourly import impute_hourly_days


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "historical_oracle"
BASELINES = ("weekly_average", "weekly_temperature")
COLORS = {"weekly_average": "tab:blue", "weekly_temperature": "tab:orange"}
LABELS = {"weekly_average": "Weekly Average", "weekly_temperature": "Weekly Temperature"}
Z68, Z95 = float(norm.ppf(0.84)), float(norm.ppf(0.975))


def score(observed, predicted, predicted_std, *, historical=False):
    """Demand metrics and sum of absolute hourly predicted-vs-observed RMSE gaps.

    Arrays are [N,k] for historical or [k] for oracle. For each hour j,
    historical observed RMSE=sqrt(mean_n(e[n,j]**2)); oracle observed
    RMSE=abs(e[j]) because the target is a single trajectory.
    Predicted RMSE at hour j is the stochastic model's marginal std.
    """
    observed, predicted = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    predicted_std = np.asarray(predicted_std, dtype=float).reshape(-1)
    if observed.shape != predicted.shape or observed.ndim != (2 if historical else 1):
        raise ValueError("Historical observations must be N x k and oracle observations must be length k.")
    if observed.shape[-1] != predicted_std.size or np.any(predicted_std < 0):
        raise ValueError("Predicted standard deviation must match the selected horizon.")
    err = predicted-observed
    observed_hourly_rmse = np.sqrt(np.mean(err**2, axis=0)) if historical else np.abs(err)
    demand = metrics(observed.reshape(-1), predicted.reshape(-1))
    return {**demand,
        "rmse_discrepancy_sum": float(np.abs(predicted_std-observed_hourly_rmse).sum()),
        "rmse_discrepancy_mean": float(np.abs(predicted_std-observed_hourly_rmse).mean()),
        "observed_hourly_rmse": observed_hourly_rmse,
    }


def fit_baseline(frame, training, history, origin, n, k, steps_per_hour, name):
    """One original N,k joint drift QP, historical free-run variance QP, oracle QP."""
    total_start = perf_counter()
    baseline = baseline_from_history(training, history, name, k)
    intervals = [interval_window(training, past, baseline, steps_per_hour, k)
                 for past in history]
    model, forecast, _, oracle_observed = interval_window(frame, origin, baseline, steps_per_hour, k)
    sample = np.arange(k, dtype=int)*steps_per_hour

    drift_start = perf_counter()
    fit = joint_qp_params(model).fit_many([item[1] for item in intervals])
    drift_seconds = perf_counter()-drift_start
    theta = np.asarray(fit["r"], dtype=float)
    historical_observed = np.vstack([item[3] for item in intervals])
    historical_predicted = np.vstack([
        np.asarray(state["I"], dtype=float)[sample]
        for state in fit["deterministic_states"]
    ])
    kernel = model.process_variance_kernel({"r": theta}, sample).sum(axis=1)

    diffusion_start = perf_counter()
    historical_errors = historical_observed-historical_predicted
    variance_data = pd.DataFrame({
        "error": historical_errors.reshape(-1),
        "unit_process_variance": np.tile(kernel, n),
    })
    tau2, sigma2, identifiable = fit_variance(variance_data)
    variance = np.maximum(tau2+sigma2*kernel, 0.0)
    predicted_std = np.sqrt(variance)
    diffusion_seconds = perf_counter()-diffusion_start

    oracle_start = perf_counter()
    base_state = model._solve_deterministic(
        [0.0, 0.0], forecast["U"], forecast["PE"], forecast["Time"], forecast["O"], theta)
    zero_demand = np.asarray(base_state["I"], dtype=float)[sample]
    sensitivity = np.asarray(model.sensitivity({"r": theta})["input"])[sample]
    oracle_initial = np.asarray(model._least_squares_qp(
        sensitivity, np.asarray(oracle_observed)-zero_demand,
        lb=np.array([-1.0, -1.0]), ub=np.array([1.0, 1.0]), ridge=model._FIT_RIDGE), dtype=float)
    oracle_state = model._solve_deterministic(
        oracle_initial, forecast["U"], forecast["PE"], forecast["Time"], forecast["O"], theta)
    oracle_predicted = np.asarray(oracle_state["I"], dtype=float)[sample]
    oracle_seconds = perf_counter()-oracle_start

    historical_score = score(historical_observed, historical_predicted, predicted_std, historical=True)
    oracle_score = score(oracle_observed, oracle_predicted, predicted_std)
    curves = []
    for index, (past, obs, pred) in enumerate(zip(history, historical_observed, historical_predicted)):
        curves.append({"kind": "historical", "interval": index, "origin": past,
                       "observed": obs, "predicted": pred, "std": predicted_std})
    curves.append({"kind": "oracle", "interval": n, "origin": origin,
                   "observed": np.asarray(oracle_observed), "predicted": oracle_predicted,
                   "std": predicted_std})
    return {
        "name": name, "theta": theta, "initial_states": np.asarray(fit["init_cond"]),
        "oracle_initial": oracle_initial, "tau2": tau2, "sigma2": sigma2,
        "identifiable": bool(identifiable), "kernel": kernel, "std": predicted_std,
        "historical_score": historical_score, "oracle_score": oracle_score,
        "curves": curves, "drift_seconds": drift_seconds,
        "diffusion_seconds": diffusion_seconds, "oracle_seconds": oracle_seconds,
        "fit_seconds": perf_counter()-total_start-oracle_seconds,
        "total_seconds": perf_counter()-total_start,
    }


def plot_comparison(results, history, origin, k, path, show=False):
    """Single BM1-style panel: historical intervals, then oracle, two bands."""
    n = len(history)
    gap = max(1.0, 0.08*k)
    width = max(17.0, 2.5*(n+1))
    fig, ax = plt.subplots(figsize=(width, 7.0))
    fig.subplots_adjust(left=0.046, right=0.99, top=0.88, bottom=0.18)
    ax.axvspan(n*(k+gap)-gap/2, (n+1)*(k+gap), facecolor="gray", alpha=0.055, zorder=0)
    xticks, ticklabels, all_observed = [], [], []
    for j, at in enumerate([*history, origin]):
        x = j*(k+gap)+np.arange(k, dtype=float)
        if j:
            ax.axvline(j*(k+gap)-gap/2, color="0.8", ls=":", lw=0.9)
        xticks.append(float(np.mean(x)))
        ticklabels.append(f"{'Oracle prediction' if j == n else f'Historical {j+1}'}\n{at:%d %b %Y %H:%M}")
        observed = results[BASELINES[0]]["curves"][j]["observed"]
        all_observed.extend(observed)
        for name in BASELINES:
            curve = results[name]["curves"][j]
            mean, std = curve["predicted"], curve["std"]
            color = COLORS[name]
            ax.fill_between(x, mean-Z95*std, mean+Z95*std, color=color, alpha=0.075, lw=0)
            ax.fill_between(x, mean-Z68*std, mean+Z68*std, color=color, alpha=0.15, lw=0)
            ax.plot(x, mean, color=color, ls="--" if j == n else "-", lw=1.9,
                    label=f"{LABELS[name]} — ARM mean" if j == 0 else None, zorder=3)
        ax.plot(x, observed, color="0.12", marker="o", markersize=2.6, lw=1.6,
                label="Observed demand" if j == 0 else None, zorder=4)
    ax.set_xticks(xticks, ticklabels, fontsize=9 if n < 9 else 7.5)
    ax.set_ylabel("Normalized aggregate demand")
    ax.set_xlabel("Fitting / prediction interval (hours concatenated; weekly gaps omitted)")
    ax.grid(axis="y", color="0.88", lw=0.75)
    ax.grid(axis="x", color="0.91", ls=":", lw=0.6)
    ax.set_xlim(-0.5, n*(k+gap)+k-0.5)
    ax.set_title(f"ARM: {n} historical fits and one oracle prediction | k={k} h | ALL | "
                 f"{origin:%Y-%m-%d %H:%M}")
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor="gray", alpha=0.15, label="Central 68% demand interval"),
                Patch(facecolor="gray", alpha=0.075, label="Central 95% demand interval")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
              ncol=3, fontsize=9, framealpha=0.95)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)


def save_tables(results, output):
    rows, summaries = [], []
    for name in BASELINES:
        result = results[name]
        for curve in result["curves"]:
            for hour, (obs, pred, std) in enumerate(zip(curve["observed"], curve["predicted"], curve["std"])):
                rows.append({"baseline": name, "kind": curve["kind"],
                             "interval": curve["interval"], "origin": str(curve["origin"]),
                             "hour": hour, "observed": float(obs), "predicted": float(pred),
                             "error_pred_minus_obs": float(pred-obs),
                             "predicted_rmse": float(std), "lower_68": float(pred-Z68*std),
                             "upper_68": float(pred+Z68*std), "lower_95": float(pred-Z95*std),
                             "upper_95": float(pred+Z95*std)})
        for part in ("historical", "oracle"):
            m = result[f"{part}_score"]
            summaries.append({"baseline": name, "kind": part,
                              "rmse": m["rmse"], "mae": m["mae"], "bias": m["bias"],
                              "abs_hourly_rmse_discrepancy_sum": m["rmse_discrepancy_sum"],
                              "abs_hourly_rmse_discrepancy_mean": m["rmse_discrepancy_mean"],
                              "tau": np.sqrt(result["tau2"]), "sigma": np.sqrt(result["sigma2"]),
                              "theta_F": result["theta"][0], "theta_U": result["theta"][1],
                              "theta_R": result["theta"][2], "fit_seconds": result["fit_seconds"],
                              "total_seconds": result["total_seconds"]})
    pd.DataFrame(rows).to_csv(output/"trajectories.csv", index=False)
    pd.DataFrame(summaries).to_csv(output/"metrics.csv", index=False)


def run(origin, n=7, k=24, steps_per_hour=60, output=None, show=False):
    started = perf_counter()
    origin, n, k, steps_per_hour = pd.Timestamp(origin), int(n), int(k), int(steps_per_hour)
    if pd.isna(origin) or origin != origin.floor("h") or origin.tzinfo is not None:
        raise ValueError("--origin must be a timezone-naive timestamp aligned to the hour.")
    if not 2 <= n <= 14 or not 1 <= k <= 24 or steps_per_hour < 2:
        raise ValueError("Require 2 <= N <= 14, 1 <= k <= 24, steps-per-hour >= 2.")
    output = Path(output) if output is not None else OUTPUT/f"{origin:%Y%m%d_%H%M}_N{n}_k{k}"
    history = historical_weekdays(origin, n)
    frame = group_frame("ALL")
    training, imputed = impute_hourly_days(
        frame, days=historical_calendar_days(history, k),
        columns=("I", "U", "Temp"), require_days=True)
    results = {name: fit_baseline(frame, training, history, origin, n, k, steps_per_hour, name)
               for name in BASELINES}
    output.mkdir(parents=True, exist_ok=True)
    plot_path = output/"arm_historical_oracle.png"
    plot_comparison(results, history, origin, k, plot_path, show=show)
    save_tables(results, output)
    total_seconds = perf_counter()-started
    for name in BASELINES:
        fit = results[name]
        print(f"\n{LABELS[name]} | N={n}, k={k} h, origin={origin:%Y-%m-%d %H:%M}")
        print("  drift theta [F, U, R]="+np.array2string(fit["theta"], precision=7))
        print(f"  tau={np.sqrt(fit['tau2']):.8f}, sigma={np.sqrt(fit['sigma2']):.8f}, "
              f"process_identifiable={fit['identifiable']}")
        print("  oracle [F0, R0]="+np.array2string(fit["oracle_initial"], precision=7))
        for part in ("historical", "oracle"):
            m = fit[f"{part}_score"]
            print(f"  {part:10s} demand RMSE={m['rmse']:.8f}, MAE={m['mae']:.8f}, "
                  f"bias(pred-obs)={m['bias']:+.8f}")
            print(f"  {part:10s} Σ|RMSE_pred(t)-RMSE_obs(t)| = {m['rmse_discrepancy_sum']:.8f} "
                  f"(mean per hour={m['rmse_discrepancy_mean']:.8f})")
        print(f"  fit time (baseline + drift + variance)={fit['fit_seconds']:.3f} s; "
              f"oracle optimization={fit['oracle_seconds']:.3f} s; "
              f"baseline total={fit['total_seconds']:.3f} s")
    print(f"\nHistorical imputed hourly slots={imputed['imputed_hourly_slots']}")
    print(f"Complete end-to-end time (data load + both fits + plot + CSV)={total_seconds:.3f} s")
    print(f"Plot: {plot_path}\nData: {output/'trajectories.csv'}\nMetrics: {output/'metrics.csv'}")
    return results


def main():
    parser = ArgumentParser(description="Fit and plot ARM historical N,k intervals and oracle for ALL using both baselines.")
    parser.add_argument("--origin", default="2022-05-05 00:00", help="Target origin, e.g. '2022-05-05 06:00'.")
    parser.add_argument("--n", type=int, default=7, help="Number of preceding corresponding weekdays (2..14).")
    parser.add_argument("--k", type=int, default=24, help="Fitting and prediction horizon in hours (1..24).")
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--output", default=None, help="Output folder for PNG and CSV files.")
    parser.add_argument("--show", action="store_true", help="Display the graph interactively after saving.")
    args = parser.parse_args()
    return run(args.origin, args.n, args.k, args.steps_per_hour, args.output, args.show)


if __name__ == "__main__": main()
