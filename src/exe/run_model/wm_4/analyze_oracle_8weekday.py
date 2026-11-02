"""WM4 full-state oracle [F0,R0] diagnostics, N=8 matching weekdays / H=7.

Historical theta is fitted only on preceding matching weekdays. Both F0/R0
are optimized jointly on the seven TARGET observations using a convex QP.
This is retrospective and must never be presented as an operational forecast.
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, norm, spearmanr

from src.exe.run_model.wm_4.moving_horizon_common import group_frame
from src.exe.run_model.wm_4.sweep_weekday_horizon import (
    BASELINE_NAMES, evaluate_configuration,
)


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "oracle_8weekday_7hour"
N_WEEKDAYS, HORIZON = 8, 7
PARAMETERS = ("theta_F", "theta_R", "theta_U")
CONTRIBUTIONS = ("restoring", "damping", "price")
QUANTILES = (0, 20, 40, 50, 60, 80, 100)
SAVED = {"daily": "daily_metrics.csv", "hourly": "hourly_predictions.csv",
         "states": "historical_initial_states.csv", "skip": "skipped_days.csv"}


def reconstruct_hourly(daily, components, steps_per_hour, horizon=HORIZON):
    """Reconstruct exactly the WM4 Euler oracle-state trajectory, not a proxy.

    WM4 normalized model time is one day. U is piecewise constant per hour.
    F and R do not depend directly on the baseline, only on U and theta.
    """
    sph = int(steps_per_hour)
    if sph < 2 or not 1 <= horizon <= 24: raise ValueError("Invalid numerical resolution/horizon.")
    subset = components.sort_values("hour").iloc[:horizon]
    if len(subset) != horizon or not np.array_equal(subset["hour"].to_numpy(dtype=int), np.arange(horizon)):
        raise ValueError("Missing hourly forecast components.")
    theta_F, theta_R, theta_U = (float(daily[name]) for name in PARAMETERS)
    F, R = float(daily["oracle_F0"]), float(daily["oracle_R0"])
    dt = 1.0/(24*sph)
    result = []
    for part in subset.itertuples(index=False):
        prediction = (float(part.prediction_at_initial_zero)
                      +F*float(part.sensitivity_F0)+R*float(part.sensitivity_R0))
        if not np.isclose(R, prediction-float(part.baseline_value), atol=1e-7, rtol=1e-7):
            raise RuntimeError("Full-state oracle trajectory does not agree with saved WM4 prediction.")
        U = float(part.price)
        cF, cR, cU = theta_F*F, theta_R*R, theta_U*U
        observed = float(part.observed)
        result.append({
            "day": pd.Timestamp(daily["day"]), "hour": int(part.hour),
            "time": pd.Timestamp(daily["day"])+pd.Timedelta(hours=int(part.hour)),
            "baseline": daily["baseline"], "training_weekdays": int(daily["training_weekdays"]),
            "horizon_hours": horizon, "observed": observed,
            "predicted": prediction, "baseline_prediction": float(part.baseline_value),
            "error": prediction-observed, "absolute_error": abs(prediction-observed),
            "baseline_error": float(part.baseline_value)-observed,
            "F": F, "R": R, "U": U,
            "restoring": cF, "damping": cR, "price": cU, "R_dot": cF+cR+cU,
        })
        for _ in range(sph):
            F, R = F+dt*R, R+dt*(theta_F*F+theta_R*R+theta_U*U)
    return result



def consolidate(fit, score, components, states, steps_per_hour, horizon=HORIZON):
    daily = {**fit, **score}
    if (int(daily["training_weekdays"]) != N_WEEKDAYS
            or int(daily["horizon_hours"]) != horizon):
        raise ValueError("The diagnostic requires the chosen historical-window and horizon.")
    hourly = reconstruct_hourly(daily, pd.DataFrame(components), steps_per_hour, horizon)
    errors = np.asarray([row["error"] for row in hourly], dtype=float)
    if not np.isclose(np.sqrt(np.mean(errors**2)), float(daily["rmse"]), atol=1e-8):
        raise RuntimeError("Reconstructed RMSE disagrees with joint-QP oracle score.")
    for component in CONTRIBUTIONS:
        values = np.asarray([row[component] for row in hourly], dtype=float)
        daily[f"{component}_rms"] = float(np.sqrt(np.mean(values**2)))
        daily[f"{component}_mean"] = float(np.mean(values))
        daily[f"{component}_mean_abs"] = float(np.mean(np.abs(values)))
    for name in ("F", "R", "U"):
        values = np.asarray([row[name] for row in hourly], dtype=float)
        daily[f"{name}_rms"] = float(np.sqrt(np.mean(values**2)))
        daily[f"{name}_mean"] = float(np.mean(values))
    return daily, hourly, states


def check_tables(daily, hourly):
    d = daily.copy()
    h = hourly.copy()
    required_daily = {"day", "baseline", "training_weekdays", "horizon_hours",
                      "oracle_F0", "oracle_R0", "historical_R0", "rmse", "mse", "baseline_mse", *PARAMETERS}
    required_hourly = {"day", "hour", "observed", "predicted", "baseline_prediction",
                       "error", "absolute_error", *CONTRIBUTIONS}
    if missing := required_daily-set(d.columns): raise ValueError(f"Missing daily columns: {sorted(missing)}")
    if missing := required_hourly-set(h.columns): raise ValueError(f"Missing hourly columns: {sorted(missing)}")
    d["day"], h["day"] = pd.to_datetime(d["day"]), pd.to_datetime(h["day"])
    d = d.drop_duplicates("day", keep="last").sort_values("day").reset_index(drop=True)
    h = h.drop_duplicates(["day", "hour"], keep="last").sort_values(["day", "hour"]).reset_index(drop=True)
    if not (d["training_weekdays"] == N_WEEKDAYS).all(): raise ValueError("Expected only eight training weekdays.")
    if not (d["horizon_hours"] == HORIZON).all(): raise ValueError("Expected only seven-hour predictions.")
    if d["baseline"].nunique() != 1: raise ValueError("Results cannot mix different baselines.")
    if (h.groupby("day")["hour"].nunique() != HORIZON).any():
        raise ValueError("At least one saved day has incomplete seven-hour predictions.")
    if not set(d["day"]) == set(h["day"]): raise ValueError("Day and hourly results must have identical dates.")
    if not np.isfinite(d[list(PARAMETERS)+["rmse", "mse"]].to_numpy(dtype=float)).all():
        raise ValueError("Saved parameters or performance contain nonfinite values.")
    return d, h


def quantile_table(daily, hourly):
    rows = []
    for hour, group in hourly.groupby("hour"):
        values = group["absolute_error"].to_numpy(dtype=float)
        errors = group["error"].to_numpy(dtype=float)
        row = {"hour": int(hour), "days": len(group),
               "pooled_rmse": float(np.sqrt(np.mean(errors**2))),
               "mae": float(np.mean(values)), "bias": float(np.mean(errors))}
        for q in QUANTILES: row[f"abs_error_q{q:02d}"] = float(np.percentile(values, q))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)


def cumulative_table(hourly):
    matrix = hourly.pivot(index="day", columns="hour", values="error").sort_index()
    matrix = matrix.reindex(columns=range(HORIZON))
    if matrix.isna().any().any(): raise ValueError("Cumulative RMSE requires all seven forecast errors.")
    errors = matrix.to_numpy(dtype=float)
    cumulative = np.sqrt(np.cumsum(errors**2, axis=1) / np.arange(1, HORIZON+1)[None, :])
    rows = []
    for hour in range(HORIZON):
        row = {"hour": hour, "days": len(errors),
               "pooled_cumulative_rmse": float(np.sqrt(np.mean(errors[:, :hour+1]**2)))}
        for q in QUANTILES: row[f"daily_cumulative_rmse_q{q:02d}"] = float(np.percentile(cumulative[:, hour], q))
        rows.append(row)
    return pd.DataFrame(rows)


def hac_test(values, lag=7):
    """Exploratory HAC test of mean=0; not an identification or causal test."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    result = {"n": n, "mean": float(np.mean(values)) if n else np.nan,
              "hac_lag": min(int(lag), max(0, n-1)),
              "hac_se": np.nan, "z": np.nan, "p_value": np.nan,
              "ci95_lower": np.nan, "ci95_upper": np.nan}
    if n < 3: return result
    centered = values-result["mean"]
    effective = min(lag, n-1)
    long_run_variance = float(centered @ centered)/n
    for k in range(1, effective+1):
        long_run_variance += 2*(1-k/(effective+1))*float(centered[k:] @ centered[:-k])/n
    se = np.sqrt(max(long_run_variance, 0.0)/n)
    result["hac_se"] = float(se)
    if se < 1e-14: return result
    result.update({"z": float(result["mean"]/se),
                   "p_value": float(2*norm.sf(abs(result["mean"]/se))),
                   "ci95_lower": float(result["mean"]-1.95996398454*se),
                   "ci95_upper": float(result["mean"]+1.95996398454*se)})
    return result


def parameter_tests(daily, lag):
    frame = daily.sort_values("day")
    rows, correlations = [], []
    tested = [*PARAMETERS, "baseline_beta_C", "oracle_F0", "oracle_R0", "historical_R0",
              "restoring_rms", "damping_rms", "price_rms"]
    for variable in tested:
        if variable not in frame: continue
        values = frame[variable].to_numpy(dtype=float)
        record = {"variable": variable, **hac_test(values, lag)}
        if variable.endswith("_rms"):
            # A nonnegative RMS is not a signed effect: a zero-mean significance
            # test is uninformative. Keep these as descriptive distributions.
            for name in ("z", "p_value", "ci95_lower", "ci95_upper", "hac_se"):
                record[name] = np.nan
        record["std"] = float(np.std(values, ddof=1)) if len(values)>1 else np.nan
        record["min"], record["max"] = float(np.min(values)), float(np.max(values))
        for q in QUANTILES: record[f"q{q:02d}"] = float(np.percentile(values, q))
        rows.append(record)

    for variable in tested:
        if variable not in frame: continue
        for comparison in ("rmse", "baseline_rmse", *PARAMETERS):
            if variable == comparison: continue
            subset = frame[[variable, comparison]].dropna()
            rho, p = (spearmanr(subset[variable], subset[comparison])
                      if len(subset)>=4 and subset[variable].nunique()>1
                      and subset[comparison].nunique()>1 else (np.nan, np.nan))
            correlations.append({"variable": variable, "comparison": comparison,
                                 "n": len(subset), "spearman_rho": rho,
                                 "p_value_exploratory": p})
    weekday_tests = []
    frame = frame.assign(weekday=frame["day"].dt.day_name())
    for variable in ("rmse", *PARAMETERS, "restoring_rms", "damping_rms", "price_rms"):
        group = [part[variable].to_numpy(dtype=float) for _, part in frame.groupby("weekday")
                 if len(part)>=2 and part[variable].nunique()>1]
        if len(group)>=2:
            try: statistic, p = kruskal(*group)
            except ValueError: statistic, p = np.nan, np.nan
        else: statistic, p = np.nan, np.nan
        weekday_tests.append({"variable": variable, "weekdays_tested": len(group),
                              "statistic": statistic, "p_value_exploratory": p})
    return pd.DataFrame(rows), pd.DataFrame(correlations), pd.DataFrame(weekday_tests)


def save_figure(fig, output, filename):
    fig.tight_layout(); fig.savefig(output/filename, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_quantiles(hour, cumulative, daily, output):
    x = hour["hour"].to_numpy(dtype=int)+1
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax = axes[0]
    ax.plot(x, hour["pooled_rmse"], lw=2.2, label="Pooled hourly RMSE")
    for q in (20, 40, 60, 80):
        ax.plot(x, hour[f"abs_error_q{q:02d}"], linestyle="--", lw=1.0, label=f"Absolute-error Q{q}")
    ax.fill_between(x, hour["abs_error_q20"].to_numpy(), hour["abs_error_q80"].to_numpy(), alpha=.12)
    ax.set_ylabel("Normalized demand error"); ax.set_title("Hour-specific error quintiles")
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=.2)
    ax = axes[1]
    ax.plot(x, cumulative["pooled_cumulative_rmse"], lw=2.2, label="Pooled cumulative RMSE")
    for q in (20, 40, 60, 80):
        ax.plot(x, cumulative[f"daily_cumulative_rmse_q{q:02d}"], "--", lw=1, label=f"Per-day cumulative Q{q}")
    ax.fill_between(x, cumulative["daily_cumulative_rmse_q20"].to_numpy(),
                    cumulative["daily_cumulative_rmse_q80"].to_numpy(), alpha=.12)
    ax.set_xlabel("Forecast lead hour (1–7)"); ax.set_ylabel("Normalized demand RMSE")
    ax.set_title("Per-day cumulative RMSE quintiles"); ax.legend(ncol=3, fontsize=8); ax.grid(alpha=.2)
    save_figure(fig, output, "forecast_error_quintiles.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    order = daily.sort_values("day")
    ax.plot(order["day"], order["rmse"], alpha=.7, lw=1.1, label="Daily seven-hour RMSE")
    for q in (20, 40, 60, 80):
        ax.axhline(np.percentile(order["rmse"], q), linestyle="--", lw=1,
                   label=f"Q{q} = {np.percentile(order['rmse'], q):.4f}")
    ax.set_ylabel("Daily oracle RMSE"); ax.set_xlabel("Prediction date")
    ax.set_title("Seven-hour full-state oracle RMSE over the evaluation period")
    ax.legend(ncol=2, fontsize=8); ax.grid(alpha=.2)
    save_figure(fig, output, "daily_rmse_quintiles.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    data = daily["rmse"].to_numpy(dtype=float)
    ax.hist(data, bins=min(30, max(5, int(np.sqrt(len(data))))), alpha=.7)
    for q in (20, 40, 60, 80):
        ax.axvline(np.percentile(data, q), linestyle="--", label=f"Q{q}")
    ax.set_xlabel("Daily seven-hour RMSE"); ax.set_ylabel("Number of days")
    ax.set_title("Distribution of daily oracle-state prediction errors")
    ax.legend(); ax.grid(alpha=.2)
    save_figure(fig, output, "rmse_distribution.png")


def plot_parameters(daily, stats, output):
    chronological = daily.sort_values("day")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, variable in zip(axes, PARAMETERS):
        values = chronological[variable].to_numpy(dtype=float)
        ax.plot(chronological["day"], values, lw=1, alpha=.8)
        for q in (20, 50, 80):
            ax.axhline(float(np.percentile(values, q)), linestyle="--", lw=1, label=f"Q{q}")
        ax.set_ylabel(variable); ax.grid(alpha=.2); ax.legend(ncol=3, fontsize=8)
    axes[0].set_title("WM4 joint-QP parameters: eight matching weekdays")
    axes[-1].set_xlabel("Prediction day")
    save_figure(fig, output, "theta_time_quantiles.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, variable in zip(axes, PARAMETERS):
        values = daily[variable].to_numpy(dtype=float)
        ax.boxplot([values], showmeans=True)
        ax.set_xticks([1]); ax.set_xticklabels([variable])
        ax.grid(alpha=.2, axis="y")
        ax.set_title(f"Median={np.median(values):.4f}")
    fig.suptitle("Parameter distributions (different regressor scales; compare contributions separately)")
    save_figure(fig, output, "theta_distributions.png")

    if "baseline_beta_C" in daily and daily["baseline_beta_C"].notna().all():
        fig, ax = plt.subplots(figsize=(12, 5))
        values = chronological["baseline_beta_C"].to_numpy(dtype=float)
        ax.plot(chronological["day"], values, lw=1, label="Baseline heating coefficient")
        for q in (20, 50, 80):
            ax.axhline(np.percentile(values, q), ls="--", lw=1, label=f"Q{q}")
        ax.set_ylabel("Baseline heating coefficient"); ax.set_xlabel("Prediction day")
        ax.set_title("Eight-weekday fitted baseline temperature-response quintiles")
        ax.grid(alpha=.2); ax.legend()
        save_figure(fig, output, "baseline_heating_coefficient_quintiles.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(PARAMETERS))
    for i, q in enumerate((20, 40, 50, 60, 80)):
        quantiles = [float(np.percentile(daily[var], q)) for var in PARAMETERS]
        ax.plot(x, quantiles, marker="o", label=f"Q{q}")
    ax.set_xticks(x, PARAMETERS); ax.set_ylabel("Estimated coefficient")
    ax.set_title("Empirical parameter quintiles (not effect-size comparisons)")
    ax.legend(); ax.grid(alpha=.2)
    save_figure(fig, output, "theta_quintiles.png")


def plot_state_quantiles(daily, hourly, output):
    days = daily.sort_values("day")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, variable in zip(axes, ("oracle_F0", "oracle_R0", "historical_R0")):
        if variable not in days: continue
        y = days[variable].to_numpy(dtype=float)
        ax.plot(days["day"], y, label=variable, lw=1)
        for q in (20, 50, 80):
            ax.axhline(np.percentile(y, q), ls="--", lw=1, label=f"Q{q}")
        ax.set_ylabel(variable); ax.legend(ncol=4, fontsize=8); ax.grid(alpha=.2)
    axes[-1].set_xlabel("Prediction day")
    fig.suptitle("Forecast oracle F0/R0 versus historical median R0: empirical quantiles")
    save_figure(fig, output, "initial_states_and_damping_quintiles.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    hours = np.arange(HORIZON)
    for variable in ("observed", "predicted", "baseline_prediction"):
        group = hourly.groupby("hour")[variable]
        median = group.median().reindex(hours).to_numpy(dtype=float)
        lower = group.quantile(.2).reindex(hours).to_numpy(dtype=float)
        upper = group.quantile(.8).reindex(hours).to_numpy(dtype=float)
        ax.plot(hours+1, median, label=f"{variable} median", marker="o")
        ax.fill_between(hours+1, lower, upper, alpha=.12)
    ax.set_xlabel("Forecast lead hour (1–7)"); ax.set_ylabel("Normalized demand")
    ax.set_title("Oracle prediction versus observed demand: cross-day 20–80% bands")
    ax.legend(); ax.grid(alpha=.2)
    save_figure(fig, output, "prediction_demand_quintiles.png")


def plot_contributions(daily, hourly, output):
    days = daily.sort_values("day")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, effect in zip(axes, CONTRIBUTIONS):
        values = days[f"{effect}_rms"].to_numpy(dtype=float)
        ax.boxplot([values], showmeans=True)
        ax.set_xticks([1]); ax.set_xticklabels([effect])
        ax.set_title(f"Median RMS={np.median(values):.4f}")
        ax.grid(alpha=.25, axis="y")
    fig.suptitle("Actual dynamical contribution RMS over seven forecast hours")
    save_figure(fig, output, "contribution_rms_quintiles.png")

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, effect in zip(axes, CONTRIBUTIONS):
        groups = hourly.groupby("hour")[effect]
        x = np.array(sorted(groups.groups), dtype=int)+1
        median = groups.median().reindex(x-1).to_numpy(dtype=float)
        q20 = groups.quantile(.2).reindex(x-1).to_numpy(dtype=float)
        q80 = groups.quantile(.8).reindex(x-1).to_numpy(dtype=float)
        ax.plot(x, median, marker="o", label=f"{effect} median")
        ax.fill_between(x, q20, q80, alpha=.2, label="20–80% band")
        ax.axhline(0, color="black", linewidth=.6, alpha=.4)
        ax.set_ylabel("Contribution to dR/dt"); ax.grid(alpha=.25); ax.legend()
    axes[0].set_title("Signed restoring, damping and price contribution quantiles")
    axes[-1].set_xlabel("Forecast lead hour (1–7)")
    save_figure(fig, output, "contribution_hourly_quantiles.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    for effect in CONTRIBUTIONS:
        ax.plot(days["day"], days[f"{effect}_rms"], lw=1, label=f"{effect} RMS")
    ax.set_xlabel("Prediction day"); ax.set_ylabel("RMS contribution to dR/dt")
    ax.set_title("Parameter-dependent contributions: state scales included")
    ax.legend(); ax.grid(alpha=.2)
    save_figure(fig, output, "contribution_rms_time.png")


def plot_residuals(hourly, output):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for effect, label in (("predicted", "WM4 oracle"), ("observed", "Observed"),
                          ("baseline_prediction", "Statistical baseline")):
        groups = hourly.groupby("hour")[effect]
        x = np.arange(HORIZON)+1
        axes[0].plot(x, groups.mean().reindex(np.arange(HORIZON)), label=label, marker=".")
    axes[0].set_ylabel("Average normalized demand"); axes[0].legend(); axes[0].grid(alpha=.25)
    for effect in ("error", "baseline_error"):
        groups = hourly.groupby("hour")[effect]
        x = np.arange(HORIZON)+1
        axes[1].plot(x, np.sqrt(groups.apply(lambda col: np.mean(col.to_numpy()**2)).reindex(np.arange(HORIZON))),
                     label=effect, marker="o")
    axes[1].set_ylabel("Hourly pooled RMSE"); axes[1].set_xlabel("Forecast lead hour (1–7)")
    axes[1].legend(); axes[1].grid(alpha=.25)
    save_figure(fig, output, "demand_and_residual_horizon.png")


def print_report(daily, hourly, stats, tests, weekday_tests, output):
    errors = hourly["error"].to_numpy(dtype=float)
    baseline = hourly["baseline_error"].to_numpy(dtype=float)
    print("\nWM4 fixed configuration: eight preceding corresponding weekdays; seven-hour FULL-STATE oracle [F0,R0].")
    print(f"Baseline: {daily['baseline'].iloc[0]} | evaluated days: {len(daily)} | samples: {len(errors)}")
    print(f"Aggregate oracle RMSE: {np.sqrt(np.mean(errors**2)):.8f} | "
          f"baseline RMSE: {np.sqrt(np.mean(baseline**2)):.8f}")
    print(f"Forecast error bias: {np.mean(errors):.8f}")
    print("\nDaily 7-hour RMSE distribution:")
    print(daily["rmse"].describe(percentiles=[.2,.4,.5,.6,.8]).to_string(float_format=lambda x: f"{x:.8f}"))
    print("\nTheta, baseline heating coefficient, and state parameters (exploratory HAC mean tests):")
    columns = ["variable", "mean", "std", "q20", "q50", "q80", "hac_se", "p_value"]
    print(stats.loc[stats["variable"].isin(PARAMETERS+("baseline_beta_C", "oracle_F0", "oracle_R0", "historical_R0")), columns]
          .to_string(index=False, float_format=lambda x: f"{x:.7f}"))
    print("\nRMS magnitude of actual restoring, damping and price contributions:")
    print(stats.loc[stats["variable"].isin(("restoring_rms", "damping_rms", "price_rms")), columns]
          .to_string(index=False, float_format=lambda x: f"{x:.7f}"))
    print("\nConstraint-sign diagnostics:")
    print(f"theta_F<0: {(daily['theta_F']<0).mean():.1%} | "
          f"theta_R<0: {(daily['theta_R']<0).mean():.1%} | "
          f"theta_U<=0: {(daily['theta_U']<=0).mean():.1%}")
    print(f"theta_U approximately zero (abs<1e-6): {(daily['theta_U'].abs()<1e-6).mean():.1%}")
    if "design_rank" in daily:
        print(f"Rank deficient fits: {int((daily['design_rank']<daily['design_columns']).sum())}/{len(daily)}")
    print("\nWeekday parameter and prediction-quality variation (exploratory Kruskal tests):")
    print(weekday_tests.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    if "imputed_training_hours" in daily:
        print(f"Training missing-hour imputations across overlapping 8-week windows: "
              f"{int(daily['imputed_training_hours'].sum())}")
    print(f"\nSaved CSVs and figures: {output}")
    print("NOTE: Forecast F0 AND R0 are jointly optimized using seven future hourly observations.")
    print("HAC and weekday test p-values are exploratory: overlapping fitting windows and "
          "the oracle can invalidate naive independent-sample interpretations.")
    print("RMS contributions are descriptive nonnegative magnitudes: no zero-mean tests are reported for them.")


def analyze(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(output/SAVED["daily"], parse_dates=["day"])
    if "oracle_R0" not in daily:
        raise ValueError("Cached report uses the old F0-only oracle. Re-run this executable "
                         "or load a NEW full-state horizon sweep with --from-sweep.")
    hourly = pd.read_csv(output/SAVED["hourly"], parse_dates=["day", "time"])
    daily, hourly = check_tables(daily, hourly)
    q = quantile_table(daily, hourly)
    cumulative = cumulative_table(hourly)
    lag = 7
    stats, corr, weekdays = parameter_tests(daily, lag)
    q.to_csv(output/"hourly_error_quintiles.csv", index=False)
    cumulative.to_csv(output/"cumulative_rmse_quintiles.csv", index=False)
    stats.to_csv(output/"parameter_distribution_and_hac_tests.csv", index=False)
    corr.to_csv(output/"parameter_spearman_tests.csv", index=False)
    weekdays.to_csv(output/"weekday_kruskal_tests.csv", index=False)
    plot_quantiles(q, cumulative, daily, output)
    plot_parameters(daily, stats, output)
    plot_state_quantiles(daily, hourly, output)
    plot_contributions(daily, hourly, output)
    plot_residuals(hourly, output)
    print_report(daily, hourly, stats, corr, weekdays, output)
    return daily, hourly, stats


def write_results(output, daily, hourly, states, skipped):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if daily: pd.DataFrame(daily).to_csv(output/SAVED["daily"], index=False)
    if hourly: pd.DataFrame(hourly).to_csv(output/SAVED["hourly"], index=False)
    if states: pd.DataFrame(states).to_csv(output/SAVED["states"], index=False)
    pd.DataFrame(skipped, columns=["day", "reason"]).to_csv(output/SAVED["skip"], index=False)


def load_from_sweep(folder, baseline, output, steps_per_hour):
    folder, output = Path(folder), Path(output)
    files = {"fit": "parameter_estimates.csv", "score": "oracle_horizon_scores.csv",
             "component": "forecast_components.csv", "states": "historical_initial_states.csv"}
    tables = {}
    provenance = folder/"configuration.json"
    if provenance.is_file():
        source_config = json.loads(provenance.read_text(encoding="utf-8"))
        if int(source_config.get("steps_per_hour", steps_per_hour)) != steps_per_hour:
            raise ValueError("Sweep numerical resolution differs. Pass --steps-per-hour "
                             "matching the saved sweep configuration.")
    else: source_config = {"warning": "Sweep configuration.json was unavailable."}
    for name, filename in files.items():
        path = folder/filename
        if not path.exists(): raise FileNotFoundError(f"Missing source sweep data: {path}")
        tables[name] = pd.read_csv(path, parse_dates=["day"])
        if "baseline" not in tables[name]: raise ValueError(f"Sweep {filename} lacks baseline method identifiers.")
        tables[name] = tables[name].loc[(tables[name]["baseline"] == baseline) &
                                         (tables[name]["training_weekdays"] == N_WEEKDAYS)].copy()
    if "oracle_R0" not in tables["score"] or "sensitivity_R0" not in tables["component"]:
        raise ValueError("Saved sweep uses an obsolete F0-only oracle. Rerun the weekday-horizon "
                         "sweep using the full-state [F0,R0] implementation.")
    scores = tables["score"].loc[tables["score"]["horizon_hours"] == HORIZON]
    fits = tables["fit"].drop_duplicates(["day"], keep="last").set_index("day")
    components = tables["component"].sort_values("hour").groupby("day")
    states = tables["states"].groupby("day")
    daily_rows, hourly_rows, history, skipped = [], [], [], []
    for day, row in scores.sort_values("day").drop_duplicates("day", keep="last").set_index("day").iterrows():
        try:
            if day not in fits.index or day not in components.groups or day not in states.groups:
                raise ValueError("Incomplete saved parameter/component/initial-state records.")
            fit = {**fits.loc[day].to_dict(), "day": day}
            score = {**row.to_dict(), "day": day}
            samples = components.get_group(day).to_dict("records")
            history_rows = states.get_group(day).to_dict("records")
            single, hour, state = consolidate(fit, score, samples, history_rows, steps_per_hour)
            daily_rows.append(single); hourly_rows.extend(hour); history.extend(state)
        except (RuntimeError, ValueError, KeyError) as exc:
            skipped.append({"day": day, "reason": str(exc)})
    if not daily_rows: raise RuntimeError("No complete eight-weekday/seven-hour configurations found in sweep.")
    write_results(output, daily_rows, hourly_rows, history, skipped)
    (output/"configuration.json").write_text(json.dumps({
        "mode": "from_sweep", "source": str(folder), "source_config": source_config,
        "baseline": baseline, "N": N_WEEKDAYS, "H": HORIZON,
        "steps_per_hour": steps_per_hour}, indent=2, default=str), encoding="utf-8")
    return len(daily_rows)


def fit_all(args):
    output = Path(args.output)
    config = {"baseline": args.baseline, "group": args.group, "N": N_WEEKDAYS,
              "H": HORIZON, "steps_per_hour": args.steps_per_hour,
              "start_day": args.start_day, "end_day": args.end_day}
    settings = output/"configuration.json"
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not settings.exists() or json.loads(settings.read_text(encoding="utf-8")) != config:
            raise ValueError("Cannot resume with a different or missing saved configuration.")
        previous = pd.read_csv(output/SAVED["daily"], parse_dates=["day"]) if (output/SAVED["daily"]).exists() else pd.DataFrame()
        done = set(previous["day"]) if not previous.empty else set()
        daily_rows = previous.to_dict("records")
        hourly_rows = (pd.read_csv(output/SAVED["hourly"]).to_dict("records")
                       if (output/SAVED["hourly"]).exists() else [])
        states = (pd.read_csv(output/SAVED["states"]).to_dict("records")
                  if (output/SAVED["states"]).exists() else [])
        skipped = (pd.read_csv(output/SAVED["skip"]).to_dict("records")
                   if (output/SAVED["skip"]).exists() else [])
    else:
        settings.write_text(json.dumps(config, indent=2), encoding="utf-8")
        daily_rows, hourly_rows, states, skipped, done = [], [], [], [], set()

    frame = group_frame(args.group)
    first = max(pd.Timestamp("2022-01-01")+pd.Timedelta(weeks=N_WEEKDAYS),
                frame.index.min().normalize()+pd.Timedelta(weeks=N_WEEKDAYS))
    last = min(pd.Timestamp("2022-10-31"), frame.index.max().normalize())
    if args.start_day: first = max(first, pd.Timestamp(args.start_day))
    if args.end_day: last = min(last, pd.Timestamp(args.end_day))
    days = pd.date_range(first, last, freq="D")
    if args.max_days: days = days[:args.max_days]
    if days.empty: raise ValueError("No dates remain in the requested period.")
    for count, day in enumerate(days, start=1):
        if day in done: continue
        try:
            fit, scores, components, train_states = evaluate_configuration(
                frame, day, N_WEEKDAYS, args.steps_per_hour, HORIZON, HORIZON, args.baseline)
            row, hourly, historical = consolidate(fit, scores[0], components, train_states,
                                                  args.steps_per_hour)
            daily_rows.append(row); hourly_rows.extend(hourly); states.extend(historical)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if args.fail_fast: raise
            skipped.append({"day": day, "reason": str(exc)})
        if count % args.progress_every == 0:
            write_results(output, daily_rows, hourly_rows, states, skipped)
            print(f"[WM4 N8/H7] {count}/{len(days)}: {len(daily_rows)} fitted, {len(skipped)} skipped", flush=True)
    write_results(output, daily_rows, hourly_rows, states, skipped)
    if not daily_rows: raise RuntimeError("No eligible forecasts. Check skipped_days.csv.")
    return len(daily_rows)


def main():
    parser = ArgumentParser(description="Full parameter tests and quantile plots for WM4: N=8 matched weekdays, H=7h oracle.")
    parser.add_argument("--group", choices=("MED", "ALL"), default="MED")
    parser.add_argument("--baseline", choices=BASELINE_NAMES, default="weekly_temperature")
    parser.add_argument("--steps-per-hour", type=int, default=4)
    parser.add_argument("--start-day", default=None)
    parser.add_argument("--end-day", default=None)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--from-sweep", default=None,
                        help="Analyze cached three-baseline sweep CSVs instead of refitting.")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate tests/quantiles/figures from this script's own saved CSVs.")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted fresh fit.")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.steps_per_hour < 2 or args.max_days < 0 or args.progress_every < 1:
        parser.error("steps-per-hour >= 2, max-days >= 0 and progress-every >= 1 are required.")
    if args.output is None: args.output = str(OUTPUT/args.group.lower()/args.baseline)
    if args.report_only:
        analyze(args.output); return
    if args.from_sweep:
        load_from_sweep(args.from_sweep, args.baseline, args.output, args.steps_per_hour)
    else: fit_all(args)
    analyze(args.output)


if __name__ == "__main__": main()
