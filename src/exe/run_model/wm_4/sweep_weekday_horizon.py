from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.moving_horizon_common import HOURS, group_frame, metrics, window_data
from src.exe.run_model.wm_4.predict_all_moving_horizon import historical_weekdays
from src.params.wm_4 import joint_qp_params
from src.model.wm_4 import wm_4
from src.tools.baseline import statistical_output, weekly_average, weekly_temperature, output_callable
from src.tools.impute_hourly import impute_hourly_days
from src.validation.exe.run_model.wm_4.sweep_weekday_horizon import sweep_weekday_horizon_validation as validation


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "weekday_horizon_sweep_expanded"
BASELINES = {"statistical_output": statistical_output, "weekly_average": weekly_average,
             "weekly_temperature": weekly_temperature}
BASELINE_NAMES = tuple(BASELINES)
PERCENTILES = (20, 40, 50, 60, 80)
RANKING_METRIC = "aggregate_rmse"  # Absolute pooled prediction error; never baseline-relative improvement.
EXPERIMENT_VERSION = "4_baseline_matched_training_per_horizon_joint_drift_oracle_v5"
COLUMNS = {
    "fit": "parameter_estimates.csv",
    "score": "oracle_horizon_scores.csv",
    "component": "forecast_components.csv",
    "state": "historical_initial_states.csv",
    "fail": "skipped_windows.csv",
}


class corresponding_weekday_baseline:
    """Piecewise-linear 24-hour demand profile: mean of N historical matching weekdays."""

    def __init__(self, profiles):
        values = np.asarray(profiles, dtype=float)
        if values.ndim != 2 or values.shape[1] != HOURS or not 2 <= len(values) <= 14:
            raise ValueError("Baseline requires 2-14 complete corresponding weekday profiles.")
        if not np.isfinite(values).all(): raise ValueError("Baseline training demand must be finite.")
        self.profile = np.mean(values, axis=0)

    def predict(self, index, temperature):
        index = pd.DatetimeIndex(index)
        if len(index) != len(np.asarray(temperature).reshape(-1)):
            raise ValueError("Baseline timestamps and temperatures must align.")
        h = (index.hour + index.minute/60.0 + index.second/3600.0 +
             index.microsecond/3.6e9).to_numpy(dtype=float)
        return np.interp(h, np.arange(HOURS+1), np.r_[self.profile, self.profile[0]])


def baseline_from_history(frame, history, name, horizon_hours=HOURS):
    """Fit the baseline ONLY to N corresponding historical intervals of k hours."""
    if name not in BASELINES: raise ValueError(f"Unknown baseline: {name}.")
    horizon_hours = int(horizon_hours)
    if not 1 <= horizon_hours <= HOURS: raise ValueError("horizon_hours must lie in [1,24].")
    samples = [frame.reindex(pd.date_range(past, periods=horizon_hours, freq="h"))[["I", "Temp"]]
               for past in history]
    train = pd.concat(samples).sort_index()
    if train[["I", "Temp"]].isna().any().any():
        raise ValueError("All selected historical baseline samples must be complete.")
    return BASELINES[name](min_samples=max(2, min(48, len(train)))).fit(
        train, max(history)+pd.Timedelta(days=1), exact_historical=True)


def interval_window(frame, start, baseline, steps_per_hour, horizon_hours):
    """Build only the requested k-hour window, not a full-day data window.

    The k hourly observations at hours 0..k-1 are scored. For k<24,
    the observation at hour k closes the k-hour historical fitting
    interval, providing the last actual rate transition. At k=24 the
    available 24 observations span 23 observed hourly transitions.
    """
    horizon_hours, steps_per_hour = int(horizon_hours), int(steps_per_hour)
    if not 1 <= horizon_hours <= HOURS: raise ValueError("horizon_hours must lie in [1,24].")
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be >= 2.")
    start = pd.Timestamp(start)
    points = min(horizon_hours+1, HOURS)
    index = pd.date_range(start, periods=points, freq="h")
    source = frame.reindex(index)[["I", "U", "Temp"]]
    if source.isna().any().any():
        raise ValueError(f"Missing consumption, price or temperature in historical/prediction interval {start} (k={horizon_hours}).")
    observed = source["I"].to_numpy(dtype=float)[:horizon_hours]
    elapsed = np.arange(horizon_hours*steps_per_hour+1, dtype=float)/steps_per_hour
    fine_index = start+pd.to_timedelta(elapsed, unit="h")
    I_hour = source["I"].to_numpy(dtype=float)
    T_hour = source["Temp"].to_numpy(dtype=float)
    U_hour = source["U"].to_numpy(dtype=float)
    if points == horizon_hours:
        I_hour, T_hour, U_hour = np.r_[I_hour, I_hour[-1]], np.r_[T_hour, T_hour[-1]], np.r_[U_hour, U_hour[-1]]
    I = np.interp(elapsed, np.arange(horizon_hours+1), I_hour)
    Temp = np.interp(elapsed, np.arange(horizon_hours+1), T_hour)
    U = U_hour[np.minimum(np.floor(elapsed+1e-10).astype(int), horizon_hours)]
    O = np.asarray(baseline.predict(fine_index, Temp), dtype=float).reshape(-1)
    count = len(elapsed)
    if O.size != count or not np.isfinite(O).all(): raise ValueError("Baseline must be finite throughout the requested horizon.")
    observation_mask = np.zeros(count, dtype=bool)
    observation_mask[np.arange(horizon_hours)*steps_per_hour] = True
    transition_mask = np.zeros(count-1, dtype=bool)
    transition_mask[:min(horizon_hours, HOURS-1)*steps_per_hour] = True
    data = {"I":I, "U":U, "PE":U.copy(), "Time":elapsed, "O":O,
            "observation_mask":observation_mask, "transition_mask":transition_mask}
    model = wm_4((0.0, horizon_hours/HOURS), count, output_callable(fine_index, O))
    return model, data, fine_index, observed


def evaluate_configuration(frame, day, n_weekdays, steps_per_hour, min_horizon, max_horizon,
                           baseline_name="weekly_average"):
    history = validation.history(day, historical_weekdays(day, n_weekdays))
    training, imputation = impute_hourly_days(frame, days=history,
                                               columns=("I", "U", "Temp"), require_days=True)
    fits, scores, components, states = [], [], [], []
    for horizon in range(min_horizon, max_horizon+1):
        baseline = baseline_from_history(training, history, baseline_name, horizon)
        datasets = [interval_window(training, past, baseline, steps_per_hour, horizon)[1] for past in history]
        model, forecast, _, observed = interval_window(frame, day, baseline, steps_per_hour, horizon)

        started = perf_counter()
        fit = joint_qp_params(model).fit_many(datasets)
        runtime = perf_counter()-started
        theta = np.asarray(fit["r"], dtype=float)   # [theta_F, theta_U, theta_R]
        sample_idx = np.arange(horizon, dtype=int)*steps_per_hour
        base_state = model._solve_deterministic([0.0, 0.0], forecast["U"], forecast["PE"],
                                                forecast["Time"], forecast["O"], theta)
        base = np.asarray(base_state["I"], dtype=float)[sample_idx]
        S = np.asarray(model.sensitivity({"r":theta})["input"], dtype=float)[sample_idx, :]
        if not np.isfinite(np.r_[observed, base, S.reshape(-1)]).all():
            raise ValueError("Oracle observations and fitted-state sensitivities must be finite.")
        baseline_hourly = np.asarray(forecast["O"], dtype=float)[sample_idx]
        if len(observed) != horizon: raise ValueError("Prediction observations and horizon do not agree.")

        origins = pd.DatetimeIndex(history)
        temps = training.reindex(origins)["Temp"].to_numpy(dtype=float)
        history_R0 = training.reindex(origins)["I"].to_numpy(dtype=float)-baseline.predict(origins, temps)
        R0 = float(np.median(history_R0))

        coeffs = {"theta_F":float(theta[0]), "theta_R":float(theta[2]), "theta_U":float(theta[1])}
        fit_row = {
            "day":day, "weekday":day.day_name(), "baseline":baseline_name,
            "training_weekdays":n_weekdays, "horizon_hours":horizon,
            "training_days":";".join(f"{d:%Y-%m-%d}" for d in history),
            "baseline_coefficients":json.dumps(np.asarray(baseline.coef, dtype=float).tolist()),
            "baseline_metadata":json.dumps(baseline.metadata, default=str),
            "baseline_feature_mean":json.dumps(None if baseline.feature_mean is None else baseline.feature_mean.tolist())
                if hasattr(baseline, "feature_mean") else "null",
            "baseline_feature_scale":json.dumps(None if baseline.feature_scale is None else baseline.feature_scale.tolist())
                if hasattr(baseline, "feature_scale") else "null",
            "baseline_beta_C":float(baseline.beta_C) if hasattr(baseline, "beta_C") else np.nan,
            "baseline_chill_center":float(baseline.chill_center) if hasattr(baseline, "chill_center") else np.nan,
            "baseline_samples":int(baseline.metadata["samples"]),
            "imputed_training_hours":int(imputation["imputed_hourly_slots"]),
            "imputed_training_I_hours":int(imputation["imputed_by_column"]["I"]),
            "imputed_training_U_hours":int(imputation["imputed_by_column"]["U"]),
            "imputed_training_Temp_hours":int(imputation["imputed_by_column"]["Temp"]),
            "hourly_imputation_method":imputation["method"], **coeffs,
            "historical_R0":R0, "fit_qp_loss":float(fit["qp_loss"]),
            "fit_rmse":float(np.sqrt(fit["deterministic_loss"])), "fit_seconds":runtime,
            "design_rank":int(fit["design_rank"]), "design_columns":len(fit["beta"]),
            "damping_ratio":float(-theta[2]/(2*np.sqrt(-theta[0]))),
            "training_F0_mean":float(np.mean(fit["init_cond"][:, 0])),
            "training_F0_std":float(np.std(fit["init_cond"][:, 0])),
        }
        # Preserve a fixed CSV schema without evaluating beyond the fitted horizon.
        for hour in range(HOURS):
            fit_row[f"baseline_hour_{hour:02d}"] = float(baseline_hourly[hour]) if hour < horizon else np.nan
        fits.append(fit_row)

        states.extend({
            "day":day, "baseline":baseline_name, "training_weekdays":n_weekdays,
            "horizon_hours":horizon, "historical_day":past,
            "historical_F0":float(state[0]), "historical_R0":float(state[1]),
            "baseline_hour_00":float(baseline.predict(pd.DatetimeIndex([past]),
                                      np.array([training.at[past, "Temp"]]))[0]),
        } for past, state in zip(history, fit["init_cond"]))
        components.extend({
            "day":day, "baseline":baseline_name, "training_weekdays":n_weekdays,
            "horizon_hours":horizon, "hour":h,
            "time":day+pd.Timedelta(hours=h), "observed":float(observed[h]),
            "baseline_value":float(baseline_hourly[h]), "price":float(forecast["U"][sample_idx[h]]),
            "prediction_at_F0_zero":float(base[h]),
            "sensitivity_F0":float(S[h, 0]),
            "prediction_at_initial_zero":float(base[h]), "sensitivity_R0":float(S[h, 1]),
        } for h in range(horizon))

        desired = observed-base
        initial = np.asarray(model._least_squares_qp(
            S, desired, lb=np.array([-1., -1.]), ub=np.array([1., 1.]),
            ridge=model._FIT_RIDGE), dtype=float)
        predicted = base+S@initial
        score, bscore = metrics(observed, predicted), metrics(observed, baseline_hourly)
        scores.append({
            "day":day, "baseline":baseline_name, "training_weekdays":n_weekdays,
            "horizon_hours":horizon, **coeffs,
            "oracle_F0":float(initial[0]), "oracle_R0":float(initial[1]),
            "historical_R0":R0, "oracle_sensitivity_norm":float(np.linalg.norm(S)),
            "oracle_state_design_rank":int(np.linalg.matrix_rank(S)),
            "oracle_state_observations":int(horizon),
            "rmse":score["rmse"], "mse":score["mse"], "mae":score["mae"], "bias":score["bias"],
            "baseline_rmse":bscore["rmse"], "baseline_mse":bscore["mse"],
            "rmse_gain":bscore["rmse"]-score["rmse"],
        })
    return fits, scores, components, states


def append_rows(path, records):
    if not records: return
    pd.DataFrame(records).to_csv(path, mode="a", index=False, header=not path.is_file())


def read_table(path, subset):
    if not path.is_file() or path.stat().st_size == 0: return pd.DataFrame()
    try: frame = pd.read_csv(path)
    except pd.errors.EmptyDataError: return pd.DataFrame()
    if frame.empty: return frame
    for col in ("day", "historical_day", "time"):
        if col in frame: frame[col] = pd.to_datetime(frame[col])
    if "baseline" not in frame: frame["baseline"] = "corresponding_weekday"  # Legacy CSV compatibility.
    if subset: frame = frame.drop_duplicates(subset=subset, keep="last")
    return frame


def load_tables(output):
    return {
        name: read_table(output/filename, {
            "fit": ["day", "baseline", "training_weekdays", "horizon_hours"],
            "score": ["day", "baseline", "training_weekdays", "horizon_hours"],
            "component": ["day", "baseline", "training_weekdays", "horizon_hours", "hour"],
            "state": ["day", "baseline", "training_weekdays", "horizon_hours", "historical_day"],
            "fail": ["day", "baseline", "training_weekdays", "error"],
        }[name])
        for name, filename in COLUMNS.items()
    }


def completed_pairs(tables, counts, horizons, baselines=BASELINE_NAMES):
    if any(tables[key].empty for key in ("fit", "score", "component", "state")): return set()
    key = ["day", "baseline", "training_weekdays"]
    fit_counts = tables["fit"].groupby(key)["horizon_hours"].nunique()
    scores = tables["score"].groupby(key)["horizon_hours"].nunique()
    components = tables["component"].groupby(key).size()
    states = tables["state"].groupby(key).size()
    required_components = sum(range(int(tables["score"]["horizon_hours"].min()),
                                    int(tables["score"]["horizon_hours"].max())+1))
    return {(day, name, int(n)) for (day, name, n), size in fit_counts.items()
            if size == horizons and scores.get((day, name, n), 0) == horizons
            and components.get((day, name, n), 0) == required_components
            and states.get((day, name, n), 0) == n*horizons and n in counts and name in baselines}


def summarise(scores):
    if scores.empty: return pd.DataFrame()
    rows = []
    if "baseline" not in scores: scores = scores.assign(baseline="corresponding_weekday")
    for (name, n, horizon), data in scores.groupby(["baseline", "training_weekdays", "horizon_hours"]):
        pooled = float(np.sqrt(np.mean(data["mse"])))
        base = float(np.sqrt(np.mean(data["baseline_mse"])))
        rmse = data["rmse"].to_numpy(dtype=float)
        row = {
            "baseline": name, "training_weekdays": int(n), "horizon_hours": int(horizon), "days": len(data),
            "aggregate_rmse": pooled, "baseline_rmse": base, "mean_daily_rmse": float(np.mean(rmse)),
            "median_daily_rmse": float(np.median(rmse)), "mean_bias": float(data["bias"].mean()),
            "baseline_gain": base-pooled, "baseline_improvement_pct": 100*(base-pooled)/base if base > 0 else np.nan,
            "wm4_better_pct": float(100*np.mean(data["rmse"] < data["baseline_rmse"])),
        }
        for percentile in PERCENTILES:
            row[f"rmse_q{percentile}"] = float(np.percentile(rmse, percentile))
        rows.append(row)
    return pd.DataFrame(rows).sort_values([RANKING_METRIC, "baseline", "training_weekdays", "horizon_hours"]).reset_index(drop=True)


def matched_cohort(tables, weekdays, horizons, baselines=None):
    fits, scores = tables["fit"], tables["score"]
    if fits.empty or scores.empty: return pd.DatetimeIndex([])
    if "baseline" not in fits: fits = fits.assign(baseline="corresponding_weekday")
    if "baseline" not in scores: scores = scores.assign(baseline="corresponding_weekday")
    if baselines is None: baselines = tuple(fits["baseline"].unique())
    counts = scores.groupby(["day", "baseline", "training_weekdays"])["horizon_hours"].nunique()
    complete = {(day, name, int(n)) for (day, name, n), size in counts.items() if size == horizons}
    eligible = fits.loc[fits.apply(lambda r: (r["day"], r["baseline"], int(r["training_weekdays"])) in complete
                             and r["baseline"] in baselines and int(r["training_weekdays"]) in weekdays, axis=1)]
    by_day = eligible.groupby("day").apply(lambda group:
        len(set(zip(group["baseline"], group["training_weekdays"]))))
    return pd.DatetimeIndex(by_day.index[by_day == len(weekdays)*len(baselines)]).sort_values()


def write_outputs(tables, output, weekdays, horizons, baselines=BASELINE_NAMES):
    # Deduplicate possibly partially written windows after an interrupted --resume.
    for name in ("fit", "score", "component", "state", "fail"):
        tables[name].to_csv(output/COLUMNS[name], index=False)
    summary_all = summarise(tables["score"])
    summary_all.to_csv(output/"ranking_all_available.csv", index=False)
    common = matched_cohort(tables, weekdays, horizons, baselines)
    matched = tables["score"].loc[tables["score"]["day"].isin(common)].copy()
    summary_matched = summarise(matched)
    summary_matched.to_csv(output/"ranking_matched.csv", index=False)
    ranking = summary_matched if not summary_matched.empty else summary_all
    basis = "matched" if not summary_matched.empty else "unmatched_fallback"
    # Rank ONLY by absolute pooled oracle-state prediction RMSE, not baseline skill.
    ranking = ranking.sort_values([RANKING_METRIC, "baseline", "training_weekdays", "horizon_hours"]).reset_index(drop=True)
    top = ranking.head(3).copy()
    top.insert(0, "rank", np.arange(1, len(top)+1))
    top.insert(1, "ranking_basis", basis)
    top.to_csv(output/"top_three.csv", index=False)
    # Keep the full requested 1-24h ranking unchanged, but also report
    # comparisons at H>=6 where a two-state oracle has more target
    # observations than optimized degrees of freedom.
    meaningful=ranking.loc[ranking["horizon_hours"]>=6].head(3).copy()
    meaningful.insert(0,"rank",np.arange(1,len(meaningful)+1))
    meaningful.to_csv(output/"top_three_h6plus.csv",index=False)
    best_by_horizon = ranking.sort_values([RANKING_METRIC, "baseline", "training_weekdays"]).drop_duplicates("horizon_hours").sort_values("horizon_hours")
    best_by_horizon.to_csv(output/"best_by_horizon.csv", index=False)
    parameters = tables["fit"].groupby(["baseline", "training_weekdays", "horizon_hours"])[
        ["theta_F", "theta_R", "theta_U", "fit_qp_loss", "fit_seconds"]
    ].agg(["mean", "std", "median", "min", "max"])
    parameters.columns = ["_".join(column) for column in parameters.columns]
    parameters.to_csv(output/"parameter_summary.csv")
    return ranking, top, common, basis


def save_heatmap(ranking, top, output, metric, filename, title, weekdays=None, horizons=None):
    if ranking.empty: return
    pivot = ranking.pivot(index="training_weekdays", columns="horizon_hours", values=metric)
    pivot = pivot.reindex(index=range(2, 15), columns=range(1, 25))
    fig, ax = plt.subplots(figsize=(16, 7))
    img = ax.imshow(np.ma.masked_invalid(pivot.to_numpy(dtype=float)), origin="lower", aspect="auto", interpolation="nearest")
    fig.colorbar(img, ax=ax, label=metric.replace("_", " "))
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Prediction horizon (hours)")
    ax.set_ylabel("Number of preceding corresponding weekdays")
    ax.set_title(title)
    for row in top.itertuples():
        if row.baseline not in set(ranking["baseline"]) or row.baseline != ranking["baseline"].iloc[0]: continue
        x = pivot.columns.get_loc(row.horizon_hours)
        y = pivot.index.get_loc(row.training_weekdays)
        ax.scatter(x, y, marker="*", s=240, edgecolor="black", linewidths=0.7, zorder=5)
        ax.annotate(f"#{row.rank}", (x, y), xytext=(7, 7), textcoords="offset points", fontsize=10, weight="bold")
    fig.tight_layout(); fig.savefig(output/filename, dpi=190, bbox_inches="tight"); plt.close(fig)


def save_horizon_curves(ranking, output):
    if ranking.empty: return
    baselines = list(ranking["baseline"].unique())
    fig, axes = plt.subplots(len(baselines), 1, figsize=(13, 5*len(baselines)), sharex=True, squeeze=False)
    for ax, name in zip(axes[:, 0], baselines):
        data = ranking.loc[ranking["baseline"] == name].sort_values("horizon_hours")
        for n, subset in data.groupby("training_weekdays"):
            ax.plot(subset["horizon_hours"], subset["aggregate_rmse"], label=f"{n} weekdays")
        ax.set_ylabel("Pooled full-state oracle RMSE"); ax.set_title(name)
        ax.grid(alpha=0.25); ax.legend(ncol=5, fontsize=8)
    axes[-1, 0].set_xlabel("Prediction horizon (hours)")
    fig.suptitle("ARM (Affine Revised Model) oracle prediction quality by historical weekdays and baseline")
    fig.tight_layout(); fig.savefig(output/"rmse_by_horizon.png", dpi=190, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, group in ranking.groupby("baseline"):
        best = group.sort_values(RANKING_METRIC).drop_duplicates("horizon_hours").sort_values("horizon_hours")
        ax.plot(best["horizon_hours"], best[RANKING_METRIC], label=f"{name} (optimal N per horizon)")
    ax.set_xlabel("Prediction horizon (hours)"); ax.set_ylabel("Lowest pooled RMSE over N")
    ax.set_title("Best corresponding-weekday count at every horizon, by baseline")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(output/"best_baseline_by_horizon.png", dpi=190, bbox_inches="tight"); plt.close(fig)


def save_top_three_plots(scores, top, common, output):
    if top.empty: return
    fig, ax = plt.subplots(figsize=(14, 6))
    samples = []
    labels = []
    for row in top.itertuples():
        subset = scores.loc[(scores["baseline"] == row.baseline) &
                            (scores["training_weekdays"] == row.training_weekdays) &
                            (scores["horizon_hours"] == row.horizon_hours) &
                            scores["day"].isin(common)].sort_values("day")
        if subset.empty: subset = scores.loc[(scores["baseline"] == row.baseline) &
                                              (scores["training_weekdays"] == row.training_weekdays) &
                                              (scores["horizon_hours"] == row.horizon_hours)].sort_values("day")
        label = f"#{row.rank}: {row.baseline}, N={row.training_weekdays}, H={row.horizon_hours}h"
        ax.plot(subset["day"], subset["rmse"], lw=0.8, alpha=0.75, label=label)
        samples.append(subset["rmse"].to_numpy(dtype=float)); labels.append(label)
    ax.set_title("Best three weekday/horizon configurations — daily oracle-state RMSE")
    ax.set_xlabel("Prediction date"); ax.set_ylabel("Daily RMSE")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(output/"top_three_daily_rmse.png", dpi=190, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(samples, showmeans=True)
    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels)
    ax.set_title("Best three configurations — across-day RMSE distributions")
    ax.set_ylabel("Per-day oracle-state RMSE"); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(output/"top_three_rmse_distribution.png", dpi=190, bbox_inches="tight"); plt.close(fig)


def print_report(top, ranking, common, tables, basis, output):
    fits, scores = tables["fit"], tables["score"]
    print("\nWM4 full-state [F0,R0] oracle historical-weekday / prediction-horizon sweep")
    print(f"completed parameter fits: {len(fits)} | horizon scores: {len(scores)} | failed/skipped: {len(tables['fail'])}")
    print(f"matched forecast days across ALL baseline methods and weekday counts: {len(common)}")
    print(f"evaluated baselines: {', '.join(sorted(fits['baseline'].unique()))}")
    if "imputed_training_hours" in fits:
        print(f"historical missing-hour imputations: {int(fits['imputed_training_hours'].sum())} "
              "(summed across overlapping training windows; whole missing days remain excluded)")
    if basis != "matched":
        print("WARNING: no fully matched cohort; ranking uses differing available dates per combination.")
    print("\nTHREE BEST COMBINATIONS (ranked by LOWEST ABSOLUTE aggregate full-state oracle RMSE):")
    print("Baseline improvement is reported separately and NEVER used to determine ranking.")
    fields = ["rank", "baseline", "training_weekdays", "horizon_hours", "days", "aggregate_rmse",
              "baseline_rmse", "rmse_q20", "median_daily_rmse", "rmse_q80"]
    print(top[fields].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    nontrivial=ranking.loc[ranking["horizon_hours"]>=6].head(3).copy()
    if len(nontrivial):
        nontrivial.insert(0,"rank",np.arange(1,len(nontrivial)+1))
        print("\nTOP THREE WITH H >= 6h (for comparison with the previous experiment):")
        print(nontrivial[fields].to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print("\nBest absolute RMSE within each baseline method:")
    for name, subset in ranking.groupby("baseline"):
        best = subset.sort_values(RANKING_METRIC).iloc[0]
        print(f"  {name:<20} N={int(best['training_weekdays']):2d}, "
              f"H={int(best['horizon_hours']):2d}h | pooled RMSE={best[RANKING_METRIC]:.8f} "
              f"| days={int(best['days'])}")
    print("\nTop configurations: historical coefficient distribution on the ranked cohort")
    for row in top.itertuples():
        subset = fits.loc[(fits["baseline"] == row.baseline) & (fits["training_weekdays"] == row.training_weekdays) & (fits["horizon_hours"] == row.horizon_hours) & fits["day"].isin(common)]
        if subset.empty: subset = fits.loc[(fits["baseline"] == row.baseline) & (fits["training_weekdays"] == row.training_weekdays) & (fits["horizon_hours"] == row.horizon_hours)]
        print(f"#{row.rank} B={row.baseline}, N={row.training_weekdays}, H={row.horizon_hours}h — theta means "
              f"[F={subset['theta_F'].mean():.6f}, R={subset['theta_R'].mean():.6f}, U={subset['theta_U'].mean():.6f}] "
              f"(std [{subset['theta_F'].std():.6f}, {subset['theta_R'].std():.6f}, {subset['theta_U'].std():.6f}])")
    print(f"\nSaved results and plots: {output}")
    print("WARNING: prediction [F0,R0] uses the target horizon's future observations: retrospective oracle, not live forecasting.")
    print("Ranking across different horizon lengths compares different forecasting tasks; best_by_horizon.csv compares N at fixed H.")
    print("CAUTION: H=1 uses only one target observation to fit TWO oracle initial states;")
    print("its initial R0 can generally reproduce that observation exactly, so H=1 RMSE")
    print("cannot measure model predictive skill. H=2 is also highly underdetermined/optimistic.")
    print("Assess parameter backlog N at a FIXED meaningful H, preferably on held-out observations.")


def report_only(output, weekdays, horizons, baselines=None):
    output = Path(output)
    tables = load_tables(output)
    if tables["score"].empty: raise RuntimeError("No saved oracle_horizon_scores.csv available.")
    if baselines is None: baselines = tuple(sorted(tables["score"]["baseline"].unique()))
    ranking, top, common, basis = write_outputs(tables, output, weekdays, horizons, baselines)
    if not ranking.empty:
        for name, group in ranking.groupby("baseline"):
            chosen = top.loc[top["baseline"] == name]
            suffix = "" if name == "corresponding_weekday" else f"_{name}"
            save_heatmap(group, chosen, output, "aggregate_rmse",
                         f"rmse_heatmap{suffix}.png", f"Pooled full-state oracle RMSE — {name}")
            save_heatmap(group, chosen, output, "baseline_improvement_pct",
                         f"baseline_skill_heatmap{suffix}.png", f"Baseline improvement (%) — {name}")
        save_horizon_curves(ranking, output)
        save_top_three_plots(tables["score"], top, common, output)
    print_report(top, ranking, common, tables, basis, output)
    return ranking, top


def run(args):
    validation.arguments(args)
    output = Path(args.output) if args.output else OUTPUT/args.group.lower()
    output.mkdir(parents=True, exist_ok=True)
    weekdays = tuple(range(args.min_weekdays, args.max_weekdays+1))
    baselines = tuple(args.baselines)
    horizons = args.max_horizon-args.min_horizon+1
    metadata = {
        "group": args.group, "min_weekdays": args.min_weekdays,
        "max_weekdays": args.max_weekdays, "min_horizon": args.min_horizon,
        "max_horizon": args.max_horizon, "steps_per_hour": args.steps_per_hour,
        "start_day": args.start_day, "end_day": args.end_day,
        "baselines": list(baselines), "version": EXPERIMENT_VERSION,
    }
    info = output/"configuration.json"
    if args.report_only:
        if not info.is_file() or json.loads(info.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Cannot report: saved configuration is absent, obsolete or differs "
                             "from the current full [F0,R0] oracle settings.")
        return report_only(output, weekdays, horizons, baselines)
    if args.resume:
        if not info.exists(): raise ValueError("Cannot resume: configuration.json is missing.")
        if json.loads(info.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Cannot resume: saved configuration differs from current arguments.")
    else:
        for file in list(COLUMNS.values()) + ["configuration.json"]:
            (output/file).unlink(missing_ok=True)
        info.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    frame = group_frame(args.group)
    earliest = max(frame.index.min().normalize(), pd.Timestamp("2022-01-01"))
    last = min(frame.index.max().normalize(), pd.Timestamp("2022-10-31"))
    # Use the SAME eligible calendar period for all N, allowing fair ranking.
    begin = earliest + pd.Timedelta(weeks=args.max_weekdays)
    if args.start_day: begin = max(begin, pd.Timestamp(args.start_day))
    if args.end_day: last = min(last, pd.Timestamp(args.end_day))
    if begin > last: raise ValueError("No prediction dates remain after matching-weekday warm-up.")
    days = pd.date_range(begin, last, freq="D")
    if args.max_days: days = days[:args.max_days]
    saved = load_tables(output) if args.resume else {name: pd.DataFrame() for name in COLUMNS}
    done = completed_pairs(saved, weekdays, horizons, baselines) if args.resume else set()
    successful, attempted = 0, 0
    total = len(days)*len(weekdays)*len(baselines)
    for day in days:
        for n in weekdays:
            for name in baselines:
                key = (day, name, n)
                if key in done: continue
                try:
                    fit, score, component, state = evaluate_configuration(
                        frame, day, n, args.steps_per_hour, args.min_horizon, args.max_horizon, name)
                    append_rows(output/COLUMNS["fit"], fit)
                    append_rows(output/COLUMNS["score"], score)
                    append_rows(output/COLUMNS["component"], component)
                    append_rows(output/COLUMNS["state"], state)
                    successful += 1
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                    if args.fail_fast: raise
                    append_rows(output/COLUMNS["fail"], [{"day": day, "baseline": name, "training_weekdays": n, "error": str(exc)}])
                    print(f"[skip] {day:%Y-%m-%d} B={name} N={n}: {exc}", flush=True)
                attempted += 1
                if attempted % args.progress_every == 0:
                    print(f"[WM4 sweep] computed {successful} new / {total} possible fits "
                          f"(resumed {len(done)}) | {day:%Y-%m-%d} B={name} N={n}", flush=True)
    return report_only(output, weekdays, horizons, baselines)


def main():
    parser = ArgumentParser(description="WM4 full-state oracle sweep over N=2..14 matching weekdays and H=1..24h.")
    parser.add_argument("--group", choices=("MED", "ALL"), default="MED")
    parser.add_argument("--baselines", nargs="+", choices=BASELINE_NAMES, default=list(BASELINE_NAMES),
                        help="Baseline methods; default all three: statistical_output weekly_average weekly_temperature.")
    parser.add_argument("--min-weekdays", type=int, default=2)
    parser.add_argument("--max-weekdays", type=int, default=14)
    parser.add_argument("--min-horizon", type=int, default=1)
    parser.add_argument("--max-horizon", type=int, default=24)
    parser.add_argument("--steps-per-hour", type=int, default=4)
    parser.add_argument("--start-day", default=None)
    parser.add_argument("--end-day", default=None)
    parser.add_argument("--max-days", type=int, default=0, help="Limit the calendar days evaluated (0=all).")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from saved CSVs and matched configuration.json.")
    parser.add_argument("--report-only", action="store_true", help="Rebuild rankings/plots from existing saved results.")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__": main()
