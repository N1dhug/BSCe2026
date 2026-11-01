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
from src.tools.baseline import statistical_output, weekly_average, weekly_temperature
from src.validation.exe.run_model.wm_4.sweep_weekday_horizon import sweep_weekday_horizon_validation as validation


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "weekday_horizon_sweep_three_baselines"
BASELINES = {"statistical_output": statistical_output, "weekly_average": weekly_average,
             "weekly_temperature": weekly_temperature}
BASELINE_NAMES = tuple(BASELINES)
PERCENTILES = (20, 40, 50, 60, 80)
RANKING_METRIC = "aggregate_rmse"  # Absolute pooled prediction error; never baseline-relative improvement.
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
        if values.ndim != 2 or values.shape[1] != HOURS or not 2 <= len(values) <= 10:
            raise ValueError("Baseline requires 2-10 complete corresponding weekday profiles.")
        if not np.isfinite(values).all(): raise ValueError("Baseline training demand must be finite.")
        self.profile = np.mean(values, axis=0)

    def predict(self, index, temperature):
        index = pd.DatetimeIndex(index)
        if len(index) != len(np.asarray(temperature).reshape(-1)):
            raise ValueError("Baseline timestamps and temperatures must align.")
        h = (index.hour + index.minute/60.0 + index.second/3600.0 +
             index.microsecond/3.6e9).to_numpy(dtype=float)
        return np.interp(h, np.arange(HOURS+1), np.r_[self.profile, self.profile[0]])


def baseline_from_history(frame, history, name):
    if name not in BASELINES: raise ValueError(f"Unknown baseline: {name}.")
    samples = [frame.reindex(pd.date_range(past, periods=HOURS, freq="h"))[["I", "Temp"]]
               for past in history]
    train = pd.concat(samples).sort_index()
    if train[["I", "Temp"]].isna().any().any():
        raise ValueError("All matched-weekday baseline training samples must be complete.")
    # The standard baseline estimators require seven historical days by default.
    # This comparative study deliberately fits each one to EXACTLY N same weekdays.
    # Explicit min_samples=48 enables N=2 without leaking other calendar days.
    baseline = BASELINES[name](min_samples=48).fit(train, max(history)+pd.Timedelta(days=1))
    return baseline


def evaluate_configuration(frame, day, n_weekdays, steps_per_hour, min_horizon, max_horizon,
                           baseline_name="weekly_average"):
    history = validation.history(day, historical_weekdays(day, n_weekdays))
    profiles = validation.training_data(frame, history)
    baseline = baseline_from_history(frame, history, baseline_name)
    datasets = [window_data(frame, earlier, baseline, steps_per_hour)[1] for earlier in history]
    model, forecast, _, observed = window_data(frame, day, baseline, steps_per_hour)

    started = perf_counter()
    fit = joint_qp_params(model).fit_many(datasets)
    runtime = perf_counter()-started
    theta = np.asarray(fit["r"], dtype=float)       # [theta_F, theta_U, theta_R]
    historical_origins = pd.DatetimeIndex(history)
    historical_temps = frame.reindex(historical_origins)["Temp"].to_numpy(dtype=float)
    history_R0 = profiles[:, 0]-baseline.predict(historical_origins, historical_temps)
    R0 = float(np.median(history_R0))              # strictly historical, never optimized

    state0 = model._solve_deterministic([0.0, R0], forecast["U"], forecast["PE"],
                                        forecast["Time"], forecast["O"], theta)
    sample_idx = np.arange(HOURS, dtype=int)*steps_per_hour
    base = np.asarray(state0["I"], dtype=float)[sample_idx]
    sensitivity = np.asarray(model.sensitivity({"r": theta})["input"], dtype=float)[sample_idx, 0]
    observed, base, sensitivity = validation.oracle(observed, base, sensitivity, min_horizon, max_horizon)
    baseline_hourly = np.asarray(forecast["O"], dtype=float)[sample_idx]

    coeffs = {"theta_F": float(theta[0]), "theta_R": float(theta[2]), "theta_U": float(theta[1])}
    fit_row = {
        "day": day, "weekday": day.day_name(), "baseline": baseline_name, "training_weekdays": n_weekdays,
        "training_days": ";".join(f"{d:%Y-%m-%d}" for d in history),
        "baseline_coefficients": json.dumps(np.asarray(baseline.coef, dtype=float).tolist()),
        "baseline_metadata": json.dumps(baseline.metadata, default=str),
        "baseline_feature_mean": json.dumps(None if baseline.feature_mean is None else baseline.feature_mean.tolist())
            if hasattr(baseline, "feature_mean") else "null",
        "baseline_feature_scale": json.dumps(None if baseline.feature_scale is None else baseline.feature_scale.tolist())
            if hasattr(baseline, "feature_scale") else "null",
        "baseline_beta_C": float(baseline.beta_C) if hasattr(baseline, "beta_C") else np.nan,
        "baseline_chill_center": float(baseline.chill_center) if hasattr(baseline, "chill_center") else np.nan,
        "baseline_samples": int(baseline.metadata["samples"]),
        **coeffs, "historical_R0": R0, "fit_qp_loss": float(fit["qp_loss"]),
        "fit_rmse": float(np.sqrt(fit["deterministic_loss"])), "fit_seconds": runtime,
        "design_rank": int(fit["design_rank"]), "design_columns": len(fit["beta"]),
        "damping_ratio": float(-theta[2]/(2*np.sqrt(-theta[0]))),
        "training_F0_mean": float(np.mean(fit["init_cond"][:, 0])),
        "training_F0_std": float(np.std(fit["init_cond"][:, 0])),
    }
    for hour, value in enumerate(baseline_hourly):
        fit_row[f"baseline_hour_{hour:02d}"] = float(value)

    states = [{"day": day, "baseline": baseline_name, "training_weekdays": n_weekdays, "historical_day": past,
               "historical_F0": float(state[0]), "historical_R0": float(state[1]),
               "baseline_hour_00": float(baseline.predict(pd.DatetimeIndex([past]),
                                              np.array([frame.at[past, "Temp"]]))[0])}
              for past, state in zip(history, fit["init_cond"])]
    components = [{"day": day, "baseline": baseline_name, "training_weekdays": n_weekdays, "hour": h,
                   "time": day+pd.Timedelta(hours=h), "observed": float(observed[h]),
                   "baseline": float(baseline_hourly[h]), "price": float(forecast["U"][sample_idx[h]]),
                   "prediction_at_F0_zero": float(base[h]), "sensitivity_F0": float(sensitivity[h])}
                  for h in range(HOURS)]

    # The model is affine in F0 for fixed R0/theta; solve each truncated oracle
    # exactly via prefix sums, rather than refitting theta for each horizon.
    numerator = np.cumsum(sensitivity*(observed-base))
    denominator = np.cumsum(sensitivity**2)
    scores = []
    for horizon in range(min_horizon, max_horizon+1):
        den = float(denominator[horizon-1])
        oracle = float(np.clip(numerator[horizon-1]/den, -1.0, 1.0)) if den > 1e-28 else 0.0
        predicted = base[:horizon]+oracle*sensitivity[:horizon]
        score = metrics(observed[:horizon], predicted)
        bscore = metrics(observed[:horizon], baseline_hourly[:horizon])
        scores.append({
            "day": day, "baseline": baseline_name, "training_weekdays": n_weekdays, "horizon_hours": horizon,
            **coeffs, "oracle_F0": oracle, "historical_R0": R0,
            "oracle_sensitivity_norm": float(np.sqrt(den)),
            "rmse": score["rmse"], "mse": score["mse"], "mae": score["mae"], "bias": score["bias"],
            "baseline_rmse": bscore["rmse"], "baseline_mse": bscore["mse"],
            "rmse_gain": bscore["rmse"]-score["rmse"],
        })
    return fit_row, scores, components, states


def append_rows(path, records):
    if not records: return
    pd.DataFrame(records).to_csv(path, mode="a", index=False, header=not path.is_file())


def read_table(path, subset):
    if not path.is_file() or path.stat().st_size == 0: return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty: return frame
    for col in ("day", "historical_day", "time"):
        if col in frame: frame[col] = pd.to_datetime(frame[col])
    if subset: frame = frame.drop_duplicates(subset=subset, keep="last")
    return frame


def load_tables(output):
    return {
        name: read_table(output/filename, {
            "fit": ["day", "training_weekdays"],
            "score": ["day", "training_weekdays", "horizon_hours"],
            "component": ["day", "training_weekdays", "hour"],
            "state": ["day", "training_weekdays", "historical_day"],
            "fail": ["day", "training_weekdays", "error"],
        }[name])
        for name, filename in COLUMNS.items()
    }


def completed_pairs(tables, counts, horizons):
    if any(tables[key].empty for key in ("fit", "score", "component", "state")): return set()
    key = ["day", "training_weekdays"]
    fit_counts = tables["fit"].groupby(key).size()
    scores = tables["score"].groupby(key)["horizon_hours"].nunique()
    components = tables["component"].groupby(key)["hour"].nunique()
    states = tables["state"].groupby(key)["historical_day"].nunique()
    return {(day, int(n)) for (day, n), size in fit_counts.items()
            if size >= 1 and scores.get((day, n), 0) == horizons
            and components.get((day, n), 0) == HOURS
            and states.get((day, n), 0) == n and n in counts}


def summarise(scores):
    if scores.empty: return pd.DataFrame()
    rows = []
    for (n, horizon), data in scores.groupby(["training_weekdays", "horizon_hours"]):
        pooled = float(np.sqrt(np.mean(data["mse"])))
        base = float(np.sqrt(np.mean(data["baseline_mse"])))
        rmse = data["rmse"].to_numpy(dtype=float)
        row = {
            "training_weekdays": int(n), "horizon_hours": int(horizon), "days": len(data),
            "aggregate_rmse": pooled, "baseline_rmse": base, "mean_daily_rmse": float(np.mean(rmse)),
            "median_daily_rmse": float(np.median(rmse)), "mean_bias": float(data["bias"].mean()),
            "baseline_gain": base-pooled, "baseline_improvement_pct": 100*(base-pooled)/base if base > 0 else np.nan,
            "wm4_better_pct": float(100*np.mean(data["rmse"] < data["baseline_rmse"])),
        }
        for percentile in PERCENTILES:
            row[f"rmse_q{percentile}"] = float(np.percentile(rmse, percentile))
        rows.append(row)
    return pd.DataFrame(rows).sort_values([RANKING_METRIC, "training_weekdays", "horizon_hours"]).reset_index(drop=True)


def matched_cohort(tables, weekdays, horizons):
    fits, scores = tables["fit"], tables["score"]
    if fits.empty or scores.empty: return pd.DatetimeIndex([])
    counts = scores.groupby(["day", "training_weekdays"])["horizon_hours"].nunique()
    complete = {(day, int(n)) for (day, n), size in counts.items() if size == horizons}
    eligible = fits.loc[fits.apply(lambda row: (row["day"], int(row["training_weekdays"])) in complete, axis=1)]
    by_day = eligible.groupby("day")["training_weekdays"].nunique()
    return pd.DatetimeIndex(by_day.index[by_day == len(weekdays)]).sort_values()


def write_outputs(tables, output, weekdays, horizons):
    # Deduplicate possibly partially written windows after an interrupted --resume.
    for name in ("fit", "score", "component", "state", "fail"):
        tables[name].to_csv(output/COLUMNS[name], index=False)
    summary_all = summarise(tables["score"])
    summary_all.to_csv(output/"ranking_all_available.csv", index=False)
    common = matched_cohort(tables, weekdays, horizons)
    matched = tables["score"].loc[tables["score"]["day"].isin(common)].copy()
    summary_matched = summarise(matched)
    summary_matched.to_csv(output/"ranking_matched.csv", index=False)
    ranking = summary_matched if not summary_matched.empty else summary_all
    basis = "matched" if not summary_matched.empty else "unmatched_fallback"
    # Rank ONLY by absolute pooled oracle-state prediction RMSE, not baseline skill.
    ranking = ranking.sort_values([RANKING_METRIC, "training_weekdays", "horizon_hours"]).reset_index(drop=True)
    top = ranking.head(3).copy()
    top.insert(0, "rank", np.arange(1, len(top)+1))
    top.insert(1, "ranking_basis", basis)
    top.to_csv(output/"top_three.csv", index=False)
    best_by_horizon = ranking.sort_values([RANKING_METRIC, "training_weekdays"]).drop_duplicates("horizon_hours").sort_values("horizon_hours")
    best_by_horizon.to_csv(output/"best_by_horizon.csv", index=False)
    parameters = tables["fit"].groupby("training_weekdays")[
        ["theta_F", "theta_R", "theta_U", "fit_qp_loss", "fit_seconds"]
    ].agg(["mean", "std", "median", "min", "max"])
    parameters.columns = ["_".join(column) for column in parameters.columns]
    parameters.to_csv(output/"parameter_summary.csv")
    return ranking, top, common, basis


def save_heatmap(ranking, top, output, metric, filename, title):
    if ranking.empty: return
    pivot = ranking.pivot(index="training_weekdays", columns="horizon_hours", values=metric)
    fig, ax = plt.subplots(figsize=(16, 7))
    img = ax.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", interpolation="nearest")
    fig.colorbar(img, ax=ax, label=metric.replace("_", " "))
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Prediction horizon (hours)")
    ax.set_ylabel("Number of preceding corresponding weekdays")
    ax.set_title(title)
    for row in top.itertuples():
        x = pivot.columns.get_loc(row.horizon_hours)
        y = pivot.index.get_loc(row.training_weekdays)
        ax.scatter(x, y, marker="*", s=240, edgecolor="black", linewidths=0.7, zorder=5)
        ax.annotate(f"#{row.rank}", (x, y), xytext=(7, 7), textcoords="offset points", fontsize=10, weight="bold")
    fig.tight_layout(); fig.savefig(output/filename, dpi=190, bbox_inches="tight"); plt.close(fig)


def save_horizon_curves(ranking, output):
    if ranking.empty: return
    fig, ax = plt.subplots(figsize=(12, 7))
    for n, subset in ranking.groupby("training_weekdays"):
        ax.plot(subset["horizon_hours"], subset["aggregate_rmse"], label=f"{n} weekdays")
    ax.set_xlabel("Prediction horizon (hours)"); ax.set_ylabel("Pooled oracle-F0 RMSE")
    ax.set_title("WM4 oracle-state prediction quality: historical weekdays vs horizon")
    ax.grid(alpha=0.25); ax.legend(ncol=3)
    fig.tight_layout(); fig.savefig(output/"rmse_by_horizon.png", dpi=190, bbox_inches="tight"); plt.close(fig)


def save_top_three_plots(scores, top, common, output):
    if top.empty: return
    fig, ax = plt.subplots(figsize=(14, 6))
    samples = []
    labels = []
    for row in top.itertuples():
        subset = scores.loc[(scores["training_weekdays"] == row.training_weekdays) &
                            (scores["horizon_hours"] == row.horizon_hours) &
                            scores["day"].isin(common)].sort_values("day")
        if subset.empty: subset = scores.loc[(scores["training_weekdays"] == row.training_weekdays) &
                                              (scores["horizon_hours"] == row.horizon_hours)].sort_values("day")
        label = f"#{row.rank}: N={row.training_weekdays}, H={row.horizon_hours}h"
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
    print("\nWM4 oracle-F0 historical-weekday / prediction-horizon sweep")
    print(f"completed parameter fits: {len(fits)} | horizon scores: {len(scores)} | failed/skipped: {len(tables['fail'])}")
    print(f"matched forecast days across ALL weekday counts: {len(common)}")
    if basis != "matched":
        print("WARNING: no fully matched cohort; ranking uses differing available dates per combination.")
    print("\nTHREE BEST COMBINATIONS (ranked by LOWEST ABSOLUTE aggregate oracle-F0 RMSE):")
    print("Baseline improvement is reported separately and NEVER used to determine ranking.")
    fields = ["rank", "training_weekdays", "horizon_hours", "days", "aggregate_rmse",
              "baseline_rmse", "rmse_q20", "median_daily_rmse", "rmse_q80"]
    print(top[fields].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTop configurations: historical coefficient distribution on the ranked cohort")
    for row in top.itertuples():
        subset = fits.loc[(fits["training_weekdays"] == row.training_weekdays) & fits["day"].isin(common)]
        if subset.empty: subset = fits.loc[fits["training_weekdays"] == row.training_weekdays]
        print(f"#{row.rank} N={row.training_weekdays}, H={row.horizon_hours}h — theta means "
              f"[F={subset['theta_F'].mean():.6f}, R={subset['theta_R'].mean():.6f}, U={subset['theta_U'].mean():.6f}] "
              f"(std [{subset['theta_F'].std():.6f}, {subset['theta_R'].std():.6f}, {subset['theta_U'].std():.6f}])")
    print(f"\nSaved results and plots: {output}")
    print("WARNING: prediction F0 uses the target horizon's future observations: retrospective oracle, not live forecasting.")
    print("Ranking across different horizon lengths compares different forecasting tasks; best_by_horizon.csv compares N at fixed H.")


def report_only(output, weekdays, horizons):
    output = Path(output)
    tables = load_tables(output)
    if tables["score"].empty: raise RuntimeError("No saved oracle_horizon_scores.csv available.")
    ranking, top, common, basis = write_outputs(tables, output, weekdays, horizons)
    if not ranking.empty:
        save_heatmap(ranking, top, output, "aggregate_rmse", "rmse_heatmap.png", "Pooled oracle-F0 RMSE")
        save_heatmap(ranking, top, output, "baseline_improvement_pct", "baseline_skill_heatmap.png", "RMSE improvement over matching-weekday baseline (%)")
        save_horizon_curves(ranking, output)
        save_top_three_plots(tables["score"], top, common, output)
    print_report(top, ranking, common, tables, basis, output)
    return ranking, top


def run(args):
    validation.arguments(args)
    output = Path(args.output) if args.output else OUTPUT/args.group.lower()
    output.mkdir(parents=True, exist_ok=True)
    weekdays = tuple(range(args.min_weekdays, args.max_weekdays+1))
    horizons = args.max_horizon-args.min_horizon+1
    metadata = {
        "group": args.group, "min_weekdays": args.min_weekdays,
        "max_weekdays": args.max_weekdays, "min_horizon": args.min_horizon,
        "max_horizon": args.max_horizon, "steps_per_hour": args.steps_per_hour,
        "start_day": args.start_day, "end_day": args.end_day,
    }
    info = output/"configuration.json"
    if args.resume:
        if not info.exists(): raise ValueError("Cannot resume: configuration.json is missing.")
        if json.loads(info.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Cannot resume: saved configuration differs from current arguments.")
    else:
        for file in list(COLUMNS.values()) + ["configuration.json"]:
            (output/file).unlink(missing_ok=True)
        info.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.report_only: return report_only(output, weekdays, horizons)

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
    done = completed_pairs(saved, weekdays, horizons) if args.resume else set()
    successful, attempted = 0, 0
    total = len(days)*len(weekdays)
    for day in days:
        for n in weekdays:
            key = (day, n)
            if key in done: continue
            try:
                fit, score, component, state = evaluate_configuration(
                    frame, day, n, args.steps_per_hour, args.min_horizon, args.max_horizon)
                append_rows(output/COLUMNS["fit"], [fit])
                append_rows(output/COLUMNS["score"], score)
                append_rows(output/COLUMNS["component"], component)
                append_rows(output/COLUMNS["state"], state)
                successful += 1
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                if args.fail_fast: raise
                append_rows(output/COLUMNS["fail"], [{"day": day, "training_weekdays": n, "error": str(exc)}])
                print(f"[skip] {day:%Y-%m-%d} N={n}: {exc}", flush=True)
            attempted += 1
            if attempted % args.progress_every == 0:
                print(f"[WM4 sweep] computed {successful} new / {total} possible fits "
                      f"(resumed {len(done)}) | {day:%Y-%m-%d} N={n}", flush=True)
    return report_only(output, weekdays, horizons)


def main():
    parser = ArgumentParser(description="WM4 oracle F0 sweep over N=2..10 matching weekdays and H=6..24h.")
    parser.add_argument("--group", choices=("MED", "ALL"), default="MED")
    parser.add_argument("--min-weekdays", type=int, default=2)
    parser.add_argument("--max-weekdays", type=int, default=10)
    parser.add_argument("--min-horizon", type=int, default=6)
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
    if args.report_only:
        validation.arguments(args)
        output = Path(args.output) if args.output else OUTPUT/args.group.lower()
        return report_only(output, tuple(range(args.min_weekdays, args.max_weekdays+1)),
                           args.max_horizon-args.min_horizon+1)
    return run(args)


if __name__ == "__main__": main()
