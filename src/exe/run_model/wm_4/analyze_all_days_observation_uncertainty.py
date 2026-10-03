from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_observation_uncertainty import DEFAULT_WEEKS, _stack_days, correlation_metrics, corresponding_days, uncertainty_frame
from src.exe.run_model.wm_4.fit_daily import aligned_wm4_data, day_start
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "uncertainty" / "observation_all_days"
VARIABLES = [
    "mean_I", "mean_U", "mean_abs_U", "std_U",
    "mean_abs_dI", "mean_abs_dU", "mean_Temp", "std_Temp",
    "time_sin", "time_cos",
]


def _window_correlations(hourly, prediction_day):
    result = correlation_metrics(hourly).copy()
    result.insert(0, "prediction_day", prediction_day)
    result.insert(1, "weekday", prediction_day.day_name())
    return result


def _demean_by_hour(frame, columns):
    result = frame.copy()
    for column in columns:
        result[f"{column}_hour_demeaned"] = result[column] - result.groupby("hour")[column].transform("mean")
    return result


def pooled_correlations(hourly):
    return correlation_metrics(hourly)


def hour_controlled_correlations(hourly):
    columns = ["s_I"] + [name for name in VARIABLES if not name.startswith("time_")]
    work = _demean_by_hour(hourly, columns)
    renamed = pd.DataFrame({"s_I": work["s_I_hour_demeaned"]})
    for variable in columns[1:]: renamed[variable] = work[f"{variable}_hour_demeaned"]
    return correlation_metrics(renamed, variables=columns[1:])


def hourly_summary(hourly):
    rows = []
    for hour, part in hourly.groupby("hour", sort=True):
        rows.append({
            "hour": int(hour), "windows": int(len(part)),
            "mean_s_I": float(part["s_I"].mean()), "median_s_I": float(part["s_I"].median()),
            "std_s_I": float(part["s_I"].std(ddof=1)), "min_s_I": float(part["s_I"].min()), "max_s_I": float(part["s_I"].max()),
            "mean_I": float(part["mean_I"].mean()), "mean_U": float(part["mean_U"].mean()),
            "mean_abs_dI": float(part["mean_abs_dI"].mean()), "mean_abs_dU": float(part["mean_abs_dU"].mean()),
            "mean_Temp": float(part["mean_Temp"].mean()),
        })
    return pd.DataFrame(rows)


def correlation_stability(window_correlations):
    rows = []
    for variable, part in window_correlations.groupby("variable", sort=False):
        pearson = part["pearson_r"].dropna()
        spearman = part["spearman_rho"].dropna()
        rows.append({
            "variable": variable, "windows": int(len(part)),
            "mean_pearson_r": float(pearson.mean()) if len(pearson) else np.nan,
            "median_pearson_r": float(pearson.median()) if len(pearson) else np.nan,
            "pearson_positive_pct": 100.0 * float((pearson > 0.0).mean()) if len(pearson) else np.nan,
            "mean_spearman_rho": float(spearman.mean()) if len(spearman) else np.nan,
            "median_spearman_rho": float(spearman.median()) if len(spearman) else np.nan,
            "spearman_positive_pct": 100.0 * float((spearman > 0.0).mean()) if len(spearman) else np.nan,
        })
    result = pd.DataFrame(rows)
    result["max_abs_median_correlation"] = np.nanmax(
        np.abs(result[["median_pearson_r", "median_spearman_rho"]].to_numpy(dtype=float)), axis=1,
    )
    return result.sort_values("max_abs_median_correlation", ascending=False).reset_index(drop=True)


def run_all_days(frame, start=START_DAY, end=END_DAY, weeks=DEFAULT_WEEKS, fail_fast=False):
    start, end, weeks = day_start(start), day_start(end), int(weeks)
    if end < start: raise ValueError("end must be on or after start.")
    if weeks < 2: raise ValueError("weeks must be at least 2.")

    hourly_frames, correlation_frames, skipped = [], [], []
    for prediction_day in pd.date_range(start, end, freq="D"):
        try:
            days = corresponding_days(prediction_day, weeks)
            hourly = uncertainty_frame(_stack_days(frame, days))
            hourly.insert(0, "prediction_day", prediction_day)
            hourly.insert(1, "weekday", prediction_day.day_name())
            hourly_frames.append(hourly)
            correlation_frames.append(_window_correlations(hourly, prediction_day))
            print(
                f"{prediction_day:%Y-%m-%d} | mean_s_I={hourly['s_I'].mean():.6f} "
                f"| max_s_I={hourly['s_I'].max():.6f} @h{int(hourly.loc[hourly['s_I'].idxmax(), 'hour']):02d}"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((prediction_day, str(exc))); print(f"{prediction_day:%Y-%m-%d} | skipped | {exc}")

    if not hourly_frames: raise RuntimeError("No rolling five-week uncertainty windows could be analyzed.")
    hourly = pd.concat(hourly_frames, ignore_index=True).sort_values(["prediction_day", "hour"]).reset_index(drop=True)
    window_correlations = pd.concat(correlation_frames, ignore_index=True)
    return hourly, window_correlations, skipped


def print_summary(hourly, pooled, controlled, stability, by_hour, skipped, weeks):
    windows = int(hourly["prediction_day"].nunique())
    mean_by_hour = by_hour.loc[by_hour["mean_s_I"].idxmax()]
    min_by_hour = by_hour.loc[by_hour["mean_s_I"].idxmin()]
    print()
    print("all-days observed uncertainty summary")
    print(f"rolling window={weeks} corresponding weekdays")
    print(f"prediction windows analyzed={windows}, skipped={len(skipped)}, pooled hour/window observations={len(hourly)}")
    print(f"overall mean s_I={hourly['s_I'].mean():.8f}, median s_I={hourly['s_I'].median():.8f}")
    print(f"highest mean hourly uncertainty: hour={int(mean_by_hour['hour'])}, mean_s_I={mean_by_hour['mean_s_I']:.8f}")
    print(f"lowest mean hourly uncertainty: hour={int(min_by_hour['hour'])}, mean_s_I={min_by_hour['mean_s_I']:.8f}")

    columns = ["variable", "observations", "pearson_r", "pearson_p", "spearman_rho", "spearman_p"]
    print()
    print("pooled correlation: s_I vs variables across all rolling windows and hours")
    print(pooled[columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print()
    print("hour-controlled correlation: within-hour deviations across rolling windows")
    print(controlled[columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print()
    print("rolling-window correlation stability")
    stable_columns = [
        "variable", "windows", "median_pearson_r", "pearson_positive_pct",
        "median_spearman_rho", "spearman_positive_pct",
    ]
    print(stability[stable_columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print()
    print("mean s_I by hour")
    print(by_hour[["hour", "windows", "mean_s_I", "median_s_I", "std_s_I"]].to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    print()
    print("s_I is always estimated directly from observed I across the corresponding weekdays; O is not used.")
    print("Pooled p-values are descriptive because rolling windows overlap and observations are temporally dependent.")
    if skipped:
        print(); print("skipped prediction dates:")
        for day, reason in skipped: print(f"{day:%Y-%m-%d}: {reason}")


def main():
    parser = ArgumentParser(description="Analyze observed WM4 demand uncertainty and correlations over all prediction dates using rolling corresponding-weekday windows.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--weeks", type=int, default=DEFAULT_WEEKS, help="Rolling number of previous corresponding weekdays; default 5.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    hourly, window_correlations, skipped = run_all_days(
        frame, start=args.start_day, end=args.end_day, weeks=args.weeks, fail_fast=args.fail_fast,
    )
    pooled = pooled_correlations(hourly)
    controlled = hour_controlled_correlations(hourly)
    stability = correlation_stability(window_correlations)
    by_hour = hourly_summary(hourly)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output / "rolling_hourly_uncertainty.csv", index=False)
    window_correlations.to_csv(output / "rolling_window_correlations.csv", index=False)
    pooled.to_csv(output / "pooled_correlations.csv", index=False)
    controlled.to_csv(output / "hour_controlled_correlations.csv", index=False)
    stability.to_csv(output / "correlation_stability.csv", index=False)
    by_hour.to_csv(output / "hourly_uncertainty_summary.csv", index=False)
    if skipped: pd.DataFrame([{"prediction_day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)

    print_summary(hourly, pooled, controlled, stability, by_hour, skipped, args.weeks)
    print(); print(f"output={output}")


if __name__ == "__main__": main()
