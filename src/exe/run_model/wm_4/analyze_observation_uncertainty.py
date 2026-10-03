from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start, selected_day


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "uncertainty" / "observation_five_weekdays"
DEFAULT_WEEKS = 5


def corresponding_days(date, weeks=DEFAULT_WEEKS):
    start, weeks = day_start(date), int(weeks)
    if weeks < 2: raise ValueError("weeks must be at least 2 to estimate observation uncertainty.")
    return [start - pd.Timedelta(weeks=k) for k in range(1, weeks + 1)]


def _stack_days(frame, days):
    selected = []
    for day in days:
        _, values = selected_day(frame, day)
        selected.append(values[["I", "U", "Temp"]].copy())
    return {
        "I": np.vstack([day["I"].to_numpy(dtype=float) for day in selected]),
        "U": np.vstack([day["U"].to_numpy(dtype=float) for day in selected]),
        "Temp": np.vstack([day["Temp"].to_numpy(dtype=float) for day in selected]),
    }


def _within_day_delta(values):
    values = np.asarray(values, dtype=float)
    delta = np.full(values.shape, np.nan, dtype=float)
    delta[:, 1:] = np.diff(values, axis=1)
    return delta


def uncertainty_frame(stacked):
    I = np.asarray(stacked["I"], dtype=float)
    U = np.asarray(stacked["U"], dtype=float)
    Temp = np.asarray(stacked["Temp"], dtype=float)
    if I.ndim != 2 or I.shape[1] != DAY_HOURS: raise ValueError("Expected one 24-hour row per corresponding weekday.")
    if U.shape != I.shape or Temp.shape != I.shape: raise ValueError("I, U and Temp weekday matrices must have identical shape.")

    dI, dU = _within_day_delta(I), _within_day_delta(U)
    hours = np.arange(DAY_HOURS, dtype=float)
    result = pd.DataFrame({
        "hour": hours.astype(int),
        "s_I": np.std(I, axis=0, ddof=1),
        "var_I": np.var(I, axis=0, ddof=1),
        "mean_I": np.mean(I, axis=0),
        "mean_U": np.mean(U, axis=0),
        "mean_abs_U": np.mean(np.abs(U), axis=0),
        "std_U": np.std(U, axis=0, ddof=1),
        "mean_abs_dI": np.nanmean(np.abs(dI), axis=0),
        "mean_abs_dU": np.nanmean(np.abs(dU), axis=0),
        "mean_Temp": np.mean(Temp, axis=0),
        "std_Temp": np.std(Temp, axis=0, ddof=1),
        "time_sin": np.sin(2.0 * np.pi * hours / DAY_HOURS),
        "time_cos": np.cos(2.0 * np.pi * hours / DAY_HOURS),
    })
    result.loc[0, ["mean_abs_dI", "mean_abs_dU"]] = np.nan
    return result


def _correlation(x, y, method):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3 or np.std(x) <= 0.0 or np.std(y) <= 0.0: return int(x.size), np.nan, np.nan
    if method == "pearson":
        result = pearsonr(x, y)
    elif method == "spearman":
        result = spearmanr(x, y)
    else: raise ValueError("method must be 'pearson' or 'spearman'.")
    return int(x.size), float(result.statistic), float(result.pvalue)


def correlation_metrics(frame):
    variables = [
        "mean_I", "mean_U", "mean_abs_U", "std_U",
        "mean_abs_dI", "mean_abs_dU", "mean_Temp", "std_Temp",
        "time_sin", "time_cos",
    ]
    rows = []
    for variable in variables:
        n_p, pearson_r, pearson_p = _correlation(frame["s_I"], frame[variable], "pearson")
        n_s, spearman_rho, spearman_p = _correlation(frame["s_I"], frame[variable], "spearman")
        rows.append({
            "variable": variable, "observations": min(n_p, n_s),
            "pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_rho": spearman_rho, "spearman_p": spearman_p,
        })
    result = pd.DataFrame(rows)
    result["max_abs_correlation"] = np.nanmax(
        np.abs(result[["pearson_r", "spearman_rho"]].to_numpy(dtype=float)), axis=1,
    )
    return result.sort_values("max_abs_correlation", ascending=False).reset_index(drop=True)


def save_plot(date, days, frame, path):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(frame["hour"], frame["s_I"], marker="o", label="Observed demand uncertainty $s_I$")
    ax.set_xticks(np.arange(0, DAY_HOURS, 2))
    ax.set_xlim(0, DAY_HOURS - 1)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("$s_I$ across corresponding weekdays")
    ax.set_title(f"Observed input uncertainty before {date:%Y-%m-%d} | {len(days)} corresponding {date.day_name()}s")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(date, days, frame, correlations, plot_path):
    print(f"prediction date={date:%Y-%m-%d} ({date.day_name()})")
    print("corresponding weekdays=" + ", ".join(day.strftime("%Y-%m-%d") for day in days))
    print(f"s_I mean={frame['s_I'].mean():.8f}, min={frame['s_I'].min():.8f} at hour={int(frame.loc[frame['s_I'].idxmin(), 'hour'])}, max={frame['s_I'].max():.8f} at hour={int(frame.loc[frame['s_I'].idxmax(), 'hour'])}")
    print()
    print("correlation of observed uncertainty s_I with empirical/model variables")
    columns = ["variable", "observations", "pearson_r", "pearson_p", "spearman_rho", "spearman_p"]
    print(correlations[columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print("s_I is estimated only from the five observed I trajectories; O is not used.")
    print(f"plot={plot_path}")


def analyze(date="2022-08-22", weeks=DEFAULT_WEEKS, output_root=OUTPUT):
    date = day_start(date)
    days = corresponding_days(date, weeks)
    frame, _ = aligned_wm4_data("MED")
    stacked = _stack_days(frame, days)
    hourly = uncertainty_frame(stacked)
    correlations = correlation_metrics(hourly)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = f"{date:%Y-%m-%d}_{int(weeks)}_weekdays"
    hourly_path = output_root / f"{stem}_hourly_uncertainty.csv"
    correlation_path = output_root / f"{stem}_correlations.csv"
    plot_path = output_root / f"{stem}_s_I.png"
    hourly.to_csv(hourly_path, index=False)
    correlations.to_csv(correlation_path, index=False)
    save_plot(date, days, hourly, plot_path)
    return {
        "date": date, "days": days, "hourly": hourly, "correlations": correlations,
        "hourly_csv": hourly_path, "correlations_csv": correlation_path, "plot": plot_path,
    }


def main():
    parser = ArgumentParser(description="Estimate observed hourly demand uncertainty from corresponding weekdays and correlate it with WM4-relevant variables.")
    parser.add_argument("--date", default="2022-08-22", help="Prediction date; the previous corresponding weekdays are analyzed.")
    parser.add_argument("--weeks", type=int, default=DEFAULT_WEEKS, help="Number of previous corresponding weekdays; default 5.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    result = analyze(args.date, args.weeks, args.output)
    print_summary(result["date"], result["days"], result["hourly"], result["correlations"], result["plot"])


if __name__ == "__main__": main()
