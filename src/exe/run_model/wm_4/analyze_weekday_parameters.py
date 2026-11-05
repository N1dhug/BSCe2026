from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days" / "daily_metrics.csv"
DEFAULT_OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days" / "weekday_parameter_analysis"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CORE_PARAMETERS = ["r_F", "r_U", "r_R"]
REQUIRED = {
    "day", "r_F", "r_U", "r_R",
    "wm4_mse", "wm4_rmse", "wm4_mae", "wm4_bias",
    "O_mse", "O_rmse", "O_mae", "O_bias",
}


def load_metrics(path):
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("daily_metrics.csv"))
    frames = []
    for file in files:
        frame = pd.read_csv(file)
        if not REQUIRED.issubset(frame.columns): continue
        frame = frame.copy()
        frame["source_file"] = str(file)
        frames.append(frame)
    if not frames: raise RuntimeError(f"No compatible daily_metrics.csv found at {path}.")
    frame = pd.concat(frames, ignore_index=True)
    frame["day"] = pd.to_datetime(frame["day"], errors="coerce")
    frame = frame.dropna(subset=["day"]).copy()
    frame["weekday"] = frame["day"].dt.day_name()
    frame["mse_gain"] = frame["O_mse"] - frame["wm4_mse"]
    frame["mse_improvement_pct"] = np.where(frame["O_mse"] > 0.0, 100.0 * frame["mse_gain"] / frame["O_mse"], np.nan)
    frame["mae_gain"] = frame["O_mae"] - frame["wm4_mae"]
    frame["abs_bias_gain"] = np.abs(frame["O_bias"]) - np.abs(frame["wm4_bias"])
    frame["wm4_better"] = frame["mse_gain"] > 0.0
    return frame.sort_values("day").reset_index(drop=True)


def weekday_summary(frame):
    rows = []
    for weekday in WEEKDAYS:
        part = frame[frame["weekday"] == weekday]
        if part.empty: continue
        rows.append({
            "weekday": weekday, "days": len(part),
            "wm4_better_days": int(part["wm4_better"].sum()),
            "wm4_better_pct": 100.0 * float(part["wm4_better"].mean()),
            "mean_wm4_mse": part["wm4_mse"].mean(), "median_wm4_mse": part["wm4_mse"].median(),
            "mean_O_mse": part["O_mse"].mean(), "median_O_mse": part["O_mse"].median(),
            "mean_mse_gain": part["mse_gain"].mean(), "median_mse_gain": part["mse_gain"].median(),
            "mean_mse_improvement_pct": part["mse_improvement_pct"].mean(),
            "mean_wm4_rmse": part["wm4_rmse"].mean(), "mean_wm4_mae": part["wm4_mae"].mean(),
            "mean_abs_wm4_bias": np.abs(part["wm4_bias"]).mean(),
        })
    result = pd.DataFrame(rows)
    if result.empty: return result
    result["mse_rank"] = result["mean_wm4_mse"].rank(method="min", ascending=True).astype(int)
    result["gain_rank"] = result["mean_mse_gain"].rank(method="min", ascending=False).astype(int)
    return result


def weekday_tests(frame):
    rows = []
    for metric in ("wm4_mse", "mse_gain", "mse_improvement_pct"):
        groups = [part[metric].dropna().to_numpy(dtype=float) for _, part in frame.groupby("weekday") if part[metric].notna().sum() >= 2]
        if len(groups) < 2:
            rows.append({"metric": metric, "groups": len(groups), "statistic": np.nan, "p_value": np.nan})
            continue
        stat, p = kruskal(*groups)
        rows.append({"metric": metric, "groups": len(groups), "statistic": float(stat), "p_value": float(p)})
    return pd.DataFrame(rows)


def performance_groups(frame):
    result = frame.copy()
    q25, q75 = result["wm4_mse"].quantile([0.25, 0.75])
    result["absolute_group"] = np.select(
        [result["wm4_mse"] <= q25, result["wm4_mse"] >= q75],
        ["best_quartile", "worst_quartile"], default="middle",
    )
    g25, g75 = result["mse_gain"].quantile([0.25, 0.75])
    result["relative_group"] = np.select(
        [result["mse_gain"] >= g75, result["mse_gain"] <= g25],
        ["best_improvement_quartile", "worst_improvement_quartile"], default="middle",
    )
    return result


def parameter_group_summary(frame, parameters):
    rows = []
    for grouping in ("absolute_group", "relative_group"):
        for group, part in frame.groupby(grouping):
            base = {
                "grouping": grouping, "group": group, "days": len(part),
                "mean_wm4_mse": part["wm4_mse"].mean(), "median_wm4_mse": part["wm4_mse"].median(),
                "mean_mse_gain": part["mse_gain"].mean(), "wm4_better_pct": 100.0 * float(part["wm4_better"].mean()),
            }
            for parameter in parameters:
                values = part[parameter].dropna()
                base[f"{parameter}_mean"] = values.mean()
                base[f"{parameter}_median"] = values.median()
                base[f"{parameter}_std"] = values.std()
                base[f"{parameter}_q25"] = values.quantile(0.25)
                base[f"{parameter}_q75"] = values.quantile(0.75)
            rows.append(base)
    return pd.DataFrame(rows)


def parameter_correlations(frame, parameters):
    rows = []
    for parameter in parameters:
        for metric in ("wm4_mse", "mse_gain", "mse_improvement_pct"):
            data = frame[[parameter, metric]].dropna()
            rho = data[parameter].corr(data[metric], method="spearman") if len(data) >= 3 else np.nan
            rows.append({"parameter": parameter, "metric": metric, "days": len(data), "spearman_rho": rho})
    return pd.DataFrame(rows)


def parameter_bins(frame, parameters, bins=4):
    rows = []
    for parameter in parameters:
        part = frame[[parameter, "wm4_mse", "O_mse", "mse_gain", "mse_improvement_pct", "wm4_better"]].dropna(subset=[parameter]).copy()
        if part[parameter].nunique() < 2: continue
        try: part["bin"] = pd.qcut(part[parameter], q=min(bins, part[parameter].nunique()), duplicates="drop")
        except ValueError: continue
        for interval, bucket in part.groupby("bin", observed=True):
            rows.append({
                "parameter": parameter, "bin": str(interval), "days": len(bucket),
                "parameter_min": bucket[parameter].min(), "parameter_max": bucket[parameter].max(),
                "parameter_mean": bucket[parameter].mean(),
                "mean_wm4_mse": bucket["wm4_mse"].mean(), "mean_O_mse": bucket["O_mse"].mean(),
                "mean_mse_gain": bucket["mse_gain"].mean(),
                "mean_mse_improvement_pct": bucket["mse_improvement_pct"].mean(),
                "wm4_better_pct": 100.0 * float(bucket["wm4_better"].mean()),
            })
    result = pd.DataFrame(rows)
    if not result.empty: result["mse_rank_within_parameter"] = result.groupby("parameter")["mean_wm4_mse"].rank(method="min", ascending=True).astype(int)
    return result


def parameter_regions(frame, parameters, bins=3, min_days=3):
    work = frame.copy()
    labels = ["low", "mid", "high"]
    region_cols = []
    for parameter in parameters:
        if work[parameter].nunique(dropna=True) < 2: continue
        try:
            col = f"{parameter}_region"
            work[col] = pd.qcut(work[parameter], q=min(bins, work[parameter].nunique()), labels=labels[:min(bins, work[parameter].nunique())], duplicates="drop")
            region_cols.append(col)
        except ValueError: continue
    if not region_cols: return pd.DataFrame()

    rows = []
    for keys, part in work.dropna(subset=region_cols).groupby(region_cols, observed=True):
        if len(part) < min_days: continue
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {column: str(value) for column, value in zip(region_cols, keys)}
        row.update({
            "days": len(part), "mean_wm4_mse": part["wm4_mse"].mean(), "median_wm4_mse": part["wm4_mse"].median(),
            "mean_O_mse": part["O_mse"].mean(), "mean_mse_gain": part["mse_gain"].mean(),
            "mean_mse_improvement_pct": part["mse_improvement_pct"].mean(),
            "wm4_better_pct": 100.0 * float(part["wm4_better"].mean()),
        })
        for parameter in parameters: row[f"{parameter}_mean"] = part[parameter].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_wm4_mse") if rows else pd.DataFrame()


def save_weekday_plot(summary, path):
    ordered = summary.sort_values("mse_rank")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ordered["weekday"], ordered["mean_wm4_mse"], label="ARM")
    ax.plot(ordered["weekday"], ordered["mean_O_mse"], marker="o", label="O(t)")
    ax.set_ylabel("Mean daily MSE")
    ax.set_xlabel("Weekday")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_parameter_plot(frame, parameter, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(frame[parameter], frame["wm4_mse"], alpha=0.65)
    ax.set_xlabel(parameter)
    ax.set_ylabel("ARM daily MSE")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(weekdays, groups, correlations, bins):
    if not weekdays.empty:
        best = weekdays.sort_values("mean_wm4_mse").iloc[0]
        strongest = weekdays.sort_values("mean_mse_gain", ascending=False).iloc[0]
        print("weekday performance")
        print(weekdays[["weekday", "days", "mean_wm4_mse", "mean_O_mse", "mean_mse_gain", "wm4_better_pct"]].to_string(index=False))
        print()
        print(f"lowest mean WM4 MSE: {best['weekday']} ({best['mean_wm4_mse']:.10f})")
        print(f"largest mean improvement over O(t): {strongest['weekday']} ({strongest['mean_mse_gain']:.10f})")

    print()
    print("best vs worst parameter quartiles")
    columns = ["grouping", "group", "days", "mean_wm4_mse", "mean_mse_gain"] + [f"{p}_median" for p in CORE_PARAMETERS]
    print(groups[columns].to_string(index=False))

    print()
    print("strongest parameter/performance relationships")
    strength = correlations.dropna().assign(abs_rho=lambda x: np.abs(x["spearman_rho"])).sort_values("abs_rho", ascending=False).head(8)
    print(strength[["parameter", "metric", "spearman_rho", "days"]].to_string(index=False))

    if not bins.empty:
        print()
        print("best parameter ranges by mean WM4 MSE")
        best_bins = bins.sort_values(["parameter", "mse_rank_within_parameter"]).groupby("parameter", as_index=False).first()
        print(best_bins[["parameter", "bin", "days", "mean_wm4_mse", "mean_mse_gain", "wm4_better_pct"]].to_string(index=False))


def main():
    parser = ArgumentParser(description="Analyze wm_4 predictive performance by weekday and by fitted parameter regime.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="daily_metrics.csv or a directory containing daily_metrics.csv files.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--parameter-bins", type=int, default=4)
    parser.add_argument("--region-bins", type=int, default=3)
    parser.add_argument("--min-region-days", type=int, default=3)
    args = parser.parse_args()

    frame = load_metrics(args.input)
    parameters = CORE_PARAMETERS
    frame = performance_groups(frame)
    weekdays = weekday_summary(frame)
    tests = weekday_tests(frame)
    groups = parameter_group_summary(frame, parameters)
    correlations = parameter_correlations(frame, parameters)
    bins = parameter_bins(frame, parameters, max(2, args.parameter_bins))
    regions = parameter_regions(frame, CORE_PARAMETERS, max(2, args.region_bins), max(1, args.min_region_days))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "daily_classified.csv", index=False)
    weekdays.to_csv(output / "weekday_performance.csv", index=False)
    tests.to_csv(output / "weekday_statistical_tests.csv", index=False)
    groups.to_csv(output / "parameter_best_vs_worst.csv", index=False)
    correlations.to_csv(output / "parameter_correlations.csv", index=False)
    bins.to_csv(output / "parameter_ranges.csv", index=False)
    if not regions.empty: regions.to_csv(output / "parameter_regions.csv", index=False)

    if not weekdays.empty: save_weekday_plot(weekdays, output / "weekday_mse.png")
    for parameter in parameters: save_parameter_plot(frame, parameter, output / f"{parameter}_vs_mse.png")
    print_summary(weekdays, groups, correlations, bins)


if __name__ == "__main__": main()
