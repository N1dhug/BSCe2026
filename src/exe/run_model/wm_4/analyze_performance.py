from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "files" / "output" / "wm_4" / "prediction"
DEFAULT_OUTPUT = DEFAULT_INPUT / "performance_analysis"
REQUIRED = {
    "day", "wm4_mse", "wm4_rmse", "wm4_mae", "wm4_bias",
    "O_mse", "O_rmse", "O_mae", "O_bias",
}


def metric_gain(frame, metric):
    if metric == "bias":
        return np.abs(frame["O_bias"]) - np.abs(frame["wm4_bias"])
    return frame[f"O_{metric}"] - frame[f"wm4_{metric}"]


def load_daily_metrics(root):
    root = Path(root)
    files = sorted(root.rglob("*.csv"))
    frames, skipped = [], []

    for path in files:
        try: frame = pd.read_csv(path)
        except Exception as exc:
            skipped.append((path, f"read error: {exc}"))
            continue

        if not REQUIRED.issubset(frame.columns):
            skipped.append((path, "missing daily metric columns"))
            continue

        frame = frame.copy()
        frame["source_file"] = str(path)
        frame["day"] = pd.to_datetime(frame["day"], errors="coerce")
        frame = frame.dropna(subset=["day"])
        if frame.empty:
            skipped.append((path, "no valid day rows"))
            continue
        frames.append(frame)

    if not frames: raise RuntimeError(f"No compatible daily metric CSV files found below {root}.")
    return pd.concat(frames, ignore_index=True), skipped


def analyze(frame, atol=1e-12):
    result = frame.copy()
    for metric in ("mse", "rmse", "mae", "bias"):
        gain = metric_gain(result, metric)
        result[f"{metric}_gain"] = gain
        result[f"wm4_better_{metric}"] = gain > atol
        result[f"wm4_equal_{metric}"] = np.abs(gain) <= atol

    result["mse_improvement_pct"] = np.where(
        result["O_mse"] > atol,
        100.0 * result["mse_gain"] / result["O_mse"],
        np.nan,
    )
    result["wm4_better_all_errors"] = (
        result["wm4_better_mse"]
        & result["wm4_better_rmse"]
        & result["wm4_better_mae"]
    )
    result["wm4_better_all_metrics"] = result["wm4_better_all_errors"] & result["wm4_better_bias"]
    result["performance"] = np.select(
        [result["wm4_better_mse"], result["wm4_equal_mse"]],
        ["better", "equal"],
        default="worse",
    )
    return result


def summary_table(frame):
    total = len(frame)
    better = int(frame["wm4_better_mse"].sum())
    equal = int(frame["wm4_equal_mse"].sum())
    worse = total - better - equal
    return pd.DataFrame([{
        "days": total,
        "wm4_better_mse_days": better,
        "wm4_equal_mse_days": equal,
        "wm4_worse_mse_days": worse,
        "wm4_better_mse_pct": 100.0 * better / total,
        "mean_wm4_mse": frame["wm4_mse"].mean(),
        "mean_O_mse": frame["O_mse"].mean(),
        "mean_mse_gain": frame["mse_gain"].mean(),
        "median_mse_gain": frame["mse_gain"].median(),
        "mean_mse_improvement_pct": frame["mse_improvement_pct"].mean(),
        "wm4_better_all_error_days": int(frame["wm4_better_all_errors"].sum()),
        "wm4_better_all_metric_days": int(frame["wm4_better_all_metrics"].sum()),
    }])


def weekday_summary(frame):
    if "weekday" not in frame.columns: return pd.DataFrame()
    grouped = frame.groupby("weekday", dropna=False)
    rows = []
    for weekday, part in grouped:
        rows.append({
            "weekday": weekday,
            "days": len(part),
            "wm4_better_mse_days": int(part["wm4_better_mse"].sum()),
            "wm4_better_mse_pct": 100.0 * float(part["wm4_better_mse"].mean()),
            "mean_wm4_mse": part["wm4_mse"].mean(),
            "mean_O_mse": part["O_mse"].mean(),
            "mean_mse_gain": part["mse_gain"].mean(),
            "mean_mse_improvement_pct": part["mse_improvement_pct"].mean(),
        })
    return pd.DataFrame(rows).sort_values("wm4_better_mse_pct", ascending=False)


def print_summary(summary, best, worst):
    row = summary.iloc[0]
    print(f"days={int(row['days'])}")
    print(f"WM4 better than O(t) on MSE: {int(row['wm4_better_mse_days'])}/{int(row['days'])} ({row['wm4_better_mse_pct']:.2f}%)")
    print(f"mean MSE: WM4={row['mean_wm4_mse']:.10f}, O(t)={row['mean_O_mse']:.10f}, gain={row['mean_mse_gain']:.10f}")
    print(f"mean MSE improvement={row['mean_mse_improvement_pct']:.2f}%")
    print()
    print("largest WM4 improvements:")
    for _, item in best.iterrows():
        print(f"{item['day']:%Y-%m-%d} | WM4={item['wm4_mse']:.10f} | O(t)={item['O_mse']:.10f} | improvement={item['mse_improvement_pct']:.2f}%")
    print()
    print("largest WM4 regressions:")
    for _, item in worst.iterrows():
        print(f"{item['day']:%Y-%m-%d} | WM4={item['wm4_mse']:.10f} | O(t)={item['O_mse']:.10f} | improvement={item['mse_improvement_pct']:.2f}%")


def main():
    parser = ArgumentParser(description="Identify when wm_4 outperforms the expected-output O(t) baseline in daily metric CSV files.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Directory to scan recursively for compatible daily metric CSV files.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Directory for analysis CSV files.")
    parser.add_argument("--top", type=int, default=10, help="Number of best/worst days printed.")
    args = parser.parse_args()

    frame, skipped = load_daily_metrics(args.input)
    analyzed = analyze(frame)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    better = analyzed[analyzed["performance"] == "better"].sort_values("mse_gain", ascending=False)
    equal = analyzed[analyzed["performance"] == "equal"].sort_values("day")
    worse = analyzed[analyzed["performance"] == "worse"].sort_values("mse_gain")
    summary = summary_table(analyzed)
    weekdays = weekday_summary(analyzed)

    analyzed.sort_values(["day", "source_file"]).to_csv(output / "performance_by_day.csv", index=False)
    better.to_csv(output / "wm4_better_days.csv", index=False)
    equal.to_csv(output / "wm4_equal_days.csv", index=False)
    worse.to_csv(output / "wm4_worse_days.csv", index=False)
    summary.to_csv(output / "performance_summary.csv", index=False)
    if not weekdays.empty: weekdays.to_csv(output / "performance_by_weekday.csv", index=False)
    if skipped:
        pd.DataFrame([{"file": str(path), "reason": reason} for path, reason in skipped]).to_csv(output / "skipped_csv_files.csv", index=False)

    print_summary(
        summary,
        better.head(max(0, args.top)),
        worse.head(max(0, args.top)),
    )


if __name__ == "__main__": main()
