from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OPTIMIZED = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_optimized"
OUTPUT = OPTIMIZED / "analysis"
PARAMETERS = ("r_F", "r_U", "r_R")
METRICS = ("mse", "rmse", "mae", "bias")


def load_results(root):
    root = Path(root)
    daily_path, hourly_path = root / "daily_metrics.csv", root / "hourly_predictions.csv"
    if not daily_path.exists(): raise FileNotFoundError(f"Missing {daily_path}. Run the optimized predictor first.")
    if not hourly_path.exists(): raise FileNotFoundError(f"Missing {hourly_path}. Run the optimized predictor first.")
    daily = pd.read_csv(daily_path, parse_dates=["day"])
    hourly = pd.read_csv(hourly_path, parse_dates=["time"]).set_index("time")
    return daily, hourly


def analyze_daily(frame, atol=1e-12):
    result = frame.copy()
    for metric in METRICS:
        if metric == "bias":
            gain = np.abs(result["O_bias"]) - np.abs(result["wm4_bias"])
        else:
            gain = result[f"O_{metric}"] - result[f"wm4_{metric}"]
        result[f"{metric}_gain"] = gain
        result[f"wm4_better_{metric}"] = gain > atol
        result[f"wm4_equal_{metric}"] = np.abs(gain) <= atol

    result["mse_improvement_pct"] = np.where(
        result["O_mse"] > atol,
        100.0 * result["mse_gain"] / result["O_mse"],
        np.nan,
    )
    result["performance"] = np.select(
        [result["wm4_better_mse"], result["wm4_equal_mse"]],
        ["better", "equal"],
        default="worse",
    )
    return result


def global_summary(daily, hourly):
    observed = hourly["I_observed"].to_numpy(dtype=float)
    wm4 = hourly["I_wm4"].to_numpy(dtype=float)
    baseline = hourly["O"].to_numpy(dtype=float)
    wm4_residual = wm4 - observed
    baseline_residual = baseline - observed
    wm4_mse = float(np.mean(wm4_residual ** 2))
    baseline_mse = float(np.mean(baseline_residual ** 2))
    better = int(daily["wm4_better_mse"].sum())
    equal = int(daily["wm4_equal_mse"].sum())
    worse = len(daily) - better - equal

    return pd.DataFrame([{
        "days": len(daily),
        "wm4_better_mse_days": better,
        "wm4_equal_mse_days": equal,
        "wm4_worse_mse_days": worse,
        "wm4_better_mse_pct": 100.0 * better / len(daily),
        "mean_daily_wm4_mse": daily["wm4_mse"].mean(),
        "mean_daily_O_mse": daily["O_mse"].mean(),
        "mean_daily_mse_gain": daily["mse_gain"].mean(),
        "median_daily_mse_gain": daily["mse_gain"].median(),
        "mean_daily_mse_improvement_pct": daily["mse_improvement_pct"].mean(),
        "global_wm4_mse": wm4_mse,
        "global_O_mse": baseline_mse,
        "global_mse_gain": baseline_mse - wm4_mse,
        "global_mse_improvement_pct": 100.0 * (baseline_mse - wm4_mse) / baseline_mse if baseline_mse > 0.0 else np.nan,
        "global_wm4_mae": float(np.mean(np.abs(wm4_residual))),
        "global_O_mae": float(np.mean(np.abs(baseline_residual))),
        "global_wm4_bias": float(np.mean(wm4_residual)),
        "global_O_bias": float(np.mean(baseline_residual)),
    }])


def weekday_summary(frame):
    rows = []
    for weekday, part in frame.groupby("weekday"):
        rows.append({
            "weekday": weekday,
            "days": len(part),
            "wm4_better_mse_days": int(part["wm4_better_mse"].sum()),
            "wm4_better_mse_pct": 100.0 * float(part["wm4_better_mse"].mean()),
            "mean_wm4_mse": part["wm4_mse"].mean(),
            "mean_O_mse": part["O_mse"].mean(),
            "mean_mse_gain": part["mse_gain"].mean(),
            "mean_mse_improvement_pct": part["mse_improvement_pct"].mean(),
            "mean_F0": part["F_0"].mean(),
            "std_F0": part["F_0"].std(),
            "mean_R0": part["R_0"].mean(),
            "std_R0": part["R_0"].std(),
        })
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    result = pd.DataFrame(rows)
    result["weekday"] = pd.Categorical(result["weekday"], categories=order, ordered=True)
    return result.sort_values("weekday").reset_index(drop=True)


def parameter_summary(frame):
    rows = []
    for parameter in PARAMETERS:
        values = frame[parameter]
        rows.append({
            "parameter": parameter,
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "median": values.median(),
            "max": values.max(),
        })
    return pd.DataFrame(rows)


def state_summary(optimized, training_states=None):
    rows = [{
        "scope": "prediction_initial_state",
        "F0_mean": optimized["F_0"].mean(), "F0_std": optimized["F_0"].std(),
        "F0_min": optimized["F_0"].min(), "F0_max": optimized["F_0"].max(),
        "R0_mean": optimized["R_0"].mean(), "R0_std": optimized["R_0"].std(),
        "R0_min": optimized["R_0"].min(), "R0_max": optimized["R_0"].max(),
    }]
    if training_states is not None and not training_states.empty:
        rows.append({
            "scope": "optimized_training_states",
            "F0_mean": training_states["F_0"].mean(), "F0_std": training_states["F_0"].std(),
            "F0_min": training_states["F_0"].min(), "F0_max": training_states["F_0"].max(),
            "R0_mean": training_states["R_0"].mean(), "R0_std": training_states["R_0"].std(),
            "R0_min": training_states["R_0"].min(), "R0_max": training_states["R_0"].max(),
        })
    return pd.DataFrame(rows)


def continuity_analysis(training_states):
    if training_states is None or training_states.empty: return pd.DataFrame()
    frame = training_states.copy()
    frame["prediction_day"] = pd.to_datetime(frame["prediction_day"])
    frame["training_day"] = pd.to_datetime(frame["training_day"])
    frame = frame.sort_values(["training_day", "prediction_day"]).drop_duplicates("training_day", keep="last")
    frame["weekday"] = frame["training_day"].dt.day_name()
    frame["previous_F0"] = frame["F_0"].shift(1)
    frame["previous_R0"] = frame["R_0"].shift(1)
    frame["F0_change"] = frame["F_0"] - frame["previous_F0"]
    frame["R0_change"] = frame["R_0"] - frame["previous_R0"]
    return frame


def save_plots(analyzed, output):
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(analyzed["O_mse"], analyzed["wm4_mse"], alpha=0.7)
    limit = max(float(analyzed["O_mse"].max()), float(analyzed["wm4_mse"].max()))
    ax.plot([0.0, limit], [0.0, limit], linestyle="--")
    ax.set_xlabel("Baseline O(t) daily MSE")
    ax.set_ylabel("Optimized-state WM4 daily MSE")
    ax.set_title("WM4 versus exogenous-output baseline")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "mse_baseline_vs_wm4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(analyzed["day"], analyzed["mse_improvement_pct"])
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Prediction day")
    ax.set_ylabel("MSE improvement over O(t) [%]")
    ax.set_title("Optimized-state WM4 improvement over baseline")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "mse_improvement_over_baseline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(analyzed["day"], analyzed["training_F0_mean"], label="Training F0 mean")
    ax.fill_between(
        analyzed["day"],
        analyzed["training_F0_mean"] - analyzed["training_F0_std"],
        analyzed["training_F0_mean"] + analyzed["training_F0_std"],
        alpha=0.2, label="+/- one training-state std",
    )
    ax.plot(analyzed["day"], analyzed["F_0"], label="Prediction F0 median")
    ax.set_xlabel("Prediction day")
    ax.set_ylabel("F0")
    ax.set_title("Optimized historical-state variation")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "optimized_F0_over_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    for ax, parameter in zip(axes, PARAMETERS):
        ax.plot(analyzed["day"], analyzed[parameter])
        ax.set_title(parameter)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Rolling five-week optimized-state WM4 parameters")
    fig.tight_layout()
    fig.savefig(output / "parameter_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary, weekday, parameter, state):
    row = summary.iloc[0]
    print(f"days={int(row['days'])}")
    print(f"WM4 better than O(t) on MSE: {int(row['wm4_better_mse_days'])}/{int(row['days'])} ({row['wm4_better_mse_pct']:.2f}%)")
    print(f"mean daily MSE: WM4={row['mean_daily_wm4_mse']:.10f}, O(t)={row['mean_daily_O_mse']:.10f}, gain={row['mean_daily_mse_gain']:.10f}")
    print(f"global MSE: WM4={row['global_wm4_mse']:.10f}, O(t)={row['global_O_mse']:.10f}, improvement={row['global_mse_improvement_pct']:.2f}%")
    print()
    print("weekday performance")
    print(weekday.to_string(index=False))
    print()
    print("optimized parameter summary")
    print(parameter.to_string(index=False))
    print()
    print("state summary")
    print(state.to_string(index=False))


def main():
    parser = ArgumentParser(description="Analyze optimized-state wm_4 predictions only against the exogenous O(t) baseline.")
    parser.add_argument("--optimized", default=str(OPTIMIZED))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    optimized, hourly = load_results(args.optimized)
    analyzed = analyze_daily(optimized)
    training_path = Path(args.optimized) / "optimized_training_states.csv"
    training = pd.read_csv(training_path, parse_dates=["prediction_day", "training_day"]) if training_path.exists() else None

    summary = global_summary(analyzed, hourly)
    weekdays = weekday_summary(analyzed)
    parameters = parameter_summary(analyzed)
    states = state_summary(analyzed, training)
    continuity = continuity_analysis(training)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    analyzed.to_csv(output / "baseline_comparison_by_day.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    weekdays.to_csv(output / "weekday_comparison.csv", index=False)
    parameters.to_csv(output / "parameter_summary.csv", index=False)
    states.to_csv(output / "state_summary.csv", index=False)
    if not continuity.empty: continuity.to_csv(output / "optimized_state_continuity.csv", index=False)

    best = analyzed.sort_values("mse_gain", ascending=False).head(20)
    worst = analyzed.sort_values("mse_gain").head(20)
    best.to_csv(output / "largest_improvements_over_baseline.csv", index=False)
    worst.to_csv(output / "largest_regressions_vs_baseline.csv", index=False)
    save_plots(analyzed, output)
    print_summary(summary, weekdays, parameters, states)


if __name__ == "__main__": main()
