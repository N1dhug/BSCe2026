from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
CURRENT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days"
OPTIMIZED = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_optimized"
OUTPUT = OPTIMIZED / "analysis"
PARAMETERS = ("r_0", "r_F", "r_U", "r_R")
METRICS = ("mse", "rmse", "mae", "bias")


def load_results(root):
    root = Path(root)
    daily_path, hourly_path = root / "daily_metrics.csv", root / "hourly_predictions.csv"
    if not daily_path.exists(): raise FileNotFoundError(f"Missing {daily_path}. Run the corresponding predictor first.")
    if not hourly_path.exists(): raise FileNotFoundError(f"Missing {hourly_path}. Run the corresponding predictor first.")
    daily = pd.read_csv(daily_path, parse_dates=["day"])
    hourly = pd.read_csv(hourly_path, parse_dates=["time"]).set_index("time")
    return daily, hourly


def paired_days(current, optimized):
    merged = current.merge(optimized, on="day", suffixes=("_current", "_optimized"), how="inner")
    if merged.empty: raise RuntimeError("Current and optimized predictions have no overlapping days.")
    merged["weekday"] = merged["day"].dt.day_name()
    for metric in METRICS:
        left, right = f"wm4_{metric}_current", f"wm4_{metric}_optimized"
        if metric == "bias":
            merged[f"{metric}_gain"] = np.abs(merged[left]) - np.abs(merged[right])
        else:
            merged[f"{metric}_gain"] = merged[left] - merged[right]
    merged["mse_improvement_pct"] = np.where(
        merged["wm4_mse_current"] > 0.0,
        100.0 * merged["mse_gain"] / merged["wm4_mse_current"],
        np.nan,
    )
    merged["optimized_better"] = merged["mse_gain"] > 0.0
    for parameter in PARAMETERS:
        merged[f"{parameter}_delta"] = merged[f"{parameter}_optimized"] - merged[f"{parameter}_current"]
    return merged


def global_summary(paired, current_hourly, optimized_hourly):
    current_hourly = current_hourly.loc[current_hourly.index.intersection(optimized_hourly.index)]
    optimized_hourly = optimized_hourly.loc[current_hourly.index]
    current_residual = current_hourly["I_wm4"].to_numpy(dtype=float) - current_hourly["I_observed"].to_numpy(dtype=float)
    optimized_residual = optimized_hourly["I_wm4"].to_numpy(dtype=float) - optimized_hourly["I_observed"].to_numpy(dtype=float)
    return pd.DataFrame([{
        "days": len(paired),
        "optimized_better_days": int(paired["optimized_better"].sum()),
        "optimized_better_pct": 100.0 * float(paired["optimized_better"].mean()),
        "mean_current_mse": paired["wm4_mse_current"].mean(),
        "mean_optimized_mse": paired["wm4_mse_optimized"].mean(),
        "mean_daily_mse_gain": paired["mse_gain"].mean(),
        "median_daily_mse_gain": paired["mse_gain"].median(),
        "mean_daily_mse_improvement_pct": paired["mse_improvement_pct"].mean(),
        "global_current_mse": float(np.mean(current_residual ** 2)),
        "global_optimized_mse": float(np.mean(optimized_residual ** 2)),
        "global_current_mae": float(np.mean(np.abs(current_residual))),
        "global_optimized_mae": float(np.mean(np.abs(optimized_residual))),
        "global_current_bias": float(np.mean(current_residual)),
        "global_optimized_bias": float(np.mean(optimized_residual)),
    }])


def weekday_summary(paired):
    rows = []
    for weekday, part in paired.groupby("weekday"):
        rows.append({
            "weekday": weekday,
            "days": len(part),
            "optimized_better_days": int(part["optimized_better"].sum()),
            "optimized_better_pct": 100.0 * float(part["optimized_better"].mean()),
            "mean_current_mse": part["wm4_mse_current"].mean(),
            "mean_optimized_mse": part["wm4_mse_optimized"].mean(),
            "mean_mse_gain": part["mse_gain"].mean(),
            "mean_mse_improvement_pct": part["mse_improvement_pct"].mean(),
            "mean_F0": part["F_0_optimized"].mean(),
            "std_F0": part["F_0_optimized"].std(),
        })
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    result = pd.DataFrame(rows)
    result["weekday"] = pd.Categorical(result["weekday"], categories=order, ordered=True)
    return result.sort_values("weekday").reset_index(drop=True)


def parameter_summary(paired):
    rows = []
    for parameter in PARAMETERS:
        current, optimized = paired[f"{parameter}_current"], paired[f"{parameter}_optimized"]
        rows.append({
            "parameter": parameter,
            "current_mean": current.mean(), "optimized_mean": optimized.mean(),
            "current_std": current.std(), "optimized_std": optimized.std(),
            "mean_delta": (optimized - current).mean(),
            "mean_abs_delta": np.abs(optimized - current).mean(),
            "correlation": current.corr(optimized),
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


def save_plots(paired, optimized, output):
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(paired["wm4_mse_current"], paired["wm4_mse_optimized"], alpha=0.7)
    limit = max(float(paired["wm4_mse_current"].max()), float(paired["wm4_mse_optimized"].max()))
    ax.plot([0.0, limit], [0.0, limit], linestyle="--")
    ax.set_xlabel("Current QP daily MSE"); ax.set_ylabel("Optimized-state QP daily MSE")
    ax.set_title("WM4 parameter-fit comparison"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "mse_current_vs_optimized.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(paired["day"], paired["mse_improvement_pct"])
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Prediction day"); ax.set_ylabel("MSE improvement over current fit [%]")
    ax.set_title("Effect of optimizing historical initial states"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "mse_improvement_by_day.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(optimized["day"], optimized["training_F0_mean"], label="Training F0 mean")
    ax.fill_between(
        optimized["day"],
        optimized["training_F0_mean"] - optimized["training_F0_std"],
        optimized["training_F0_mean"] + optimized["training_F0_std"],
        alpha=0.2, label="+/- one training-state std",
    )
    ax.plot(optimized["day"], optimized["F_0"], label="Prediction F0 median")
    ax.set_xlabel("Prediction day"); ax.set_ylabel("F0"); ax.set_title("Optimized historical-state variation")
    ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "optimized_F0_over_time.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, parameter in zip(axes.flat, PARAMETERS):
        ax.plot(paired["day"], paired[f"{parameter}_current"], label="Current")
        ax.plot(paired["day"], paired[f"{parameter}_optimized"], label="Optimized-state")
        ax.set_title(parameter); ax.grid(True, alpha=0.25)
    axes[0, 0].legend(); fig.suptitle("Rolling five-week WM4 parameter estimates")
    fig.tight_layout(); fig.savefig(output / "parameter_comparison.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def print_summary(summary, weekday, parameter, state):
    row = summary.iloc[0]
    print(f"paired days={int(row['days'])}")
    print(f"optimized-state parameterization better on {int(row['optimized_better_days'])}/{int(row['days'])} days ({row['optimized_better_pct']:.2f}%)")
    print(f"mean daily MSE: current={row['mean_current_mse']:.10f}, optimized={row['mean_optimized_mse']:.10f}, gain={row['mean_daily_mse_gain']:.10f}")
    print(f"global MSE: current={row['global_current_mse']:.10f}, optimized={row['global_optimized_mse']:.10f}")
    print()
    print("weekday performance")
    print(weekday.to_string(index=False))
    print()
    print("parameter comparison")
    print(parameter.to_string(index=False))
    print()
    print("state summary")
    print(state.to_string(index=False))


def main():
    parser = ArgumentParser(description="Complete paired analysis of current and optimized-state wm_4 QP parameter fits.")
    parser.add_argument("--current", default=str(CURRENT))
    parser.add_argument("--optimized", default=str(OPTIMIZED))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    current, current_hourly = load_results(args.current)
    optimized, optimized_hourly = load_results(args.optimized)
    training_path = Path(args.optimized) / "optimized_training_states.csv"
    training = pd.read_csv(training_path, parse_dates=["prediction_day", "training_day"]) if training_path.exists() else None

    paired = paired_days(current, optimized)
    summary = global_summary(paired, current_hourly, optimized_hourly)
    weekdays = weekday_summary(paired)
    parameters = parameter_summary(paired)
    states = state_summary(optimized, training)
    continuity = continuity_analysis(training)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)

    paired.to_csv(output / "paired_daily_comparison.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    weekdays.to_csv(output / "weekday_comparison.csv", index=False)
    parameters.to_csv(output / "parameter_comparison.csv", index=False)
    states.to_csv(output / "state_summary.csv", index=False)
    if not continuity.empty: continuity.to_csv(output / "optimized_state_continuity.csv", index=False)

    best = paired.sort_values("mse_gain", ascending=False).head(20)
    worst = paired.sort_values("mse_gain").head(20)
    best.to_csv(output / "largest_improvements.csv", index=False)
    worst.to_csv(output / "largest_regressions.csv", index=False)
    save_plots(paired, optimized, output)
    print_summary(summary, weekdays, parameters, states)


if __name__ == "__main__": main()
