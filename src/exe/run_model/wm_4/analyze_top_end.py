from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
WARMUP = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_top_end"
OPTIMIZED = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_optimized_top_end"
OUTPUT = OPTIMIZED / "top_end_analysis"
PARAMETERS = ("r_0", "r_F", "r_U", "r_R")


def load_daily(root):
    path = Path(root) / "daily_metrics.csv"
    if not path.exists(): raise FileNotFoundError(f"Missing {path}. Run the corresponding top-end predictor first.")
    return pd.read_csv(path, parse_dates=["day"])


def paired_results(warmup, optimized):
    merged = warmup.merge(optimized, on="day", suffixes=("_warmup", "_optimized"), how="inner")
    if merged.empty: raise RuntimeError("Warmup and optimized parameterizations have no overlapping top-end prediction days.")
    merged["mse_gain"] = merged["wm4_mse_warmup"] - merged["wm4_mse_optimized"]
    merged["mse_improvement_pct"] = np.where(
        merged["wm4_mse_warmup"] > 0.0,
        100.0 * merged["mse_gain"] / merged["wm4_mse_warmup"],
        np.nan,
    )
    merged["optimized_better"] = merged["mse_gain"] > 0.0
    merged["F0_delta"] = merged["F_0_optimized"] - merged["F_0_warmup"]
    merged["R0_delta"] = merged["R_0_optimized"] - merged["R_0_warmup"]
    for parameter in PARAMETERS:
        merged[f"{parameter}_delta"] = merged[f"{parameter}_optimized"] - merged[f"{parameter}_warmup"]
    return merged


def summary_table(frame):
    return pd.DataFrame([{
        "days": len(frame),
        "optimized_better_days": int(frame["optimized_better"].sum()),
        "optimized_better_pct": 100.0 * float(frame["optimized_better"].mean()),
        "mean_warmup_mse": frame["wm4_mse_warmup"].mean(),
        "mean_optimized_mse": frame["wm4_mse_optimized"].mean(),
        "mean_mse_gain": frame["mse_gain"].mean(),
        "median_mse_gain": frame["mse_gain"].median(),
        "mean_mse_improvement_pct": frame["mse_improvement_pct"].mean(),
        "mean_abs_F0_delta": np.abs(frame["F0_delta"]).mean(),
        "mean_abs_R0_delta": np.abs(frame["R0_delta"]).mean(),
    }])


def state_summary(frame):
    return pd.DataFrame([
        {
            "method": "warmup_parameters",
            "F0_mean": frame["F_0_warmup"].mean(), "F0_std": frame["F_0_warmup"].std(),
            "R0_mean": frame["R_0_warmup"].mean(), "R0_std": frame["R_0_warmup"].std(),
            "state_fit_mse_mean": frame["prediction_state_fit_mse_warmup"].mean(),
        },
        {
            "method": "optimized_parameters",
            "F0_mean": frame["F_0_optimized"].mean(), "F0_std": frame["F_0_optimized"].std(),
            "R0_mean": frame["R_0_optimized"].mean(), "R0_std": frame["R_0_optimized"].std(),
            "state_fit_mse_mean": frame["prediction_state_fit_mse_optimized"].mean(),
        },
    ])


def parameter_summary(frame):
    rows = []
    for parameter in PARAMETERS:
        warmup, optimized = frame[f"{parameter}_warmup"], frame[f"{parameter}_optimized"]
        rows.append({
            "parameter": parameter,
            "warmup_mean": warmup.mean(),
            "optimized_mean": optimized.mean(),
            "warmup_std": warmup.std(),
            "optimized_std": optimized.std(),
            "mean_delta": (optimized - warmup).mean(),
            "mean_abs_delta": np.abs(optimized - warmup).mean(),
            "correlation": warmup.corr(optimized),
        })
    return pd.DataFrame(rows)


def weekday_summary(frame):
    rows = []
    for weekday, part in frame.groupby("weekday_warmup"):
        rows.append({
            "weekday": weekday,
            "days": len(part),
            "optimized_better_days": int(part["optimized_better"].sum()),
            "optimized_better_pct": 100.0 * float(part["optimized_better"].mean()),
            "mean_warmup_mse": part["wm4_mse_warmup"].mean(),
            "mean_optimized_mse": part["wm4_mse_optimized"].mean(),
            "mean_mse_gain": part["mse_gain"].mean(),
            "mean_mse_improvement_pct": part["mse_improvement_pct"].mean(),
        })
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    result = pd.DataFrame(rows)
    result["weekday"] = pd.Categorical(result["weekday"], categories=order, ordered=True)
    return result.sort_values("weekday").reset_index(drop=True)


def save_plots(frame, output):
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(frame["wm4_mse_warmup"], frame["wm4_mse_optimized"], alpha=0.7)
    limit = max(float(frame["wm4_mse_warmup"].max()), float(frame["wm4_mse_optimized"].max()))
    ax.plot([0.0, limit], [0.0, limit], linestyle="--")
    ax.set_xlabel("Warmup-parameter fit with oracle state MSE")
    ax.set_ylabel("Optimized-parameter fit with oracle state MSE")
    ax.set_title("WM4 top-end parameterization comparison")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "top_end_mse_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(frame["day"], frame["F_0_warmup"], label="Warmup-parameter oracle F0")
    ax.plot(frame["day"], frame["F_0_optimized"], label="Optimized-parameter oracle F0")
    ax.set_xlabel("Prediction day"); ax.set_ylabel("Oracle F0")
    ax.set_title("Prediction-day optimized initial flexibility state")
    ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "top_end_F0_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(frame["day"], frame["R_0_warmup"], label="Warmup-parameter oracle R0")
    ax.plot(frame["day"], frame["R_0_optimized"], label="Optimized-parameter oracle R0")
    ax.set_xlabel("Prediction day"); ax.set_ylabel("Oracle R0")
    ax.set_title("Prediction-day optimized initial flexibility rate")
    ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "top_end_R0_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(summary, states):
    row = summary.iloc[0]
    print(f"days={int(row['days'])}")
    print(f"optimized parameter fit better with oracle state on {int(row['optimized_better_days'])}/{int(row['days'])} days ({row['optimized_better_pct']:.2f}%)")
    print(f"mean top-end MSE: warmup={row['mean_warmup_mse']:.10f}, optimized={row['mean_optimized_mse']:.10f}, gain={row['mean_mse_gain']:.10f}")
    print(f"mean MSE improvement={row['mean_mse_improvement_pct']:.2f}%")
    print()
    print("oracle prediction-state summary")
    print(states.to_string(index=False))


def main():
    parser = ArgumentParser(description="Compare top-end WM4 performance when both parameterizations receive prediction-day optimized initial states.")
    parser.add_argument("--warmup", default=str(WARMUP))
    parser.add_argument("--optimized", default=str(OPTIMIZED))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    warmup = load_daily(args.warmup)
    optimized = load_daily(args.optimized)
    paired = paired_results(warmup, optimized)
    summary = summary_table(paired)
    states = state_summary(paired)
    parameters = parameter_summary(paired)
    weekdays = weekday_summary(paired)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)

    paired.to_csv(output / "paired_top_end_comparison.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    states.to_csv(output / "oracle_state_comparison.csv", index=False)
    parameters.to_csv(output / "parameter_comparison.csv", index=False)
    weekdays.to_csv(output / "weekday_comparison.csv", index=False)
    paired.sort_values("mse_gain", ascending=False).head(20).to_csv(output / "largest_optimized_improvements.csv", index=False)
    paired.sort_values("mse_gain").head(20).to_csv(output / "largest_optimized_regressions.csv", index=False)
    save_plots(paired, output)
    print_summary(summary, states)


if __name__ == "__main__": main()
