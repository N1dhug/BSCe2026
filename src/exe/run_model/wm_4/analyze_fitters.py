from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
PREDICTION = ROOT / "files" / "output" / "wm_4" / "prediction"
WARMUP = PREDICTION / "all_days"
OPTIMIZED = PREDICTION / "all_days_optimized"
BRUTE_FORCE = PREDICTION / "all_days_brute_force"
OUTPUT = PREDICTION / "fitter_analysis"
METHODS = ("warmup", "optimized", "brute_force")
PARAMETERS = ("r_F", "r_U", "r_R")
REQUIRED = {"day", "weekday", "wm4_mse", "wm4_rmse", "wm4_mae", "wm4_bias", "O_mse", "F_0", "R_0", *PARAMETERS}


def load_method(name, root):
    path = Path(root) / "daily_metrics.csv"
    if not path.exists(): raise FileNotFoundError(f"Missing {path}. Run the {name} all-days predictor first.")
    frame = pd.read_csv(path, parse_dates=["day"])
    missing = sorted(REQUIRED.difference(frame.columns))
    if missing: raise RuntimeError(f"{path} is missing required columns: {', '.join(missing)}.")
    frame = frame.sort_values("day").drop_duplicates("day", keep="last").reset_index(drop=True)
    frame["method"] = name
    return frame


def common_days(frames):
    common = set(frames[METHODS[0]]["day"])
    for method in METHODS[1:]: common &= set(frames[method]["day"])
    if not common: raise RuntimeError("The three fitters have no overlapping prediction days.")
    return sorted(common)


def aligned(frames):
    days = common_days(frames)
    aligned = {}
    for method, frame in frames.items():
        aligned[method] = frame.loc[frame["day"].isin(days)].sort_values("day").reset_index(drop=True)
    return aligned


def method_summary(frames):
    rows = []
    for method, frame in frames.items():
        better = frame["wm4_mse"] < frame["O_mse"]
        rows.append({
            "method": method, "days": len(frame),
            "mean_mse": frame["wm4_mse"].mean(), "median_mse": frame["wm4_mse"].median(),
            "mean_rmse": frame["wm4_rmse"].mean(), "mean_mae": frame["wm4_mae"].mean(),
            "mean_abs_bias": np.abs(frame["wm4_bias"]).mean(),
            "mean_O_mse": frame["O_mse"].mean(),
            "aggregate_improvement_vs_O_pct": 100.0 * (frame["O_mse"].mean() - frame["wm4_mse"].mean()) / frame["O_mse"].mean(),
            "better_than_O_days": int(better.sum()), "better_than_O_pct": 100.0 * float(better.mean()),
            "F0_mean": frame["F_0"].mean(), "F0_std": frame["F_0"].std(),
            "R0_mean": frame["R_0"].mean(), "R0_std": frame["R_0"].std(),
        })
    return pd.DataFrame(rows)


def pairwise_summary(frames):
    rows, daily = [], []
    for a, b in combinations(METHODS, 2):
        left, right = frames[a], frames[b]
        merged = left[["day", "wm4_mse"]].merge(right[["day", "wm4_mse"]], on="day", suffixes=(f"_{a}", f"_{b}"))
        a_mse, b_mse = merged[f"wm4_mse_{a}"], merged[f"wm4_mse_{b}"]
        delta = b_mse - a_mse
        merged["method_a"], merged["method_b"] = a, b
        merged["mse_a"], merged["mse_b"], merged["mse_gain_a_over_b"] = a_mse, b_mse, delta
        merged["winner"] = np.where(delta > 0.0, a, np.where(delta < 0.0, b, "equal"))
        merged["winning_margin"] = np.abs(delta)
        merged["winning_margin_pct"] = np.where(np.maximum(a_mse, b_mse) > 0.0, 100.0 * np.abs(delta) / np.maximum(a_mse, b_mse), np.nan)
        daily.append(merged[["day", "method_a", "method_b", "mse_a", "mse_b", "mse_gain_a_over_b", "winner", "winning_margin", "winning_margin_pct"]])
        rows.append({
            "method_a": a, "method_b": b, "days": len(merged),
            "a_better_days": int((delta > 0.0).sum()), "b_better_days": int((delta < 0.0).sum()), "equal_days": int((delta == 0.0).sum()),
            "a_better_pct": 100.0 * float((delta > 0.0).mean()),
            "mean_mse_a": a_mse.mean(), "mean_mse_b": b_mse.mean(),
            "mean_mse_gain_a_over_b": delta.mean(),
            "aggregate_improvement_a_over_b_pct": 100.0 * (b_mse.mean() - a_mse.mean()) / b_mse.mean(),
            "mean_winning_margin": np.abs(delta).mean(),
            "mean_winning_margin_pct": merged["winning_margin_pct"].mean(),
        })
    return pd.DataFrame(rows), pd.concat(daily, ignore_index=True)


def daily_comparison(frames):
    base = frames["warmup"][["day", "weekday", "O_mse"]].copy()
    for method in METHODS:
        frame = frames[method]
        for column in ("wm4_mse", "wm4_rmse", "wm4_mae", "wm4_bias", "F_0", "R_0", *PARAMETERS):
            base[f"{column}_{method}"] = frame[column].to_numpy()
    mse_columns = [f"wm4_mse_{method}" for method in METHODS]
    mse = base[mse_columns].to_numpy(dtype=float)
    best = np.argmin(mse, axis=1)
    base["best_method"] = np.asarray(METHODS, dtype=object)[best]
    base["best_mse"] = np.min(mse, axis=1)
    base["mse_spread"] = np.max(mse, axis=1) - np.min(mse, axis=1)
    return base


def parameter_stability(frames):
    rows = []
    for method, frame in frames.items():
        for parameter in PARAMETERS:
            values = frame[parameter].to_numpy(dtype=float)
            mean, std = float(np.mean(values)), float(np.std(values, ddof=1))
            rows.append({
                "method": method, "parameter": parameter, "mean": mean, "std": std,
                "variance": float(np.var(values, ddof=1)),
                "relative_std_pct": 100.0 * std / abs(mean) if abs(mean) > 1e-12 else np.nan,
                "median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values)),
            })
    return pd.DataFrame(rows)


def win_summary(daily):
    counts = daily["best_method"].value_counts()
    rows = [{"method": method, "best_days": int(counts.get(method, 0)), "best_pct": 100.0 * float(counts.get(method, 0)) / len(daily)} for method in METHODS]
    return pd.DataFrame(rows)


def save_plots(daily, output):
    fig, ax = plt.subplots(figsize=(16, 6))
    for method in METHODS: ax.plot(daily["day"], daily[f"wm4_mse_{method}"], label=method)
    ax.set_xlabel("Prediction day"); ax.set_ylabel("Daily MSE"); ax.set_title("WM4 fitter prediction error")
    ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
    fig.savefig(output / "daily_mse_comparison.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    for method in METHODS: ax.plot(daily["day"], daily[f"F_0_{method}"], label=method)
    ax.axhline(0.0, linestyle="--"); ax.set_xlabel("Prediction day"); ax.set_ylabel("Prediction F0")
    ax.set_title("WM4 initial flexibility state by fitter"); ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
    fig.savefig(output / "F0_comparison.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    for parameter in PARAMETERS:
        fig, ax = plt.subplots(figsize=(16, 6))
        for method in METHODS: ax.plot(daily["day"], daily[f"{parameter}_{method}"], label=method)
        ax.set_xlabel("Prediction day"); ax.set_ylabel(parameter); ax.set_title(f"WM4 {parameter} by fitter")
        ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
        fig.savefig(output / f"{parameter}_comparison.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def print_summary(methods, pairwise, wins):
    print("method performance")
    print(methods.to_string(index=False))
    print(); print("pairwise performance")
    print(pairwise.to_string(index=False))
    print(); print("lowest daily MSE counts")
    print(wins.to_string(index=False))


def main():
    parser = ArgumentParser(description="Compare warmup, optimized-state and brute-force-F0 WM4 fitters on their common prediction days.")
    parser.add_argument("--warmup", default=str(WARMUP))
    parser.add_argument("--optimized", default=str(OPTIMIZED))
    parser.add_argument("--brute-force", default=str(BRUTE_FORCE))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    frames = aligned({
        "warmup": load_method("warmup", args.warmup),
        "optimized": load_method("optimized", args.optimized),
        "brute_force": load_method("brute_force", args.brute_force),
    })
    daily = daily_comparison(frames)
    methods = method_summary(frames)
    pairwise, pairwise_daily = pairwise_summary(frames)
    stability = parameter_stability(frames)
    wins = win_summary(daily)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_comparison.csv", index=False)
    methods.to_csv(output / "method_summary.csv", index=False)
    pairwise.to_csv(output / "pairwise_summary.csv", index=False)
    pairwise_daily.to_csv(output / "pairwise_by_day.csv", index=False)
    stability.to_csv(output / "parameter_stability.csv", index=False)
    wins.to_csv(output / "daily_best_counts.csv", index=False)
    save_plots(daily, output)
    print_summary(methods, pairwise, wins)


if __name__ == "__main__": main()
