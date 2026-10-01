from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
WARMUP = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_top_end"
OPTIMIZED = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_optimized_top_end"
WARMUP_R0_ZERO = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_top_end_r0_zero"
OPTIMIZED_R0_ZERO = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_optimized_top_end_r0_zero"
OUTPUT = OPTIMIZED / "top_end_analysis"
OUTPUT_R0_ZERO = OPTIMIZED_R0_ZERO / "top_end_analysis"
PARAMETERS = ("r_0", "r_F", "r_U", "r_R")


def load_daily(root):
    path = Path(root) / "daily_metrics.csv"
    if not path.exists(): raise FileNotFoundError(f"Missing {path}. Run the corresponding top-end predictor first.")
    return pd.read_csv(path, parse_dates=["day"])


def paired_results(warmup, optimized):
    merged = warmup.merge(optimized, on="day", suffixes=("_warmup", "_optimized"), how="inner")
    if merged.empty: raise RuntimeError("Warmup and optimized parameterizations have no overlapping top-end prediction days.")
    if len(merged) < 4:
        raise RuntimeError(
            f"Top-end comparison requires at least 4 overlapping days for quartile analysis; "
            f"warmup rows={len(warmup)}, optimized rows={len(optimized)}, overlapping days={len(merged)}. "
            "Re-run both top-end predictors over the full date range and inspect their skipped-day output."
        )
    merged["mse_gain"] = merged["wm4_mse_warmup"] - merged["wm4_mse_optimized"]
    merged["mse_improvement_pct"] = np.where(
        merged["wm4_mse_warmup"] > 0.0,
        100.0 * merged["mse_gain"] / merged["wm4_mse_warmup"],
        np.nan,
    )
    merged["optimized_better"] = merged["mse_gain"] > 0.0
    merged["winner"] = np.where(merged["mse_gain"] > 0.0, "optimized", np.where(merged["mse_gain"] < 0.0, "warmup", "equal"))
    merged["pair_mean_mse"] = 0.5 * (merged["wm4_mse_warmup"] + merged["wm4_mse_optimized"])
    merged["winning_margin"] = np.abs(merged["mse_gain"])
    merged["winner_mse"] = np.minimum(merged["wm4_mse_warmup"], merged["wm4_mse_optimized"])
    merged["loser_mse"] = np.maximum(merged["wm4_mse_warmup"], merged["wm4_mse_optimized"])
    merged["winning_margin_pct"] = np.where(
        merged["loser_mse"] > 0.0,
        100.0 * merged["winning_margin"] / merged["loser_mse"],
        np.nan,
    )
    merged["error_quartile"] = pd.qcut(
        merged["pair_mean_mse"].rank(method="first"),
        4,
        labels=("Q1 low error", "Q2", "Q3", "Q4 high error"),
    )
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
        "aggregate_mse_improvement_pct": 100.0 * (frame["wm4_mse_warmup"].mean() - frame["wm4_mse_optimized"].mean()) / frame["wm4_mse_warmup"].mean(),
        "mean_winning_margin": frame["winning_margin"].mean(),
        "median_winning_margin": frame["winning_margin"].median(),
        "mean_winning_margin_pct": frame["winning_margin_pct"].mean(),
        "median_winning_margin_pct": frame["winning_margin_pct"].median(),
        "mean_abs_F0_delta": np.abs(frame["F0_delta"]).mean(),
        "mean_abs_R0_delta": np.abs(frame["R0_delta"]).mean(),
    }])


def winner_margin_summary(frame):
    rows = []
    for winner in ("optimized", "warmup"):
        part = frame.loc[frame["winner"] == winner]
        if part.empty: continue
        rows.append({
            "winner": winner,
            "wins": len(part),
            "mean_winner_mse": part["winner_mse"].mean(),
            "mean_loser_mse": part["loser_mse"].mean(),
            "mean_pair_mse": part["pair_mean_mse"].mean(),
            "median_pair_mse": part["pair_mean_mse"].median(),
            "mean_winning_margin": part["winning_margin"].mean(),
            "median_winning_margin": part["winning_margin"].median(),
            "mean_winning_margin_pct": part["winning_margin_pct"].mean(),
            "median_winning_margin_pct": part["winning_margin_pct"].median(),
        })
    return pd.DataFrame(rows)


def error_level_summary(frame):
    rows = []
    for error_quartile, part in frame.groupby("error_quartile", observed=True):
        optimized = part.loc[part["winner"] == "optimized"]
        warmup = part.loc[part["winner"] == "warmup"]
        rows.append({
            "error_quartile": str(error_quartile),
            "days": len(part),
            "mean_pair_mse": part["pair_mean_mse"].mean(),
            "min_pair_mse": part["pair_mean_mse"].min(),
            "max_pair_mse": part["pair_mean_mse"].max(),
            "optimized_wins": len(optimized),
            "warmup_wins": len(warmup),
            "optimized_win_pct": 100.0 * len(optimized) / len(part),
            "mean_winning_margin": part["winning_margin"].mean(),
            "mean_winning_margin_pct": part["winning_margin_pct"].mean(),
            "optimized_mean_margin": optimized["winning_margin"].mean() if not optimized.empty else np.nan,
            "optimized_mean_margin_pct": optimized["winning_margin_pct"].mean() if not optimized.empty else np.nan,
            "warmup_mean_margin": warmup["winning_margin"].mean() if not warmup.empty else np.nan,
            "warmup_mean_margin_pct": warmup["winning_margin_pct"].mean() if not warmup.empty else np.nan,
        })
    return pd.DataFrame(rows)


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


def parameter_stability(frame, f_reference=0.5):
    rows = []
    for method in ("warmup", "optimized"):
        values = {parameter: frame[f"{parameter}_{method}"].to_numpy(dtype=float) for parameter in PARAMETERS}
        centered = values["r_0"] + float(f_reference) * values["r_F"]
        kappa = -values["r_R"]
        derived = {
            **values,
            f"r_0+{float(f_reference):g}r_F": centered,
            "kappa_R": kappa,
        }
        for parameter, series in derived.items():
            mean = float(np.mean(series))
            std = float(np.std(series, ddof=1))
            rows.append({
                "method": method,
                "parameter": parameter,
                "mean": mean,
                "variance": float(np.var(series, ddof=1)),
                "std": std,
                "relative_std_pct": 100.0 * std / abs(mean) if abs(mean) > 1e-12 else np.nan,
                "q25": float(np.quantile(series, 0.25)),
                "median": float(np.median(series)),
                "q75": float(np.quantile(series, 0.75)),
                "iqr": float(np.quantile(series, 0.75) - np.quantile(series, 0.25)),
                "min": float(np.min(series)),
                "max": float(np.max(series)),
            })
    return pd.DataFrame(rows)


def same_weekday_parameter_stability(frame):
    rows = []
    weekdays = frame["weekday_warmup"].astype(str)
    for method in ("warmup", "optimized"):
        for parameter in PARAMETERS:
            changes = []
            levels = []
            for weekday in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"):
                part = frame.loc[weekdays == weekday].sort_values("day")
                values = part[f"{parameter}_{method}"].to_numpy(dtype=float)
                if values.size < 2: continue
                changes.extend(np.diff(values))
                levels.extend(values)
            changes = np.asarray(changes, dtype=float)
            levels = np.asarray(levels, dtype=float)
            rows.append({
                "method": method,
                "parameter": parameter,
                "paired_weekly_changes": int(changes.size),
                "mean_abs_weekly_change": float(np.mean(np.abs(changes))) if changes.size else np.nan,
                "median_abs_weekly_change": float(np.median(np.abs(changes))) if changes.size else np.nan,
                "rms_weekly_change": float(np.sqrt(np.mean(changes ** 2))) if changes.size else np.nan,
                "weekly_change_std": float(np.std(changes, ddof=1)) if changes.size > 1 else np.nan,
                "level_std": float(np.std(levels, ddof=1)) if levels.size > 1 else np.nan,
            })
    return pd.DataFrame(rows)


def r0_rf_identifiability(frame, f_reference=0.5):
    rows = []
    for method in ("warmup", "optimized"):
        r0 = frame[f"r_0_{method}"].to_numpy(dtype=float)
        rF = frame[f"r_F_{method}"].to_numpy(dtype=float)
        centered = r0 + float(f_reference) * rF
        r0_std = float(np.std(r0, ddof=1))
        rF_std = float(np.std(rF, ddof=1))
        correlation = float(np.corrcoef(r0, rF)[0, 1]) if r0_std > 1e-12 and rF_std > 1e-12 else np.nan
        rows.append({
            "method": method,
            "r0_rF_correlation": correlation,
            "r0_variance": float(np.var(r0, ddof=1)),
            "rF_variance": float(np.var(rF, ddof=1)),
            "centered_drift_reference_F": float(f_reference),
            "centered_drift_mean": float(np.mean(centered)),
            "centered_drift_variance": float(np.var(centered, ddof=1)),
            "centered_drift_std": float(np.std(centered, ddof=1)),
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

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(frame["pair_mean_mse"], frame["mse_gain"], alpha=0.7)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Mean MSE of the two models on the day")
    ax.set_ylabel("Warmup MSE - optimized MSE")
    ax.set_title("Which parameterization wins as day-level error increases")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "mse_gain_vs_error_level.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(frame["pair_mean_mse"], frame["winning_margin_pct"], alpha=0.7)
    ax.set_xlabel("Mean MSE of the two models on the day")
    ax.set_ylabel("Winner margin relative to worse MSE [%]")
    ax.set_title("Relative winning margin versus day-level error")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "relative_margin_vs_error_level.png", dpi=300, bbox_inches="tight")
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


def print_summary(summary, states, margins, error_levels, stability, weekly_stability, identifiability):
    row = summary.iloc[0]
    print(f"days={int(row['days'])}")
    print(f"optimized parameter fit better with oracle state on {int(row['optimized_better_days'])}/{int(row['days'])} days ({row['optimized_better_pct']:.2f}%)")
    print(f"mean top-end MSE: warmup={row['mean_warmup_mse']:.10f}, optimized={row['mean_optimized_mse']:.10f}, gain={row['mean_mse_gain']:.10f}")
    print(f"aggregate MSE improvement={row['aggregate_mse_improvement_pct']:.2f}%")
    print(f"mean absolute winning margin={row['mean_winning_margin']:.10f}, mean relative winning margin={row['mean_winning_margin_pct']:.2f}%")
    print()
    print("margin conditional on which model wins")
    print(margins.to_string(index=False))
    print()
    print("performance by day-level error quartile")
    print(error_levels.to_string(index=False))
    print()
    print("oracle prediction-state summary")
    print(states.to_string(index=False))
    print()
    print("parameter stability")
    print(stability.to_string(index=False))
    print()
    print("same-weekday week-to-week parameter changes")
    print(weekly_stability.to_string(index=False))
    print()
    print("r0/rF identifiability around F=0.5")
    print(identifiability.to_string(index=False))


def main():
    parser = ArgumentParser(description="Compare top-end WM4 performance when both parameterizations receive prediction-day optimized initial states.")
    parser.add_argument("--warmup", default=str(WARMUP))
    parser.add_argument("--optimized", default=str(OPTIMIZED))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--zero-intercept", action="store_true", help="Compare the r_0=0 top-end runs.")
    args = parser.parse_args()

    warmup_root = WARMUP_R0_ZERO if args.zero_intercept and args.warmup == str(WARMUP) else Path(args.warmup)
    optimized_root = OPTIMIZED_R0_ZERO if args.zero_intercept and args.optimized == str(OPTIMIZED) else Path(args.optimized)
    warmup = load_daily(warmup_root)
    optimized = load_daily(optimized_root)
    print(f"warmup rows={len(warmup)}, optimized rows={len(optimized)}, overlapping days={len(set(warmup['day']).intersection(set(optimized['day'])))}")
    paired = paired_results(warmup, optimized)
    summary = summary_table(paired)
    states = state_summary(paired)
    margins = winner_margin_summary(paired)
    error_levels = error_level_summary(paired)
    parameters = parameter_summary(paired)
    stability = parameter_stability(paired)
    weekly_stability = same_weekday_parameter_stability(paired)
    identifiability = r0_rf_identifiability(paired)
    weekdays = weekday_summary(paired)
    output = OUTPUT_R0_ZERO if args.zero_intercept and args.output == str(OUTPUT) else Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    paired.to_csv(output / "paired_top_end_comparison.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    states.to_csv(output / "oracle_state_comparison.csv", index=False)
    margins.to_csv(output / "winner_margin_summary.csv", index=False)
    error_levels.to_csv(output / "error_level_summary.csv", index=False)
    parameters.to_csv(output / "parameter_comparison.csv", index=False)
    stability.to_csv(output / "parameter_stability.csv", index=False)
    weekly_stability.to_csv(output / "same_weekday_parameter_stability.csv", index=False)
    identifiability.to_csv(output / "r0_rF_identifiability.csv", index=False)
    weekdays.to_csv(output / "weekday_comparison.csv", index=False)
    paired.sort_values("mse_gain", ascending=False).head(20).to_csv(output / "largest_optimized_improvements.csv", index=False)
    paired.sort_values("mse_gain").head(20).to_csv(output / "largest_optimized_regressions.csv", index=False)
    save_plots(paired, output)
    print_summary(summary, states, margins, error_levels, stability, weekly_stability, identifiability)


if __name__ == "__main__": main()
