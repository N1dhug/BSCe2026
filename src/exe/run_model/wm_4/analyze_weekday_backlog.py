"""Rank historical backlog N using saved WM4 oracle N-k sweep scores.

The same forecast dates are used for every baseline, backlog and horizon.
Pooled RMSE weights each k-hour forecast by its k observed hourly errors.
The equal-horizon alternative gives each horizon k the same weight.
No model fitting or initial-state optimization is performed here.
"""
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT / "files" / "output" / "wm_4" / "weekday_horizon_sweep_expanded"
BASELINES = ("statistical_output", "weekly_average", "weekly_temperature")
LABELS = {"statistical_output": "Statistical", "weekly_average": "Weekly average",
          "weekly_temperature": "Temperature-adjusted weekly", "all_baselines": "All baselines (pooled)"}
KEY = ["day", "baseline", "training_weekdays", "horizon_hours"]


def read_scores(directory):
    path = Path(directory)/"oracle_horizon_scores.csv"
    if not path.is_file(): raise FileNotFoundError(f"Saved sweep scores not found: {path}")
    scores = pd.read_csv(path)
    missing = set(KEY + ["mse"])-set(scores.columns)
    if missing: raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if scores.empty: raise ValueError(f"No saved sweep scores in {path}")
    scores["day"] = pd.to_datetime(scores["day"]).dt.normalize()
    for col in ("training_weekdays", "horizon_hours", "mse"):
        scores[col] = pd.to_numeric(scores[col], errors="raise")
    if (scores["mse"] < 0).any() or not np.isfinite(scores["mse"]).all():
        raise ValueError("Saved MSE values must be finite and nonnegative.")
    if scores["day"].isna().any(): raise ValueError("Saved sweep contains invalid dates.")
    return scores.drop_duplicates(KEY, keep="last")


def evaluate(scores, *, baselines=BASELINES, n_values=range(2, 15), horizons=range(1, 25)):
    """Return backlog, per-horizon and best-N tables on a strictly matched date cohort."""
    baselines = tuple(baselines)
    n_values, horizons = tuple(sorted(set(n_values))), tuple(sorted(set(horizons)))
    if len(baselines) != 3 or set(baselines) != set(BASELINES):
        raise ValueError("The complete comparison requires all three sweep baselines.")
    if not n_values or not horizons or min(n_values) < 2 or min(horizons) < 1 or max(horizons) > 24:
        raise ValueError("Specify valid nonempty backlog N >= 2 and horizons k in 1..24.")
    selection = scores.loc[scores["baseline"].isin(baselines) &
                           scores["training_weekdays"].isin(n_values) &
                           scores["horizon_hours"].isin(horizons)].drop_duplicates(KEY, keep="last").copy()
    required = set(product(baselines, n_values, horizons))
    existing = set(zip(selection["baseline"], selection["training_weekdays"], selection["horizon_hours"]))
    missing = required-existing
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(f"Sweep is missing {len(missing)} baseline/N/k configurations (e.g. {examples}). "
                         "Run the complete sweep or narrow the selected N/k range.")
    counts = selection.groupby("day").size()
    common_dates = counts.index[counts == len(required)]
    if not len(common_dates):
        raise ValueError("No forecast dates have all three baselines and every requested N and k. "
                         "Complete/resume the sweep or narrow the selected range.")
    matched = selection.loc[selection["day"].isin(common_dates)].copy()
    matched["weighted_mse"] = matched["mse"] * matched["horizon_hours"]

    def pooled(frame, grouping):
        result = (frame.groupby(grouping, sort=True).agg(
            weighted_sse=("weighted_mse", "sum"), hourly_errors=("horizon_hours", "sum"),
            average_mse=("mse", "mean"), origins=("day", "nunique")).reset_index())
        result["pooled_rmse"] = np.sqrt(result.pop("weighted_sse") / result["hourly_errors"])
        result["equal_horizon_rmse"] = np.sqrt(result.pop("average_mse"))
        return result

    # Every (baseline, N, k) has exactly the same dates and k observations per date.
    by_horizon = (matched.groupby(["baseline", "training_weekdays", "horizon_hours"], sort=True)
                  .agg(mse=("mse", "mean"), origins=("day", "nunique")).reset_index())
    by_horizon["pooled_rmse"] = np.sqrt(by_horizon["mse"])
    common_per_k = (matched.groupby(["training_weekdays", "horizon_hours"], sort=True)
                    .agg(mse=("mse", "mean"), origins=("day", "nunique")).reset_index())
    common_per_k["baseline"] = "all_baselines"
    common_per_k["pooled_rmse"] = np.sqrt(common_per_k["mse"])
    by_horizon = pd.concat([by_horizon, common_per_k[by_horizon.columns]], ignore_index=True)

    by_baseline = pooled(matched, ["baseline", "training_weekdays"])
    combined = pooled(matched, ["training_weekdays"])
    combined["baseline"] = "all_baselines"
    backlog = pd.concat([by_baseline, combined[by_baseline.columns]], ignore_index=True)
    backlog["origins"] = backlog["origins"].astype(int)
    backlog["hourly_errors"] = backlog["hourly_errors"].astype(int)
    optimum = backlog.groupby("baseline")["pooled_rmse"].transform("min")
    backlog["gap_to_best_pct"] = np.where(optimum > 0,
                                           100*(backlog["pooled_rmse"]-optimum)/optimum, 0.0)

    # A tied RMSE is assigned to the smaller N, as a parsimonious tie break.
    winners = (by_horizon.sort_values(["baseline", "horizon_hours", "pooled_rmse", "training_weekdays"])
               .drop_duplicates(["baseline", "horizon_hours"])
               .rename(columns={"training_weekdays": "best_backlog_N", "pooled_rmse": "best_rmse"})
               [["baseline", "horizon_hours", "best_backlog_N", "best_rmse", "origins"]]
               .sort_values(["baseline", "horizon_hours"]).reset_index(drop=True))
    wins = (winners.groupby(["baseline", "best_backlog_N"]).size()
            .rename("horizons_won").reset_index().rename(columns={"best_backlog_N": "training_weekdays"}))
    backlog = backlog.merge(wins, on=["baseline", "training_weekdays"], how="left")
    backlog["horizons_won"] = backlog["horizons_won"].fillna(0).astype(int)
    backlog = backlog.sort_values(["baseline", "pooled_rmse", "training_weekdays"]).reset_index(drop=True)
    return backlog, by_horizon, winners, common_dates


def plot(backlog, winners, output, *, horizons):
    ns = sorted(backlog["training_weekdays"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.35, 1]})
    order = BASELINES + ("all_baselines",)
    for baseline in order:
        part = backlog.loc[backlog["baseline"] == baseline].sort_values("training_weekdays")
        axes[0].plot(part["training_weekdays"], part["pooled_rmse"], marker="o", ms=4,
                     lw=2.2 if baseline == "all_baselines" else 1.5,
                     ls="--" if baseline == "all_baselines" else "-", label=LABELS[baseline])
        best = part.nsmallest(1, "pooled_rmse").iloc[0]
        axes[0].scatter([best["training_weekdays"]], [best["pooled_rmse"]],
                        s=110, facecolors="none", edgecolors="black", linewidths=1.2, zorder=5)
    axes[0].set_xlabel("Historical matching weekdays N")
    axes[0].set_ylabel("Pooled RMSE (normalized demand)")
    axes[0].set_title("Pooled over all selected forecast horizons")
    axes[0].set_xticks(ns); axes[0].grid(alpha=.2); axes[0].legend(fontsize=8)
    width = .2
    x = np.arange(len(ns))
    for index, baseline in enumerate(order):
        values = winners.loc[winners["baseline"] == baseline, "best_backlog_N"].value_counts()
        axes[1].bar(x+(index-1.5)*width, [int(values.get(n, 0)) for n in ns], width,
                    label=LABELS[baseline])
    axes[1].set_xticks(x, ns); axes[1].set_xlabel("Historical matching weekdays N")
    axes[1].set_ylabel("Number of horizons with lowest RMSE")
    axes[1].set_title("Winning backlog by horizon (ties favor smaller N)")
    axes[1].grid(axis="y", alpha=.2); axes[1].legend(fontsize=8)
    fig.suptitle(f"Affine revised-model oracle: backlog comparison, k={min(horizons)}..{max(horizons)} h")
    fig.tight_layout(); fig.savefig(output, dpi=190, bbox_inches="tight"); plt.close(fig)


def run(*, group="ALL", input_dir=None, output_dir=None, min_weekdays=2, max_weekdays=14,
        min_horizon=1, max_horizon=24):
    group = group.upper()
    if group not in ("ALL", "MED"): raise ValueError("group must be ALL or MED.")
    if min_weekdays < 2 or max_weekdays < min_weekdays:
        raise ValueError("Require 2 <= min_weekdays <= max_weekdays.")
    if not 1 <= min_horizon <= max_horizon <= 24:
        raise ValueError("Require 1 <= min_horizon <= max_horizon <= 24.")
    source = Path(input_dir) if input_dir is not None else INPUT/group.lower()
    output = Path(output_dir) if output_dir is not None else source/"backlog_analysis"
    output.mkdir(parents=True, exist_ok=True)
    backlog, by_horizon, winners, days = evaluate(
        read_scores(source), n_values=range(min_weekdays, max_weekdays+1),
        horizons=range(min_horizon, max_horizon+1))
    backlog.to_csv(output/"backlog_summary.csv", index=False)
    by_horizon.to_csv(output/"backlog_by_horizon.csv", index=False)
    winners.to_csv(output/"best_backlog_by_horizon.csv", index=False)
    plot(backlog, winners, output/"backlog_comparison.png", horizons=range(min_horizon, max_horizon+1))
    print(f"\nWM4 oracle backlog analysis: {group}, matched prediction dates={len(days)}, "
          f"N={min_weekdays}..{max_weekdays}, k={min_horizon}..{max_horizon} h")
    print("Primary ranking: sqrt(sum(k*MSE) / sum(k)) over matched origins and all horizons.")
    print("Equal-horizon RMSE: sqrt(mean(MSE)) across matched origins and horizons.")
    for baseline in BASELINES + ("all_baselines",):
        part = backlog.loc[backlog["baseline"] == baseline]
        ranked = part.sort_values(["pooled_rmse", "training_weekdays"])
        best = ranked.iloc[0]
        near = part.loc[part["gap_to_best_pct"] <= 1.0, "training_weekdays"].min()
        print(f"\n{LABELS[baseline]}: BEST N={int(best['training_weekdays'])} "
              f"pooled RMSE={best['pooled_rmse']:.7f}; smallest N within 1%={int(near)}")
        print(part[["training_weekdays", "pooled_rmse", "equal_horizon_rmse",
                    "gap_to_best_pct", "horizons_won"]].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nOutputs: {output}")
    print("Note: 1-2 h oracle horizons optimize two state variables using at most two observations.")
    print("For a less optimistic comparison, repeat using --min-horizon 3 (or 6).")
    return backlog, by_horizon, winners


def main():
    parser = ArgumentParser(description="Analyze which historical backlog N minimizes pooled WM4 oracle RMSE across k-hour horizons.")
    parser.add_argument("--group", choices=("ALL", "MED"), default="ALL")
    parser.add_argument("--input", default=None, help="Folder containing saved oracle_horizon_scores.csv.")
    parser.add_argument("--output", default=None, help="Directory for figures and summary CSVs.")
    parser.add_argument("--min-weekdays", type=int, default=2)
    parser.add_argument("--max-weekdays", type=int, default=14)
    parser.add_argument("--min-horizon", type=int, default=1)
    parser.add_argument("--max-horizon", type=int, default=24)
    args = parser.parse_args()
    return run(group=args.group, input_dir=args.input, output_dir=args.output,
               min_weekdays=args.min_weekdays, max_weekdays=args.max_weekdays,
               min_horizon=args.min_horizon, max_horizon=args.max_horizon)


if __name__ == "__main__": main()
