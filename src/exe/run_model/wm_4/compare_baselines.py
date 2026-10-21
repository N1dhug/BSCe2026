from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.moving_horizon_common import GROUPS, HOURS, group_frame, metrics, origins, save_table, window_data
from src.params.wm_4 import optimized_qp_params
from src.tools.baseline import statistical_output, weekly_average, weekly_temperature
from src.validation.exe.run_model.wm_4.compare_baselines import compare_baselines_validation


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "baseline_comparison"
BASELINES = {
    "statistical_output": statistical_output,
    "weekly_average": weekly_average,
    "weekly_temperature": weekly_temperature,
}
METHODS = ("baseline", "standard", "oracle")


def baseline_before(frame, origin, name, cache):
    """One fit per method and calendar day, using only dates strictly before the day."""
    day = pd.Timestamp(origin).normalize()
    key = (name, day)
    if key not in cache: cache[key] = BASELINES[name]().fit(frame, day)
    return cache[key]


def fit_and_predict(frame, origin, name, baseline, steps_per_hour, seed_rate, max_state_iter, state_tol):
    """Fit the previous 24h; score an unadjusted continuation and a future-state oracle."""
    start = origin - pd.Timedelta(hours=HOURS)
    historical_model, train, _, historical_I = window_data(frame, start, baseline, steps_per_hour)
    forecast_model, future, _, future_I = window_data(frame, origin, baseline, steps_per_hour)
    fit_solver = optimized_qp_params(historical_model)
    seed_R0 = float(train["I"][0] - train["O"][0]) if seed_rate == "observed" else 0.0
    fit = fit_solver.fit(train, [0.0, seed_R0], fit_initial_state=True,
                         f_bounds=(-1.0, 1.0), r_bounds=(-1.0, 1.0),
                         max_state_iter=max_state_iter, state_tol=state_tol)
    history = fit.get("state_optimization_history", [])
    convergence = bool(history and history[-1]["max_state_change"] <= state_tol and
                        history[-1]["loss_change"] <= state_tol)

    # Genuine, non-oracle prediction: propagate the historically fitted trajectory
    # to the forecast origin, then hold its inferred terminal [F,R] as the new
    # initial state. No future consumption enters this branch.
    fitted = fit["deterministic_state"]
    terminal = np.array([fitted["F"][-1], fitted["R"][-1]], dtype=float)
    standard = forecast_model._solve_deterministic(
        terminal, future["U"], future["PE"], future["Time"], future["O"], fit["r"], r_0=0.0)

    # Retrospective optimistic bound: fit BOTH prediction initial states to
    # the *future* observations, with historical drift coefficients frozen.
    oracle = optimized_qp_params(forecast_model).optimize_prediction_initial_state(
        future, fit["r"], f_bounds=(-1.0, 1.0), r_bounds=(-1.0, 1.0))
    indices = np.arange(HOURS) * int(steps_per_hour)
    reference = np.asarray(future_I, dtype=float)
    predictions = {
        "baseline": np.asarray(future["O"][indices], dtype=float),
        "standard": np.asarray(standard["I"][indices], dtype=float),
        "oracle": np.asarray(oracle["deterministic_state"]["I"][indices], dtype=float),
    }
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("Nonfinite forecast from a WM4 fitting method.")

    train_pred = np.asarray(fitted["I"][indices], dtype=float)
    train_score = metrics(historical_I, train_pred)
    state_seed, state_fit = np.asarray(fit["initial_condition_seed"]), np.asarray(fit["init_cond"])
    row = {
        "group": None, "origin": pd.Timestamp(origin), "baseline": name,
        "r_F": float(fit["r"][0]), "r_U": float(fit["r"][1]), "r_R": float(fit["r"][2]),
        "F0_seed": float(state_seed[0]), "R0_seed": float(state_seed[1]),
        "F0_fit": float(state_fit[0]), "R0_fit": float(state_fit[1]),
        "F0_terminal": float(terminal[0]), "R0_terminal": float(terminal[1]),
        "F0_oracle": float(oracle["init_cond"][0]), "R0_oracle": float(oracle["init_cond"][1]),
        "terminal_F_outside_bounds": bool(not -1.0 <= terminal[0] <= 1.0),
        "terminal_R_outside_bounds": bool(not -1.0 <= terminal[1] <= 1.0),
        "fit_rmse": train_score["rmse"], "fit_mse": train_score["mse"],
        "fit_mae": train_score["mae"], "fit_bias": train_score["bias"],
        "qp_loss": float(fit["qp_loss"]),
        "fit_time": float(fit.get("qp_fit_time", np.nan)),
        "qp_solve_time": float(fit["qp_solve_time"]),
        "state_iterations": int(fit.get("state_iterations", 0)),
        "state_converged": convergence,
        "baseline_training_samples": int(baseline.metadata["samples"]),
        "baseline_training_mse": float(baseline.metadata["mse"]),
    }
    errors = {"group": None, "origin": pd.Timestamp(origin), "baseline": name}
    for method in METHODS:
        error = predictions[method] - reference
        for key, value in metrics(reference, predictions[method]).items():
            row[f"{method}_{key}"] = value
        for hour, value in enumerate(error, start=1):
            errors[f"{method}_error_{hour:02d}"] = float(value)
    sample = {"origin": pd.Timestamp(origin), "reference": reference, **predictions}
    return row, errors, sample


def matched_cohort(rows, errors, baseline_names):
    """Retain exactly the common forecast origins for each group and baseline."""
    if rows.empty: return rows, errors
    required = len(baseline_names)
    counts = rows.groupby(["group", "origin"])["baseline"].nunique()
    matched_index = counts.index[counts == required]
    if len(matched_index) == 0: return rows.iloc[:0].copy(), errors.iloc[:0].copy()
    keys = pd.DataFrame(list(matched_index), columns=["group", "origin"])
    return (rows.merge(keys, on=["group", "origin"], how="inner"),
            errors.merge(keys, on=["group", "origin"], how="inner"))


def summaries(rows, errors):
    summaries, horizons = [], []
    if rows.empty: return pd.DataFrame(), pd.DataFrame()
    for (group, name), subset in rows.groupby(["group", "baseline"], sort=True):
        score = {"group": group, "baseline": name, "windows": len(subset),
                 "observations": int(len(subset) * HOURS),
                 "mean_fit_rmse": float(subset["fit_rmse"].mean()),
                 "pooled_fit_rmse": float(np.sqrt(subset["fit_mse"].mean())),
                 "mean_qp_loss": float(subset["qp_loss"].mean()),
                 "mean_fit_time": float(subset["fit_time"].mean()),
                 "converged_pct": float(100 * subset["state_converged"].mean()),
                 "terminal_F_out_of_bounds_pct": float(100 * subset["terminal_F_outside_bounds"].mean()),
                 "terminal_R_out_of_bounds_pct": float(100 * subset["terminal_R_outside_bounds"].mean())}
        for method in METHODS:
            rmses = subset[f"{method}_rmse"]
            score[f"{method}_aggregate_rmse"] = float(np.sqrt(subset[f"{method}_mse"].mean()))
            score[f"{method}_mean_window_rmse"] = float(rmses.mean())
            score[f"{method}_median_window_rmse"] = float(rmses.median())
            score[f"{method}_q20"] = float(rmses.quantile(0.2))
            score[f"{method}_q80"] = float(rmses.quantile(0.8))
            score[f"{method}_mean_bias"] = float(subset[f"{method}_bias"].mean())
        score["standard_rmse_gain_vs_baseline_pct"] = float(
            100 * (1 - score["standard_aggregate_rmse"] / score["baseline_aggregate_rmse"]))
        score["oracle_rmse_gain_vs_baseline_pct"] = float(
            100 * (1 - score["oracle_aggregate_rmse"] / score["baseline_aggregate_rmse"]))
        summaries.append(score)

        indexed = errors.loc[(errors["group"] == group) & (errors["baseline"] == name)]
        for method in METHODS:
            cols = [f"{method}_error_{hour:02d}" for hour in range(1, HOURS+1)]
            matrix = indexed[cols].to_numpy(dtype=float)
            for hour in range(HOURS):
                values = matrix[:, hour]
                horizons.append({"group": group, "baseline": name, "method": method,
                                 "horizon": hour+1, "windows": len(values),
                                 "rmse": float(np.sqrt(np.mean(values**2))),
                                 "mae": float(np.mean(np.abs(values))), "bias": float(np.mean(values))})
    return (pd.DataFrame(summaries).sort_values(["group", "oracle_aggregate_rmse"]).reset_index(drop=True),
            pd.DataFrame(horizons))


def paired_gains(rows):
    """Paired per-window differences; negative MSE difference favors the first baseline.

    Rolling 24-hour windows overlap, so these descriptive win rates are not
    independent-observation hypothesis tests.
    """
    comparisons = []
    for group, subset in rows.groupby("group"):
        names = sorted(subset["baseline"].unique())
        for i, first in enumerate(names):
            for second in names[i+1:]:
                left = subset.loc[subset["baseline"] == first].set_index("origin")
                right = subset.loc[subset["baseline"] == second].set_index("origin")
                common = left.index.intersection(right.index)
                if len(common) < 1: continue
                for method in METHODS:
                    delta_mse = left.loc[common, f"{method}_mse"].to_numpy() - right.loc[common, f"{method}_mse"].to_numpy()
                    delta_rmse = left.loc[common, f"{method}_rmse"].to_numpy() - right.loc[common, f"{method}_rmse"].to_numpy()
                    comparisons.append({"group": group, "first": first, "second": second,
                                        "method": method, "windows": len(common),
                                        "mean_mse_first_minus_second": float(np.mean(delta_mse)),
                                        "mean_rmse_first_minus_second": float(np.mean(delta_rmse)),
                                        "first_better_pct": float(100 * np.mean(delta_mse < 0.0)),
                                        "equal_pct": float(100 * np.mean(delta_mse == 0.0))})
    return pd.DataFrame(comparisons)


def save_plots(summary, horizon, examples, output):
    for group in GROUPS:
        subset = summary.loc[summary["group"] == group].sort_values("baseline")
        if subset.empty: continue
        directory = output / group.lower(); directory.mkdir(parents=True, exist_ok=True)
        names = subset["baseline"].tolist()
        x = np.arange(len(names))
        width = 0.25
        fig, ax = plt.subplots(figsize=(11, 5))
        for offset, method in zip((-width, 0, width), METHODS):
            ax.bar(x+offset, subset[f"{method}_aggregate_rmse"], width, label=method)
        ax.set_xticks(x, names)
        ax.set_title(f"{group}: WM4 baseline vs standard forecast vs future-state oracle")
        ax.set_ylabel("Matched 24-hour aggregate RMSE"); ax.grid(axis="y", alpha=0.2); ax.legend()
        fig.tight_layout(); fig.savefig(directory / "aggregate_rmse.png", dpi=180); plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        for ax, method in zip(axes, METHODS):
            cases = horizon.loc[(horizon["group"] == group) & (horizon["method"] == method)]
            for name, values in cases.groupby("baseline"):
                ax.plot(values["horizon"], values["rmse"], label=name)
            ax.set_ylabel("RMSE"); ax.set_title(f"{method} forecast")
            ax.grid(True, alpha=0.2); ax.legend()
        axes[-1].set_xlabel("Forecast hour")
        fig.suptitle(f"{group}: RMSE by forecast horizon (common origins)")
        fig.tight_layout(); fig.savefig(directory / "rmse_by_horizon.png", dpi=180); plt.close(fig)

        example_items = [(name, examples[(group, name)]) for name in names if (group, name) in examples]
        if example_items:
            fig, axes = plt.subplots(len(example_items), 1, figsize=(13, 4*len(example_items)), sharex=True,
                                     squeeze=False)
            for ax, (name, sample) in zip(axes[:, 0], example_items):
                hours = np.arange(1, HOURS+1)
                ax.plot(hours, sample["reference"], label="observed", linewidth=1.5)
                for method in METHODS: ax.plot(hours, sample[method], label=method, alpha=0.8)
                ax.set_title(f"{name}, origin {sample['origin']:%Y-%m-%d %H:%M}")
                ax.set_ylabel("Normalized consumption"); ax.grid(True, alpha=0.2); ax.legend()
            axes[-1, 0].set_xlabel("Forecast hour")
            fig.tight_layout(); fig.savefig(directory / "example_24h_predictions.png", dpi=180); plt.close(fig)


def run(args):
    groups, names = compare_baselines_validation.arguments(args, BASELINES)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    timeline = origins(args.start, args.end, args.stride)
    if args.max_origins: timeline = timeline[:args.max_origins]
    rows, errors, failures, examples = [], [], [], {}
    for group in groups:
        frame, cache = group_frame(group), {}
        for count, origin in enumerate(timeline, start=1):
            for name in names:
                try:
                    baseline = baseline_before(frame, origin, name, cache)
                    row, err, example = fit_and_predict(
                        frame, origin, name, baseline, args.steps_per_hour,
                        args.seed_rate, args.max_state_iter, args.state_tol)
                    row["group"], err["group"] = group, group
                    rows.append(row); errors.append(err)
                    if (group, name) not in examples: examples[(group, name)] = example
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    if args.fail_fast: raise
                    failures.append({"group": group, "origin": origin, "baseline": name, "error": str(exc)})
            if count % args.progress_every == 0:
                print(f"[baseline comparison] {group}: {count}/{len(timeline)} origins, failures={len(failures)}")
    save_table(rows, output / "all_available_window_metrics.csv")
    save_table(errors, output / "all_available_hourly_errors_wide.csv")
    save_table(failures, output / "skipped.csv")
    if not rows: raise RuntimeError("No WM4 baseline-comparison windows completed successfully.")
    full_rows, full_errors = pd.DataFrame(rows), pd.DataFrame(errors)
    full_summary, full_horizon = summaries(full_rows, full_errors)
    full_summary.to_csv(output / "all_available_summary.csv", index=False)
    full_horizon.to_csv(output / "all_available_horizon.csv", index=False)

    matched_rows, matched_errors = matched_cohort(full_rows, full_errors, names)
    if matched_rows.empty:
        raise RuntimeError("No common forecast origins across all requested baselines.")
    summary, horizon = summaries(matched_rows, matched_errors)
    matched_rows.to_csv(output / "matched_window_metrics.csv", index=False)
    matched_errors.to_csv(output / "matched_hourly_errors_wide.csv", index=False)
    summary.to_csv(output / "matched_summary.csv", index=False)
    horizon.to_csv(output / "matched_horizon.csv", index=False)
    paired_gains(matched_rows).to_csv(output / "paired_baseline_gains.csv", index=False)
    save_plots(summary, horizon, examples, output)

    print(f"WM4 baseline comparison: {output}")
    print(f"Matched windows per baseline: {matched_rows.groupby('group')['origin'].nunique().to_dict()}")
    print(f"Unmatched/skipped fits: {len(failures)}")
    display = ("group", "baseline", "windows", "baseline_aggregate_rmse",
               "standard_aggregate_rmse", "oracle_aggregate_rmse",
               "mean_fit_rmse", "mean_qp_loss", "mean_fit_time", "converged_pct")
    print(summary[list(display)].to_string(index=False))
    return summary


def main():
    parser = ArgumentParser(description="Compare statistical, weekly-average and weekly-temperature baselines with WM4 QP, standard forward prediction and 24-hour oracle.")
    parser.add_argument("--start", default="2022-02-28")
    parser.add_argument("--end", default="2022-08-22 23:00")
    parser.add_argument("--groups", default="ALL,MED")
    parser.add_argument("--baselines", default="all", help="Comma-separated baseline names or 'all'.")
    parser.add_argument("--stride", type=int, default=1, help="Hours between prediction origins.")
    parser.add_argument("--steps-per-hour", type=int, default=4)
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument("--seed-rate", choices=("zero", "observed"), default="observed")
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--progress-every", type=int, default=24)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--fail-fast", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__": main()
