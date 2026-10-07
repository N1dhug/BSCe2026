from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import DAY_HOURS, aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY, fit_history_optimized, historical_optimized_state
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup
from src.params.wm_4 import optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "adaptive_traceback"
DEFAULT_TRACEBACK_WEEKS = [3, 4, 5, 6, 7, 8]
SELECTION_METRICS = ("historical_mse", "qp_loss")


def normalized_tracebacks(values):
    weeks = []
    for value in values:
        value = int(value)
        if value < 1: raise ValueError("Every traceback week count must be at least 1.")
        if value not in weeks: weeks.append(value)
    if not weeks: raise ValueError("At least one traceback week count is required.")
    return weeks


def candidate_result(frame, day, weeks, steps_per_hour, oracle_state=False, max_state_iter=50, state_tol=1e-8):
    output_model, output_fit, fit, _ = fit_history_optimized(
        frame, day, weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )
    _, actual, U, Time, model, fine_index = prediction_setup(frame, day, output_model, steps_per_hour)
    if oracle_state:
        prediction_data, fine_index, _, _ = model_input(frame, day, steps_per_hour, 0, output_model)
        oracle = optimized_qp_params(model).optimize_prediction_initial_state(prediction_data, fit["r"])
        init = np.asarray(oracle["init_cond"], dtype=float)
        state = oracle["deterministic_state"]
        state_mode = "oracle"
        prediction_state_fit_mse = float(oracle["loss"])
    else:
        init = historical_optimized_state(fit)
        state = model.deterministic_solver(init, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})
        state_mode = "historical_median"
        prediction_state_fit_mse = np.nan

    hourly_index = np.arange(DAY_HOURS, dtype=int) * int(steps_per_hour)
    observed = actual["I"].to_numpy(dtype=float)
    predicted = np.asarray(state["I"], dtype=float)[hourly_index]
    baseline = np.asarray(state["O"], dtype=float)[hourly_index]
    wm4 = deviation_metrics(observed, predicted)
    output = deviation_metrics(observed, baseline)
    r = np.asarray(fit["r"], dtype=float)

    candidate = {
        "day": day, "weekday": day.day_name(), "traceback_weeks": int(weeks),
        "historical_mse": float(fit["deterministic_loss"]), "qp_loss": float(fit["qp_loss"]),
        "state_iterations": int(fit["state_iterations"]),
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "F_0": init[0], "R_0": init[1],
        "prediction_state_mode": state_mode, "prediction_state_fit_mse": prediction_state_fit_mse,
        "wm4_mse": wm4["mse"], "wm4_rmse": wm4["rmse"], "wm4_mae": wm4["mae"], "wm4_bias": wm4["bias"],
        "O_mse": output["mse"], "O_rmse": output["rmse"], "O_mae": output["mae"], "O_bias": output["bias"],
        "O_fit_mse": float(output_fit["mse"]),
    }
    hourly = pd.DataFrame({
        "day": day, "traceback_weeks": int(weeks),
        "I_observed": observed, "I_wm4": predicted, "O": baseline,
        "U": np.asarray(state["U"], dtype=float)[hourly_index],
        "F": np.asarray(state["F"], dtype=float)[hourly_index],
        "R": np.asarray(state["R"], dtype=float)[hourly_index],
    }, index=actual.index)
    hourly.index.name = "time"
    return candidate, hourly


def select_day(frame, day, tracebacks, steps_per_hour, selection_metric="historical_mse", oracle_state=False, max_state_iter=50, state_tol=1e-8):
    if selection_metric not in SELECTION_METRICS: raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}.")
    candidates, hourly_by_weeks, failures = [], {}, []

    for weeks in tracebacks:
        try:
            candidate, hourly = candidate_result(
                frame, day, weeks, steps_per_hour,
                oracle_state=oracle_state,
                max_state_iter=max_state_iter, state_tol=state_tol,
            )
            candidates.append(candidate); hourly_by_weeks[int(weeks)] = hourly
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failures.append({"day": day, "traceback_weeks": int(weeks), "reason": str(exc)})

    if not candidates: raise RuntimeError("No traceback candidate produced a valid WM4 fit.")
    table = pd.DataFrame(candidates)
    best_index = int(table[selection_metric].idxmin())
    selected = dict(table.loc[best_index])
    selected_weeks = int(selected["traceback_weeks"])

    hindsight_index = int(table["wm4_mse"].idxmin())
    hindsight = table.loc[hindsight_index]
    selected.update({
        "selection_metric": selection_metric,
        "selected_fit_score": float(selected[selection_metric]),
        "hindsight_best_prediction_weeks": int(hindsight["traceback_weeks"]),
        "hindsight_best_prediction_mse": float(hindsight["wm4_mse"]),
        "selected_matches_hindsight_best": selected_weeks == int(hindsight["traceback_weeks"]),
        "valid_candidates": int(len(table)), "failed_candidates": int(len(failures)),
    })
    table["selected"] = table["traceback_weeks"].astype(int) == selected_weeks
    table["hindsight_best_prediction"] = table["traceback_weeks"].astype(int) == int(hindsight["traceback_weeks"])
    return selected, hourly_by_weeks[selected_weeks], table, failures


def run_all_days(frame, start=START_DAY, end=END_DAY, tracebacks=DEFAULT_TRACEBACK_WEEKS, steps_per_hour=60, selection_metric="historical_mse", oracle_state=False, fail_fast=False, max_state_iter=50, state_tol=1e-8):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")
    tracebacks = normalized_tracebacks(tracebacks)

    selected_rows, hourly_frames, candidate_frames, failures, skipped = [], [], [], [], []
    for day in pd.date_range(start, end, freq="D"):
        try:
            selected, hourly, candidates, day_failures = select_day(
                frame, day, tracebacks, steps_per_hour,
                selection_metric=selection_metric, oracle_state=oracle_state,
                max_state_iter=max_state_iter, state_tol=state_tol,
            )
            selected_rows.append(selected); hourly_frames.append(hourly); candidate_frames.append(candidates); failures.extend(day_failures)
            print(
                f"{day:%Y-%m-%d} | selected={int(selected['traceback_weeks'])} weeks | "
                f"{selection_metric}={selected['selected_fit_score']:.8g} | "
                f"WM4 MSE={selected['wm4_mse']:.8f} | "
                f"hindsight best={int(selected['hindsight_best_prediction_weeks'])} weeks"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc))); print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not selected_rows: raise RuntimeError("No days in the requested range could be fitted.")
    selected = pd.DataFrame(selected_rows).sort_values("day").reset_index(drop=True)
    hourly = pd.concat(hourly_frames).sort_index()
    candidates = pd.concat(candidate_frames, ignore_index=True)
    failures = pd.DataFrame(failures)
    return selected, hourly, candidates, failures, skipped


def save_plot(selected, candidates, path):
    fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=False)

    axes[0].step(selected["day"], selected["traceback_weeks"], where="mid", label="Selected traceback")
    axes[0].scatter(selected["day"], selected["traceback_weeks"], s=14)
    axes[0].set_ylabel("Weeks"); axes[0].legend(); axes[0].grid(True, alpha=0.25)

    axes[1].plot(selected["day"], selected["wm4_mse"], label="Adaptive selected prediction MSE")
    axes[1].plot(selected["day"], selected["hindsight_best_prediction_mse"], label="Hindsight best candidate MSE", alpha=0.8)
    axes[1].set_ylabel("Daily MSE"); axes[1].legend(); axes[1].grid(True, alpha=0.25)

    counts = selected["traceback_weeks"].value_counts().sort_index()
    axes[2].bar(counts.index.astype(str), counts.to_numpy(dtype=int))
    axes[2].set_xlabel("Selected traceback weeks"); axes[2].set_ylabel("Days selected")
    axes[2].grid(True, axis="y", alpha=0.25)

    metric = str(selected["selection_metric"].iloc[0])
    fig.suptitle(f"WM4 adaptive parameter traceback | selection by {metric}")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def print_summary(selected, hourly, candidates, failures, skipped):
    adaptive = deviation_metrics(hourly["I_observed"], hourly["I_wm4"])
    print()
    print(f"days fitted={len(selected)}, skipped={len(skipped)}, candidate fits={len(candidates)}, failed candidates={len(failures)}, hourly observations={len(hourly)}")
    print(f"selection metric={selected['selection_metric'].iloc[0]}")
    print(f"adaptive MSE={adaptive['mse']:.10f}, RMSE={adaptive['rmse']:.10f}, MAE={adaptive['mae']:.10f}, bias={adaptive['bias']:.10f}")
    print(f"median selected daily MSE={selected['wm4_mse'].median():.10f}")
    print(f"selected traceback matches hindsight-best prediction on {100.0 * selected['selected_matches_hindsight_best'].mean():.2f}% of days")
    print()
    print("selected traceback counts:")
    for weeks, count in selected["traceback_weeks"].value_counts().sort_index().items():
        print(f"  {int(weeks)} weeks: {int(count)} days ({100.0 * count / len(selected):.2f}%)")

    print()
    print("fixed traceback aggregate prediction quality over valid candidate days:")
    grouped = candidates.groupby("traceback_weeks")
    for weeks, group in grouped:
        print(
            f"  {int(weeks)} weeks | days={len(group):3d} | "
            f"mean MSE={group['wm4_mse'].mean():.10f} | "
            f"RMSE={np.sqrt(group['wm4_mse'].mean()):.10f} | "
            f"mean MAE={group['wm4_mae'].mean():.10f} | "
            f"mean historical MSE={group['historical_mse'].mean():.10f}"
        )

    if skipped:
        print(); print("skipped days:")
        for day, reason in skipped: print(f"{day:%Y-%m-%d}: {reason}")


def main():
    parser = ArgumentParser(description="Select the WM4 historical traceback length per prediction day using the best in-sample parameter fit, then evaluate out-of-sample prediction quality.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--traceback-weeks", type=int, nargs="+", default=DEFAULT_TRACEBACK_WEEKS, help="Candidate counts of corresponding historical weekdays. Default: 3 4 5 6 7 8.")
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="historical_mse", help="Historical-only criterion used to select the traceback. Default: historical_mse.")
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--oracle-state", action="store_true", help="Use the prediction-day oracle [F0,R0] for every traceback candidate, isolating parameter-fit quality from initial-state estimation.")
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    tracebacks = normalized_tracebacks(args.traceback_weeks)
    frame, _ = aligned_wm4_data("MED")
    selected, hourly, candidates, failures, skipped = run_all_days(
        frame, args.start_day, args.end_day, tracebacks, args.steps_per_hour,
        selection_metric=args.selection_metric, oracle_state=args.oracle_state, fail_fast=args.fail_fast,
        max_state_iter=args.max_state_iter, state_tol=args.state_tol,
    )

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "selected_daily_metrics.csv", index=False)
    hourly.to_csv(output / "selected_hourly_predictions.csv")
    candidates.to_csv(output / "candidate_traceback_fits.csv", index=False)
    if not failures.empty: failures.to_csv(output / "failed_candidates.csv", index=False)
    if skipped: pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)
    save_plot(selected, candidates, output / "adaptive_traceback_performance.png")
    print_summary(selected, hourly, candidates, failures, skipped)


if __name__ == "__main__": main()
