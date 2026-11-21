from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from src.method.moving_horizon import OUTPUT, GROUPS, baseline_before, fit_row, group_frame, origins, plot_parameter_time, save_table, window_data


def fit_windows(group, start, end, stride, steps_per_hour, max_iter, max_origins=0, fail_fast=False):
    frame = group_frame(group)
    cache, rows, failures = {}, [], []
    timeline = origins(start, end, stride)
    if max_origins: timeline = timeline[:max_origins]

    for count, origin in enumerate(timeline, start=1):
        try:
            baseline = baseline_before(frame, origin, cache)
            model, data, _, observed = window_data(frame, origin - pd.Timedelta(hours=24), baseline, steps_per_hour)
            for intercept, variant in ((False, "zero"), (True, "zero_intercept")):
                fit = model.fit_params_numerical(data, [0.0, 0.0], fit_intercept=intercept, max_iter=max_iter)
                rows.append(fit_row(group, origin, variant, fit, observed, steps_per_hour))
                if not fit["success"]: failures.append({"group": group, "origin": origin, "variant": variant, "error": fit["message"]})
            if count % 24 == 0: print(f"[numerical] {group} {count}/{len(timeline)} origins; latest={origin}")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            failures.append({"group": group, "origin": origin, "variant": "both", "error": str(exc)})
            print(f"[numerical] {group} {origin} skipped: {exc}")
    return rows, failures


def main():
    parser = ArgumentParser(description="24-hour hourly moving direct-consumption numerical WM4 fitter, with optional drift intercept.")
    parser.add_argument("--start", default="2022-02-28")
    parser.add_argument("--end", default="2022-08-22 23:00")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--steps-per-hour", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument("--output", default=str(OUTPUT / "numerical"))
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for group in GROUPS:
        data, missed = fit_windows(group, args.start, args.end, args.stride, args.steps_per_hour,
                                   args.max_iter, args.max_origins, args.fail_fast)
        rows.extend(data); failures.extend(missed)
        if data:
            save_table(data, output / f"{group.lower()}_fit_parameters.csv")
            subdir = output / group.lower(); subdir.mkdir(parents=True, exist_ok=True)
            plot_parameter_time(pd.DataFrame(data), subdir, f"{group} numerical")
    save_table(rows, output / "fit_parameters.csv")
    save_table(failures, output / "skipped.csv")
    if not rows: raise RuntimeError("No numerical fitting windows succeeded.")
    result = pd.DataFrame(rows)
    summary = result.groupby(["group", "variant"], as_index=False).agg(
        windows=("fit_mse", "size"), mean_mse=("fit_mse", "mean"), mean_rmse=("fit_rmse", "mean"),
        mean_fit_time=("fit_time", "mean"), converged=("success", "sum"),
    )
    summary.to_csv(output / "fit_summary.csv", index=False)
    print(f"Numerical results: {output}; rows={len(result)}; warnings/failures={len(failures)}")
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
