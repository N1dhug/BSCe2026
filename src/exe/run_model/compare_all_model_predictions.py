"""Run BM1 day-ahead and hourly-updated WM4 MPC predictions across a date range.

Run: python -m src.exe.run_model.compare_all_model_predictions --start-day 2022-02-28 --end-day 2022-08-22 --resume

Both models are evaluated on the identical WM4-normalized demand scale and
the same 24 target hours. BM1 predicts 24 hours ahead; WM4 MPC uses the
latest observed hourly demand for each one-hour-ahead forecast. Therefore
their information sets differ: compare operational performance, not skill
under equal information. Metrics pool all individual observations rather
than averaging daily RMSE or daily coverage percentages.
"""

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.compare_model_predictions import compare_day, save_comparison
from src.exe.run_model.wm_4.fit_daily import aligned_wm4_data
from src.exe.run_model.wm_4.predict_all_days_optimized import START_DAY, END_DAY
from src.tools.forecast_metrics import forecast_metrics
from src.validation.exe.run_model.compare_all_model_predictions import model_comparison_all_days_validation as checks


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "files" / "output" / "compare_model_predictions"
MODELS = (("BM1", "BM1"), ("WM4_MPC", "WM4"))


def scores(hourly, model):
    """Evaluate one model on the supplied rows, using exact pooled observations."""
    name, prefix = next((name, prefix) for name, prefix in MODELS if name == model)
    intervals = {
        level: (hourly[f"{prefix}_lower_{level}"].to_numpy(dtype=float),
            hourly[f"{prefix}_upper_{level}"].to_numpy(dtype=float))
        for level in (68, 95)
    }
    return forecast_metrics(hourly["I_actual"].to_numpy(dtype=float),
        hourly[name].to_numpy(dtype=float), intervals)


def metrics_table(hourly, groups=None, label=None):
    """Return pooled scores by group. RMSE is recomputed from all hourly errors."""
    if groups is None:
        subsets = [("all", hourly)]
    else:
        subsets = list(hourly.groupby(groups, sort=True))
    rows = []
    for value, part in subsets:
        for model, _ in MODELS:
            rows.append({**({label: value} if label is not None else {}),
                "model": model, "days": int(part.index.normalize().nunique()), **scores(part, model)})
    return pd.DataFrame(rows)


def report_plot(daily, aggregate, path):
    """Plot per-day RMSE and empirical interval coverage; mark nominal coverage."""
    days = sorted(pd.DatetimeIndex(daily["day"].unique()))
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0, 1.0]})
    styles = (("BM1", "#3269b0"), ("WM4_MPC", "#c85e37"))
    for name, color in styles:
        part = daily.loc[daily["model"] == name].set_index("day").reindex(days)
        axes[0].plot(days, part["rmse"], color=color, linewidth=1.3, label=name)
        axes[1].plot(days, part["coverage_68_pct"], color=color, linewidth=1.0, alpha=0.75, label=name)
        axes[2].plot(days, part["coverage_95_pct"], color=color, linewidth=1.0, alpha=0.75, label=name)
    axes[1].axhline(68.0, color="0.25", linestyle="--", linewidth=1, label="Nominal 68%")
    axes[2].axhline(95.0, color="0.25", linestyle="--", linewidth=1, label="Nominal 95%")
    axes[0].set_ylabel("Daily RMSE")
    axes[1].set_ylabel("68% coverage (%)")
    axes[2].set_ylabel("95% coverage (%)")
    axes[2].set_xlabel("Target day")
    for ax in axes:
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")
    axes[1].set_ylim(-5, 105)
    axes[2].set_ylim(-5, 105)
    by_name = aggregate.set_index("model")
    detail = " | ".join(f"{name}: RMSE={by_name.loc[name, 'rmse']:.4f}, 95% coverage={by_name.loc[name, 'coverage_95_pct']:.1f}%"
        for name, _ in styles)
    fig.suptitle("Daily BM1 vs hourly-updated WM4 MPC | " + detail + "\n"
        "Daily coverage uses 24 observations; BM1 day-ahead, WM4 1-hour-ahead")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_aggregate(hourly, daily, aggregate, skipped, start, end, settings, output_root):
    directory = Path(output_root) / "aggregate" / f"{start:%Y-%m-%d}_to_{end:%Y-%m-%d}"
    directory.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(directory / "hourly_predictions.csv", index_label="time")
    daily.to_csv(directory / "daily_metrics.csv", index=False)
    aggregate.to_csv(directory / "aggregate_metrics.csv", index=False)
    by_hour = metrics_table(hourly, hourly.index.hour, "hour")
    by_weekday = metrics_table(hourly, hourly.index.day_name(), "weekday")
    by_month = metrics_table(hourly, hourly.index.to_period("M").astype(str), "month")
    by_hour.to_csv(directory / "hourly_metrics.csv", index=False)
    by_weekday.to_csv(directory / "weekday_metrics.csv", index=False)
    by_month.to_csv(directory / "monthly_metrics.csv", index=False)
    pd.DataFrame(skipped, columns=["day", "reason"]).to_csv(directory / "skipped_days.csv", index=False)
    report_plot(daily, aggregate, directory / "aggregate_performance.png")
    metadata = {
        "first_day": str(start.date()), "last_day": str(end.date()),
        "requested_days": int((end - start).days + 1),
        "evaluated_days": int(hourly.index.normalize().nunique()),
        "evaluated_hours": int(len(hourly)), "skipped_days": len(skipped),
        "settings": settings,
        "interpretation": "BM1 uses no target-day demand (24h forecast); WM4 MPC uses the previous hourly observation (1h lead). These are unequal-information forecasts.",
        "intervals": "BM1 central equal-tail empirical/FP quantiles; WM4 Gaussian +/-1 sigma (68.27%) and +/-1.96 sigma (95%).",
        "aggregation": "Scores and empirical coverages recomputed over all successfully predicted hourly observations; no averaging of daily RMSE.",
        "aggregate": aggregate.set_index("model").to_dict(orient="index"),
    }
    (directory / "summary.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return directory


def print_aggregate(aggregate, attempted, skipped):
    successful = int(aggregate["days"].iloc[0])
    hours = int(aggregate["n"].iloc[0])
    print(f"\nAggregate comparison: {successful}/{attempted} days, {hours} matched hourly observations; {len(skipped)} skipped.")
    print("BM1 day-ahead vs WM4 hourly-updated MPC (one-hour lead). All scores use the SAME normalized demand scale.")
    print(f"{'model':<12}{'MSE':>12}{'RMSE':>12}{'MAE':>12}{'BIAS':>12}{'68% cover':>12}{'95% cover':>12}")
    for _, row in aggregate.iterrows():
        print(f"{row['model']:<12}{row['mse']:>12.7f}{row['rmse']:>12.7f}{row['mae']:>12.7f}{row['bias']:>+12.7f}{row['coverage_68_pct']:>11.2f}%{row['coverage_95_pct']:>11.2f}%")
        print(f"  intervals: 68% {int(row['coverage_68_count'])}/{hours} (width={row['width_68_mean']:.5f}); "
            f"95% {int(row['coverage_95_count'])}/{hours} (width={row['width_95_mean']:.5f})")
    print("Coverage is empirical; nominal 68% and 95% are benchmarks, not guaranteed per 24-hour day.")


def run_all_days(start=START_DAY, end=END_DAY, *, frame=None, tag="MED", X0=0.5, sigma_x=0.0,
    steps_per_hour=2, max_state_iter=12, state_tol=1e-8, bm1_max_iterations=6,
    bm1_max_nfev=150, bm1_state_maxiter=100, substeps=2, x_steps=101, fp_max_step=0.25,
    wm4_forecasts=None, bm1_saved_root=None, output_root=OUTPUT, resume=False,
    fail_fast=False, save_daily_plots=False, save_bm1=False):
    """Run, checkpoint, and aggregate every complete target day in [start, end]."""
    start, end = checks.dates(start, end)
    checks.settings(steps_per_hour, max_state_iter, state_tol, tag, sigma_x,
        bm1_max_iterations, bm1_max_nfev, bm1_state_maxiter, substeps, x_steps, fp_max_step)
    if frame is None: frame, _ = aligned_wm4_data(tag)
    frame = checks.source_frame(frame)
    path = wm4_forecasts if isinstance(wm4_forecasts, (str, Path)) else None
    wm4_rows = pd.read_csv(path) if path is not None else wm4_forecasts
    wm4_rows = checks.wm4_cache(wm4_rows, start, end)
    settings = {
        "tag": tag, "X0": float(X0), "sigma_x": float(sigma_x),
        "steps_per_hour": int(steps_per_hour), "max_state_iter": int(max_state_iter),
        "state_tol": float(state_tol), "bm1_max_iterations": int(bm1_max_iterations),
        "bm1_max_nfev": int(bm1_max_nfev), "bm1_state_maxiter": int(bm1_state_maxiter),
        "substeps": int(substeps), "x_steps": int(x_steps), "fp_max_step": float(fp_max_step),
        "wm4_forecasts": str(Path(path).resolve()) if path is not None else ("supplied DataFrame" if wm4_rows is not None else None),
        "bm1_saved_root": str(Path(bm1_saved_root).resolve()) if bm1_saved_root is not None else None,
    }
    forecast_days = pd.date_range(start, end, freq="D")
    collected, skipped = [], []
    print(f"Comparing {len(forecast_days)} days from {start:%Y-%m-%d} through {end:%Y-%m-%d}; model=BM1 vs WM4 MPC", flush=True)
    for number, day in enumerate(forecast_days, start=1):
        directory = Path(output_root) / f"{day:%Y-%m-%d}"
        try:
            cached = checks.cached(directory, day, settings) if resume else None
            if cached is None:
                result = compare_day(day, frame=frame, tag=tag, X0=X0, sigma_x=sigma_x,
                    steps_per_hour=steps_per_hour, max_state_iter=max_state_iter, state_tol=state_tol,
                    bm1_max_iterations=bm1_max_iterations, bm1_max_nfev=bm1_max_nfev,
                    bm1_state_maxiter=bm1_state_maxiter, substeps=substeps, x_steps=x_steps,
                    fp_max_step=fp_max_step, wm4_forecasts=wm4_rows,
                    bm1_saved_root=bm1_saved_root, save_bm1=save_bm1)
                result["settings"] = settings
                saved = save_comparison(result, output_root=output_root, save_plot=save_daily_plots)
                day_hourly = checks.hourly(result["hourly"], day)
                status = "fitted"
            else:
                day_hourly, status = cached, "resumed"
            collected.append(day_hourly)
            bm = scores(day_hourly, "BM1")
            wm = scores(day_hourly, "WM4_MPC")
            print(f"[{number:3d}/{len(forecast_days)}] {day:%Y-%m-%d} {status} | "
                f"RMSE BM1={bm['rmse']:.5f} WM4={wm['rmse']:.5f} | "
                f"95% coverage BM1={bm['coverage_95_pct']:.1f}% WM4={wm['coverage_95_pct']:.1f}%", flush=True)
        except (ValueError, TypeError, RuntimeError, OSError, KeyError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((str(day.date()), str(exc)))
            print(f"[{number:3d}/{len(forecast_days)}] {day:%Y-%m-%d} skipped: {exc}", flush=True)

    if not collected: raise RuntimeError("No days produced a complete matched BM1/WM4 forecast.")
    hourly = pd.concat(collected).sort_index()
    daily = metrics_table(hourly, hourly.index.normalize(), "day")
    daily["weekday"] = pd.DatetimeIndex(daily["day"]).day_name()
    aggregate = metrics_table(hourly)
    aggregate = aggregate[["model", "days", "n", "mse", "rmse", "mae", "bias",
        "coverage_68_pct", "coverage_68_count", "width_68_mean", "coverage_68_gap_pp",
        "coverage_95_pct", "coverage_95_count", "width_95_mean", "coverage_95_gap_pp"]]
    directory = save_aggregate(hourly, daily, aggregate, skipped, start, end, settings, output_root)
    print_aggregate(aggregate, len(forecast_days), skipped)
    print(f"Saved aggregate results: {directory}")
    return {"hourly": hourly, "daily": daily, "aggregate": aggregate,
        "skipped": skipped, "directory": directory, "settings": settings}


def main():
    parser = ArgumentParser(description="Predict and compare BM1 and WM4 MPC for every day in a date range, with pooled error and coverage metrics.")
    parser.add_argument("--start-day", default=START_DAY, help="First target date, inclusive.")
    parser.add_argument("--end-day", default=END_DAY, help="Last target date, inclusive.")
    parser.add_argument("--tag", default="MED", help="Consumption group (default MED).")
    parser.add_argument("--X0", type=float, default=0.5, help="BM1 pooled initial-state guess.")
    parser.add_argument("--sigma-x", type=float, default=0.0, help="Fixed BM1 process diffusion; 0 means initial-state uncertainty only.")
    parser.add_argument("--steps-per-hour", type=int, default=2, help="WM4 MPC integration grid resolution.")
    parser.add_argument("--max-state-iter", type=int, default=12, help="WM4 MPC alternating fit iterations per origin.")
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--bm1-max-iterations", type=int, default=6)
    parser.add_argument("--bm1-max-nfev", type=int, default=150)
    parser.add_argument("--bm1-state-maxiter", type=int, default=100)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--x-steps", type=int, default=101, help="BM1 FV cells for positive sigma-x.")
    parser.add_argument("--fp-max-step", type=float, default=0.25)
    parser.add_argument("--wm4-forecasts", type=Path, help="Optional existing WM4 mpc_forecasts.csv to avoid fitting thousands of hourly MPC origins.")
    parser.add_argument("--bm1-saved-root", type=Path, help="Optional saved BM1 prediction directory; use only forecasts on the same normalized data.")
    parser.add_argument("--save-bm1", action="store_true", help="Also save the BM1 individual fit for each target day.")
    parser.add_argument("--resume", action="store_true", help="Reuse matching per-day comparison files if a run was interrupted.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first missing or failed forecast instead of skipping the day.")
    parser.add_argument("--save-daily-plots", action="store_true", help="Save one BM1/WM4 figure per target day as well.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    return run_all_days(args.start_day, args.end_day, tag=args.tag, X0=args.X0,
        sigma_x=args.sigma_x, steps_per_hour=args.steps_per_hour,
        max_state_iter=args.max_state_iter, state_tol=args.state_tol,
        bm1_max_iterations=args.bm1_max_iterations, bm1_max_nfev=args.bm1_max_nfev,
        bm1_state_maxiter=args.bm1_state_maxiter, substeps=args.substeps,
        x_steps=args.x_steps, fp_max_step=args.fp_max_step, wm4_forecasts=args.wm4_forecasts,
        bm1_saved_root=args.bm1_saved_root, save_bm1=args.save_bm1,
        output_root=args.output_root, resume=args.resume, fail_fast=args.fail_fast,
        save_daily_plots=args.save_daily_plots)


if __name__ == "__main__": main()
