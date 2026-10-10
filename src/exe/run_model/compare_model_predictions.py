"""Compare BM1 day-ahead and hourly-updated WM4 MPC forecasts for one target day.

Run: python -m src.exe.run_model.compare_model_predictions --day 2022-08-22
WM4 updates its state each hour using the latest observed demand (one-hour
lead); BM1 predicts the whole day without target-day demand. Their information
sets differ, so the scores are descriptive rather than a fair skill ranking.
"""

from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.bm_1.predict_daily import predict_day as bm1_predict_day, save_result as save_bm1_result
from src.exe.run_model.bm_1.plot_prediction import reconstructed_demand
from src.exe.run_model.wm_4.fit_daily import aligned_wm4_data, fit_expected_output
from src.exe.run_model.wm_4.predict_all_days_mpc import _mpc_origin
from src.tools.forecast_metrics import forecast_metrics, predictive_intervals
from src.validation.exe.run_model.bm_1.plot_prediction import bm_1_plot_validation
from src.validation.exe.run_model.compare_model_predictions import model_comparison_validation as checks


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "files" / "output" / "compare_model_predictions"
BM1_OUTPUT = ROOT / "files" / "output" / "bm_1" / "prediction"


def wm4_mpc_day(frame, day, *, steps_per_hour=2, max_state_iter=12, state_tol=1e-8):
    """Build 24 hourly WM4 one-step forecasts with five shifted historical days each."""
    first_origin = day - pd.Timedelta(hours=1)
    origins = pd.date_range(first_origin, periods=24, freq="h")
    baseline_models, rows = {}, []
    for j, origin in enumerate(origins, start=1):
        date = origin.normalize()
        if date not in baseline_models: baseline_models[date] = fit_expected_output(frame, date)[0]
        try:
            predictions, state = _mpc_origin(frame, origin, 5, steps_per_hour, baseline_models[date], 1,
                max_state_iter=max_state_iter, state_tol=state_tol)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(f"WM4 MPC origin {origin} failed ({j}/24): {exc}") from exc
        rows.extend(predictions)
        print(f"WM4 MPC origin {j:02d}/24: {origin:%Y-%m-%d %H:%M} -> {origin + pd.Timedelta(hours=1):%H:%M}", flush=True)
    return pd.DataFrame(rows)


def bm1_outcomes_from_fit(result):
    """Reuse BM1 solver output; quantiles are computed before display smoothing."""
    hourly, fit, ensemble = result["hourly"], result["fit"], result["ensemble"]
    weights = np.asarray(result["initial_weights"], dtype=float)
    if result["sigma_x"] == 0:
        paths = np.asarray(ensemble["ensemble"], dtype=float)
        supports = [paths[:, k] for k in range(len(hourly))]
        return supports, [weights for _ in supports]

    x = np.asarray(ensemble["fp"]["x"], dtype=float)
    masses = np.asarray(ensemble["fp"]["cell_mass"], dtype=float)
    supports = []
    for k, row in enumerate(hourly.itertuples()):
        delta = np.tanh(0.5 * fit["k"] * (fit["f"](x) + fit["g"](row.U)))
        weight = np.where(delta >= 0.0, 1.0 - row.B, row.B)
        supports.append(row.B + fit["Xi"] * delta * weight)
    return supports, [masses[k] for k in range(len(hourly))]


def bm1_same_day(frame, day, *, tag, X0, sigma_x, max_iterations, max_nfev, state_maxiter, substeps, x_steps, fp_max_step, save_bm1):
    normalized, calibration = checks.bm1_scale(frame, day)
    result = bm1_predict_day(day, frame=normalized, tag=tag, X0=X0, sigma_x=sigma_x,
        max_iterations=max_iterations, max_nfev=max_nfev, state_maxiter=state_maxiter,
        substeps=substeps, x_steps=x_steps, fp_max_step=fp_max_step, score=False)
    result["normalization"] = calibration
    if save_bm1:
        output = save_bm1_result(result, plot=False)
        print(f"Saved BM1 fit: {output}")
    support, weights = bm1_outcomes_from_fit(result)
    return result["hourly"], support, weights, calibration, "new BM1 five-day fit"


def bm1_from_saved(day, *, output_root=BM1_OUTPUT, tag="MED", x_steps=101, fp_max_step=0.25, substeps=2):
    folder = Path(output_root) / f"{day:%Y-%m-%d}"
    hourly, metadata = bm_1_plot_validation.saved(folder, day)
    if metadata.get("tag", tag) != tag: raise ValueError("Saved BM1 measurement tag differs from requested tag.")
    support, weights, reconstructed_mean, method = reconstructed_demand(
        hourly, metadata, x_steps=x_steps, fp_max_step=fp_max_step, substeps=substeps)
    discrepancy = float(np.max(np.abs(reconstructed_mean - hourly["I_predicted"].to_numpy(dtype=float))))
    if discrepancy > 1e-3: raise ValueError(f"Reconstructed BM1 distribution differs from saved means by {discrepancy:.6f}; check original numerical settings.")
    return hourly, support, weights, metadata["normalization"], "saved BM1 fitted distribution"


def compare_day(day="2022-08-22", *, tag="MED", frame=None, X0=0.5, sigma_x=0.0, steps_per_hour=2,
    max_state_iter=12, state_tol=1e-8, bm1_max_iterations=6, bm1_max_nfev=150,
    bm1_state_maxiter=100, substeps=2, x_steps=101, fp_max_step=0.25,
    wm4_forecasts=None, bm1_saved_root=None, save_bm1=False):
    """Run or load both forecast models, then score all 24 hours on one demand scale."""
    day = checks.day(day)
    checks.configuration(steps_per_hour, max_state_iter, state_tol, tag)
    if frame is None: frame, _ = aligned_wm4_data(tag)
    frame = checks.common_frame(frame)

    if bm1_saved_root is None:
        bm_hourly, bm_support, bm_weights, calibration, bm_source = bm1_same_day(
            frame, day, tag=tag, X0=X0, sigma_x=sigma_x, max_iterations=bm1_max_iterations,
            max_nfev=bm1_max_nfev, state_maxiter=bm1_state_maxiter, substeps=substeps,
            x_steps=x_steps, fp_max_step=fp_max_step, save_bm1=save_bm1)
        bm_sigma = sigma_x
    else:
        bm_hourly, bm_support, bm_weights, calibration, bm_source = bm1_from_saved(
            day, output_root=bm1_saved_root, tag=tag, substeps=substeps,
            x_steps=x_steps, fp_max_step=fp_max_step)
        bm_sigma = bm_1_plot_validation.saved(Path(bm1_saved_root) / f"{day:%Y-%m-%d}", day)[1]["sigma_x"]

    if wm4_forecasts is None:
        mpc = wm4_mpc_day(frame, day, steps_per_hour=steps_per_hour, max_state_iter=max_state_iter, state_tol=state_tol)
        wm_source = "new hourly one-step WM4 MPC fit"
    else:
        mpc = wm4_forecasts if isinstance(wm4_forecasts, pd.DataFrame) else pd.read_csv(wm4_forecasts)
        wm_source = "preloaded WM4 MPC forecasts" if isinstance(wm4_forecasts, pd.DataFrame) else str(wm4_forecasts)

    rows, actual, timestamps = checks.hourly_wm4(frame, day, mpc)
    bm_hourly = bm_hourly.reindex(timestamps)
    if bm_hourly["I_predicted"].isna().any(): raise ValueError("BM1 forecast is missing one or more target hours.")
    bm_predicted = checks.bm1_to_wm4(bm_hourly["I_predicted"].to_numpy(dtype=float), calibration)
    bm_intervals_raw = predictive_intervals(bm_support, bm_weights)
    bm_intervals = {level: tuple(checks.bm1_to_wm4(bound, calibration) for bound in pair)
        for level, pair in bm_intervals_raw.items()}
    wm_intervals = {68: (rows["lower_68"].to_numpy(dtype=float), rows["upper_68"].to_numpy(dtype=float)),
        95: (rows["lower_95"].to_numpy(dtype=float), rows["upper_95"].to_numpy(dtype=float))}
    wm_predicted = rows["I_mpc"].to_numpy(dtype=float)
    metrics = {
        "BM1": forecast_metrics(actual, bm_predicted, bm_intervals),
        "WM4_MPC": forecast_metrics(actual, wm_predicted, wm_intervals),
    }
    hourly = pd.DataFrame({"I_actual": actual, "BM1": bm_predicted, "WM4_MPC": wm_predicted,
        "BM1_error": bm_predicted - actual, "WM4_error": wm_predicted - actual}, index=timestamps)
    for label, bounds in (("BM1", bm_intervals), ("WM4", wm_intervals)):
        for level, (lo, hi) in bounds.items():
            hourly[f"{label}_lower_{level}"], hourly[f"{label}_upper_{level}"] = lo, hi
            hourly[f"{label}_within_{level}"] = (actual >= lo - 1e-12) & (actual <= hi + 1e-12)

    return {"day": day, "hourly": hourly, "metrics": metrics, "sources": {"BM1": bm_source, "WM4_MPC": wm_source},
        "bm1_sigma_x": float(bm_sigma), "wm4_forecasts": rows, "wm4_fit_rows": mpc,
        "normalization": {"comparison": "WM4 normalized I (Jan-Oct scale)", "bm1_training_scale": calibration},
        "interpretation": "WM4 MPC: hourly updated from observed I at t-1, 1h ahead; BM1: 24h day-ahead without target demand."}


def save_comparison(result, output_root=OUTPUT, *, save_plot=True):
    directory = checks.save_directory(output_root, result["day"])
    directory.mkdir(parents=True, exist_ok=True)
    hourly, metrics = result["hourly"], result["metrics"]
    hourly.to_csv(directory / "predictions_hourly.csv", index_label="time")
    pd.DataFrame.from_dict(metrics, orient="index").rename_axis("model").to_csv(directory / "metrics.csv")
    result["wm4_forecasts"].to_csv(directory / "wm4_mpc_one_hour.csv", index=False)
    serialized = {key: val for key, val in result.items() if key in ("sources", "bm1_sigma_x", "normalization", "interpretation", "settings")}
    serialized.update({"day": str(result["day"].date()), "metrics": metrics})
    (directory / "comparison.json").write_text(json.dumps(serialized, indent=2, default=str), encoding="utf-8")
    if save_plot:
        fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [2.6, 1.0]})
        top, bottom = axes
        top.plot(hourly.index, hourly["I_actual"], label="Observed demand", color="black", linewidth=2.1, marker="o", markersize=3)
        for label, color, name in (("BM1", "#3477b8", "BM1 day-ahead"), ("WM4", "#d46835", "WM4 MPC 1h-ahead")):
            forecast = hourly[label if label == "BM1" else "WM4_MPC"]
            top.fill_between(hourly.index, hourly[f"{label}_lower_95"].to_numpy(), hourly[f"{label}_upper_95"].to_numpy(), color=color, alpha=0.1)
            top.fill_between(hourly.index, hourly[f"{label}_lower_68"].to_numpy(), hourly[f"{label}_upper_68"].to_numpy(), color=color, alpha=0.22)
            top.plot(hourly.index, forecast, color=color, linewidth=1.8, label=name)
        top.set_ylabel("Normalized aggregate demand (WM4 I scale)")
        top.set_title(f"BM1 vs WM4 MPC predictions | {result['day']:%Y-%m-%d}\nShading: central 95% and 68% predictive intervals")
        top.legend(loc="upper right"); top.grid(True, alpha=0.16)
        bottom.plot(hourly.index, hourly["BM1_error"], color="#3477b8", label="BM1 error")
        bottom.plot(hourly.index, hourly["WM4_error"], color="#d46835", label="WM4 MPC error")
        bottom.axhline(0.0, color="black", linewidth=0.8)
        bottom.set_ylabel("Prediction − actual"); bottom.set_xlabel("Target-day hour")
        bottom.grid(True, alpha=0.2); bottom.legend(loc="upper right")
        fig.tight_layout(); fig.savefig(directory / "comparison.png", dpi=190, bbox_inches="tight"); plt.close(fig)
    return directory


def print_comparison(result):
    print(f"\nTarget day: {result['day']:%Y-%m-%d}; n=24 hourly forecasts")
    print(result["interpretation"])
    print("Intervals: BM1 central equal-tail 68%/95%; WM4 Gaussian ±1σ (68.27%) / ±1.96σ (95.0%).")
    sigma = result["wm4_forecasts"].get("diffusion_sigma")
    if sigma is not None: print(f"WM4 fitted diffusion sigma: median {float(np.median(np.asarray(sigma, dtype=float))):.6f} across 24 hourly origins.")
    if result["bm1_sigma_x"] == 0.0:
        print("BM1: sigma_x=0; coverage reflects fitted initial-state uncertainty, not process diffusion.")
    else: print(f"BM1: supplied sigma_x={result['bm1_sigma_x']:.6g} (not estimated in this comparison).")
    print(f"{'model':<12}{'MSE':>12}{'RMSE':>12}{'MAE':>12}{'BIAS':>12}{'68% cover':>12}{'95% cover':>12}")
    for name, metrics in result["metrics"].items():
        print(f"{name:<12}{metrics['mse']:>12.7f}{metrics['rmse']:>12.7f}{metrics['mae']:>12.7f}{metrics['bias']:>+12.7f}{metrics['coverage_68_pct']:>11.2f}%{metrics['coverage_95_pct']:>11.2f}%")
        print(f"  {name}: 68% covered {metrics['coverage_68_count']}/24 (gap {metrics['coverage_68_gap_pp']:+.2f}pp), "
              f"95% covered {metrics['coverage_95_count']}/24 (gap {metrics['coverage_95_gap_pp']:+.2f}pp); "
              f"mean widths {metrics['width_68_mean']:.5f}/{metrics['width_95_mean']:.5f}")


def main():
    parser = ArgumentParser(description="Run BM1 day-ahead and WM4 hourly one-step MPC forecasts on the same normalized demand scale.")
    parser.add_argument("--day", default="2022-08-22")
    parser.add_argument("--tag", default="MED")
    parser.add_argument("--X0", type=float, default=0.5)
    parser.add_argument("--sigma-x", type=float, default=0.0, help="BM1 process diffusion amplitude (fixed, not fitted).")
    parser.add_argument("--steps-per-hour", type=int, default=2, help="WM4 MPC integration resolution (>=2).")
    parser.add_argument("--max-state-iter", type=int, default=12, help="WM4 state fitting iterations per origin.")
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--bm1-max-iterations", type=int, default=6)
    parser.add_argument("--bm1-max-nfev", type=int, default=150)
    parser.add_argument("--bm1-state-maxiter", type=int, default=100)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--x-steps", type=int, default=101)
    parser.add_argument("--fp-max-step", type=float, default=0.25)
    parser.add_argument("--wm4-forecasts", type=Path, help="Optional saved mpc_forecasts.csv containing the required 24 lead-1 observations.")
    parser.add_argument("--bm1-saved-root", type=Path, help="Optional saved BM1 prediction root to bypass parameter refitting.")
    parser.add_argument("--save-bm1", action="store_true", help="Also store newly fitted BM1 hourly.csv and fit.json.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = compare_day(args.day, tag=args.tag, X0=args.X0, sigma_x=args.sigma_x,
        steps_per_hour=args.steps_per_hour, max_state_iter=args.max_state_iter, state_tol=args.state_tol,
        bm1_max_iterations=args.bm1_max_iterations, bm1_max_nfev=args.bm1_max_nfev,
        bm1_state_maxiter=args.bm1_state_maxiter, substeps=args.substeps, x_steps=args.x_steps,
        fp_max_step=args.fp_max_step, wm4_forecasts=args.wm4_forecasts,
        bm1_saved_root=args.bm1_saved_root, save_bm1=args.save_bm1)
    print_comparison(result)
    print("Saved:", save_comparison(result, args.output_root))
    return result


if __name__ == "__main__": main()
