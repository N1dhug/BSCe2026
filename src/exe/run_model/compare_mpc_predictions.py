"""Compare BM1 and WM4 rolling MPC forecasts for exact leads 1 through 8 hours.

At each hourly origin, both models refit their parameters on five corresponding
24-hour historical windows. WM4 uses its existing Gaussian uncertainty from
estimated process variance; BM1 estimates sigma_x² from historical residuals,
updates the X0 distribution using observed demand at the origin, and evolves
its distribution using BM1's finite-volume Fokker-Planck solver.

Every comparison row uses the SAME origin, target time, observed I, and
normalized WM4 demand scale. Per-lead and cumulative scores pool all
matched windows. 24 origins per day, predicting lead hours 1..8.
"""

from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.bm_1.predict_mpc import predict_origin as bm1_predict_origin
from src.exe.run_model.wm_4.fit_daily import aligned_wm4_data, fit_expected_output
from src.exe.run_model.wm_4.predict_all_days_mpc import _mpc_origin
from src.exe.run_model.wm_4.predict_all_days_optimized import START_DAY, END_DAY
from src.tools.baseline import statistical_output
from src.tools.forecast_metrics import forecast_metrics
from src.validation.exe.run_model.compare_model_predictions import model_comparison_validation as normalization_checks
from src.validation.exe.run_model.compare_mpc_predictions import compare_mpc_validation as checks


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "files" / "output" / "compare_model_predictions" / "mpc_eight_hour"
LABELS = (("BM1", "I_BM1"), ("WM4", "I_WM4"))


def matched_origin(frame, normalized, origin, bm_output, wm_output, *, horizon=8,
    steps_per_hour=2, max_state_iter=12, state_tol=1e-8, x_steps=101,
    fp_max_step=0.25, prior_bandwidth=0.025, X0=0.5, C=None, Xi=None,
    max_iterations=3, max_nfev=120, state_maxiter=60, substeps=2, cached_wm4=None, min_capacity=0.05):
    """Construct matching BM1 and WM4 forecasts at exactly the same origin."""
    bm, extra = bm1_predict_origin(normalized, origin, bm_output, X0=X0,
        horizon=horizon, x_steps=x_steps, fp_max_step=fp_max_step,
        prior_bandwidth=prior_bandwidth, max_iterations=max_iterations,
        max_nfev=max_nfev, state_maxiter=state_maxiter, substeps=substeps, C=C, Xi=Xi,
        min_capacity=min_capacity)
    if cached_wm4 is None:
        wm, state = _mpc_origin(frame, origin, 5, steps_per_hour, wm_output, horizon,
            max_state_iter=max_state_iter, state_tol=state_tol)
    else:
        try: wm = cached_wm4.loc[[(origin, lead) for lead in range(1, horizon + 1)]].to_dict(orient="records")
        except KeyError as exc: raise ValueError(f"WM4 cached forecasts lack a complete 8-hour horizon at {origin}.") from exc
    bm, wm = checks.complete_pair(bm, wm, origin, horizon)
    return bm, wm, extra


def normalized_rows(bm, wm, calibration, raw_frame):
    """Transform BM1 outputs/quantiles to WM4 scale and compare on one target grid."""
    out = []
    for (_, b), (_, w) in zip(bm.iterrows(), wm.iterrows()):
        time = pd.Timestamp(b["target_time"])
        actual = checks.observed(raw_frame["I"].reindex([time]).iloc[0], time)
        converted = normalization_checks.bm1_to_wm4(
            np.array([b["I_BM1"], b["lower_68"], b["upper_68"], b["lower_95"], b["upper_95"]]), calibration)
        out.append({
            "origin": pd.Timestamp(b["origin"]), "origin_day": pd.Timestamp(b["origin"]).normalize(),
            "origin_hour": int(b["origin_hour"]), "target_time": time, "lead_hours": int(b["lead_hours"]),
            "I_observed": actual, "I_BM1": float(converted[0]), "I_WM4": float(w["I_mpc"]),
            "BM1_lower_68": float(converted[1]), "BM1_upper_68": float(converted[2]),
            "BM1_lower_95": float(converted[3]), "BM1_upper_95": float(converted[4]),
            "WM4_lower_68": float(w["lower_68"]), "WM4_upper_68": float(w["upper_68"]),
            "WM4_lower_95": float(w["lower_95"]), "WM4_upper_95": float(w["upper_95"]),
            "BM1_sigma_x": float(b["sigma_x"]), "WM4_diffusion_sigma": float(w.get("diffusion_sigma", np.nan)),
            "BM1_history_rmse": float(b["historical_fit_rmse"]),
            "BM1_observation_sd": float(b["observation_sd"]),
            "BM1_post_state_mean": float(b["posterior_state_mean"]),
            "BM1_post_state_std": float(b["posterior_state_std"]),
            "BM1_fit_converged": bool(b["fit_converged"]),
            "BM1_fit_stop_reason": str(b["fit_stop_reason"]),
            "BM1_fit_message": str(b["fit_message"]),
            "BM1_theta_nfev": int(b["theta_nfev"]),
        })
    return out


def metrics(frame, model):
    prefix, mean_col = next((key, col) for key, col in LABELS if key == model)
    bounds = {p: (frame[f"{prefix}_lower_{p}"].to_numpy(dtype=float),
        frame[f"{prefix}_upper_{p}"].to_numpy(dtype=float)) for p in (68, 95)}
    return forecast_metrics(frame["I_observed"].to_numpy(dtype=float),
        frame[mean_col].to_numpy(dtype=float), bounds)


def grouped_scores(frame, group=None, group_name="horizon"):
    """Compute RMSE from all matched hourly squared errors (never mean daily RMSE)."""
    parts = [("all", frame)] if group is None else list(frame.groupby(group, sort=True))
    result = []
    for value, data in parts:
        for model, _ in LABELS:
            result.append({group_name: value, "model": model,
                "origins": int(data["origin"].nunique()),
                "days": int(pd.to_datetime(data["origin_day"]).nunique()),
                **metrics(data, model)})
    return pd.DataFrame(result)


def plot_leads(exact, cumulative, output):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex="col")
    series = (("BM1", "#2376ac"), ("WM4", "#ca593b"))
    for col, (summary, xkey, title) in enumerate(((exact, "lead_hours", "Exact prediction lead"),
        (cumulative, "horizon_hours", "Cumulative window length"))):
        for model, color in series:
            group = summary.loc[summary["model"] == model].sort_values(xkey)
            leads = group[xkey].to_numpy(dtype=int)
            axes[0, col].plot(leads, group["rmse"], marker="o", color=color, label=model)
            axes[1, col].plot(leads, group["coverage_68_pct"], marker="o", color=color, label=model)
            axes[2, col].plot(leads, group["coverage_95_pct"], marker="o", color=color, label=model)
        axes[0, col].set_title(title); axes[2, col].set_xlabel("Prediction hours")
        for row in range(3):
            axes[row, col].grid(True, alpha=0.2)
            axes[row, col].set_xticks(range(1, 9))
        axes[1, col].axhline(68.0, linestyle="--", linewidth=0.9, color="0.25")
        axes[2, col].axhline(95.0, linestyle="--", linewidth=0.9, color="0.25")
    axes[0, 0].set_ylabel("RMSE")
    axes[1, 0].set_ylabel("68% empirical coverage (%)")
    axes[2, 0].set_ylabel("95% empirical coverage (%)")
    axes[0, 0].legend(); axes[0, 1].legend()
    for col in range(2):
        axes[1, col].set_ylim(-3, 103); axes[2, col].set_ylim(-3, 103)
    fig.suptitle("BM1 vs WM4 | 24-hour shifted historical fitting, 8-hour receding-horizon forecasts")
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def save_results(rows, skipped, start, end, output_root):
    directory = Path(output_root) / f"{start:%Y-%m-%d}_to_{end:%Y-%m-%d}"
    directory.mkdir(parents=True, exist_ok=True)
    rows = rows.sort_values(["origin", "lead_hours"]).reset_index(drop=True)
    exact = grouped_scores(rows, "lead_hours", "lead_hours")
    windows = []
    for horizon in range(1, int(rows["lead_hours"].max()) + 1):
        part = grouped_scores(rows[rows["lead_hours"] <= horizon])
        part = part.rename(columns={"horizon": "horizon_hours"})
        part["horizon_hours"] = horizon
        windows.append(part)
    cumulative = pd.concat(windows, ignore_index=True)
    by_origin_hour = grouped_scores(rows, "origin_hour", "origin_hour")
    by_day_lead = grouped_scores(rows, [rows["origin_day"], rows["lead_hours"]], "day_lead")
    by_day_lead[["origin_day", "lead_hours"]] = pd.DataFrame(by_day_lead.pop("day_lead").tolist(), index=by_day_lead.index)
    overall = grouped_scores(rows)
    rows.to_csv(directory / "matched_forecasts.csv", index=False)
    exact.to_csv(directory / "rmse_coverage_by_lead.csv", index=False)
    cumulative.to_csv(directory / "rmse_coverage_by_horizon.csv", index=False)
    by_origin_hour.to_csv(directory / "rmse_coverage_by_origin_hour.csv", index=False)
    by_day_lead.to_csv(directory / "daily_lead_metrics.csv", index=False)
    overall.to_csv(directory / "aggregate_metrics.csv", index=False)
    pd.DataFrame(skipped, columns=["origin", "reason"]).to_csv(directory / "skipped_origins.csv", index=False)
    sigma = rows.groupby("origin", as_index=False).agg(
        BM1_sigma_x=("BM1_sigma_x", "first"), WM4_diffusion_sigma=("WM4_diffusion_sigma", "first"),
        BM1_observation_sd=("BM1_observation_sd", "first"),
        BM1_fit_converged=("BM1_fit_converged", "first"),
        BM1_fit_stop_reason=("BM1_fit_stop_reason", "first"),
        BM1_theta_nfev=("BM1_theta_nfev", "first"))
    sigma.to_csv(directory / "historical_diffusion_estimates.csv", index=False)
    plot_leads(exact, cumulative, directory / "lead_time_rmse_coverage.png")
    summary = {
        "start": str(start.date()), "end": str(end.date()), "matched_origins": int(rows["origin"].nunique()),
        "matched_forecasts": int(len(rows)), "skipped_origins": int(len(skipped)),
        "BM1_nonconverged_origins": int((~sigma["BM1_fit_converged"].astype(bool)).sum()),
        "information": "Both refit five prior 24h aligned weekly windows at each hour. WM4 updates R with observed I(origin); BM1 updates a Bayesian X0 distribution with observed I(origin).",
        "uncertainty": "BM1 sigma_x estimated from historical linear-noise demand variance regression; BM1 probability intervals from FP. WM4 ±1σ and ±1.96σ Gaussian intervals.",
        "aggregation": "All exact and cumulative coverage and RMSE are calculated over paired prediction observations.",
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return directory, exact, cumulative, overall


def print_summary(exact, cumulative, overall, skipped, directory):
    print(f"\nMatched origins={int(overall['origins'].iloc[0])}; paired hours={int(overall['n'].iloc[0])}; skipped={len(skipped)}")
    print("Both models use the same hourly origin and 1-8h horizon with five 24h aligned history windows.")
    print(f"{'Lead':>4} {'Model':<6} {'RMSE':>10} {'MSE':>10} {'MAE':>10} {'BIAS':>10} {'68%cov':>9} {'95%cov':>9} {'N':>7}")
    for _, row in exact.iterrows():
        print(f"{int(row['lead_hours']):4d} {row['model']:<6} {row['rmse']:10.6f} {row['mse']:10.6f} {row['mae']:10.6f} {row['bias']:+10.6f} {row['coverage_68_pct']:8.1f}% {row['coverage_95_pct']:8.1f}% {int(row['n']):7d}")
    print("\nCumulative 1..H forecast window scores:")
    for _, row in cumulative.iterrows():
        print(f"H={int(row['horizon_hours'])} {row['model']:<4} RMSE={row['rmse']:.6f} coverage68={row['coverage_68_pct']:.1f}% coverage95={row['coverage_95_pct']:.1f}%")
    stats = pd.read_csv(directory / "historical_diffusion_estimates.csv")
    nonconverged = int((~stats["BM1_fit_converged"].astype(bool)).sum())
    print(f"BM1 pooled fits stopped before convergence: {nonconverged}/{len(stats)} origins.")
    if nonconverged:
        print("WARNING: Predictions from capped fits are retained and marked BM1_fit_converged=False in saved forecasts.")
        print(stats.loc[~stats["BM1_fit_converged"].astype(bool), "BM1_fit_stop_reason"].value_counts().to_string())
    print(f"Saved: {directory}")


def run_all_days(start=START_DAY, end=END_DAY, *, frame=None, tag="MED", horizon=8,
    steps_per_hour=2, X0=0.5, C=None, Xi=None, x_steps=101,
    max_iterations=3, max_nfev=120, state_maxiter=60, substeps=2,
    max_state_iter=12, state_tol=1e-8, fp_max_step=0.25, prior_bandwidth=0.025,
    wm4_forecasts=None, output_root=OUTPUT, resume=True, fail_fast=False, min_capacity=0.05):
    start, end = checks.date_range(start, end, horizon, steps_per_hour, x_steps, fp_max_step, prior_bandwidth, substeps)
    if not np.isfinite(min_capacity) or not 0.001 <= min_capacity < 100.0:
        raise ValueError("min_capacity must be finite and in [0.001, 100).")
    if frame is None: frame, _ = aligned_wm4_data(tag)
    frame = normalization_checks.common_frame(frame)
    cached_wm4 = checks.wm4_saved(wm4_forecasts)
    options = {"tag": tag, "horizon": horizon, "steps_per_hour": steps_per_hour, "X0": X0,
        "C": C, "Xi": Xi, "x_steps": x_steps, "max_iterations": max_iterations,
        "max_nfev": max_nfev, "state_maxiter": state_maxiter, "substeps": substeps,
        "max_state_iter": max_state_iter, "state_tol": state_tol,
        "fp_max_step": fp_max_step, "prior_bandwidth": prior_bandwidth,
        "fit_status_schema": 3, "min_capacity": float(min_capacity),
        "wm4_forecasts": str(Path(wm4_forecasts).resolve()) if isinstance(wm4_forecasts, (str, Path))
            else ("supplied DataFrame" if cached_wm4 is not None else None)}
    all_rows, skipped = [], []
    days = pd.date_range(start, end, freq="D")
    for number, day in enumerate(days, start=1):
        folder = Path(output_root) / f"{day:%Y-%m-%d}"
        loaded = checks.saved_day(folder, options) if resume else None
        if loaded is not None:
            all_rows.extend(loaded.to_dict(orient="records"))
            missing_path = folder / "skipped_origins.csv"
            if missing_path.exists():
                missing = pd.read_csv(missing_path)
                skipped.extend([(pd.Timestamp(row["origin"]), str(row["reason"])) for _, row in missing.iterrows()])
            print(f"[{number}/{len(days)}] {day:%Y-%m-%d}: restored {len(loaded)} paired forecasts", flush=True)
            continue
        day_rows, day_skipped = [], []
        try:
            normalized, calibration = normalization_checks.bm1_scale(frame, day)
            bm_baseline = statistical_output().fit(normalized, day)
            wm_baseline = None if cached_wm4 is not None else fit_expected_output(frame, day)[0]
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.extend([(day + pd.Timedelta(hours=h), str(exc)) for h in range(24)])
            print(f"[{number}/{len(days)}] {day:%Y-%m-%d}: skipped all origins ({exc})", flush=True)
            continue
        for hour in range(24):
            origin = day + pd.Timedelta(hours=hour)
            try:
                bm, wm, extra = matched_origin(frame, normalized, origin, bm_baseline, wm_baseline,
                    horizon=horizon, steps_per_hour=steps_per_hour,
                    max_state_iter=max_state_iter, state_tol=state_tol,
                    x_steps=x_steps, fp_max_step=fp_max_step, prior_bandwidth=prior_bandwidth,
                    X0=X0, C=C, Xi=Xi, max_iterations=max_iterations,
                    max_nfev=max_nfev, state_maxiter=state_maxiter,
                    substeps=substeps, cached_wm4=cached_wm4, min_capacity=min_capacity)
                day_rows.extend(normalized_rows(bm, wm, calibration, frame))
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, KeyError) as exc:
                if fail_fast: raise
                day_skipped.append((origin, str(exc)))
        skipped.extend(day_skipped)
        if day_rows:
            folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(day_rows).to_csv(folder / "matched_forecasts.csv", index=False)
            pd.DataFrame(day_skipped, columns=["origin", "reason"]).to_csv(folder / "skipped_origins.csv", index=False)
            (folder / "settings.json").write_text(json.dumps(options, indent=2), encoding="utf-8")
            all_rows.extend(day_rows)
        print(f"[{number}/{len(days)}] {day:%Y-%m-%d} matched={len(day_rows)//horizon}/24 "
            f"skipped={len(day_skipped)}", flush=True)

    if not all_rows: raise RuntimeError("No paired BM1/WM4 MPC forecasts were produced.")
    frame_out = pd.DataFrame(all_rows)
    directory, exact, cumulative, overall = save_results(frame_out, skipped, start, end, output_root)
    print_summary(exact, cumulative, overall, skipped, directory)
    return {"matched": frame_out, "exact": exact, "cumulative": cumulative,
        "aggregate": overall, "skipped": skipped, "output": directory}


def main():
    parser = ArgumentParser(description="Compare BM1 and WM4 MPC on identical hourly origins, each fitted over five prior 24h windows and evaluated 1-8h ahead.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--tag", default="MED")
    parser.add_argument("--horizon", type=int, default=8, help="Exact 1..8h prediction leads.")
    parser.add_argument("--steps-per-hour", type=int, default=2, help="WM4 integration resolution.")
    parser.add_argument("--X0", type=float, default=0.5)
    parser.add_argument("--C", type=float)
    parser.add_argument("--Xi", type=float)
    parser.add_argument("--x-steps", type=int, default=101)
    parser.add_argument("--fp-max-step", type=float, default=0.25)
    parser.add_argument("--prior-bandwidth", type=float, default=0.025)
    parser.add_argument("--min-capacity", type=float, default=0.05,
        help="BM1 C lower bound to limit nonlinear RK4 work; default 0.05.")
    parser.add_argument("--bm1-max-iterations", type=int, default=3)
    parser.add_argument("--bm1-max-nfev", type=int, default=120)
    parser.add_argument("--bm1-state-maxiter", type=int, default=60)
    parser.add_argument("--bm1-substeps", type=int, default=2)
    parser.add_argument("--wm4-max-state-iter", type=int, default=12)
    parser.add_argument("--wm4-state-tol", type=float, default=1e-8)
    parser.add_argument("--wm4-forecasts", type=Path, help="Existing WM4 mpc_forecasts.csv with all 8 leads; avoids WM4 refitting.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--no-resume", action="store_true", help="Refit all origins even if matching day checkpoints exist.")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    return run_all_days(args.start_day, args.end_day, tag=args.tag, horizon=args.horizon,
        steps_per_hour=args.steps_per_hour, X0=args.X0, C=args.C, Xi=args.Xi,
        x_steps=args.x_steps, max_iterations=args.bm1_max_iterations,
        max_nfev=args.bm1_max_nfev, state_maxiter=args.bm1_state_maxiter,
        substeps=args.bm1_substeps, max_state_iter=args.wm4_max_state_iter,
        state_tol=args.wm4_state_tol, fp_max_step=args.fp_max_step,
        prior_bandwidth=args.prior_bandwidth, wm4_forecasts=args.wm4_forecasts,
        min_capacity=args.min_capacity,
        output_root=args.output_root, resume=not args.no_resume, fail_fast=args.fail_fast)


if __name__ == "__main__": main()
