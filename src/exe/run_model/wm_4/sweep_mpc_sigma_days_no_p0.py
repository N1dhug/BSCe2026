"""Fit WM4 hourly MPC once, then independently re-estimate no-P0 Wiener q for 7..28 days.

Each MPC origin receives one fresh 24-hour deterministic parameter/baseline fit
and a genuine six-hour forecast. The historical MPC archive contains completed
forward errors and Wiener kernels from THIS run, not an older CSV. Afterward,
for EACH origin and EACH window length 7..28, a new causal six-lead QP is solved
with zero initial-state variance:

    min_{q>=0} sum_{h=1}^6 [sample_var(e_{T-1:T-N,h}) - q * mean(K_{T-1:T-N,h})]^2

The scalar QP uses the same exact closed-form ridge solver as the WM4 rolling
sweep; neither P0 nor its propagated contribution enters calibration or
prediction. We avoid rerunning the expensive deterministic optimizer 22 times.
The original 24-day no-P0 q from run_all_days is not used for the sweep fits.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data
from src.exe.run_model.wm_4.predict_all_days_optimized import START_DAY, END_DAY
from src.exe.run_model.wm_4.predict_all_days_mpc import ROOT, run_all_days
from src.exe.run_model.wm_4.predict_all_days_mpc_no_p0 import MODE
from src.exe.run_model.wm_4.sweep_mpc_sigma_days import HORIZON, save_plots, sweep


OUTPUT = ROOT / "files/output/wm_4/prediction/sigma_days_sweep_no_p0"


def plot_origin_scatter(fits, archive, path):
    """One panel per N; same origins, axis scales, and observed six-hour MSE."""
    days = sorted(fits["history_days"].unique())
    errors = {origin: float(np.mean(values[:, 0] ** 2)) for origin, values in archive.items()}
    values = fits.copy()
    values["observed_mse"] = values["origin"].map(errors)
    values["predicted_mean_variance"] = values[
        [f"variance_h{h}" for h in range(1, HORIZON + 1)]].mean(axis=1)
    high = 1.05 * max(float(values["observed_mse"].max()),
        float(values["predicted_mean_variance"].max()), 1e-8)
    ncols = 4
    fig, axes = plt.subplots(int(np.ceil(len(days) / ncols)), ncols,
        figsize=(16, 3.5 * int(np.ceil(len(days) / ncols))), squeeze=False)
    for ax, day in zip(axes.flat, days):
        sub = values[values["history_days"] == day]
        ax.scatter(sub["predicted_mean_variance"], sub["observed_mse"],
            s=9, alpha=0.45)
        ax.plot([0, high], [0, high], "k--", lw=0.8)
        ax.set(xlim=(0, high), ylim=(0, high), title=f"{day}-day Wiener fit")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(days):]: ax.axis("off")
    fig.supxlabel("Predicted six-hour mean variance (Wiener only)")
    fig.supylabel("Realized six-hour mean squared MPC error")
    fig.suptitle("ARM (Affine Revised Model) | no initial-state covariance | matched origins in every panel")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def run(frame, start=START_DAY, end=END_DAY, *, steps_per_hour=60,
    baseline_weeks=BASELINE_WEEKS, min_days=7, max_days=28, quantiles=5,
    output=OUTPUT, fail_fast=False):
    """Generate fresh MPC paths, refit ALL no-P0 diffusion windows, score quintiles."""
    min_days, max_days, quantiles = int(min_days), int(max_days), int(quantiles)
    if not 2 <= min_days <= max_days: raise ValueError("Require 2 <= min_days <= max_days.")
    if quantiles < 2: raise ValueError("At least two uncertainty quantiles required.")
    output = Path(output)
    archive = {}
    forecasts, states, skipped = run_all_days(frame, start=start, end=end,
        baseline_weeks=baseline_weeks, steps_per_hour=steps_per_hour,
        max_horizon=HORIZON, fit_hours=DAY_HOURS, fail_fast=fail_fast,
        sigma_fit_mode=MODE, sigma_rolling_days=24, mpc_archive=archive)
    if not len(archive):
        raise RuntimeError("No completed six-hour MPC trajectories were archived.")
    if (forecasts["sigma_fit_mode"] != MODE).any():
        raise RuntimeError("Unexpected sigma mode in fresh WM4 MPC forecasts.")
    if not np.allclose(forecasts["initial_variance"], 0.0, atol=1e-12):
        raise RuntimeError("Initial-state covariance propagated into new forecasts.")
    for origin, record in archive.items():
        sample = np.asarray(record, dtype=float)
        if sample.shape != (HORIZON, 3):
            raise RuntimeError(f"Incomplete six-hour MPC archive at {origin}.")
        if not np.allclose(sample[:, 1], 0.0, atol=1e-12):
            raise RuntimeError(f"Initial-state variance entered sigma training at {origin}.")
        if np.any(sample[:, 2] < 0.0) or not np.all(np.isfinite(sample)):
            raise RuntimeError(f"Invalid Wiener kernels / historical errors at {origin}.")

    # Every candidate q is RE-ESTIMATED from this run's historical MPC errors,
    # not read from diffusion_sigma / diffusion_q in forecasts.
    result = sweep(archive, min_days=min_days, max_days=max_days, quantiles=quantiles)
    if not np.allclose(result["matched_fits"][
        [f"variance_h{h}" for h in range(1, HORIZON + 1)]].to_numpy(),
        result["matched_fits"]["diffusion_q"].to_numpy()[:, None] *
        np.stack([archive[stamp][:, 2] for stamp in result["matched_fits"]["origin"]]),
        atol=1e-9, rtol=1e-6):
        raise RuntimeError("No-P0 sweep forecast variances contain a non-Wiener contribution.")

    output.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output / "mpc_forecasts.csv", index=False)
    states.to_csv(output / "mpc_states.csv", index=False)
    if skipped:
        pd.DataFrame([{"origin": origin, "reason": reason} for origin, reason in skipped
            ]).to_csv(output / "skipped_origins.csv", index=False)
    save_plots(result, output)
    for name, table in result.items():
        table.to_csv(output / f"{name}.csv", index=False)
    plot_origin_scatter(result["matched_fits"], archive,
        output / "matched_scatter_all_windows.png")

    report = result["matched_summary"]
    print(f"Fresh hourly WM4 MPC origins: {forecasts['origin'].nunique()}; "
        f"no-P0 six-hour history trajectories: {len(archive)}")
    print(f"Separately re-estimated no-P0 Wiener QPs: {len(result['fits'])}; "
        f"matched origin cohort: {int(report.iloc[0]['n_origins'])}")
    print(report[["history_days", "n_origins", "predicted_to_observed_mse",
        "coverage_68_pct", "coverage_95_pct", "gaussian_nll", "mean_sigma"]].to_string(
        index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"Saved 7–28d fresh no-P0 QP fits, quintiles, exact-lead scores and plots: {output}")
    return {"forecasts": forecasts, "states": states, "skipped": skipped,
        "archive": archive, **result, "output": output}


def main():
    parser = ArgumentParser(description="Fresh WM4 six-hour MPC and independent 7–28d Wiener QPs WITHOUT any initial-state covariance propagation.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--min-days", type=int, default=7)
    parser.add_argument("--max-days", type=int, default=28)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data, _ = aligned_wm4_data("MED")
    return run(data, args.start_day, args.end_day,
        steps_per_hour=args.steps_per_hour, baseline_weeks=args.baseline_weeks,
        min_days=args.min_days, max_days=args.max_days,
        quantiles=args.quantiles, output=args.output, fail_fast=args.fail_fast)


if __name__ == "__main__": main()
