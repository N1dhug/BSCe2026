"""Re-estimate WM4 Wiener diffusion at EACH hourly MPC origin with NO P0 propagation.

This is a genuine forward-MPC run: refits the 24-hour deterministic WM4
parameters at every origin, reconstructs the initial F/R centre, and refits
the six-hour Wiener coefficient from the previous 24 days of historical
six-hour MPC forecast errors. The covariance J P0 J^T is explicitly ZERO
in both the historical sigma QP and the forward variance prediction.

Unlike compare_mpc_p0_ablation.py, no saved sigma estimates are reused.
Outputs a new mpc_forecasts.csv and independently ranked uncertainty quintiles.
"""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.exe.run_model.wm_4.fit_daily import (
    BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data,
)
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY
from src.exe.run_model.wm_4.predict_all_days_mpc import (
    HORIZONS, ROOT, cumulative_summary, print_summary, run_all_days,
    save_plot, save_stochastic_plot, summary_by_lead, summary_by_origin_hour,
)
from src.exe.run_model.wm_4.plot_mpc_stochastic_quantiles import analyze as plot_quantiles


MODE = "mpc_replay_24d_6h_no_p0"
OUTPUT = ROOT / "files/output/wm_4/prediction/all_days_mpc_sigma_24d_6h_no_p0"


def run(frame, start=START_DAY, end=END_DAY, steps_per_hour=60,
    baseline_weeks=BASELINE_WEEKS, rolling_days=24, output=OUTPUT,
    fail_fast=False, quantiles=5):
    """Generate newly fitted 6h no-P0 MPC distributions and their quintiles."""
    rolling_days = int(rolling_days)
    if rolling_days < 2: raise ValueError("rolling_days must be >= 2.")
    if int(quantiles) < 2: raise ValueError("At least two uncertainty quantiles required.")
    output = Path(output)
    forecasts, states, skipped = run_all_days(frame, start=start, end=end,
        baseline_weeks=baseline_weeks, steps_per_hour=steps_per_hour,
        max_horizon=6, fit_hours=DAY_HOURS, fail_fast=fail_fast,
        sigma_fit_mode=MODE, sigma_rolling_days=rolling_days)
    if not (forecasts["initial_variance"] == 0).all():
        raise RuntimeError("No-P0 run generated nonzero initial variance.")
    if not (forecasts["sigma_fit_mode"] == MODE).all():
        raise RuntimeError("No-P0 run included forecasts from a different sigma mode.")

    lead = summary_by_lead(forecasts)
    cumulative = cumulative_summary(forecasts, tuple(h for h in HORIZONS if h <= 6))
    hourly = summary_by_origin_hour(forecasts, 6)
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "mpc_forecasts.csv"
    forecasts.to_csv(predictions, index=False)
    states.to_csv(output / "mpc_states.csv", index=False)
    lead.to_csv(output / "lead_time_summary.csv", index=False)
    cumulative.to_csv(output / "cumulative_horizon_summary.csv", index=False)
    hourly.to_csv(output / "origin_hour_summary.csv", index=False)
    if skipped:
        pd.DataFrame([{"origin": origin, "reason": reason} for origin, reason in skipped
            ]).to_csv(output / "skipped_origins.csv", index=False)
    save_plot(cumulative, lead, output / "mpc_vs_output_rmse.png")
    save_stochastic_plot(lead, output / "stochastic_variance_calibration.png")
    print_summary(cumulative, lead, skipped)

    # Rank only forecasts with a complete causal rolling calibration.
    calibration = plot_quantiles(predictions, output / "stochastic_quantiles",
        horizon=6, quantiles=int(quantiles), mode=MODE,
        mature_only=True, rolling_days=rolling_days)
    print(f"New six-hour sigma QPs: {forecasts['origin'].nunique()} hourly MPC origins; "
        f"24-hour deterministic fitting, no P0 propagation; output={output}")
    return {"forecasts": forecasts, "states": states, "skipped": skipped,
        "lead": lead, "quintiles": calibration, "output": output}


def main():
    parser = ArgumentParser(description="Run and refit every hourly WM4 six-hour Wiener MPC prediction WITHOUT initial-state covariance propagation; produce new uncertainty quintiles.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--rolling-days", type=int, default=24)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data, _ = aligned_wm4_data("MED")
    return run(data, args.start_day, args.end_day, args.steps_per_hour,
        args.baseline_weeks, args.rolling_days, args.output,
        args.fail_fast, args.quantiles)


if __name__ == "__main__": main()
