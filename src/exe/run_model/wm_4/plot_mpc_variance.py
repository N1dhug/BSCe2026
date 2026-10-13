"""Compare WM4 predicted stochastic variance with empirical MPC forecast-error variance.

The observed quantity is the sample variance (ddof=1) of errors
e_{t,h} = I_{t+h} - I_hat_{t+h|t} across MPC origins at fixed lead h.
It is NOT the variance of demand levels and NOT mean squared error.
Predicted variance is the arithmetic mean of each origin's predicted_variance.
No WM4 model, baseline, or stochastic parameters are refitted here.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_mpc_intervals import FORECASTS, load_forecasts


OUTPUT = FORECASTS.parent / "variance_calibration"


def select_origins(frame, fully_mpc=False, history_windows=5):
    """Optionally exclude origins still using oracle-bootstrap calibration."""
    if not fully_mpc: return frame.copy()
    if int(history_windows) < 1: raise ValueError("history_windows must be positive.")
    fields = ("calibration_mpc_count", "calibration_oracle_count",
        "F_calibration_mpc_count", "F_calibration_oracle_count")
    if missing := set(fields) - set(frame.columns):
        raise ValueError(f"Fully-MPC filtering requires saved calibration columns {sorted(missing)}.")
    ok = (
        (frame["calibration_mpc_count"] >= history_windows)
        & (frame["calibration_oracle_count"] == 0)
        & (frame["F_calibration_mpc_count"] >= history_windows)
        & (frame["F_calibration_oracle_count"] == 0)
    )
    selected = frame.loc[ok].copy()
    if selected.empty:
        raise ValueError("No origins have fully MPC-calibrated initial states and Wiener diffusion. "
            "Evaluate later dates or omit --fully-mpc.")
    # Rows for each origin must remain complete after filtering.
    if selected.groupby("origin")["lead_hours"].nunique().nunique() != 1:
        raise ValueError("Calibration metadata must be constant for all leads within an origin.")
    return selected


def variance_scores(frame):
    """Observed variance uses actual forward errors; do not substitute observed MSE."""
    error = np.asarray(frame["error"], dtype=float)
    predicted = np.asarray(frame["predicted_variance"], dtype=float)
    if error.size < 2: raise ValueError("At least two forecast errors are required to estimate empirical variance.")
    if not np.all(np.isfinite(error)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Forecast errors and predicted variances must be finite.")
    if np.any(predicted < 0.0): raise ValueError("Predicted variances must be nonnegative.")
    empirical = float(np.var(error, ddof=1))
    model = float(np.mean(predicted))
    bias = float(np.mean(error))
    mse = float(np.mean(error ** 2))
    return {
        "n": int(len(frame)),
        "origins": int(frame["origin"].nunique()),
        "predicted_variance": model,
        "empirical_error_variance": empirical,
        "empirical_error_variance_population": float(np.var(error, ddof=0)),
        "observed_mse": mse,
        "forecast_bias": bias,
        "bias_squared": bias ** 2,
        "variance_difference": model - empirical,
        "variance_ratio_predicted_to_empirical": model / empirical if empirical > 0 else np.nan,
        "predicted_rms_std": float(np.sqrt(model)),
        "empirical_error_std": float(np.sqrt(empirical)),
    }


def summarize_variance(frame, horizon):
    """Exact leads and pooled 1..H windows, with variance not MSE as ground truth."""
    lead, cumulative = [], []
    for h in range(1, int(horizon) + 1):
        exact = frame[frame["lead_hours"] == h]
        pooled = frame[frame["lead_hours"] <= h]
        if len(exact) < 2: raise ValueError(f"Need at least two complete MPC origins at lead {h}.")
        lead.append({"lead_hours": h, **variance_scores(exact)})
        cumulative.append({"horizon_hours": h, **variance_scores(pooled)})
    return pd.DataFrame(lead), pd.DataFrame(cumulative)


def plot_variance(lead, cumulative, output, fully_mpc=False):
    """Show predicted and measured forecast-error variances on identical scales."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True)
    for ax, frame, x_col, title in (
        (axes[0], lead, "lead_hours", "Exact forecast lead"),
        (axes[1], cumulative, "horizon_hours", "Cumulative window (pooled leads 1..H)"),
    ):
        h = frame[x_col].to_numpy(dtype=int)
        ax.plot(h, frame["predicted_variance"], "o-", linewidth=2,
            label="Predicted variance (mean across origins)")
        ax.plot(h, frame["empirical_error_variance"], "s-", linewidth=2,
            label="Actual forecast-error variance (sample, ddof=1)")
        ax.plot(h, frame["observed_mse"], "^:", linewidth=1.3, alpha=0.65,
            label="Observed MSE (variance + squared bias, adjusted for ddof)")
        ax.set_title(title)
        ax.set_ylabel("Variance (normalized demand squared)")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("Forecast horizon H (hours)")
    axes[-1].set_xticks(lead["lead_hours"].to_numpy(dtype=int))
    axes[0].set_ylim(bottom=0)
    caption = " | fully MPC-calibrated origins only" if fully_mpc else " | all complete origins"
    fig.suptitle("WM4 MPC | predicted vs actual forecast-error variance" + caption)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return output


def analyze(forecasts=FORECASTS, output=OUTPUT, horizon=8, fully_mpc=False, history_windows=5,
    *, expected_f0_variance_mode="mpc_reconstruction",
    expected_r0_variance_mode="oracle_reconstruction",
    expected_sigma_fit_mode="mpc_replay"):
    frame, dropped, mode = load_forecasts(forecasts, horizon,
        expected_f0_variance_mode=expected_f0_variance_mode,
        expected_r0_variance_mode=expected_r0_variance_mode,
        expected_sigma_fit_mode=expected_sigma_fit_mode)
    frame = select_origins(frame, fully_mpc=fully_mpc, history_windows=history_windows)
    lead, cumulative = summarize_variance(frame, horizon)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lead.to_csv(output / "variance_by_exact_lead.csv", index=False)
    cumulative.to_csv(output / "variance_cumulative_1_to_H.csv", index=False)
    figure = plot_variance(lead, cumulative,
        output / "predicted_vs_empirical_variance.png", fully_mpc=fully_mpc)
    print(f"WM4 sigma fit={mode}; {frame['origin'].nunique()} complete MPC origins; "
        f"{dropped} incomplete origins discarded.")
    print(lead[["lead_hours", "n", "predicted_variance", "empirical_error_variance",
        "observed_mse", "forecast_bias", "variance_ratio_predicted_to_empirical"]].to_string(
            index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved {figure} and exact/cumulative variance CSVs in {output}")
    return {"lead": lead, "cumulative": cumulative, "figure": figure, "output": output}


def main():
    parser = ArgumentParser(description="Plot WM4 predicted variance against empirical forward MPC error variance (not MSE).")
    parser.add_argument("--forecasts", type=Path, default=FORECASTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--horizon", type=int, default=8, help="Hours (1..8).")
    parser.add_argument("--fully-mpc", action="store_true",
        help="Only use origins with no oracle bootstrap in either state or diffusion calibration.")
    parser.add_argument("--history-windows", type=int, default=5,
        help="Required number of historical MPC windows for --fully-mpc.")
    parser.add_argument("--expected-f0-variance-mode", choices=("mpc_reconstruction", "oracle_spread", "any"),
        default="mpc_reconstruction")
    parser.add_argument("--expected-r0-variance-mode", choices=("oracle_reconstruction", "baseline_residual", "zero", "any"),
        default="oracle_reconstruction")
    parser.add_argument("--expected-sigma-fit-mode", choices=("mpc_replay", "mpc_replay_24d_24h", "mpc_replay_24d_6h", "historical_p0", "zero_p0", "any"),
        default="mpc_replay")
    args = parser.parse_args()
    if args.horizon < 1 or args.horizon > 8: parser.error("horizon must be in [1, 8].")
    return analyze(args.forecasts, args.output, args.horizon, args.fully_mpc, args.history_windows,
        expected_f0_variance_mode=args.expected_f0_variance_mode,
        expected_r0_variance_mode=args.expected_r0_variance_mode,
        expected_sigma_fit_mode=args.expected_sigma_fit_mode)


if __name__ == "__main__": main()
