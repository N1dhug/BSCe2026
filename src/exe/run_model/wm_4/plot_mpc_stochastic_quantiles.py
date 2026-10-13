"""Plot stochastic WM4 RMSE and interval coverage by predicted-uncertainty quintile.

This reproduces stochastic_variance_calibration.png at the *conditional*
level: one row per predicted-uncertainty quantile, with exact-lead RMSE vs
predicted RMS standard deviation and observed 1-sigma / 95% coverage.

An origin is ranked by its mean predicted variance across its full forecast horizon.
Quantiles are fixed across leads: the same MPC origins appear in every
hour of a given quintile. This avoids mixing forecast difficulty with lead.

The default 28-day/6h evaluation excludes origins before all 28 previous
days of calibration data were available. All plots use existing predictions,
without refitting the WM4 model.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_mpc_intervals import FORECASTS, load_forecasts


ROLLING_MODES = {"mpc_replay_24d_6h": 6, "mpc_replay_24d_24h": 24,
    "mpc_replay_30d_6h": 6, "mpc_replay_28d_6h": 6}
ROLLING_DAYS = {"mpc_replay_24d_6h": 24, "mpc_replay_24d_24h": 24,
    "mpc_replay_30d_6h": 30, "mpc_replay_28d_6h": 28}
MODES = ("mpc_replay", *ROLLING_MODES)
DEFAULT_FORECASTS = FORECASTS.parent.parent / "all_days_mpc_sigma_28d_6h_fit12" / "mpc_forecasts.csv"
DEFAULT_OUTPUT = DEFAULT_FORECASTS.parent / "stochastic_quantiles"


def select_mpc_origins(frame, horizon=8, *, quantiles=5, mode="mpc_replay_28d_6h",
    mature_only=True, rolling_days=None):
    """Keep complete origins, optionally require full rolling sigma history."""
    horizon, quantiles = int(horizon), int(quantiles)
    if not 1 <= horizon <= 8: raise ValueError("horizon must be in [1, 8].")
    if quantiles < 2: raise ValueError("At least two quantiles are required.")
    if mode not in MODES: raise ValueError(f"Unknown sigma fit mode {mode!r}.")
    rolling_days = ROLLING_DAYS.get(mode, 24) if rolling_days is None else int(rolling_days)
    if rolling_days < 2: raise ValueError("rolling_days must be >= 2.")
    data = frame.copy()
    data["origin"] = pd.to_datetime(data["origin"])
    data["lead_hours"] = pd.to_numeric(data["lead_hours"], errors="raise").astype(int)
    if data.duplicated(["origin", "lead_hours"]).any():
        raise ValueError("Duplicate MPC origin/forecast-lead records.")
    data = data.loc[data["lead_hours"].between(1, horizon)].copy()

    if mature_only and mode in ROLLING_MODES:
        required = {"sigma_rolling_complete", "sigma_rolling_count",
            "sigma_rolling_days", "calibration_horizon"}
        if missing := required - set(data):
            raise ValueError(f"Cannot select mature rolling sigma fits: missing {sorted(missing)}.")
        ready = (data["sigma_rolling_complete"].astype(str).str.lower().eq("true")
            & (pd.to_numeric(data["sigma_rolling_count"]) >= int(rolling_days))
            & (pd.to_numeric(data["sigma_rolling_days"]) == int(rolling_days))
            & (pd.to_numeric(data["calibration_horizon"]) == ROLLING_MODES[mode]))
        data = data.loc[ready].copy()

    complete = data.groupby("origin")["lead_hours"].agg(
        lambda v: np.array_equal(np.sort(v.to_numpy()), np.arange(1, horizon + 1)))
    data = data.loc[data["origin"].isin(complete[complete].index)].copy()
    count = data["origin"].nunique()
    if count < quantiles:
        raise ValueError(f"Only {count} complete MPC origins; {quantiles} quantiles required. "
            "For a rolling fit, use --include-warmup or generate more historical forecasts.")
    required = {"predicted_variance", "I_observed", "I_mpc"}
    if missing := required - set(data):
        raise ValueError(f"MPC forecast CSV lacks columns: {sorted(missing)}.")
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.all(np.isfinite(data[col])):
            raise ValueError(f"Nonfinite forecast data in {col}.")
    if (data["predicted_variance"] <= 0).any():
        raise ValueError("Positive predicted variance required for coverage and NLL.")

    by_origin = data.groupby("origin")["predicted_variance"].mean().sort_index()
    # Rank-then-qcut makes equal-size bins even if multiple forecasts have
    # identical variance. Timestamp order is a deterministic tie-breaker.
    q = pd.qcut(by_origin.rank(method="first"), quantiles, labels=False).astype(int) + 1
    mapping = q.rename("uncertainty_quantile")
    data = data.merge(mapping, left_on="origin", right_index=True, validate="many_to_one")
    origins = mapping.reset_index()
    origins["predicted_rms_std"] = np.sqrt(by_origin.to_numpy(dtype=float))
    origins["n_leads"] = horizon
    return data.sort_values(["uncertainty_quantile", "origin", "lead_hours"]), origins


def summarize_quantiles(frame, horizon=8):
    """Exact-lead conditional error/RMS and Gaussian interval coverage."""
    rows = []
    for (quantile, lead), part in frame.groupby(["uncertainty_quantile", "lead_hours"], sort=True):
        errors = (part["I_observed"] - part["I_mpc"]).to_numpy(dtype=float)
        variance = part["predicted_variance"].to_numpy(dtype=float)
        std = np.sqrt(variance)
        rms = float(np.sqrt(np.mean(errors ** 2)))
        pred = float(np.sqrt(np.mean(variance)))
        rows.append({
            "uncertainty_quantile": int(quantile),
            "lead_hours": int(lead),
            "n_origins": int(part["origin"].nunique()),
            "observed_rmse": rms,
            "predicted_rms_std": pred,
            "observed_mse": rms ** 2,
            "predicted_variance": pred ** 2,
            "predicted_to_observed_mse": pred ** 2 / rms ** 2 if rms > 0 else np.nan,
            "error_bias": float(np.mean(errors)),
            "coverage_68_pct": 100 * float(np.mean(np.abs(errors) <= std)),
            "coverage_95_pct": 100 * float(np.mean(np.abs(errors) <= 1.96 * std)),
            "mean_gaussian_nll": float(np.mean(
                0.5 * (np.log(2 * np.pi * variance) + errors ** 2 / variance))),
        })
    result = pd.DataFrame(rows)
    if result.empty: raise ValueError("No matching forecasts in uncertainty quantiles.")
    for quantile, part in result.groupby("uncertainty_quantile"):
        if not np.array_equal(part["lead_hours"].to_numpy(), np.arange(1, int(horizon) + 1)):
            raise ValueError(f"Incomplete lead coverage in quantile {quantile}.")
        if part["n_origins"].nunique() != 1:
            raise ValueError("Quantile membership must stay fixed for all forecast leads.")
    return result


def plot_quantiles(summary, output, *, mode="mpc_replay_28d_6h", mature_only=True):
    """Five rows (Q1–Q5) x two columns (RMSE/std, interval coverage)."""
    groups = sorted(summary["uncertainty_quantile"].unique())
    rows = len(groups)
    fig, axes = plt.subplots(rows, 2, figsize=(16, max(4 * rows, 7)),
        sharex="col", sharey="col", squeeze=False)
    for row, quantile in enumerate(groups):
        part = summary.loc[summary["uncertainty_quantile"] == quantile].sort_values("lead_hours")
        h = part["lead_hours"].to_numpy(dtype=int)
        ax = axes[row, 0]
        ax.plot(h, part["observed_rmse"], "o-", label="Observed MPC RMSE", color="tab:blue")
        ax.plot(h, part["predicted_rms_std"], "o-", label="Predicted RMS std", color="tab:orange")
        ax.set_title(f"Q{quantile} | {int(part.iloc[0]['n_origins'])} MPC origins | forecast-error scale")
        ax.set_ylabel("Normalized demand")
        ax.grid(alpha=0.25)
        if row == 0: ax.legend(fontsize=9)

        ax = axes[row, 1]
        ax.plot(h, part["coverage_68_pct"], "o-", label="Observed ±1σ", color="tab:blue")
        ax.plot(h, part["coverage_95_pct"], "o-", label="Observed 95%", color="tab:orange")
        ax.axhline(68.27, ls="--", lw=1, color="tab:blue", alpha=0.65,
            label="Gaussian 68.27%")
        ax.axhline(95, ls="--", lw=1, color="tab:orange", alpha=0.65,
            label="Gaussian 95%")
        ax.set_title(f"Q{quantile} | observed prediction interval coverage")
        ax.set_ylabel("Coverage (%)")
        ax.set_ylim(0, 101)
        ax.grid(alpha=0.25)
        if row == 0: ax.legend(fontsize=9, ncol=2)
        for col in range(2): axes[row, col].set_xticks(h)
    for ax in axes[-1]: ax.set_xlabel("Exact forecast lead (hours)")
    status = "mature rolling fits only" if mature_only and mode in ROLLING_MODES else "all complete origins"
    fig.suptitle(
        f"WM4 MPC | RMSE and Gaussian interval coverage by predicted-uncertainty quantile\n"
        f"{mode} | {status} | fixed {int(summary['lead_hours'].max())}-hour MPC-origin cohorts (Q1 lowest uncertainty)",
        fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=175, bbox_inches="tight")
    plt.close(fig)
    return output


def analyze(forecasts=DEFAULT_FORECASTS, output=DEFAULT_OUTPUT, horizon=6,
    quantiles=5, mode="mpc_replay_28d_6h", mature_only=True, rolling_days=None):
    frame, excluded, verified_mode = load_forecasts(forecasts, horizon,
        expected_sigma_fit_mode=mode)
    selected, origin_bins = select_mpc_origins(frame, horizon=horizon, quantiles=quantiles,
        mode=verified_mode, mature_only=mature_only, rolling_days=rolling_days)
    summary = summarize_quantiles(selected, horizon)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "stochastic_calibration_by_quantile_and_lead.csv", index=False)
    origin_bins.to_csv(output / "mpc_origin_uncertainty_quantiles.csv", index=False)
    figure = plot_quantiles(summary, output / "stochastic_variance_calibration_quantiles.png",
        mode=verified_mode, mature_only=mature_only)
    print(f"WM4 {verified_mode} | {len(origin_bins)} MPC origins | {quantiles} equal-count "
        f"uncertainty quantiles | {excluded} incomplete origins excluded")
    print(summary.loc[summary["lead_hours"] == horizon,
        ["uncertainty_quantile", "n_origins", "observed_rmse", "predicted_rms_std",
         "predicted_to_observed_mse", "coverage_68_pct", "coverage_95_pct"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Plot: {figure}\nTables: {output}")
    return {"summary": summary, "origins": origin_bins, "figure": figure, "output": output}


def main():
    parser = ArgumentParser(description="Plot WM4 stochastic RMSE and 68%/95% interval coverage by uncertainty quintile, not pooled across all MPC forecasts.")
    parser.add_argument("--forecasts", type=Path, default=DEFAULT_FORECASTS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sigma-fit-mode", choices=MODES, default="mpc_replay_28d_6h")
    parser.add_argument("--horizon", type=int, default=None,
        help="Forecast horizon (6 for 28d/6h, otherwise 8 by default).")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--rolling-days", type=int, default=None,
        help="Required calibration history; defaults to 24, 28 or 30 by sigma-fit-mode.")
    parser.add_argument("--include-warmup", action="store_true",
        help="Include origins without the full mode-specific rolling history.")
    args = parser.parse_args()
    if args.rolling_days is not None and args.rolling_days < 2:
        parser.error("--rolling-days must be at least 2")
    output = args.output or Path(args.forecasts).parent / "stochastic_quantiles"
    horizon = ((6 if args.sigma_fit_mode == "mpc_replay_28d_6h" else 8)
        if args.horizon is None else args.horizon)
    return analyze(args.forecasts, output, horizon, args.quantiles,
        args.sigma_fit_mode, not args.include_warmup, args.rolling_days)


if __name__ == "__main__": main()
