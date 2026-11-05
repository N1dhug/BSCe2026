"""Fit constant Wiener diffusion on historical 6..24h MPC paths, evaluate at 6h.

Run 24-hour deterministic WM4 parameter/baseline fits and six-hour MPC
evaluations once. At every MPC origin additionally archive the entire REAL
24-hour forward MPC trajectory. For each diffusion fit horizon H=6..24,
estimate a new q=sigma² from the 25 previous days' corresponding-hour
24-hour MPC errors, using only their first H forecast leads:

    min_{q>=0} sum_{h=1}^H (sample_variance(e_{1:25,h}) - q mean(K_{1:25,h}))².

No initial-state covariance propagation in the fit or forecast. Every H
is evaluated on the SAME six-hour current-origin MPC errors, so parameter
and initial-condition reconstructions are not repeated across horizons.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY
from src.exe.run_model.wm_4.predict_all_days_mpc import ROOT, run_all_days
from src.exe.run_model.wm_4.predict_all_days_mpc_no_p0 import MODE
from src.exe.run_model.wm_4.sweep_mpc_sigma_days import RIDGE


BACKLOG_DAYS = 25
FORECAST_HOURS = 6
ARCHIVE_HOURS = 24
OUTPUT = ROOT / "files/output/wm_4/prediction/sigma_fit_hours_6_24_no_p0"


def fit_diffusion_horizons(history, min_hours=6, max_hours=24, ridge=RIDGE):
    """Fit scalar constant diffusion QP separately using first H historical leads.

    history: (25,24,3) with observed errors, zero P0 contributions and
    origin-specific unit-q process kernels, newest historical day first.
    Exact constrained scalar QP solution uses the WM4 1e-10 ridge.
    """
    values = np.asarray(history, dtype=float)
    lo, hi = int(min_hours), int(max_hours)
    if (values.ndim != 3 or values.shape[2] != 3
        or not 2 <= lo <= hi <= values.shape[1] or values.shape[0] < 2):
        raise ValueError("History must have >=2 complete MPC paths of >=max_hours leads.")
    if not np.isfinite(values).all() or np.any(values[:, :, 2] < 0.):
        raise ValueError("Historical errors/kernels must be finite with nonnegative kernels.")
    if not np.allclose(values[:, :, 1], 0., atol=1e-12):
        raise ValueError("No-P0 diffusion fit must not include initial-state variance.")
    S = np.var(values[:, :, 0], axis=0, ddof=1)
    K = np.mean(values[:, :, 2], axis=0)
    sk = np.cumsum(S * K)
    kk = np.cumsum(K ** 2)
    hs = np.arange(lo, hi + 1)
    q = np.maximum(sk[hs - 1] / np.maximum(
        kk[hs - 1] + float(ridge) * hs / 2., np.finfo(float).tiny), 0.)
    return hs, q, S, K


def six_hour_evaluation(forecasts):
    """Use saved actual six-hour MPC evaluation errors, not 24h archive targets."""
    required = {"origin", "lead_hours", "I_observed", "I_mpc",
        "initial_variance", "process_kernel_unit", "parameter_fit_hours"}
    if missing := required - set(forecasts):
        raise ValueError(f"Forecast output missing {sorted(missing)}.")
    frame = forecasts.copy()
    frame["origin"] = pd.to_datetime(frame["origin"])
    if (frame["parameter_fit_hours"] != DAY_HOURS).any():
        raise ValueError("Expected a 24-hour deterministic WM4 fit.")
    if not np.allclose(frame["initial_variance"].to_numpy(dtype=float), 0., atol=1e-12):
        raise ValueError("Current MPC evaluations contain propagated initial-state variance.")
    result = {}
    wanted = np.arange(1, FORECAST_HOURS + 1)
    for origin, part in frame.groupby("origin", sort=True):
        part = part.sort_values("lead_hours")
        if not np.array_equal(part["lead_hours"].to_numpy(dtype=int), wanted): continue
        error = (part["I_observed"] - part["I_mpc"]).to_numpy(dtype=float)
        K = part["process_kernel_unit"].to_numpy(dtype=float)
        if not np.isfinite(error).all() or not np.isfinite(K).all() or (K < 0).any():
            raise ValueError(f"Invalid MPC errors or Wiener kernels at {origin}.")
        result[pd.Timestamp(origin)] = np.column_stack((error, K))
    if not result: raise ValueError("No complete six-hour MPC evaluation origins.")
    return result


def replay(archive, evaluations, *, days=BACKLOG_DAYS, min_hours=6, max_hours=24):
    """Each fit uses exactly 25 older, completed, same-clock-hour MPC paths."""
    days = int(days)
    if days < 2: raise ValueError("At least two historical days required.")
    rows = []
    for origin in sorted(evaluations):
        training = []
        for offset in range(1, days + 1):
            historical = origin - pd.Timedelta(days=offset)
            values = archive.get(historical)
            # Historical 24h forecasts are complete by the current origin.
            if values is None or historical + pd.Timedelta(hours=ARCHIVE_HOURS) > origin:
                break
            values = np.asarray(values, dtype=float)
            if values.shape != (ARCHIVE_HOURS, 3):
                raise ValueError(f"Expected 24 historical MPC leads at {historical}.")
            training.append(values)
        if len(training) != days: continue
        hours, qs, _, _ = fit_diffusion_horizons(
            np.asarray(training), min_hours, max_hours)
        errors, current_kernels = evaluations[origin][:, 0], evaluations[origin][:, 1]
        for hours_fit, q in zip(hours, qs):
            forecast_var = np.maximum(q * current_kernels, 0.)
            row = {"origin": origin, "calibration_hours": int(hours_fit),
                "history_days": days, "n_history_mpcs": days,
                "diffusion_q": float(q), "diffusion_sigma": float(np.sqrt(q)),
                "observed_mse": float(np.mean(errors ** 2)),
                "predicted_variance": float(np.mean(forecast_var))}
            row.update({f"variance_h{h}": float(forecast_var[h - 1])
                for h in range(1, FORECAST_HOURS + 1)})
            rows.append(row)
    if not rows:
        raise ValueError(f"No six-hour evaluation origin has a full {days}-day backlog "
            "of completed 24-hour historical MPC trajectories.")
    return pd.DataFrame(rows)


def _score(errors, variances, origins, diffusion_q):
    """Gaussian forecast diagnostics across exact forecast leads, not variance of daily means."""
    e = np.asarray(errors, dtype=float)
    v = np.asarray(variances, dtype=float)
    if e.shape != v.shape: raise ValueError("Error and variance shapes differ.")
    mse, mean_v = float(np.mean(e ** 2)), float(np.mean(v))
    std = np.sqrt(np.maximum(v, 0.))
    safe = np.maximum(v, 1e-12)
    return {"n_origins": int(origins), "n_forecasts": int(e.size),
        "observed_rmse": float(np.sqrt(mse)),
        "predicted_rms_std": float(np.sqrt(mean_v)),
        "observed_mse": mse, "predicted_variance": mean_v,
        "predicted_to_observed_mse": mean_v / mse if mse > 0 else np.nan,
        "coverage_68_pct": 100. * float(np.mean(np.abs(e) <= std)),
        "coverage_95_pct": 100. * float(np.mean(np.abs(e) <= 1.96 * std)),
        "gaussian_nll": .5 * float(np.mean(np.log(2 * np.pi * safe) + e ** 2 / safe)),
        "mean_sigma": float(np.mean(np.sqrt(diffusion_q)))}


def summarize(fits, evaluations, quantiles=5):
    """Full matched-cohort scores, exact-lead scores and independent Q1–Q5 for each H."""
    quantiles = int(quantiles)
    if quantiles < 2: raise ValueError("At least two quantiles required.")
    overall, quintile_rows, leads = [], [], []
    for hours, part in fits.groupby("calibration_hours", sort=True):
        part = part.sort_values("origin").reset_index(drop=True)
        origins = part["origin"].tolist()
        errors = np.stack([evaluations[o][:, 0] for o in origins])
        variance = part[[f"variance_h{h}" for h in range(1, FORECAST_HOURS + 1)]
            ].to_numpy(dtype=float)
        if (variance < 0).any(): raise ValueError("Negative predicted variances.")
        q = part["diffusion_q"].to_numpy(dtype=float)
        overall.append({"calibration_hours": int(hours),
            **_score(errors, variance, len(origins), q)})
        for h in range(FORECAST_HOURS):
            leads.append({"calibration_hours": int(hours), "lead_hours": h + 1,
                **_score(errors[:, h], variance[:, h], len(origins), q)})
        if len(origins) < quantiles: continue
        # Entire MPC origin's mean 1..6h variance determines its quintile.
        ranks = pd.Series(variance.mean(axis=1)).rank(method="first")
        bins = pd.qcut(ranks, quantiles, labels=False).astype(int).to_numpy() + 1
        for group in range(1, quantiles + 1):
            mask = bins == group
            quintile_rows.append({"calibration_hours": int(hours),
                "uncertainty_quantile": group,
                **_score(errors[mask], variance[mask], int(mask.sum()), q[mask])})
    return pd.DataFrame(overall), pd.DataFrame(quintile_rows), pd.DataFrame(leads)


def save_plots(fits, overview, quantiles, leads, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    h = overview["calibration_hours"].to_numpy(dtype=int)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs[0, 0].plot(h, overview["observed_rmse"], "o--", label="Observed six-hour RMSE")
    axs[0, 0].plot(h, overview["predicted_rms_std"], "o-", label="Predicted RMS std")
    axs[0, 0].set_ylabel("Normalized demand")
    axs[0, 0].legend()
    axs[0, 1].plot(h, overview["predicted_to_observed_mse"], "o-")
    axs[0, 1].axhline(1., color="black", ls="--")
    axs[0, 1].set_ylabel("Predicted variance / observed MSE")
    axs[1, 0].plot(h, overview["coverage_68_pct"], "o-", label="±1σ")
    axs[1, 0].plot(h, overview["coverage_95_pct"], "o-", label="±1.96σ")
    axs[1, 0].axhline(68.27, ls="--", alpha=0.5)
    axs[1, 0].axhline(95., ls="--", alpha=0.5)
    axs[1, 0].set_ylabel("Observed coverage (%)")
    axs[1, 0].legend()
    axs[1, 1].plot(h, overview["gaussian_nll"], "o-")
    axs[1, 1].set_ylabel("Mean Gaussian NLL (lower is better)")
    for ax in axs.flat:
        ax.set_xlabel("Diffusion fitting horizon (hours)")
        ax.set_xticks(h[::max(1, len(h) // 9)])
        ax.grid(alpha=0.2)
    fig.suptitle(f"ARM (Affine Revised Model) | constant Wiener q, {int(fits['history_days'].iloc[0])}-day backlog, 24h historical MPC paths\n"
        f"Evaluation: same {int(overview['n_origins'].iloc[0])} origins, next six hours, no P0")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output / "fit_hours_sweep.png", dpi=165, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for group, part in quantiles.groupby("uncertainty_quantile"):
        axs[0].plot(part["calibration_hours"], part["predicted_to_observed_mse"],
            "o-", label=f"Q{group}")
        axs[1].plot(part["calibration_hours"], part["coverage_95_pct"],
            "o-", label=f"Q{group}")
    axs[0].axhline(1., color="black", ls="--")
    axs[1].axhline(95., color="black", ls="--")
    axs[0].set_ylabel("Predicted variance / observed MSE")
    axs[1].set_ylabel("95% interval coverage (%)")
    for ax in axs:
        ax.set_xlabel("Diffusion fitting horizon (hours)")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
    fig.suptitle("ARM (Affine Revised Model) | conditional six-hour reliability by 6–24h diffusion fitting horizon")
    fig.tight_layout()
    fig.savefig(output / "fit_hours_quintile_sweep.png", dpi=165, bbox_inches="tight")
    plt.close(fig)

    table = leads.pivot(index="lead_hours", columns="calibration_hours",
        values="predicted_to_observed_mse").sort_index()
    fig, ax = plt.subplots(figsize=(13, 4.8))
    heat = ax.imshow(table.to_numpy(), origin="lower", aspect="auto", vmin=0.25, vmax=2.)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set(xlabel="Diffusion fitting horizon (hours)", ylabel="Evaluation lead (hours)",
        title="Predicted variance / observed MSE by calibration horizon and evaluation lead")
    fig.colorbar(heat, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "fit_hours_lead_heatmap.png", dpi=165, bbox_inches="tight")
    plt.close(fig)

    vals = fits[[f"variance_h{x}" for x in range(1, FORECAST_HOURS + 1)]].mean(axis=1)
    max_value = 1.05 * max(float(fits["observed_mse"].max()), float(vals.max()), 1e-8)
    hours = sorted(fits["calibration_hours"].unique())
    fig, axs = plt.subplots(int(np.ceil(len(hours) / 4)), 4,
        figsize=(17, 3.5 * int(np.ceil(len(hours) / 4))), squeeze=False)
    for ax, hour in zip(axs.flat, hours):
        subset = fits.loc[fits["calibration_hours"] == hour]
        v = subset[[f"variance_h{x}" for x in range(1, FORECAST_HOURS + 1)]].mean(axis=1)
        ax.scatter(v, subset["observed_mse"], s=10, alpha=0.5)
        ax.plot([0, max_value], [0, max_value], ls="--", color="black", lw=0.8)
        ax.set(xlim=(0, max_value), ylim=(0, max_value), title=f"Fit 1–{hour}h")
        ax.grid(alpha=0.2)
    for ax in list(axs.flat)[len(hours):]: ax.axis("off")
    fig.supxlabel("Predicted six-hour mean variance")
    fig.supylabel("Realized six-hour MPC MSE")
    fig.suptitle("ARM (Affine Revised Model) | same origins across all diffusion fitting horizons | no initial P0")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output / "fit_hours_scatter.png", dpi=145, bbox_inches="tight")
    plt.close(fig)


def run(frame, start=START_DAY, end=END_DAY, *, steps_per_hour=60,
    baseline_weeks=BASELINE_WEEKS, history_days=BACKLOG_DAYS,
    min_hours=6, max_hours=24, quantiles=5, output=OUTPUT, fail_fast=False):
    """One full deterministic MPC pass, then independent constant-q calibration QPs."""
    min_hours, max_hours = int(min_hours), int(max_hours)
    history_days, quantiles = int(history_days), int(quantiles)
    if not 2 <= min_hours <= max_hours <= ARCHIVE_HOURS:
        raise ValueError("Require 2 <= min_hours <= max_hours <= 24.")
    if history_days < 2 or quantiles < 2:
        raise ValueError("Require at least 2 historical days and 2 quantiles.")
    archive = {}
    forecasts, states, skipped = run_all_days(frame, start=start, end=end,
        steps_per_hour=steps_per_hour, baseline_weeks=baseline_weeks,
        max_horizon=FORECAST_HOURS, fit_hours=DAY_HOURS,
        sigma_fit_mode=MODE, sigma_rolling_days=history_days,
        mpc_archive=archive, archive_horizon=ARCHIVE_HOURS,
        fail_fast=fail_fast)
    evaluations = six_hour_evaluation(forecasts)
    for origin, arr in archive.items():
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (ARCHIVE_HOURS, 3) or not np.all(np.isfinite(arr)):
            raise ValueError(f"Missing/invalid 24h historical MPC path at {origin}.")
        if not np.allclose(arr[:, 1], 0., atol=1e-12):
            raise ValueError(f"Unexpected initial-state covariance at {origin}.")
    fits = replay(archive, evaluations, days=history_days,
        min_hours=min_hours, max_hours=max_hours)
    if fits["origin"].nunique() < quantiles:
        raise ValueError(f"Only {fits['origin'].nunique()} full-history MPC origins; "
            f"at least {quantiles} are needed for uncertainty quantiles.")
    score, groups, leads = summarize(fits, evaluations, quantiles=quantiles)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = {"fits": fits, "summary": score, "quantiles": groups,
        "leads": leads, "mpc_forecasts": forecasts, "mpc_states": states}
    for name, table in tables.items(): table.to_csv(output / f"{name}.csv", index=False)
    # Preserve the real 24-hour historical MPC paths for later experiments
    # with different diffusion fitting/evaluation horizons, without refitting WM4.
    archive_rows = []
    for origin, values in sorted(archive.items()):
        for h, (error, initial, kernel) in enumerate(values, 1):
            archive_rows.append({"origin": origin, "lead_hours": h,
                "forecast_error": float(error),
                "initial_variance": float(initial),
                "process_kernel_unit": float(kernel)})
    pd.DataFrame(archive_rows).to_csv(output / "mpc_calibration_24h.csv", index=False)
    if skipped:
        pd.DataFrame([{"origin": o, "reason": reason} for o, reason in skipped]
            ).to_csv(output / "skipped_origins.csv", index=False)
    save_plots(fits, score, groups, leads, output)
    print(f"WM4: 24h deterministic parameter fitting; "
        f"{history_days} historical full 24h MPC trajectories per origin; "
        f"evaluation next {FORECAST_HOURS}h; no initial-state variance.")
    print(f"Full-history eligible origins: {fits['origin'].nunique()}; "
        f"fresh constant-diffusion QP fits: {len(fits)}.")
    print(score[["calibration_hours", "n_origins", "predicted_to_observed_mse",
        "coverage_68_pct", "coverage_95_pct", "gaussian_nll", "mean_sigma"]].to_string(
        index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"Saved results: {output}")
    return {"archive": archive, "evaluations": evaluations,
        **tables, "skipped": skipped, "output": output}


def main():
    parser = ArgumentParser(description="WM4: use 25 days of genuine 24h MPC error paths to fit constant Wiener diffusion over horizons 6..24h; evaluate every fit on identical six-hour MPC predictions without P0.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--history-days", type=int, default=BACKLOG_DAYS)
    parser.add_argument("--min-hours", type=int, default=6)
    parser.add_argument("--max-hours", type=int, default=24)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    data, _ = aligned_wm4_data("MED")
    return run(data, args.start_day, args.end_day,
        steps_per_hour=args.steps_per_hour, baseline_weeks=args.baseline_weeks,
        history_days=args.history_days, min_hours=args.min_hours,
        max_hours=args.max_hours, quantiles=args.quantiles,
        output=args.output, fail_fast=args.fail_fast)


if __name__ == "__main__": main()
