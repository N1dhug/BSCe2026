"""Daily WM4 MPC uncertainty quantiles from already-saved 7–28-day diffusion fits.

One observation is an origin CALENDAR DAY containing 24 hourly MPC origins
times six forecast leads (144 forecast errors). Daily predicted variance is
the MEAN of the 144 forecast variances; observed MSE is the MEAN of their
squared errors. These are NOT the variance/error of a daily-mean consumption
forecast, since overlapping MPC errors are correlated.

Uses sigma_days_sweep_no_p0/fits.csv and mpc_forecasts.csv by default. No
deterministic WM4 fit, Wiener QP, or MPC trajectory is recomputed. Works also
with the original P0 sweep by passing its CSVs.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.sweep_mpc_sigma_days import HORIZON, ROOT, read_mpc_archive


SOURCE = ROOT / "files/output/wm_4/prediction/sigma_days_sweep_no_p0"
OUTPUT = SOURCE / "daily_mpc_quantiles"
FORECASTS = SOURCE / "mpc_forecasts.csv"
FITS = SOURCE / "fits.csv"
HOURS = 24


def load_inputs(fits=FITS, forecasts=FORECASTS, fit_hours=24):
    """Read saved scalar sigma fits and six-lead MPC forecasts; never refit."""
    archive, raw = read_mpc_archive(forecasts, horizon=HORIZON, fit_hours=fit_hours)
    frame = pd.read_csv(fits, parse_dates=["origin"])
    required = {"origin", "history_days", "diffusion_q",
        *(f"variance_h{h}" for h in range(1, HORIZON + 1))}
    if missing := required - set(frame):
        raise ValueError(f"Historical sigma fit table missing: {sorted(missing)}")
    if frame.duplicated(["history_days", "origin"]).any():
        raise ValueError("Duplicate origin/window combinations in sigma fits.")
    numeric = ["history_days", "diffusion_q"] + [
        f"variance_h{h}" for h in range(1, HORIZON + 1)]
    for field in numeric: frame[field] = pd.to_numeric(frame[field], errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Saved diffusion fit data contain nonfinite values.")
    if (frame[["diffusion_q"] + numeric[2:]] < 0).any().any():
        raise ValueError("Diffusion coefficients and forecast variances must be nonnegative.")
    if (frame["history_days"] < 2).any() or not np.equal(
        frame["history_days"], np.floor(frame["history_days"])).all():
        raise ValueError("Historical fitting days must be integer values >= 2.")
    frame["history_days"] = frame["history_days"].astype(int)
    unknown = set(frame["origin"]) - set(archive)
    if unknown:
        raise ValueError(f"{len(unknown)} fitted MPC origins lack complete saved six-hour forecasts.")
    if not len(frame): raise ValueError("No saved diffusion fits.")
    return frame, archive, raw


def daily_metrics(fits, archive, raw, *, origins_per_day=HOURS):
    """Pool all 24 origins x six leads per origin, for every fit window.

    Only full 24-origin calendar days are retained. The day is determined
    by the MPC ORIGIN, not by target timestamps (the last six-hour paths
    extend into the next calendar day).
    """
    origins_per_day = int(origins_per_day)
    if not 1 <= origins_per_day <= HOURS:
        raise ValueError("origins_per_day must be in [1,24].")
    if fits.duplicated(["history_days", "origin"]).any():
        raise ValueError("Every MPC origin must occur only once in each window.")
    forecast_means = {}
    complete = raw.loc[raw["origin"].isin(archive)].copy()
    for origin, block in complete.groupby("origin", sort=True):
        block = block.sort_values("lead_hours")
        forecast_means[origin] = (
            float(block["I_mpc"].mean()), float(block["I_observed"].mean()))
    rows = []
    table = fits.copy()
    table["date"] = table["origin"].dt.normalize()
    variance_columns = [f"variance_h{h}" for h in range(1, HORIZON + 1)]
    skipped = []
    for (n, day), block in table.groupby(["history_days", "date"], sort=True):
        hours = block["origin"].dt.hour.to_numpy(dtype=int)
        valid = (len(block) == origins_per_day
            and (np.array_equal(np.sort(hours), np.arange(HOURS))
                if origins_per_day == HOURS else len(set(hours)) == origins_per_day))
        if not valid:
            skipped.append({"history_days": int(n), "date": day,
                "available_origins": int(len(block))})
            continue
        block = block.sort_values("origin")
        origins = block["origin"].tolist()
        error = np.stack([archive[o][:, 0] for o in origins])
        variances = block[variance_columns].to_numpy(dtype=float)
        if variances.shape != error.shape:
            raise ValueError("Saved six-hour variance and realized-error trajectories differ.")
        # Audit exact decomposition for each origin/lead. This permits either
        # P0 or no-P0 data without silently mixing forecast CSV generations.
        B = np.stack([archive[o][:, 1] for o in origins])
        K = np.stack([archive[o][:, 2] for o in origins])
        estimated = B + block["diffusion_q"].to_numpy(dtype=float)[:, None] * K
        if not np.allclose(variances, estimated, atol=1e-8, rtol=1e-6):
            raise ValueError(f"Saved forecast variances do not match diffusion fit at {day}, N={n}.")
        predicted = np.array([forecast_means[o][0] for o in origins])
        observed = np.array([forecast_means[o][1] for o in origins])
        mse = float(np.mean(error ** 2))
        v = float(np.mean(variances))
        sd = np.sqrt(variances)
        safe = np.maximum(variances, 1e-12)
        rows.append({"history_days": int(n), "date": day,
            "n_origins": int(len(origins)), "n_forecasts": int(error.size),
            "mean_predicted_consumption": float(predicted.mean()),
            "mean_observed_consumption": float(observed.mean()),
            "mean_forecast_error": float(error.mean()),
            "squared_mean_forecast_error": float(np.mean(error) ** 2),
            "observed_mse": mse, "observed_rmse": float(np.sqrt(mse)),
            "predicted_variance": v, "predicted_rms_std": float(np.sqrt(v)),
            "predicted_to_observed_mse": v / mse if mse > 0 else np.nan,
            "coverage_68_pct": float(100 * np.mean(np.abs(error) <= sd)),
            "coverage_95_pct": float(100 * np.mean(np.abs(error) <= 1.96 * sd)),
            "gaussian_nll": float(np.mean(0.5 * (
                np.log(2 * np.pi * safe) + error ** 2 / safe))),
            "mean_sigma": float(np.mean(np.sqrt(
                block["diffusion_q"].to_numpy(dtype=float)))),
            "zero_sigma_pct": float(100 * np.mean(block["diffusion_q"] <= 1e-14))})
    if not rows:
        raise ValueError("No full calendar days with 24 complete six-hour MPC origins. "
            "Check that you saved hourly MPC forecasts covering consecutive full days.")
    return pd.DataFrame(rows).sort_values(["history_days", "date"]).reset_index(drop=True), pd.DataFrame(skipped)


def _summarize(daily, quantiles=5, cohort="all"):
    """Rank DAYS by their mean predicted variance for every sigma backlog."""
    quantiles = int(quantiles)
    if quantiles < 2: raise ValueError("At least two daily quantiles required.")
    metrics, groups = [], []
    for n, block in daily.groupby("history_days", sort=True):
        block = block.sort_values("date").copy()
        # Each day's 144 forecast errors contribute equally because every
        # retained day contains exactly 24 origins and six forecast leads.
        def score(sub, label):
            real = float(sub["observed_mse"].mean())
            pred = float(sub["predicted_variance"].mean())
            return {"cohort": cohort, "history_days": int(n),
                "uncertainty_quantile": int(label),
                "n_days": int(len(sub)), "n_origins": int(sub["n_origins"].sum()),
                "n_forecasts": int(sub["n_forecasts"].sum()),
                "observed_rmse": float(np.sqrt(real)),
                "predicted_rms_std": float(np.sqrt(pred)),
                "observed_mse": real, "predicted_variance": pred,
                "predicted_to_observed_mse": pred / real if real > 0 else np.nan,
                "coverage_68_pct": float(sub["coverage_68_pct"].mean()),
                "coverage_95_pct": float(sub["coverage_95_pct"].mean()),
                "gaussian_nll": float(sub["gaussian_nll"].mean()),
                "mean_sigma": float(sub["mean_sigma"].mean()),
                "mean_daily_bias": float(sub["mean_forecast_error"].mean())}
        metrics.append(score(block, 0))
        if len(block) < quantiles: continue
        ranks = block["predicted_variance"].rank(method="first")
        block["uncertainty_quantile"] = pd.qcut(
            ranks, quantiles, labels=False).astype(int) + 1
        for quintile, group in block.groupby("uncertainty_quantile", sort=True):
            groups.append(score(group, quintile))
    return pd.DataFrame(metrics), pd.DataFrame(groups)


def analyze(fits=FITS, forecasts=FORECASTS, output=OUTPUT,
    quantiles=5, fit_hours=24):
    """Evaluate all available full days and the exact same days for all windows."""
    frame, archive, raw = load_inputs(fits, forecasts, fit_hours)
    daily, incomplete = daily_metrics(frame, archive, raw)
    windows = sorted(daily["history_days"].unique())
    # A date must have all 24 origins for EVERY window length to qualify for
    # the matched cohort. Unlike an origin-level matched cohort, this can
    # contain fewer days, especially if the saved sweep spans few full days.
    counts = daily.groupby("date")["history_days"].nunique()
    dates = counts[counts == len(windows)].index
    matched = daily[daily["date"].isin(dates)].copy()
    all_summary, all_quantiles = _summarize(daily, quantiles, "all")
    matched_summary, matched_quantiles = _summarize(
        matched, quantiles, "matched") if not matched.empty else (pd.DataFrame(), pd.DataFrame())

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = {"daily_metrics_all": daily, "daily_metrics_matched": matched,
        "incomplete_days": incomplete, "daily_summary_all": all_summary,
        "daily_quantiles_all": all_quantiles,
        "daily_summary_matched": matched_summary,
        "daily_quantiles_matched": matched_quantiles}
    for name, value in tables.items(): value.to_csv(output / f"{name}.csv", index=False)

    # Plot both cohorts independently; matching is the fair window-length
    # comparison, whereas 'all' offers much stronger daily quantile samples.
    for cohort, detail, overview, quintile_table in (
        ("all", daily, all_summary, all_quantiles),
        ("matched", matched, matched_summary, matched_quantiles)):
        if detail.empty: continue
        _plot_daily_scatter(detail, output / f"{cohort}_daily_scatter.png", cohort)
        _plot_mean_forecasts(detail,
            output / f"{cohort}_daily_mean_forecasts.png", cohort)
        if quintile_table.empty: continue
        _plot_daily_quantiles(quintile_table,
            output / f"{cohort}_daily_quintile_sweep.png", cohort)
        _plot_reference_quintiles(quintile_table,
            output / f"{cohort}_daily_quantiles_reference.png", cohort)
        _plot_overall(overview, output / f"{cohort}_daily_calibration.png", cohort)

    print(f"Saved daily MPC metrics: {len(daily)} complete day/window pairs; "
        f"{daily['date'].nunique()} distinct calendar days; "
        f"{len(matched['date'].unique()) if len(matched) else 0} fully matched days.")
    print(f"All estimates use 24 hourly origins x {HORIZON} leads = 144 forecast errors per day.")
    if len(matched) < quantiles * len(windows):
        print(f"WARNING: fewer than {quantiles} matched complete days. "
            "The matched daily quintile plot is unavailable; use all_daily_quintile_sweep.")
    print(all_summary[["history_days", "n_days", "predicted_to_observed_mse",
        "coverage_95_pct", "gaussian_nll"]].to_string(index=False,
        float_format=lambda x: f"{x:.4f}"))
    print(f"Saved daily quintile CSVs and plots at {output}")
    return {**tables, "output": output}


def _plot_daily_scatter(daily, path, cohort):
    """One dot per calendar day; fixed axes for all diffusion fitting windows."""
    windows = sorted(daily["history_days"].unique())
    cols = 4
    count_rows = int(np.ceil(len(windows) / cols))
    fig, axes = plt.subplots(count_rows, cols, figsize=(17, 3.6 * count_rows),
        squeeze=False)
    high = max(float(daily["observed_mse"].max()),
        float(daily["predicted_variance"].max()), 1e-8) * 1.05
    for ax, n in zip(axes.flat, windows):
        sub = daily.loc[daily["history_days"] == n]
        ax.scatter(sub["predicted_variance"], sub["observed_mse"],
            s=17, alpha=0.65)
        ax.plot([0, high], [0, high], "k--", lw=0.8)
        ax.set(xlim=(0, high), ylim=(0, high), title=f"{n} days | {len(sub)} daily averages")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(windows):]: ax.axis("off")
    fig.supxlabel("Daily mean predicted MPC variance (24 origins × 6 leads)")
    fig.supylabel("Daily mean observed MPC squared error")
    fig.suptitle(f"WM4 | {cohort} days | each point is one origin calendar day")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_mean_forecasts(daily, path, cohort, reference=24):
    """Visualize the 144-prediction daily mean AND pooled forecast-error scale."""
    window = reference if reference in set(daily["history_days"]) else int(
        daily["history_days"].max())
    sub = daily.loc[daily["history_days"] == window].sort_values("date")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(sub["date"], sub["mean_observed_consumption"], "o-",
        label="Mean observed demand")
    axes[0].plot(sub["date"], sub["mean_predicted_consumption"], "o-",
        label="Mean MPC predicted demand")
    axes[0].set(ylabel="Mean normalized demand",
        title=f"{window}d: average of 24 hourly MPC origins × six leads")
    axes[1].plot(sub["date"], sub["observed_rmse"], "o-",
        label="RMSE across 144 forecasts")
    axes[1].plot(sub["date"], sub["predicted_rms_std"], "o-",
        label="Predicted RMS std across 144 forecasts")
    axes[1].set(ylabel="Normalized demand", xlabel="MPC origin calendar day",
        title="Daily pooled forecast-error scale (not uncertainty of the mean demand)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"WM4 | {cohort} daily MPC averages | forecasts saved, no refitting")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_daily_quantiles(q, path, cohort):
    """Each Q1–Q5 group contains DAYS, not hourly MPC forecast origins."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for label, sub in q.groupby("uncertainty_quantile", sort=True):
        sub = sub.sort_values("history_days")
        axes[0].plot(sub["history_days"], sub["predicted_to_observed_mse"],
            "o-", label=f"Q{label}")
        axes[1].plot(sub["history_days"], sub["coverage_95_pct"],
            "o-", label=f"Q{label}")
    axes[0].axhline(1, color="black", ls="--")
    axes[0].set(title="Daily variance ratio (ideal = 1)",
        ylabel="Mean predicted variance / mean realized MSE")
    axes[1].axhline(95, color="black", ls="--")
    axes[1].set(title="Daily cohorts: pooled 95% MPC interval coverage",
        ylabel="Forecast-level coverage (%)")
    for ax in axes:
        ax.set_xlabel("Historical diffusion fitting days")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
    fig.suptitle(f"WM4 | {cohort} | conditional reliability by daily-average uncertainty")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_reference_quintiles(q, path, cohort, reference=24):
    window = reference if reference in set(q["history_days"]) else int(
        q["history_days"].max())
    sub = q.loc[q["history_days"] == window].sort_values("uncertainty_quantile")
    x = sub["uncertainty_quantile"].to_numpy(dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].plot(x, sub["observed_rmse"], "o-", label="Observed MPC RMSE")
    axes[0].plot(x, sub["predicted_rms_std"], "o-", label="Predicted RMS std")
    axes[0].set(ylabel="Normalized demand", title=f"{window}d: error scale by daily Q1–Q5")
    axes[1].plot(x, sub["coverage_68_pct"], "o-", label="±1σ interval")
    axes[1].plot(x, sub["coverage_95_pct"], "o-", label="±1.96σ interval")
    axes[1].axhline(68.27, ls="--", alpha=0.6)
    axes[1].axhline(95., ls="--", alpha=0.6)
    axes[1].set(ylabel="Forecast-level interval coverage (%)",
        title=f"{window}d: daily-quintile interval coverage")
    for ax in axes:
        ax.set_xlabel("Daily predicted-uncertainty quintile")
        ax.set_xticks(x)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"WM4 | {cohort} | one quintile observation = one complete MPC day")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_overall(summary, path, cohort):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    axes[0].plot(summary["history_days"], summary["predicted_to_observed_mse"], "o-")
    axes[0].axhline(1, color="black", ls="--")
    axes[0].set(title="All complete days: aggregate variance calibration",
        ylabel="Predicted variance / observed MSE")
    axes[1].plot(summary["history_days"], summary["gaussian_nll"], "o-")
    axes[1].set(title="Mean Gaussian NLL", ylabel="NLL (lower is better)")
    for ax in axes:
        ax.set_xlabel("Historical diffusion fitting days")
        ax.grid(alpha=0.25)
    fig.suptitle(f"WM4 | {cohort} daily MPC-origin aggregation")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Analyze daily-averaged 24×6 MPC forecasts and uncertainty quintiles for saved 7–28-day diffusion fits; NO refitting.")
    parser.add_argument("--fits", type=Path, default=FITS)
    parser.add_argument("--forecasts", type=Path, default=FORECASTS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--fit-hours", type=int, default=24)
    args = parser.parse_args()
    if args.quantiles < 2: parser.error("quantiles must be at least 2")
    output = args.output or Path(args.fits).parent / "daily_mpc_quantiles"
    return analyze(args.fits, args.forecasts, output, args.quantiles, args.fit_hours)


if __name__ == "__main__": main()
