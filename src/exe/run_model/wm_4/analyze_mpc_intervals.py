"""Analyze saved WM4 1-8h receding forecasts and cumulative uncertainty.

No model refitting is performed. Process variance is the time-accumulated
Wiener contribution K_h sigma^2; cumulative 1..H statistics instead pool
forecast errors/variances over all leads up to H (not a sum of variances).
"""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
MPC_OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_mpc"
FORECASTS = MPC_OUTPUT / "mpc_forecasts.csv"
OUTPUT = MPC_OUTPUT / "interval_analysis"
REQUIRED = {
    "origin", "target_time", "lead_hours", "I_observed", "I_mpc",
    "initial_variance", "process_variance", "predicted_variance", "diffusion_sigma",
    "lower_68", "upper_68", "lower_95", "upper_95",
}
P0_REQUIRED = {
    "sigma_fit_mode", "r0_variance_mode", "initial_F_variance", "initial_R_variance",
    "initial_F_component_variance", "initial_R_component_variance",
    "baseline_origin_residual_variance",
}


def load_forecasts(path, horizon=8, *, expected_r0_variance_mode="baseline_residual", allow_legacy=False):
    """Validate saved distributions; reject incompatible pre-update CSVs by default.

    The analyzer does not refit WM4 or recompute covariance. It must not plot
    stale forecast intervals as if they contained baseline-derived R0 uncertainty.
    """
    horizon = int(horizon)
    if not 1 <= horizon <= 8: raise ValueError("horizon must be in [1, 8].")
    if expected_r0_variance_mode not in ("baseline_residual", "zero", "any"):
        raise ValueError("expected_r0_variance_mode must be baseline_residual, zero, or any.")
    path = Path(path).resolve()
    if not path.is_file(): raise FileNotFoundError(
        f"WM4 MPC forecasts not found at {path}. Run python -m "
        "src.exe.run_model.wm_4.predict_all_days_mpc first, or specify --forecasts.")
    frame = pd.read_csv(path)
    missing_p0 = P0_REQUIRED - set(frame.columns)
    if missing_p0 and not allow_legacy:
        raise ValueError(
            f"OUTDATED WM4 MPC FORECAST DATA: {path}\n"
            f"Missing baseline-P0 columns: {sorted(missing_p0)}.\n"
            "The analysis reads an existing CSV and does not refit WM4. "
            "Regenerate it with: python -m src.exe.run_model.wm_4.predict_all_days_mpc "
            "--r0-variance-mode baseline_residual --sigma-fit-mode historical_p0\n"
            "If MPC was saved with --output, pass its mpc_forecasts.csv via --forecasts. "
            "To analyze older predictions explicitly, pass --allow-legacy.")
    if missing_p0: print(
        f"WARNING: legacy WM4 MPC forecast data missing {sorted(missing_p0)}; "
        "baseline-derived R0 variance is NOT verified.", flush=True)
    if "r0_variance_mode" in frame:
        modes = frame["r0_variance_mode"].dropna().astype(str).unique()
        if len(modes) != 1: raise ValueError(f"Mixed or missing r0_variance_mode values: {list(modes)}")
        if expected_r0_variance_mode != "any" and modes[0] != expected_r0_variance_mode:
            raise ValueError(
                f"WM4 forecast CSV has r0_variance_mode={modes[0]!r}, but analysis expects "
                f"{expected_r0_variance_mode!r}.\n"
                "Regenerate MPC forecasts with --r0-variance-mode "
                f"{expected_r0_variance_mode}, or use --expected-r0-variance-mode "
                f"{modes[0]} to analyze these results.")
    elif not allow_legacy:
        raise ValueError("WM4 forecast CSV is missing r0_variance_mode.")
    missing = REQUIRED - set(frame.columns)
    if missing: raise ValueError(f"Forecast CSV is missing required columns: {sorted(missing)}")
    frame["origin"] = pd.to_datetime(frame["origin"])
    frame["target_time"] = pd.to_datetime(frame["target_time"])
    if frame["origin"].dt.tz is not None or frame["target_time"].dt.tz is not None:
        raise ValueError("Forecast timestamps must be timezone-naive.")
    frame["lead_hours"] = pd.to_numeric(frame["lead_hours"], errors="raise").astype(int)
    if frame.duplicated(["origin", "lead_hours"]).any():
        raise ValueError("Duplicate origin/lead combinations found in forecast CSV.")
    frame = frame[frame["lead_hours"].between(1, horizon)].copy()
    if frame.empty: raise ValueError("Forecast CSV has no predictions within the requested horizon.")

    numeric = (REQUIRED - {"origin", "target_time", "lead_hours"}) | (
        (P0_REQUIRED - {"sigma_fit_mode", "r0_variance_mode"}) & set(frame.columns))
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.all(np.isfinite(frame[name].to_numpy(dtype=float))):
            raise ValueError(f"Forecast column {name} contains nonfinite values.")
    for name in (P0_REQUIRED - {"sigma_fit_mode", "r0_variance_mode"}) & set(frame.columns):
        if (frame[name] < 0).any(): raise ValueError(f"{name} must be nonnegative.")
    for name in ("initial_variance", "process_variance", "predicted_variance", "diffusion_sigma"):
        if (frame[name] < 0).any(): raise ValueError(f"{name} must be nonnegative.")
    for confidence in (68, 95):
        if (frame[f"lower_{confidence}"] > frame[f"upper_{confidence}"]).any():
            raise ValueError(f"Invalid {confidence}% interval ordering.")
    components = frame["initial_variance"] + frame["process_variance"]
    if not np.allclose(frame["predicted_variance"], components, atol=1e-8, rtol=1e-5):
        raise ValueError("Saved predicted_variance disagrees with initial_variance + process_variance.")
    if {"initial_F_component_variance", "initial_R_component_variance"} <= set(frame.columns):
        derived = frame["initial_F_component_variance"] + frame["initial_R_component_variance"]
        if not np.allclose(frame["initial_variance"], derived, atol=1e-8, rtol=1e-5):
            raise ValueError("Saved initial_variance disagrees with propagated F0 + R0 variance.")
        if "initial_R_variance" in frame and "r0_variance_mode" in frame:
            zero = frame["r0_variance_mode"].eq("zero")
            if np.any(zero) and not (
                np.allclose(frame.loc[zero, "initial_R_variance"], 0.0, atol=1e-10)
                and np.allclose(frame.loc[zero, "initial_R_component_variance"], 0.0, atol=1e-10)):
                raise ValueError("R0=zero mode contains nonzero initial R0 uncertainty.")
    expected = frame["origin"] + pd.to_timedelta(frame["lead_hours"], unit="h")
    if not (frame["target_time"] == expected).all():
        raise ValueError("Saved target_time must equal origin + lead_hours.")

    modes = frame["sigma_fit_mode"].dropna().unique() if "sigma_fit_mode" in frame else []
    if len(modes) > 1: raise ValueError(f"Forecast data mixes diffusion fitting modes: {list(modes)}")
    mode = str(modes[0]) if len(modes) else "legacy/unspecified"
    if not allow_legacy and mode != "historical_p0":
        print(f"NOTE: WM4 sigma_fit_mode={mode!r}; 'historical_p0' is the P0-aware diffusion QP.", flush=True)
    complete = frame.groupby("origin")["lead_hours"].agg(
        lambda values: np.array_equal(np.sort(values.to_numpy()), np.arange(1, horizon + 1)))
    discarded = int((~complete).sum())
    frame = frame[frame["origin"].isin(complete[complete].index)].copy()
    if frame.empty: raise ValueError(f"No complete {horizon}-hour forecast origins in CSV.")

    frame["origin_day"] = frame["origin"].dt.normalize()
    frame["origin_hour"] = frame["origin"].dt.hour
    frame["error"] = frame["I_observed"] - frame["I_mpc"]
    frame["squared_error"] = frame["error"] ** 2
    for confidence in (68, 95):
        frame[f"covered_{confidence}"] = (
            (frame["I_observed"] >= frame[f"lower_{confidence}"])
            & (frame["I_observed"] <= frame[f"upper_{confidence}"]))
    frame["process_share"] = np.divide(frame["process_variance"], frame["predicted_variance"],
        out=np.full(len(frame), np.nan), where=frame["predicted_variance"] > 0.0)
    return frame.sort_values(["origin", "lead_hours"]).reset_index(drop=True), discarded, mode


def scores(frame):
    """Properly pool errors and variances; standard deviations never get averaged."""
    error = frame["error"].to_numpy(dtype=float)
    initial = float(frame["initial_variance"].mean())
    process = float(frame["process_variance"].mean())
    variance = float(frame["predicted_variance"].mean())
    mse = float(np.mean(error ** 2))
    ratio = float(np.sqrt(variance / mse)) if mse > 0.0 else np.nan
    return {
        "origins": int(frame["origin"].nunique()),
        "days": int(frame["origin_day"].nunique()),
        "n": int(len(frame)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "bias_observed_minus_predicted": float(np.mean(error)),
        "coverage_68_pct": 100.0 * float(frame["covered_68"].mean()),
        "coverage_95_pct": 100.0 * float(frame["covered_95"].mean()),
        "mean_initial_variance": initial,
        "mean_initial_F_component_variance": float(frame["initial_F_component_variance"].mean())
            if "initial_F_component_variance" in frame else np.nan,
        "mean_initial_R_component_variance": float(frame["initial_R_component_variance"].mean())
            if "initial_R_component_variance" in frame else np.nan,
        "mean_process_variance": process,
        "mean_predicted_variance": variance,
        "initial_rms_std": float(np.sqrt(initial)),
        "process_rms_std": float(np.sqrt(process)),
        "predicted_rms_std": float(np.sqrt(variance)),
        "predicted_rms_std_to_rmse": ratio,
        "process_variance_share_pct": 100.0 * process / variance if variance > 0.0 else np.nan,
        "mean_sigma": float(frame["diffusion_sigma"].mean()),
    }


def summaries(frame, horizon):
    exact, cumulative, by_hour = [], [], []
    for lead in range(1, horizon + 1):
        exact.append({"lead_hours": lead, **scores(frame[frame["lead_hours"] == lead])})
        cumulative.append({"horizon_hours": lead, **scores(frame[frame["lead_hours"] <= lead])})
    for (hour, lead), part in frame.groupby(["origin_hour", "lead_hours"], sort=True):
        by_hour.append({"origin_hour": int(hour), "lead_hours": int(lead), **scores(part)})
    origins = frame.drop_duplicates("origin").sort_values("origin")[
        ["origin", "origin_day", "origin_hour", "diffusion_sigma"]].copy()
    if "sigma_fit_mode" in frame:
        origins["sigma_fit_mode"] = frame.drop_duplicates("origin").sort_values("origin")["sigma_fit_mode"].to_numpy()
    daily = origins.groupby("origin_day")["diffusion_sigma"].agg(
        origins="size", sigma_mean="mean", sigma_median="median",
        sigma_q25=lambda x: x.quantile(0.25), sigma_q75=lambda x: x.quantile(0.75)).reset_index()
    return pd.DataFrame(exact), pd.DataFrame(cumulative), pd.DataFrame(by_hour), origins, daily


def selected_day(frame, day=None):
    if day is None:
        sizes = frame.groupby("origin_day")["origin"].nunique().sort_index()
        full = sizes[sizes == 24]
        day = full.index[0] if not full.empty else sizes.idxmax()
    else:
        day = pd.Timestamp(day).normalize()
    part = frame[frame["origin_day"] == day]
    if part.empty: raise ValueError(f"No complete MPC origins for selected day {day:%Y-%m-%d}.")
    return pd.Timestamp(day), part


def _save_fig(fig, path):
    path = Path(path)
    try: fig.savefig(path, dpi=190, bbox_inches="tight")
    except PermissionError:
        path = path.with_name(f"{path.stem}.{uuid4().hex[:8]}{path.suffix}")
        fig.savefig(path, dpi=190, bbox_inches="tight")
        print(f"Output locked; saved figure as {path.name}", flush=True)
    finally: plt.close(fig)
    return path


def _save_csv(frame, path):
    try: frame.to_csv(path, index=False)
    except PermissionError:
        path = path.with_name(f"{path.stem}.{uuid4().hex[:8]}{path.suffix}")
        frame.to_csv(path, index=False)
        print(f"Output locked; saved table as {path.name}", flush=True)
    return path


def plot_day_fans(day, frame, output, horizon):
    """All 24 hourly origins, showing the individual 1..8h forecast fan."""
    fig, axes = plt.subplots(6, 4, figsize=(18, 22), sharex=True, sharey=True)
    axes = axes.ravel()
    y = np.concatenate([frame["I_observed"].to_numpy(), frame["lower_95"].to_numpy(),
        frame["upper_95"].to_numpy()])
    ymin, ymax = np.quantile(y, [0.002, 0.998])
    margin = 0.06 * max(ymax - ymin, 1e-5)
    by_origin = {int(origin.hour): part.sort_values("lead_hours")
        for origin, part in frame.groupby("origin", sort=True)}
    for hour, ax in enumerate(axes):
        ax.set_title(f"{hour:02d}:00 origin", fontsize=10)
        if hour not in by_origin:
            ax.text(0.5, 0.5, "No complete forecast", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
            continue
        part = by_origin[hour]
        h = part["lead_hours"].to_numpy(dtype=float)
        ax.fill_between(h, part["lower_95"].to_numpy(), part["upper_95"].to_numpy(),
            alpha=0.16, label="95% interval")
        ax.fill_between(h, part["lower_68"].to_numpy(), part["upper_68"].to_numpy(),
            alpha=0.3, label="68% interval")
        ax.plot(h, part["I_mpc"].to_numpy(), linewidth=1.3, label="WM4 mean")
        ax.plot(h, part["I_observed"].to_numpy(), marker="o", markersize=2.6,
            linewidth=1.05, label="Observed")
        ax.set_xlim(1, horizon)
        ax.set_xticks(np.arange(1, horizon + 1, 2))
        ax.grid(alpha=0.2)
    for ax in axes[-4:]: ax.set_xlabel("Lead (hours)")
    for ax in axes[::4]: ax.set_ylabel("Normalized demand")
    for ax in axes: ax.set_ylim(ymin - margin, ymax + margin)
    handles, labels = next(ax.get_legend_handles_labels() for ax in axes if ax.get_legend_handles_labels()[0])
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.992))
    fig.suptitle(f"WM4 MPC | every receding {horizon}h prediction interval | {day:%Y-%m-%d}",
        y=1.008)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    return _save_fig(fig, output)


def plot_rolling_one_hour(day, frame, output):
    part = frame[frame["lead_hours"] == 1].sort_values("target_time")
    t = part["target_time"].to_numpy()
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.fill_between(t, part["lower_95"], part["upper_95"], alpha=0.14, label="95%")
    ax.fill_between(t, part["lower_68"], part["upper_68"], alpha=0.30, label="68%")
    ax.plot(t, part["I_mpc"], marker=".", label="Rolling 1h predicted mean")
    ax.plot(t, part["I_observed"], marker="o", markersize=3, label="Observed")
    ax.set_xlabel("Target time"); ax.set_ylabel("Normalized demand")
    ax.set_title(f"WM4 MPC | hourly receding 1h predictions | {day:%Y-%m-%d}")
    ax.grid(alpha=0.25); ax.legend(ncol=4)
    fig.autofmt_xdate(); fig.tight_layout()
    return _save_fig(fig, output)


def plot_diffusion(exact, cumulative, output, *, covariance_mode="legacy/unspecified"):
    """Separate variance accumulated by the SDE from 1..H pooled forecast scores."""
    h = exact["lead_hours"].to_numpy(dtype=int)
    initial = exact["mean_initial_variance"].to_numpy(dtype=float)
    process = exact["mean_process_variance"].to_numpy(dtype=float)
    total = exact["mean_predicted_variance"].to_numpy(dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
    F = exact["mean_initial_F_component_variance"].to_numpy(dtype=float)
    R = exact["mean_initial_R_component_variance"].to_numpy(dtype=float)
    if np.all(np.isfinite(F)) and np.all(np.isfinite(R)):
        axes[0].fill_between(h, 0, F, alpha=0.65, label="Historical F0 variance contribution")
        axes[0].fill_between(h, F, F + R, alpha=0.55, label="Baseline-derived R0 variance contribution")
    else:
        axes[0].fill_between(h, 0, initial, alpha=0.65, label="Propagated initial-state variance")
    axes[0].fill_between(h, initial, initial + process, alpha=0.6, label="Cumulative process (Wiener) variance")
    axes[0].plot(h, exact["mse"], marker="o", linestyle="--", label="Observed MSE")
    axes[0].set_ylabel("Variance"); axes[0].set_title("Variance budget at exact prediction lead")
    axes[1].plot(h, np.sqrt(initial), marker="o", label="Initial-state RMS std")
    axes[1].plot(h, np.sqrt(process), marker="o", label="Diffusion RMS std")
    axes[1].plot(h, np.sqrt(total), marker="o", label="Combined predicted RMS std")
    axes[1].plot(h, exact["rmse"], marker="s", linestyle="--", label="Observed RMSE")
    axes[1].set_ylabel("Normalized demand"); axes[1].set_title("Uncertainty growth after MPC initialization")
    c = cumulative
    axes[2].plot(h, c["initial_rms_std"], marker="o", label="Initial-state contribution")
    axes[2].plot(h, c["process_rms_std"], marker="o", label="Diffusion contribution")
    axes[2].plot(h, c["predicted_rms_std"], marker="o", label="Predicted RMS std, leads 1..H")
    axes[2].plot(h, c["rmse"], marker="s", linestyle="--", label="Observed RMSE, leads 1..H")
    axes[2].set_ylabel("Normalized demand")
    axes[2].set_title("Cumulative forecast window: pooled 1..H variance and error")
    for ax in axes:
        ax.grid(alpha=0.25); ax.legend(loc="best")
        ax.set_xticks(h)
    axes[2].set_xlabel("Horizon H (hours)")
    if not (np.all(np.isfinite(F)) and np.all(np.isfinite(R))):
        fig.suptitle("WM4 | LEGACY covariance data (F/R decomposition unavailable)")
    else:
        fig.suptitle(f"WM4 | initial covariance and accumulated stochastic diffusion | R0: {covariance_mode}")
    fig.tight_layout()
    return _save_fig(fig, output)


def plot_coverage(exact, cumulative, output):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, table, title in ((axes[0], exact, "Exact-lead interval coverage"),
                             (axes[1], cumulative, "Cumulative 1..H interval coverage")):
        h = table["lead_hours"] if "lead_hours" in table else table["horizon_hours"]
        ax.plot(h, table["coverage_68_pct"], marker="o", label="Empirical 68%")
        ax.plot(h, table["coverage_95_pct"], marker="o", label="Empirical 95%")
        ax.axhline(68.27, linestyle=":", label="Gaussian 68.27% target")
        ax.axhline(95.0, linestyle=":", label="95% target")
        ax.set_ylim(0, 102); ax.set_ylabel("Empirical coverage (%)"); ax.set_title(title)
        ax.grid(alpha=0.2); ax.legend(ncol=2)
    axes[-1].set_xticks(exact["lead_hours"])
    axes[-1].set_xlabel("Lead / cumulative horizon (hours)")
    fig.tight_layout()
    return _save_fig(fig, output)


def plot_hour_lead_heatmaps(by_hour, output, horizon):
    """Show where undercoverage or underdispersed diffusion is concentrated."""
    plots = (("coverage_95_pct", "95% interval coverage (%)", 0.0, 100.0),
             ("predicted_rms_std_to_rmse", "Predicted RMS std / observed RMSE", 0.0, None),
             ("process_variance_share_pct", "Diffusion share of predicted variance (%)", 0.0, 100.0))
    fig, axes = plt.subplots(1, 3, figsize=(19, 8), sharey=True)
    for ax, (field, label, vmin, vmax) in zip(axes, plots):
        matrix = by_hour.pivot(index="origin_hour", columns="lead_hours", values=field)
        matrix = matrix.reindex(index=range(24), columns=range(1, horizon + 1))
        image = ax.imshow(matrix.to_numpy(dtype=float), origin="lower", aspect="auto",
            interpolation="none", vmin=vmin, vmax=vmax)
        ax.set_title(label); ax.set_xlabel("Forecast lead (h)")
        ax.set_xticks(range(horizon)); ax.set_xticklabels(range(1, horizon + 1))
        ax.set_yticks(range(0, 24, 2))
        fig.colorbar(image, ax=ax, shrink=0.75)
    axes[0].set_ylabel("MPC origin hour")
    fig.suptitle("WM4 MPC | calibration and diffusion across origin time and lead")
    fig.tight_layout()
    return _save_fig(fig, output)


def plot_sigma(origins, daily, output):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    axes[0].scatter(origins["origin"], origins["diffusion_sigma"], s=5, alpha=0.12,
        label="Origin-level fitted sigma")
    dates = daily["origin_day"].to_numpy()
    axes[0].plot(dates, daily["sigma_median"], linewidth=1.3, label="Daily median")
    axes[0].fill_between(dates, daily["sigma_q25"], daily["sigma_q75"], alpha=0.25,
        label="Daily 25th–75th percentile")
    axes[0].set_ylabel("Fitted Wiener diffusion sigma")
    axes[0].set_title("Historical diffusion estimate at successive MPC origins")
    axes[0].grid(alpha=0.25); axes[0].legend()
    hourly = origins.groupby("origin_hour")["diffusion_sigma"].agg(
        mean="mean", median="median", q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)).reset_index()
    h = hourly["origin_hour"].to_numpy()
    axes[1].fill_between(h, hourly["q25"], hourly["q75"], alpha=0.25,
        label="25th–75th percentile")
    axes[1].plot(h, hourly["median"], marker="o", label="Median fitted sigma")
    axes[1].set_xticks(range(0, 24, 2)); axes[1].set_xlabel("MPC origin hour")
    axes[1].set_ylabel("Fitted Wiener diffusion sigma")
    axes[1].set_title("Distribution of diffusion estimates by hourly update")
    axes[1].grid(alpha=0.25); axes[1].legend()
    fig.tight_layout()
    return _save_fig(fig, output)


def analyze(path=FORECASTS, output=OUTPUT, day=None, horizon=8,
    *, expected_r0_variance_mode="baseline_residual", allow_legacy=False):
    path, output = Path(path).resolve(), Path(output)
    frame, incomplete, mode = load_forecasts(path, horizon,
        expected_r0_variance_mode=expected_r0_variance_mode, allow_legacy=allow_legacy)
    r0_mode = str(frame["r0_variance_mode"].iloc[0]) if "r0_variance_mode" in frame else "legacy/unspecified"
    print(f"Loaded WM4 forecasts: {path} | last modified: "
          f"{datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M:%S} (system local) | "
          f"sigma_fit_mode={mode} | r0_variance_mode={r0_mode}", flush=True)
    horizon = int(horizon)
    exact, cumulative, by_hour, origins, daily = summaries(frame, horizon)
    day, selected = selected_day(frame, day)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "exact_lead_metrics.csv": exact,
        "cumulative_horizon_metrics.csv": cumulative,
        "origin_hour_lead_metrics.csv": by_hour,
        "origin_diffusion_estimates.csv": origins,
        "daily_diffusion_summary.csv": daily,
        f"day_{day:%Y-%m-%d}_forecast_intervals.csv": selected,
    }
    saved = [_save_csv(table, output / name) for name, table in tables.items()]
    saved += [
        plot_day_fans(day, selected, output / f"day_{day:%Y-%m-%d}_{horizon}h_fans.png", horizon),
        plot_rolling_one_hour(day, selected, output / f"day_{day:%Y-%m-%d}_rolling_1h.png"),
        plot_diffusion(exact, cumulative, output / "diffusion_variance_by_horizon.png",
            covariance_mode=r0_mode),
        plot_coverage(exact, cumulative, output / "interval_coverage_by_horizon.png"),
        plot_hour_lead_heatmaps(by_hour, output / "origin_hour_lead_heatmaps.png", horizon),
        plot_sigma(origins, daily, output / "fitted_sigma_over_time.png"),
    ]
    summary = {
        "source": str(path), "sigma_fit_mode": mode, "r0_variance_mode": r0_mode,
        "source_modified_timestamp": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        "forecast_horizon_hours": horizon,
        "matched_origins": int(origins.shape[0]), "paired_forecast_rows": int(len(frame)),
        "days": int(frame["origin_day"].nunique()), "incomplete_origins_discarded": incomplete,
        "example_day": day.strftime("%Y-%m-%d"),
        "variance_definition": "predicted_variance = initial_variance + process_variance, where the process component accumulates stochastic forcing since each MPC origin",
        "cumulative_definition": "1..H scores pool leads 1 through H; variances are averaged, not added across forecast leads",
        "exact_horizon": exact.iloc[-1].to_dict(), "cumulative_horizon": cumulative.iloc[-1].to_dict(),
    }
    info = output / "summary.json"
    try: info.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    except PermissionError:
        info = output / f"summary.{uuid4().hex[:8]}.json"
        info.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(f"WM4 MPC {horizon}h interval analysis | sigma mode={mode} | R0 mode={r0_mode} | "
          f"origins={len(origins)} | days={frame['origin_day'].nunique()} | "
          f"discarded incomplete origins={incomplete}")
    print(f"{'Lead':>4} {'RMSE':>10} {'pred std':>10} {'diff std':>10} {'68%cov':>9} {'95%cov':>9}")
    if "initial_R_variance" in frame:
        print(f"Mean origin baseline-derived R0 std: "
              f"{np.sqrt(frame.drop_duplicates('origin')['initial_R_variance'].mean()):.6f}")
    for row in exact.itertuples(index=False):
        print(f"{row.lead_hours:>4d} {row.rmse:>10.6f} {row.predicted_rms_std:>10.6f} "
              f"{row.process_rms_std:>10.6f} {row.coverage_68_pct:>8.1f}% {row.coverage_95_pct:>8.1f}%")
    last = cumulative.iloc[-1]
    print(f"cumulative 1..{horizon}: RMSE={last['rmse']:.6f}, "
          f"predicted RMS std={last['predicted_rms_std']:.6f}, "
          f"coverage68={last['coverage_68_pct']:.1f}%, "
          f"coverage95={last['coverage_95_pct']:.1f}%")
    print(f"Saved {len(saved)} tables/figures and summary to {output}")
    return {"exact": exact, "cumulative": cumulative, "by_hour": by_hour,
        "origins": origins, "daily": daily, "example_day": day, "output": output}


def main():
    parser = ArgumentParser(description="Analyze saved WM4 1..8h MPC intervals, calibration and accumulated diffusion variance; no refitting.")
    parser.add_argument("--forecasts", type=Path, default=FORECASTS,
        help="CSV written by wm_4.predict_all_days_mpc")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--day", help="Day for 24 receding forecast fans; defaults to first complete day.")
    parser.add_argument("--horizon", type=int, default=8, help="Max lead horizon (1..8).")
    parser.add_argument("--expected-r0-variance-mode", choices=("baseline_residual", "zero", "any"),
        default="baseline_residual", help="Reject a CSV produced using a different R0 covariance assumption.")
    parser.add_argument("--allow-legacy", action="store_true",
        help="Permit plotting pre-update CSVs, explicitly labeled legacy; does not recompute covariance.")
    args = parser.parse_args()
    return analyze(args.forecasts, args.output, args.day, args.horizon,
        expected_r0_variance_mode=args.expected_r0_variance_mode, allow_legacy=args.allow_legacy)


if __name__ == "__main__": main()
