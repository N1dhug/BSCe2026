"""WM4: fit constant Wiener q on historical horizons 2..8h; evaluate 4h MPC.

24h deterministic WM4 fitting; 25 complete previous days' corresponding-hour
MPC paths; no initial-state covariance propagation. Every candidate horizon
H fits an independent scalar QP using historical errors h=1..H, but scores the
SAME current-origin forecasts at leads h=1..4.

If a saved 24h MPC calibration archive and six-hour MPC forecast CSV are
available, reuse them without refitting WM4. Older 6..24h sweep outputs did
not save the 24h calibration paths; in that case run the MPC ONCE and persist
both files so subsequent experiments can reuse them. Never reuse saved q.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data
from src.exe.run_model.wm_4.predict_all_days_optimized import START_DAY, END_DAY
from src.exe.run_model.wm_4.predict_all_days_mpc import ROOT, run_all_days
from src.exe.run_model.wm_4.predict_all_days_mpc_no_p0 import MODE
from src.exe.run_model.wm_4.sweep_mpc_sigma_fit_hours_no_p0 import (
    ARCHIVE_HOURS, BACKLOG_DAYS, fit_diffusion_horizons, _score,
)


EVALUATION_HOURS = 4
MIN_HOURS = 2
MAX_HOURS = 8
SOURCE = ROOT / "files/output/wm_4/prediction/sigma_fit_hours_6_24_no_p0"
OUTPUT = ROOT / "files/output/wm_4/prediction/sigma_fit_hours_2_8_eval4_no_p0"
FORECAST_FILE = "mpc_forecasts.csv"
ARCHIVE_FILE = "mpc_calibration_24h.csv"


def save_archive(archive, path):
    """Persist every genuine 24-lead historical error / unit Wiener kernel."""
    rows = []
    for origin, matrix in sorted(archive.items()):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (ARCHIVE_HOURS, 3):
            raise ValueError(f"Expected 24 MPC calibration leads at {origin}.")
        if not np.isfinite(matrix).all() or not np.allclose(matrix[:, 1], 0., atol=1e-12):
            raise ValueError(f"Invalid/no-P0 calibration trajectory at {origin}.")
        for h, (err, initial, kernel) in enumerate(matrix, 1):
            rows.append({"origin": origin, "lead_hours": h,
                "forecast_error": float(err), "initial_variance": float(initial),
                "process_kernel_unit": float(kernel)})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["origin", "lead_hours", "forecast_error",
        "initial_variance", "process_kernel_unit"]).to_csv(path, index=False)
    return path


def load_archive(path):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["origin"])
    needed = {"origin", "lead_hours", "forecast_error", "initial_variance",
        "process_kernel_unit"}
    if missing := needed - set(frame):
        raise ValueError(f"Missing historical MPC calibration fields: {sorted(missing)}")
    if frame.duplicated(["origin", "lead_hours"]).any():
        raise ValueError("Duplicate origin/lead in saved historical MPC archive.")
    archive = {}
    for origin, part in frame.groupby("origin", sort=True):
        part = part.sort_values("lead_hours")
        if not np.array_equal(part["lead_hours"].to_numpy(dtype=int),
            np.arange(1, ARCHIVE_HOURS + 1)):
            raise ValueError(f"Incomplete historical 24-hour MPC path at {origin}.")
        matrix = part[["forecast_error", "initial_variance",
            "process_kernel_unit"]].to_numpy(dtype=float)
        if not np.isfinite(matrix).all() or np.any(matrix[:, 2] < 0.):
            raise ValueError(f"Invalid Wiener unit kernel at {origin}.")
        if not np.allclose(matrix[:, 1], 0., atol=1e-12):
            raise ValueError(f"Saved historical MPC path includes propagated P0 at {origin}.")
        archive[pd.Timestamp(origin)] = matrix
    if not archive: raise ValueError("No saved 24-hour MPC calibration paths.")
    return archive


def four_hour_evaluation(forecasts):
    """Use four actual MPC forecast leads; original CSV can contain six leads."""
    required = {"origin", "lead_hours", "I_observed", "I_mpc", "initial_variance",
        "process_kernel_unit", "parameter_fit_hours", "sigma_fit_mode"}
    if missing := required - set(forecasts):
        raise ValueError(f"Missing MPC evaluation fields: {sorted(missing)}")
    data = forecasts.copy()
    data["origin"] = pd.to_datetime(data["origin"])
    if not (pd.to_numeric(data["parameter_fit_hours"]) == DAY_HOURS).all():
        raise ValueError("MPC deterministic parameter fit must span 24 hours.")
    if not (data["sigma_fit_mode"] == MODE).all():
        raise ValueError("Expected six-hour no-P0 baseline MPC mode.")
    data = data.loc[data["lead_hours"].between(1, EVALUATION_HOURS)].copy()
    if data.duplicated(["origin", "lead_hours"]).any():
        raise ValueError("Duplicate evaluation origin/forecast lead.")
    if not np.allclose(data["initial_variance"], 0., atol=1e-12):
        raise ValueError("Current four-hour MPC evaluation includes P0 propagation.")
    evaluations = {}
    wanted = np.arange(1, EVALUATION_HOURS + 1)
    for origin, part in data.groupby("origin", sort=True):
        part = part.sort_values("lead_hours")
        if not np.array_equal(part["lead_hours"].to_numpy(dtype=int), wanted):
            continue
        err = (part["I_observed"] - part["I_mpc"]).to_numpy(dtype=float)
        kernel = part["process_kernel_unit"].to_numpy(dtype=float)
        if not np.isfinite(err).all() or not np.isfinite(kernel).all() or (kernel < 0).any():
            raise ValueError(f"Invalid four-hour error or Wiener kernel at {origin}.")
        evaluations[pd.Timestamp(origin)] = np.column_stack((err, kernel))
    if not evaluations: raise ValueError("No complete four-hour MPC evaluation origins.")
    return evaluations


def fit_per_origin(archive, evaluations, history_days=BACKLOG_DAYS,
    min_hours=MIN_HOURS, max_hours=MAX_HOURS):
    """Causally refit independent constant-q estimates for every H and origin."""
    history_days = int(history_days)
    if history_days < 2: raise ValueError("At least two historical days are required.")
    if not 2 <= min_hours <= max_hours <= ARCHIVE_HOURS:
        raise ValueError("Require 2 <= min_hours <= max_hours <= 24.")
    rows = []
    for origin in sorted(evaluations):
        history = []
        for lag in range(1, history_days + 1):
            old = origin - pd.Timedelta(days=lag)
            past = archive.get(old)
            if past is None or old + pd.Timedelta(hours=ARCHIVE_HOURS) > origin: break
            past = np.asarray(past, dtype=float)
            if past.shape != (ARCHIVE_HOURS, 3):
                raise ValueError(f"Invalid 24-hour historical MPC path at {old}.")
            history.append(past)
        if len(history) != history_days: continue
        horizons, qs, _, _ = fit_diffusion_horizons(
            np.asarray(history), min_hours=min_hours, max_hours=max_hours)
        actual, current_kernel = evaluations[origin].T
        for horizon, q in zip(horizons, qs):
            v = q * current_kernel
            row = {"origin": origin, "calibration_hours": int(horizon),
                "evaluation_hours": EVALUATION_HOURS,
                "history_days": history_days, "n_history_mpcs": history_days,
                "diffusion_q": float(q), "diffusion_sigma": float(np.sqrt(q)),
                "observed_mse": float(np.mean(actual ** 2)),
                "predicted_variance": float(np.mean(v))}
            row.update({f"variance_h{h}": float(v[h - 1])
                for h in range(1, EVALUATION_HOURS + 1)})
            rows.append(row)
    if not rows:
        raise ValueError(f"No four-hour origin has {history_days} completed "
            "historical 24-hour MPC trajectories.")
    return pd.DataFrame(rows)


def summarize(fits, evaluations, quantiles=5):
    """Compare H=2..8 over identical four-hour MPC forecast errors/origins."""
    quantiles = int(quantiles)
    if quantiles < 2: raise ValueError("Need at least two uncertainty quantiles.")
    if fits["origin"].nunique() < quantiles:
        raise ValueError("Too few complete MPC origins for uncertainty quantiles.")
    overall, by_lead, by_quantile = [], [], []
    variance_columns = [f"variance_h{h}" for h in range(1, EVALUATION_HOURS + 1)]
    first_origins = None
    for horizon, group in fits.groupby("calibration_hours", sort=True):
        group = group.sort_values("origin").reset_index(drop=True)
        origins = group["origin"].tolist()
        if first_origins is None: first_origins = origins
        if origins != first_origins:
            raise ValueError("Fitting horizons do not share the same evaluation origins.")
        errors = np.stack([evaluations[o][:, 0] for o in origins])
        variance = group[variance_columns].to_numpy(dtype=float)
        q = group["diffusion_q"].to_numpy(dtype=float)
        overall.append({"calibration_hours": int(horizon),
            **_score(errors, variance, len(origins), q)})
        for h in range(EVALUATION_HOURS):
            by_lead.append({"calibration_hours": int(horizon), "lead_hours": h + 1,
                **_score(errors[:, h], variance[:, h], len(origins), q)})
        order = pd.Series(variance.mean(axis=1)).rank(method="first")
        bins = pd.qcut(order, quantiles, labels=False).astype(int).to_numpy() + 1
        for index in range(1, quantiles + 1):
            use = bins == index
            by_quantile.append({"calibration_hours": int(horizon),
                "uncertainty_quantile": index,
                **_score(errors[use], variance[use], int(use.sum()), q[use])})
    return pd.DataFrame(overall), pd.DataFrame(by_quantile), pd.DataFrame(by_lead)


def save_figures(fits, scores, quantiles, leads, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hours = scores["calibration_hours"].to_numpy(dtype=int)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0, 0].plot(hours, scores["observed_rmse"], "o--", label="Observed 4h RMSE")
    ax[0, 0].plot(hours, scores["predicted_rms_std"], "o-", label="Predicted RMS std")
    ax[0, 0].set_ylabel("Normalized demand")
    ax[0, 0].legend()
    ax[0, 1].plot(hours, scores["predicted_to_observed_mse"], "o-")
    ax[0, 1].axhline(1., color="black", ls="--")
    ax[0, 1].set_ylabel("Predicted variance / observed MSE")
    ax[1, 0].plot(hours, scores["coverage_68_pct"], "o-", label="±1σ coverage")
    ax[1, 0].plot(hours, scores["coverage_95_pct"], "o-", label="±1.96σ coverage")
    ax[1, 0].axhline(68.27, ls="--", alpha=.6)
    ax[1, 0].axhline(95., ls="--", alpha=.6)
    ax[1, 0].legend()
    ax[1, 0].set_ylabel("Empirical coverage (%)")
    ax[1, 1].plot(hours, scores["gaussian_nll"], "o-")
    ax[1, 1].set_ylabel("Gaussian NLL (lower better)")
    for a in ax.flat:
        a.set_xlabel("Diffusion fitting horizon (hours)")
        a.set_xticks(hours)
        a.grid(alpha=.2)
    fig.suptitle("WM4 constant Wiener | 25-day backlog | 24h deterministic fit | "
        f"4h evaluation | {int(scores['n_origins'].iloc[0])} identical origins | no P0")
    fig.tight_layout()
    fig.savefig(output / "4h_evaluation_fit_2_8_sweep.png", dpi=175, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for quintile, group in quantiles.groupby("uncertainty_quantile", sort=True):
        group = group.sort_values("calibration_hours")
        ax[0].plot(group["calibration_hours"], group["predicted_to_observed_mse"],
            "o-", label=f"Q{quintile}")
        ax[1].plot(group["calibration_hours"], group["coverage_95_pct"],
            "o-", label=f"Q{quintile}")
    ax[0].axhline(1., color="black", ls="--")
    ax[0].set_ylabel("Predicted variance / observed MSE")
    ax[1].axhline(95., color="black", ls="--")
    ax[1].set_ylabel("95% interval coverage (%)")
    for a in ax:
        a.set_xlabel("Diffusion fitting horizon (hours)")
        a.set_xticks(hours)
        a.grid(alpha=.2)
        a.legend()
    fig.suptitle("WM4 | 4h MPC evaluation | own predicted-uncertainty quintiles")
    fig.tight_layout()
    fig.savefig(output / "4h_evaluation_quintiles.png", dpi=175, bbox_inches="tight")
    plt.close(fig)

    table = leads.pivot(index="lead_hours", columns="calibration_hours",
        values="predicted_to_observed_mse").sort_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    heat = ax.imshow(table.to_numpy(), origin="lower", aspect="auto", vmin=.25, vmax=2.)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set(xlabel="Diffusion fitting horizon (hours)", ylabel="MPC evaluation lead (hours)",
        title="Variance calibration: 4h MPC evaluation, 2–8h diffusion fit")
    fig.colorbar(heat, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "4h_evaluation_lead_calibration.png", dpi=175,
        bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    high = 1.05 * max(float(fits["observed_mse"].max()),
        float(fits["predicted_variance"].max()), 1e-8)
    for a, horizon in zip(ax.flat, hours):
        group = fits.loc[fits["calibration_hours"] == horizon]
        a.scatter(group["predicted_variance"], group["observed_mse"], s=11, alpha=.55)
        a.plot([0, high], [0, high], color="black", ls="--", lw=.8)
        a.set(xlim=(0, high), ylim=(0, high), title=f"{horizon}-hour diffusion fit")
        a.grid(alpha=.2)
    for a in list(ax.flat)[len(hours):]: a.axis("off")
    fig.supxlabel("Predicted mean variance over 4h")
    fig.supylabel("Observed mean squared error over 4h")
    fig.suptitle("WM4 | matched-origin predicted vs realized 4h MPC variance")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / "4h_evaluation_scatter.png", dpi=175, bbox_inches="tight")
    plt.close(fig)


def _find_saved(source, output):
    for folder in (Path(output), Path(source)):
        if (folder / FORECAST_FILE).is_file() and (folder / ARCHIVE_FILE).is_file():
            forecasts = pd.read_csv(folder / FORECAST_FILE, parse_dates=["origin"])
            archive = load_archive(folder / ARCHIVE_FILE)
            return forecasts, archive, folder
    return None


def run(frame=None, start=START_DAY, end=END_DAY, *, source=SOURCE,
    steps_per_hour=60, baseline_weeks=BASELINE_WEEKS,
    history_days=BACKLOG_DAYS, min_hours=MIN_HOURS, max_hours=MAX_HOURS,
    quantiles=5, output=OUTPUT, reuse_only=False, fail_fast=False):
    """Load already fitted 24h MPC trajectories or generate/cache them ONCE."""
    min_hours, max_hours, history_days, quantiles = (int(min_hours),
        int(max_hours), int(history_days), int(quantiles))
    if not MIN_HOURS <= min_hours <= max_hours <= ARCHIVE_HOURS:
        raise ValueError("Require 2 <= min_hours <= max_hours <= 24.")
    if history_days < 2 or quantiles < 2:
        raise ValueError("Need >=2 historical MPC days and >=2 quantiles.")
    output = Path(output)
    cached = _find_saved(source, output)
    if cached is not None:
        forecasts, archive, used = cached
        print(f"Reusing saved MPC forecasts and 24h historical archive from {used}")
    else:
        if reuse_only:
            raise FileNotFoundError("No saved 24h MPC calibration archive. "
                "Rerun the 6–24h experiment after updating the code, or omit --reuse-only.")
        if frame is None: frame, _ = aligned_wm4_data("MED")
        archive = {}
        forecasts, states, skipped = run_all_days(frame, start=start, end=end,
            steps_per_hour=steps_per_hour, baseline_weeks=baseline_weeks,
            max_horizon=6, fit_hours=DAY_HOURS,
            sigma_fit_mode=MODE, sigma_rolling_days=history_days,
            mpc_archive=archive, archive_horizon=ARCHIVE_HOURS,
            fail_fast=fail_fast)
        output.mkdir(parents=True, exist_ok=True)
        forecasts.to_csv(output / FORECAST_FILE, index=False)
        states.to_csv(output / "mpc_states.csv", index=False)
        save_archive(archive, output / ARCHIVE_FILE)
        if skipped:
            pd.DataFrame([{"origin": origin, "reason": reason}
                for origin, reason in skipped]).to_csv(output / "skipped_origins.csv",
                index=False)
    evaluations = four_hour_evaluation(forecasts)
    fits = fit_per_origin(archive, evaluations, history_days=history_days,
        min_hours=min_hours, max_hours=max_hours)
    scores, quantile_table, lead_table = summarize(fits, evaluations, quantiles=quantiles)
    output.mkdir(parents=True, exist_ok=True)
    for key, value in {"fits": fits, "summary": scores, "quantiles": quantile_table,
        "leads": lead_table}.items():
        value.to_csv(output / f"{key}.csv", index=False)
    save_figures(fits, scores, quantile_table, lead_table, output)
    print(f"4h evaluation: {fits['origin'].nunique()} identical MPC origins; "
        f"{len(fits)} fresh scalar Wiener QP fits using {history_days} historical days.")
    print(scores[["calibration_hours", "n_origins", "predicted_to_observed_mse",
        "coverage_68_pct", "coverage_95_pct", "gaussian_nll", "mean_sigma"]].to_string(
        index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"Results: {output}")
    return {"fits": fits, "summary": scores, "quantiles": quantile_table,
        "leads": lead_table, "evaluations": evaluations, "archive": archive,
        "output": output}


def main():
    parser = ArgumentParser(description="Evaluate four-hour WM4 MPC variance using separate Wiener diffusion fits over 2..8 historical MPC leads; 25-day backlog, no P0.")
    parser.add_argument("--source", type=Path, default=SOURCE,
        help="Directory with saved mpc_forecasts.csv and mpc_calibration_24h.csv.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--history-days", type=int, default=BACKLOG_DAYS)
    parser.add_argument("--min-hours", type=int, default=MIN_HOURS)
    parser.add_argument("--max-hours", type=int, default=MAX_HOURS)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--reuse-only", action="store_true",
        help="Require a saved 24h MPC calibration archive; do not run WM4.")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    return run(start=args.start_day, end=args.end_day, source=args.source,
        output=args.output, steps_per_hour=args.steps_per_hour,
        baseline_weeks=args.baseline_weeks, history_days=args.history_days,
        min_hours=args.min_hours, max_hours=args.max_hours,
        quantiles=args.quantiles, reuse_only=args.reuse_only,
        fail_fast=args.fail_fast)


if __name__ == "__main__": main()
