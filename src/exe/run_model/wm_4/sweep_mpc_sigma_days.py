"""Replay WM4 six-hour Wiener fits for every historical window 7..28 days.

Fit WM4 deterministic parameters, initial states and baseline ONCE using
predict_all_days_mpc, then replay only the scalar diffusion variance QP from
saved mpc_forecasts.csv. No deterministic refitting and no second simulation.

Reports both all eligible fits for each window and the SAME fixed-origin
cohort with full 28-day histories for fair across-window comparisons.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "files/output/wm_4/prediction/all_days_mpc_sigma_24d_6h/mpc_forecasts.csv"
OUTPUT = ROOT / "files/output/wm_4/prediction/sigma_days_sweep"
HORIZON = 6
RIDGE = 1e-10  # Matches wm_4._FIT_RIDGE and wm_4._least_squares_qp.
COLUMNS = ("forecast_error", "initial_variance", "process_kernel_unit")


def read_mpc_archive(path, horizon=HORIZON, fit_hours=24):
    """Read complete, timestamp-aligned forward MPC trajectories from a single CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No archived WM4 forecasts at {path}. Run predict_all_days_mpc first.")
    frame = pd.read_csv(path)
    required = {"origin", "lead_hours", "target_time", "I_observed", "I_mpc",
        "initial_variance", "process_kernel_unit"}
    if missing := required - set(frame):
        raise ValueError(f"Input MPC forecasts missing fields: {sorted(missing)}.")
    if "parameter_fit_hours" in frame:
        hours = pd.to_numeric(frame["parameter_fit_hours"], errors="raise")
        if not (hours == fit_hours).all():
            raise ValueError(f"Saved WM4 parameters were not all fitted for {fit_hours}h.")
    frame["origin"] = pd.to_datetime(frame["origin"])
    frame["target_time"] = pd.to_datetime(frame["target_time"])
    if frame["origin"].dt.tz is not None or frame["target_time"].dt.tz is not None:
        raise ValueError("MPC timestamps must be timezone-naive.")
    frame["lead_hours"] = pd.to_numeric(frame["lead_hours"], errors="raise").astype(int)
    if frame.duplicated(["origin", "lead_hours"]).any():
        raise ValueError("Duplicate origin / horizon in source predictions.")
    frame = frame.loc[frame["lead_hours"].between(1, int(horizon))].copy()
    for col in ("I_observed", "I_mpc", "initial_variance", "process_kernel_unit"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.all(np.isfinite(frame[["I_observed", "I_mpc", "initial_variance",
        "process_kernel_unit"]].to_numpy(dtype=float))):
        raise ValueError("MPC errors, P0 propagation and Wiener kernels must be finite.")
    if (frame[["initial_variance", "process_kernel_unit"]] < 0).any().any():
        raise ValueError("P0 propagation and Wiener kernels must be nonnegative.")
    frame["forecast_error"] = frame["I_observed"] - frame["I_mpc"]
    if not (frame["target_time"] == frame["origin"] +
        pd.to_timedelta(frame["lead_hours"], unit="h")).all():
        raise ValueError("MPC origin/target timestamps do not match exact forecast leads.")

    archive = {}
    wanted = np.arange(1, int(horizon) + 1)
    for origin, group in frame.groupby("origin", sort=True):
        group = group.sort_values("lead_hours")
        if not np.array_equal(group["lead_hours"].to_numpy(), wanted): continue
        archive[pd.Timestamp(origin)] = group[list(COLUMNS)].to_numpy(dtype=float)
    if not archive: raise ValueError("No complete six-hour MPC trajectories.")
    return archive, frame


def fit_window_variances(history, min_days=7, max_days=28, ridge=RIDGE):
    """Vectorized closed-form nonnegative scalar WM4 diffusion QP.

    History is newest first, shape (max_days,6,3), columns e, B, K.
    The QP uses sample variance across days, mean historical P0 variance,
    mean historical unit Wiener kernel, and the same tiny ridge as WM4.
    """
    history = np.asarray(history, dtype=float)
    min_days, max_days = int(min_days), int(max_days)
    if history.ndim != 3 or history.shape[1:] != (HORIZON, 3):
        raise ValueError("Expected historical MPC samples with shape (days,6,3).")
    if not 2 <= min_days <= max_days <= len(history):
        raise ValueError("Invalid fitting window range for historical MPC samples.")
    errors = history[:, :, 0]
    count = np.arange(1, len(history) + 1, dtype=float)[:, None]
    csum = np.cumsum(errors, axis=0)
    csq = np.cumsum(errors ** 2, axis=0)
    cB = np.cumsum(history[:, :, 1], axis=0)
    cK = np.cumsum(history[:, :, 2], axis=0)
    ns = np.arange(min_days, max_days + 1)
    ix = ns - 1
    S = np.maximum((csq[ix] - csum[ix] ** 2 / count[ix]) / (count[ix] - 1), 0.0)
    B = cB[ix] / count[ix]
    K = cK[ix] / count[ix]
    numerator = np.sum(K * (S - B), axis=1)
    denominator = np.sum(K ** 2, axis=1) + float(ridge) * HORIZON / 2.0
    q = np.maximum(numerator / np.maximum(denominator, np.finfo(float).tiny), 0.0)
    return ns, q, S, B, K


def replay(archive, min_days=7, max_days=28):
    """Sweep every eligible origin, keeping all complete six-hour predictions."""
    min_days, max_days = int(min_days), int(max_days)
    if not 2 <= min_days <= max_days:
        raise ValueError("Require 2 <= min_days <= max_days.")
    rows = []
    deltas = [pd.Timedelta(days=k) for k in range(1, max_days + 1)]
    for origin in sorted(archive):
        history = []
        for delta in deltas:
            old_origin = origin - delta
            historical = archive.get(old_origin)
            # Every used historical six-hour target must precede the origin.
            if historical is None or old_origin + pd.Timedelta(hours=HORIZON) > origin:
                break
            history.append(historical)
        if len(history) < min_days: continue
        ns, qs, _, _, _ = fit_window_variances(
            np.asarray(history), min_days=min_days, max_days=len(history))
        current = archive[origin]
        for n, q in zip(ns, qs):
            row = {"origin": origin, "history_days": int(n), "n_history_mpcs": int(n),
                "diffusion_q": float(q), "diffusion_sigma": float(np.sqrt(q))}
            variance = np.maximum(current[:, 1] + q * current[:, 2], 0.0)
            for h, v in enumerate(variance, 1): row[f"variance_h{h}"] = float(v)
            rows.append(row)
    if not rows:
        raise ValueError("No origins have sufficient complete historical MPC forecasts. "
            "Generate a longer MPC forecast archive before sweeping.")
    return pd.DataFrame(rows)


def _evaluate(fits, archive, quantiles=5, cohort="all"):
    """Score every eligible fit on identical six-hour realized errors."""
    metrics, bins = [], []
    std_floor = 1e-12
    for days, part in fits.groupby("history_days", sort=True):
        part = part.sort_values("origin").reset_index(drop=True)
        origins = part["origin"].tolist()
        errors = np.stack([archive[stamp][:, 0] for stamp in origins])
        predicted = part[[f"variance_h{h}" for h in range(1, HORIZON + 1)]].to_numpy(dtype=float)
        squared = errors ** 2
        nll = 0.5 * (np.log(2.0 * np.pi * np.maximum(predicted, std_floor)) +
            squared / np.maximum(predicted, std_floor))

        def row_metrics(subset, quantile=0):
            e, v, n = errors[subset], predicted[subset], nll[subset]
            mse = float(np.mean(e ** 2))
            vbar = float(np.mean(v))
            z = np.sqrt(np.maximum(v, 0.0))
            return {"cohort": cohort, "history_days": int(days),
                "uncertainty_quantile": int(quantile),
                "n_origins": int(len(e)), "n_forecasts": int(e.size),
                "observed_rmse": float(np.sqrt(mse)),
                "predicted_rms_std": float(np.sqrt(vbar)),
                "predicted_to_observed_mse": float(vbar / mse) if mse > 0 else np.nan,
                "coverage_68_pct": float(100 * np.mean(np.abs(e) <= z)),
                "coverage_95_pct": float(100 * np.mean(np.abs(e) <= 1.96 * z)),
                "gaussian_nll": float(np.mean(n)),
                "mean_sigma": float(part.iloc[subset]["diffusion_sigma"].mean()),
                "mean_q": float(part.iloc[subset]["diffusion_q"].mean()),
                "zero_q_pct": float(100 * np.mean(part.iloc[subset]["diffusion_q"] <= 1e-14))}

        metrics.append(row_metrics(np.arange(len(part))))
        if len(part) < quantiles: continue
        ranks = np.sqrt(np.mean(predicted, axis=1))
        order = pd.Series(ranks).rank(method="first").to_numpy()
        groups = pd.qcut(order, quantiles, labels=False).astype(int) + 1
        for quantile in range(1, quantiles + 1):
            ix = np.flatnonzero(groups == quantile)
            bins.append(row_metrics(ix, quantile))
    return pd.DataFrame(metrics), pd.DataFrame(bins)


def summarize_by_lead(fits, archive, cohort):
    """Exact-lead variance/coverage scores for every historical fit length."""
    rows = []
    for days, part in fits.groupby("history_days", sort=True):
        errors = np.stack([archive[o][:, 0] for o in part["origin"]])
        variance = part[[f"variance_h{h}" for h in range(1, HORIZON + 1)]].to_numpy(dtype=float)
        for lead in range(HORIZON):
            e, v = errors[:, lead], variance[:, lead]
            mse, vbar = float(np.mean(e ** 2)), float(np.mean(v))
            std = np.sqrt(v)
            safe = np.maximum(v, 1e-12)
            rows.append({"cohort": cohort, "history_days": int(days),
                "lead_hours": lead + 1, "n_origins": int(len(e)),
                "observed_rmse": float(np.sqrt(mse)),
                "predicted_rms_std": float(np.sqrt(vbar)),
                "predicted_to_observed_mse": vbar / mse if mse > 0 else np.nan,
                "coverage_68_pct": 100.0 * float(np.mean(np.abs(e) <= std)),
                "coverage_95_pct": 100.0 * float(np.mean(np.abs(e) <= 1.96 * std)),
                "gaussian_nll": 0.5 * float(np.mean(
                    np.log(2.0 * np.pi * safe) + e ** 2 / safe))})
    return pd.DataFrame(rows)


def sweep(archive, min_days=7, max_days=28, quantiles=5):
    fits = replay(archive, min_days, max_days)
    # Full-cohort scores use every eligible origin for each window (sample size
    # changes). The common cohort uses only origins with all max_days present.
    common = set(fits.loc[fits["history_days"] == max_days, "origin"])
    if len(common) < quantiles:
        raise ValueError(f"Only {len(common)} common {max_days}-day origins; "
            f"need at least {quantiles} for uncertainty quantiles.")
    matched = fits.loc[fits["origin"].isin(common)].copy()
    full_summary, full_bins = _evaluate(fits, archive, quantiles, cohort="all")
    paired_summary, paired_bins = _evaluate(matched, archive, quantiles, cohort="matched")
    return {"fits": fits, "matched_fits": matched,
        "all_summary": full_summary, "all_quantiles": full_bins,
        "matched_summary": paired_summary, "matched_quantiles": paired_bins,
        "all_leads": summarize_by_lead(fits, archive, "all"),
        "matched_leads": summarize_by_lead(matched, archive, "matched")}


def save_plots(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    score = result["matched_summary"]
    q = result["matched_quantiles"]
    days = score["history_days"].to_numpy(dtype=int)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0, 0].plot(days, score["predicted_rms_std"], "o-", label="Predicted RMS std")
    ax[0, 0].axhline(float(score["observed_rmse"].iloc[0]), ls="--",
        color="black", label="Observed MPC RMSE")
    ax[0, 0].set_title("Uncertainty scale (same six-hour MPC origins)")
    ax[0, 0].set_ylabel("Normalized demand")
    ax[0, 0].legend()
    ax[0, 1].plot(days, score["predicted_to_observed_mse"], "o-")
    ax[0, 1].axhline(1., ls="--", color="black")
    ax[0, 1].set_title("Variance calibration (ideal = 1)")
    ax[0, 1].set_ylabel("Predicted variance / realized MSE")
    ax[1, 0].plot(days, score["coverage_68_pct"], "o-", label="Observed 68% coverage")
    ax[1, 0].plot(days, score["coverage_95_pct"], "o-", label="Observed 95% coverage")
    ax[1, 0].axhline(68.27, ls="--", color="tab:blue", alpha=0.6)
    ax[1, 0].axhline(95., ls="--", color="tab:orange", alpha=0.6)
    ax[1, 0].set_title("Gaussian coverage on matched origins")
    ax[1, 0].set_ylabel("Observed coverage (%)")
    ax[1, 0].legend()
    ax[1, 1].plot(days, score["gaussian_nll"], "o-")
    ax[1, 1].set_title("Mean Gaussian NLL (lower is better)")
    ax[1, 1].set_ylabel("Gaussian NLL")
    for a in ax.flat:
        a.set_xlabel("Historical diffusion fitting days")
        a.set_xticks(days[::max(1, len(days) // 11)])
        a.grid(alpha=0.25)
    fig.suptitle(f"WM4 | six-hour Wiener sigma fit | rolling {int(days.min())}–{int(days.max())} day sweep\n"
        f"Identical {int(score['n_origins'].iloc[0])} complete MPC origins across all windows")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "matched_sigma_sweep.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for quantile, part in q.groupby("uncertainty_quantile", sort=True):
        part = part.sort_values("history_days")
        ax[0].plot(part["history_days"], part["predicted_to_observed_mse"],
            "o-", label=f"Q{quantile}")
        ax[1].plot(part["history_days"], part["coverage_95_pct"],
            "o-", label=f"Q{quantile}")
    ax[0].axhline(1., ls="--", color="black")
    ax[0].set_title("Variance calibration by own predicted uncertainty quintile")
    ax[0].set_ylabel("Predicted variance / observed MSE")
    ax[1].axhline(95., ls="--", color="black")
    ax[1].set_title("Gaussian 95% interval coverage by quintile")
    ax[1].set_ylabel("Coverage (%)")
    for a in ax:
        a.set_xlabel("Historical diffusion fitting days")
        a.grid(alpha=0.25)
        a.legend(ncol=2)
    fig.suptitle("WM4 | matched-cohort conditional reliability over all diffusion backlogs")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "matched_quintile_sweep.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    full = result["all_summary"]
    ax.plot(full["history_days"], full["n_origins"], "o-", label="All eligible fits")
    ax.plot(score["history_days"], score["n_origins"], "o-",
        label=f"Matched {int(days.max())}d cohort")
    ax.set_xlabel("Historical diffusion fitting days")
    ax.set_ylabel("Number of six-hour MPC origins fitted")
    ax.set_title("Total fitted origins by rolling calibration window")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "fitted_origins_by_window.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    leads = result["matched_leads"]
    fig, ax = plt.subplots(figsize=(12, 5))
    table = leads.pivot(index="lead_hours", columns="history_days",
        values="predicted_to_observed_mse").sort_index()
    heat = ax.imshow(table.to_numpy(), origin="lower", aspect="auto", vmin=0.25, vmax=2.0)
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=90)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("Historical diffusion fitting days")
    ax.set_ylabel("MPC forecast lead (hours)")
    ax.set_title("Variance calibration by fit backlog and exact MPC lead (ideal = 1)")
    fig.colorbar(heat, ax=ax, label="Predicted variance / empirical MSE")
    fig.tight_layout()
    fig.savefig(output / "matched_lead_calibration_heatmap.png", dpi=170,
        bbox_inches="tight")
    plt.close(fig)
    return output


def analyze(forecasts=SOURCE, output=OUTPUT, min_days=7, max_days=28,
    quantiles=5, fit_hours=24):
    archive, _ = read_mpc_archive(forecasts, horizon=HORIZON, fit_hours=fit_hours)
    result = sweep(archive, min_days=min_days, max_days=max_days, quantiles=quantiles)
    output = save_plots(result, output)
    for name, value in result.items():
        value.to_csv(output / f"{name}.csv", index=False)
    report = result["matched_summary"]
    print(f"MPC origins in forecast archive: {len(archive)}")
    print(f"All diffusion QP fits across {min_days}..{max_days} days: {len(result['fits'])}")
    print(f"Matched origin cohort: {int(report.iloc[0]['n_origins'])} origins (all {max_days} historical days)")
    print(report[["history_days", "n_origins", "predicted_to_observed_mse",
        "coverage_68_pct", "coverage_95_pct", "gaussian_nll", "mean_sigma"]].to_string(
        index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"Saved complete fit tables, exact-lead scores, and four plots: {output}")
    return {**result, "output": output}


def main():
    parser = ArgumentParser(description="Sweep 7–28 historical days for six-hour WM4 Wiener variance using a single saved MPC forecast CSV.")
    parser.add_argument("--forecasts", type=Path, default=SOURCE,
        help="Saved WM4 mpc_forecasts.csv from a 24-hour parameter fit (first six MPC leads used).")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--min-days", type=int, default=7)
    parser.add_argument("--max-days", type=int, default=28)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--fit-hours", type=int, default=24)
    args = parser.parse_args()
    if not 2 <= args.min_days <= args.max_days:
        parser.error("Require 2 <= min-days <= max-days.")
    if args.quantiles < 2: parser.error("--quantiles must be >=2.")
    return analyze(args.forecasts, args.output, args.min_days, args.max_days,
        args.quantiles, args.fit_hours)


if __name__ == "__main__": main()
