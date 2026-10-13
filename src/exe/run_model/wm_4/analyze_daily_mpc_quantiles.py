"""Rank full days of WM4 MPC predictions by accuracy, calibration and sharpness.

Each daily score pools all 24 hourly MPC origins and their full 1..H forecasts
(24*H paired prediction errors). Each error belongs to its forecast origin day,
even when its target is on the following calendar day.

Three distinct lower-is-better rankings:
* RMSE: accuracy of daily pooled 1..H forecasts.
* Variance calibration: |log(mean predicted variance / empirical error variance)|.
* Predicted RMS std: model-reported uncertainty magnitude ("sharpness").

Calibration and sharpness are NOT equivalent: large, correctly anticipated
forecast errors can have excellent calibration but high RMSE. The quantile
overlap tables make these relationships measurable without refitting WM4.

The bottom-left scatter is DIFFERENT: each point is one hourly MPC origin's
complete 1..H forecast, not the pooled quality of a whole day. All complete
MPC origins are shown, including origins from days excluded by daily ranking.
"""

from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_mpc_intervals import FORECASTS, load_forecasts
from src.exe.run_model.wm_4.plot_mpc_variance import select_origins, variance_scores


OUTPUT = FORECASTS.parent / "daily_quality"


def daily_metrics(frame, horizon=8):
    """Require 24 complete hourly MPC origins per day, each with exactly 1..H leads."""
    horizon = int(horizon)
    if not 1 <= horizon <= 8: raise ValueError("horizon must be in [1, 8].")
    required = {"origin", "lead_hours", "error", "predicted_variance"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Daily WM4 MPC metrics missing forecast columns: {sorted(missing)}.")
    frame = frame.copy()
    frame["origin"] = pd.to_datetime(frame["origin"])
    frame["origin_day"] = frame["origin"].dt.normalize()
    rows, excluded = [], []
    expected_hours = list(range(24))
    expected_leads = list(range(1, horizon + 1))
    for day, group in frame.groupby("origin_day", sort=True):
        count = group["origin"].nunique()
        hours = sorted(group["origin"].dt.hour.unique().tolist())
        complete = count == 24 and hours == expected_hours and len(group) == 24 * horizon
        if complete:
            complete = all(sorted(part["lead_hours"].tolist()) == expected_leads
                for _, part in group.groupby("origin"))
        if not complete:
            excluded.append({"day": day, "complete_origins": int(count),
                "rows": int(len(group)), "expected_rows": 24 * horizon})
            continue
        score = variance_scores(group)
        predicted = score["predicted_variance"]
        empirical = score["empirical_error_variance"]
        if predicted <= 0 or empirical <= 0:
            excluded.append({"day": day, "complete_origins": int(count),
                "rows": int(len(group)), "expected_rows": 24 * horizon,
                "reason": "nonpositive variance; logarithmic calibration undefined"})
            continue
        ratio = predicted / empirical
        errors = group["error"].to_numpy(dtype=float)
        pvar = group["predicted_variance"].to_numpy(dtype=float)
        records = {
            "day": day, "weekday": pd.Timestamp(day).day_name(),
            "month": pd.Timestamp(day).strftime("%Y-%m"),
            "horizon_hours": horizon, "n_origins": count, "n_predictions": len(group),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mae": float(np.mean(np.abs(errors))),
            "bias": float(np.mean(errors)),
            "observed_mse": score["observed_mse"],
            "actual_error_variance": empirical,
            "predicted_variance": predicted,
            "predicted_rms_std": score["predicted_rms_std"],
            "actual_error_std": score["empirical_error_std"],
            "variance_ratio": ratio,
            "variance_calibration_error": abs(float(np.log(ratio))),
            "variance_calibration_error_pct": 100.0 * abs(ratio - 1.0),
            "nll_mean": float(np.mean(0.5 * (np.log(2 * np.pi * pvar)
                + errors ** 2 / pvar))),
        }
        if "covered_68" in group: records["coverage_68_pct"] = 100.0 * float(group["covered_68"].mean())
        if "covered_95" in group: records["coverage_95_pct"] = 100.0 * float(group["covered_95"].mean())
        rows.append(records)
    if not rows:
        raise ValueError("No full MPC days with 24 complete hourly forecast origins. "
            "Check the requested horizon and --fully-mpc filter.")
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True), pd.DataFrame(excluded)


def origin_metrics(frame, horizon=8):
    """One observation per hourly MPC origin, pooling that origin's 1..H leads.

    Observed RMSE is sqrt(mean(e_h^2)); predicted RMS uncertainty is
    sqrt(mean(V_h)). A single eight-lead realization cannot reliably
    identify an empirical error *variance*, so the color instead depicts
    absolute log(predicted MSE / realized MSE), not variance calibration.
    """
    horizon = int(horizon)
    if not 1 <= horizon <= 8: raise ValueError("horizon must be in [1, 8].")
    needed = {"origin", "lead_hours", "error", "predicted_variance"}
    if missing := needed - set(frame.columns):
        raise ValueError(f"MPC origin quality requires columns {sorted(missing)}.")
    rows = []
    for origin, group in frame.groupby("origin", sort=True):
        if sorted(group["lead_hours"].tolist()) != list(range(1, horizon + 1)):
            raise ValueError(f"Expected one full 1..{horizon} forecast at origin {origin}.")
        errors = group["error"].to_numpy(dtype=float)
        variances = group["predicted_variance"].to_numpy(dtype=float)
        if not np.all(np.isfinite(errors)) or not np.all(np.isfinite(variances)):
            raise ValueError(f"Nonfinite forecast errors or predicted variance at {origin}.")
        if np.any(variances < 0.0):
            raise ValueError(f"Negative predicted variance at {origin}.")
        mse, predicted = float(np.mean(errors ** 2)), float(np.mean(variances))
        ratio = predicted / mse if mse > 0 else np.nan
        rows.append({
            "origin": pd.Timestamp(origin),
            "origin_day": pd.Timestamp(origin).normalize(),
            "origin_hour": pd.Timestamp(origin).hour,
            "horizon_hours": horizon,
            "n_predictions": len(group),
            "rmse": float(np.sqrt(mse)),
            "predicted_rms_std": float(np.sqrt(predicted)),
            "observed_mse": mse,
            "predicted_variance": predicted,
            "mse_ratio_predicted_to_observed": ratio,
            "mse_calibration_error": abs(float(np.log(ratio))) if ratio > 0 else np.nan,
        })
    if not rows: raise ValueError("No complete individual MPC forecasts to plot.")
    return pd.DataFrame(rows).sort_values("origin").reset_index(drop=True)


def assign_quantiles(daily, quantiles=5):
    """Equally sized rank quantiles; 1 is lowest/best for each distinct criterion."""
    quantiles = int(quantiles)
    if quantiles < 2: raise ValueError("quantiles must be at least 2.")
    if len(daily) < quantiles:
        raise ValueError(f"At least {quantiles} full days required; found {len(daily)}.")
    result = daily.copy().sort_values("day").reset_index(drop=True)
    names = {
        "rmse_quantile": "rmse",
        "calibration_quantile": "variance_calibration_error",
        "uncertainty_quantile": "predicted_rms_std",
    }
    for label, name in names.items():
        if not np.all(np.isfinite(result[name].to_numpy(dtype=float))):
            raise ValueError(f"Cannot rank nonfinite {name}.")
        # The deterministic chronological tie-breaker preserves equal counts.
        ranks = result[name].rank(method="first", ascending=True)
        result[label] = pd.qcut(ranks, quantiles, labels=False).astype(int) + 1
    return result


def quantile_spread(daily, group):
    """Show distribution within quantile, not just the average-ranked day."""
    cols = ("rmse", "predicted_rms_std", "actual_error_std",
        "variance_calibration_error", "variance_ratio", "nll_mean")
    rows = []
    for q, part in daily.groupby(group, sort=True):
        row = {"quantile": int(q), "n_days": int(len(part)),
            "start_day": part["day"].min(), "end_day": part["day"].max()}
        for metric in cols:
            value = part[metric].to_numpy(dtype=float)
            row.update({f"{metric}_mean": float(np.mean(value)),
                f"{metric}_min": float(np.min(value)),
                f"{metric}_p25": float(np.percentile(value, 25)),
                f"{metric}_median": float(np.median(value)),
                f"{metric}_p75": float(np.percentile(value, 75)),
                f"{metric}_max": float(np.max(value))})
        rows.append(row)
    return pd.DataFrame(rows)


def quantile_overlap(daily, x, y, quantiles):
    """Rows are daily RMSE quantiles; columns are calibration or uncertainty quantiles."""
    counts = pd.crosstab(daily[x], daily[y]).reindex(
        index=range(1, quantiles + 1), columns=range(1, quantiles + 1), fill_value=0)
    counts.index.name, counts.columns.name = x, y
    return counts


def overlap_summary(daily, quantiles):
    """Compare best/bad-day coincidences to chance under independent ranks."""
    n = len(daily)
    def correlation(first, second):
        value = float(daily[first].corr(daily[second], method="spearman"))
        return value if np.isfinite(value) else None
    best = daily["rmse_quantile"] == 1
    worst = daily["rmse_quantile"] == quantiles
    info = {"days": int(n), "quantiles": int(quantiles),
        "expected_best_overlap_independent": float(best.sum() * (daily["calibration_quantile"] == 1).sum() / n),
        "spearman_rmse_vs_predicted_std": correlation("rmse", "predicted_rms_std"),
        "spearman_rmse_vs_calibration_error": correlation("rmse", "variance_calibration_error")}
    for label, name in (("calibration", "calibration_quantile"),
        ("uncertainty", "uncertainty_quantile")):
        good = daily[name] == 1
        bad = daily[name] == quantiles
        info[f"best_rmse_and_best_{label}_days"] = int((best & good).sum())
        info[f"best_rmse_days"] = int(best.sum())
        info[f"best_{label}_days"] = int(good.sum())
        info[f"expected_best_{label}_overlap_independent"] = float(best.sum() * good.sum() / n)
        info[f"fraction_best_rmse_also_best_{label}"] = float((best & good).sum() / best.sum())
        info[f"worst_rmse_and_worst_{label}_days"] = int((worst & bad).sum())
    return info


def _heatmap(ax, counts, label):
    q = counts.shape[0]
    values = counts.to_numpy(dtype=int)
    ax.imshow(values, cmap="Blues", vmin=0, vmax=max(int(values.max()), 1), aspect="auto")
    for i in range(q):
        for j in range(q):
            count = int(values[i, j])
            row_n = int(values[i].sum())
            color = "white" if count >= 0.6 * max(values.max(), 1) else "black"
            ax.text(j, i, f"{count}\n({count / row_n:.0%})" if row_n else "0",
                ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(q), labels=[str(k) for k in range(1, q + 1)])
    ax.set_yticks(range(q), labels=[str(k) for k in range(1, q + 1)])
    ax.set_xlabel(label + " quantile (1 = best / lowest)")
    ax.set_ylabel("RMSE quantile (1 = lowest)")
    ax.set_title(f"Daily RMSE vs {label.lower()}\ncount (share of RMSE-quantile days)")


def plot_quantiles(daily, overlap_calibration, overlap_uncertainty, origins, output):
    """Daily heatmaps/timeline plus one scatter point per eight-hour MPC origin."""
    q = len(overlap_calibration)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    _heatmap(axes[0, 0], overlap_calibration, "Variance calibration")
    _heatmap(axes[0, 1], overlap_uncertainty, "Predicted uncertainty")
    # Each dot is ONE hourly-origin MPC forecast, not a daily average.
    # Within an origin, eight forecast leads are pooled as RMSE and RMS std.
    color = origins["mse_calibration_error"].to_numpy(dtype=float)
    if not np.all(np.isfinite(color)):
        raise ValueError("Origin-level forecast MSE ratio is undefined for zero realized or predicted MSE.")
    sc = axes[1, 0].scatter(origins["predicted_rms_std"], origins["rmse"],
        c=color, cmap="viridis", s=15, alpha=0.55,
        edgecolors="none", rasterized=True)
    lo = float(min(origins["predicted_rms_std"].min(), origins["rmse"].min()))
    hi = float(max(origins["predicted_rms_std"].max(), origins["rmse"].max()))
    axes[1, 0].plot([lo, hi], [lo, hi], "k--", alpha=0.45, label="RMSE = predicted std")
    axes[1, 0].set_xlabel("Predicted RMS standard deviation (per MPC origin)")
    axes[1, 0].set_ylabel("Observed 1..H MPC RMSE (per origin)")
    axes[1, 0].set_title(f"Individual 1..H MPC forecasts | {len(origins)} hourly origins")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=axes[1, 0], label="|log(predicted MSE / observed MSE)|")

    for col, label, marker in (
        ("rmse_quantile", "RMSE", "o"),
        ("uncertainty_quantile", "Predicted uncertainty", "s"),
        ("calibration_quantile", "Variance calibration", "^")):
        axes[1, 1].scatter(daily["day"], daily[col], s=17,
            alpha=0.55, marker=marker, label=label)
    axes[1, 1].set_yticks(range(1, q + 1))
    axes[1, 1].set_ylim(q + 0.35, 0.65)
    axes[1, 1].set_ylabel("Daily quantile (1 = best / lowest)")
    axes[1, 1].set_xlabel("MPC forecast origin date")
    axes[1, 1].set_title("Quality quantiles through the evaluation period")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].tick_params(axis="x", labelrotation=35)
    fig.suptitle("WM4 MPC | daily quality quantiles and individual 1..H-hour MPC forecasts", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.5, w_pad=2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return output


def analyze(forecasts=FORECASTS, output=OUTPUT, horizon=8, quantiles=5,
    fully_mpc=False, history_windows=5, expected_sigma_fit_mode="mpc_replay"):
    frame, incomplete_origins, mode = load_forecasts(forecasts, horizon,
        expected_sigma_fit_mode=expected_sigma_fit_mode)
    frame = select_origins(frame, fully_mpc=fully_mpc, history_windows=history_windows)
    origins = origin_metrics(frame, horizon)
    days, skipped = daily_metrics(frame, horizon)
    ranked = assign_quantiles(days, quantiles)
    q = int(quantiles)
    cal = quantile_overlap(ranked, "rmse_quantile", "calibration_quantile", q)
    unc = quantile_overlap(ranked, "rmse_quantile", "uncertainty_quantile", q)
    info = overlap_summary(ranked, q)
    info.update({"horizon_hours": int(horizon), "fully_mpc": bool(fully_mpc),
        "sigma_fit_mode": mode, "incomplete_origins_excluded": int(incomplete_origins),
        "incomplete_days_excluded": int(len(skipped)),
        "individual_mpc_origins_plotted": int(len(origins))})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output / "daily_quality_quantiles.csv", index=False)
    origins.to_csv(output / "mpc_origin_quality.csv", index=False)
    skipped.to_csv(output / "excluded_incomplete_days.csv", index=False)
    cal.to_csv(output / "rmse_vs_calibration_quantiles.csv")
    unc.to_csv(output / "rmse_vs_uncertainty_quantiles.csv")
    for label, col in (("rmse", "rmse_quantile"),
        ("calibration", "calibration_quantile"),
        ("uncertainty", "uncertainty_quantile")):
        quantile_spread(ranked, col).to_csv(output / f"{label}_quantile_spread.csv", index=False)
    with (output / "quantile_overlap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, allow_nan=False)
    fig = plot_quantiles(ranked, cal, unc, origins, output / "daily_mpc_quality_quantiles.png")
    print(f"WM4 daily MPC quality | {len(ranked)} complete days, {len(skipped)} partial days excluded; "
        f"each day = 24 origins x {horizon} horizons; fully MPC={fully_mpc}")
    print(f"Bottom-left scatter: {len(origins)} individual hourly MPC-origin forecasts, "
        f"each pooling {horizon} leads (including origins from incomplete days).")
    print(f"Best RMSE quantile overlaps: best-calibrated "
        f"{info['best_rmse_and_best_calibration_days']}/{info['best_rmse_days']}, "
        f"lowest predicted uncertainty "
        f"{info['best_rmse_and_best_uncertainty_days']}/{info['best_rmse_days']}")
    corr_std = info["spearman_rmse_vs_predicted_std"]
    corr_cal = info["spearman_rmse_vs_calibration_error"]
    print(f"Spearman RMSE vs predicted std={corr_std:.3f}" if corr_std is not None
        else "Spearman RMSE vs predicted std=undefined (constant ranks)")
    print(f"Spearman RMSE vs calibration error={corr_cal:.3f}" if corr_cal is not None
        else "Spearman RMSE vs calibration error=undefined (constant ranks)")
    print(f"Figure: {fig}\nDaily rankings, per-origin metrics, overlap matrices "
        f"and quantile-spread tables: {output}")
    return {"daily": ranked, "origins": origins, "excluded": skipped, "calibration_overlap": cal,
        "uncertainty_overlap": unc, "summary": info, "figure": fig, "output": output}


def main():
    parser = ArgumentParser(description="Quantile-rank full-day WM4 MPC accuracy, variance calibration and predicted uncertainty.")
    parser.add_argument("--forecasts", type=Path, default=FORECASTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--horizon", type=int, default=8, help="MPC horizon in hours, 1..8.")
    parser.add_argument("--quantiles", type=int, default=5, help="Number of equal-count day quantiles (default: 5).")
    parser.add_argument("--fully-mpc", action="store_true",
        help="Require five historical MPC samples (no oracle bootstrap) for every hourly origin in a day.")
    parser.add_argument("--history-windows", type=int, default=5)
    parser.add_argument("--expected-sigma-fit-mode",
        choices=("mpc_replay", "mpc_replay_24d_24h", "mpc_replay_24d_6h", "mpc_replay_30d_6h", "mpc_replay_28d_6h", "mpc_replay_32d_6h_clip4", "any"),
        default="mpc_replay")
    args = parser.parse_args()
    return analyze(args.forecasts, args.output, args.horizon, args.quantiles,
        args.fully_mpc, args.history_windows, args.expected_sigma_fit_mode)


if __name__ == "__main__": main()
