"""Paired WM4 uncertainty-quintile comparison: 5-week/8h vs 24-day/24h sigma.

Only evaluate origins with forecasts from BOTH fits; by default require that
the rolling estimator used a FULL 24 or 30 historical days, not its warmup.

Main 2x3 figure stratifies by the OLD estimator's predicted eight-hour
variance. Every panel uses exactly the same origins and empirical errors,
so differences cannot result from changed evaluation cohorts.

A separate reliability diagram recomputes quintiles independently for each
estimator, revealing whether the new sigma gives better conditional forecasts.
No WM4 model fitting is performed in this module.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_mpc_intervals import FORECASTS, load_forecasts


NEW_FORECASTS = FORECASTS.parent.parent / "all_days_mpc_sigma_24d_24h" / "mpc_forecasts.csv"
NEW_FORECASTS_6H = FORECASTS.parent.parent / "all_days_mpc_sigma_24d_6h" / "mpc_forecasts.csv"
NEW_FORECASTS_30D_6H = FORECASTS.parent.parent / "all_days_mpc_sigma_30d_6h" / "mpc_forecasts.csv"
NEW_FORECASTS_32D_CLIP4 = FORECASTS.parent.parent / "all_days_mpc_sigma_32d_6h_clip4" / "mpc_forecasts.csv"
NEW_FORECASTS_NO_P0 = FORECASTS.parent.parent / "all_days_mpc_sigma_24d_6h_no_p0" / "mpc_forecasts.csv"
OUTPUT = FORECASTS.parent.parent / "sigma_estimator_comparison"
ROLLING_MODES = {"mpc_replay_24d_24h": (24, "24-day/24h"),
    "mpc_replay_24d_6h": (6, "24-day/6h"),
    "mpc_replay_30d_6h": (6, "30-day/6h"),
    "mpc_replay_32d_6h_clip4": (6, "32-day/6h clip4"),
    "mpc_replay_24d_6h_no_p0": (6, "24-day/6h no P0")}
ROLLING_DAYS = {"mpc_replay_24d_24h": 24, "mpc_replay_24d_6h": 24,
    "mpc_replay_30d_6h": 30,
    "mpc_replay_32d_6h_clip4": 32,
    "mpc_replay_24d_6h_no_p0": 24}


def _mature_rolling(frame, days, calibration_horizon):
    required = {"sigma_rolling_complete", "sigma_rolling_count",
        "sigma_rolling_days", "calibration_horizon"}
    if missing := required - set(frame):
        raise ValueError(f"New sigma forecast data missing rolling maturity columns: {sorted(missing)}.")
    ready = (frame["sigma_rolling_complete"].astype(str).str.lower().eq("true")
        & (pd.to_numeric(frame["sigma_rolling_count"]) >= int(days))
        & (pd.to_numeric(frame["sigma_rolling_days"]) == int(days))
        & (pd.to_numeric(frame["calibration_horizon"]) == int(calibration_horizon)))
    mature = frame.loc[ready].copy()
    if mature.empty: raise ValueError(f"No fully mature {days}-day/{calibration_horizon}-hour sigma fits. "
        f"Run the rolling MPC for at least {int(days) + 1} complete days or use --include-warmup.")
    return mature


def matched_forecasts(old, new, horizon=8, require_full_history=True, days=24,
    calibration_horizon=24, old_calibration_horizon=None, old_days=None):
    """Align matching MPC origins/leads; require full rolling history if selected."""
    if require_full_history and old_calibration_horizon is not None:
        old = _mature_rolling(old, days if old_days is None else old_days,
            old_calibration_horizon)
    if require_full_history:
        new = _mature_rolling(new, days, calibration_horizon)
    keys = ["origin", "lead_hours"]
    retain = keys + ["I_observed", "I_mpc", "initial_variance", "process_variance",
        "predicted_variance", "diffusion_sigma"]
    for label, data in (("old", old), ("new", new)):
        if missing := set(retain) - set(data):
            raise ValueError(f"Missing {label} forecast comparison columns: {sorted(missing)}.")
        if data.duplicated(keys).any(): raise ValueError(f"{label} forecasts contain duplicate origin/lead.")
    merged = old[retain].merge(new[retain], on=keys, how="inner",
        suffixes=("_old", "_new"), validate="one_to_one")
    count = merged.groupby("origin")["lead_hours"].nunique()
    origins = count[count == int(horizon)].index
    merged = merged.loc[merged["origin"].isin(origins)].sort_values(keys).reset_index(drop=True)
    if merged.empty: raise ValueError("No matching complete MPC origins between sigma estimators.")
    for name in ("I_observed", "I_mpc"):
        if not np.allclose(merged[f"{name}_old"], merged[f"{name}_new"],
            rtol=1e-7, atol=1e-7):
            raise ValueError(f"Paired {name} differs. The sigma comparison must use identical "
                "deterministic fit, initialization, origin and forecasts.")
    merged["error"] = merged["I_observed_old"] - merged["I_mpc_old"]
    merged["squared_error"] = merged["error"] ** 2
    return merged


def summarize_origins(data, quantiles=5):
    """Assign common baseline quantiles and method-specific quantiles at lead 8."""
    at8 = data.loc[data["lead_hours"] == data["lead_hours"].max()].copy()
    g = data.groupby("origin", sort=True)
    for col in ("predicted_variance_old", "predicted_variance_new", "squared_error"):
        at8[f"pooled_{col}"] = at8["origin"].map(g[col].mean())
    at8 = at8[["origin", "pooled_predicted_variance_old",
        "pooled_predicted_variance_new", "pooled_squared_error"]].sort_values("origin")
    at8 = at8.rename(columns={"pooled_predicted_variance_old": "old_predicted_variance",
        "pooled_predicted_variance_new": "new_predicted_variance",
        "pooled_squared_error": "realized_mse"})
    if int(quantiles) < 2 or len(at8) < int(quantiles):
        raise ValueError(f"Need at least {quantiles} common MPC origins for quantile ranking.")
    for method in ("old", "new"):
        # Equal-count bins; use chronological tie-breaking if sigma ties.
        rank = at8[f"{method}_predicted_variance"].rank(method="first")
        at8[f"{method}_quantile"] = pd.qcut(rank, int(quantiles),
            labels=False).astype(int) + 1
    return at8.reset_index(drop=True)


def quantile_curves(data, origin_bins, quantiles=5, independent=False):
    """Paired cumulative variance reliability, either shared or independent bins."""
    merged = data.merge(origin_bins[["origin", "old_quantile", "new_quantile"]],
        on="origin", validate="many_to_one")
    rows = []
    for method in ("old", "new"):
        col = f"{method}_quantile" if independent else "old_quantile"
        for (q, h), part in merged.groupby([col, "lead_hours"], sort=True):
            # One realized MSE value per matched forecast; compare mean of
            # cumulative 1..H curves, not variance across 8 error samples.
            selected = merged.loc[merged["origin"].isin(part["origin"])]
            selected = selected.loc[selected["lead_hours"] <= int(h)]
            rows.append({"method": method, "quantile": int(q), "horizon": int(h),
                "n_origins": int(part["origin"].nunique()),
                "predicted_variance": float(selected[f"predicted_variance_{method}"].mean()),
                "initial_variance": float(selected[f"initial_variance_{method}"].mean()),
                "wiener_variance": float(selected[f"process_variance_{method}"].mean()),
                "realized_mse": float(selected["squared_error"].mean())})
    result = pd.DataFrame(rows).sort_values(["method", "quantile", "horizon"])
    result["predicted_to_observed"] = np.divide(result["predicted_variance"],
        result["realized_mse"], out=np.full(len(result), np.nan),
        where=result["realized_mse"] > 0)
    return result


def compare_metrics(data, origins):
    """Proper probabilistic log score and conditional calibration on paired rows."""
    stats = []
    for method in ("old", "new"):
        pred = data[f"predicted_variance_{method}"].to_numpy(dtype=float)
        if np.any(pred <= 0) or not np.all(np.isfinite(pred)):
            raise ValueError(f"{method} predicted variance must be strictly positive for NLL.")
        mse = data["squared_error"].to_numpy(dtype=float)
        first = data["lead_hours"].to_numpy(dtype=int)
        for h in sorted(np.unique(first)):
            idx = first <= h
            p, y = pred[idx], mse[idx]
            stats.append({"method": method, "cumulative_horizon": int(h),
                "n_origins": int(data.loc[idx, "origin"].nunique()),
                "predicted_variance": float(p.mean()),
                "realized_mse": float(y.mean()),
                "predicted_to_observed": float(p.mean() / y.mean()) if y.mean() > 0 else np.nan,
                "mean_gaussian_nll": float(np.mean(0.5 * (
                    np.log(2 * np.pi * p) + y / p)))})
    return pd.DataFrame(stats)


def plot_paired(curves, output, quantiles=5, new_label="24-day/24h", old_label="5-week/8h"):
    """Same old-estimator quintile membership for both sigma forecasts."""
    columns = 3
    rows = int(np.ceil((int(quantiles) + 1) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(17, 5 * rows))
    axes = np.asarray(axes).ravel()
    for i, q in enumerate(range(1, int(quantiles) + 1)):
        ax = axes[i]
        group = curves[curves["quantile"] == q]
        old = group[group["method"] == "old"].sort_values("horizon")
        new = group[group["method"] == "new"].sort_values("horizon")
        h = old["horizon"].to_numpy(dtype=int)
        ax.plot(h, old["predicted_variance"], "o-", label=f"{old_label} predicted")
        ax.plot(h, new["predicted_variance"], "o-", label=f"{new_label} predicted")
        ax.plot(h, old["realized_mse"], "ks--", label="Same origins: observed MSE")
        ax.set_title(f"Original predicted-variance Q{q} | {int(old.iloc[0]['n_origins'])} origins")
        ax.set_xlabel("Cumulative MPC horizon (hours)")
        ax.set_ylabel("Normalized demand squared")
        ax.set_xticks(h)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    ax = axes[int(quantiles)]
    last = curves[curves["horizon"] == curves["horizon"].max()]
    for method, label in (("old", old_label), ("new", new_label)):
        group = last[last["method"] == method].sort_values("quantile")
        ax.plot(group["predicted_variance"], group["realized_mse"], "o-",
            label=label)
        for _, row in group.iterrows():
            ax.annotate(f"Q{int(row['quantile'])}",
                (row["predicted_variance"], row["realized_mse"]), xytext=(4, 4),
                textcoords="offset points", fontsize=8)
    lim = float(max(last["predicted_variance"].max(), last["realized_mse"].max())) * 1.08
    ax.plot([0, lim], [0, lim], "k--", alpha=0.45, label="Perfect calibration")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title("8h reliability | identical baseline quintile cohorts")
    ax.set_xlabel("Mean predicted 1..8h variance")
    ax.set_ylabel("Mean observed 1..8h MSE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    for unused in axes[int(quantiles) + 1:]: unused.axis("off")
    fig.suptitle(f"WM4 | {new_label} versus {old_label} sigma calibration"
        "\nMatched 8h MPC origins, stratified by ORIGINAL predicted variance",
        fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=185, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_independent(curves, output, new_label="24-day/24h", old_label="5-week/8h"):
    """Separate quantile rankings show each estimator's own conditional calibration."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    last = curves[curves["horizon"] == curves["horizon"].max()]
    for method, label in (("old", old_label), ("new", new_label)):
        g = last[last["method"] == method].sort_values("quantile")
        axes[0].plot(g["predicted_variance"], g["realized_mse"], "o-", label=label)
        axes[1].plot(g["quantile"], g["predicted_to_observed"], "o-", label=label)
    lim = float(max(last["predicted_variance"].max(), last["realized_mse"].max())) * 1.08
    axes[0].plot([0, lim], [0, lim], "k--", label="Perfect calibration", alpha=0.5)
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].set_xlabel("Predicted cumulative eight-hour variance")
    axes[0].set_ylabel("Observed cumulative eight-hour MSE")
    axes[0].set_title("Independently ranked uncertainty quintiles")
    axes[1].axhline(1, color="black", ls="--", lw=1)
    axes[1].set_xlabel("Own predicted-uncertainty quantile")
    axes[1].set_ylabel("Predicted / observed MSE")
    axes[1].set_title("Conditional calibration ratio (ideal 1)")
    axes[1].set_xticks(sorted(last["quantile"].unique()))
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("WM4 | conditional calibration on identical evaluation origins")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=185, bbox_inches="tight")
    plt.close(fig)
    return output


def analyze(old_path=FORECASTS, new_path=NEW_FORECASTS, output=OUTPUT,
    quantiles=5, horizon=8, full_history=True, history_days=None,
    new_mode="mpc_replay_24d_24h", old_mode="mpc_replay"):
    if new_mode not in ROLLING_MODES:
        raise ValueError(f"Expected new sigma mode in {tuple(ROLLING_MODES)}.")
    if old_mode != "mpc_replay" and old_mode not in ROLLING_MODES:
        raise ValueError("Old sigma mode must be mpc_replay or a rolling sigma mode.")
    cal_horizon, new_label = ROLLING_MODES[new_mode]
    history_days = ROLLING_DAYS[new_mode] if history_days is None else int(history_days)
    old_label = ROLLING_MODES[old_mode][1] if old_mode in ROLLING_MODES else "5-week/8h"
    old, _, _ = load_forecasts(old_path, horizon, expected_sigma_fit_mode=old_mode)
    new, _, _ = load_forecasts(new_path, horizon, expected_sigma_fit_mode=new_mode)
    paired = matched_forecasts(old, new, horizon, require_full_history=full_history,
        days=history_days, calibration_horizon=cal_horizon,
        old_calibration_horizon=ROLLING_MODES[old_mode][0] if old_mode in ROLLING_MODES else None,
        old_days=ROLLING_DAYS.get(old_mode))
    origins = summarize_origins(paired, quantiles)
    paired_curves = quantile_curves(paired, origins, quantiles, independent=False)
    independent = quantile_curves(paired, origins, quantiles, independent=True)
    stats = compare_metrics(paired, origins)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "matched_mpc_forecasts.csv", index=False)
    origins.to_csv(output / "matched_origin_quintiles.csv", index=False)
    paired_curves.to_csv(output / "paired_baseline_quintile_curves.csv", index=False)
    independent.to_csv(output / "independent_quintile_curves.csv", index=False)
    stats.to_csv(output / "paired_cumulative_nll_mse.csv", index=False)
    figure = plot_paired(paired_curves, output / "paired_sigma_quintiles.png", quantiles,
        new_label, old_label)
    reliability = plot_independent(independent,
        output / "independent_sigma_reliability.png", new_label, old_label)
    print(f"Paired sigma: {old_label} vs {new_label}, {len(origins)} identical 1..{horizon}h "
        f"MPC origins; full {history_days}d history={full_history}")
    print(stats.loc[stats["cumulative_horizon"] == horizon].to_string(
        index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved {figure}, {reliability} and CSV comparisons at {output}")
    return {"paired": paired, "origins": origins,
        "paired_curves": paired_curves, "independent": independent,
        "stats": stats, "figure": figure, "reliability": reliability, "output": output}


def main():
    parser = ArgumentParser(description="Compare historical 24d/6h and 32d/6h clip4 Wiener estimates on matched MPC origins, or other rolling modes.")
    parser.add_argument("--old", type=Path, default=FORECASTS,
        help="Original five-week/8h or alternative existing WM4 mpc_forecasts.csv")
    parser.add_argument("--old-mode", choices=("mpc_replay", *ROLLING_MODES),
        default="mpc_replay", help="Sigma mode of --old CSV.")
    parser.add_argument("--new", type=Path, default=None,
        help="New rolling sigma mpc_forecasts.csv; default is inferred from --new-mode.")
    parser.add_argument("--new-mode", choices=tuple(ROLLING_MODES), default="mpc_replay_24d_24h",
        help="Choose the rolling Wiener calibration mode, including 32d/6h clip4.")
    parser.add_argument("--output", type=Path, default=None,
        help="Comparison output folder; defaults to a mode-specific directory.")
    parser.add_argument("--horizon", type=int, default=None,
        help="Matched forecast horizon: 6h for 32d/6h clip4, 8h otherwise.")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--history-days", type=int, default=None,
        help="Required rolling history for new mode (default: 24, 30 or 32 by mode).")
    parser.add_argument("--include-warmup", action="store_true",
        help="Include origins before the full rolling calibration history is available.")
    args = parser.parse_args()
    defaults = {"mpc_replay_24d_24h": NEW_FORECASTS,
        "mpc_replay_24d_6h": NEW_FORECASTS_6H,
        "mpc_replay_30d_6h": NEW_FORECASTS_30D_6H,
        "mpc_replay_32d_6h_clip4": NEW_FORECASTS_32D_CLIP4,
        "mpc_replay_24d_6h_no_p0": NEW_FORECASTS_NO_P0}
    new_path = args.new or defaults[args.new_mode]
    output = args.output or (OUTPUT / "24d_6h_no_p0" if args.new_mode == "mpc_replay_24d_6h_no_p0"
        else OUTPUT / "32d_6h_clip4" if args.new_mode == "mpc_replay_32d_6h_clip4"
        else OUTPUT / "30d_6h" if args.new_mode == "mpc_replay_30d_6h"
        else OUTPUT / "24d_6h" if args.new_mode == "mpc_replay_24d_6h" else OUTPUT)
    horizon = ((6 if args.new_mode in ("mpc_replay_32d_6h_clip4",
        "mpc_replay_24d_6h_no_p0") else 8)
        if args.horizon is None else args.horizon)
    return analyze(args.old, new_path, output, args.quantiles, horizon,
        not args.include_warmup, args.history_days, args.new_mode, args.old_mode)


if __name__ == "__main__": main()
