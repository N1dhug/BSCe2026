"""Inspect entire WM4 1..8h MPC forecasts and their cumulative uncertainty.

One forecast origin contains ONE realized trajectory. Its squared errors and
cumulative MSE can be compared to model variance, but they are not an empirical
estimate of that single trajectory's conditional variance.

Additionally, aggregate complete forecasts into bins of predicted 8h RMS
uncertainty to assess conditional calibration with many realized trajectories.
All calculations use saved MPC predictions. No parameter or state refitting.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_mpc_intervals import FORECASTS, load_forecasts
from src.exe.run_model.wm_4.plot_mpc_variance import select_origins


OUTPUT = FORECASTS.parent / "trajectory_analysis"
PARTS = ("initial_F_component_variance", "initial_R_component_variance",
    "initial_FR_component_variance", "process_variance")
PARAMS = ("r_F", "r_U", "r_R", "qp_loss", "diffusion_sigma",
    "F_initialized_origin", "R_initialized_origin",
    "initial_F_variance", "initial_R_variance", "initial_FR_covariance")


def prepare_trajectories(frame, horizon=8):
    """Compute exact-lead and cumulative scores separately for every MPC origin."""
    horizon = int(horizon)
    if not 1 <= horizon <= 8: raise ValueError("horizon must be between 1 and 8.")
    needed = {"origin", "lead_hours", "I_observed", "I_mpc", "predicted_variance",
        "initial_variance", "process_variance"} | set(PARTS)
    if missing := needed - set(frame.columns):
        raise ValueError(f"MPC trajectory analysis missing columns: {sorted(missing)}.")
    data = frame.copy()
    data["origin"] = pd.to_datetime(data["origin"])
    data["lead_hours"] = pd.to_numeric(data["lead_hours"], errors="raise").astype(int)
    columns = list(needed - {"origin"})
    if not np.all(np.isfinite(data[columns].to_numpy(dtype=float))):
        raise ValueError("MPC trajectory analysis contains nonfinite prediction values.")
    if np.any(data["predicted_variance"] < 0.0):
        raise ValueError("Predicted variance must be nonnegative.")
    if data.duplicated(["origin", "lead_hours"]).any():
        raise ValueError("MPC forecast data contain duplicate origin/lead pairs.")
    data = data[data["lead_hours"].between(1, horizon)].sort_values(
        ["origin", "lead_hours"]).reset_index(drop=True)
    complete = data.groupby("origin")["lead_hours"].agg(
        lambda a: np.array_equal(a.to_numpy(), np.arange(1, horizon + 1)))
    data = data.loc[data["origin"].isin(complete[complete].index)].copy()
    if data.empty: raise ValueError("No complete MPC origins for the requested horizon.")
    if not np.allclose(data["initial_variance"] + data["process_variance"],
        data["predicted_variance"], rtol=1e-5, atol=1e-8):
        raise ValueError("Initial-state plus Wiener variance must equal predicted variance.")
    if not np.allclose(data[list(PARTS)].sum(axis=1),
        data["predicted_variance"], rtol=1e-5, atol=1e-8):
        raise ValueError("F0, R0, signed F/R covariance and Wiener terms must sum to total variance.")

    data["error"] = data["I_observed"] - data["I_mpc"]
    data["squared_error"] = data["error"] ** 2
    by_origin = data.groupby("origin", sort=False)
    for source, target in (("predicted_variance", "cumulative_predicted_variance"),
        ("squared_error", "cumulative_realized_mse"),
        ("initial_variance", "cumulative_initial_variance"),
        ("process_variance", "cumulative_wiener_variance"),
        ("initial_F_component_variance", "cumulative_F0_variance"),
        ("initial_R_component_variance", "cumulative_R0_variance"),
        ("initial_FR_component_variance", "cumulative_FR_covariance")):
        data[target] = by_origin[source].cumsum() / data["lead_hours"]
    data["cumulative_rmse"] = np.sqrt(data["cumulative_realized_mse"])
    data["cumulative_predicted_rms_std"] = np.sqrt(data["cumulative_predicted_variance"])
    data["cumulative_mse_ratio"] = np.divide(
        data["cumulative_predicted_variance"], data["cumulative_realized_mse"],
        out=np.full(len(data), np.nan), where=data["cumulative_realized_mse"] > 0)
    data["cumulative_mse_difference"] = (
        data["cumulative_predicted_variance"] - data["cumulative_realized_mse"])
    return data


def origin_scores(trajectories, horizon=8):
    """One summary row per origin; include fitted model coefficients when saved."""
    summary = trajectories.loc[trajectories["lead_hours"] == horizon].copy()
    if summary.empty: raise ValueError("No complete 1..H trajectories to summarize.")
    kept = ["origin", "cumulative_predicted_variance", "cumulative_realized_mse",
        "cumulative_rmse", "cumulative_predicted_rms_std", "cumulative_mse_ratio",
        "cumulative_mse_difference", "cumulative_initial_variance",
        "cumulative_wiener_variance", "cumulative_F0_variance",
        "cumulative_R0_variance", "cumulative_FR_covariance"]
    kept += [col for col in PARAMS if col in summary]
    summary = summary[kept].rename(columns={
        "cumulative_rmse": "rmse",
        "cumulative_predicted_rms_std": "predicted_rms_std",
        "cumulative_predicted_variance": "predicted_variance",
        "cumulative_realized_mse": "realized_mse",
        "cumulative_mse_ratio": "predicted_to_realized_mse_ratio",
    })
    summary["origin_day"] = summary["origin"].dt.normalize()
    summary["origin_hour"] = summary["origin"].dt.hour
    summary["overconfidence_ratio"] = np.divide(
        summary["realized_mse"], summary["predicted_variance"],
        out=np.full(len(summary), np.nan),
        where=summary["predicted_variance"] > 0)
    summary["absolute_mse_difference"] = abs(summary["realized_mse"] -
        summary["predicted_variance"])
    return summary.sort_values("origin").reset_index(drop=True)


def select_examples(scores, top_k=3, origin=None):
    """Select illustrative best, worst, overconfident, conservative trajectories."""
    if origin is not None:
        timestamp = pd.Timestamp(origin)
        matches = scores.loc[scores["origin"] == timestamp]
        if len(matches) != 1:
            raise ValueError(f"No complete MPC forecast beginning at {timestamp}. "
                "Pass an origin timestamp from the generated origin_summary.csv.")
        return [("requested", timestamp)]
    count = int(top_k)
    if count < 1: raise ValueError("top_k must be at least one.")
    results, used = [], set()
    criteria = (
        ("best_rmse", "rmse", True),
        ("worst_rmse", "rmse", False),
        ("overconfident", "overconfidence_ratio", False),
        ("conservative", "overconfidence_ratio", True),
    )
    for label, column, ascending in criteria:
        ordered = scores.loc[np.isfinite(scores[column])].sort_values(
            [column, "origin"], ascending=[ascending, True])
        selected = 0
        for stamp in ordered["origin"]:
            if stamp in used: continue
            used.add(stamp)
            results.append((label, pd.Timestamp(stamp)))
            selected += 1
            if selected == count: break
    return results


def plot_trajectory(frame, origin, output, label="MPC origin"):
    """Full eight-hour trajectory, cumulative budget, exact error and MSE ratio."""
    forecast = frame.loc[frame["origin"] == pd.Timestamp(origin)].sort_values("lead_hours")
    if forecast.empty: raise ValueError(f"No forecast saved for origin {origin}.")
    h = forecast["lead_hours"].to_numpy(dtype=int)
    obs = forecast["I_observed"].to_numpy(dtype=float)
    mean = forecast["I_mpc"].to_numpy(dtype=float)
    variance = forecast["predicted_variance"].to_numpy(dtype=float)
    std = np.sqrt(variance)
    exact_error = forecast["squared_error"].to_numpy(dtype=float)
    c_pred = forecast["cumulative_predicted_variance"].to_numpy(dtype=float)
    c_mse = forecast["cumulative_realized_mse"].to_numpy(dtype=float)
    init = forecast["cumulative_initial_variance"].to_numpy(dtype=float)
    wiener = forecast["cumulative_wiener_variance"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax = axes[0, 0]
    ax.plot(h, obs, "s-", label="Observed demand", markersize=4)
    ax.plot(h, mean, "o-", label="WM4 deterministic prediction", markersize=4)
    ax.fill_between(h, mean - 1.96 * std, mean + 1.96 * std,
        alpha=0.18, label="Predicted 95% marginal interval")
    ax.set_title("Eight-hour demand trajectory (no state reinitialization)")
    ax.set_ylabel("Normalized consumption")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.fill_between(h, 0, init, alpha=0.40,
        label="Cumulative initial-state variance")
    ax.fill_between(h, init, init + wiener, alpha=0.45,
        label="Cumulative Wiener variance")
    ax.plot(h, c_pred, "ko-", markersize=4, label="Cumulative predicted variance")
    ax.plot(h, c_mse, "rs--", markersize=4,
        label="Cumulative realized MSE (one trajectory)")
    ax.set_title("Mean variance vs mean squared error over leads 1..H")
    ax.set_ylabel("Normalized demand squared")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(h, variance, "o-", label="Predicted variance at lead h")
    ax.plot(h, exact_error, "s--", label="Realized squared error at lead h")
    ax.plot(h, forecast["initial_F_component_variance"], ":", label="F0 variance")
    ax.plot(h, forecast["initial_R_component_variance"], ":", label="R0 variance")
    ax.plot(h, forecast["initial_FR_component_variance"], ":", label="F/R covariance (signed)")
    ax.plot(h, forecast["process_variance"], ":", label="Wiener variance")
    ax.set_title("Exact-lead squared errors and uncertainty components")
    ax.set_xlabel("Forecast lead (hours)")
    ax.set_ylabel("Normalized demand squared")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    # Ratio > 1 means observed errors exceeded predicted variance.
    ratio = np.divide(c_mse, c_pred, out=np.full(len(h), np.nan), where=c_pred > 0)
    ax.plot(h, ratio, "o-", label="Realized MSE / predicted variance")
    ax.axhline(1.0, color="black", ls="--", lw=1, label="Expected ratio = 1")
    ax.set_title("Cumulative uncertainty calibration of one realization")
    ax.set_xlabel("Forecast lead (hours)")
    ax.set_ylabel("Realized / predicted squared-error scale")
    ax.legend(fontsize=9)

    for ax in axes.flat:
        ax.set_xticks(h)
        ax.grid(alpha=0.22)
    row = forecast.iloc[0]
    ptxt = " | ".join(f"{p}={row[p]:.3g}" for p in
        ("r_F", "r_U", "r_R", "F_initialized_origin", "R_initialized_origin")
        if p in row and pd.notna(row[p]))
    text = f"Fitted parameters and initial state: {ptxt}" if ptxt else "Parameters fixed within this MPC origin"
    fig.suptitle(f"WM4 | {label} | origin {pd.Timestamp(origin):%Y-%m-%d %H:%M}\n{text}",
        fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.3, w_pad=1.6)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def reliability_bins(scores, trajectories, quantiles=5):
    """Conditional calibration grouped by predicted uncertainty of full MPC origins."""
    quantiles = int(quantiles)
    if not 2 <= quantiles <= len(scores):
        raise ValueError(f"Quantiles must be between 2 and the number of origins ({len(scores)}).")
    scores = scores.copy().sort_values("origin")
    ranks = scores["predicted_variance"].rank(method="first")
    scores["uncertainty_quantile"] = pd.qcut(ranks, quantiles, labels=False).astype(int) + 1
    data = trajectories.merge(scores[["origin", "uncertainty_quantile"]], on="origin",
        validate="many_to_one")
    rows = []
    for (q, h), part in data.groupby(["uncertainty_quantile", "lead_hours"], sort=True):
        # Aggregate the SAME complete set of origins at every h. Hourly
        # origins overlap and are not statistically independent.
        rows.append({
            "uncertainty_quantile": int(q), "horizon_hours": int(h),
            "n_origins": int(part["origin"].nunique()),
            "mean_cumulative_predicted_variance": float(part["cumulative_predicted_variance"].mean()),
            "mean_cumulative_realized_mse": float(part["cumulative_realized_mse"].mean()),
            "mean_cumulative_initial_variance": float(part["cumulative_initial_variance"].mean()),
            "mean_cumulative_wiener_variance": float(part["cumulative_wiener_variance"].mean()),
            "mean_cumulative_F0_variance": float(part["cumulative_F0_variance"].mean()),
            "mean_cumulative_R0_variance": float(part["cumulative_R0_variance"].mean()),
            "mean_cumulative_FR_covariance": float(part["cumulative_FR_covariance"].mean()),
            "mean_exact_predicted_variance": float(part["predicted_variance"].mean()),
            "mean_exact_realized_squared_error": float(part["squared_error"].mean()),
        })
    grouped = pd.DataFrame(rows)
    grouped["ratio_realized_to_predicted"] = np.divide(
        grouped["mean_cumulative_realized_mse"],
        grouped["mean_cumulative_predicted_variance"],
        out=np.full(len(grouped), np.nan),
        where=grouped["mean_cumulative_predicted_variance"] > 0)
    return scores, grouped


def plot_reliability(grouped, output):
    """Compare the full cumulative uncertainty curve within each uncertainty bin."""
    qs = sorted(grouped["uncertainty_quantile"].unique())
    n = len(qs)
    fig, axes = plt.subplots(2, int(np.ceil((n + 1) / 2)),
        figsize=(17, 9))
    axes = np.asarray(axes).ravel()
    for j, q in enumerate(qs):
        g = grouped.loc[grouped["uncertainty_quantile"] == q].sort_values("horizon_hours")
        h = g["horizon_hours"]
        pred = g["mean_cumulative_predicted_variance"]
        actual = g["mean_cumulative_realized_mse"]
        initial = g["mean_cumulative_initial_variance"]
        wiener = g["mean_cumulative_wiener_variance"]
        ax = axes[j]
        ax.fill_between(h.to_numpy(), 0, initial.to_numpy(), alpha=0.25,
            label="Initial-state uncertainty")
        ax.fill_between(h.to_numpy(), initial.to_numpy(), (initial + wiener).to_numpy(),
            alpha=0.25, label="Wiener diffusion")
        ax.plot(h, pred, "ko-", markersize=4, label="Predicted variance")
        ax.plot(h, actual, "rs--", markersize=4, label="Observed mean squared error")
        ax.set_title(f"Predicted uncertainty Q{q} | {int(g.iloc[0]['n_origins'])} origins")
        ax.grid(alpha=0.2)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    ax = axes[n]
    last = grouped.loc[grouped["horizon_hours"] == grouped["horizon_hours"].max()]
    x = last["mean_cumulative_predicted_variance"].to_numpy()
    y = last["mean_cumulative_realized_mse"].to_numpy()
    ax.plot(x, y, "o-", markersize=7, label="Uncertainty quantiles")
    bound = float(max(np.max(x), np.max(y))) * 1.07
    ax.plot([0, bound], [0, bound], "k--", alpha=0.5, label="Perfect calibration")
    for q, v, w in zip(last["uncertainty_quantile"], x, y):
        ax.annotate(f"Q{q}", (v, w), xytext=(5, 3), textcoords="offset points")
    ax.set_xlim(0, bound)
    ax.set_ylim(0, bound)
    ax.set_title("Eight-hour variance reliability across origins")
    ax.set_xlabel("Mean predicted 1..8h variance")
    ax.set_ylabel("Mean realized 1..8h MSE")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    for j in range(n + 1, len(axes)): axes[j].axis("off")
    for ax in axes[:n]:
        ax.set_xlabel("Cumulative forecast horizon (hours)")
        ax.set_ylabel("Normalized demand squared")
        ax.set_xticks(grouped["horizon_hours"].unique())
    fig.suptitle("WM4 MPC | cumulative variance calibration by predicted uncertainty"
        "\nMeans across origins (not single-trajectory empirical variance)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.1)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=185, bbox_inches="tight")
    plt.close(fig)
    return output


def parameter_associations(scores):
    """Exploratory rank associations, not parameter posterior uncertainty."""
    metrics = ("rmse", "overconfidence_ratio", "absolute_mse_difference")
    rows = []
    for par in PARAMS:
        if par not in scores: continue
        for metric in metrics:
            a, b = pd.to_numeric(scores[par], errors="coerce"), scores[metric]
            usable = np.isfinite(a) & np.isfinite(b)
            subset = scores.loc[usable]
            correlation = subset[par].corr(subset[metric], method="spearman") if (
                len(subset) >= 3 and subset[par].nunique() > 1 and subset[metric].nunique() > 1) else np.nan
            rows.append({"parameter": par, "metric": metric, "origins": int(len(subset)),
                "spearman_correlation": float(correlation)})
    return pd.DataFrame(rows)


def analyze(forecasts=FORECASTS, output=OUTPUT, horizon=8, fully_mpc=False,
    history_windows=5, quantiles=5, top_k=3, origin=None,
    expected_sigma_fit_mode="mpc_replay"):
    frame, dropped, _ = load_forecasts(forecasts, horizon,
        expected_sigma_fit_mode=expected_sigma_fit_mode)
    frame = select_origins(frame, fully_mpc=fully_mpc, history_windows=history_windows)
    trajectories = prepare_trajectories(frame, horizon)
    scores = origin_scores(trajectories, horizon)
    scores, bins = reliability_bins(scores, trajectories, quantiles)
    chosen = select_examples(scores, top_k=top_k, origin=origin)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(output / "origin_cumulative_curves.csv", index=False)
    scores.to_csv(output / "origin_summary.csv", index=False)
    bins.to_csv(output / "conditional_variance_reliability.csv", index=False)
    parameter_associations(scores).to_csv(output / "parameter_associations.csv", index=False)
    reliability = plot_reliability(bins, output / "conditional_variance_reliability.png")
    selections = []
    for label, stamp in chosen:
        name = f"{label}_{stamp:%Y-%m-%d_%H-%M}.png"
        figure = plot_trajectory(trajectories, stamp, output / "examples" / name, label)
        selections.append({"category": label, "origin": stamp, "figure": str(figure)})
    pd.DataFrame(selections).to_csv(output / "selected_origins.csv", index=False)
    print(f"WM4 full 1..{horizon}h trajectories: {len(scores)} MPC origins "
        f"({dropped} incomplete discarded); fully MPC={fully_mpc}.")
    print(f"Reliability across predicted-variance quantiles: {reliability}")
    print("Examples: " + ", ".join(
        f"{entry['category']}={entry['origin']:%Y-%m-%d %H:%M}" for entry in selections))
    print(f"Cumulative curves, origin scores and fitted-parameter associations: {output}")
    print("Note: individual trajectories provide realized MSE, not observed conditional "
        "variance; parameter-estimation uncertainty is not included in WM4's "
        "saved stochastic forecast variance.")
    return {"trajectories": trajectories, "scores": scores, "bins": bins,
        "reliability": reliability, "selected": selections, "output": output}


def main():
    parser = ArgumentParser(description="Diagnose full eight-hour WM4 MPC trajectories, cumulative variance and squared errors.")
    parser.add_argument("--forecasts", type=Path, default=FORECASTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--horizon", type=int, default=8, help="Horizon in hours (1..8).")
    parser.add_argument("--fully-mpc", action="store_true",
        help="Exclude origins still using historical oracle bootstrap for diffusion or initial states.")
    parser.add_argument("--history-windows", type=int, default=5)
    parser.add_argument("--quantiles", type=int, default=5,
        help="Number of predicted-uncertainty reliability bins.")
    parser.add_argument("--top-k", type=int, default=3,
        help="Number of origins to show in each best/worst/overconfident/conservative category.")
    parser.add_argument("--origin", type=str,
        help="Plot this exact MPC origin timestamp, e.g. '2022-05-17 14:00'.")
    parser.add_argument("--expected-sigma-fit-mode", choices=("mpc_replay", "mpc_replay_24d_24h", "mpc_replay_24d_6h", "mpc_replay_30d_6h", "mpc_replay_28d_6h", "mpc_replay_32d_6h_clip4", "any"), default="mpc_replay")
    args = parser.parse_args()
    return analyze(args.forecasts, args.output, args.horizon, args.fully_mpc,
        args.history_windows, args.quantiles, args.top_k, args.origin,
        args.expected_sigma_fit_mode)


if __name__ == "__main__": main()
