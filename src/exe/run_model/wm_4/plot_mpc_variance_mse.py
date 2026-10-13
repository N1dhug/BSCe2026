"""Plot WM4 cumulative and exact-lead MPC variance / MSE calibration in a 2x2 grid.

Rows: cumulative pooled forecast horizons 1..H, then exact prediction leads H.
Columns: predicted versus empirical error variance (with F0/R0/covariance/Wiener
decomposition), then expected squared prediction error versus observed MSE.

The model predicts E[(I - I_hat)^2] = predicted_variance if its conditional
forecast mean is unbiased; no empirical bias is used to tune this prediction.
Actual MSE decomposes exactly into population error variance plus squared
aggregate bias. Actual error variance uses ddof=1 for the comparison.
Only archived MPC forecasts are read: no WM4 refitting.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_mpc_intervals import FORECASTS, load_forecasts
from src.exe.run_model.wm_4.plot_mpc_variance import select_origins, variance_scores


OUTPUT = FORECASTS.parent / "variance_calibration"
COMPONENTS = ("initial_F_component_variance", "initial_R_component_variance",
    "initial_FR_component_variance", "process_variance")


def budget_scores(frame):
    """Calculate one comparable empirical/predicted forecast-error budget."""
    metrics = variance_scores(frame)
    needed = {"initial_variance", "predicted_variance"} | set(COMPONENTS)
    if missing := needed - set(frame.columns):
        raise ValueError(f"Forecasts missing state/diffusion variance contributions: {sorted(missing)}.")
    components = frame[list(COMPONENTS)].to_numpy(dtype=float)
    if not np.all(np.isfinite(components)): raise ValueError("Nonfinite model variance contributions.")
    predicted = float(metrics["predicted_variance"])
    by_component = np.mean(components, axis=0)
    if not np.isclose(np.sum(by_component), predicted, atol=1e-8, rtol=1e-5):
        raise ValueError("Initial-state F/R/covariance and Wiener variances do not sum to predicted variance.")
    metrics.update({
        "F0_variance": float(by_component[0]),
        "R0_variance": float(by_component[1]),
        "FR_covariance_contribution": float(by_component[2]),
        "Wiener_variance": float(by_component[3]),
        "initial_state_variance": float(frame["initial_variance"].mean()),
        "predicted_mse_unbiased": predicted,
        "actual_mse": float(metrics["observed_mse"]),
        "actual_population_error_variance": float(metrics["empirical_error_variance_population"]),
        "actual_bias_squared": float(metrics["bias_squared"]),
        "mse_ratio_predicted_to_observed": (
            predicted / metrics["observed_mse"] if metrics["observed_mse"] > 0 else np.nan),
    })
    if not np.isclose(metrics["actual_population_error_variance"] + metrics["actual_bias_squared"],
        metrics["actual_mse"], rtol=1e-10, atol=1e-10):
        raise ValueError("Observed MSE must equal population error variance plus squared bias.")
    return metrics


def summarize_budgets(frame, horizon):
    """Exact h and cumulative pooled 1..H budgets, each with matching origins."""
    exact, cumulative = [], []
    for h in range(1, int(horizon) + 1):
        lead = frame[frame["lead_hours"] == h]
        pooled = frame[frame["lead_hours"] <= h]
        if len(lead) < 2: raise ValueError(f"At least two MPC origins are required at lead {h}.")
        exact.append({"lead_hours": h, **budget_scores(lead)})
        cumulative.append({"horizon_hours": h, **budget_scores(pooled)})
    return pd.DataFrame(exact), pd.DataFrame(cumulative)


def _variance_panel(ax, data, x, title, *, show_quality=False):
    """Stack physical variance components while retaining signed F/R covariance."""
    h = data[x].to_numpy(dtype=int)
    F = data["F0_variance"].to_numpy(dtype=float)
    R = data["R0_variance"].to_numpy(dtype=float)
    cross = data["FR_covariance_contribution"].to_numpy(dtype=float)
    W = data["Wiener_variance"].to_numpy(dtype=float)
    initial = F + R + cross
    total = initial + W

    ax.fill_between(h, 0, F, color="tab:blue", alpha=0.46, label="F0 reconstruction")
    ax.fill_between(h, F, F + R, color="tab:orange", alpha=0.53, label="R0 reconstruction")
    ax.fill_between(h, F + R, initial, color="tab:purple", alpha=0.58,
        label="F/R covariance (signed)")
    ax.fill_between(h, initial, total, color="tab:green", alpha=0.42,
        label="Accumulated Wiener")
    ax.plot(h, data["predicted_variance"], color="black", linewidth=1.7,
        marker="o", markersize=5, label="Predicted variance")
    ax.plot(h, data["empirical_error_variance"], color="tab:red", linewidth=1.8,
        linestyle="--", marker="s", markersize=5, label="Empirical error variance (ddof=1)")
    if show_quality:
        for hour, observed, ratio in zip(h, data["empirical_error_variance"],
            data["variance_ratio_predicted_to_empirical"]):
            if np.isfinite(ratio):
                ax.annotate(f"{ratio:.0%}", (hour, observed), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Variance (normalized demand²)")
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper left", ncol=2)


def _mse_panel(ax, data, x, title, *, show_quality=False):
    """Display the empirical MSE decomposition and theoretical model MSE."""
    h = data[x].to_numpy(dtype=int)
    error_var = data["actual_population_error_variance"].to_numpy(dtype=float)
    bias = data["actual_bias_squared"].to_numpy(dtype=float)
    mse = error_var + bias
    ax.fill_between(h, 0, error_var, color="tab:orange", alpha=0.28,
        label="Actual error variance (ddof=0)")
    ax.fill_between(h, error_var, mse, color="tab:purple", alpha=0.55,
        label="Squared aggregate bias")
    ax.plot(h, data["predicted_mse_unbiased"], color="black", marker="o",
        linewidth=1.9, markersize=5, label="Predicted MSE (unbiased WM4)")
    ax.plot(h, data["actual_mse"], color="tab:red", marker="s",
        linestyle="--", linewidth=1.8, markersize=5, label="Observed MSE")
    if show_quality:
        for hour, observed, ratio in zip(h, data["actual_mse"],
            data["mse_ratio_predicted_to_observed"]):
            if np.isfinite(ratio):
                ax.annotate(f"{ratio:.0%}", (hour, observed), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("MSE (normalized demand²)")
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper left", ncol=2)


def plot_budget(exact, cumulative, output, *, fully_mpc=False):
    """One 2x2 plot: cumulative versus exact lead, variance versus MSE."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), sharex=True, sharey="row")
    _variance_panel(axes[0, 0], cumulative, "horizon_hours",
        "Cumulative 1..H | predicted vs actual variance")
    _mse_panel(axes[0, 1], cumulative, "horizon_hours",
        "Cumulative 1..H | predicted vs observed MSE")
    _variance_panel(axes[1, 0], exact, "lead_hours",
        "Exact lead h | variance and prediction quality", show_quality=True)
    _mse_panel(axes[1, 1], exact, "lead_hours",
        "Exact lead h | MSE and prediction quality", show_quality=True)

    hours = exact["lead_hours"].to_numpy(dtype=int)
    for ax in axes.flat:
        ax.set_xticks(hours)
        ax.set_xlim(float(hours[0]) - 0.35, float(hours[-1]) + 0.35)
        ax.set_ylim(bottom=0)
    for ax in axes[1]: ax.set_xlabel("Forecast horizon H (hours)")
    subtitle = "fully MPC-calibrated" if fully_mpc else "all complete origins"
    fig.suptitle("WM4 MPC | uncertainty budget and forecast-error calibration"
        f" | {subtitle}\n"
        "Per-hour labels: predicted / observed (%). Predicted MSE assumes unbiased conditional means.",
        fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.7, w_pad=2.5)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return output


def analyze(forecasts=FORECASTS, output=OUTPUT, horizon=8, fully_mpc=False, history_windows=5,
    *, expected_f0_variance_mode="mpc_reconstruction",
    expected_r0_variance_mode="oracle_reconstruction",
    expected_sigma_fit_mode="mpc_replay"):
    frame, incomplete, _ = load_forecasts(forecasts, horizon,
        expected_f0_variance_mode=expected_f0_variance_mode,
        expected_r0_variance_mode=expected_r0_variance_mode,
        expected_sigma_fit_mode=expected_sigma_fit_mode)
    frame = select_origins(frame, fully_mpc=fully_mpc, history_windows=history_windows)
    exact, cumulative = summarize_budgets(frame, horizon)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    exact.to_csv(output / "variance_mse_exact_lead.csv", index=False)
    cumulative.to_csv(output / "variance_mse_cumulative_1_to_H.csv", index=False)
    figure = plot_budget(exact, cumulative, output / "variance_mse_2x2.png",
        fully_mpc=fully_mpc)
    print(f"WM4 variance/MSE 2x2 | {frame['origin'].nunique()} origins"
        f" | fully MPC={fully_mpc} | {incomplete} incomplete origins excluded")
    columns = ["lead_hours", "predicted_variance", "empirical_error_variance",
        "predicted_mse_unbiased", "actual_mse",
        "variance_ratio_predicted_to_empirical", "mse_ratio_predicted_to_observed"]
    print(exact[columns].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"Saved {figure} and the corresponding exact/cumulative budget CSVs")
    return {"exact": exact, "cumulative": cumulative, "figure": figure, "output": output}


def main():
    parser = ArgumentParser(description="Plot WM4 cumulative and per-hour variance/MSE budgets in one 2x2 grid.")
    parser.add_argument("--forecasts", type=Path, default=FORECASTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--horizon", type=int, default=8, help="Forecast lead in hours, 1..8.")
    parser.add_argument("--fully-mpc", action="store_true",
        help="Exclude origins with oracle bootstrap in F0/R0 reconstruction or diffusion calibration.")
    parser.add_argument("--history-windows", type=int, default=5)
    parser.add_argument("--expected-f0-variance-mode",
        choices=("mpc_reconstruction", "oracle_spread", "any"), default="mpc_reconstruction")
    parser.add_argument("--expected-r0-variance-mode",
        choices=("oracle_reconstruction", "baseline_residual", "zero", "any"),
        default="oracle_reconstruction")
    parser.add_argument("--expected-sigma-fit-mode",
        choices=("mpc_replay", "mpc_replay_24d_24h", "mpc_replay_24d_6h", "mpc_replay_30d_6h", "mpc_replay_28d_6h", "mpc_replay_32d_6h_clip4", "historical_p0", "zero_p0", "any"), default="mpc_replay")
    args = parser.parse_args()
    if not 1 <= args.horizon <= 8: parser.error("horizon must be in [1, 8].")
    return analyze(args.forecasts, args.output, args.horizon, args.fully_mpc, args.history_windows,
        expected_f0_variance_mode=args.expected_f0_variance_mode,
        expected_r0_variance_mode=args.expected_r0_variance_mode,
        expected_sigma_fit_mode=args.expected_sigma_fit_mode)


if __name__ == "__main__": main()
