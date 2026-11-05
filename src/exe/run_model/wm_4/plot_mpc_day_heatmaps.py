from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.calibrate_mpc_sigma import _prepare, evaluate_variance_components


ROOT = Path(__file__).resolve().parents[4]
MPC_OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_mpc"
DEFAULT_FORECASTS = MPC_OUTPUT / "mpc_forecasts.csv"
DEFAULT_STATES = MPC_OUTPUT / "mpc_states.csv"
DEFAULT_ESTIMATES = MPC_OUTPUT / "sigma_from_mpc" / "variance_component_estimates.csv"
OUTPUT = MPC_OUTPUT / "day_heatmaps"


def _estimate_row(path, mode):
    estimates = pd.read_csv(path)
    selected = estimates[estimates["mode"].astype(str) == str(mode)]
    if selected.empty: raise ValueError(f"No variance estimate with mode={mode!r} in {path}.")
    row = selected.iloc[0]
    return {
        "r0_variance": float(row["R0_variance"]),
        "sigma_R0": float(row["sigma_R0"]),
        "q": float(row["wiener_q"]),
        "sigma_W": float(row["sigma_W"]),
    }


def _target_day_rows(forecasts, states, day, max_horizon):
    day = pd.Timestamp(day).normalize()
    forecasts = forecasts.copy()
    states = states.copy()
    forecasts["origin"] = pd.to_datetime(forecasts["origin"])
    forecasts["origin_day"] = pd.to_datetime(forecasts["origin_day"]).dt.normalize()
    forecasts["target_time"] = pd.to_datetime(forecasts["target_time"])
    states["origin"] = pd.to_datetime(states["origin"])
    states["origin_day"] = pd.to_datetime(states["origin_day"]).dt.normalize()

    forecasts = forecasts[
        (forecasts["origin_day"] == day)
        & (forecasts["lead_hours"].astype(int) <= int(max_horizon))
    ].copy()
    states = states[states["origin_day"] == day].copy()

    origins = sorted(pd.unique(forecasts["origin"]))
    if len(origins) != 24: raise ValueError(f"Expected 24 MPC origins for {day:%Y-%m-%d}, found {len(origins)}.")
    if states["origin"].nunique() != 24: raise ValueError(f"Expected 24 MPC state rows for {day:%Y-%m-%d}, found {states['origin'].nunique()}.")

    forecasts = forecasts.sort_values(["origin", "lead_hours"]).reset_index(drop=True)
    states = states.sort_values("origin").reset_index(drop=True)
    return day, forecasts, states


def _normalized_density(mean, std, y_grid):
    mean = np.asarray(mean, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 1e-10)
    z = (y_grid[:, None] - mean[None, :]) / std[None, :]
    return np.exp(-0.5 * z ** 2)


def _y_grid(forecasts, states):
    mu = forecasts["I_mpc"].to_numpy(dtype=float)
    std = forecasts["calibrated_std"].to_numpy(dtype=float)
    observed = np.concatenate((
        forecasts["I_observed"].to_numpy(dtype=float),
        states["I_observed"].to_numpy(dtype=float),
    ))
    lower = min(float(np.min(mu - 4.0 * std)), float(np.min(observed)))
    upper = max(float(np.max(mu + 4.0 * std)), float(np.max(observed)))
    margin = 0.04 * max(upper - lower, 1e-6)
    return np.linspace(lower - margin, upper + margin, 260)


def _bar_edges(x):
    x = np.asarray(x, dtype=float)
    if x.size == 1: return np.array([x[0] - 0.45, x[0] + 0.45], dtype=float)
    middle = 0.5 * (x[:-1] + x[1:])
    return np.concatenate(([x[0] - 0.5 * (x[1] - x[0])], middle, [x[-1] + 0.5 * (x[-1] - x[-2])]))


def save_plot(day, forecasts, states, estimate, path, max_horizon):
    y_grid = _y_grid(forecasts, states)
    y_edges = _bar_edges(y_grid)
    fig, axes = plt.subplots(6, 4, figsize=(22, 28), sharey=True)
    axes = axes.ravel()
    mesh = None

    state_by_origin = states.set_index("origin")
    for panel, origin in enumerate(sorted(pd.unique(forecasts["origin"]))):
        ax = axes[panel]
        part = forecasts[forecasts["origin"] == origin].sort_values("lead_hours")
        lead = part["lead_hours"].to_numpy(dtype=float)
        mean = part["I_mpc"].to_numpy(dtype=float)
        std = part["calibrated_std"].to_numpy(dtype=float)
        density = _normalized_density(mean, std, y_grid)
        x_edges = _bar_edges(lead)

        mesh = ax.pcolormesh(
            x_edges, y_edges, density,
            shading="flat", vmin=0.0, vmax=1.0,
        )
        ax.plot(lead, mean, linewidth=1.0, label="ARM mean")
        ax.plot(lead, part["O"].to_numpy(dtype=float), linestyle="--", linewidth=0.9, label="O(t)")
        ax.scatter(lead, part["I_observed"].to_numpy(dtype=float), s=15, edgecolors="white", linewidths=0.4, label="Observed")

        current = state_by_origin.loc[pd.Timestamp(origin)]
        current_I = float(current["I_observed"])
        ax.scatter([0.0], [current_I], s=30, marker="D", edgecolors="white", linewidths=0.5, label="Current observation")
        ax.axvline(0.0, linewidth=1.1, linestyle=":")
        ax.set_xlim(-0.45, float(max_horizon) + 0.5)
        ax.set_xticks(np.arange(0, int(max_horizon) + 1, 2))
        ax.set_title(f"k={pd.Timestamp(origin):%H:%M}")
        ax.grid(False)

        clock = [pd.Timestamp(origin) + pd.Timedelta(hours=int(h)) for h in np.arange(0, int(max_horizon) + 1, 2)]
        ax.set_xticklabels([t.strftime("%H") for t in clock])

    for row in range(6): axes[row * 4].set_ylabel("Normalized input I")
    for col in range(20, 24): axes[col].set_xlabel("Physical hour from current k")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.975))
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.tolist(), fraction=0.012, pad=0.01)
        cbar.set_label("Normalized Gaussian density (peak = 1)")

    fig.suptitle(
        f"ARM (Affine Revised Model) receding MPC predictive distributions | {day:%Y-%m-%d} | "
        f"sigma_R0={estimate['sigma_R0']:.4f}, sigma_W={estimate['sigma_W']:.4f}",
        y=0.995,
    )
    fig.subplots_adjust(top=0.955, bottom=0.045, left=0.06, right=0.94, hspace=0.34, wspace=0.13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Plot all 24 hourly WM4 MPC predictive distributions for one target day as normalized Gaussian probability heatmap bars using the saved MPC forecasts and calibrated stochastic variance components.")
    parser.add_argument("--day", default="2022-08-22")
    parser.add_argument("--forecasts", default=str(DEFAULT_FORECASTS))
    parser.add_argument("--states", default=str(DEFAULT_STATES))
    parser.add_argument("--estimates", default=str(DEFAULT_ESTIMATES))
    parser.add_argument("--estimate-mode", choices=("all", "chronological_holdout"), default="all")
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    if args.max_horizon < 1 or args.max_horizon > 8: raise ValueError("max-horizon must lie in [1, 8].")

    forecast_path, state_path, estimate_path = Path(args.forecasts), Path(args.states), Path(args.estimates)
    for path in (forecast_path, state_path, estimate_path):
        if not path.exists(): raise FileNotFoundError(path)

    raw_forecasts = pd.read_csv(forecast_path)
    prepared = _prepare(raw_forecasts, args.steps_per_hour)
    estimate = _estimate_row(estimate_path, args.estimate_mode)
    calibrated, _ = evaluate_variance_components(prepared, estimate["r0_variance"], estimate["q"])
    states = pd.read_csv(state_path)
    day, forecasts, states = _target_day_rows(calibrated, states, args.day, args.max_horizon)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{day:%Y-%m-%d}_mpc_{int(args.max_horizon)}h"
    plot_path = output / f"{stem}_24_heatmaps.png"
    csv_path = output / f"{stem}_forecasts.csv"

    forecasts.to_csv(csv_path, index=False)
    save_plot(day, forecasts, states, estimate, plot_path, args.max_horizon)

    print(f"day={day:%Y-%m-%d}")
    print(f"origins={forecasts['origin'].nunique()}, forecasts={len(forecasts)}, horizon={args.max_horizon}h")
    print(f"sigma_R0={estimate['sigma_R0']:.8f}, sigma_W={estimate['sigma_W']:.8f}, q={estimate['q']:.8f}")
    print("Each panel is one hourly MPC update. The vertical bar at lead 0 marks the current hour k; forecast heatmap bars use normalized Gaussian density with peak 1.")
    print(f"forecast_csv={csv_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__": main()
