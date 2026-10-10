"""Plot a saved BM1 demand forecast over a reconstructed probability heatmap.

Use: python -m src.exe.run_model.bm_1.plot_prediction --day 2022-08-22

Loads the existing hourly.csv and fit.json without rerunning baseline fitting or
parameter optimization. The heatmap is a display-smoothed representation of the
saved five-state ensemble, or the BM1 Fokker-Planck demand distribution when
the saved sigma_x is positive. Actual demand comes from saved I_observed or,
if absent, the input workbook using the saved normalization.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from src.model.bm_1 import bm_1
from src.params.bm_1 import lsq_params
from src.exe.run_model.bm_1.predict_daily import held_signal
from src.validation.exe.run_model.bm_1.plot_prediction import bm_1_plot_validation as checks


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "bm_1" / "prediction"


def reconstructed_demand(hourly, metadata, *, x_steps=101, fp_max_step=0.25, substeps=2):
    """Reconstruct the saved conditional demand law; do not refit model parameters."""
    p = metadata["theta"]
    states, weights, sigma = metadata["state_X0"], metadata["initial_weights"], metadata["sigma_x"]
    b = hourly["B"].to_numpy(dtype=float)
    u = hourly["U"].to_numpy(dtype=float)
    data = {"Time": np.arange(25, dtype=float), "B": np.r_[b, b[-1]], "U": np.r_[u, u[-1]]}
    f, g = lsq_params.response(p["alpha"]), lsq_params.response(p["beta"])

    if sigma == 0.0:
        ensemble = np.stack([lsq_params._predict(data, float(x0), p, substeps)[1][:24] for x0 in states])
        mean = weights @ ensemble
        return [ensemble[:, k] for k in range(24)], [weights for _ in range(24)], mean, "Five-state empirical ensemble"

    model = bm_1(held_signal(data["B"]), lambda t: 0.0 * np.asarray(t, dtype=float) + 0.5, (0.0, 24.0), 25)
    mass0 = np.zeros(x_steps, dtype=float)
    np.add.at(mass0, np.minimum((states * x_steps).astype(int), x_steps - 1), weights)
    parameters = [held_signal(data["U"]), f, g, p["C"], p["k"], p["Xi"]]
    t, x, density, masses, mean_x, var_x, price, baseline, mean_d, diagnostics = model.fokker_planc_reference_solver(
        mass0, parameters, sigma, t_eval=data["Time"], max_step=fp_max_step,
    )
    demand_values = []
    for k in range(24):
        delta = model.l(f(x), g(u[k]), p["k"])
        demand_values.append(np.asarray(model.D(b[k], p["Xi"], delta), dtype=float))
    return demand_values, list(masses[:24]), mean_d[:24], "Fokker-Planck demand distribution"


def demand_heatmap(values, weights, *, bins=180, bandwidth=0.012):
    """Create a per-hour normalized demand PDF using visualization-only smoothing."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    width = edges[1] - edges[0]
    pdf = np.empty((len(values), bins), dtype=float)
    for hour, (points, probabilities) in enumerate(zip(values, weights)):
        mass = np.histogram(points, bins=edges, weights=probabilities)[0]
        blurred = gaussian_filter1d(mass, sigma=bandwidth / width, mode="constant")
        pdf[hour] = blurred / (blurred.sum() * width)
    return edges, pdf


def plot_saved(day, *, output_root=OUTPUT, tag="MED", bins=180, bandwidth=0.012, x_steps=101, fp_max_step=0.25, substeps=2, import_actual=True, show=False, save=True):
    """Produce a demand probability heatmap with saved mean and observed demand."""
    start = checks.date(day)
    bins, bandwidth, x_steps, fp_max_step, substeps = checks.settings(bins, bandwidth, x_steps, fp_max_step, substeps)
    directory = Path(output_root) / f"{start:%Y-%m-%d}"
    hourly, metadata = checks.saved(directory, start)
    actual, actual_source = checks.actual_from_saved_or_import(hourly, metadata, metadata.get("tag", tag), import_actual)
    demand_values, weights, reconstructed_mean, method = reconstructed_demand(hourly, metadata,
        x_steps=x_steps, fp_max_step=fp_max_step, substeps=substeps)
    y_edges, pdf = demand_heatmap(demand_values, weights, bins=bins, bandwidth=bandwidth)

    predicted = hourly["I_predicted"].to_numpy(dtype=float)
    baseline = hourly["B"].to_numpy(dtype=float)
    errors = predicted - actual
    mse, bias, mae = float(np.mean(errors ** 2)), float(np.mean(errors)), float(np.mean(np.abs(errors)))
    difference = float(np.max(np.abs(reconstructed_mean - predicted)))
    dates = hourly.index
    date_edges = pd.date_range(start, periods=25, freq="h")
    values = np.concatenate((predicted, actual, baseline, np.array([np.quantile(a, [0.02, 0.98]) for a in demand_values]).ravel()))
    low, high = float(np.clip(values.min() - 0.06, 0.0, 1.0)), float(np.clip(values.max() + 0.06, 0.0, 1.0))
    if high - low < 0.2:
        midpoint = (low + high) / 2.0
        low, high = max(0.0, midpoint - 0.1), min(1.0, midpoint + 0.1)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.0]})
    ax, error_ax = axes
    mesh = ax.pcolormesh(mdates.date2num(date_edges), y_edges, pdf.T, cmap="magma", shading="flat", rasterized=True)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.012, fraction=0.027)
    colorbar.set_label("Demand probability density (normalized)")
    ax.plot(dates, predicted, color="#39e4f1", linewidth=2.3, label="Saved BM1 mean forecast")
    ax.plot(dates, actual, color="white", linewidth=2.2, linestyle="-", marker="o", markersize=3.8,
        markeredgecolor="black", markeredgewidth=0.5, label="Actual demand")
    ax.plot(dates, baseline, color="#a5ffb3", linewidth=1.4, linestyle="--", alpha=0.9, label="Saved baseline B(t)")
    ax.set_ylim(low, high)
    ax.set_xlim(date_edges[0], date_edges[-1])
    ax.set_ylabel("Normalized electricity demand")
    ax.set_title(f"BM1 demand forecast | {start:%Y-%m-%d}\n{method} (display-smoothed, bandwidth={bandwidth:g})")
    ax.legend(loc="upper right", framealpha=0.9)

    error_ax.plot(dates, errors, color="#b3243a", marker="o", markersize=3, linewidth=1.6, label="Prediction − actual")
    error_ax.axhline(0, color="0.35", linewidth=0.9)
    error_ax.fill_between(dates, 0, errors, color="#b3243a", alpha=0.14)
    error_ax.set_ylabel("Forecast error")
    error_ax.set_xlabel(f"Hour on {start:%Y-%m-%d}")
    error_ax.grid(True, alpha=0.22)
    error_ax.text(0.01, 0.96, f"MSE={mse:.5f}   MAE={mae:.5f}   bias={bias:+.5f}", va="top",
        ha="left", transform=error_ax.transAxes, fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.85})
    error_ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    error_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()

    destination = directory / "demand_heatmap.png"
    if save:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=190, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)
    return {"plot": destination if save else None, "mse": mse, "mae": mae, "bias": bias,
        "heatmap": pdf, "heatmap_density_edges": y_edges, "method": method,
        "actual_source": actual_source, "max_reconstruction_difference": difference}


def main():
    parser = ArgumentParser(description="Plot a saved BM1 forecast over its reconstructed demand-distribution heatmap.")
    parser.add_argument("--day", required=True, help="Target day YYYY-MM-DD (must have saved prediction files).")
    parser.add_argument("--output-root", type=Path, default=OUTPUT, help="Directory containing date-named prediction folders.")
    parser.add_argument("--tag", default="MED", help="Consumption group for actual-demand fallback imports (default MED).")
    parser.add_argument("--bins", type=int, default=180, help="Number of normalized-demand heatmap bins.")
    parser.add_argument("--bandwidth", type=float, default=0.012, help="Heatmap display smoothing width in normalized demand units.")
    parser.add_argument("--x-steps", type=int, default=101, help="State grid cells when saved sigma_x is positive.")
    parser.add_argument("--fp-max-step", type=float, default=0.25, help="Maximum FP internal step in hours.")
    parser.add_argument("--substeps", type=int, default=2, help="RK4 substeps used in original deterministic prediction.")
    parser.add_argument("--saved-only", action="store_true", help="Require I_observed in hourly.csv; do not import raw data.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively after saving.")
    args = parser.parse_args()
    result = plot_saved(args.day, output_root=args.output_root, tag=args.tag, bins=args.bins,
        bandwidth=args.bandwidth, x_steps=args.x_steps, fp_max_step=args.fp_max_step,
        substeps=args.substeps, import_actual=not args.saved_only, show=args.show)
    print(f"Plot: {result['plot']}")
    print(f"Demand source: {result['actual_source']}; heatmap: {result['method']}")
    print(f"MSE={result['mse']:.7f}; MAE={result['mae']:.7f}; bias={result['bias']:+.7f}")
    if result["max_reconstruction_difference"] > 1e-3:
        print(f"Note: reconstructed distribution mean differs from saved prediction by up to {result['max_reconstruction_difference']:.6f}; verify original --substeps/--x-steps/--fp-max-step settings.")
    return result


if __name__ == "__main__": main()
