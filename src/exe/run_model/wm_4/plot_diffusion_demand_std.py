"""Plot historical ARM model-implied demand standard deviation by (N,k).

For each saved drift/diffusion configuration, the hourly standard deviation is
sqrt(tau^2 + sigma^2*K_j). The plotted tile is the arithmetic mean of these
hourly standard deviations across all selected origins and scored hours.

Reads saved drift and diffusion CSVs in their original sweep order. No data
import, deterministic fitting, stochastic refitting or oracle optimization.
"""
from argparse import ArgumentParser
from collections import defaultdict
from csv import DictReader
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_moving_horizon_diffusion import (
    OUTPUT as DIFFUSION_OUTPUT, TITLES, variance_kernel,
)
from src.exe.run_model.wm_4.plot_moving_horizon_quality import configuration, bounds, selected
from src.exe.run_model.wm_4.sweep_weekday_horizon import OUTPUT as SWEEP_OUTPUT


BASELINES = ("weekly_average", "weekly_temperature")
PNG_NAME = "diffusion_demand_std_heatmaps.png"
CSV_NAME = "diffusion_demand_std_summary.csv"


def key(row):
    """The diffusion fitter copies these four CSV fields from each drift row."""
    return (str(row["day"]), str(row["baseline"]),
            int(row["training_weekdays"]), int(row["horizon_hours"]))


def matched_rows(drift_file, diffusion_file):
    """Stream filtered diffusion rows against the original ordered drift sweep.

    The diffusion executable preserves the order of the input drift sweep and
    may omit rows because of baseline/horizon/cohort filters or test-run limits.
    This merge uses constant memory and rejects unmatched/duplicate saved keys.
    """
    with Path(drift_file).open(encoding="utf-8-sig", newline="") as fd, \
         Path(diffusion_file).open(encoding="utf-8-sig", newline="") as fq:
        drifts, noise = iter(DictReader(fd)), iter(DictReader(fq))
        pending = next(noise, None)
        prior = None
        for drift in drifts:
            if pending is None: break
            if key(drift) != key(pending): continue
            current = key(pending)
            if current == prior:
                raise ValueError(f"Duplicate diffusion configuration: {current}")
            yield drift, pending
            prior = current
            pending = next(noise, None)
        if pending is not None:
            raise ValueError("A diffusion configuration could not be matched in drift CSV order: "
                             f"{key(pending)}. Check that both CSV files come from the same sweep.")


def interval_std(drift, noise, steps_per_hour):
    """Model-implied hourly demand std, conditional on fitted initial state."""
    if key(drift) != key(noise):
        raise ValueError("Drift and diffusion rows describe different configurations.")
    k = int(drift["horizon_hours"])
    tau2, q = float(noise["observation_variance"]), float(noise["process_variance_rate"])
    if not np.isfinite([tau2, q]).all() or min(tau2, q) < 0.:
        raise ValueError(f"Invalid nonnegative diffusion variance at {key(drift)}.")
    K = variance_kernel(float(drift["theta_F"]), float(drift["theta_R"]),
                        int(steps_per_hour), k)
    variance = tau2+q*K
    if not np.isfinite(variance).all() or np.min(variance) < 0.:
        raise ValueError(f"Invalid demand variance at {key(drift)}.")
    return np.sqrt(variance)


def aggregate(drift_file, diffusion_file, steps_per_hour, selection,
              progress_every=10000):
    """Pool the mean of sqrt(variance), not sqrt(mean variance)."""
    pool = defaultdict(lambda: {"sum":0., "sum_squares":0., "last":0.,
                                "first":0., "origins":0, "observations":0})
    matched = 0
    for drift, noise in matched_rows(drift_file, diffusion_file):
        matched += 1
        if not selected(drift, *selection): continue
        deviations = interval_std(drift, noise, steps_per_hour)
        group = (drift["baseline"], int(drift["training_weekdays"]),
                 int(drift["horizon_hours"]))
        item = pool[group]
        item["sum"] += float(deviations.sum())
        item["sum_squares"] += float(np.dot(deviations, deviations))
        item["first"] += float(deviations[0])
        item["last"] += float(deviations[-1])
        item["origins"] += 1
        item["observations"] += len(deviations)
        if matched % progress_every == 0:
            print(f"[ARM demand std] matched={matched:,}; "
                  f"selected={sum(x['origins'] for x in pool.values()):,}", flush=True)
    if not pool: raise ValueError("No matching saved diffusion configurations for selected N, k and baselines.")
    rows = []
    for (baseline, n, k), value in sorted(pool.items()):
        origins, observations = value["origins"], value["observations"]
        rows.append({
            "baseline":baseline, "training_weekdays":n, "horizon_hours":k,
            "origins":origins, "observations":observations,
            "mean_demand_std":value["sum"]/observations,
            "rms_demand_std":float(np.sqrt(value["sum_squares"]/observations)),
            "mean_initial_demand_std":value["first"]/origins,
            "mean_final_demand_std":value["last"]/origins,
        })
    return pd.DataFrame(rows)


def plot(summary, destination, ns, ks):
    """Two side-by-side heatmaps with a common color scale."""
    if any(name not in summary["baseline"].unique() for name in BASELINES):
        raise ValueError("Both weekly_average and weekly_temperature are required for a 1x2 comparison.")
    maximum = max(float(summary["mean_demand_std"].max()), 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    for ax, name in zip(axes, BASELINES):
        part = summary.loc[summary["baseline"] == name]
        matrix = part.pivot(index="training_weekdays", columns="horizon_hours",
                            values="mean_demand_std").reindex(index=ns, columns=ks)
        heatmap = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
                            origin="lower", aspect="auto", interpolation="nearest",
                            cmap="viridis", vmin=0., vmax=maximum)
        ax.set_title(f"Mean demand standard deviation | {TITLES[name]}")
        ax.set_xticks(np.arange(len(ks)), labels=ks, fontsize=8)
        ax.set_yticks(np.arange(len(ns)), labels=ns, fontsize=8)
        ax.set_xlabel("Historical fitting horizon k (hours)")
        ax.set_ylabel("N preceding matching weekdays")
    fig.colorbar(heatmap, ax=axes, shrink=0.87, pad=0.015,
                 label="Mean model-implied demand std (normalized demand)")
    fig.suptitle("ARM diffusion | mean historical demand standard deviation",
                 fontsize=13)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def run(args):
    source = Path(args.source) if args.source else SWEEP_OUTPUT/args.group.lower()
    diffusion = Path(args.diffusion_source) if args.diffusion_source else DIFFUSION_OUTPUT/args.group.lower()
    output = Path(args.output) if args.output else diffusion
    config = configuration(source)
    if config["group"] != args.group:
        raise ValueError(f"--group {args.group} differs from saved drift group {config['group']}.")
    noise_file = diffusion/"diffusion_interval_parameters.csv"
    drift_file = source/"parameter_estimates.csv"
    for path in (noise_file, drift_file):
        if not path.is_file(): raise FileNotFoundError(path)
    # Verify the discretization is identical to the diffusion fit.
    metadata = diffusion/"diffusion_configuration.json"
    if metadata.is_file():
        import json
        previous_steps = json.loads(metadata.read_text(encoding="utf-8")).get("steps_per_hour")
        if previous_steps is not None and int(previous_steps) != int(config["steps_per_hour"]):
            raise ValueError("Diffusion and drift saved configurations use different steps_per_hour.")
    names = BASELINES
    selection = bounds(config, args)
    _, ns, ks, _, _ = selection
    summary = aggregate(drift_file, noise_file, int(config["steps_per_hour"]),
                        selection, progress_every=args.progress_every)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output/CSV_NAME, index=False)
    image = plot(summary, output/PNG_NAME, ns, ks)
    print(f"[ARM demand std] configurations={int(summary['origins'].sum()):,} "
          f"| saved={image} | table={output/CSV_NAME}", flush=True)
    return summary


def main():
    parser = ArgumentParser(description="Plot ARM historically fitted total demand standard deviation in a 2x1 N,k heatmap (no refitting).")
    parser.add_argument("--group", choices=("ALL", "MED"), default="ALL")
    parser.add_argument("--source", default=None, help="Drift sweep CSV directory.")
    parser.add_argument("--diffusion-source", default=None, help="Saved diffusion fitting directory.")
    parser.add_argument("--output", default=None, help="Output directory; defaults to diffusion source.")
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--min-weekdays", type=int, default=None)
    parser.add_argument("--max-weekdays", type=int, default=None)
    parser.add_argument("--min-horizon", type=int, default=None)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--start-day", default=None)
    parser.add_argument("--end-day", default=None)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    if args.progress_every < 1: parser.error("--progress-every must be positive.")
    if set(args.baselines) != set(BASELINES) or len(args.baselines) != 2:
        parser.error("The 2x1 heatmap requires both weekly_average and weekly_temperature.")
    return run(args)


if __name__ == "__main__": main()
