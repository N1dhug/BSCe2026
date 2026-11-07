"""Chapter 4 ARM historical-fit and oracle-prediction cumulative error heatmaps.

Each tile (N,k) evaluates the independently fitted N matching historical
k-hour intervals and/or the target k-hour oracle prediction, using residuals
at hourly offsets 0..k-1. Every origin advances by one hour in the original
sweep; this executable never solves a new parameter-estimation QP.

Historical fit quality must be reconstructed: the saved "fit_rmse" includes
an endpoint observation at hour k for k<24, so it is NOT the k-hour metric.
"""
from argparse import ArgumentParser
from collections import defaultdict
from csv import DictReader
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.sweep_weekday_horizon import (
    OUTPUT as SWEEP_OUTPUT, EXPERIMENT_VERSION,
)
from src.tools.baseline import statistical_output, weekly_average, weekly_temperature
from src.tools.impute_hourly import impute_hourly_days


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/"files"/"output"/"wm_4"/"moving_horizon_quality"
METRICS = ("rmse", "mae", "bias")
BASELINES = ("weekly_average", "weekly_temperature", "statistical_output")
TITLES = {"weekly_average":"Weekly average", "weekly_temperature":"Weekly temperature",
          "statistical_output":"Statistical output"}
LABELS = {"rmse":"RMSE", "mae":"MAE", "bias":"Bias (model - observed)"}


def configuration(source):
    path = Path(source)/"configuration.json"
    if not path.is_file(): raise FileNotFoundError(f"Missing experiment configuration: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != EXPERIMENT_VERSION:
        raise ValueError("Source is not the N,k-specific hourly-origin ARM experiment. Expected version "
                         f"{EXPERIMENT_VERSION}, got {data.get('version')!r}.")
    return data


def bounds(config, args):
    names = tuple(args.baselines or config["baselines"])
    if not names or len(set(names)) != len(names) or any(name not in BASELINES for name in names):
        raise ValueError("Select one or more unique supported baselines.")
    n_values = tuple(range(args.min_weekdays or config["min_weekdays"],
                           (args.max_weekdays or config["max_weekdays"])+1))
    k_values = tuple(range(args.min_horizon or config["min_horizon"],
                           (args.max_horizon or config["max_horizon"])+1))
    if not n_values or not 2 <= n_values[0] <= n_values[-1] <= 14:
        raise ValueError("Invalid historical backlog N range.")
    if not k_values or not 1 <= k_values[0] <= k_values[-1] <= 24:
        raise ValueError("Invalid horizon k range.")
    first = pd.Timestamp(args.start_day) if args.start_day else None
    end = pd.Timestamp(args.end_day) if args.end_day else None
    until = (end+pd.Timedelta(days=1) if end == end.normalize()
             else end+pd.Timedelta(hours=1)) if end is not None else None
    if first is not None and until is not None and first >= until:
        raise ValueError("Invalid selected hourly origin interval.")
    return names, n_values, k_values, first, until


def selected(row, names, n_values, k_values, first=None, until=None):
    if row["baseline"] not in names: return False
    if int(row["training_weekdays"]) not in n_values: return False
    if int(row["horizon_hours"]) not in k_values: return False
    if first is not None or until is not None:
        origin = pd.Timestamp(row["day"])
        if first is not None and origin < first: return False
        if until is not None and origin >= until: return False
    return True


def pooled_row(key, record):
    name, n, k = key
    count = int(record["count"])
    if count < 1: raise ValueError("Cannot aggregate an empty group.")
    return {"baseline":name, "training_weekdays":n, "horizon_hours":k,
            "observations":count, "origins":int(record["origins"]),
            "rmse":float(np.sqrt(record["square"]/count)),
            "mae":float(record["absolute"]/count),
            "bias":float(record["signed"]/count)}


def accumulate(pool, key, errors, origin=None):
    errors = np.asarray(errors, dtype=float).reshape(-1)
    if not errors.size or not np.isfinite(errors).all():
        raise ValueError(f"Non-finite/empty residuals for {key}, origin={origin}.")
    item = pool[key]
    item["square"] += float(np.dot(errors, errors))
    item["absolute"] += float(np.abs(errors).sum())
    item["signed"] += float(errors.sum())
    item["count"] += int(errors.size)
    item["origins"] += 1


def empty_pool():
    return defaultdict(lambda: {"square":0., "absolute":0., "signed":0., "count":0, "origins":0})


def prediction_summary(source, selection, cohort=None):
    """Aggregate saved complete k-hour oracle errors, not a 24-hour fit prefix."""
    file = Path(source)/"oracle_horizon_scores.csv"
    if not file.is_file(): raise FileNotFoundError(file)
    names, ns, ks, first, until = selection
    pool = empty_pool()
    for chunk in pd.read_csv(file, chunksize=100000):
        for column in ("day", "baseline", "training_weekdays", "horizon_hours", "mse", "mae", "bias"):
            if column not in chunk: raise ValueError(f"Saved oracle scores missing column {column}.")
        chunk = chunk.loc[chunk["baseline"].isin(names) &
                          chunk["training_weekdays"].isin(ns) &
                          chunk["horizon_hours"].isin(ks)]
        if chunk.empty: continue
        if cohort is not None or first is not None or until is not None:
            origin = pd.to_datetime(chunk["day"], format="mixed")
            mask = pd.Series(True, index=chunk.index)
            if first is not None: mask &= origin >= first
            if until is not None: mask &= origin < until
            if cohort is not None: mask &= origin.isin(cohort)
            chunk = chunk.loc[mask]
        if chunk.empty: continue
        for key, data in chunk.groupby(["baseline", "training_weekdays", "horizon_hours"]):
            count = len(data)
            if not np.isfinite(data[["mse", "mae", "bias"]].to_numpy(dtype=float)).all():
                raise ValueError(f"Invalid oracle score for {key}.")
            value = pool[(str(key[0]), int(key[1]), int(key[2]))]
            k = int(key[2])
            value["square"] += float(data["mse"].sum())*k
            value["absolute"] += float(data["mae"].sum())*k
            value["signed"] += float(data["bias"].sum())*k
            value["count"] += count*k
            value["origins"] += count
    return pd.DataFrame([pooled_row(key, record) for key, record in pool.items()])


def common_origins(source, selection):
    """Optional strict matched set across every selected baseline, N and k."""
    file = Path(source)/"oracle_horizon_scores.csv"
    names, ns, ks, first, until = selection
    required = len(names)*len(ns)*len(ks)
    counts = defaultdict(int)
    for chunk in pd.read_csv(file, chunksize=100000):
        chunk = chunk.loc[chunk["baseline"].isin(names) &
                          chunk["training_weekdays"].isin(ns) &
                          chunk["horizon_hours"].isin(ks)]
        if chunk.empty: continue
        times = pd.to_datetime(chunk["day"], format="mixed")
        if first is not None: chunk = chunk.loc[times >= first]; times = times.loc[chunk.index]
        if until is not None: chunk = chunk.loc[times < until]; times = times.loc[chunk.index]
        for origin, size in times.value_counts().items(): counts[origin] += int(size)
    matched = pd.DatetimeIndex([origin for origin, size in counts.items() if size == required])
    if matched.empty: raise ValueError("No hourly origin has all selected baseline/N/k configurations. "
                                       "Use --cohort all or restrict the N/k range.")
    return matched.sort_values()


def saved_baseline(row):
    """Restore the EXACT baseline fitted during the original sweep, without refitting."""
    name = row["baseline"]
    coef = np.asarray(json.loads(row["baseline_coefficients"]), dtype=float)
    if name == "statistical_output":
        obj = statistical_output(min_samples=2)
        obj.coef = coef
        obj.feature_mean = np.asarray(json.loads(row["baseline_feature_mean"]), dtype=float)
        obj.feature_scale = np.asarray(json.loads(row["baseline_feature_scale"]), dtype=float)
        if obj.coef.shape != obj.feature_mean.shape or obj.coef.shape != obj.feature_scale.shape:
            raise ValueError("Inconsistent saved statistical baseline coefficients.")
    elif name in ("weekly_average", "weekly_temperature"):
        obj = weekly_average(min_samples=2) if name == "weekly_average" else weekly_temperature(min_samples=2)
        if coef.size not in (168, 169): raise ValueError("Invalid saved weekly baseline profile.")
        obj.levels = coef[:168].copy()
        obj.coef = coef.copy()
        obj.beta_C = float(row["baseline_beta_C"])
        obj.chill_center = float(row["baseline_chill_center"])
    else: raise ValueError(f"Unsupported saved baseline {name}.")
    return obj


def historical_errors(frame, row, states, steps_per_hour):
    """Replay N saved historical [F0,R0] states with saved theta, score hours 0..k-1.

    A vectorized forward-Euler solver reproduces wm_4._solve_deterministic
    at every hourly output timestamp. No parameter or state is re-estimated.
    """
    n, k = int(row["training_weekdays"]), int(row["horizon_hours"])
    if len(states) != n: raise ValueError(f"Expected N={n} historical initial states.")
    origins = pd.DatetimeIndex([pd.Timestamp(item["historical_day"]) for item in states])
    target = pd.Timestamp(row["day"])
    if origins.has_duplicates or any(origin >= target or
          origin.dayofweek != target.dayofweek or origin.hour != target.hour
          for origin in origins):
        raise ValueError(f"Historical state timestamps do not match target {target}.")
    index = pd.DatetimeIndex([origin+pd.Timedelta(hours=hour)
                               for origin in origins for hour in range(k)])
    historical = frame.reindex(index)[["I", "U", "Temp"]]
    if historical.isna().any().any():
        raise ValueError(f"Missing restored historical observations for target={target}, N={n}, k={k}.")
    baseline = saved_baseline(row)
    O = np.asarray(baseline.predict(index, historical["Temp"].to_numpy(dtype=float)), dtype=float).reshape(n, k)
    I = historical["I"].to_numpy(dtype=float).reshape(n, k)
    U = historical["U"].to_numpy(dtype=float).reshape(n, k)
    F = np.asarray([float(item["historical_F0"]) for item in states], dtype=float)
    R = np.asarray([float(item["historical_R0"]) for item in states], dtype=float)
    theta_F, theta_U, theta_R = float(row["theta_F"]), float(row["theta_U"]), float(row["theta_R"])
    if not np.isfinite(np.r_[F, R, theta_F, theta_U, theta_R]).all():
        raise ValueError("Non-finite saved state or fitted dynamical coefficients.")
    dt = 1.0/(24.0*int(steps_per_hour))
    residual = np.empty((n, k), dtype=float)
    for hour in range(k):
        residual[:, hour] = O[:, hour]+R-I[:, hour]
        if hour == k-1: break
        for _ in range(int(steps_per_hour)):
            dR = theta_F*F+theta_U*U[:, hour]+theta_R*R
            F, R = F+dt*R, R+dt*dR
    return residual.reshape(-1)


def historical_summary(source, selection, steps_per_hour, group, cohort=None, progress_every=10000):
    """Stream saved fit/state CSVs in original append order; never refit a QP.

    The sweep appends one fit row followed by N initial-state rows for that fit,
    preserving the same configuration and horizon order in both CSVs.
    """
    from src.exe.run_model.wm_4.moving_horizon_common import group_frame

    fits = Path(source)/"parameter_estimates.csv"
    states = Path(source)/"historical_initial_states.csv"
    if not fits.is_file() or not states.is_file(): raise FileNotFoundError("Missing saved fit or historical state CSV.")
    raw = group_frame(group)
    historical, _ = impute_hourly_days(raw, columns=("I", "U", "Temp"))
    names, ns, ks, first, until = selection
    pool = empty_pool()
    processed, selected_count = 0, 0
    with fits.open(newline="", encoding="utf-8-sig") as fit_file, \
         states.open(newline="", encoding="utf-8-sig") as state_file:
        fit_rows, state_rows = DictReader(fit_file), iter(DictReader(state_file))
        for row in fit_rows:
            n = int(row["training_weekdays"])
            key = tuple(row[field] for field in ("day", "baseline", "training_weekdays", "horizon_hours"))
            historical_states = []
            for _ in range(n):
                state = next(state_rows, None)
                if state is None: raise ValueError(f"Truncated historical_initial_states.csv at {key}.")
                matched = tuple(state[field] for field in ("day", "baseline", "training_weekdays", "horizon_hours"))
                if matched != key:
                    raise ValueError("Fit and historical state CSVs are not aligned at "
                                     f"{key}: found {matched}. Keep the original saved CSVs together.")
                historical_states.append(state)
            processed += 1
            if not selected(row, names, ns, ks, first, until): continue
            origin = pd.Timestamp(row["day"])
            if cohort is not None and origin not in cohort: continue
            error = historical_errors(historical, row, historical_states, steps_per_hour)
            accumulate(pool, (row["baseline"], n, int(row["horizon_hours"])), error, origin)
            selected_count += 1
            if selected_count % progress_every == 0:
                print(f"[ARM historical replay] selected configurations={selected_count:,}, scanned={processed:,}", flush=True)
        if next(state_rows, None) is not None:
            raise ValueError("Historical state CSV contains excess rows; saved inputs may be misaligned.")
    print(f"[ARM historical replay] scanned={processed:,}, selected={selected_count:,}; QP fits performed=0", flush=True)
    return pd.DataFrame([pooled_row(key, item) for key, item in pool.items()])


def plot_quality(summary, kind, names, ns, ks, output):
    """Matched axes/color scales across baselines, one row per cumulative metric."""
    if summary.empty: raise ValueError(f"No {kind} configurations available for plotting.")
    names = tuple(names)
    fig, axes = plt.subplots(3, len(names), figsize=(max(9, 7*len(names)), 12),
                             squeeze=False, constrained_layout=True)
    for row_idx, metric in enumerate(METRICS):
        for col_idx, name in enumerate(names):
            ax = axes[row_idx, col_idx]
            part = summary.loc[summary["baseline"] == name]
            matrix = part.pivot(index="training_weekdays", columns="horizon_hours",
                                values=metric).reindex(index=ns, columns=ks)
            vals = matrix.to_numpy(dtype=float)
            vmax = float(np.nanmax(np.abs(summary[metric].to_numpy(dtype=float))))
            vmax = max(vmax, 1e-12)
            cmap = "coolwarm" if metric == "bias" else "viridis"
            norm = (-vmax, vmax) if metric == "bias" else (0.0, vmax)
            image = ax.imshow(np.ma.masked_invalid(vals), origin="lower", aspect="auto",
                              interpolation="nearest", cmap=cmap, vmin=norm[0], vmax=norm[1])
            ax.set_title(f"{LABELS[metric]} | {TITLES[name]}")
            ax.set_xticks(np.arange(len(ks)), labels=ks, fontsize=8)
            ax.set_yticks(np.arange(len(ns)), labels=ns, fontsize=8)
            ax.set_xlabel("Historical fitting and evaluation horizon k (hours)")
            if col_idx == 0: ax.set_ylabel("N preceding matching weekdays")
            fig.colorbar(image, ax=ax, shrink=0.86, pad=0.015,
                         label=f"Pooled cumulative {LABELS[metric]} (normalized demand)")
    fig.suptitle(f"ARM {kind}: cumulative historical/target demand errors at each independently fitted (N,k)",
                 fontsize=13)
    destination = output/f"{kind}_cumulative_quality.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def main():
    parser = ArgumentParser(description="Plot Chapter 4 ARM N,k cumulative historical-fit and prediction error; no parameter refits.")
    parser.add_argument("--group", choices=("MED", "ALL"), default="ALL")
    parser.add_argument("--source", default=None, help="Completed sweep CSV directory; defaults to weekday_horizon_sweep_expanded/<group>.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--mode", choices=("fit", "prediction", "both"), default="both")
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=None)
    parser.add_argument("--min-weekdays", type=int, default=None)
    parser.add_argument("--max-weekdays", type=int, default=None)
    parser.add_argument("--min-horizon", type=int, default=None)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--start-day", default=None)
    parser.add_argument("--end-day", default=None)
    parser.add_argument("--cohort", choices=("all", "matched"), default="all",
                        help="all: use each cell's eligible origins; matched: require all selected (baseline,N,k) at each origin.")
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    if args.progress_every < 1: parser.error("--progress-every must be positive.")
    source = Path(args.source) if args.source else SWEEP_OUTPUT/args.group.lower()
    output = Path(args.output) if args.output else OUTPUT/args.group.lower()
    config = configuration(source)
    if config["group"] != args.group:
        parser.error(f"Selected --group {args.group} differs from saved sweep group {config['group']}.")
    selection = bounds(config, args)
    names, ns, ks, _, _ = selection
    output.mkdir(parents=True, exist_ok=True)
    cohort = common_origins(source, selection) if args.cohort == "matched" else None
    if cohort is not None: print(f"[ARM quality] fully matched hourly origins={len(cohort):,}", flush=True)
    if args.mode in ("prediction", "both"):
        prediction = prediction_summary(source, selection, cohort)
        prediction.to_csv(output/"prediction_cumulative_quality.csv", index=False)
        print(f"[ARM quality] prediction={plot_quality(prediction, 'prediction', names, ns, ks, output)}", flush=True)
    if args.mode in ("fit", "both"):
        fit = historical_summary(source, selection, int(config["steps_per_hour"]), args.group,
                                 cohort, args.progress_every)
        fit.to_csv(output/"fit_cumulative_quality.csv", index=False)
        print(f"[ARM quality] historical fit={plot_quality(fit, 'fit', names, ns, ks, output)}", flush=True)
    print(f"[ARM quality] results={output} | origin cohort={args.cohort}", flush=True)


if __name__ == "__main__": main()
