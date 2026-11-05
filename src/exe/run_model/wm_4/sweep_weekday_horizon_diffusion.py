"""WM4 constant observation + Brownian diffusion sweep on saved weekday-horizon forecasts.

Consumes the row-level forecasts and fitted drift coefficients from the weekday
sweep; never refits theta. A row is one forecast at a specific lead, with an
observed demand and deterministic WM4 mean. By default all available tags,
dates and weekdays are used (chronological calibration/validation split).
"""
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import kstest


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "uncertainty" / "weekday_horizon_diffusion"
CANDIDATES = ("sweep_weekday_horizon*forecast*.csv", "sweep_weekday_horizon*.csv",
              "weekday_horizon*forecast*.csv", "weekday_horizon*.csv")
DEFAULT_BACKLOG = tuple(range(2, 15))
DEFAULT_HORIZONS = tuple(range(1, 25))


def _column(frame, names, required=True):
    lookup = {str(name).lower(): name for name in frame.columns}
    for name in names:
        if name.lower() in lookup: return lookup[name.lower()]
    if required: raise ValueError(f"Missing column; expected one of {names}. Available: {list(frame.columns)}")
    return None


def _discover(path=None):
    if path is not None:
        result = Path(path)
        if not result.is_file(): raise FileNotFoundError(result)
        return result
    root = ROOT / "files" / "output" / "wm_4"
    for pattern in CANDIDATES:
        matches = sorted(root.rglob(pattern)) if root.exists() else []
        if matches: return matches[-1]
    raise FileNotFoundError(
        "No saved sweep_weekday_horizon row-level forecasts found. "
        "Run the sweep first, or pass --input PATH to its forecast CSV. "
        "The CSV must contain N, lead k, observed/predicted demand and fitted r_F/r_R."
    )


def _standardize(raw):
    names = {
        "N": ("N", "backlog", "backlog_N", "preceding_weekdays", "baseline_weeks", "parameter_weeks"),
        "lead": ("lead_hours", "lead", "k", "horizon_k", "prediction_horizon", "horizon"),
        "observed": ("I_observed", "D_observed", "observed", "actual", "demand_observed"),
        "predicted": ("I_mpc", "I_wm4", "I_predicted", "D_predicted", "predicted", "wm4_prediction"),
        "r_F": ("r_F", "theta_F", "theta_f"),
        "r_R": ("r_R", "theta_R", "theta_r"),
    }
    rename = {_column(raw, choices): key for key, choices in names.items()}
    optional = {
        "variant": ("variant", "baseline", "baseline_kind", "method", "model", "strategy"),
        "tag": ("tag", "consumer_tag", "dataset"),
        "origin": ("origin", "forecast_origin", "origin_time", "prediction_date", "day", "date"),
        "r_U": ("r_U", "theta_U", "theta_u"),
    }
    for key, choices in optional.items():
        col = _column(raw, choices, required=False)
        if col is not None: rename[col] = key
    data = raw.rename(columns=rename).copy()
    if "variant" not in data: data["variant"] = "All forecasts"
    if "tag" not in data: data["tag"] = "unspecified"
    if "origin" not in data: data["origin"] = np.arange(len(data)).astype(str)
    for key in ("N", "lead", "observed", "predicted", "r_F", "r_R"):
        data[key] = pd.to_numeric(data[key], errors="coerce")
    if data[list(names)].isna().any().any(): raise ValueError("Missing/non-numeric values in required saved sweep columns.")
    if not np.isfinite(data[["N", "lead", "observed", "predicted", "r_F", "r_R"]].to_numpy()).all():
        raise ValueError("Sweep predictions and fitted theta values must be finite.")
    if ((data["N"] < 2) | (data["lead"] < 1)).any(): raise ValueError("N must be >=2 and lead k must be >=1.")
    if not np.array_equal(data["N"].to_numpy(), data["N"].to_numpy(dtype=int)):
        raise ValueError("N must be integer.")
    if not np.array_equal(data["lead"].to_numpy(), data["lead"].to_numpy(dtype=int)):
        raise ValueError("Lead k must be integer.")
    data["N"], data["lead"] = data["N"].astype(int), data["lead"].astype(int)
    data["variant"] = data["variant"].fillna("All forecasts").astype(str)
    data["tag"] = data["tag"].fillna("unspecified").astype(str)
    data["error"] = data["observed"] - data["predicted"]
    return data


@lru_cache(maxsize=32768)
def _unit_process_variance(r_f, r_r, lead, steps_per_hour):
    """Euler covariance of R for unit q with model time measured in days."""
    lead, steps_per_hour = int(lead), int(steps_per_hour)
    dt = 1.0 / (24.0 * steps_per_hour)
    transition = np.array([[1.0, dt], [dt * r_f, 1.0 + dt * r_r]], dtype=float)
    response = np.array([0.0, 1.0], dtype=float)
    variance = 0.0
    for _ in range(lead * steps_per_hour):
        variance += dt * response[1] ** 2
        response = transition @ response
    return float(variance)


def _prepare(frame, steps_per_hour):
    data = _standardize(frame)
    data["unit_process_variance"] = [
        _unit_process_variance(float(rf), float(rr), int(lead), int(steps_per_hour))
        for rf, rr, lead in zip(data["r_F"], data["r_R"], data["lead"])
    ]
    if not np.isfinite(data["unit_process_variance"]).all():
        raise ValueError("Nonfinite covariance kernel; check theta stability and time-step scaling.")
    return data


def _split(frame, fraction):
    """Split chronologically by forecast origin, never by correlated forecast rows."""
    if not 0.0 < fraction < 1.0: raise ValueError("calibration_fraction must lie strictly between 0 and 1.")
    origin = frame["origin"].astype(str)
    distinct = sorted(origin.unique())
    if len(distinct) < 3:
        raise ValueError("At least three distinct forecast origins are required for calibration/validation.")
    cut = max(2, min(len(distinct) - 1, int(np.floor(fraction * len(distinct)))))
    train = frame[origin.isin(set(distinct[:cut]))]
    test = frame[origin.isin(set(distinct[cut:]))]
    return train, test


def _distribution_ks(error, variance):
    """Descriptive KS distance between standardized residuals and standard normal."""
    error = np.asarray(error, dtype=float)
    variance = np.asarray(variance, dtype=float)
    valid = np.isfinite(error) & np.isfinite(variance) & (variance > 1e-14)
    if np.count_nonzero(valid) < 3: return np.nan
    return float(kstest(error[valid] / np.sqrt(variance[valid]), "norm").statistic)


def _calibrate(train):
    X = np.column_stack((np.ones(len(train)), train["unit_process_variance"].to_numpy(dtype=float)))
    y = train["error"].to_numpy(dtype=float) ** 2
    if len(train) < 3 or np.linalg.matrix_rank(X, tol=1e-10) < 2:
        return None
    # Nonnegative two-parameter LS is a convex QP in observation q and Wiener q.
    observation_q, process_q = nnls(X, y)[0]
    return float(observation_q), float(process_q)


def _metrics(train, test, N, horizon, variant, tag):
    fitted = _calibrate(train)
    if fitted is None: return None
    observation_q, process_q = fitted
    residual = test["error"].to_numpy(dtype=float)
    unit = test["unit_process_variance"].to_numpy(dtype=float)
    predicted_variance = observation_q + process_q * unit
    actual_rmse = float(np.sqrt(np.mean(residual ** 2)))
    predicted_rmse = float(np.sqrt(np.mean(predicted_variance)))
    covered = float(np.mean(np.abs(residual) <= 1.959963984540054 * np.sqrt(predicted_variance)))
    return {
        "variant": variant, "tag": tag, "N": int(N), "k": int(horizon),
        "train_rows": len(train), "test_rows": len(test),
        "observation_q": observation_q, "observation_sigma": float(np.sqrt(observation_q)),
        "process_q": process_q, "process_sigma": float(np.sqrt(process_q)),
        "observed_rmse": actual_rmse, "reported_rmse": predicted_rmse,
        "rmse_absolute_error": abs(predicted_rmse - actual_rmse),
        "rmse_ratio": predicted_rmse / actual_rmse if actual_rmse > 0 else np.nan,
        "ks_distance": _distribution_ks(residual, predicted_variance),
        "distribution_similarity": 1.0 - _distribution_ks(residual, predicted_variance),
        "coverage_95": covered, "bias": float(np.mean(residual)),
    }


def sweep(data, backlogs=DEFAULT_BACKLOG, horizons=DEFAULT_HORIZONS, calibration_fraction=0.7, min_samples=6):
    results = []
    for (variant, tag, N), part in data.groupby(["variant", "tag", "N"], sort=True):
        if int(N) not in backlogs: continue
        for k in horizons:
            # k is the cumulative scored prediction horizon; fit/evaluate leads <= k.
            subset = part[part["lead"] <= int(k)]
            if subset.empty: continue
            try: train, test = _split(subset, calibration_fraction)
            except ValueError: continue
            if len(train) < min_samples or len(test) < min_samples: continue
            result = _metrics(train, test, N, k, variant, tag)
            if result is not None: results.append(result)
    return pd.DataFrame(results)


def _matrix(result, variant, key, backlogs, horizons):
    grid = np.full((len(backlogs), len(horizons)), np.nan)
    part = result[result["variant"] == variant]
    for _, row in part.iterrows():
        n, k = int(row["N"]), int(row["k"])
        if n in backlogs and k in horizons: grid[backlogs.index(n), horizons.index(k)] = float(row[key])
    return grid


def save_plot(summary, path, backlogs, horizons):
    variants = list(summary["variant"].drop_duplicates())
    if len(variants) == 1: variants = variants * 2
    elif len(variants) != 2:
        raise ValueError(f"2x2 plot needs one or two variants, found {variants}. Filter with --variants.")
    fig, axes = plt.subplots(2, 2, figsize=(19, 10), sharex=True, sharey=True, constrained_layout=True)
    specifications = (("reported_rmse", "Reported stochastic RMSE (normalized demand)", "viridis"),
                      ("distribution_similarity", "Gaussian distribution similarity (1 − KS)", "viridis"))
    for row, (metric, label, cmap_name) in enumerate(specifications):
        grids = [_matrix(summary, name, metric, backlogs, horizons) for name in variants]
        valid = np.concatenate([g[np.isfinite(g)] for g in grids])
        vmin, vmax = (float(valid.min()), float(valid.max())) if valid.size else (0.0, 1.0)
        if vmin == vmax: vmax += 1e-10
        for col, name in enumerate(variants):
            ax = axes[row, col]
            im = ax.imshow(np.ma.masked_invalid(grids[col]), origin="lower", aspect="auto",
                           interpolation="nearest", cmap=cmap_name, vmin=vmin, vmax=vmax)
            ax.set_title(f"{label.split(' (')[0]} | {name}")
            ax.set_xticks(np.arange(len(horizons)), labels=horizons, fontsize=8)
            ax.set_yticks(np.arange(len(backlogs)), labels=backlogs)
            if row == 1: ax.set_xlabel("Prediction horizon k (hours scored)")
            if col == 0: ax.set_ylabel("Backlog N (preceding matching weekdays)")
        fig.colorbar(im, ax=axes[row, :], fraction=0.025, pad=0.015, label=label)
    fig.suptitle("WM4 diffusion calibrated from historical demand errors | held-out forecasts")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Calibrate WM4 observation and Brownian variances for existing weekday-horizon sweep forecasts. Fitted theta values are reused without refitting.")
    parser.add_argument("--input", default=None, help="Row-level sweep CSV (auto-discovered by default).")
    parser.add_argument("--tags", nargs="+", default=["all"], help="Consumer datasets to include; default all.")
    parser.add_argument("--variants", nargs="+", default=None, help="Select two baseline variants for the 2x2 heatmap.")
    parser.add_argument("--backlog-min", type=int, default=2)
    parser.add_argument("--backlog-max", type=int, default=14)
    parser.add_argument("--horizon-min", type=int, default=1)
    parser.add_argument("--horizon-max", type=int, default=24)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--calibration-fraction", type=float, default=0.7)
    parser.add_argument("--min-samples", type=int, default=6)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if args.steps_per_hour < 2: parser.error("--steps-per-hour must be >= 2.")
    if args.backlog_min < 2 or args.backlog_max < args.backlog_min: parser.error("Invalid backlog range.")
    if args.horizon_min < 1 or args.horizon_max < args.horizon_min: parser.error("Invalid horizon range.")
    path = _discover(args.input)
    data = _prepare(pd.read_csv(path), args.steps_per_hour)
    if [x.lower() for x in args.tags] != ["all"]: data = data[data["tag"].isin(args.tags)]
    if args.variants is not None: data = data[data["variant"].isin(args.variants)]
    if data.empty: raise ValueError("No sweep rows remain after tag/variant filtering.")
    backlogs = tuple(range(args.backlog_min, args.backlog_max + 1))
    horizons = tuple(range(args.horizon_min, args.horizon_max + 1))
    # Pool all selected consumer datasets as the default aggregate analysis;
    # keep dataset-specific results as additional diagnostics when available.
    aggregate = data.copy()
    aggregate["tag"] = "all"
    summary_all = sweep(aggregate, backlogs, horizons, args.calibration_fraction, args.min_samples)
    summary_tags = sweep(data, backlogs, horizons, args.calibration_fraction, args.min_samples)
    summary = pd.concat([summary_all, summary_tags], ignore_index=True)
    if summary.empty: raise RuntimeError("No identifiable variance fits were possible; check N/k coverage, origins and sample counts.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Produce one figure per dataset; include all datasets by default.
    for tag, part in summary.groupby("tag", sort=True):
        suffix = str(tag).replace("/", "_").replace(" ", "_")
        part.to_csv(output / f"diffusion_sweep_{suffix}.csv", index=False)
        save_plot(part, output / f"diffusion_sweep_{suffix}.png", backlogs, horizons)
        print(f"tag={tag} | cells={len(part)} | output={output / f'diffusion_sweep_{suffix}.png'}")
    summary.to_csv(output / "diffusion_sweep_all.csv", index=False)
    print(f"source={path} | fitted theta reused | result={output / 'diffusion_sweep_all.csv'}")


if __name__ == "__main__": main()
