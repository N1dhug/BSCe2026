"""Retrospective BM1 price-optimization diagnostic on real 2022 data.

The experiment imports the repository's measured consumption data and DK1 spot
prices directly through src.tools.import_data. For each selected calendar day:

* A(t) is the observed normalized hourly aggregate consumption;
* B(t) is a history-only baseline built from the previous matching weekdays;
* the observed normalized DK1 spot price is evaluated as one candidate policy;
* BM1 optimizes a 24-value hourly price signal with the fast FV evaluator;
* optional Monte Carlo validation compares observed and optimized policies with
  common random numbers.

This is a retrospective tracking/inverse-control diagnostic, not a forecast or
causal price-response validation. The BM1 response functions, C, k, Xi,
sigma_x and initial Beta law remain the controlled F2 reference choices.

Run from the repository root:

    python -B -m src.exe.run_model.bm_1.real_data_daily_price_experiment

Use explicit days if desired:

    python -B -m src.exe.run_model.bm_1.real_data_daily_price_experiment --days 2022-03-15 2022-06-15 2022-09-15
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.bm_1.f2_reference_case import (
    INITIAL_ALPHA,
    INITIAL_BETA,
    SIGMA_X,
    beta_initial_cell_mass,
)
from src.exe.run_model.bm_1.stochastic_optimization_reference import (
    C,
    K,
    T,
    XI,
    evaluate_mc_candidates_common_random_numbers,
    linear_price_response,
    linear_state_response,
    paired_mc_improvement,
)
from src.method.bm_1_fast_fv import FastFVStochasticPriceEvaluator
from src.method.bm_1_stochastic_optimization import multistart_powell
from src.model.bm_1 import bm_1
from src.tools.import_data.import_input import input_data
from src.tools.import_data.import_prices import spot_price_data
from src.tools.input_generator.input_builder import input_builder


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "files" / "input" / "data"
OUTPUT_DIR = ROOT / "files" / "output" / "bm_1" / "real_data"

SEASON_START = pd.Timestamp("2022-01-01")
SEASON_END = pd.Timestamp("2022-11-01")
HOURLY_EDGES = np.linspace(T[0], T[1], 25, dtype=float)

DEFAULT_BASELINE_WEEKS = 5
DEFAULT_DAYS = 3
DEFAULT_CELLS = 201
DEFAULT_MC_PATHS = 20_000
OBJECTIVE_MAX_STEP = 1.0 / 96.0
SOLVER_MAX_STEP = 1.0 / 96.0
MC_INITIAL_SEED = 2091
MC_BROWNIAN_SEED = 2092


@dataclass(frozen=True)
class Normalization:
    consumption_min: float
    consumption_max: float
    price_min: float
    price_max: float


@dataclass(frozen=True)
class DaySignals:
    day: pd.Timestamp
    target: np.ndarray
    baseline: np.ndarray
    observed_price: np.ndarray
    raw_target: np.ndarray
    raw_price: np.ndarray
    baseline_days: tuple


@dataclass(frozen=True)
class DayExperiment:
    signals: DaySignals
    evaluator: FastFVStochasticPriceEvaluator
    optimization: object
    evaluations: dict
    mc: dict | None
    paired_mc: dict | None
    plot_path: Path


def _normalize(series, lower, upper, name):
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"{name} normalization range must be finite with upper > lower.")
    return (series.astype(float) - lower) / (upper - lower)


def _consumption_series(data):
    columns = data.measurement_columns
    if not columns:
        raise ValueError("No consumption measurement columns found in input.xlsx.")
    values = (
        data.observed_data.loc[:, columns]
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=1)
        .dropna()
        .sort_index()
    )
    values.name = "Consumption"
    return values


def _price_series(data):
    values = data.prices("DKK")
    values.index = values.index.tz_localize(None)
    values = values.groupby(level=0).mean().sort_index()
    values.name = "SpotPriceDKK"
    return values


def load_real_signals(price_area="DK1"):
    """Load and season-normalize real aggregate consumption and spot price."""
    consumption_import = input_data(DATA_DIR / "input.xlsx")
    price_import = spot_price_data(
        DATA_DIR / "prices.xlsx",
        price_area=price_area,
    )

    consumption_raw = _consumption_series(consumption_import)
    price_raw = _price_series(price_import)

    consumption_season = consumption_raw.loc[
        (consumption_raw.index >= SEASON_START)
        & (consumption_raw.index < SEASON_END)
    ]
    price_season = price_raw.loc[
        (price_raw.index >= SEASON_START)
        & (price_raw.index < SEASON_END)
    ]
    if consumption_season.empty or price_season.empty:
        raise ValueError("No Jan-Oct 2022 real data are available for normalization.")

    normalization = Normalization(
        consumption_min=float(consumption_season.min()),
        consumption_max=float(consumption_season.max()),
        price_min=float(price_season.min()),
        price_max=float(price_season.max()),
    )

    consumption = _normalize(
        consumption_raw,
        normalization.consumption_min,
        normalization.consumption_max,
        "consumption",
    )
    price = _normalize(
        price_raw,
        normalization.price_min,
        normalization.price_max,
        "price",
    )
    return consumption_raw, price_raw, consumption, price, normalization


def _day_index(day):
    day = pd.Timestamp(day).normalize()
    return pd.date_range(
        day,
        day + pd.Timedelta(days=1),
        freq="h",
        inclusive="left",
    )


def _complete_day_values(series, day, name):
    index = _day_index(day)
    values = series.reindex(index)
    if values.isna().any():
        missing = list(values.index[values.isna()])
        raise ValueError(
            f"{pd.Timestamp(day):%Y-%m-%d} has incomplete {name}; "
            f"missing={missing[:5]}."
        )
    array = values.to_numpy(dtype=float)
    if array.shape != (24,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain 24 finite hourly values.")
    return array


def build_day_signals(
    consumption_raw,
    price_raw,
    consumption,
    price,
    day,
    *,
    baseline_weeks=DEFAULT_BASELINE_WEEKS,
):
    """Build a real target day and history-only matching-weekday baseline."""
    day = pd.Timestamp(day).normalize()
    baseline_weeks = int(baseline_weeks)
    if baseline_weeks < 3:
        raise ValueError("baseline_weeks must be at least 3 for trimmed averaging.")

    target = _complete_day_values(consumption, day, "normalized consumption")
    observed_price = _complete_day_values(price, day, "normalized DK1 price")
    raw_target = _complete_day_values(consumption_raw, day, "raw consumption")
    raw_price = _complete_day_values(price_raw, day, "raw DK1 price")

    if np.min(target) < -1.0e-12 or np.max(target) > 1.0 + 1.0e-12:
        raise ValueError("Selected target day lies outside the Jan-Oct normalization range.")
    if np.min(observed_price) < -1.0e-12 or np.max(observed_price) > 1.0 + 1.0e-12:
        raise ValueError("Selected price day lies outside the Jan-Oct normalization range.")

    history_days = tuple(
        day - pd.Timedelta(weeks=week)
        for week in range(1, baseline_weeks + 1)
    )
    history = np.vstack(
        [
            _complete_day_values(
                consumption,
                history_day,
                "historical normalized consumption",
            )
            for history_day in history_days
        ]
    )
    ordered = np.sort(history, axis=0)
    baseline = np.mean(ordered[1:-1], axis=0)

    if (
        not np.all(np.isfinite(baseline))
        or np.min(baseline) < -1.0e-12
        or np.max(baseline) > 1.0 + 1.0e-12
    ):
        raise ValueError("History-derived baseline is invalid after normalization.")

    return DaySignals(
        day=day,
        target=target.copy(),
        baseline=baseline.copy(),
        observed_price=observed_price.copy(),
        raw_target=raw_target.copy(),
        raw_price=raw_price.copy(),
        baseline_days=history_days,
    )


def _representative_quantiles(n_days):
    n_days = int(n_days)
    if n_days < 1:
        raise ValueError("n_days must be positive.")
    if n_days == 1:
        return np.array([0.5], dtype=float)
    if n_days == 2:
        return np.array([0.25, 0.75], dtype=float)
    return np.linspace(0.15, 0.85, n_days)


def select_representative_days(
    consumption_raw,
    price_raw,
    consumption,
    price,
    *,
    n_days=DEFAULT_DAYS,
    baseline_weeks=DEFAULT_BASELINE_WEEKS,
):
    """Select complete low/typical/high-consumption days reproducibly."""
    earliest = SEASON_START + pd.Timedelta(weeks=int(baseline_weeks))
    latest = SEASON_END - pd.Timedelta(days=1)

    candidates = []
    for day in pd.date_range(earliest, latest, freq="D"):
        try:
            signals = build_day_signals(
                consumption_raw,
                price_raw,
                consumption,
                price,
                day,
                baseline_weeks=baseline_weeks,
            )
        except ValueError:
            continue
        candidates.append(
            {
                "day": day,
                "mean_consumption": float(np.mean(signals.target)),
            }
        )

    if len(candidates) < int(n_days):
        raise ValueError(
            f"Only {len(candidates)} complete real-data days satisfy the "
            f"history requirements; requested {n_days}."
        )

    candidates = sorted(candidates, key=lambda row: row["mean_consumption"])
    means = np.array([row["mean_consumption"] for row in candidates], dtype=float)
    quantiles = _representative_quantiles(n_days)

    selected = []
    used = set()
    for quantile in quantiles:
        target_mean = float(np.quantile(means, quantile))
        order = np.argsort(np.abs(means - target_mean))
        for index in order:
            day = candidates[int(index)]["day"]
            if day not in used:
                selected.append(day)
                used.add(day)
                break
    return selected


def _hourly_hold(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape != (24,):
        raise ValueError("Hourly held signals must contain 24 values.")
    return input_builder.price_intervals((HOURLY_EDGES, values))


def _parameter_template():
    neutral = input_builder.price_intervals(
        (HOURLY_EDGES, np.full(24, 0.5, dtype=float))
    )
    return [
        neutral,
        linear_state_response,
        linear_price_response,
        C,
        K,
        XI,
    ]


def build_evaluator(signals, *, cells=DEFAULT_CELLS):
    baseline = _hourly_hold(signals.baseline)
    target = _hourly_hold(signals.target)
    model = bm_1(baseline, target, T, 2)

    spatial_edges, _, _ = model.fokker_planc_reference.grid(int(cells))
    initial_mass = beta_initial_cell_mass(
        spatial_edges,
        alpha=INITIAL_ALPHA,
        beta=INITIAL_BETA,
    )
    return FastFVStochasticPriceEvaluator(
        model,
        _parameter_template(),
        HOURLY_EDGES,
        SIGMA_X,
        initial_mass,
        objective_max_step=OBJECTIVE_MAX_STEP,
        solver_max_step=SOLVER_MAX_STEP,
        breakpoints=HOURLY_EDGES[1:-1],
    )


def _objective_dict(evaluation):
    objective = evaluation.objective
    return {
        "total": float(objective.total),
        "bias": float(objective.bias),
        "variance": float(objective.variance),
    }


def _price_correlation(observed, optimized):
    observed = np.asarray(observed, dtype=float)
    optimized = np.asarray(optimized, dtype=float)
    if np.std(observed) <= 1.0e-15 or np.std(optimized) <= 1.0e-15:
        return float("nan")
    return float(np.corrcoef(observed, optimized)[0, 1])


def save_day_plot(signals, evaluator, evaluations, optimized_prices):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    ax_demand, ax_price, ax_objective = axes

    hours = 24.0 * evaluator.midpoints
    plot_t = np.linspace(0.0, 1.0, 1441)
    plot_hours = 24.0 * plot_t
    target = _hourly_hold(signals.target)
    baseline = _hourly_hold(signals.baseline)

    ax_demand.step(
        plot_hours,
        target(plot_t),
        where="post",
        linewidth=2.2,
        label="Observed target A(t)",
    )
    ax_demand.step(
        plot_hours,
        baseline(plot_t),
        where="post",
        linestyle="--",
        label="History baseline B(t)",
    )
    ax_demand.plot(
        hours,
        evaluations["observed_spot"].objective.expected_demand,
        label="E[D] with observed DK1 price",
    )
    ax_demand.plot(
        hours,
        evaluations["optimized"].objective.expected_demand,
        label="E[D] with optimized price",
    )
    ax_demand.set_ylabel("Normalized consumption")
    ax_demand.set_title(f"Real-data BM1 tracking — {signals.day:%Y-%m-%d}")
    ax_demand.grid(True, alpha=0.25)
    ax_demand.legend(loc="best")

    edge_hours = np.arange(25, dtype=float)
    observed_plot = np.append(signals.observed_price, signals.observed_price[-1])
    optimized_plot = np.append(optimized_prices, optimized_prices[-1])
    ax_price.step(
        edge_hours,
        observed_plot,
        where="post",
        label="Observed normalized DK1 spot price",
    )
    ax_price.step(
        edge_hours,
        optimized_plot,
        where="post",
        label="BM1 optimized hourly price",
    )
    ax_price.set_xlim(0.0, 24.0)
    ax_price.set_ylim(0.0, 1.0)
    ax_price.set_ylabel("Normalized price")
    ax_price.set_xlabel("Hour")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="best")

    names = ("Observed DK1", "Neutral 0.5", "Optimized")
    keys = ("observed_spot", "neutral", "optimized")
    bias = np.array([evaluations[key].objective.bias for key in keys])
    variance = np.array([evaluations[key].objective.variance for key in keys])
    positions = np.arange(len(keys))
    ax_objective.bar(positions, bias, label="Bias contribution")
    ax_objective.bar(positions, variance, bottom=bias, label="Variance contribution")
    ax_objective.set_xticks(positions, names)
    ax_objective.set_ylabel("Stochastic tracking objective")
    ax_objective.set_title("Objective decomposition")
    ax_objective.grid(True, axis="y", alpha=0.25)
    ax_objective.legend(loc="best")

    fig.tight_layout()
    path = OUTPUT_DIR / f"bm_1_real_data_{signals.day:%Y-%m-%d}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def run_day(
    signals,
    *,
    cells=DEFAULT_CELLS,
    mc_paths=DEFAULT_MC_PATHS,
    validate_mc=True,
    progress=print,
):
    evaluator = build_evaluator(signals, cells=cells)
    neutral = np.full(24, 0.5, dtype=float)
    starts = np.vstack((signals.observed_price, neutral))

    if progress is not None:
        progress(
            f"{signals.day:%Y-%m-%d}: optimizing 24 hourly prices from "
            "observed DK1 and neutral starts"
        )

    optimization = multistart_powell(
        evaluator,
        starts,
        progress=progress,
        options={
            "xtol": 5.0e-5,
            "ftol": 1.0e-9,
            "maxiter": 100,
        },
    )
    optimized = np.asarray(optimization.best.x, dtype=float)

    evaluations = {
        "observed_spot": evaluator.evaluate(signals.observed_price),
        "neutral": evaluator.evaluate(neutral),
        "optimized": evaluator.evaluate(optimized),
    }

    mc = None
    paired = None
    if validate_mc:
        mc = evaluate_mc_candidates_common_random_numbers(
            evaluator,
            {
                "observed_spot": signals.observed_price,
                "neutral": neutral,
                "optimized": optimized,
            },
            n_paths=int(mc_paths),
            initial_seed=MC_INITIAL_SEED,
            brownian_seed=MC_BROWNIAN_SEED,
        )
        paired = {}
        for reference_name in ("observed_spot", "neutral"):
            mean, standard_error = paired_mc_improvement(
                mc[reference_name],
                mc["optimized"],
            )
            paired[reference_name] = {
                "improvement": float(mean),
                "standard_error": float(standard_error),
                "ratio": (
                    float(mean / standard_error)
                    if standard_error > 0.0
                    else float("inf")
                ),
            }

    path = save_day_plot(signals, evaluator, evaluations, optimized)
    return DayExperiment(
        signals=signals,
        evaluator=evaluator,
        optimization=optimization,
        evaluations=evaluations,
        mc=mc,
        paired_mc=paired,
        plot_path=path,
    )


def _json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _result_record(result):
    signals = result.signals
    optimized = np.asarray(result.optimization.best.x, dtype=float)
    record = {
        "day": signals.day,
        "baseline_days": list(signals.baseline_days),
        "raw_consumption_mean": float(np.mean(signals.raw_target)),
        "raw_price_mean_dkk_per_mwh": float(np.mean(signals.raw_price)),
        "raw_price_min_dkk_per_mwh": float(np.min(signals.raw_price)),
        "raw_price_max_dkk_per_mwh": float(np.max(signals.raw_price)),
        "normalized_target": signals.target,
        "normalized_baseline": signals.baseline,
        "normalized_observed_price": signals.observed_price,
        "optimized_price": optimized,
        "price_correlation": _price_correlation(
            signals.observed_price,
            optimized,
        ),
        "objective": {
            name: _objective_dict(evaluation)
            for name, evaluation in result.evaluations.items()
        },
        "optimization": {
            "objective_evaluations": int(result.optimization.objective_evaluations),
            "cache_hits": int(result.optimization.cache_hits),
            "elapsed_seconds": float(result.optimization.elapsed_seconds),
        },
        "plot": result.plot_path,
    }
    if result.mc is not None:
        record["mc"] = {
            name: {
                "total": float(item.objective.total),
                "standard_error": float(item.objective.standard_error),
            }
            for name, item in result.mc.items()
        }
        record["paired_mc"] = result.paired_mc
    return record


def run_real_data_experiment(
    *,
    days=None,
    n_days=DEFAULT_DAYS,
    baseline_weeks=DEFAULT_BASELINE_WEEKS,
    price_area="DK1",
    cells=DEFAULT_CELLS,
    mc_paths=DEFAULT_MC_PATHS,
    validate_mc=True,
    progress=print,
):
    (
        consumption_raw,
        price_raw,
        consumption,
        price,
        normalization,
    ) = load_real_signals(price_area=price_area)

    if days:
        selected = [pd.Timestamp(day).normalize() for day in days]
    else:
        selected = select_representative_days(
            consumption_raw,
            price_raw,
            consumption,
            price,
            n_days=n_days,
            baseline_weeks=baseline_weeks,
        )

    experiments = []
    for day in selected:
        signals = build_day_signals(
            consumption_raw,
            price_raw,
            consumption,
            price,
            day,
            baseline_weeks=baseline_weeks,
        )
        experiments.append(
            run_day(
                signals,
                cells=cells,
                mc_paths=mc_paths,
                validate_mc=validate_mc,
                progress=progress,
            )
        )
    return experiments, normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        nargs="+",
        default=None,
        help="Explicit YYYY-MM-DD days. Omit to auto-select representative days.",
    )
    parser.add_argument(
        "--n-days",
        type=int,
        default=DEFAULT_DAYS,
        help="Number of automatically selected low/typical/high-consumption days.",
    )
    parser.add_argument(
        "--baseline-weeks",
        type=int,
        default=DEFAULT_BASELINE_WEEKS,
        help="Previous matching weekdays used for the trimmed baseline.",
    )
    parser.add_argument("--price-area", default="DK1")
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS)
    parser.add_argument("--mc-paths", type=int, default=DEFAULT_MC_PATHS)
    parser.add_argument("--no-mc", action="store_true")
    args = parser.parse_args()

    print("BM1 real-data retrospective diagnostic", flush=True)
    print("=====================================", flush=True)
    print(
        "Target is observed consumption; baseline uses only previous matching "
        "weekdays. Reference BM parameters are not fitted to these days.",
        flush=True,
    )

    experiments, normalization = run_real_data_experiment(
        days=args.days,
        n_days=args.n_days,
        baseline_weeks=args.baseline_weeks,
        price_area=args.price_area,
        cells=args.cells,
        mc_paths=args.mc_paths,
        validate_mc=not args.no_mc,
        progress=lambda message: print(message, flush=True),
    )

    print()
    print(
        "Season normalization: "
        f"consumption=[{normalization.consumption_min:.6g}, "
        f"{normalization.consumption_max:.6g}], "
        f"DK1 price=[{normalization.price_min:.6g}, "
        f"{normalization.price_max:.6g}] DKK/MWh"
    )

    records = []
    for result in experiments:
        signals = result.signals
        optimized = np.asarray(result.optimization.best.x, dtype=float)
        observed = result.evaluations["observed_spot"].objective
        neutral = result.evaluations["neutral"].objective
        optimum = result.evaluations["optimized"].objective

        print()
        print(f"{signals.day:%Y-%m-%d}")
        print("-" * 32)
        print(
            "baseline days: "
            + ", ".join(day.strftime("%Y-%m-%d") for day in signals.baseline_days)
        )
        print(
            f"raw mean consumption={np.mean(signals.raw_target):.6g}; "
            f"DK1 price mean/min/max="
            f"{np.mean(signals.raw_price):.3f}/"
            f"{np.min(signals.raw_price):.3f}/"
            f"{np.max(signals.raw_price):.3f} DKK/MWh"
        )
        print(
            f"FV J observed={observed.total:.10e}, "
            f"neutral={neutral.total:.10e}, optimized={optimum.total:.10e}"
        )
        print(
            f"optimized bias={optimum.bias:.10e}, "
            f"variance={optimum.variance:.10e}"
        )
        print(
            f"FV improvement vs observed DK1="
            f"{(observed.total - optimum.total) / observed.total:.3%}; "
            f"vs neutral="
            f"{(neutral.total - optimum.total) / neutral.total:.3%}"
        )
        print(
            "observed normalized DK1="
            + np.array2string(
                signals.observed_price,
                precision=4,
                floatmode="fixed",
                max_line_width=120,
            )
        )
        print(
            "optimized hourly price="
            + np.array2string(
                optimized,
                precision=4,
                floatmode="fixed",
                max_line_width=120,
            )
        )
        print(
            f"observed/optimized price correlation="
            f"{_price_correlation(signals.observed_price, optimized):.4f}"
        )
        print(
            f"Powell evaluations={result.optimization.objective_evaluations}, "
            f"time={result.optimization.elapsed_seconds:.3f}s"
        )

        if result.mc is not None:
            print(
                f"MC J observed={result.mc['observed_spot'].objective.total:.10e} "
                f"(SE={result.mc['observed_spot'].objective.standard_error:.3e}), "
                f"optimized={result.mc['optimized'].objective.total:.10e} "
                f"(SE={result.mc['optimized'].objective.standard_error:.3e})"
            )
            paired = result.paired_mc["observed_spot"]
            print(
                f"paired MC observed -> optimized improvement="
                f"{paired['improvement']:.6e}, "
                f"SE={paired['standard_error']:.3e}, "
                f"ratio={paired['ratio']:.3f}"
            )
        print(f"plot={result.plot_path}")
        records.append(_result_record(result))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "bm_1_real_data_daily_results.json"
    payload = {
        "note": (
            "Retrospective tracking diagnostic; target day is observed and "
            "reference BM parameters are not fitted to real data."
        ),
        "normalization": normalization.__dict__,
        "baseline_weeks": int(args.baseline_weeks),
        "cells": int(args.cells),
        "mc_paths": 0 if args.no_mc else int(args.mc_paths),
        "days": records,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Saved summary: {output_path}")


if __name__ == "__main__":
    main()
