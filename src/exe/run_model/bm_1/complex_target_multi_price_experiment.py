"""BM1 multi-price experiment with a genuinely time-varying 24-hour target.

A deterministic target demand is generated from a deliberately structured
24-hour reference-price schedule with many price changes.  The stochastic
optimization problem then tries to track that same target with 2, 4, 6, 12
and 24 equally sized price intervals.

All candidate problems use the same hourly breakpoint set, the same pure-FV
objective, the same Beta(20,20) initial law and the same process noise.
Final candidates are optionally checked with common-random-number Monte Carlo.

Run from the repository root:

    python -B -m src.exe.run_model.bm_1.complex_target_multi_price_experiment
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from src.exe.run_model.bm_1.f2_reference_case import (
    INITIAL_ALPHA,
    INITIAL_BETA,
    SIGMA_X,
    beta_initial_cell_mass,
    sample_beta_initial_states,
)
from src.exe.run_model.bm_1.stochastic_optimization_reference import (
    BASELINE,
    C,
    K,
    T,
    XI,
    X0_DETERMINISTIC,
    linear_price_response,
    linear_state_response,
)
from src.method.bm_1_fast_fv import FastFVStochasticPriceEvaluator
from src.method.bm_1_stochastic_objective import monte_carlo_tracking_objective
from src.method.bm_1_stochastic_optimization import (
    FVStochasticPriceEvaluator,
    multistart_powell,
)
from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder


INTERVAL_COUNTS = (2, 4, 6, 12, 24)
HOURLY_EDGES = np.linspace(T[0], T[1], 25, dtype=float)

# Deliberately nontrivial intraday signal: two broad waves plus several
# reversals.  It is not intended as an empirical electricity tariff; it is a
# controlled stress-test for the price parametrization.
REFERENCE_PRICES_24 = np.array(
    [
        0.78, 0.74, 0.68, 0.58, 0.46, 0.34,
        0.24, 0.30, 0.44, 0.62, 0.78, 0.84,
        0.74, 0.60, 0.46, 0.32, 0.24, 0.30,
        0.46, 0.66, 0.82, 0.72, 0.56, 0.42,
    ],
    dtype=float,
)

FV_X_STEPS = 201
OBJECTIVE_MAX_STEP = 1.0 / 96.0
SOLVER_MAX_STEP = 1.0 / 96.0
TARGET_T_STEPS = 1537

MC_PATHS = 20_000
MC_INITIAL_SEED = 2051
MC_BROWNIAN_SEED = 2052


@dataclass(frozen=True)
class ComplexTargetCase:
    n_intervals: int
    edges: np.ndarray
    evaluator: FVStochasticPriceEvaluator
    optimization: object
    evaluation: object


def _edges(n_intervals):
    n_intervals = int(n_intervals)
    if n_intervals < 1 or 24 % n_intervals != 0:
        raise ValueError("n_intervals must be a positive divisor of 24.")
    return np.linspace(T[0], T[1], n_intervals + 1, dtype=float)


def _baseline():
    return input_builder.energy(
        (np.array(T, dtype=float), np.array([BASELINE, BASELINE], dtype=float))
    )


def _parameter_template(edges):
    placeholder = input_builder.price_intervals(
        (edges, np.full(edges.size - 1, 0.5, dtype=float))
    )
    return [
        placeholder,
        linear_state_response,
        linear_price_response,
        C,
        K,
        XI,
    ]


def _parameters_for_prices(edges, prices):
    parameters = _parameter_template(edges)
    parameters[0] = input_builder.price_intervals(
        (edges, np.asarray(prices, dtype=float))
    )
    return parameters


def _coarsen_reference_prices(n_intervals):
    """Average the 24 hourly reference prices inside each candidate interval."""
    n_intervals = int(n_intervals)
    if n_intervals < 1 or 24 % n_intervals != 0:
        raise ValueError("n_intervals must divide 24.")
    block = 24 // n_intervals
    return REFERENCE_PRICES_24.reshape(n_intervals, block).mean(axis=1)


def _sample_piecewise_prices(old_edges, old_prices, new_edges):
    """Sample an existing held-price signal at the new interval midpoints."""
    old_edges = np.asarray(old_edges, dtype=float)
    old_prices = np.asarray(old_prices, dtype=float)
    new_edges = np.asarray(new_edges, dtype=float)
    midpoints = 0.5 * (new_edges[:-1] + new_edges[1:])

    if old_prices.size + 1 != old_edges.size:
        raise ValueError("old_edges and old_prices have incompatible lengths.")

    indices = np.searchsorted(old_edges[1:], midpoints, side="right")
    indices = np.minimum(indices, old_prices.size - 1)
    return old_prices[indices].copy()


def build_complex_deterministic_target():
    """Generate a jump-aware target from the 24-hour reference price signal."""
    baseline = _baseline()
    generator = bm_1(baseline, baseline, T, TARGET_T_STEPS)
    target_t = np.linspace(T[0], T[1], TARGET_T_STEPS, dtype=float)

    result = generator.deterministic_solver(
        [X0_DETERMINISTIC],
        _parameters_for_prices(HOURLY_EDGES, REFERENCE_PRICES_24),
        t_eval=target_t,
        breakpoints=HOURLY_EDGES[1:-1],
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=1.0 / (TARGET_T_STEPS - 1),
    )
    target_demand = np.asarray(result[7], dtype=float)

    # One interpolator per hour prevents interpolation across demand jumps.
    interpolators = []
    for hour in range(24):
        left = HOURLY_EDGES[hour]
        right = HOURLY_EDGES[hour + 1]
        if hour < 23:
            mask = (target_t >= left) & (target_t < right)
        else:
            mask = (target_t >= left) & (target_t <= right)

        local_t = target_t[mask]
        local_demand = target_demand[mask]
        if local_t.size < 3:
            raise RuntimeError("Target grid is too coarse inside an hourly interval.")
        interpolators.append(
            PchipInterpolator(local_t, local_demand, extrapolate=True)
        )

    def target(query):
        array = np.asarray(query, dtype=float)
        flat = array.reshape(-1)
        indices = np.searchsorted(
            HOURLY_EDGES[1:],
            flat,
            side="right",
        )
        indices = np.minimum(indices, 23)

        values = np.empty_like(flat, dtype=float)
        for hour in range(24):
            mask = indices == hour
            if np.any(mask):
                values[mask] = interpolators[hour](flat[mask])

        values = values.reshape(array.shape)
        return float(values) if array.ndim == 0 else values

    return target, target_t, target_demand


def build_stochastic_model(target):
    baseline = _baseline()
    return bm_1(baseline, target, T, 2)


def _build_evaluator(model, n_intervals, *, fast_fv=False):
    edges = _edges(n_intervals)
    spatial_edges, _, _ = model.fokker_planc_reference.grid(FV_X_STEPS)
    initial_mass = beta_initial_cell_mass(
        spatial_edges,
        alpha=INITIAL_ALPHA,
        beta=INITIAL_BETA,
    )
    evaluator_class = FastFVStochasticPriceEvaluator if fast_fv else FVStochasticPriceEvaluator
    return evaluator_class(
        model,
        _parameter_template(edges),
        edges,
        SIGMA_X,
        initial_mass,
        objective_max_step=OBJECTIVE_MAX_STEP,
        solver_max_step=SOLVER_MAX_STEP,
        # Every candidate uses the same 23 hourly boundaries, including
        # target jumps and all possible N=24 price jumps.
        breakpoints=HOURLY_EDGES[1:-1],
    )


def _unique_rows(*groups):
    rows = []
    for group in groups:
        array = np.asarray(group, dtype=float)
        if array.ndim == 1:
            array = array[None, :]
        for row in array:
            if not any(np.array_equal(row, previous) for previous in rows):
                rows.append(row.copy())
    return np.asarray(rows, dtype=float)


def _starts_for_case(n_intervals, previous_case=None):
    reference = _coarsen_reference_prices(n_intervals)
    neutral = np.full(n_intervals, 0.5, dtype=float)

    if previous_case is None:
        warm = reference
    else:
        warm = _sample_piecewise_prices(
            previous_case.edges,
            previous_case.optimization.best.x,
            _edges(n_intervals),
        )

    # 12D and 24D are already expensive.  Use the previous optimum plus the
    # coarsened/exact target-generating reference as strong independent starts.
    if n_intervals >= 12:
        return _unique_rows(warm, reference)

    phase = np.linspace(0.0, 2.0 * np.pi, n_intervals, endpoint=False)
    perturb_up = np.clip(warm + 0.05 * np.sin(phase), 0.0, 1.0)
    perturb_down = np.clip(warm - 0.05 * np.sin(phase), 0.0, 1.0)
    return _unique_rows(warm, reference, neutral, perturb_up, perturb_down)


def _lift_half_day_policy(two_prices, n_intervals):
    two_prices = np.asarray(two_prices, dtype=float).reshape(-1)
    n_intervals = int(n_intervals)
    if two_prices.size != 2 or n_intervals % 2:
        raise ValueError("Need two prices and an even interval count.")
    return np.concatenate(
        (
            np.full(n_intervals // 2, two_prices[0], dtype=float),
            np.full(n_intervals // 2, two_prices[1], dtype=float),
        )
    )


def _verify_common_numerical_partition(evaluators):
    """The same physical held signal must give the same cost for every N."""
    test_prices = np.array([0.72, 0.28], dtype=float)
    values = {}
    for count, evaluator in evaluators.items():
        prices = _lift_half_day_policy(test_prices, count)
        values[count] = evaluator.evaluate(prices).objective.total

    reference = values[min(values)]
    for count, value in values.items():
        if not np.isclose(value, reference, atol=5.0e-13, rtol=0.0):
            raise RuntimeError(
                "Common-policy check failed: "
                f"N={count}, J={value:.16e}, reference={reference:.16e}."
            )
    return values


def _evaluate_mc_common_random_numbers(model, cases):
    reference_evaluator = cases[min(cases)].evaluator
    solver_t_eval = reference_evaluator.solver_t_eval
    widths = reference_evaluator.widths
    target = reference_evaluator.target

    for case in cases.values():
        np.testing.assert_array_equal(case.evaluator.solver_t_eval, solver_t_eval)
        np.testing.assert_array_equal(case.evaluator.widths, widths)
        np.testing.assert_array_equal(case.evaluator.target, target)

    initial_states = sample_beta_initial_states(
        MC_PATHS,
        MC_INITIAL_SEED,
        alpha=INITIAL_ALPHA,
        beta=INITIAL_BETA,
    )
    integration_t, _ = model.euler_maruyama.build_time_grid(
        solver_t_eval,
        breakpoints=HOURLY_EDGES[1:-1],
        max_step=SOLVER_MAX_STEP,
    )
    dW = model.euler_maruyama.brownian_increments(
        integration_t,
        n_paths=MC_PATHS,
        seed=MC_BROWNIAN_SEED,
    )

    results = {}
    for count, case in cases.items():
        parameters = case.evaluator.parameters_for_prices(
            case.optimization.best.x
        )
        output = model.euler_maruyama_solver(
            initial_states,
            parameters,
            SIGMA_X,
            n_paths=MC_PATHS,
            boundary="none",
            t_eval=solver_t_eval,
            breakpoints=HOURLY_EDGES[1:-1],
            max_step=SOLVER_MAX_STEP,
            brownian_increments=dW,
            return_diagnostics=True,
        )
        _, _, _, _, _, _, _, demand, diagnostics = output
        if diagnostics["total_violations"]:
            raise RuntimeError(
                f"N={count} MC validation produced raw boundary violations."
            )

        results[count] = monte_carlo_tracking_objective(
            widths,
            np.asarray(demand, dtype=float)[:, 1:-1],
            target,
        )

    return results


def _paired_improvement(reference, candidate):
    difference = (
        np.asarray(reference.path_objectives, dtype=float)
        - np.asarray(candidate.path_objectives, dtype=float)
    )
    mean = float(np.mean(difference))
    se = float(np.std(difference, ddof=1) / np.sqrt(difference.size))
    return mean, se


def run_complex_target_experiment(progress=print, validate_mc=True, *, fast_fv=False):
    target, target_t, target_demand = build_complex_deterministic_target()
    model = build_stochastic_model(target)

    evaluators = {
        count: _build_evaluator(model, count, fast_fv=fast_fv)
        for count in INTERVAL_COUNTS
    }

    common_policy_values = _verify_common_numerical_partition(evaluators)
    if progress is not None:
        progress("Common numerical-partition check:")
        for count in INTERVAL_COUNTS:
            progress(f"  N={count}: J={common_policy_values[count]:.10e}")

    cases = {}
    previous = None
    for index, count in enumerate(INTERVAL_COUNTS, start=1):
        evaluator = evaluators[count]
        starts = _starts_for_case(count, previous)

        if progress is not None:
            progress("")
            progress(
                f"[{index}/{len(INTERVAL_COUNTS)}] Optimizing "
                f"{count} price intervals ({count - 1} possible changes)"
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
        evaluation = evaluator.evaluate(optimization.best.x)
        case = ComplexTargetCase(
            n_intervals=count,
            edges=evaluator.interval_edges.copy(),
            evaluator=evaluator,
            optimization=optimization,
            evaluation=evaluation,
        )
        cases[count] = case
        previous = case

    mc = _evaluate_mc_common_random_numbers(model, cases) if validate_mc else None

    return {
        "model": model,
        "target_t": target_t,
        "target_demand": target_demand,
        "common_policy_values": common_policy_values,
        "cases": cases,
        "mc": mc,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-fv", action="store_true", help="Use the numerically equivalent LAPACK/blocked FV evaluator.")
    args = parser.parse_args()
    print("FV backend: " + ("fast" if args.fast_fv else "reference"), flush=True)
    result = run_complex_target_experiment(
        progress=lambda message: print(message, flush=True), fast_fv=args.fast_fv
    )
    cases = result["cases"]

    print()
    print("BM1 complex-target multi-price stochastic optimization")
    print("------------------------------------------------------")
    print("Target-generating hourly reference prices:")
    print(
        np.array2string(
            REFERENCE_PRICES_24,
            precision=3,
            floatmode="fixed",
            max_line_width=120,
        )
    )
    print()
    print(
        "intervals  changes  FV objective       bias               variance           "
        "actual evals  cache hits"
    )
    for count, case in cases.items():
        objective = case.evaluation.objective
        optimization = case.optimization
        print(
            f"{count:9d}  {count - 1:7d}  "
            f"{objective.total:.10e}  "
            f"{objective.bias:.10e}  "
            f"{objective.variance:.10e}  "
            f"{optimization.objective_evaluations:12d}  "
            f"{optimization.cache_hits:10d}"
        )
        print(
            " " * 20
            + "prices="
            + np.array2string(
                np.asarray(optimization.best.x, dtype=float),
                precision=6,
                floatmode="fixed",
                max_line_width=120,
            )
        )

    print()
    print("Marginal pure-FV improvement")
    previous = None
    for count, case in cases.items():
        if previous is not None:
            improvement = (
                previous.evaluation.objective.total
                - case.evaluation.objective.total
            )
            relative = improvement / previous.evaluation.objective.total
            print(
                f"N={previous.n_intervals} -> N={count}: "
                f"dJ={improvement:.6e} ({relative:.3%})"
            )
        previous = case

    if result["mc"] is not None:
        print()
        print(f"Common-random-number Monte Carlo ({MC_PATHS} paths)")
        for count in INTERVAL_COUNTS:
            item = result["mc"][count]
            print(
                f"N={count}: J={item.objective.total:.10e}, "
                f"SE={item.standard_error:.3e}"
            )

        reference = result["mc"][INTERVAL_COUNTS[0]]
        print()
        print(f"Paired MC improvement relative to N={INTERVAL_COUNTS[0]}")
        for count in INTERVAL_COUNTS[1:]:
            mean, se = _paired_improvement(reference, result["mc"][count])
            ratio = mean / se if se > 0.0 else np.inf
            print(
                f"N={count}: improvement={mean:.6e}, "
                f"SE={se:.3e}, ratio={ratio:.3f}"
            )

    fig, axes = plt.subplots(3, 1, figsize=(13, 14))
    ax_price, ax_objective, ax_demand = axes

    plot_t = np.linspace(T[0], T[1], 1441, dtype=float)
    reference_price = input_builder.price_intervals(
        (HOURLY_EDGES, REFERENCE_PRICES_24)
    )
    ax_price.step(
        24.0 * plot_t,
        reference_price(plot_t),
        where="post",
        linewidth=2.5,
        linestyle="--",
        label="Target-generating N=24 reference",
    )
    for count, case in cases.items():
        price = input_builder.price_intervals(
            (case.edges, case.optimization.best.x)
        )
        ax_price.step(
            24.0 * plot_t,
            price(plot_t),
            where="post",
            label=f"Optimized N={count}",
        )
    ax_price.set_ylabel("Normalized price")
    ax_price.set_ylim(0.0, 1.0)
    ax_price.set_title("Reference and optimized stochastic price schedules")
    ax_price.grid(True)
    ax_price.legend(loc="best")

    counts = np.asarray(INTERVAL_COUNTS, dtype=int)
    totals = np.array(
        [cases[count].evaluation.objective.total for count in counts]
    )
    bias = np.array(
        [cases[count].evaluation.objective.bias for count in counts]
    )
    variance = np.array(
        [cases[count].evaluation.objective.variance for count in counts]
    )
    ax_objective.plot(counts, totals, marker="o", label="Total J")
    ax_objective.plot(counts, bias, marker="o", label="Bias contribution")
    ax_objective.plot(counts, variance, marker="o", label="Variance contribution")
    ax_objective.set_xlabel("Number of price intervals")
    ax_objective.set_ylabel("Pure-FV stochastic objective")
    ax_objective.set_title("Value of price resolution for a complex target")
    ax_objective.grid(True)
    ax_objective.legend(loc="best")

    reference_evaluator = cases[INTERVAL_COUNTS[0]].evaluator
    hours = 24.0 * reference_evaluator.midpoints
    ax_demand.plot(
        hours,
        reference_evaluator.target,
        linewidth=2.5,
        label="Target A(t)",
    )
    for count, case in cases.items():
        ax_demand.plot(
            hours,
            case.evaluation.objective.expected_demand,
            label=f"E[D], N={count}",
        )
    ax_demand.set_xlabel("Time [h]")
    ax_demand.set_ylabel("Expected demand")
    ax_demand.set_title("Expected demand under optimized price schedules")
    ax_demand.grid(True)
    ax_demand.legend(loc="best")

    fig.tight_layout()
    graph_dir = (
        Path(__file__).resolve().parents[4]
        / "files"
        / "output"
        / "graphs"
        / "bm_1"
    )
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "bm_1_complex_target_multi_price_experiment.png"
    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print()
    print(f"Saved graph to {graph_path}")


if __name__ == "__main__":
    main()
