"""Explore stochastic BM1 optimization with more than one price change.

The completed F2-I reference uses two half-day price intervals.  This experiment
keeps the same stochastic model and deterministic target but increases the
number of equally sized price intervals to 2, 4, 6 and 24.  The 24-interval
case corresponds to one price decision per hour.

All cases use the union of every candidate interval boundary as additional
internal breakpoints.  This keeps the numerical time partition comparable
across parameterizations.  The two-price policy space is embedded exactly in
the four- and six-price spaces.

The experiment uses pure FV inside Powell.  Final candidates are optionally
validated with common-random-number Monte Carlo.

Run from the repository root:

    python -B -m src.exe.run_model.bm_1.multi_price_optimization_experiment
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    build_deterministic_target,
    build_stochastic_model,
    linear_price_response,
    linear_state_response,
)
from src.method.bm_1_stochastic_objective import monte_carlo_tracking_objective
from src.method.bm_1_stochastic_optimization import (
    FVStochasticPriceEvaluator,
    multistart_powell,
)
from src.tools.input_generator.input_builder import input_builder


INTERVAL_COUNTS = (2, 4, 6, 24)
FV_X_STEPS = 201
OBJECTIVE_MAX_STEP = 1.0 / 96.0
SOLVER_MAX_STEP = 1.0 / 96.0

# Exploratory MC check. Increase later if the paired differences are marginal.
MC_PATHS = 20_000
MC_INITIAL_SEED = 2041
MC_BROWNIAN_SEED = 2042

DETERMINISTIC_REFERENCE_PRICES = np.array([0.72, 0.28], dtype=float)
NEUTRAL_PRICES = np.array([0.5, 0.5], dtype=float)


@dataclass(frozen=True)
class MultiPriceCase:
    n_intervals: int
    edges: np.ndarray
    evaluator: FVStochasticPriceEvaluator
    optimization: object
    evaluation: object


def _edges(n_intervals):
    n_intervals = int(n_intervals)
    if n_intervals < 1:
        raise ValueError("n_intervals must be positive.")
    return np.linspace(T[0], T[1], n_intervals + 1, dtype=float)


def _union_breakpoints(interval_counts=INTERVAL_COUNTS):
    points = []
    for count in interval_counts:
        points.extend(_edges(count)[1:-1])
    return np.unique(np.asarray(points, dtype=float))


def _lift_half_day_prices(two_prices, n_intervals):
    """Represent one first-half and one second-half price on an even grid."""
    two_prices = np.asarray(two_prices, dtype=float).reshape(-1)
    n_intervals = int(n_intervals)
    if two_prices.size != 2:
        raise ValueError("two_prices must contain exactly two values.")
    if n_intervals < 2 or n_intervals % 2:
        raise ValueError("n_intervals must be an even integer >= 2.")
    return np.concatenate(
        (
            np.full(n_intervals // 2, two_prices[0], dtype=float),
            np.full(n_intervals // 2, two_prices[1], dtype=float),
        )
    )


def _sample_piecewise_prices(old_edges, old_prices, new_edges):
    """Sample an old held-price signal at the midpoints of a new grid."""
    old_edges = np.asarray(old_edges, dtype=float)
    old_prices = np.asarray(old_prices, dtype=float)
    new_edges = np.asarray(new_edges, dtype=float)
    midpoints = 0.5 * (new_edges[:-1] + new_edges[1:])

    if old_prices.size + 1 != old_edges.size:
        raise ValueError("old_edges and old_prices have incompatible lengths.")

    indices = np.searchsorted(old_edges[1:], midpoints, side="right")
    indices = np.minimum(indices, old_prices.size - 1)
    return old_prices[indices].copy()


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


def _build_evaluator(model, n_intervals, common_breakpoints):
    edges = _edges(n_intervals)
    spatial_edges, _, _ = model.fokker_planc_reference.grid(FV_X_STEPS)
    initial_mass = beta_initial_cell_mass(
        spatial_edges,
        alpha=INITIAL_ALPHA,
        beta=INITIAL_BETA,
    )
    return FVStochasticPriceEvaluator(
        model,
        _parameter_template(edges),
        edges,
        SIGMA_X,
        initial_mass,
        objective_max_step=OBJECTIVE_MAX_STEP,
        solver_max_step=SOLVER_MAX_STEP,
        breakpoints=common_breakpoints,
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
    deterministic = _lift_half_day_prices(
        DETERMINISTIC_REFERENCE_PRICES,
        n_intervals,
    )
    neutral = np.full(n_intervals, 0.5, dtype=float)

    if previous_case is None:
        warm = deterministic
    else:
        warm = _sample_piecewise_prices(
            previous_case.edges,
            previous_case.optimization.best.x,
            _edges(n_intervals),
        )

    # Deterministic perturbations give Powell another local-basin probe without
    # introducing randomness into the experiment.
    phase = np.linspace(0.0, np.pi, n_intervals)
    up_down = np.clip(warm + 0.04 * np.cos(phase), 0.0, 1.0)
    down_up = np.clip(warm - 0.04 * np.cos(phase), 0.0, 1.0)

    # At 24 hourly decisions, full multistart Powell becomes very expensive.
    # The previous lower-dimensional optimum is already a strong warm start,
    # and the lifted deterministic reference supplies an independent second
    # start.  Keep the richer set for the smaller 2/4/6 experiments.
    if n_intervals >= 24:
        return _unique_rows(warm, deterministic)

    return _unique_rows(warm, deterministic, neutral, up_down, down_up)


def _verify_nested_parameterizations(evaluators):
    """Check that the same held two-price signal has the same FV objective."""
    values = {}
    for count, evaluator in evaluators.items():
        prices = _lift_half_day_prices(
            DETERMINISTIC_REFERENCE_PRICES,
            count,
        )
        values[count] = evaluator.evaluate(prices).objective.total

    reference = values[min(values)]
    for count, value in values.items():
        if not np.isclose(value, reference, atol=5.0e-13, rtol=0.0):
            raise RuntimeError(
                "Nested price parameterizations do not evaluate the same "
                f"physical two-price policy consistently: N={count}, "
                f"J={value:.16e}, reference={reference:.16e}."
            )
    return values


def _evaluate_mc_common_random_numbers(model, cases, common_breakpoints):
    """Validate all optimized cases on one common EM time/Wiener grid."""
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
        breakpoints=common_breakpoints,
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
            breakpoints=common_breakpoints,
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


def run_multi_price_experiment(progress=print, validate_mc=True):
    target, _, _ = build_deterministic_target()
    model = build_stochastic_model(target)

    common_breakpoints = _union_breakpoints()
    evaluators = {
        count: _build_evaluator(model, count, common_breakpoints)
        for count in INTERVAL_COUNTS
    }

    nested_values = _verify_nested_parameterizations(evaluators)
    if progress is not None:
        progress("Nested two-price policy check:")
        for count in INTERVAL_COUNTS:
            progress(f"  N={count}: J={nested_values[count]:.10e}")

    cases = {}
    previous = None
    for index, count in enumerate(INTERVAL_COUNTS, start=1):
        evaluator = evaluators[count]
        starts = _starts_for_case(count, previous)

        if progress is not None:
            progress("")
            progress(
                f"[{index}/{len(INTERVAL_COUNTS)}] Optimizing "
                f"{count} price intervals ({count - 1} changes)"
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
        case = MultiPriceCase(
            n_intervals=count,
            edges=evaluator.interval_edges.copy(),
            evaluator=evaluator,
            optimization=optimization,
            evaluation=evaluation,
        )
        cases[count] = case
        previous = case

    mc = (
        _evaluate_mc_common_random_numbers(model, cases, common_breakpoints)
        if validate_mc
        else None
    )

    return {
        "model": model,
        "common_breakpoints": common_breakpoints,
        "nested_values": nested_values,
        "cases": cases,
        "mc": mc,
    }


def main():
    result = run_multi_price_experiment(progress=lambda message: print(message, flush=True))
    cases = result["cases"]

    print()
    print("BM1 multi-price stochastic optimization experiment")
    print("------------------------------------------------")
    print(
        "All cases: same target, Beta(20,20), sigma=0.5, pure FV objective, "
        "common internal breakpoints."
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

    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    ax_price, ax_objective, ax_demand = axes

    for count, case in cases.items():
        price = input_builder.price_intervals(
            (case.edges, case.optimization.best.x)
        )
        plot_t = np.linspace(T[0], T[1], 721)
        ax_price.step(
            24.0 * plot_t,
            price(plot_t),
            where="post",
            label=f"N={count}",
        )
    ax_price.set_ylabel("Normalized price")
    ax_price.set_ylim(0.0, 1.0)
    ax_price.set_title("Optimized stochastic price schedules")
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
    ax_objective.set_title("Value of additional price decisions")
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
    graph_path = graph_dir / "bm_1_multi_price_optimization_experiment.png"
    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print()
    print(f"Saved graph to {graph_path}")


if __name__ == "__main__":
    main()
