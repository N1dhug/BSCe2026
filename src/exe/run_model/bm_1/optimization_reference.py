"""Reference verification for deterministic bm_1 price optimization (F1-D).

The target demand is generated from a known two-interval price signal.  The
optimizer starts from different prices and must recover the feasible target
without tying the number of price decisions to the ODE output grid.

Run this module from the repository root:

    python -B -m src.exe.run_model.bm_1.optimization_reference
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from src.method.objective import tracking_error
from src.tools.input_generator.input_builder import input_builder
from src.model.bm_1 import bm_1


T = (0.0, 1.0)
INTERVAL_EDGES = np.array([T[0], 0.5, T[1]], dtype=float)
KNOWN_PRICES = np.array([0.72, 0.28], dtype=float)
INITIAL_PRICES = np.array([0.55, 0.45], dtype=float)
OBJECTIVE_T = np.linspace(T[0], T[1], 193)

X0 = 0.5
BASELINE = 0.3
C = 1.0
K = 2.0
XI = 1.0
RTOL = 1.0e-9
ATOL = 1.0e-11

TARGET_TOL = 1.0e-16
OBJECTIVE_TOL = 1.0e-12
PRICE_TOL = 1.0e-5


def linear_state_response(x):
    """Return the benchmark response f(x) = 1 - 2x."""
    return 1.0 - 2.0 * np.asarray(x, dtype=float)


def linear_price_response(u):
    """Return the benchmark response g(u) = 1 - 2u."""
    return 1.0 - 2.0 * np.asarray(u, dtype=float)


def build_baseline():
    """Return the constant normalized baseline used by the benchmark."""
    return input_builder.energy(
        (np.array(T, dtype=float), np.array([BASELINE, BASELINE], dtype=float))
    )


def build_parameters(prices):
    """Build bm_1 parameters for one price value per interval."""
    price = input_builder.price_intervals((INTERVAL_EDGES, prices))
    return [
        price,
        linear_state_response,
        linear_price_response,
        C,
        K,
        XI,
    ]


def solve_prices(model, prices):
    """Solve bm_1 for a candidate price vector on the objective grid."""
    return model.deterministic_solver(
        [X0],
        build_parameters(prices),
        t_eval=OBJECTIVE_T,
        breakpoints=INTERVAL_EDGES[1:-1],
        rtol=RTOL,
        atol=ATOL,
    )


def raises_value_error(action):
    """Return True when an input-validation probe raises ValueError."""
    try:
        action()
    except ValueError:
        return True
    return False


def save_figure(target_result, initial_result, optimized_result, optimized_prices):
    """Save the feasible target, recovered demand, state, and prices."""
    hours = 24.0 * OBJECTIVE_T
    target_demand = target_result[7]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(hours, target_demand, linewidth=2.5, label="Target A(t)")
    axes[0].plot(
        hours,
        initial_result[7],
        linestyle="--",
        label="Demand from initial prices",
    )
    axes[0].plot(
        hours,
        optimized_result[7],
        linestyle=":",
        linewidth=2.5,
        label="Optimized demand",
    )
    axes[0].set_ylabel("Normalized demand")
    axes[0].set_title("bm_1 deterministic price optimization (F1-D)")
    axes[0].grid(True)
    axes[0].legend(loc="best")

    axes[1].plot(hours, target_result[1], linewidth=2.5, label="Target state")
    axes[1].plot(
        hours,
        optimized_result[1],
        linestyle="--",
        label="Optimized state",
    )
    axes[1].set_ylabel("State X")
    axes[1].grid(True)
    axes[1].legend(loc="best")

    known_price = input_builder.price_intervals((INTERVAL_EDGES, KNOWN_PRICES))
    initial_price = input_builder.price_intervals((INTERVAL_EDGES, INITIAL_PRICES))
    optimized_price = input_builder.price_intervals(
        (INTERVAL_EDGES, optimized_prices)
    )
    axes[2].step(
        hours,
        known_price(OBJECTIVE_T),
        where="post",
        linewidth=2.5,
        label="Known prices",
    )
    axes[2].step(
        hours,
        initial_price(OBJECTIVE_T),
        where="post",
        linestyle="--",
        label="Initial prices",
    )
    axes[2].step(
        hours,
        optimized_price(OBJECTIVE_T),
        where="post",
        linestyle=":",
        linewidth=2.5,
        label="Optimized prices",
    )
    axes[2].set_xlabel("Time [hours]")
    axes[2].set_ylabel("Normalized price")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True)
    axes[2].legend(loc="best")

    for axis in axes:
        for edge in INTERVAL_EDGES[1:-1]:
            axis.axvline(24.0 * edge, color="black", alpha=0.3, linestyle="--")

    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / "files" / "output" / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "BM1_optimization_reference.png"
    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return graph_path


def main():
    """Generate a feasible target and verify deterministic optimization."""
    baseline = build_baseline()
    generator = bm_1(baseline, baseline, T, OBJECTIVE_T.size)
    target_result = solve_prices(generator, KNOWN_PRICES)
    target_demand = target_result[7]

    target = interp1d(
        OBJECTIVE_T,
        target_demand,
        kind="linear",
        bounds_error=True,
    )
    model = bm_1(baseline, target, T, OBJECTIVE_T.size)

    initial_result = solve_prices(model, INITIAL_PRICES)
    known_objective = tracking_error(
        OBJECTIVE_T, target_demand, target_demand
    )
    initial_objective = tracking_error(
        OBJECTIVE_T, initial_result[7], target_demand
    )

    result = model.optimize(
        INITIAL_PRICES,
        [X0],
        build_parameters(INITIAL_PRICES),
        interval_edges=INTERVAL_EDGES,
        t_eval=OBJECTIVE_T,
        rtol=RTOL,
        atol=ATOL,
        options={"ftol": 1.0e-15, "gtol": 1.0e-10, "maxiter": 200},
    )
    optimized_result = solve_prices(model, result.x)
    optimized_objective = tracking_error(
        OBJECTIVE_T, optimized_result[7], target_demand
    )

    graph_path = save_figure(
        target_result,
        initial_result,
        optimized_result,
        result.x,
    )

    checks = {
        "optimizer reports success": bool(result.success),
        "known feasible target has zero objective": (
            known_objective <= TARGET_TOL
        ),
        "optimized objective reaches the target tolerance": (
            optimized_objective <= OBJECTIVE_TOL
        ),
        "reported and recomputed objectives agree": np.isclose(
            result.fun,
            optimized_objective,
            rtol=1.0e-6,
            atol=1.0e-16,
        ),
        "optimization improves the initial objective": (
            optimized_objective < initial_objective
        ),
        "optimizer recovers the known price decisions": (
            np.max(np.abs(result.x - KNOWN_PRICES)) <= PRICE_TOL
        ),
        "optimized prices remain in [0, 1]": (
            np.all(result.x >= 0.0) and np.all(result.x <= 1.0)
        ),
        "price decisions remain independent of the objective grid": (
            result.x.size + 1 == INTERVAL_EDGES.size
            and result.x.size != OBJECTIVE_T.size
        ),
        "mismatched interval counts are rejected": raises_value_error(
            lambda: model.optimize(
                INITIAL_PRICES,
                [X0],
                build_parameters(INITIAL_PRICES),
                interval_edges=np.array(T, dtype=float),
            )
        ),
        "interval edges must cover the model horizon": raises_value_error(
            lambda: model.optimize(
                INITIAL_PRICES,
                [X0],
                build_parameters(INITIAL_PRICES),
                interval_edges=np.array([0.1, 0.5, 1.0]),
            )
        ),
        "objective grid must cover the model horizon": raises_value_error(
            lambda: model.optimize(
                INITIAL_PRICES,
                [X0],
                build_parameters(INITIAL_PRICES),
                interval_edges=INTERVAL_EDGES,
                t_eval=OBJECTIVE_T[1:],
            )
        ),
        "reference figure is saved": (
            graph_path.is_file() and graph_path.stat().st_size > 0
        ),
    }

    improvement = (
        initial_objective / optimized_objective
        if optimized_objective > 0.0
        else np.inf
    )
    price_error = float(np.max(np.abs(result.x - KNOWN_PRICES)))

    print("bm_1 deterministic optimization F1-D verification")
    print("================================================")
    print()
    print(f"known prices             = {KNOWN_PRICES}")
    print(f"initial prices           = {INITIAL_PRICES}")
    print(f"optimized prices         = {result.x}")
    print(f"max price error          = {price_error:.6e}")
    print(f"known objective          = {known_objective:.6e}")
    print(f"initial objective        = {initial_objective:.6e}")
    print(f"optimized objective      = {optimized_objective:.6e}")
    print(f"objective improvement    = {improvement:.3e}")
    print(f"iterations / evaluations = {result.nit} / {result.nfev}")
    print()
    print("Checks")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print()
    print(f"Saved graph to {graph_path}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print()
        print("One or more F1-D checks failed:")
        for name in failed:
            print(f"- {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
