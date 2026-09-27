"""F3-A reference: common BM1 control-problem contract.

The script reconstructs the controlled two-price F2/F3 reference through the
new reusable contract. It first evaluates the attached fixed external price
path without optimization. With --optimize it then uses the same evaluator as
an open-loop decision problem.

Run from the repository root:

    python -B -m src.exe.run_model.bm_1.f3_a_control_problem_reference
    python -B -m src.exe.run_model.bm_1.f3_a_control_problem_reference --optimize
"""

import argparse

import numpy as np

from src.exe.run_model.bm_1.f2_reference_case import (
    INITIAL_ALPHA,
    INITIAL_BETA,
    SIGMA_X,
    beta_initial_cell_mass,
)
from src.exe.run_model.bm_1.stochastic_optimization_reference import (
    C,
    INITIAL_PRICES,
    INTERVAL_EDGES,
    K,
    KNOWN_DETERMINISTIC_PRICES,
    T,
    XI,
    build_deterministic_target,
    build_stochastic_model,
    linear_price_response,
    linear_state_response,
)
from src.method.bm_1_control_problem import (
    BM1ControlNumerics,
    BM1ControlProblem,
    BM1Parameters,
)
from src.method.bm_1_stochastic_optimization import multistart_powell


def _initial_mass_builder(edges):
    return beta_initial_cell_mass(
        edges,
        alpha=INITIAL_ALPHA,
        beta=INITIAL_BETA,
    )


def build_f3_a_problem():
    """Return the F3-A problem using the established two-price reference."""
    target, _, _ = build_deterministic_target()
    baseline = build_stochastic_model(target).B
    parameters = BM1Parameters(
        state_response=linear_state_response,
        price_response=linear_price_response,
        C=C,
        k=K,
        Xi=XI,
        sigma_x=SIGMA_X,
    )
    problem = BM1ControlProblem(
        baseline=baseline,
        target=target,
        horizon=T,
        price_edges=INTERVAL_EDGES,
        parameters=parameters,
        initial_mass_builder=_initial_mass_builder,
        # In F3-A these are a known fixed external/reference history. The
        # optimizer receives separate candidate vectors and never overwrites
        # this field.
        observed_prices=KNOWN_DETERMINISTIC_PRICES,
    )
    numerics = BM1ControlNumerics(
        x_steps=201,
        objective_max_step=1.0 / 96.0,
        solver_max_step=1.0 / 96.0,
    )
    return problem, numerics


def run_reference(*, optimize=False, progress=print):
    problem, numerics = build_f3_a_problem()

    fixed = problem.evaluate_observed(numerics, fast=True)
    result = {
        "problem": problem,
        "numerics": numerics,
        "fixed": fixed,
        "optimization": None,
    }

    if optimize:
        evaluator = problem.build_evaluator(numerics, fast=True)
        starts = np.vstack(
            (
                problem.observed_prices,
                INITIAL_PRICES,
                np.full(problem.n_intervals, 0.5, dtype=float),
            )
        )
        result["optimization"] = multistart_powell(
            evaluator,
            starts,
            progress=progress,
            options={
                "xtol": 2.0e-5,
                "ftol": 1.0e-9,
                "maxiter": 100,
            },
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="also solve the bounded open-loop price problem",
    )
    args = parser.parse_args()

    result = run_reference(
        optimize=args.optimize,
        progress=lambda message: print(message, flush=True),
    )
    problem = result["problem"]
    fixed = result["fixed"]

    print("F3-A common BM1 control problem")
    print("-------------------------------")
    print(f"price intervals: {problem.n_intervals}")
    print(
        "fixed external/reference prices: "
        + np.array2string(problem.observed_prices, precision=6)
    )
    print(
        f"fixed-price J={fixed.objective.total:.8e}, "
        f"bias={fixed.objective.bias:.8e}, "
        f"variance={fixed.objective.variance:.8e}"
    )
    print(
        "The fixed path above was evaluated unchanged; "
        "no price optimization was performed."
    )

    if result["optimization"] is not None:
        optimization = result["optimization"]
        print()
        print("Open-loop candidate optimization")
        print("--------------------------------")
        print(
            "best prices: "
            + np.array2string(
                np.asarray(optimization.best.x, dtype=float),
                precision=6,
            )
        )
        print(f"best J: {float(optimization.best.fun):.8e}")
        print(
            f"actual objective evaluations/cache hits: "
            f"{optimization.objective_evaluations}/{optimization.cache_hits}"
        )


if __name__ == "__main__":
    main()
