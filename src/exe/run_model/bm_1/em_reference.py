"""Controlled Euler-Maruyama / Monte Carlo reference for stochastic bm_1.

This F2-B experiment keeps its original point initial state, but now reuses the
shared BM-only F2 reference inputs from f2_reference_case.

Run from the repository root:

    python -B -m src.exe.run_model.bm_1.em_reference
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.exe.run_model.bm_1.f2_reference_case import (
    MAX_STEP,
    PRICE_BREAKPOINTS,
    PRICE_DROP_HOUR,
    PRICE_DROP_TIME,
    SIGMA_X,
    T_EVAL,
    X0,
    build_reference_case,
)


N_PATHS = 20_000
SEED = 2026

def main():
    """Run the reference experiment and save its diagnostic figure."""
    model, parameters = build_reference_case()

    integration_t, output_indices = model.euler_maruyama.build_time_grid(
        T_EVAL,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=MAX_STEP,
    )
    dW = model.euler_maruyama.brownian_increments(
        integration_t,
        n_paths=N_PATHS,
        seed=SEED,
    )

    result = model.euler_maruyama_solver(
        [X0],
        parameters,
        SIGMA_X,
        n_paths=N_PATHS,
        boundary="none",
        t_eval=T_EVAL,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=MAX_STEP,
        brownian_increments=dW,
        return_diagnostics=True,
    )

    (
        t,
        state,
        price,
        baseline,
        _,
        _,
        _,
        demand,
        diagnostics,
    ) = result

    state_mean = np.mean(state, axis=0)
    state_var = np.var(state, axis=0, ddof=1)
    state_se = np.sqrt(state_var / N_PATHS)
    demand_mean = np.mean(demand, axis=0)
    demand_var = np.var(demand, axis=0, ddof=1)
    demand_se = np.sqrt(demand_var / N_PATHS)

    print("F2-B controlled Euler-Maruyama reference")
    print("----------------------------------------")
    print(f"Output points:                 {len(t)}")
    print(f"Internal EM steps:             {diagnostics['n_steps']}")
    print(f"Maximum internal step [h]:     {24.0 * np.max(np.diff(diagnostics['integration_t'])):.6f}")
    print(f"Price breakpoint included:     {bool(np.any(diagnostics['integration_t'] == PRICE_DROP_TIME))}")
    print(f"Raw proposal minimum:          {diagnostics['minimum_raw_proposal']:.6f}")
    print(f"Raw proposal maximum:          {diagnostics['maximum_raw_proposal']:.6f}")
    print(f"Lower-bound violations:        {diagnostics['lower_violations']}")
    print(f"Upper-bound violations:        {diagnostics['upper_violations']}")
    print(f"Paths with any violation:      {diagnostics['violating_paths']}")
    print(f"Boundary corrections applied:  {diagnostics['boundary_corrections']}")
    print()
    print(f"Terminal E[X]:                 {state_mean[-1]:.6f}")
    print(f"Terminal SE(E[X]):             {state_se[-1]:.3e}")
    print(f"Terminal E[D]:                 {demand_mean[-1]:.6f}")
    print(f"Terminal SE(E[D]):             {demand_se[-1]:.3e}")

    if diagnostics["total_violations"]:
        raise RuntimeError(
            "The uncorrected EM reference left [0, 1]. Reduce max_step or noise "
            "before using this run as a bounded-state reference."
        )

    hours = 24.0 * t
    fig, (ax_state, ax_paths, ax_demand) = plt.subplots(nrows=3, figsize=(11, 11))

    ax_state.step(hours, price, where="post", label="Price U(t)")
    ax_state.plot(hours, state_mean, linewidth=2.0, label="Monte Carlo E[X(t)]")
    ax_state.fill_between(
        hours,
        state_mean - 1.96 * state_se,
        state_mean + 1.96 * state_se,
        alpha=0.2,
        label="95% MC interval for E[X]",
    )
    ax_state.axvline(PRICE_DROP_HOUR, linestyle="--", label="Price drop")
    ax_state.set_ylabel("Price / SoC")
    ax_state.set_title("Controlled Euler-Maruyama mean and sampling uncertainty")
    ax_state.grid(True)
    ax_state.legend(loc="best")

    for path in state[:30]:
        ax_paths.plot(hours, path, alpha=0.2)
    ax_paths.plot(hours, state_mean, linewidth=2.5, label="E[X(t)]")
    ax_paths.axvline(PRICE_DROP_HOUR, linestyle="--")
    ax_paths.set_ylabel("State of charge")
    ax_paths.set_title("30 uncorrected Euler-Maruyama sample paths")
    ax_paths.set_ylim(0.0, 1.0)
    ax_paths.grid(True)
    ax_paths.legend(loc="best")

    ax_demand.plot(hours, demand_mean, linewidth=2.0, label="Monte Carlo E[D(t)]")
    ax_demand.fill_between(
        hours,
        demand_mean - 1.96 * demand_se,
        demand_mean + 1.96 * demand_se,
        alpha=0.2,
        label="95% MC interval for E[D]",
    )
    ax_demand.plot(hours, baseline, linestyle="--", label="Baseline B(t)")
    ax_demand.axvline(PRICE_DROP_HOUR, linestyle="--")
    ax_demand.set_xlabel("Time [h]")
    ax_demand.set_ylabel("Normalized demand")
    ax_demand.set_title("Expected demand and Monte Carlo sampling uncertainty")
    ax_demand.grid(True)
    ax_demand.legend(loc="best")

    fig.tight_layout()
    graph_dir = Path(__file__).resolve().parents[4] / "files" / "output" / "graphs" / "bm_1"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "bm_1_em_reference.png"
    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    np.testing.assert_array_equal(t, T_EVAL)
    np.testing.assert_array_equal(diagnostics["integration_t"][output_indices], T_EVAL)
    print(f"Saved graph to {graph_path}")


if __name__ == "__main__":
    main()
