"""F2-H stochastic tracking-objective reference for BM1.

The same fixed price path and stochastic initial law are evaluated with

1. Euler-Maruyama Monte Carlo,
2. pure finite-volume Fokker-Planck,
3. dynamic Beta-mixture approximation.

The objective is

    J(U) = integral E[(D_t - A(t))^2] dt

and is reported as bias-squared plus demand-variance contributions.

Run from the repository root:

    python -B -m src.exe.run_model.bm_1.stochastic_objective_reference
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.exe.run_model.bm_1.beta_dynamic_reference import (
    DEFAULT_CFL,
    run_dynamic_mixture_study,
)
from src.exe.run_model.bm_1.f2_reference_case import (
    MAX_STEP,
    PRICE_BREAKPOINTS,
    PRICE_DROP_HOUR,
    SIGMA_X,
    T,
    X_STEPS,
    build_reference_case,
    sample_beta_initial_states,
)
from src.method.bm_1_stochastic_objective import (
    discrete_demand_moments,
    demand_grid as _demand_grid,
    midpoint_quadrature,
    monte_carlo_tracking_objective,
    stochastic_tracking_objective,
)


DEFAULT_N_PATHS = 50_000
DEFAULT_DEGREES = (20, 30, 38)
OBJECTIVE_MAX_STEP = MAX_STEP
INITIAL_SEED = 2031
BROWNIAN_SEED = 2032




def _distribution_objective(widths, target, demand_values, cell_mass):
    expected, second, variance = discrete_demand_moments(
        demand_values,
        cell_mass,
    )
    result = stochastic_tracking_objective(
        widths,
        expected,
        second,
        target,
    )
    np.testing.assert_allclose(
        result.demand_variance,
        variance,
        atol=1.0e-14,
        rtol=0.0,
    )
    return result


def run_objective_reference(
    *,
    n_paths=DEFAULT_N_PATHS,
    degrees=DEFAULT_DEGREES,
    x_steps=X_STEPS,
    objective_max_step=OBJECTIVE_MAX_STEP,
    solver_max_step=MAX_STEP,
    cfl=DEFAULT_CFL,
    initial_seed=INITIAL_SEED,
    brownian_seed=BROWNIAN_SEED,
):
    """Evaluate the same stochastic BM1 objective with MC, FV and mixtures."""
    degrees = tuple(int(value) for value in degrees)
    if not degrees or any(value < 0 for value in degrees):
        raise ValueError("degrees must contain non-negative integers.")

    edges, midpoints, widths, solver_t_eval = midpoint_quadrature(
        T,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=objective_max_step,
    )

    model, parameters = build_reference_case()
    target = np.asarray(model.A(midpoints), dtype=float)

    # Monte Carlo reference.
    initial_states = sample_beta_initial_states(n_paths, initial_seed)
    integration_t, _ = model.euler_maruyama.build_time_grid(
        solver_t_eval,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=solver_max_step,
    )
    brownian_increments = model.euler_maruyama.brownian_increments(
        integration_t,
        n_paths=n_paths,
        seed=brownian_seed,
    )
    em_result = model.euler_maruyama_solver(
        initial_states,
        parameters,
        SIGMA_X,
        n_paths=n_paths,
        boundary="none",
        t_eval=solver_t_eval,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=solver_max_step,
        brownian_increments=brownian_increments,
        return_diagnostics=True,
    )
    em_t, _, _, _, _, _, _, em_demand, em_diagnostics = em_result
    np.testing.assert_array_equal(em_t, solver_t_eval)

    if em_diagnostics["total_violations"]:
        raise RuntimeError(
            "F2-H Monte Carlo produced raw boundary violations; "
            "the stochastic objective reference does not hide them."
        )

    midpoint_demand_paths = np.asarray(em_demand, dtype=float)[:, 1:-1]
    mc = monte_carlo_tracking_objective(
        widths,
        midpoint_demand_paths,
        target,
    )

    # Pure FV plus dynamic mixture on the same state grid and output times.
    density_result = run_dynamic_mixture_study(
        degrees=degrees,
        x_steps=x_steps,
        max_step=solver_max_step,
        t_eval=solver_t_eval,
        cfl=cfl,
    )
    np.testing.assert_array_equal(density_result["t"], solver_t_eval)

    x = density_result["x"]
    demand_values = _demand_grid(
        model,
        parameters,
        midpoints,
        x,
    )

    fv = _distribution_objective(
        widths,
        target,
        demand_values,
        density_result["fp_cell_mass"][1:-1],
    )

    mixture = {}
    for degree in degrees:
        mixture[degree] = _distribution_objective(
            widths,
            target,
            demand_values,
            density_result["dynamic"][degree]["cell_mass"][1:-1],
        )

    return {
        "edges": edges,
        "midpoints": midpoints,
        "widths": widths,
        "solver_t_eval": solver_t_eval,
        "target": target,
        "price": np.asarray(parameters[0](midpoints), dtype=float),
        "mc": mc,
        "fv": fv,
        "mixture": mixture,
        "degrees": degrees,
        "n_paths": int(n_paths),
        "x_steps": int(x_steps),
        "em_diagnostics": em_diagnostics,
        "density_result": density_result,
    }


def _relative_difference(left, right):
    scale = max(abs(float(right)), np.finfo(float).tiny)
    return abs(float(left) - float(right)) / scale


def main():
    result = run_objective_reference()

    print("F2-H stochastic BM1 tracking objective")
    print("-------------------------------------")
    print(f"Monte Carlo paths:       {result['n_paths']}")
    print(f"FV cells:                {result['x_steps']}")
    print(f"Objective intervals:     {len(result['widths'])}")
    print(
        f"Maximum objective step:  "
        f"{24.0 * 60.0 * np.max(result['widths']):.3f} min"
    )
    print(
        f"Price breakpoint on edge: "
        f"{bool(np.any(result['edges'] == PRICE_BREAKPOINTS[0]))}"
    )
    print(f"EM raw violations:       {result['em_diagnostics']['total_violations']}")
    print()

    mc = result["mc"]
    fv = result["fv"]

    print("Objective decomposition")
    print("-----------------------")
    print(
        f"MC:  J={mc.objective.total:.8e}, "
        f"bias={mc.objective.bias:.8e}, "
        f"variance={mc.objective.variance:.8e}, "
        f"SE(J)={mc.standard_error:.3e}"
    )
    print(
        f"FV:  J={fv.total:.8e}, "
        f"bias={fv.bias:.8e}, "
        f"variance={fv.variance:.8e}"
    )
    for degree in result["degrees"]:
        item = result["mixture"][degree]
        print(
            f"M={degree:2d}: J={item.total:.8e}, "
            f"bias={item.bias:.8e}, "
            f"variance={item.variance:.8e}"
        )

    print()
    print("Cross-method differences")
    print("------------------------")
    mc_fv_gap = abs(mc.objective.total - fv.total)
    mc_fv_se = (
        mc_fv_gap / mc.standard_error
        if mc.standard_error > 0.0
        else np.inf
    )
    print(
        f"|J_MC-J_FV| = {mc_fv_gap:.6e} "
        f"({mc_fv_se:.3f} MC standard errors, "
        f"relative {_relative_difference(mc.objective.total, fv.total):.3%})"
    )
    for degree in result["degrees"]:
        item = result["mixture"][degree]
        print(
            f"M={degree:2d}: |J_mix-J_FV|="
            f"{abs(item.total - fv.total):.6e}, "
            f"relative={_relative_difference(item.total, fv.total):.3%}"
        )

    print()
    print("Algebra diagnostics")
    print("-------------------")
    print(
        f"MC direct-vs-decomposed J error: "
        f"{abs(mc.direct_total - mc.objective.total):.3e}"
    )
    print(
        f"MC max pointwise decomposition error: "
        f"{mc.maximum_decomposition_error:.3e}"
    )

    if mc.maximum_decomposition_error > 1.0e-12:
        raise RuntimeError("MC bias-variance decomposition failed.")
    if abs(mc.direct_total - mc.objective.total) > 1.0e-12:
        raise RuntimeError("MC direct and decomposed stochastic objectives disagree.")

    hours = 24.0 * result["midpoints"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    ax_total, ax_parts, ax_cumulative = axes

    ax_total.step(
        hours,
        result["price"],
        where="mid",
        label="Price U(t)",
    )
    ax_total.plot(
        hours,
        mc.objective.total_integrand,
        label="MC E[(D-A)^2]",
    )
    ax_total.plot(
        hours,
        fv.total_integrand,
        linestyle="--",
        label="FV E[(D-A)^2]",
    )
    best_degree = max(result["degrees"])
    ax_total.plot(
        hours,
        result["mixture"][best_degree].total_integrand,
        linestyle=":",
        label=f"Dynamic mixture M={best_degree}",
    )
    ax_total.axvline(PRICE_DROP_HOUR, linestyle="--")
    ax_total.set_ylabel("Price / cost density")
    ax_total.set_title("Stochastic tracking-cost integrand")
    ax_total.grid(True)
    ax_total.legend(loc="best")

    ax_parts.plot(
        hours,
        fv.bias_integrand,
        label="FV bias-squared contribution",
    )
    ax_parts.plot(
        hours,
        fv.variance_integrand,
        label="FV demand-variance contribution",
    )
    ax_parts.axvline(PRICE_DROP_HOUR, linestyle="--")
    ax_parts.set_ylabel("Cost density")
    ax_parts.set_title("Why the stochastic objective differs from mean tracking")
    ax_parts.grid(True)
    ax_parts.legend(loc="best")

    ax_cumulative.plot(
        24.0 * result["edges"][1:],
        np.cumsum(result["widths"] * mc.objective.total_integrand),
        label="MC cumulative J",
    )
    ax_cumulative.plot(
        24.0 * result["edges"][1:],
        np.cumsum(result["widths"] * fv.total_integrand),
        linestyle="--",
        label="FV cumulative J",
    )
    ax_cumulative.plot(
        24.0 * result["edges"][1:],
        np.cumsum(
            result["widths"]
            * result["mixture"][best_degree].total_integrand
        ),
        linestyle=":",
        label=f"Mixture M={best_degree} cumulative J",
    )
    ax_cumulative.axvline(PRICE_DROP_HOUR, linestyle="--")
    ax_cumulative.set_xlabel("Time [h]")
    ax_cumulative.set_ylabel("Accumulated objective")
    ax_cumulative.set_title("Accumulation of the stochastic objective")
    ax_cumulative.grid(True)
    ax_cumulative.legend(loc="best")

    fig.tight_layout()
    graph_dir = (
        Path(__file__).resolve().parents[4]
        / "files"
        / "output"
        / "graphs"
        / "bm_1"
    )
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "bm_1_stochastic_objective_reference.png"
    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved graph to {graph_path}")


if __name__ == "__main__":
    main()
