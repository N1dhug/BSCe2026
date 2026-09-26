"""F2-E static Beta-mixture projection study for stochastic BM1.

Each pure finite-volume density snapshot is projected independently onto the
existing degree-M Beta-mixture basis.  No mixture time stepping is performed,
so the reported error isolates representation/projection error.

Run from the repository root:

    python -B -m src.exe.run_model.bm_1.beta_projection_reference
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.diagnostics.bm_1 import (
    beta_mixture_variance_floor,
    cell_mass_cdf_sup,
    cell_mass_l1,
    cell_mass_wasserstein_1,
    density_l2,
)
from src.exe.run_model.bm_1.f2_reference_case import (
    MAX_STEP,
    PRICE_BREAKPOINTS,
    PRICE_DROP_HOUR,
    SIGMA_X,
    T_EVAL,
    X_STEPS,
    beta_initial_cell_mass,
    build_reference_case,
)


DEFAULT_DEGREES = (5, 10, 20, 30, 38)


def _expected_demand(model, parameters, t, x, cell_mass):
    values = np.empty(len(t), dtype=float)
    for index, time in enumerate(t):
        Ut = parameters[0](time)
        Bt = model.B(time)
        fXt = parameters[1](x)
        gUt = parameters[2](Ut)
        delta = model.l(fXt, gUt, parameters[4])
        demand = model.D(Bt, parameters[5], delta)
        values[index] = np.sum(np.asarray(demand, dtype=float) * cell_mass[index])
    return values


def run_projection_study(
    degrees=DEFAULT_DEGREES,
    *,
    x_steps=X_STEPS,
    max_step=MAX_STEP,
    t_eval=T_EVAL,
):
    """Project pure-FV snapshots onto independent degree-M Beta mixtures."""
    degrees = tuple(int(value) for value in degrees)
    if not degrees or any(value < 0 for value in degrees):
        raise ValueError("degrees must contain non-negative integers.")

    t_eval = np.asarray(t_eval, dtype=float)
    model, parameters = build_reference_case()
    edges, x, dx = model.fokker_planc_reference.grid(x_steps)
    initial_mass = beta_initial_cell_mass(edges)

    (
        t,
        fp_x,
        fp_density,
        fp_cell_mass,
        fp_mean,
        fp_variance,
        price,
        baseline,
        fp_expected_demand,
        fp_diagnostics,
    ) = model.fokker_planc_reference_solver(
        initial_mass,
        parameters,
        SIGMA_X,
        t_eval=t_eval,
        breakpoints=PRICE_BREAKPOINTS,
        max_step=max_step,
    )
    np.testing.assert_allclose(fp_x, x, atol=0.0, rtol=0.0)

    results = {}
    for degree in degrees:
        raw_basis = model.fokker_planc.beta_basis(x, degree)
        basis = model.fokker_planc._normalise_basis_discretely(raw_basis, dx)

        coefficients = np.empty((len(t), degree + 1), dtype=float)
        projected_density = np.empty_like(fp_density)

        for index in range(len(t)):
            coefficients[index], projected_density[index] = (
                model.fokker_planc.project_density(
                    fp_density[index],
                    basis,
                    dx,
                )
            )

        projected_mass = projected_density * dx
        projected_mean = np.sum(projected_mass * x[None, :], axis=1)
        projected_variance = np.sum(
            projected_mass
            * np.square(x[None, :] - projected_mean[:, None]),
            axis=1,
        )
        projected_expected_demand = _expected_demand(
            model,
            parameters,
            t,
            x,
            projected_mass,
        )

        l2 = np.empty(len(t), dtype=float)
        l1 = np.empty(len(t), dtype=float)
        cdf_sup = np.empty(len(t), dtype=float)
        wasserstein = np.empty(len(t), dtype=float)
        for index in range(len(t)):
            l2[index] = density_l2(
                projected_density[index],
                fp_density[index],
                dx,
            )
            l1[index] = cell_mass_l1(
                projected_mass[index],
                fp_cell_mass[index],
            )
            cdf_sup[index] = cell_mass_cdf_sup(
                projected_mass[index],
                fp_cell_mass[index],
            )
            wasserstein[index] = cell_mass_wasserstein_1(
                projected_mass[index],
                fp_cell_mass[index],
                dx,
            )

        variance_floor = beta_mixture_variance_floor(fp_mean, degree)
        results[degree] = {
            "basis": basis,
            "coefficients": coefficients,
            "density": projected_density,
            "cell_mass": projected_mass,
            "mean": projected_mean,
            "variance": projected_variance,
            "expected_demand": projected_expected_demand,
            "l2": l2,
            "l1": l1,
            "cdf_sup": cdf_sup,
            "wasserstein_1": wasserstein,
            "variance_floor": variance_floor,
            "variance_floor_gap": np.maximum(
                variance_floor - fp_variance,
                0.0,
            ),
            "maximum_coefficient_sum_error": float(
                np.max(np.abs(np.sum(coefficients, axis=1) - 1.0))
            ),
            "minimum_coefficient": float(np.min(coefficients)),
            "maximum_mass_error": float(
                np.max(np.abs(np.sum(projected_mass, axis=1) - 1.0))
            ),
        }

    return {
        "t": t,
        "x": x,
        "edges": edges,
        "dx": dx,
        "price": price,
        "baseline": baseline,
        "fp_density": fp_density,
        "fp_cell_mass": fp_cell_mass,
        "fp_mean": fp_mean,
        "fp_variance": fp_variance,
        "fp_expected_demand": fp_expected_demand,
        "fp_diagnostics": fp_diagnostics,
        "degrees": degrees,
        "projection": results,
    }


def main():
    result = run_projection_study()

    print("F2-E static Beta-mixture projection study")
    print("-----------------------------------------")
    print(f"Spatial cells: {len(result['x'])}")
    print(f"Snapshots:     {len(result['t'])}")
    print()
    print(
        "degree    max L2       max L1       max W1       "
        "max |dE[X]|   max |dVar|    max |dE[D]|"
    )
    for degree in result["degrees"]:
        item = result["projection"][degree]
        print(
            f"{degree:6d}  "
            f"{np.max(item['l2']):.3e}  "
            f"{np.max(item['l1']):.3e}  "
            f"{np.max(item['wasserstein_1']):.3e}  "
            f"{np.max(np.abs(item['mean'] - result['fp_mean'])):.3e}  "
            f"{np.max(np.abs(item['variance'] - result['fp_variance'])):.3e}  "
            f"{np.max(np.abs(item['expected_demand'] - result['fp_expected_demand'])):.3e}"
        )

    print()
    print("Initial-law variance representation")
    print("-----------------------------------")
    print(f"FV initial variance: {result['fp_variance'][0]:.8e}")
    for degree in result["degrees"]:
        item = result["projection"][degree]
        print(
            f"M={degree:2d}: floor={item['variance_floor'][0]:.8e}, "
            f"projected={item['variance'][0]:.8e}, "
            f"L1={item['l1'][0]:.8e}"
        )

    for degree in result["degrees"]:
        item = result["projection"][degree]
        if item["maximum_coefficient_sum_error"] > 1.0e-9:
            raise RuntimeError("Beta-mixture coefficients failed unit-sum constraint.")
        if item["minimum_coefficient"] < -1.0e-9:
            raise RuntimeError("Beta-mixture projection produced a materially negative coefficient.")
        if item["maximum_mass_error"] > 1.0e-9:
            raise RuntimeError("Projected Beta-mixture density lost probability mass.")

    hours = 24.0 * result["t"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 15))
    ax_degree, ax_time, ax_variance, ax_terminal = axes

    degrees = np.asarray(result["degrees"], dtype=float)
    max_l2 = np.array(
        [np.max(result["projection"][degree]["l2"]) for degree in result["degrees"]]
    )
    terminal_l2 = np.array(
        [result["projection"][degree]["l2"][-1] for degree in result["degrees"]]
    )
    ax_degree.plot(degrees, max_l2, marker="o", label="Maximum density L2")
    ax_degree.plot(degrees, terminal_l2, marker="o", label="Terminal density L2")
    ax_degree.set_xlabel("Beta-mixture degree M")
    ax_degree.set_ylabel("Density L2 error")
    ax_degree.set_title("Pure representation error versus Beta-mixture degree")
    ax_degree.grid(True)
    ax_degree.legend(loc="best")

    for degree in result["degrees"]:
        ax_time.plot(
            hours,
            result["projection"][degree]["l1"],
            label=f"M={degree}",
        )
    ax_time.axvline(PRICE_DROP_HOUR, linestyle="--")
    ax_time.set_xlabel("Time [h]")
    ax_time.set_ylabel("Cell-mass L1 error")
    ax_time.set_title("Static projection error over the BM1 reference trajectory")
    ax_time.grid(True)
    ax_time.legend(loc="best")

    ax_variance.plot(
        degrees,
        [result["projection"][degree]["variance_floor"][0] for degree in result["degrees"]],
        marker="o",
        label="Theoretical variance floor at mean 0.5",
    )
    ax_variance.axhline(
        result["fp_variance"][0],
        linestyle="--",
        label="FV initial variance",
    )
    ax_variance.set_xlabel("Beta-mixture degree M")
    ax_variance.set_ylabel("Variance")
    ax_variance.set_title("Why low-degree mixtures cannot represent the narrow initial law")
    ax_variance.grid(True)
    ax_variance.legend(loc="best")

    ax_terminal.plot(
        result["x"],
        result["fp_density"][-1],
        linewidth=2.5,
        label="Pure FV",
    )
    for degree in result["degrees"]:
        if degree in (10, 20, 38):
            ax_terminal.plot(
                result["x"],
                result["projection"][degree]["density"][-1],
                label=f"Projected M={degree}",
            )
    ax_terminal.set_xlabel("State of charge x")
    ax_terminal.set_ylabel("Probability density")
    ax_terminal.set_title("Terminal pure-FV density and static Beta projections")
    ax_terminal.grid(True)
    ax_terminal.legend(loc="best")

    fig.tight_layout()
    graph_dir = (
        Path(__file__).resolve().parents[4]
        / "files"
        / "output"
        / "graphs"
        / "bm_1"
    )
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "bm_1_beta_projection_reference.png"
    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved graph to {graph_path}")


if __name__ == "__main__":
    main()
