"""F2-F dynamic Beta-mixture evolution for stochastic BM1.

The mixture itself is propagated in time.  Every internal step performs

    current mixture
        -> explicit Euler Fokker-Planck prediction
        -> simplex-constrained L2 projection back to the same Beta basis.

The pure finite-volume solution is the external reference.  F2-E's independent
static projection at each output time is also retained as the best-in-space
comparison, allowing representation error and additional dynamic error to be
distinguished.

No clipping or renormalization is applied to the raw Euler prediction.

Run from the repository root:

    python -B -m src.exe.run_model.bm_1.beta_dynamic_reference
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.diagnostics.bm_1 import (
    cell_mass_cdf_sup,
    cell_mass_l1,
    cell_mass_wasserstein_1,
    density_l2,
)
from src.exe.run_model.bm_1.beta_projection_reference import (
    _expected_demand,
    run_projection_study,
)
from src.exe.run_model.bm_1.f2_reference_case import (
    MAX_STEP,
    PRICE_BREAKPOINTS,
    PRICE_DROP_HOUR,
    SIGMA_X,
    T_EVAL,
    X_STEPS,
    build_reference_case,
)
# Compatibility imports retain the F2-F helper names used by existing tests.
from src.method.bm_1_beta_mixture import (
    _projection_matrices,
    _project_with_warm_start,
    _positive_explicit_step_limit,
    _validate_dynamic_controls,
    evolve_projected_density as _evolve_projected_density,
)


DEFAULT_DEGREES = (20, 30, 38)
DEFAULT_CFL = 0.40
_NEGATIVE_PREDICTION_TOL = 1.0e-10


def _distribution_errors(left_density, right_density, dx):
    left_mass = np.asarray(left_density, dtype=float) * dx
    right_mass = np.asarray(right_density, dtype=float) * dx
    n_times = left_density.shape[0]

    l2 = np.empty(n_times, dtype=float)
    l1 = np.empty(n_times, dtype=float)
    cdf_sup = np.empty(n_times, dtype=float)
    wasserstein = np.empty(n_times, dtype=float)
    for index in range(n_times):
        l2[index] = density_l2(
            left_density[index],
            right_density[index],
            dx,
        )
        l1[index] = cell_mass_l1(
            left_mass[index],
            right_mass[index],
        )
        cdf_sup[index] = cell_mass_cdf_sup(
            left_mass[index],
            right_mass[index],
        )
        wasserstein[index] = cell_mass_wasserstein_1(
            left_mass[index],
            right_mass[index],
            dx,
        )
    return {
        "l2": l2,
        "l1": l1,
        "cdf_sup": cdf_sup,
        "wasserstein_1": wasserstein,
    }


def run_dynamic_mixture_study(
    degrees=DEFAULT_DEGREES,
    *,
    x_steps=X_STEPS,
    max_step=MAX_STEP,
    t_eval=T_EVAL,
    cfl=DEFAULT_CFL,
):
    """Compare dynamic Beta mixtures with pure FV and static best projections."""
    degrees = tuple(int(value) for value in degrees)
    if not degrees or any(value < 0 for value in degrees):
        raise ValueError("degrees must contain non-negative integers.")

    t_eval = np.asarray(t_eval, dtype=float)
    static = run_projection_study(
        degrees=degrees,
        x_steps=x_steps,
        max_step=max_step,
        t_eval=t_eval,
    )

    model, parameters = build_reference_case()
    x = static["x"]
    dx = static["dx"]

    def drift(time, state):
        return model._drift(time, state, parameters)

    def diffusion(time, state):
        del time
        return model.diffusion_coefficient(state, SIGMA_X)

    dynamic = {}
    for degree in degrees:
        static_item = static["projection"][degree]
        basis = static_item["basis"]
        initial_coefficients = static_item["coefficients"][0]

        evolved = _evolve_projected_density(
            method=model.fokker_planc,
            t_eval=t_eval,
            breakpoints=PRICE_BREAKPOINTS,
            x=x,
            dx=dx,
            basis=basis,
            initial_coefficients=initial_coefficients,
            drift=drift,
            diffusion=diffusion,
            max_step=max_step,
            cfl=cfl,
        )

        density = evolved["density"]
        cell_mass = density * dx
        mean = np.sum(cell_mass * x[None, :], axis=1)
        variance = np.sum(
            cell_mass * np.square(x[None, :] - mean[:, None]),
            axis=1,
        )
        expected_demand = _expected_demand(
            model,
            parameters,
            static["t"],
            x,
            cell_mass,
        )

        total_error = _distribution_errors(
            density,
            static["fp_density"],
            dx,
        )
        dynamic_gap = _distribution_errors(
            density,
            static_item["density"],
            dx,
        )

        dynamic[degree] = {
            "density": density,
            "cell_mass": cell_mass,
            "coefficients": evolved["coefficients"],
            "mean": mean,
            "variance": variance,
            "expected_demand": expected_demand,
            "total_error": total_error,
            "dynamic_gap": dynamic_gap,
            "static_error": {
                "l2": static_item["l2"].copy(),
                "l1": static_item["l1"].copy(),
                "cdf_sup": static_item["cdf_sup"].copy(),
                "wasserstein_1": static_item["wasserstein_1"].copy(),
            },
            "diagnostics": evolved["diagnostics"],
        }

    result = dict(static)
    result["dynamic"] = dynamic
    result["cfl"] = float(cfl)
    return result


def main():
    result = run_dynamic_mixture_study()

    print("F2-F dynamic Beta-mixture evolution study")
    print("-----------------------------------------")
    print(f"Spatial cells: {len(result['x'])}")
    print(f"Snapshots:     {len(result['t'])}")
    print(f"CFL:           {result['cfl']:.3f}")
    print()
    print(
        "degree   cond(H)      steps   max total L2   max static L2   "
        "max dyn-gap L2   max |dE[X]|   max |dE[D]|"
    )
    for degree in result["degrees"]:
        item = result["dynamic"][degree]
        static_item = result["projection"][degree]
        diagnostics = item["diagnostics"]
        print(
            f"{degree:6d}  "
            f"{diagnostics['hessian_condition_number']:.3e}  "
            f"{diagnostics['n_steps']:6d}  "
            f"{np.max(item['total_error']['l2']):.3e}      "
            f"{np.max(static_item['l2']):.3e}       "
            f"{np.max(item['dynamic_gap']['l2']):.3e}       "
            f"{np.max(np.abs(item['mean'] - result['fp_mean'])):.3e}      "
            f"{np.max(np.abs(item['expected_demand'] - result['fp_expected_demand'])):.3e}"
        )

    print()
    print("Numerical diagnostics")
    print("---------------------")
    for degree in result["degrees"]:
        diagnostics = result["dynamic"][degree]["diagnostics"]
        print(
            f"M={degree:2d}: raw min={diagnostics['minimum_raw_prediction']:.3e}, "
            f"raw mass err={diagnostics['maximum_abs_raw_prediction_mass_error']:.3e}, "
            f"proj mass err={diagnostics['maximum_abs_projected_mass_error']:.3e}, "
            f"min c={diagnostics['minimum_coefficient']:.3e}, "
            f"max boundary c={diagnostics['maximum_boundary_coefficient']:.3e}, "
            f"max step={24.0 * 60.0 * diagnostics['maximum_internal_step']:.3f} min"
        )

        if diagnostics["used_clipping"] or diagnostics["used_renormalization"]:
            raise RuntimeError("F2-F must not use clipping or renormalization.")
        if diagnostics["minimum_raw_prediction"] < -_NEGATIVE_PREDICTION_TOL:
            raise RuntimeError("F2-F produced a materially negative raw prediction.")
        if diagnostics["maximum_abs_raw_prediction_mass_error"] > 1.0e-9:
            raise RuntimeError("F2-F raw Euler prediction failed probability conservation.")
        if diagnostics["maximum_abs_projected_mass_error"] > 1.0e-9:
            raise RuntimeError("F2-F projected mixture failed probability conservation.")
        if diagnostics["minimum_coefficient"] < -1.0e-9:
            raise RuntimeError("F2-F produced a materially negative mixture coefficient.")
        if diagnostics["maximum_abs_coefficient_sum_error"] > 1.0e-9:
            raise RuntimeError("F2-F mixture coefficients failed the simplex sum constraint.")

    hours = 24.0 * result["t"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 15))
    ax_total, ax_gap, ax_mean, ax_terminal = axes

    for degree in result["degrees"]:
        ax_total.plot(
            hours,
            result["dynamic"][degree]["total_error"]["l2"],
            label=f"Dynamic M={degree}",
        )
        ax_total.plot(
            hours,
            result["projection"][degree]["l2"],
            linestyle="--",
            label=f"Static best M={degree}",
        )
    ax_total.axvline(PRICE_DROP_HOUR, linestyle=":")
    ax_total.set_ylabel("Density L2 error")
    ax_total.set_title("Dynamic mixture error versus static representation limit")
    ax_total.grid(True)
    ax_total.legend(loc="best", ncol=2)

    for degree in result["degrees"]:
        ax_gap.plot(
            hours,
            result["dynamic"][degree]["dynamic_gap"]["l2"],
            label=f"M={degree}",
        )
    ax_gap.axvline(PRICE_DROP_HOUR, linestyle=":")
    ax_gap.set_ylabel("Density L2 distance")
    ax_gap.set_title("Additional dynamic gap to the best same-degree projection")
    ax_gap.grid(True)
    ax_gap.legend(loc="best")

    ax_mean.plot(
        hours,
        result["fp_mean"],
        linewidth=2.5,
        label="Pure FV E[X]",
    )
    for degree in result["degrees"]:
        ax_mean.plot(
            hours,
            result["dynamic"][degree]["mean"],
            label=f"Dynamic M={degree}",
        )
    ax_mean.axvline(PRICE_DROP_HOUR, linestyle=":")
    ax_mean.set_ylabel("Expected state")
    ax_mean.set_title("Expected state under dynamic mixture evolution")
    ax_mean.grid(True)
    ax_mean.legend(loc="best")

    ax_terminal.plot(
        result["x"],
        result["fp_density"][-1],
        linewidth=2.5,
        label="Pure FV",
    )
    for degree in result["degrees"]:
        ax_terminal.plot(
            result["x"],
            result["dynamic"][degree]["density"][-1],
            label=f"Dynamic M={degree}",
        )
    ax_terminal.set_xlabel("State of charge x")
    ax_terminal.set_ylabel("Probability density")
    ax_terminal.set_title("Terminal dynamic Beta mixtures")
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
    graph_path = graph_dir / "bm_1_beta_dynamic_reference.png"
    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved graph to {graph_path}")


if __name__ == "__main__":
    main()
