"""F2-G conditioning and sensitivity study for BM1 Beta mixtures.

The full Beta Gram/Hessian matrix becomes extremely ill-conditioned at high
degree.  This experiment checks whether that algebraic conditioning materially
affects the quantities that matter:

1. full basis-matrix/Hessian condition numbers;
2. local Hessian conditioning on the active simplex tangent space;
3. dependence of the QP solution on a very different feasible warm start;
4. sensitivity of coefficients and projected density to tiny positive,
   mass-preserving perturbations of the target density.

The dynamic F2-F study is reused without modifying shared WM/BM infrastructure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space

from src.diagnostics.bm_1 import density_l2
from src.exe.run_model.bm_1.beta_dynamic_reference import (
    DEFAULT_CFL,
    DEFAULT_DEGREES,
    _project_with_warm_start,
    _projection_matrices,
    run_dynamic_mixture_study,
)
from src.exe.run_model.bm_1.f2_reference_case import (
    MAX_STEP,
    T_EVAL,
    X_STEPS,
)


DEFAULT_EPSILONS = (1.0e-10, 1.0e-8, 1.0e-6)
DEFAULT_DIRECTIONS = 4
DEFAULT_SEED = 2030
DEFAULT_ACTIVE_TOL = 1.0e-8


def _simplex_tangent_condition(hessian, coefficients, active_tol=DEFAULT_ACTIVE_TOL):
    """Condition number of local QP curvature along feasible free directions.

    Coefficients larger than active_tol define the local free set.  The simplex
    equality removes the all-ones direction.  The returned condition number is
    therefore computed for Z.T @ H_FF @ Z, where columns of Z span
    {d : 1.T d = 0} on the free set.
    """
    hessian = np.asarray(hessian, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    active_tol = float(active_tol)

    if (
        hessian.ndim != 2
        or hessian.shape[0] != hessian.shape[1]
        or hessian.shape[0] != coefficients.size
    ):
        raise ValueError("hessian and coefficients have incompatible shapes.")
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(coefficients)):
        raise ValueError("hessian and coefficients must be finite.")
    if not np.isfinite(active_tol) or active_tol < 0.0:
        raise ValueError("active_tol must be finite and non-negative.")

    free = np.flatnonzero(coefficients > active_tol)
    if free.size <= 1:
        return 1.0, int(free.size), 0

    reduced_hessian = hessian[np.ix_(free, free)]
    tangent = null_space(np.ones((1, free.size), dtype=float))
    if tangent.shape[1] == 0:
        return 1.0, int(free.size), 0

    curvature = tangent.T @ reduced_hessian @ tangent
    condition = float(np.linalg.cond(curvature))
    return condition, int(free.size), int(tangent.shape[1])


def _positive_mass_preserving_perturbation(density, dx, epsilon, direction):
    """Apply a tiny multiplicative log-density perturbation.

    p_eps is proportional to p * exp(epsilon*r), where r is centered with
    respect to p and scaled to max |r| = 1.  The construction preserves
    non-negativity and then enforces unit mass exactly up to roundoff.
    """
    density = np.asarray(density, dtype=float)
    direction = np.asarray(direction, dtype=float)
    dx = float(dx)
    epsilon = float(epsilon)

    if density.ndim != 1 or direction.shape != density.shape:
        raise ValueError("density and direction must be one-dimensional with equal shape.")
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("density must be finite and non-negative.")
    if not np.all(np.isfinite(direction)):
        raise ValueError("direction must be finite.")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be finite and strictly positive.")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be finite and non-negative.")

    mass = float(np.sum(density) * dx)
    if not np.isclose(mass, 1.0, atol=1.0e-9, rtol=0.0):
        raise ValueError("density must have unit discrete mass.")

    centered = direction - float(np.sum(density * direction) * dx)
    scale = float(np.max(np.abs(centered)))
    if scale == 0.0:
        raise ValueError("direction must contain non-constant variation.")
    centered /= scale

    perturbed = density * np.exp(epsilon * centered)
    perturbed_mass = float(np.sum(perturbed) * dx)
    if not np.isfinite(perturbed_mass) or perturbed_mass <= 0.0:
        raise RuntimeError("perturbed density lost positive finite mass.")
    return perturbed / perturbed_mass


def _density_moments(density, x, dx):
    mass = np.asarray(density, dtype=float) * dx
    mean = float(np.sum(mass * x))
    variance = float(np.sum(mass * np.square(x - mean)))
    return mean, variance


def run_conditioning_study(
    degrees=(20, 30, 38),
    *,
    epsilons=DEFAULT_EPSILONS,
    n_directions=DEFAULT_DIRECTIONS,
    seed=DEFAULT_SEED,
    active_tol=DEFAULT_ACTIVE_TOL,
    x_steps=X_STEPS,
    max_step=MAX_STEP,
    t_eval=T_EVAL,
    cfl=DEFAULT_CFL,
):
    """Return conditioning and projection-sensitivity diagnostics."""
    degrees = tuple(int(value) for value in degrees)
    epsilons = tuple(float(value) for value in epsilons)
    n_directions = int(n_directions)
    if not degrees or any(value < 0 for value in degrees):
        raise ValueError("degrees must contain non-negative integers.")
    if not epsilons or any(not np.isfinite(value) or value < 0.0 for value in epsilons):
        raise ValueError("epsilons must contain finite non-negative values.")
    if n_directions < 1:
        raise ValueError("n_directions must be at least one.")

    base = run_dynamic_mixture_study(
        degrees=degrees,
        x_steps=x_steps,
        max_step=max_step,
        t_eval=np.asarray(t_eval, dtype=float),
        cfl=cfl,
    )
    x = base["x"]
    dx = base["dx"]
    rng = np.random.default_rng(seed)

    conditioning = {}
    for degree in degrees:
        static_item = base["projection"][degree]
        dynamic_item = base["dynamic"][degree]
        basis = static_item["basis"]
        matrix, hessian, hessian_condition = _projection_matrices(basis, dx)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        matrix_condition = float(singular_values[0] / singular_values[-1])

        static_local_conditions = []
        static_free_sizes = []
        dynamic_local_conditions = []
        dynamic_free_sizes = []
        for static_c, dynamic_c in zip(
            static_item["coefficients"],
            dynamic_item["coefficients"],
        ):
            value, free_size, _ = _simplex_tangent_condition(
                hessian,
                static_c,
                active_tol,
            )
            static_local_conditions.append(value)
            static_free_sizes.append(free_size)

            value, free_size, _ = _simplex_tangent_condition(
                hessian,
                dynamic_c,
                active_tol,
            )
            dynamic_local_conditions.append(value)
            dynamic_free_sizes.append(free_size)

        uniform = np.full(degree + 1, 1.0 / (degree + 1.0))
        alternative_start = {
            "failures": 0,
            "maximum_coefficient_l1": 0.0,
            "maximum_density_l2": 0.0,
            "maximum_mean_difference": 0.0,
            "maximum_variance_difference": 0.0,
        }
        for index, target in enumerate(base["fp_density"]):
            try:
                coefficients, density = _project_with_warm_start(
                    target,
                    basis,
                    matrix,
                    hessian,
                    dx,
                    uniform,
                )
            except RuntimeError:
                alternative_start["failures"] += 1
                continue

            baseline_coefficients = static_item["coefficients"][index]
            baseline_density = static_item["density"][index]
            alternative_start["maximum_coefficient_l1"] = max(
                alternative_start["maximum_coefficient_l1"],
                float(np.sum(np.abs(coefficients - baseline_coefficients))),
            )
            alternative_start["maximum_density_l2"] = max(
                alternative_start["maximum_density_l2"],
                density_l2(density, baseline_density, dx),
            )
            mean, variance = _density_moments(density, x, dx)
            baseline_mean, baseline_variance = _density_moments(
                baseline_density,
                x,
                dx,
            )
            alternative_start["maximum_mean_difference"] = max(
                alternative_start["maximum_mean_difference"],
                abs(mean - baseline_mean),
            )
            alternative_start["maximum_variance_difference"] = max(
                alternative_start["maximum_variance_difference"],
                abs(variance - baseline_variance),
            )

        perturbations = {}
        for epsilon in epsilons:
            summary = {
                "failures": 0,
                "maximum_input_l2": 0.0,
                "maximum_coefficient_l1": 0.0,
                "maximum_projected_density_l2": 0.0,
                "maximum_density_amplification": 0.0,
                "maximum_coefficient_amplification": 0.0,
                "maximum_mean_difference": 0.0,
                "maximum_variance_difference": 0.0,
            }
            for index, target in enumerate(base["fp_density"]):
                baseline_coefficients = static_item["coefficients"][index]
                baseline_density = static_item["density"][index]
                baseline_mean, baseline_variance = _density_moments(
                    baseline_density,
                    x,
                    dx,
                )

                for _ in range(n_directions):
                    direction = rng.standard_normal(target.size)
                    perturbed_target = _positive_mass_preserving_perturbation(
                        target,
                        dx,
                        epsilon,
                        direction,
                    )
                    input_l2 = density_l2(
                        perturbed_target,
                        target,
                        dx,
                    )

                    try:
                        coefficients, density = _project_with_warm_start(
                            perturbed_target,
                            basis,
                            matrix,
                            hessian,
                            dx,
                            baseline_coefficients,
                        )
                    except RuntimeError:
                        summary["failures"] += 1
                        continue

                    coefficient_l1 = float(
                        np.sum(np.abs(coefficients - baseline_coefficients))
                    )
                    projected_l2 = density_l2(
                        density,
                        baseline_density,
                        dx,
                    )
                    scale = max(input_l2, np.finfo(float).tiny)
                    summary["maximum_input_l2"] = max(
                        summary["maximum_input_l2"],
                        input_l2,
                    )
                    summary["maximum_coefficient_l1"] = max(
                        summary["maximum_coefficient_l1"],
                        coefficient_l1,
                    )
                    summary["maximum_projected_density_l2"] = max(
                        summary["maximum_projected_density_l2"],
                        projected_l2,
                    )
                    summary["maximum_density_amplification"] = max(
                        summary["maximum_density_amplification"],
                        projected_l2 / scale,
                    )
                    summary["maximum_coefficient_amplification"] = max(
                        summary["maximum_coefficient_amplification"],
                        coefficient_l1 / scale,
                    )

                    mean, variance = _density_moments(density, x, dx)
                    summary["maximum_mean_difference"] = max(
                        summary["maximum_mean_difference"],
                        abs(mean - baseline_mean),
                    )
                    summary["maximum_variance_difference"] = max(
                        summary["maximum_variance_difference"],
                        abs(variance - baseline_variance),
                    )

            perturbations[epsilon] = summary

        conditioning[degree] = {
            "matrix_condition_number": matrix_condition,
            "hessian_condition_number": hessian_condition,
            "minimum_singular_value": float(singular_values[-1]),
            "maximum_singular_value": float(singular_values[0]),
            "static_local_condition_numbers": np.asarray(
                static_local_conditions,
                dtype=float,
            ),
            "static_free_sizes": np.asarray(static_free_sizes, dtype=int),
            "dynamic_local_condition_numbers": np.asarray(
                dynamic_local_conditions,
                dtype=float,
            ),
            "dynamic_free_sizes": np.asarray(dynamic_free_sizes, dtype=int),
            "alternative_start": alternative_start,
            "perturbations": perturbations,
        }

    result = dict(base)
    result["conditioning"] = conditioning
    result["epsilons"] = epsilons
    result["n_directions"] = n_directions
    result["active_tol"] = float(active_tol)
    return result


def main():
    result = run_conditioning_study()

    print("F2-G Beta-mixture conditioning and sensitivity")
    print("----------------------------------------------")
    print(f"Spatial cells:       {len(result['x'])}")
    print(f"Snapshots:           {len(result['t'])}")
    print(f"Directions/epsilon:  {result['n_directions']}")
    print(f"Active threshold:    {result['active_tol']:.1e}")
    print()
    print(
        "degree   cond(A)      cond(H)      max local static   "
        "max local dynamic   free static   free dynamic"
    )
    for degree in result["degrees"]:
        item = result["conditioning"][degree]
        print(
            f"{degree:6d}  "
            f"{item['matrix_condition_number']:.3e}  "
            f"{item['hessian_condition_number']:.3e}  "
            f"{np.max(item['static_local_condition_numbers']):.3e}        "
            f"{np.max(item['dynamic_local_condition_numbers']):.3e}         "
            f"{np.max(item['static_free_sizes']):6d}        "
            f"{np.max(item['dynamic_free_sizes']):6d}"
        )

    print()
    print("Dependence on feasible QP start")
    print("-------------------------------")
    for degree in result["degrees"]:
        item = result["conditioning"][degree]["alternative_start"]
        print(
            f"M={degree:2d}: failures={item['failures']}, "
            f"max ||dc||_1={item['maximum_coefficient_l1']:.3e}, "
            f"max density L2={item['maximum_density_l2']:.3e}, "
            f"max |dmean|={item['maximum_mean_difference']:.3e}, "
            f"max |dvar|={item['maximum_variance_difference']:.3e}"
        )

    print()
    print("Positive mass-preserving target perturbations")
    print("---------------------------------------------")
    for degree in result["degrees"]:
        print(f"M={degree}:")
        for epsilon in result["epsilons"]:
            item = result["conditioning"][degree]["perturbations"][epsilon]
            print(
                f"  eps={epsilon:.0e}: failures={item['failures']}, "
                f"input L2={item['maximum_input_l2']:.3e}, "
                f"||dc||_1={item['maximum_coefficient_l1']:.3e}, "
                f"density L2={item['maximum_projected_density_l2']:.3e}, "
                f"density amp={item['maximum_density_amplification']:.3e}, "
                f"coeff amp={item['maximum_coefficient_amplification']:.3e}, "
                f"|dmean|={item['maximum_mean_difference']:.3e}, "
                f"|dvar|={item['maximum_variance_difference']:.3e}"
            )

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    ax_condition, ax_coeff, ax_density = axes

    degrees = np.asarray(result["degrees"], dtype=float)
    ax_condition.semilogy(
        degrees,
        [
            result["conditioning"][degree]["hessian_condition_number"]
            for degree in result["degrees"]
        ],
        marker="o",
        label="Full cond(H)",
    )
    ax_condition.semilogy(
        degrees,
        [
            np.max(
                result["conditioning"][degree][
                    "dynamic_local_condition_numbers"
                ]
            )
            for degree in result["degrees"]
        ],
        marker="o",
        label="Max active-tangent cond",
    )
    ax_condition.set_xlabel("Beta-mixture degree M")
    ax_condition.set_ylabel("Condition number")
    ax_condition.set_title("Full versus locally relevant conditioning")
    ax_condition.grid(True)
    ax_condition.legend(loc="best")

    for degree in result["degrees"]:
        ax_coeff.loglog(
            result["epsilons"],
            [
                result["conditioning"][degree]["perturbations"][epsilon][
                    "maximum_coefficient_l1"
                ]
                for epsilon in result["epsilons"]
            ],
            marker="o",
            label=f"M={degree}",
        )
    ax_coeff.set_xlabel("Log-density perturbation epsilon")
    ax_coeff.set_ylabel("Maximum coefficient L1 change")
    ax_coeff.set_title("Coefficient sensitivity")
    ax_coeff.grid(True)
    ax_coeff.legend(loc="best")

    for degree in result["degrees"]:
        ax_density.loglog(
            result["epsilons"],
            [
                result["conditioning"][degree]["perturbations"][epsilon][
                    "maximum_projected_density_l2"
                ]
                for epsilon in result["epsilons"]
            ],
            marker="o",
            label=f"M={degree}",
        )
    ax_density.set_xlabel("Log-density perturbation epsilon")
    ax_density.set_ylabel("Maximum projected density L2 change")
    ax_density.set_title("Density sensitivity")
    ax_density.grid(True)
    ax_density.legend(loc="best")

    fig.tight_layout()
    graph_dir = (
        Path(__file__).resolve().parents[4]
        / "files"
        / "output"
        / "graphs"
        / "bm_1"
    )
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "bm_1_beta_conditioning_reference.png"
    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved graph to {graph_path}")


if __name__ == "__main__":
    main()
