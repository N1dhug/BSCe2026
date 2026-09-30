"""F4-D mixture representation studies for BM1 state estimation.

This validation module chooses and tests approximation dictionaries.  It does
not implement the recursive mixture filter and it does not alter the F2/F3
mixture defaults.
"""

from __future__ import annotations

import numpy as np

from src.diagnostics.bm_1 import (
    cell_mass_cdf_sup,
    cell_mass_l1,
    cell_mass_wasserstein_1,
    density_l2,
)
from src.method.beta_mixture import PreparedBetaProjector, beta_cell_average_basis
from src.solver.finite_volume import finite_volume_1d
from src.validation import estimation as checks
from src.validation.bm_1.estimation import build_f4_b_case


D1_X_STEPS = 801
D1_GAP_TOL = 1.0e-9
D1_MASS_TOL = 1.0e-9
D1_L1_GATE = 0.05
D1_W1_GATE = 0.01
D1_CONDITION_ALARM = 1.0e12
D1_ACTIVE_TOL = 1.0e-10

_INNER_MEANS = np.arange(1, 20, dtype=float) / 20.0
_BOUNDARY_MEANS = np.array([0.01, 0.025, 0.975, 0.99], dtype=float)
_CANDIDATE_LEVELS = (
    ("multiscale_2", (20.0, 100.0), False),
    ("multiscale_3", (20.0, 100.0, 500.0), False),
    ("multiscale_4", (20.0, 100.0, 500.0, 2500.0), False),
    ("multiscale_4_boundary", (20.0, 100.0, 500.0, 2500.0), True),
)


def _moments(cell_mass, centers):
    """Discrete center moments under D1's declared 1e-9 mass tolerance."""
    mass = checks.finite_vector(cell_mass, "cell_mass")
    centers = checks.finite_vector(centers, "centers")
    if mass.shape != centers.shape:
        raise ValueError("cell_mass and centers must have matching shapes.")
    if np.any(mass < 0.0) or abs(float(np.sum(mass)) - 1.0) > D1_MASS_TOL:
        raise ValueError(
            "cell_mass must be non-negative and sum to one within the declared D1 "
            "mass tolerance; no repair is applied."
        )
    mean = float(mass @ centers)
    variance = float(mass @ np.square(centers - mean))
    return mean, variance


def _target_suite(edges):
    """Return the preregistered six D1 density targets on one cell grid."""
    component_specs = {
        "broad_center": ((4.0,), (4.0,), (1.0,)),
        "narrow_center": ((900.0,), (900.0,), (1.0,)),
        "shifted_narrow": ((296.0,), (504.0,), (1.0,)),
        "bimodal": ((33.6, 86.4), (86.4, 33.6), (0.5, 0.5)),
        "left_boundary": ((1.5,), (73.5,), (1.0,)),
        "right_boundary": ((73.5,), (1.5,), (1.0,)),
    }
    widths = np.diff(edges)
    dx = float(widths[0])
    centers = 0.5 * (edges[:-1] + edges[1:])
    result = {}
    for name, (alpha, beta, weights) in component_specs.items():
        basis = beta_cell_average_basis(edges, alpha, beta)
        weights = np.asarray(weights, dtype=float)
        density = weights @ basis
        cell_mass = density * dx
        cell_mass = checks.probability_mass(cell_mass, name + " target mass")
        mean, variance = _moments(cell_mass, centers)
        density = cell_mass / dx
        density.setflags(write=False)
        cell_mass.setflags(write=False)
        result[name] = {
            "density": density,
            "cell_mass": cell_mass,
            "mean": mean,
            "variance": variance,
        }
    return result


def _multiscale_dictionary(edges, concentrations, *, boundary_enriched):
    means = _INNER_MEANS.copy()
    if boundary_enriched:
        means = np.sort(np.concatenate((means, _BOUNDARY_MEANS)))
    concentrations = np.asarray(concentrations, dtype=float)
    mean_grid, concentration_grid = np.meshgrid(means, concentrations, indexing="ij")
    flat_mean = mean_grid.reshape(-1)
    flat_concentration = concentration_grid.reshape(-1)
    alpha = flat_concentration * flat_mean
    beta = flat_concentration * (1.0 - flat_mean)
    basis = beta_cell_average_basis(edges, alpha, beta)
    return {
        "basis": basis,
        "means": flat_mean,
        "concentrations": flat_concentration,
        "alpha": alpha,
        "beta": beta,
        "components": int(alpha.size),
        "selectable": True,
    }


def _legacy_dictionary(model, centers, dx, degree):
    raw = model.fokker_planc.beta_basis(centers, degree)
    basis = model.fokker_planc._normalise_basis_discretely(raw, dx)
    return {
        "basis": basis,
        "means": (np.arange(degree + 1, dtype=float) + 1.0) / (degree + 2.0),
        "concentrations": np.full(degree + 1, degree + 2.0),
        "alpha": None,
        "beta": None,
        "components": degree + 1,
        "selectable": False,
    }


def build_d1_dictionaries(edges):
    """Build comparison and nested multiscale dictionaries fixed before D1 results."""
    edges = checks.increasing_grid(edges, "edges")
    widths = np.diff(edges)
    dx = float(widths[0])
    if edges[0] != 0.0 or edges[-1] != 1.0 or not np.allclose(widths, dx, atol=1e-14, rtol=1e-10):
        raise ValueError("D1 dictionaries require a uniform grid spanning exactly [0, 1].")
    centers = 0.5 * (edges[:-1] + edges[1:])
    model, _ = build_f4_b_case().build_model()

    dictionaries = {
        "legacy_degree_30": _legacy_dictionary(model, centers, dx, 30),
        "legacy_degree_38": _legacy_dictionary(model, centers, dx, 38),
    }
    for name, concentrations, boundary_enriched in _CANDIDATE_LEVELS:
        dictionaries[name] = _multiscale_dictionary(
            edges,
            concentrations,
            boundary_enriched=boundary_enriched,
        )
    return dictionaries



def _adaptive_dictionary(edges, high_step):
    """Scale-dependent D1b dictionary fixed after D1 and before D1b results."""
    high_step = checks.finite_scalar(high_step, "high_step", positive=True)
    low_means = np.arange(1, 10, dtype=float) / 10.0
    medium_means = np.arange(1, 40, dtype=float) / 40.0
    medium_means = np.sort(np.concatenate((medium_means, np.array([0.02, 0.98]))))
    count = int(round(0.8 / high_step)) + 1
    high_means = np.linspace(0.1, 0.9, count)
    if not np.isclose(high_means[1] - high_means[0], high_step, atol=1e-14, rtol=0.0):
        raise ValueError("high_step must divide [0.1, 0.9] exactly.")

    means = np.concatenate((
        low_means,
        medium_means,
        high_means,
        high_means,
    ))
    concentrations = np.concatenate((
        np.full(low_means.size, 20.0),
        np.full(medium_means.size, 100.0),
        np.full(high_means.size, 500.0),
        np.full(high_means.size, 2000.0),
    ))
    alpha = concentrations * means
    beta = concentrations * (1.0 - means)
    basis = beta_cell_average_basis(edges, alpha, beta)
    return {
        "basis": basis,
        "means": means,
        "concentrations": concentrations,
        "alpha": alpha,
        "beta": beta,
        "components": int(alpha.size),
        "selectable": True,
    }


def build_d1b_dictionaries(edges):
    """Build the two nested adaptive D1b dictionaries preregistered after D1."""
    edges = checks.increasing_grid(edges, "edges")
    widths = np.diff(edges)
    dx = float(widths[0])
    if edges[0] != 0.0 or edges[-1] != 1.0 or not np.allclose(widths, dx, atol=1e-14, rtol=1e-10):
        raise ValueError("D1b dictionaries require a uniform grid spanning exactly [0, 1].")
    return {
        "adaptive_A": _adaptive_dictionary(edges, 0.025),
        "adaptive_B": _adaptive_dictionary(edges, 0.0125),
    }


def _fit_target(projector, target, centers):
    projected = projector.project(target["density"])
    density = projected["density"]
    mass = density * projector.dx
    mean, variance = _moments(mass, centers)
    diagnostics = projected["diagnostics"]
    coefficients = projected["coefficients"]
    return {
        "status": "ok",
        "density_l2": density_l2(density, target["density"], projector.dx),
        "cell_mass_l1": cell_mass_l1(mass, target["cell_mass"]),
        "cdf_sup": cell_mass_cdf_sup(mass, target["cell_mass"]),
        "wasserstein_1": cell_mass_wasserstein_1(mass, target["cell_mass"], projector.dx),
        "mean_error": mean - target["mean"],
        "variance_error": variance - target["variance"],
        "optimality_gap": float(diagnostics["optimality_gap"]),
        "mass_error": float(diagnostics["mass_error"]),
        "used_fallback": bool(diagnostics["used_fallback"]),
        "active_components": int(np.sum(coefficients > D1_ACTIVE_TOL)),
        "maximum_coefficient": float(np.max(coefficients)),
    }



def _evaluate_static_dictionaries(
    *,
    edges,
    centers,
    dx,
    targets,
    dictionaries,
    fallback_disqualifies,
):
    """Evaluate fixed dictionaries without changing the declared projection problem."""
    results = {}
    selected = None
    for name, dictionary in dictionaries.items():
        projector = PreparedBetaProjector(dictionary["basis"], dx)
        fits = {}
        for target_name, target in targets.items():
            try:
                fits[target_name] = _fit_target(projector, target, centers)
            except (FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
                fits[target_name] = {
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                }
        failed_targets = tuple(
            target_name for target_name, item in fits.items()
            if item["status"] != "ok"
        )
        successful = [item for item in fits.values() if item["status"] == "ok"]
        if successful:
            max_l1 = max(item["cell_mass_l1"] for item in successful)
            max_w1 = max(item["wasserstein_1"] for item in successful)
            max_gap = max(item["optimality_gap"] for item in successful)
            max_mass_error = max(item["mass_error"] for item in successful)
            any_fallback = any(item["used_fallback"] for item in successful)
        else:
            max_l1 = max_w1 = max_gap = max_mass_error = float("inf")
            any_fallback = False
        if failed_targets:
            max_l1 = max_w1 = max_gap = max_mass_error = float("inf")
        matrix_condition = float(projector.projection_solver.matrix_condition_number)
        hessian_condition = float(projector.projection_data[2])
        numerical_ok = (
            not failed_targets
            and max_gap <= D1_GAP_TOL
            and max_mass_error <= D1_MASS_TOL
            and (not fallback_disqualifies or not any_fallback)
        )
        representation_ok = (
            not failed_targets
            and max_l1 <= D1_L1_GATE
            and max_w1 <= D1_W1_GATE
        )
        eligible = bool(dictionary["selectable"] and numerical_ok and representation_ok)
        summary = {
            "components": dictionary["components"],
            "selectable": dictionary["selectable"],
            "numerical_ok": numerical_ok,
            "representation_ok": representation_ok,
            "eligible": eligible,
            "maximum_cell_mass_l1": max_l1,
            "maximum_wasserstein_1": max_w1,
            "maximum_optimality_gap": max_gap,
            "maximum_mass_error": max_mass_error,
            "any_fallback": any_fallback,
            "failed_targets": failed_targets,
            "projection_matrix_condition_number": matrix_condition,
            "hessian_condition_number": hessian_condition,
            "conditioning_alarm": (
                not np.isfinite(matrix_condition)
                or matrix_condition > D1_CONDITION_ALARM
            ),
            "worst_l1_target": (
                None if failed_targets else max(fits, key=lambda key: fits[key]["cell_mass_l1"])
            ),
            "worst_w1_target": (
                None if failed_targets else max(fits, key=lambda key: fits[key]["wasserstein_1"])
            ),
        }
        results[name] = {
            "summary": summary,
            "fits": fits,
        }
        if selected is None and eligible:
            selected = name
    return results, selected


def _representation_payload(*, x_steps, edges, centers, dx, targets, dictionaries,
                            results, selected, fallback_disqualifies, stage):
    return {
        "stage": stage,
        "x_steps": x_steps,
        "edges": edges,
        "centers": centers,
        "dx": dx,
        "targets": targets,
        "dictionary_order": tuple(dictionaries),
        "results": results,
        "selected_dictionary": selected,
        "fallback_disqualifies": bool(fallback_disqualifies),
        "gates": {
            "optimality_gap": D1_GAP_TOL,
            "mass_error": D1_MASS_TOL,
            "cell_mass_l1": D1_L1_GATE,
            "wasserstein_1": D1_W1_GATE,
            "condition_alarm": D1_CONDITION_ALARM,
        },
    }


def run_f4_d_representation_study(*, x_steps=D1_X_STEPS):
    """Run the original preregistered static D1 representation study."""
    x_steps = checks.positive_integer(x_steps, "x_steps", minimum=21)
    edges, centers, dx = finite_volume_1d.grid(x_steps)
    targets = _target_suite(edges)
    dictionaries = build_d1_dictionaries(edges)
    results, selected = _evaluate_static_dictionaries(
        edges=edges,
        centers=centers,
        dx=dx,
        targets=targets,
        dictionaries=dictionaries,
        fallback_disqualifies=True,
    )
    return _representation_payload(
        x_steps=x_steps,
        edges=edges,
        centers=centers,
        dx=dx,
        targets=targets,
        dictionaries=dictionaries,
        results=results,
        selected=selected,
        fallback_disqualifies=True,
        stage="D1",
    )


def run_f4_d_adaptive_representation_study(*, x_steps=D1_X_STEPS):
    """Run D1b after D1; same targets/gates, adaptive dictionaries only."""
    x_steps = checks.positive_integer(x_steps, "x_steps", minimum=21)
    edges, centers, dx = finite_volume_1d.grid(x_steps)
    targets = _target_suite(edges)
    dictionaries = build_d1b_dictionaries(edges)
    results, selected = _evaluate_static_dictionaries(
        edges=edges,
        centers=centers,
        dx=dx,
        targets=targets,
        dictionaries=dictionaries,
        fallback_disqualifies=False,
    )
    return _representation_payload(
        x_steps=x_steps,
        edges=edges,
        centers=centers,
        dx=dx,
        targets=targets,
        dictionaries=dictionaries,
        results=results,
        selected=selected,
        fallback_disqualifies=False,
        stage="D1b",
    )
