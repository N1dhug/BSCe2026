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
from src.method.bm_1_estimation import (
    bm1_observation_update,
    run_fv_filter,
    run_mixture_filter,
)
from src.solver.finite_volume import finite_volume_1d
from src.validation import estimation as checks
from src.validation.bm_1.estimation import (
    beta_initial_mass,
    build_f4_b_case,
    generate_synthetic_sequence,
)


D1_X_STEPS = 801
D1_GAP_TOL = 1.0e-9
D1_MASS_TOL = 1.0e-9
D1_L1_GATE = 0.05
D1_W1_GATE = 0.01
D1_CONDITION_ALARM = 1.0e12
D1_ACTIVE_TOL = 1.0e-10
D2_SEQUENCE_START = 1001
D2_SEQUENCE_COUNT = 10
D2_X_STEPS = 801
D2_MAX_STEP = 1.0 / 192.0
D2_QUADRATURE_ORDER = 1
D2_DICTIONARY = "shape_D"
D3A_SEQUENCE_ID = 0
D3A_X_STEPS = 101
D3A_MAX_STEP = 1.0 / 384.0
D3A_CFL = 0.4
D3A_QUADRATURE_ORDER = 1
D3B_TIME_STEPS = (1.0 / 384.0, 1.0 / 768.0, 1.0 / 1536.0)
D3B_SPATIAL_CELLS = (51, 101, 151)
D3B_SPATIAL_MAX_STEP = 1.0 / 1536.0
D3B_STABILITY_RATIO = 0.10
D3C_TIME_STEPS = (1.0 / 768.0, 1.0 / 1536.0, 1.0 / 3072.0)
D3C_SPATIAL_CELLS = (101, 151, 201)
D3C_SPATIAL_MAX_STEP = 1.0 / 3072.0
D3C_LIMIT_RATIO = 0.10
D3C_PREDICTION_REL_TOL = 0.20
D3C_SPATIAL_REL_TOL = 0.10
D3C_POPULATION_X_STEPS = 151
D3C_POPULATION_MAX_STEP = 1.0 / 3072.0

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



def _shape_aware_dictionary(edges, high_step):
    """D1c scale-aware dictionary fixed after D1b and before D1c results."""
    high_step = checks.finite_scalar(high_step, "high_step", positive=True)
    low_means = np.arange(1, 10, dtype=float) / 10.0
    medium_means = np.arange(1, 40, dtype=float) / 40.0
    medium_means = np.sort(np.concatenate((medium_means, np.array([0.02, 0.98]))))
    boundary_means = np.array([0.02, 0.05, 0.95, 0.98], dtype=float)

    count = int(round(0.8 / high_step)) + 1
    high_means = np.linspace(0.1, 0.9, count)
    if not np.isclose(high_means[1] - high_means[0], high_step, atol=1e-14, rtol=0.0):
        raise ValueError("high_step must divide [0.1, 0.9] exactly.")

    means = np.concatenate((
        low_means,
        boundary_means,
        medium_means,
        high_means,
        high_means,
    ))
    concentrations = np.concatenate((
        np.full(low_means.size, 20.0),
        np.full(boundary_means.size, 50.0),
        np.full(medium_means.size, 100.0),
        np.full(high_means.size, 800.0),
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


def build_d1c_dictionaries(edges):
    """Build the two nested D1c shape-aware dictionaries preregistered after D1b."""
    edges = checks.increasing_grid(edges, "edges")
    widths = np.diff(edges)
    dx = float(widths[0])
    if edges[0] != 0.0 or edges[-1] != 1.0 or not np.allclose(widths, dx, atol=1e-14, rtol=1e-10):
        raise ValueError("D1c dictionaries require a uniform grid spanning exactly [0, 1].")
    return {
        "shape_C": _shape_aware_dictionary(edges, 0.025),
        "shape_D": _shape_aware_dictionary(edges, 0.0125),
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



def _snapshot_projection(projector, reference_mass, centers):
    """Project one FV cell-mass snapshot and return metrics plus projected mass."""
    reference_mass = checks.probability_mass(reference_mass, "reference_mass")
    if reference_mass.shape != centers.shape:
        raise ValueError("reference_mass and centers must have matching shapes.")
    target_mean, target_variance = _moments(reference_mass, centers)
    target_density = reference_mass / projector.dx
    projected = projector.project(target_density)
    density = projected["density"]
    projected_mass = density * projector.dx
    mean, variance = _moments(projected_mass, centers)
    diagnostics = projected["diagnostics"]
    coefficients = projected["coefficients"]
    fit = {
        "status": "ok",
        "density_l2": density_l2(density, target_density, projector.dx),
        "cell_mass_l1": cell_mass_l1(projected_mass, reference_mass),
        "cdf_sup": cell_mass_cdf_sup(projected_mass, reference_mass),
        "wasserstein_1": cell_mass_wasserstein_1(
            projected_mass, reference_mass, projector.dx,
        ),
        "mean_error": mean - target_mean,
        "variance_error": variance - target_variance,
        "optimality_gap": float(diagnostics["optimality_gap"]),
        "mass_error": float(diagnostics["mass_error"]),
        "used_fallback": bool(diagnostics["used_fallback"]),
        "active_components": int(np.sum(coefficients > D1_ACTIVE_TOL)),
        "maximum_coefficient": float(np.max(coefficients)),
    }
    return fit, projected_mass



def _mass_distribution_errors(left_mass, right_mass, centers, dx):
    left_mass = checks.probability_mass(left_mass, "left_mass")
    right_mass = checks.probability_mass(right_mass, "right_mass")
    if left_mass.shape != right_mass.shape or left_mass.shape != centers.shape:
        raise ValueError("mass distributions and centers must have matching shapes.")
    left_mean, left_variance = _moments(left_mass, centers)
    right_mean, right_variance = _moments(right_mass, centers)
    return {
        "cell_mass_l1": cell_mass_l1(left_mass, right_mass),
        "cdf_sup": cell_mass_cdf_sup(left_mass, right_mass),
        "wasserstein_1": cell_mass_wasserstein_1(left_mass, right_mass, dx),
        "density_l2": density_l2(left_mass / dx, right_mass / dx, dx),
        "mean_error": left_mean - right_mean,
        "variance_error": left_variance - right_variance,
    }


def _error_history(left, right, centers, dx):
    return [
        _mass_distribution_errors(lrow, rrow, centers, dx)
        for lrow, rrow in zip(left, right)
    ]


def _max_error_history(history):
    return {
        "maximum_cell_mass_l1": max(item["cell_mass_l1"] for item in history),
        "maximum_cdf_sup": max(item["cdf_sup"] for item in history),
        "maximum_wasserstein_1": max(item["wasserstein_1"] for item in history),
        "maximum_density_l2": max(item["density_l2"] for item in history),
        "maximum_abs_mean_error": max(abs(item["mean_error"]) for item in history),
        "maximum_abs_variance_error": max(abs(item["variance_error"]) for item in history),
    }


def _aggregate_snapshot_fits(fits):
    """Summarize a fixed set of successful/failed snapshot fits."""
    failed = tuple(index for index, item in enumerate(fits) if item["status"] != "ok")
    successful = [item for item in fits if item["status"] == "ok"]
    if failed or not successful:
        return {
            "failed_snapshot_indices": failed,
            "maximum_cell_mass_l1": float("inf"),
            "maximum_wasserstein_1": float("inf"),
            "maximum_cdf_sup": float("inf"),
            "maximum_density_l2": float("inf"),
            "maximum_abs_mean_error": float("inf"),
            "maximum_abs_variance_error": float("inf"),
            "maximum_optimality_gap": float("inf"),
            "maximum_mass_error": float("inf"),
            "fallback_count": sum(
                int(item.get("used_fallback", False)) for item in successful
            ),
            "passes_representation_gate": False,
        }
    max_l1 = max(item["cell_mass_l1"] for item in successful)
    max_w1 = max(item["wasserstein_1"] for item in successful)
    max_gap = max(item["optimality_gap"] for item in successful)
    max_mass_error = max(item["mass_error"] for item in successful)
    return {
        "failed_snapshot_indices": (),
        "maximum_cell_mass_l1": max_l1,
        "maximum_wasserstein_1": max_w1,
        "maximum_cdf_sup": max(item["cdf_sup"] for item in successful),
        "maximum_density_l2": max(item["density_l2"] for item in successful),
        "maximum_abs_mean_error": max(abs(item["mean_error"]) for item in successful),
        "maximum_abs_variance_error": max(abs(item["variance_error"]) for item in successful),
        "maximum_optimality_gap": max_gap,
        "maximum_mass_error": max_mass_error,
        "fallback_count": sum(int(item["used_fallback"]) for item in successful),
        "passes_representation_gate": bool(
            max_l1 <= D1_L1_GATE
            and max_w1 <= D1_W1_GATE
            and max_gap <= D1_GAP_TOL
            and max_mass_error <= D1_MASS_TOL
        ),
    }


def run_f4_d_snapshot_holdout(
    *,
    sequence_start=D2_SEQUENCE_START,
    n_sequences=D2_SEQUENCE_COUNT,
    x_steps=D2_X_STEPS,
    max_step=D2_MAX_STEP,
    likelihood_quadrature_order=D2_QUADRATURE_ORDER,
):
    """Run preregistered D2 static holdout on FV predictions/posteriors.

    The selected shape-D dictionary is frozen.  FV truth states are not used.
    Projected predictions are scored through the same observation operator only
    to diagnose evidence distortion; those projected posteriors are never
    carried forward.
    """
    sequence_start = checks.positive_integer(sequence_start, "sequence_start", minimum=0)
    n_sequences = checks.positive_integer(n_sequences, "n_sequences", minimum=1)
    x_steps = checks.positive_integer(x_steps, "x_steps", minimum=21)
    max_step = checks.finite_scalar(max_step, "max_step", positive=True)
    likelihood_quadrature_order = checks.positive_integer(
        likelihood_quadrature_order, "likelihood_quadrature_order", minimum=1,
    )

    case = build_f4_b_case()
    model, parameters = case.build_model()
    edges, centers, dx = model.fokker_planc_reference.grid(x_steps, bounds=(0.0, 1.0))
    dictionary = build_d1c_dictionaries(edges)[D2_DICTIONARY]
    projector = PreparedBetaProjector(dictionary["basis"], dx)

    sequence_ids = tuple(range(sequence_start, sequence_start + n_sequences))
    prediction_fits = []
    posterior_fits = []
    evidence_errors = []
    daily_evidence_errors = []
    invalid_truth = []

    for sequence_id in sequence_ids:
        sequence = generate_synthetic_sequence(case, sequence_id=sequence_id)
        if not sequence.truth.valid:
            invalid_truth.append(sequence_id)
            continue
        initial_mass = beta_initial_mass(case, model, x_steps)
        filtered = run_fv_filter(
            model,
            parameters,
            case.sigma_x,
            case.sigma_y,
            initial_mass,
            case.horizon[0],
            sequence.observations,
            breakpoints=case.breakpoints,
            max_step=max_step,
            likelihood_quadrature_order=likelihood_quadrature_order,
        )
        previous_time = case.horizon[0]
        sequence_evidence_error = 0.0
        for index, (time, value, observed) in enumerate(zip(
            sequence.observations.times,
            sequence.observations.values,
            sequence.observations.observed,
        )):
            try:
                pred_fit, projected_pred_mass = _snapshot_projection(
                    projector, filtered.predicted_mass[index], centers,
                )
            except (FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
                pred_fit = {
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                }
                projected_pred_mass = None
            prediction_fits.append(pred_fit)

            try:
                post_fit, _ = _snapshot_projection(
                    projector, filtered.posterior_mass[index], centers,
                )
            except (FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
                post_fit = {
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                }
            posterior_fits.append(post_fit)

            if observed and projected_pred_mass is not None:
                update = bm1_observation_update(
                    model,
                    parameters,
                    case.sigma_y,
                    projected_pred_mass,
                    previous_time,
                    float(time),
                    float(value),
                    price_side=sequence.observations.price_side,
                    likelihood_quadrature_order=likelihood_quadrature_order,
                )
                error = float(update.log_evidence - filtered.log_evidence[index])
                evidence_errors.append(error)
                sequence_evidence_error += error
            previous_time = float(time)
        daily_evidence_errors.append(sequence_evidence_error)

    prediction_summary = _aggregate_snapshot_fits(prediction_fits)
    posterior_summary = _aggregate_snapshot_fits(posterior_fits)
    evidence_errors = np.asarray(evidence_errors, dtype=float)
    daily_evidence_errors = np.asarray(daily_evidence_errors, dtype=float)
    expected_evidence_count = (
        (n_sequences - len(invalid_truth)) * case.observation_times.size
    )
    evidence_complete = evidence_errors.size == expected_evidence_count
    if evidence_errors.size:
        evidence_summary = {
            "complete": bool(evidence_complete),
            "count": int(evidence_errors.size),
            "expected_count": int(expected_evidence_count),
            "mean_absolute_per_observation": float(np.mean(np.abs(evidence_errors))),
            "maximum_absolute_per_observation": float(np.max(np.abs(evidence_errors))),
            "mean_signed_per_observation": float(np.mean(evidence_errors)),
            "maximum_absolute_daily_sum_difference": (
                float(np.max(np.abs(daily_evidence_errors)))
                if evidence_complete else float("inf")
            ),
        }
    else:
        evidence_summary = {
            "complete": False,
            "count": 0,
            "expected_count": int(expected_evidence_count),
            "mean_absolute_per_observation": float("inf"),
            "maximum_absolute_per_observation": float("inf"),
            "mean_signed_per_observation": float("nan"),
            "maximum_absolute_daily_sum_difference": float("inf"),
        }

    return {
        "stage": "D2",
        "dictionary": D2_DICTIONARY,
        "components": dictionary["components"],
        "sequence_ids": sequence_ids,
        "x_steps": x_steps,
        "max_step": max_step,
        "likelihood_quadrature_order": likelihood_quadrature_order,
        "prediction": prediction_summary,
        "posterior": posterior_summary,
        "predictive_log_evidence_error": evidence_summary,
        "invalid_truth_sequences": tuple(invalid_truth),
        "projection_matrix_condition_number": float(
            projector.projection_solver.matrix_condition_number
        ),
        "hessian_condition_number": float(projector.projection_data[2]),
        "passes_holdout": bool(
            not invalid_truth
            and prediction_summary["passes_representation_gate"]
            and posterior_summary["passes_representation_gate"]
        ),
    }


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


def run_f4_d_shape_aware_representation_study(*, x_steps=D1_X_STEPS):
    """Run D1c after D1b; same targets/gates, shape-aware dictionaries only."""
    x_steps = checks.positive_integer(x_steps, "x_steps", minimum=21)
    edges, centers, dx = finite_volume_1d.grid(x_steps)
    targets = _target_suite(edges)
    dictionaries = build_d1c_dictionaries(edges)
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
        stage="D1c",
    )


def run_f4_d_recursive_sanity(
    *,
    sequence_id=D3A_SEQUENCE_ID,
    x_steps=D3A_X_STEPS,
    max_step=D3A_MAX_STEP,
    cfl=D3A_CFL,
    likelihood_quadrature_order=D3A_QUADRATURE_ORDER,
):
    """Run preregistered D3a recursive mixture sanity against matched-grid FV."""
    sequence_id = checks.positive_integer(sequence_id, "sequence_id", minimum=0)
    x_steps = checks.positive_integer(x_steps, "x_steps", minimum=21)
    max_step = checks.finite_scalar(max_step, "max_step", positive=True)
    cfl = checks.finite_scalar(cfl, "cfl", positive=True)
    likelihood_quadrature_order = checks.positive_integer(
        likelihood_quadrature_order, "likelihood_quadrature_order", minimum=1,
    )
    case = build_f4_b_case()
    sequence = generate_synthetic_sequence(case, sequence_id=sequence_id)
    model, parameters = case.build_model()
    edges, centers, dx = model.fokker_planc_reference.grid(x_steps, bounds=(0.0, 1.0))
    dictionary = build_d1c_dictionaries(edges)[D2_DICTIONARY]
    mixture_projector = PreparedBetaProjector(dictionary["basis"], dx)
    static_projector = PreparedBetaProjector(dictionary["basis"], dx)
    initial_mass = beta_initial_mass(case, model, x_steps)

    mixture = run_mixture_filter(
        model,
        parameters,
        case.sigma_x,
        case.sigma_y,
        initial_mass,
        case.horizon[0],
        sequence.observations,
        projector=mixture_projector,
        breakpoints=case.breakpoints,
        max_step=max_step,
        cfl=cfl,
        likelihood_quadrature_order=likelihood_quadrature_order,
    )
    fv = run_fv_filter(
        model,
        parameters,
        case.sigma_x,
        case.sigma_y,
        initial_mass,
        case.horizon[0],
        sequence.observations,
        breakpoints=case.breakpoints,
        max_step=max_step,
        likelihood_quadrature_order=likelihood_quadrature_order,
    )

    static_pred_mass = []
    static_post_mass = []
    static_pred_fits = []
    static_post_fits = []
    for pred, post in zip(fv.predicted_mass, fv.posterior_mass):
        pred_fit, pred_mass = _snapshot_projection(static_projector, pred, centers)
        post_fit, post_mass = _snapshot_projection(static_projector, post, centers)
        static_pred_fits.append(pred_fit)
        static_post_fits.append(post_fit)
        static_pred_mass.append(pred_mass)
        static_post_mass.append(post_mass)
    static_pred_mass = np.stack(static_pred_mass)
    static_post_mass = np.stack(static_post_mass)

    original_initial = initial_mass
    projected_initial = mixture.initial_mass
    initial_error = _mass_distribution_errors(projected_initial, original_initial, centers, dx)
    static_prediction = _aggregate_snapshot_fits(static_pred_fits)
    static_posterior = _aggregate_snapshot_fits(static_post_fits)
    dynamic_prediction = _max_error_history(
        _error_history(mixture.predicted_mass, static_pred_mass, centers, dx)
    )
    dynamic_posterior = _max_error_history(
        _error_history(mixture.posterior_mass, static_post_mass, centers, dx)
    )
    total_prediction = _max_error_history(
        _error_history(mixture.predicted_mass, fv.predicted_mass, centers, dx)
    )
    total_posterior = _max_error_history(
        _error_history(mixture.posterior_mass, fv.posterior_mass, centers, dx)
    )

    observed = sequence.observations.observed
    evidence_error = mixture.log_evidence[observed] - fv.log_evidence[observed]
    truth = sequence.truth.state[1:]
    rmse = lambda values: float(np.sqrt(np.mean(np.square(values - truth))))

    pred_diag = mixture.prediction_diagnostics
    post_diag = mixture.posterior_projection_diagnostics
    numerical = {
        "total_internal_steps": int(sum(item["n_steps"] for item in pred_diag)),
        "maximum_internal_step": float(max(item["maximum_internal_step"] for item in pred_diag)),
        "minimum_internal_step": float(min(item["minimum_internal_step"] for item in pred_diag)),
        "prediction_projection_fallback_count": int(
            sum(item["projection_fallback_count"] for item in pred_diag)
        ),
        "posterior_projection_fallback_count": int(
            sum(int(item["used_fallback"]) for item in post_diag)
        ),
        "maximum_raw_prediction_negativity": float(
            min(item["minimum_raw_prediction"] for item in pred_diag)
        ),
        "maximum_abs_raw_prediction_mass_error": float(
            max(item["maximum_abs_raw_prediction_mass_error"] for item in pred_diag)
        ),
        "maximum_abs_projected_prediction_mass_error": float(
            max(item["maximum_abs_projected_mass_error"] for item in pred_diag)
        ),
        "maximum_posterior_projection_mass_error": float(
            max(item["mass_error"] for item in post_diag)
        ),
        "maximum_bayes_to_projected_posterior_l1": float(
            max(item["raw_to_projected_cell_mass_l1"] for item in post_diag)
        ),
        "projection_matrix_condition_number": float(
            mixture_projector.projection_solver.matrix_condition_number
        ),
        "hessian_condition_number": float(mixture_projector.projection_data[2]),
    }

    return {
        "stage": "D3a",
        "sequence_id": sequence_id,
        "dictionary": D2_DICTIONARY,
        "components": dictionary["components"],
        "x_steps": x_steps,
        "max_step": max_step,
        "cfl": cfl,
        "likelihood_quadrature_order": likelihood_quadrature_order,
        "initial_projection_error": initial_error,
        "static_best_fit": {
            "prediction": static_prediction,
            "posterior": static_posterior,
        },
        "dynamic_gap": {
            "prediction": dynamic_prediction,
            "posterior": dynamic_posterior,
        },
        "total_fv_gap": {
            "prediction": total_prediction,
            "posterior": total_posterior,
        },
        "log_evidence_error": {
            "mean_absolute_per_observation": float(np.mean(np.abs(evidence_error))),
            "maximum_absolute_per_observation": float(np.max(np.abs(evidence_error))),
            "mean_signed_per_observation": float(np.mean(evidence_error)),
            "absolute_total_log_likelihood_difference": abs(
                mixture.total_log_likelihood - fv.total_log_likelihood
            ),
        },
        "state_rmse": {
            "fv_predictive": rmse(fv.predicted_mean),
            "fv_posterior": rmse(fv.posterior_mean),
            "mixture_predictive": rmse(mixture.predicted_mean),
            "mixture_posterior": rmse(mixture.posterior_mean),
        },
        "numerical": numerical,
        "mixture": mixture,
        "fv": fv,
    }



def _d3_refinement_row(result, *, axis, level):
    """Extract compact metrics from one D3a configuration for refinement."""
    total_pred = result["total_fv_gap"]["prediction"]
    total_post = result["total_fv_gap"]["posterior"]
    dynamic_pred = result["dynamic_gap"]["prediction"]
    dynamic_post = result["dynamic_gap"]["posterior"]
    static_pred = result["static_best_fit"]["prediction"]
    static_post = result["static_best_fit"]["posterior"]
    evidence = result["log_evidence_error"]
    rmse = result["state_rmse"]
    numerical = result["numerical"]
    return {
        "axis": axis,
        "level": level,
        "x_steps": result["x_steps"],
        "requested_max_step": result["max_step"],
        "actual_max_step": numerical["maximum_internal_step"],
        "actual_min_step": numerical["minimum_internal_step"],
        "internal_steps": numerical["total_internal_steps"],
        "static_prediction_l1": static_pred["maximum_cell_mass_l1"],
        "static_posterior_l1": static_post["maximum_cell_mass_l1"],
        "dynamic_prediction_l1": dynamic_pred["maximum_cell_mass_l1"],
        "dynamic_posterior_l1": dynamic_post["maximum_cell_mass_l1"],
        "total_prediction_l1": total_pred["maximum_cell_mass_l1"],
        "total_posterior_l1": total_post["maximum_cell_mass_l1"],
        "total_prediction_w1": total_pred["maximum_wasserstein_1"],
        "total_posterior_w1": total_post["maximum_wasserstein_1"],
        "evidence_mean_abs": evidence["mean_absolute_per_observation"],
        "evidence_max_abs": evidence["maximum_absolute_per_observation"],
        "evidence_total_abs": evidence["absolute_total_log_likelihood_difference"],
        "fv_predictive_rmse": rmse["fv_predictive"],
        "fv_posterior_rmse": rmse["fv_posterior"],
        "mixture_predictive_rmse": rmse["mixture_predictive"],
        "mixture_posterior_rmse": rmse["mixture_posterior"],
        "prediction_fallbacks": numerical["prediction_projection_fallback_count"],
        "posterior_fallbacks": numerical["posterior_projection_fallback_count"],
        "max_raw_mass_error": numerical["maximum_abs_raw_prediction_mass_error"],
        "max_projected_mass_error": numerical["maximum_abs_projected_prediction_mass_error"],
        "max_posterior_projection_mass_error": numerical["maximum_posterior_projection_mass_error"],
        "max_bayes_projection_l1": numerical["maximum_bayes_to_projected_posterior_l1"],
        "projection_matrix_condition_number": numerical["projection_matrix_condition_number"],
        "hessian_condition_number": numerical["hessian_condition_number"],
    }


def _d3_refinement_metric_summary(rows, metric):
    reference = rows[-1]
    reference_value = float(reference[metric])
    effects = [abs(float(row[metric]) - reference_value) for row in rows[:-1]]
    maximum_effect = max(effects, default=0.0)
    ratio = np.nan if abs(reference_value) < 1e-14 else maximum_effect / abs(reference_value)
    return {
        "reference_level": reference["level"],
        "reference_value": reference_value,
        "maximum_effect": maximum_effect,
        "effect_to_reference_ratio": ratio,
        "passes_10pct": bool(np.isfinite(ratio) and ratio <= D3B_STABILITY_RATIO),
    }


def _d3_refinement_axis_summary(rows):
    posterior = _d3_refinement_metric_summary(rows, "total_posterior_l1")
    evidence = _d3_refinement_metric_summary(rows, "evidence_mean_abs")
    return {
        "posterior_l1": posterior,
        "evidence_mean_abs": evidence,
        "passes_stability_gate": bool(
            posterior["passes_10pct"] and evidence["passes_10pct"]
        ),
    }


def run_f4_d_recursive_refinement(
    sequence_id=D3A_SEQUENCE_ID,
    *,
    time_steps=D3B_TIME_STEPS,
    spatial_cells=D3B_SPATIAL_CELLS,
    spatial_max_step=D3B_SPATIAL_MAX_STEP,
    cfl=D3A_CFL,
    likelihood_quadrature_order=D3A_QUADRATURE_ORDER,
):
    """Run preregistered D3b time/projection-frequency and spatial refinement.

    The exact (x_steps, max_step) configuration is memoized within this study,
    so the shared (101, 1/1536) fine-time/spatial-middle run is computed once.
    """
    sequence_id = checks.positive_integer(sequence_id, "sequence_id", minimum=0)
    time_steps = tuple(checks.finite_scalar(v, "time step", positive=True) for v in time_steps)
    spatial_cells = tuple(checks.positive_integer(v, "spatial cell count", minimum=21) for v in spatial_cells)
    spatial_max_step = checks.finite_scalar(spatial_max_step, "spatial_max_step", positive=True)
    cfl = checks.finite_scalar(cfl, "cfl", positive=True)
    order = checks.positive_integer(
        likelihood_quadrature_order, "likelihood_quadrature_order", minimum=1,
    )
    if not time_steps or not spatial_cells:
        raise ValueError("D3b refinement levels must be non-empty.")
    if tuple(sorted(set(time_steps), reverse=True)) != time_steps:
        raise ValueError("time_steps must be unique and strictly refined (decreasing).")
    if tuple(sorted(set(spatial_cells))) != spatial_cells:
        raise ValueError("spatial_cells must be unique and increasing.")

    cache = {}
    def run_config(x_steps, max_step):
        key = (int(x_steps), float(max_step))
        if key not in cache:
            cache[key] = run_f4_d_recursive_sanity(
                sequence_id=sequence_id,
                x_steps=x_steps,
                max_step=max_step,
                cfl=cfl,
                likelihood_quadrature_order=order,
            )
        return cache[key]

    time_rows = [
        _d3_refinement_row(
            run_config(D3A_X_STEPS, h), axis="time", level=h,
        )
        for h in time_steps
    ]
    spatial_rows = [
        _d3_refinement_row(
            run_config(n, spatial_max_step), axis="space", level=n,
        )
        for n in spatial_cells
    ]
    return {
        "stage": "D3b",
        "sequence_id": sequence_id,
        "cfl": cfl,
        "likelihood_quadrature_order": order,
        "time": time_rows,
        "spatial": spatial_rows,
        "summary": {
            "time": _d3_refinement_axis_summary(time_rows),
            "spatial": _d3_refinement_axis_summary(spatial_rows),
        },
        "unique_configurations": tuple(cache),
        "stability_ratio": D3B_STABILITY_RATIO,
    }



def _d3c_first_order_summary(rows, metric):
    """Check the preregistered first-order Richardson prediction on three halvings."""
    if len(rows) != 3:
        raise ValueError("D3c temporal Richardson summary requires exactly three levels.")
    h0, h1, h2 = (float(row["requested_max_step"]) for row in rows)
    if not (
        np.isclose(h0, 2.0 * h1, atol=1e-15, rtol=1e-12)
        and np.isclose(h1, 2.0 * h2, atol=1e-15, rtol=1e-12)
    ):
        raise ValueError("D3c temporal levels must halve exactly.")
    e0_obs, e1_obs, e2_obs = (float(row[metric]) for row in rows)
    extrapolated_limit = 2.0 * e1_obs - e0_obs
    predicted_fine = extrapolated_limit + 0.5 * (e1_obs - extrapolated_limit)
    prediction_scale = max(abs(predicted_fine), 1e-14)
    relative_prediction_error = abs(e2_obs - predicted_fine) / prediction_scale
    middle_scale = max(abs(e1_obs), 1e-14)
    limit_to_middle_ratio = abs(extrapolated_limit) / middle_scale
    monotone_decrease = bool(e0_obs > e1_obs > e2_obs >= 0.0)
    step_limit_binding = all(
        np.isclose(
            float(row["actual_max_step"]),
            float(row["requested_max_step"]),
            atol=1e-14,
            rtol=1e-10,
        )
        for row in rows
    )
    passes = bool(
        monotone_decrease
        and step_limit_binding
        and limit_to_middle_ratio <= D3C_LIMIT_RATIO
        and relative_prediction_error <= D3C_PREDICTION_REL_TOL
    )
    return {
        "metric": metric,
        "extrapolated_limit": extrapolated_limit,
        "predicted_fine": predicted_fine,
        "observed_fine": e2_obs,
        "relative_prediction_error": relative_prediction_error,
        "limit_to_middle_ratio": limit_to_middle_ratio,
        "monotone_decrease": monotone_decrease,
        "step_limit_binding": step_limit_binding,
        "passes": passes,
    }


def _d3c_spatial_summary(rows, metric):
    """Check two successive relative spatial changes at common requested max-step."""
    if len(rows) != 3:
        raise ValueError("D3c spatial summary requires exactly three levels.")
    changes = []
    for left, right in zip(rows[:-1], rows[1:]):
        left_value = float(left[metric])
        right_value = float(right[metric])
        scale = max(abs(right_value), 1e-14)
        changes.append(abs(right_value - left_value) / scale)
    step_limit_binding = all(
        np.isclose(
            float(row["actual_max_step"]),
            float(row["requested_max_step"]),
            atol=1e-14,
            rtol=1e-10,
        )
        for row in rows
    )
    maximum_change = max(changes)
    return {
        "metric": metric,
        "successive_relative_changes": tuple(changes),
        "maximum_relative_change": maximum_change,
        "step_limit_binding": step_limit_binding,
        "passes": bool(
            step_limit_binding and maximum_change <= D3C_SPATIAL_REL_TOL
        ),
    }


def run_f4_d_recursive_convergence(
    sequence_id=D3A_SEQUENCE_ID,
    *,
    time_steps=D3C_TIME_STEPS,
    spatial_cells=D3C_SPATIAL_CELLS,
    spatial_max_step=D3C_SPATIAL_MAX_STEP,
    cfl=D3A_CFL,
    likelihood_quadrature_order=D3A_QUADRATURE_ORDER,
):
    """Run preregistered D3c Richardson/time and successive-spatial convergence checks."""
    sequence_id = checks.positive_integer(sequence_id, "sequence_id", minimum=0)
    time_steps = tuple(checks.finite_scalar(v, "time step", positive=True) for v in time_steps)
    spatial_cells = tuple(
        checks.positive_integer(v, "spatial cell count", minimum=21)
        for v in spatial_cells
    )
    spatial_max_step = checks.finite_scalar(
        spatial_max_step, "spatial_max_step", positive=True,
    )
    cfl = checks.finite_scalar(cfl, "cfl", positive=True)
    order = checks.positive_integer(
        likelihood_quadrature_order, "likelihood_quadrature_order", minimum=1,
    )
    if len(time_steps) != 3 or len(spatial_cells) != 3:
        raise ValueError("D3c requires exactly three temporal and three spatial levels.")
    if tuple(sorted(set(time_steps), reverse=True)) != time_steps:
        raise ValueError("D3c time_steps must be unique and decreasing.")
    if tuple(sorted(set(spatial_cells))) != spatial_cells:
        raise ValueError("D3c spatial_cells must be unique and increasing.")

    cache = {}
    def run_config(x_steps, max_step):
        key = (int(x_steps), float(max_step))
        if key not in cache:
            cache[key] = run_f4_d_recursive_sanity(
                sequence_id=sequence_id,
                x_steps=x_steps,
                max_step=max_step,
                cfl=cfl,
                likelihood_quadrature_order=order,
            )
        return cache[key]

    time_rows = [
        _d3_refinement_row(
            run_config(D3A_X_STEPS, h), axis="time", level=h,
        )
        for h in time_steps
    ]
    spatial_rows = [
        _d3_refinement_row(
            run_config(n, spatial_max_step), axis="space", level=n,
        )
        for n in spatial_cells
    ]

    time_posterior = _d3c_first_order_summary(time_rows, "total_posterior_l1")
    time_evidence = _d3c_first_order_summary(time_rows, "evidence_mean_abs")
    spatial_posterior = _d3c_spatial_summary(spatial_rows, "total_posterior_l1")
    spatial_evidence = _d3c_spatial_summary(spatial_rows, "evidence_mean_abs")
    time_passes = bool(time_posterior["passes"] and time_evidence["passes"])
    spatial_passes = bool(spatial_posterior["passes"] and spatial_evidence["passes"])
    passes = bool(time_passes and spatial_passes)

    return {
        "stage": "D3c",
        "sequence_id": sequence_id,
        "cfl": cfl,
        "likelihood_quadrature_order": order,
        "time": time_rows,
        "spatial": spatial_rows,
        "summary": {
            "time": {
                "posterior_l1": time_posterior,
                "evidence_mean_abs": time_evidence,
                "passes": time_passes,
            },
            "spatial": {
                "posterior_l1": spatial_posterior,
                "evidence_mean_abs": spatial_evidence,
                "passes": spatial_passes,
            },
            "passes_convergence": passes,
            "population_setting": (
                {
                    "x_steps": D3C_POPULATION_X_STEPS,
                    "max_step": D3C_POPULATION_MAX_STEP,
                    "cfl": D3A_CFL,
                    "likelihood_quadrature_order": D3A_QUADRATURE_ORDER,
                }
                if passes else None
            ),
        },
        "unique_configurations": tuple(cache),
        "criteria": {
            "limit_to_middle_ratio": D3C_LIMIT_RATIO,
            "fine_prediction_relative_error": D3C_PREDICTION_REL_TOL,
            "spatial_successive_relative_change": D3C_SPATIAL_REL_TOL,
        },
    }
