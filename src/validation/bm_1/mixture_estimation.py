"""Reusable synthetic/numerical reference calculations; retired campaigns use original sources."""
from __future__ import annotations
import numpy as np
from src.diagnostics.bm_1 import (
    cell_mass_cdf_sup,
    cell_mass_l1,
    cell_mass_wasserstein_1,
    density_l2,
)

from src.method.beta_mixture import PreparedBetaProjector, beta_cell_average_basis
from src.method.bm_1_estimation import run_fv_filter, run_mixture_filter
from src.validation import estimation as checks
from src.validation.bm_1.estimation import beta_initial_mass, build_f4_b_case, generate_synthetic_sequence, _paired_bootstrap_interval

D1_GAP_TOL = 1.0e-9

D1_MASS_TOL = 1.0e-9

D1_L1_GATE = 0.05

D1_W1_GATE = 0.01

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

D3C_LIMIT_RATIO = 0.10

D3C_PREDICTION_REL_TOL = 0.20

D3C_SPATIAL_REL_TOL = 0.10

D3C_POPULATION_X_STEPS = 151

D3C_POPULATION_MAX_STEP = 1.0 / 3072.0

D3D_SEQUENCE_START = 1011

D3D_SEQUENCE_COUNT = 200

D3D_BOOTSTRAP_REPLICATES = 5000

D3D_BOOTSTRAP_SEED = 4301

D3D_RMSE_EQUIVALENCE_LOW = 0.95

D3D_RMSE_EQUIVALENCE_HIGH = 1.05

D3D_SEQUENCE_MAX_L1_Q95 = 0.02

D3D_GLOBAL_MAX_L1 = 0.05

D3D_GLOBAL_MAX_W1 = 0.01

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

def _multiresolution_dictionary(edges, high_step, *, shape_aware):
    """Shared geometry builder; historical D1b/D1c specifications stay explicit."""
    high_step = checks.finite_scalar(high_step, "high_step", positive=True)
    low = np.arange(1, 10, dtype=float) / 10.0
    medium = np.sort(np.concatenate((np.arange(1, 40) / 40.0, [0.02, 0.98])))
    count = int(round(0.8 / high_step)) + 1
    if count < 2:
        raise ValueError("high_step must divide [0.1, 0.9] exactly.")
    high = np.linspace(0.1, 0.9, count)
    if not np.isclose(high[1] - high[0], high_step, atol=1e-14, rtol=0.0):
        raise ValueError("high_step must divide [0.1, 0.9] exactly.")
    layers = [(20.0, low)]
    if shape_aware:
        layers.append((50.0, np.array([0.02, 0.05, 0.95, 0.98])))
    layers.extend([(100.0, medium), (800.0 if shape_aware else 500.0, high),
                   (2000.0, high)])
    means = np.concatenate([values for _, values in layers])
    concentrations = np.concatenate([np.full(len(values), kappa) for kappa, values in layers])
    alpha, beta = concentrations * means, concentrations * (1.0 - means)
    return dict(basis=beta_cell_average_basis(edges, alpha, beta), means=means,
                concentrations=concentrations, alpha=alpha, beta=beta,
                components=int(alpha.size), selectable=True)

def _adaptive_dictionary(edges, high_step):
    """Declared D1b layers (20, 100, 500, 2000)."""
    return _multiresolution_dictionary(edges, high_step, shape_aware=False)

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
    """Declared D1c layers, including the frozen shape_D at high_step=0.0125."""
    return _multiresolution_dictionary(edges, high_step, shape_aware=True)

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

def _projection_metrics(projector, projected, reference_mass, target_density, centers):
    target_mean, target_variance = _moments(reference_mass, centers)
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
    return fit, projected_mass, coefficients

def _snapshot_projection(projector, reference_mass, centers, *, initial_coefficients=None):
    """Solve the same snapshot QP, optionally from explicit feasible coefficients.

    Warm starts change the numerical starting point, never the target or gates.
    The returned coefficients may initialize the next *static* reference fit;
    they are never supplied to the independent recursive mixture filter.
    """
    reference_mass = checks.probability_mass(reference_mass, "reference_mass")
    if reference_mass.shape != centers.shape:
        raise ValueError("reference_mass and centers must have matching shapes.")
    target_density = reference_mass / projector.dx
    projected = projector.project(target_density, initial_coefficients=initial_coefficients)
    return _projection_metrics(projector, projected, reference_mass, target_density, centers)

def _mass_distribution_errors(left_mass, right_mass, centers, dx):
    left_mass = checks.probability_mass(left_mass, "left_mass")
    right_mass = checks.probability_mass(right_mass, "right_mass")
    if left_mass.shape != right_mass.shape or left_mass.shape != centers.shape:
        raise ValueError("mass distributions and centers must have matching shapes.")
    left_mean, left_variance = _moments(left_mass, centers)
    right_mean, right_variance = _moments(right_mass, centers)
    difference = left_mass - right_mass
    cdf_difference = np.cumsum(left_mass) - np.cumsum(right_mass)
    return {
        "cell_mass_l1": float(np.sum(np.abs(difference))),
        "cdf_sup": float(np.max(np.abs(cdf_difference))),
        "wasserstein_1": float(dx * np.sum(np.abs(cdf_difference))),
        "density_l2": density_l2(left_mass / dx, right_mass / dx, dx),
        "mean_error": left_mean - right_mean,
        "variance_error": left_variance - right_variance,
    }

def _error_history(left, right, centers, dx):
    if np.shape(left) != np.shape(right) or len(np.shape(left)) != 2:
        raise ValueError("Distribution histories must have identical (times, cells) shapes.")
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
    dictionary = _shape_aware_dictionary(edges, 0.0125)
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
    static_coefficients = None
    for pred, post in zip(fv.predicted_mass, fv.posterior_mass):
        pred_fit, pred_mass, static_coefficients = _snapshot_projection(
            static_projector, pred, centers, initial_coefficients=static_coefficients,
        )
        post_fit, post_mass, static_coefficients = _snapshot_projection(
            static_projector, post, centers, initial_coefficients=static_coefficients,
        )
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
    common_step = float(rows[0]["requested_max_step"])
    if not all(float(row["requested_max_step"]) == common_step for row in rows):
        raise ValueError("Spatial convergence requires a common requested time step.")
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

def _d3d_rmse_summary(mixture_mse, fv_mse, *, replicates, seed):
    mixture_mse = checks.finite_vector(mixture_mse, "mixture_mse")
    fv_mse = checks.finite_vector(fv_mse, "fv_mse")
    if mixture_mse.shape != fv_mse.shape:
        raise ValueError("Paired D3d MSE vectors must match.")
    if np.any(mixture_mse < 0.0) or np.any(fv_mse <= 0.0):
        raise ValueError("MSEs must be non-negative and FV MSEs positive for a finite ratio bootstrap.")
    mixture_rmse = float(np.sqrt(np.mean(mixture_mse)))
    fv_rmse = float(np.sqrt(np.mean(fv_mse)))
    ratio = mixture_rmse / fv_rmse
    ratio_ci = _paired_bootstrap_interval(
        mixture_mse,
        fv_mse,
        lambda left, right: (
            np.sqrt(np.mean(left, axis=1)) / np.sqrt(np.mean(right, axis=1))
        ),
        replicates=replicates,
        seed=seed,
    )
    difference_ci = _paired_bootstrap_interval(
        mixture_mse,
        fv_mse,
        lambda left, right: (
            np.sqrt(np.mean(left, axis=1)) - np.sqrt(np.mean(right, axis=1))
        ),
        replicates=replicates,
        seed=seed + 1,
    )
    return {
        "fv": fv_rmse,
        "mixture": mixture_rmse,
        "ratio": ratio,
        "ratio_bootstrap95": ratio_ci,
        "difference": mixture_rmse - fv_rmse,
        "difference_bootstrap95": difference_ci,
        "passes_equivalence": bool(
            ratio_ci[0] >= D3D_RMSE_EQUIVALENCE_LOW
            and ratio_ci[1] <= D3D_RMSE_EQUIVALENCE_HIGH
        ),
    }

def _d3d_gap_summary(values, *, l1):
    values = checks.finite_vector(values, "gap values")
    result = {
        "mean": float(np.mean(values)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }
    if l1:
        result["passes"] = bool(
            result["q95"] <= D3D_SEQUENCE_MAX_L1_Q95
            and result["maximum"] <= D3D_GLOBAL_MAX_L1
        )
    else:
        result["passes"] = bool(result["maximum"] <= D3D_GLOBAL_MAX_W1)
    return result
