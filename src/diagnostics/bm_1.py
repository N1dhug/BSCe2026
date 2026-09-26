"""Reusable feasibility diagnostics for the deterministic bm_1 model.

The price diagnostics below are exact for the linear response functions
``f(X) = 1 - 2X`` and ``g(U) = 1 - 2U`` used by the F1 benchmark.  Their names
make that restriction explicit so they are not mistaken for inverses of an
arbitrary response pair.
"""

from dataclasses import dataclass

import numpy as np

from src.validation.diagnostics.bm_1 import bm_1_diagnostics_validation


def _scalar_if_needed(values):
    """Return a float for scalar results and an array otherwise."""
    values = np.asarray(values, dtype=float)
    return float(values) if values.ndim == 0 else values


def linear_response_required_price(
    state,
    demand,
    baseline,
    flexibility,
    steepness,
):
    """Invert the F1 linear responses and return the required price.

    A returned value outside ``[0, 1]`` is a feasibility certificate: the
    requested state-demand pair cannot be produced by an admissible price.
    """
    (
        state,
        demand,
        baseline,
        flexibility,
        steepness,
    ) = bm_1_diagnostics_validation.response_inputs(
        state,
        demand,
        baseline,
        flexibility,
        steepness,
    )

    positive = demand >= baseline
    scale = np.where(
        positive,
        flexibility * (1.0 - baseline),
        flexibility * baseline,
    )
    normalized_response = (demand - baseline) / scale
    if np.any(np.abs(normalized_response) >= 1.0):
        raise ValueError("demand requires a normalized response outside (-1, 1).")

    price = 1.0 - state - np.arctanh(normalized_response) / steepness
    return _scalar_if_needed(price)


def linear_response_demand_envelope(
    state,
    baseline,
    flexibility,
    steepness,
):
    """Return attainable demand bounds for the F1 responses and ``U in [0, 1]``."""
    state, baseline, flexibility, steepness = (
        bm_1_diagnostics_validation.envelope_inputs(
            state,
            baseline,
            flexibility,
            steepness,
        )
    )
    minimum = baseline * (
        1.0 - flexibility * np.tanh(steepness * state)
    )
    maximum = baseline + flexibility * (1.0 - baseline) * np.tanh(
        steepness * (1.0 - state)
    )
    return _scalar_if_needed(minimum), _scalar_if_needed(maximum)


def constant_demand_implied_state(
    time,
    target_demand,
    baseline,
    capacity,
    initial_state,
    initial_time=0.0,
):
    """Return the state trajectory required by a constant target demand."""
    (
        time,
        target_demand,
        baseline,
        capacity,
        initial_state,
        initial_time,
    ) = bm_1_diagnostics_validation.implied_state_inputs(
        time,
        target_demand,
        baseline,
        capacity,
        initial_state,
        initial_time,
    )
    state = initial_state + (
        (target_demand - baseline) * (time - initial_time) / capacity
    )
    return _scalar_if_needed(state)


@dataclass(frozen=True)
class CapacityCertificate:
    """Analytical state-capacity certificate for a constant demand target."""

    required_final_state: float
    target_energy: float
    available_energy: float
    excess_energy: float
    objective_lower_bound: float
    violation_time: float | None
    violated_boundary: str | None

    @property
    def feasible(self):
        """Return whether the target respects state capacity over the interval."""
        return self.violated_boundary is None


def constant_demand_capacity_certificate(
    target_demand,
    baseline,
    capacity,
    initial_state,
    interval,
):
    """Certify whether a constant demand target exceeds state capacity.

    When capacity is exceeded, ``objective_lower_bound`` follows from the
    unavoidable energy mismatch and the Cauchy--Schwarz inequality.
    """
    (
        target_demand,
        baseline,
        capacity,
        initial_state,
        interval,
    ) = bm_1_diagnostics_validation.capacity_inputs(
        target_demand,
        baseline,
        capacity,
        initial_state,
        interval,
    )

    start, end = interval
    duration = end - start
    net_power = target_demand - baseline
    target_energy = net_power * duration
    required_final_state = initial_state + target_energy / capacity

    if net_power > 0.0:
        available_energy = capacity * (1.0 - initial_state)
        raw_excess = target_energy - available_energy
        tolerance = 10.0 * np.finfo(float).eps * max(
            abs(target_energy), abs(available_energy), 1.0
        )
        excess_energy = raw_excess if raw_excess > tolerance else 0.0
        violated_boundary = "upper" if excess_energy > 0.0 else None
        violation_time = (
            start + available_energy / net_power
            if violated_boundary is not None
            else None
        )
    elif net_power < 0.0:
        available_energy = capacity * initial_state
        raw_excess = -target_energy - available_energy
        tolerance = 10.0 * np.finfo(float).eps * max(
            abs(target_energy), abs(available_energy), 1.0
        )
        excess_energy = raw_excess if raw_excess > tolerance else 0.0
        violated_boundary = "lower" if excess_energy > 0.0 else None
        violation_time = (
            start + available_energy / (-net_power)
            if violated_boundary is not None
            else None
        )
    else:
        available_energy = 0.0
        excess_energy = 0.0
        violated_boundary = None
        violation_time = None

    return CapacityCertificate(
        required_final_state=float(required_final_state),
        target_energy=float(target_energy),
        available_energy=float(available_energy),
        excess_energy=float(excess_energy),
        objective_lower_bound=float(excess_energy**2 / duration),
        violation_time=(
            None if violation_time is None else float(violation_time)
        ),
        violated_boundary=violated_boundary,
    )



def empirical_cell_mass(samples, edges):
    """Bin one-dimensional Monte Carlo samples on supplied FV cell edges."""
    samples = np.asarray(samples, dtype=float).reshape(-1)
    edges = np.asarray(edges, dtype=float)
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain finite values.")
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("edges must be a strictly increasing one-dimensional array.")
    if np.any(samples < edges[0]) or np.any(samples > edges[-1]):
        raise ValueError("samples must lie inside the supplied histogram edges.")
    counts, _ = np.histogram(samples, bins=edges)
    return counts.astype(float) / samples.size


def cell_mass_l1(left, right):
    """Return L1 distance between two discrete cell-mass distributions."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("cell-mass arrays must be one-dimensional with equal shape.")
    return float(np.sum(np.abs(left - right)))


def cell_mass_cdf_sup(left, right):
    """Return maximum absolute difference between discrete cumulative masses."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("cell-mass arrays must be one-dimensional with equal shape.")
    return float(np.max(np.abs(np.cumsum(left) - np.cumsum(right))))


def cell_mass_wasserstein_1(left, right, dx):
    """Approximate 1D Wasserstein-1 distance on an equal-width cell grid."""
    dx = float(dx)
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be finite and strictly positive.")
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("cell-mass arrays must be one-dimensional with equal shape.")
    return float(dx * np.sum(np.abs(np.cumsum(left) - np.cumsum(right))))



def density_l2(left, right, dx):
    """Return the discrete L2 distance between two densities."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    dx = float(dx)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("density arrays must be one-dimensional with equal shape.")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("density arrays must contain finite values.")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be finite and strictly positive.")
    return float(np.sqrt(dx * np.sum(np.square(left - right))))


def beta_mixture_variance_floor(mean, degree):
    """Return the theoretical variance floor for a degree-M Beta mixture.

    For basis components Beta(j+1, M-j+1), any convex mixture with mean m
    satisfies Var(X) >= m(1-m)/(M+3).
    """
    mean = np.asarray(mean, dtype=float)
    if not np.all(np.isfinite(mean)) or np.any(mean < 0.0) or np.any(mean > 1.0):
        raise ValueError("mean must contain finite values in [0, 1].")
    if isinstance(degree, (bool, np.bool_)):
        raise TypeError("degree must be a non-negative integer.")
    try:
        degree = int(degree)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("degree must be a non-negative integer.") from exc
    if degree < 0:
        raise ValueError("degree must be a non-negative integer.")
    return mean * (1.0 - mean) / (degree + 3.0)
