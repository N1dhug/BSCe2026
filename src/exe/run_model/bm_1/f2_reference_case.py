"""Shared controlled reference case for stochastic BM1 experiments.

This module is intentionally BM-only.  It centralizes the F2 reference inputs
used by the Euler-Maruyama, finite-volume and MC-vs-FV experiments without
changing shared method/solver/tool layers used by Water Models.
"""

import numpy as np
from scipy.special import betainc

from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder


# Common F2 reference horizon and output grid.
T = (0.0, 1.0)
T_EVAL = np.linspace(T[0], T[1], 25)
MAX_STEP = 1.0 / (24.0 * 4.0)

# Deliberately non-hourly price discontinuity.
PRICE_DROP_HOUR = 12.5
PRICE_DROP_TIME = PRICE_DROP_HOUR / 24.0
PRICE_EDGES = np.array([T[0], PRICE_DROP_TIME, T[1]], dtype=float)
PRICE_VALUES = np.array([0.75, 0.25], dtype=float)
PRICE_BREAKPOINTS = PRICE_EDGES[1:-1]

# BM1 parameters.
BASELINE = 0.3
X0 = 0.5
C = 1.0
K = 2.0
XI = 1.0
SIGMA_X = 0.5

# Shared smooth initial law used when comparing Monte Carlo and finite volume.
INITIAL_ALPHA = 20.0
INITIAL_BETA = 20.0

# Default finite-volume comparison grid.
X_STEPS = 201


def build_reference_case():
    """Return the controlled BM1 model and parameter list."""
    inputs = input_builder()
    B = inputs.energy((np.array(T), np.array([BASELINE, BASELINE])))
    U = inputs.price_intervals((PRICE_EDGES, PRICE_VALUES))

    poi_x, pr_x = inputs.md_vals(5, 10)
    poi_u, pr_u = inputs.md_vals(5, 10)
    f = inputs.md(poi_x, pr_x)
    g = inputs.md(poi_u, pr_u)

    parameters = [U, f, g, C, K, XI]
    model = bm_1(B, B, T, len(T_EVAL))
    return model, parameters


def beta_initial_cell_mass(
    edges,
    alpha=INITIAL_ALPHA,
    beta=INITIAL_BETA,
):
    """Return exact Beta(alpha,beta) probability in each supplied cell."""
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges must be a one-dimensional array with at least two values.")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("edges must be finite and strictly increasing.")
    if edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("Beta initial-law edges must span [0, 1].")

    alpha = float(alpha)
    beta = float(beta)
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0.0 or beta <= 0.0:
        raise ValueError("Beta shape parameters must be finite and strictly positive.")

    cdf = betainc(alpha, beta, edges)
    mass = np.diff(cdf)
    if np.any(mass < 0.0):
        raise RuntimeError("Beta CDF produced negative cell probability.")
    if not np.isclose(np.sum(mass), 1.0, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("Beta cell probabilities do not sum to one.")
    return mass


def sample_beta_initial_states(
    n_paths,
    seed,
    alpha=INITIAL_ALPHA,
    beta=INITIAL_BETA,
):
    """Sample Monte Carlo initial states from the shared F2-D initial law."""
    if isinstance(n_paths, (bool, np.bool_)):
        raise TypeError("n_paths must be a positive integer.")
    original_n_paths = n_paths
    try:
        n_paths = int(n_paths)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("n_paths must be a positive integer.") from exc
    if float(n_paths) != float(original_n_paths):
        raise ValueError("n_paths must be an integer, not a fractional value.")
    if n_paths < 1:
        raise ValueError("n_paths must be a positive integer.")

    alpha = float(alpha)
    beta = float(beta)
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0.0 or beta <= 0.0:
        raise ValueError("Beta shape parameters must be finite and strictly positive.")

    rng = np.random.default_rng(seed)
    return rng.beta(alpha, beta, size=n_paths)
