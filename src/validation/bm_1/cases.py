"""Controlled BM1 reference inputs; no simulations or file I/O at import."""

import numpy as np
from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder

T = (0.0, 1.0)
MAX_STEP = 1.0 / (24.0 * 4.0)
PRICE_DROP_HOUR = 12.5
PRICE_VALUES = np.array([0.75, 0.25], dtype=float)
BASELINE = 0.3
X0 = 0.5
C = 1.0
K = 2.0
XI = 1.0
SIGMA_X = 0.5
INITIAL_ALPHA = 20.0
INITIAL_BETA = 20.0
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
    return (model, parameters)

def linear_state_response(x):
    """Benchmark state response f(x) = 1 - 2x."""
    return 1.0 - 2.0 * np.asarray(x, dtype=float)

def linear_price_response(u):
    """Benchmark price response g(u) = 1 - 2u."""
    return 1.0 - 2.0 * np.asarray(u, dtype=float)
T_EVAL = np.linspace(T[0], T[1], 25)
PRICE_DROP_TIME = PRICE_DROP_HOUR / 24.0
PRICE_EDGES = np.array([T[0], PRICE_DROP_TIME, T[1]], dtype=float)
PRICE_BREAKPOINTS = PRICE_EDGES[1:-1]
