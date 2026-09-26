"""Controlled Euler-Maruyama / Monte Carlo reference for stochastic bm_1.

This F2-B experiment separates output times from internal EM steps, inserts
known price discontinuities as integration boundaries, supplies Wiener
increments explicitly, and reports raw boundary proposals before any numerical
correction.  It intentionally uses boundary="none".

Run from the repository root:

    python -B -m src.exe.run_model.bm_1.em_reference
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder


# Time is measured in days. Return hourly output, but integrate at most every
# 15 minutes. The 12.5 h price change is deliberately not an output time.
T = (0.0, 1.0)
T_EVAL = np.linspace(T[0], T[1], 25)
MAX_STEP = 1.0 / (24.0 * 4.0)
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
N_PATHS = 20_000
SEED = 2026

# Build inputs with the same project helpers used by the deterministic model.
inputs = input_builder()
B = inputs.energy((np.array(T), np.array([BASELINE, BASELINE])))
A = B
U = inputs.price_intervals((PRICE_EDGES, PRICE_VALUES))
poi_x, pr_x = inputs.md_vals(5, 10)
poi_u, pr_u = inputs.md_vals(5, 10)
f = inputs.md(poi_x, pr_x)
g = inputs.md(poi_u, pr_u)
parameters = [U, f, g, C, K, XI]

# Build the internal grid explicitly so the same grid can be used to generate
# and, in later convergence studies, couple Brownian increments.
model = bm_1(B, A, T, len(T_EVAL))
integration_t, output_indices = model.euler_maruyama.build_time_grid(
    T_EVAL,
    breakpoints=PRICE_BREAKPOINTS,
    max_step=MAX_STEP,
)
dW = model.euler_maruyama.brownian_increments(
    integration_t,
    n_paths=N_PATHS,
    seed=SEED,
)

result = model.euler_maruyama_solver(
    [X0],
    parameters,
    SIGMA_X,
    n_paths=N_PATHS,
    boundary="none",
    t_eval=T_EVAL,
    breakpoints=PRICE_BREAKPOINTS,
    max_step=MAX_STEP,
    brownian_increments=dW,
    return_diagnostics=True,
)

(
    t,
    state,
    price,
    baseline,
    _,
    _,
    _,
    demand,
    diagnostics,
) = result

# Monte Carlo summaries. Standard errors quantify sampling uncertainty only;
# they do not include Euler-Maruyama time-discretization error.
state_mean = np.mean(state, axis=0)
state_var = np.var(state, axis=0, ddof=1)
state_se = np.sqrt(state_var / N_PATHS)
demand_mean = np.mean(demand, axis=0)
demand_var = np.var(demand, axis=0, ddof=1)
demand_se = np.sqrt(demand_var / N_PATHS)

print("F2-B controlled Euler-Maruyama reference")
print("----------------------------------------")
print(f"Output points:                 {len(t)}")
print(f"Internal EM steps:             {diagnostics['n_steps']}")
print(f"Maximum internal step [h]:     {24.0 * np.max(np.diff(diagnostics['integration_t'])):.6f}")
print(f"Price breakpoint included:     {bool(np.any(diagnostics['integration_t'] == PRICE_DROP_TIME))}")
print(f"Raw proposal minimum:          {diagnostics['minimum_raw_proposal']:.6f}")
print(f"Raw proposal maximum:          {diagnostics['maximum_raw_proposal']:.6f}")
print(f"Lower-bound violations:        {diagnostics['lower_violations']}")
print(f"Upper-bound violations:        {diagnostics['upper_violations']}")
print(f"Paths with any violation:      {diagnostics['violating_paths']}")
print(f"Boundary corrections applied:  {diagnostics['boundary_corrections']}")
print()
print(f"Terminal E[X]:                 {state_mean[-1]:.6f}")
print(f"Terminal SE(E[X]):             {state_se[-1]:.3e}")
print(f"Terminal E[D]:                 {demand_mean[-1]:.6f}")
print(f"Terminal SE(E[D]):             {demand_se[-1]:.3e}")

if diagnostics["total_violations"]:
    raise RuntimeError(
        "The uncorrected EM reference left [0, 1]. Reduce max_step or noise "
        "before using this run as a bounded-state reference."
    )

# Build a compact diagnostic figure.
hours = 24.0 * t
fig, (ax_state, ax_paths, ax_demand) = plt.subplots(nrows=3, figsize=(11, 11))

ax_state.step(hours, price, where="post", label="Price U(t)")
ax_state.plot(hours, state_mean, linewidth=2.0, label="Monte Carlo E[X(t)]")
ax_state.fill_between(
    hours,
    state_mean - 1.96 * state_se,
    state_mean + 1.96 * state_se,
    alpha=0.2,
    label="95% MC interval for E[X]",
)
ax_state.axvline(PRICE_DROP_HOUR, linestyle="--", label="Price drop")
ax_state.set_ylabel("Price / SoC")
ax_state.set_title("Controlled Euler-Maruyama mean and sampling uncertainty")
ax_state.grid(True)
ax_state.legend(loc="best")

for path in state[:30]:
    ax_paths.plot(hours, path, alpha=0.2)
ax_paths.plot(hours, state_mean, linewidth=2.5, label="E[X(t)]")
ax_paths.axvline(PRICE_DROP_HOUR, linestyle="--")
ax_paths.set_ylabel("State of charge")
ax_paths.set_title("30 uncorrected Euler-Maruyama sample paths")
ax_paths.set_ylim(0.0, 1.0)
ax_paths.grid(True)
ax_paths.legend(loc="best")

ax_demand.plot(hours, demand_mean, linewidth=2.0, label="Monte Carlo E[D(t)]")
ax_demand.fill_between(
    hours,
    demand_mean - 1.96 * demand_se,
    demand_mean + 1.96 * demand_se,
    alpha=0.2,
    label="95% MC interval for E[D]",
)
ax_demand.plot(hours, baseline, linestyle="--", label="Baseline B(t)")
ax_demand.axvline(PRICE_DROP_HOUR, linestyle="--")
ax_demand.set_xlabel("Time [h]")
ax_demand.set_ylabel("Normalized demand")
ax_demand.set_title("Expected demand and Monte Carlo sampling uncertainty")
ax_demand.grid(True)
ax_demand.legend(loc="best")

fig.tight_layout()
graph_dir = Path(__file__).resolve().parents[4] / "files" / "output" / "graphs" / "bm_1"
graph_dir.mkdir(parents=True, exist_ok=True)
graph_path = graph_dir / "bm_1_em_reference.png"
fig.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# The public wrapper must return the requested output grid, not the internal one.
np.testing.assert_array_equal(t, T_EVAL)
np.testing.assert_array_equal(diagnostics["integration_t"][output_indices], T_EVAL)
print(f"Saved graph to {graph_path}")
