"""Pure finite-volume Fokker-Planck reference for stochastic bm_1.

This F2-C experiment reuses the BM-only shared F2 reference case. It evolves
cell probability masses directly and does not use beta projection, clipping,
or renormalization.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.exe.run_model.bm_1.f2_reference_case import (
    MAX_STEP,
    PRICE_BREAKPOINTS,
    PRICE_DROP_HOUR,
    PRICE_DROP_TIME,
    SIGMA_X,
    T_EVAL,
    X_STEPS,
    beta_initial_cell_mass,
    build_reference_case,
)


model, parameters = build_reference_case()
x_edges, _, _ = model.fokker_planc_reference.grid(X_STEPS)
initial_mass = beta_initial_cell_mass(x_edges)

(
    t,
    x,
    density,
    cell_mass,
    mean,
    variance,
    price,
    baseline,
    expected_demand,
    diagnostics,
) = model.fokker_planc_reference_solver(
    initial_mass,
    parameters,
    SIGMA_X,
    t_eval=T_EVAL,
    breakpoints=PRICE_BREAKPOINTS,
    max_step=MAX_STEP,
)

mass = np.sum(cell_mass, axis=1)
std = np.sqrt(variance)
hours = 24.0 * t

print("F2-C pure finite-volume Fokker-Planck reference")
print("-----------------------------------------------")
print(f"Spatial cells:                  {X_STEPS}")
print(f"Output points:                  {len(t)}")
print(f"Internal implicit-Euler steps:  {diagnostics['n_steps']}")
print(f"Maximum internal step [h]:      {24.0 * np.max(np.diff(diagnostics['integration_t'])):.6f}")
print(f"Price breakpoint included:      {bool(np.any(diagnostics['integration_t'] == PRICE_DROP_TIME))}")
print(f"Maximum |mass - 1|:             {np.max(np.abs(mass - 1.0)):.3e}")
print(f"Minimum cell probability:       {np.min(cell_mass):.3e}")
print(f"Minimum density value:          {np.min(density):.3e}")
print(f"Max generator column-sum error: {diagnostics['maximum_abs_generator_column_sum']:.3e}")
print(f"Minimum off-diagonal rate:      {diagnostics['minimum_off_diagonal_rate']:.3e}")
print(f"Clipping used:                  {diagnostics['used_clipping']}")
print(f"Renormalization used:           {diagnostics['used_renormalization']}")
print()
print(f"Initial E[X]:                   {mean[0]:.6f}")
print(f"Terminal E[X]:                  {mean[-1]:.6f}")
print(f"Terminal std[X]:                {std[-1]:.6f}")
print(f"Terminal E[D]:                  {expected_demand[-1]:.6f}")

if diagnostics["used_clipping"] or diagnostics["used_renormalization"]:
    raise RuntimeError("The F2-C reference must not use post-correction.")
if diagnostics["minimum_cell_mass"] < -1.0e-12:
    raise RuntimeError("The F2-C reference produced a negative cell probability.")
if diagnostics["maximum_abs_mass_error"] > 1.0e-10:
    raise RuntimeError("The F2-C reference failed to conserve total probability.")

fig, (ax_pdf, ax_state, ax_terminal) = plt.subplots(
    nrows=3,
    figsize=(11, 11),
    gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]},
)

normalized_density = density / np.max(density)
mesh = ax_pdf.pcolormesh(
    hours,
    x,
    normalized_density.T,
    shading="auto",
    vmin=0.0,
    vmax=1.0,
)
ax_pdf.plot(hours, mean, linewidth=2.0, label="Finite-volume E[X(t)]")
ax_pdf.axvline(PRICE_DROP_HOUR, linestyle="--", label="Price drop")
ax_pdf.set_ylabel("State of charge x")
ax_pdf.set_title("Pure finite-volume Fokker-Planck density")
ax_pdf.set_ylim(0.0, 1.0)
ax_pdf.legend(loc="best")
fig.colorbar(mesh, ax=ax_pdf, label="Normalized probability density")

ax_state.step(hours, price, where="post", label="Price U(t)")
ax_state.plot(hours, mean, linewidth=2.0, label="E[X(t)]")
ax_state.fill_between(
    hours,
    np.clip(mean - std, 0.0, 1.0),
    np.clip(mean + std, 0.0, 1.0),
    alpha=0.2,
    label="E[X] +/- std",
)
ax_state.axvline(PRICE_DROP_HOUR, linestyle="--")
ax_state.set_ylabel("Price / SoC")
ax_state.set_ylim(0.0, 1.0)
ax_state.set_title("Price and finite-volume state moments")
ax_state.grid(True)
ax_state.legend(loc="best")

ax_terminal.plot(x, density[0], label="Initial density")
ax_terminal.plot(x, density[-1], linewidth=2.0, label="Terminal density")
ax_terminal.axvline(mean[-1], linestyle="--", label="Terminal E[X]")
ax_terminal.set_xlabel("State of charge x")
ax_terminal.set_ylabel("Probability density")
ax_terminal.set_title("Initial and terminal finite-volume density")
ax_terminal.set_xlim(0.0, 1.0)
ax_terminal.grid(True)
ax_terminal.legend(loc="best")

fig.tight_layout()
graph_dir = Path(__file__).resolve().parents[4] / "files" / "output" / "graphs" / "bm_1"
graph_dir.mkdir(parents=True, exist_ok=True)
graph_path = graph_dir / "bm_1_fp_reference.png"
fig.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved graph to {graph_path}")
