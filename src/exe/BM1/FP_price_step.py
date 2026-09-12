# src/exe/BM1/FP_price_drop.py
# ------------------------------------------------------------
# BM1 stochastic response to a single price drop.
#
# The probability density p(x,t) is solved directly from the
# Fokker-Planck equation using a Beta-mixture approximation.
#
# This run uses the same BM1 setup as EM_price_drop.py
# so the two solvers can later be compared directly.
# ------------------------------------------------------------

from pathlib import Path

from src.models.BM1 import BM1
from src.models.Simulate import Simulate
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# 1) problem size and time grid
# ------------------------------------------------------------

days = 4
T = (0.0, float(days))

# 24 one-hour intervals per day + initial point.
t_steps = 24 * days + 1
t = np.linspace(T[0], T[1], t_steps)
hours = 24.0 * t


# ------------------------------------------------------------
# 2) baseline consumption
# ------------------------------------------------------------

sim = Simulate()

B_data = (np.array([0.0, 1.0]), np.array([0.3, 0.3]))
B = sim.energy(B_data)

# Available energy is not needed by the Fokker-Planck solver
# itself, but BM1 expects it when the model object is created.
A = B


# ------------------------------------------------------------
# 3) fixed price signal: 0.75 -> 0.25 at hour 12
# ------------------------------------------------------------

price_drop_hour = 12.0
price_drop_time = price_drop_hour / 24.0

U_data = (np.array([0.0, price_drop_time, 1.0]), np.array([0.75, 0.25, 0.25]))
U = sim.price(U_data)


# ------------------------------------------------------------
# 4) monotonically decreasing f and g
# ------------------------------------------------------------

n = 10
kappa = 5

poi_x, pr_x = sim.md_vals(kappa, n)
poi_u, pr_u = sim.md_vals(kappa, n)

f = sim.md(poi_x, pr_x)
g = sim.md(poi_u, pr_u)


# ------------------------------------------------------------
# 5) BM1 parameters and initial state
# ------------------------------------------------------------

C = 1.0
k = 2.0
Xi = 1.0
sigma_x = 0.5

init_cond = [0.5]
parameters = [U, f, g, C, k, Xi]


# ------------------------------------------------------------
# 6) Fokker-Planck / Beta-mixture settings
# ------------------------------------------------------------

# Beta-series degree M.
#
# The density is approximated by:
#
#        M
# p(x,t) = sum c_m(t) l_m(x)
#        m=0
#
# degree=20 gives 21 Beta basis functions.
degree = 20

# Number of finite-volume cells on x in [0, 1].
x_steps = 201

# CFL stability factor for the internal explicit PDE solver.
cfl = 0.45


# ------------------------------------------------------------
# 7) Fokker-Planck simulation
# ------------------------------------------------------------

model = BM1(B, A, T, t_steps)

fp_t, x_grid, density, coefficients, mean, Ut, Bt, expected_Dt = model.fokker_planc_solver(
    init_cond=init_cond,
    parameters=parameters,
    sigma_x=sigma_x,
    degree=degree,
    x_steps=x_steps,
    cfl=cfl,
)

fp_hours = 24.0 * fp_t


# ------------------------------------------------------------
# 8) probability-density moments
# ------------------------------------------------------------

# FokkerPlanc uses cell centres on [0, 1].
dx = 1.0 / len(x_grid)

# E[X^2]
second_moment = np.sum(density * (x_grid[None, :] ** 2), axis=1) * dx

# Var[X] = E[X^2] - E[X]^2
variance = second_moment - mean**2

# Protect against tiny negative values caused by
# floating-point roundoff.
variance = np.maximum(variance, 0.0)

std = np.sqrt(variance)


# ------------------------------------------------------------
# 9) probability-density checks
# ------------------------------------------------------------

# Integral p(x,t) dx should remain approximately one.
mass = np.sum(density, axis=1) * dx

# Because the Beta basis functions are normalized probability
# densities, the mixture coefficients should also sum to one.
coefficient_mass = np.sum(coefficients, axis=1)

print("Fokker-Planck probability checks")
print("--------------------------------")
print(f"Minimum density mass:       {np.min(mass):.12f}")
print(f"Maximum density mass:       {np.max(mass):.12f}")
print(f"Maximum |mass - 1|:         {np.max(np.abs(mass - 1.0)):.3e}")

print()

print(f"Minimum coefficient sum:    {np.min(coefficient_mass):.12f}")
print(f"Maximum coefficient sum:    {np.max(coefficient_mass):.12f}")
print(f"Maximum |sum(c) - 1|:       {np.max(np.abs(coefficient_mass - 1.0)):.3e}")

print()

print(f"Minimum coefficient:        {np.min(coefficients):.3e}")
print(f"Minimum density value:      {np.min(density):.3e}")


# ------------------------------------------------------------
# 10) plots
# ------------------------------------------------------------

fig, (ax_pdf, ax_state, ax_coefficients, ax_terminal) = plt.subplots(
    nrows=4,
    figsize=(11, 15),
    gridspec_kw={"height_ratios": [2.0, 1.0, 1.4, 1.0]},
)


# ------------------------------------------------------------
# Plot 1: Fokker-Planck probability density p(x,t)
# ------------------------------------------------------------

heatmap_probability = density / np.max(density)

mesh = ax_pdf.pcolormesh(
    fp_hours,
    x_grid,
    heatmap_probability.T,
    shading="auto",
    vmin=0.0,
    vmax=1.0,
)

ax_pdf.plot(fp_hours, mean, linewidth=2.0, label="Fokker-Planck E[X(t)]")
ax_pdf.axvline(price_drop_hour, linestyle="--", label="Price drop")

ax_pdf.set_ylabel("State of charge x")
ax_pdf.set_title("BM1 Fokker-Planck probability density p(x,t)")
ax_pdf.set_ylim(0.0, 1.0)
ax_pdf.legend(loc="best")

fig.colorbar(mesh, ax=ax_pdf, label="Normalized probability density")


# ------------------------------------------------------------
# Plot 2: price signal and distribution moments
# ------------------------------------------------------------

ax_state.step(fp_hours, Ut, where="post", label="Price U(t)")
ax_state.plot(fp_hours, mean, linewidth=2.0, label="E[X(t)]")
ax_state.fill_between(
    fp_hours,
    np.clip(mean - std, 0.0, 1.0),
    np.clip(mean + std, 0.0, 1.0),
    alpha=0.2,
    label="E[X] +/- std",
)

ax_state.axvline(price_drop_hour, linestyle="--")

ax_state.set_ylabel("Price / SoC")
ax_state.set_title("Price drop and Fokker-Planck distribution moments")
ax_state.set_ylim(0.0, 1.0)
ax_state.grid(True)
ax_state.legend(loc="best")


# ------------------------------------------------------------
# Plot 3: Beta-mixture coefficient evolution
# ------------------------------------------------------------

for m in range(coefficients.shape[1]):
    ax_coefficients.plot(fp_hours, coefficients[:, m], label=f"c{m}")

ax_coefficients.axvline(price_drop_hour, linestyle="--")

ax_coefficients.set_ylabel("Coefficient value")
ax_coefficients.set_title(f"Beta-mixture coefficients, degree M = {degree}")
ax_coefficients.set_ylim(bottom=0.0)
ax_coefficients.grid(True)
ax_coefficients.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    ncol=2,
    fontsize=8,
)


# ------------------------------------------------------------
# Plot 4: terminal probability density
# ------------------------------------------------------------

terminal_hour = fp_hours[-1]

ax_terminal.plot(
    x_grid,
    density[-1],
    linewidth=2.0,
    label=f"p(x, {terminal_hour:.0f} h)",
)
ax_terminal.axvline(
    mean[-1],
    linestyle="--",
    label=f"E[X({terminal_hour:.0f} h)]",
)

ax_terminal.set_xlabel("State of charge x")
ax_terminal.set_ylabel("Probability density")
ax_terminal.set_title("Terminal Fokker-Planck probability density")
ax_terminal.set_xlim(0.0, 1.0)
ax_terminal.grid(True)
ax_terminal.legend(loc="best")


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

fig.tight_layout()

# This file lives in src/exe/BM1, so parents[3] is the repository root.
graph_dir = Path(__file__).resolve().parents[3] / "files" / "output" / "graphs"
graph_dir.mkdir(parents=True, exist_ok=True)
graph_path = graph_dir / "BM1_FP_price_step.png"

fig.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved graph to {graph_path}")
