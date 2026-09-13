"""Simulate a BM1 dynamic-price step with the Fokker-Planck Beta-mixture method."""

from pathlib import Path

from src.model.BM1 import BM1
from src.exe.tool.InputBuilder import InputBuilder
import matplotlib.pyplot as plt
import numpy as np

# Configure the time grid.
days = 4
T = (0.0, float(days))

t_steps = 24 * days + 1
t = np.linspace(T[0], T[1], t_steps)
hours = 24.0 * t

# Build the constant baseline.
inputs = InputBuilder()

B_data = (np.array([0.0, 1.0]), np.array([0.3, 0.3]))
B = inputs.energy(B_data)

# Reuse the baseline as available energy; the Fokker-Planck solver does not use A directly.
A = B

# Build the dynamic-price step.
price_drop_hour = 12.0
price_drop_time = price_drop_hour / 24.0

U_data = (np.array([0.0, price_drop_time, 1.0]), np.array([0.75, 0.25, 0.25]))
U = inputs.price(U_data)

# Build the monotone response functions.
n = 10
kappa = 5

poi_x, pr_x = inputs.md_vals(kappa, n)
poi_u, pr_u = inputs.md_vals(kappa, n)

f = inputs.md(poi_x, pr_x)
g = inputs.md(poi_u, pr_u)

# Configure the BM1 parameters and initial state.
C = 1.0
k = 2.0
Xi = 1.0
sigma_x = 0.5

init_cond = [0.5]
parameters = [U, f, g, C, k, Xi]

# Configure the Beta-mixture Fokker-Planck approximation.
degree = 20

x_steps = 201

cfl = 0.45

# Solve the Fokker-Planck system.
model = BM1(B, A, T, t_steps)

fp_t, x_grid, density, coefficients, mean, Ut, Bt, expected_Dt = model.fokker_planc_solver(init_cond=init_cond, parameters=parameters, sigma_x=sigma_x, degree=degree, x_steps=x_steps, cfl=cfl,)

fp_hours = 24.0 * fp_t

# Compute density moments on the finite-volume grid.
dx = 1.0 / len(x_grid)

second_moment = np.sum(density * (x_grid[None, :] ** 2), axis=1) * dx

variance = second_moment - mean**2

# Clamp roundoff-scale negative variances to zero.
variance = np.maximum(variance, 0.0)

std = np.sqrt(variance)

# Check density and mixture-weight normalization.
mass = np.sum(density, axis=1) * dx

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

# Build the diagnostic plots.
fig, (ax_pdf, ax_state, ax_coefficients, ax_terminal) = plt.subplots(nrows=4, figsize=(11, 15), gridspec_kw={"height_ratios": [2.0, 1.0, 1.4, 1.0]},)

# Plot the Fokker-Planck density.
heatmap_probability = density / np.max(density)

mesh = ax_pdf.pcolormesh(fp_hours, x_grid, heatmap_probability.T, shading="auto", vmin=0.0, vmax=1.0,)

ax_pdf.plot(fp_hours, mean, linewidth=2.0, label="Fokker-Planck E[X(t)]")
ax_pdf.axvline(price_drop_hour, linestyle="--", label="Price drop")

ax_pdf.set_ylabel("State of charge x")
ax_pdf.set_title("BM1 Fokker-Planck probability density p(x,t)")
ax_pdf.set_ylim(0.0, 1.0)
ax_pdf.legend(loc="best")

fig.colorbar(mesh, ax=ax_pdf, label="Normalized probability density")

# Plot the price signal and distribution moments.
ax_state.step(fp_hours, Ut, where="post", label="Price U(t)")
ax_state.plot(fp_hours, mean, linewidth=2.0, label="E[X(t)]")
ax_state.fill_between(fp_hours, np.clip(mean - std, 0.0, 1.0), np.clip(mean + std, 0.0, 1.0), alpha=0.2, label="E[X] +/- std",)

ax_state.axvline(price_drop_hour, linestyle="--")

ax_state.set_ylabel("Price / SoC")
ax_state.set_title("Price drop and Fokker-Planck distribution moments")
ax_state.set_ylim(0.0, 1.0)
ax_state.grid(True)
ax_state.legend(loc="best")

# Plot Beta-mixture coefficient evolution.
for m in range(coefficients.shape[1]):
    ax_coefficients.plot(fp_hours, coefficients[:, m], label=f"c{m}")

ax_coefficients.axvline(price_drop_hour, linestyle="--")

ax_coefficients.set_ylabel("Coefficient value")
ax_coefficients.set_title(f"Beta-mixture coefficients, degree M = {degree}")
ax_coefficients.set_ylim(bottom=0.0)
ax_coefficients.grid(True)
ax_coefficients.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=2, fontsize=8,)

# Plot the terminal density.
terminal_hour = fp_hours[-1]

ax_terminal.plot(x_grid, density[-1], linewidth=2.0, label=f"p(x, {terminal_hour:.0f} h)",)
ax_terminal.axvline(mean[-1], linestyle="--", label=f"E[X({terminal_hour:.0f} h)]",)

ax_terminal.set_xlabel("State of charge x")
ax_terminal.set_ylabel("Probability density")
ax_terminal.set_title("Terminal Fokker-Planck probability density")
ax_terminal.set_xlim(0.0, 1.0)
ax_terminal.grid(True)
ax_terminal.legend(loc="best")

# Save the figure.
fig.tight_layout()

# Resolve the output directory from the repository root.
graph_dir = Path(__file__).resolve().parents[3] / "files" / "output" / "graphs"
graph_dir.mkdir(parents=True, exist_ok=True)
graph_path = graph_dir / "BM1_FP_price_step.png"

fig.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved graph to {graph_path}")
