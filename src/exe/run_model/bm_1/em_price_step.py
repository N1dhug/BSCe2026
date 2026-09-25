"""Simulate a bm_1 dynamic-price step with Euler-Maruyama and Monte Carlo density estimates."""

from pathlib import Path

from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder
import matplotlib.pyplot as plt
import numpy as np

# Configure the time grid in days; convert to hours only for plotting.
# The price jump at 12 h is on this hourly grid. General splitting is F2-B.
days = 4
T = (0.0, float(days))

t_steps = 24 * days + 1
t = np.linspace(T[0], T[1], t_steps)
hours = 24.0 * t

# Build the constant baseline.
inputs = input_builder()

B_data = (np.array(T, dtype=float), np.array([0.3, 0.3]))
B = inputs.energy(B_data)
A = B

# Build the dynamic-price step.
price_drop_hour = 12.0
price_drop_time = price_drop_hour / 24.0

# Reuse F1's right-continuous builder: N+1 edges for N prices.
price_edges = np.array([T[0], price_drop_time, T[1]])
price_values = np.array([0.75, 0.25])
U = inputs.price_intervals((price_edges, price_values))

# Build the monotone response functions.
n = 10
kappa = 5

poi_x, pr_x = inputs.md_vals(kappa, n)
poi_u, pr_u = inputs.md_vals(kappa, n)

f = inputs.md(poi_x, pr_x)
g = inputs.md(poi_u, pr_u)

# Configure the bm_1 parameters and initial state.
C = 1.0  # Normalized capacity in days.
k = 2.0
Xi = 1.0
sigma_x = 0.5  # Process-noise amplitude in day**(-1/2).

init_cond = [0.5]
parameters = [U, f, g, C, k, Xi]

# Simulate Euler-Maruyama paths.
model = bm_1(B, A, T, t_steps)

n_paths = 100_000
seed = 1

# Preserve the existing demonstration; this is NOT an uncorrected reference.
print('EM demonstration: boundary="reflect" is a numerical correction.')
print("All paths start at x0; the FP demo uses a spread-out beta initial law.")
print("Uncorrected proposals and matched initial distributions are F2-B/F2-D.")

em_t, Xt, Ut, Bt, fXt, gUt, delta_t, Dt = model.euler_maruyama_solver(init_cond, parameters, sigma_x, n_paths=n_paths, seed=seed, boundary="reflect")

em_hours = 24.0 * em_t

# Compute Monte Carlo moments.
mean = np.mean(Xt, axis=0)
variance = np.var(Xt, axis=0)
std = np.sqrt(variance)

# Estimate p(x, t) from the simulated paths.
n_x_bins = 100

x_edges = np.linspace(0.0, 1.0, n_x_bins + 1)
x_grid = 0.5 * (x_edges[:-1] + x_edges[1:])
dx = x_edges[1] - x_edges[0]

density = np.zeros((t_steps, n_x_bins), dtype=float)

for i in range(t_steps):
    density[i], _ = np.histogram(Xt[:, i], bins=x_edges, density=True)

# Check density normalization.
mass = np.sum(density, axis=1) * dx

# Build the diagnostic plots.
fig, (ax_pdf, ax_state, ax_paths, ax_terminal) = plt.subplots(nrows=4, figsize=(11, 15), gridspec_kw={"height_ratios": [2.0, 1.0, 1.4, 1.0]})

# Plot the empirical density.
heatmap_probability = density / np.max(density)

mesh = ax_pdf.pcolormesh(em_hours, x_grid, heatmap_probability.T, shading="auto", vmin=0.0, vmax=1.0,)

ax_pdf.plot(em_hours, mean, label="Monte-Carlo E[X(t)]")
ax_pdf.axvline(price_drop_hour, linestyle="--", label="Price drop")

ax_pdf.set_ylabel("State of charge x")
ax_pdf.set_title("Euler-Maruyama estimate of bm_1 probability density p(x,t)")
ax_pdf.set_ylim(0.0, 1.0)
ax_pdf.legend(loc="best")

fig.colorbar(mesh, ax=ax_pdf, label="Normalized probability density")

# Plot the price signal and distribution moments.
ax_state.step(em_hours, Ut, where="post", label="Price U(t)")
ax_state.plot(em_hours, mean, label="E[X(t)]")

ax_state.fill_between(em_hours, np.clip(mean - std, 0.0, 1.0), np.clip(mean + std, 0.0, 1.0), alpha=0.2, label="E[X] +/- std",)

ax_state.axvline(price_drop_hour, linestyle="--")

ax_state.set_ylabel("Price / SoC")
ax_state.set_title("Price drop and Euler-Maruyama distribution moments")
ax_state.set_ylim(0.0, 1.0)
ax_state.grid(True)
ax_state.legend(loc="best")

# Plot sample Euler-Maruyama trajectories.
n_paths_to_plot = min(50, n_paths)

for i in range(n_paths_to_plot):
    ax_paths.plot(em_hours, Xt[i], alpha=0.15)

ax_paths.plot(em_hours, mean, linewidth=2.5, label="E[X(t)]")
ax_paths.axvline(price_drop_hour, linestyle="--", label="Price drop")

ax_paths.set_ylabel("State of charge")
ax_paths.set_title(f"{n_paths_to_plot} Euler-Maruyama sample paths")
ax_paths.set_ylim(0.0, 1.0)
ax_paths.grid(True)
ax_paths.legend(loc="best")

# Plot the terminal density.
terminal_hour = em_hours[-1]
ax_terminal.plot(x_grid, density[-1], label=f"p(x, {terminal_hour:.0f} h)")
ax_terminal.axvline(mean[-1], linestyle="--", label=f"E[X({terminal_hour:.0f} h)]")

ax_terminal.set_xlabel("State of charge x")
ax_terminal.set_ylabel("Probability density")
ax_terminal.set_title("Terminal Euler-Maruyama probability density")
ax_terminal.grid(True)
ax_terminal.legend(loc="best")

# Save the figure.
fig.tight_layout()

# Resolve the output directory from the repository root.
graph_dir = Path(__file__).resolve().parents[4] / "files" / "output" / "graphs" / "bm_1"
graph_dir.mkdir(parents=True, exist_ok=True)
graph_path = graph_dir / "bm_1_em_price_step.png"

fig.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved graph to {graph_path}")
