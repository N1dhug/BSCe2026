# src/exe/BM1/EM_price_drop.py
# ------------------------------------------------------------
# BM1 stochastic response to a single price drop.
#
# The stochastic SDE is solved using Euler-Maruyama.
# The probability density p(x,t) is estimated from many
# Monte-Carlo trajectories.
# ------------------------------------------------------------

from pathlib import Path

from src.model.BM1 import BM1
from src.exe.tool.InputBuilder import InputBuilder
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# 1) problem size and time grid
# ------------------------------------------------------------

days = 4
T = (0.0, float(days))

# 24 one-hour intervals -> 25 points including both endpoints.
t_steps = 24 * days + 1
t = np.linspace(T[0], T[1], t_steps)
hours = 24.0 * t


# ------------------------------------------------------------
# 2) baseline consumption
# ------------------------------------------------------------

inputs = InputBuilder()

B_data = (np.array([0.0, 1.0]), np.array([0.3, 0.3]))
B = inputs.energy(B_data)
A = B


# ------------------------------------------------------------
# 3) fixed price signal: 0.75 -> 0.25 at hour 12
# ------------------------------------------------------------

price_drop_hour = 12.0
price_drop_time = price_drop_hour / 24.0

U_data = (np.array([0.0, price_drop_time, 1.0]), np.array([0.75, 0.25, 0.25]))
U = inputs.price(U_data)


# ------------------------------------------------------------
# 4) monotonically decreasing f and g
# ------------------------------------------------------------

n = 10
kappa = 5

poi_x, pr_x = inputs.md_vals(kappa, n)
poi_u, pr_u = inputs.md_vals(kappa, n)

f = inputs.md(poi_x, pr_x)
g = inputs.md(poi_u, pr_u)


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
# 6) Euler-Maruyama simulation
# ------------------------------------------------------------

model = BM1(B, A, T, t_steps)

n_paths = 100_000
seed = 1

em_t, Xt, Ut, Bt, fXt, gUt, delta_t, Dt = model.euler_maruyama_solver(
    init_cond, parameters, sigma_x, n_paths=n_paths, seed=seed, boundary="reflect"
)

em_hours = 24.0 * em_t


# ------------------------------------------------------------
# 7) Monte-Carlo mean and variance
# ------------------------------------------------------------

mean = np.mean(Xt, axis=0)
variance = np.var(Xt, axis=0)
std = np.sqrt(variance)


# ------------------------------------------------------------
# 8) estimate p(x,t) from Euler-Maruyama trajectories
# ------------------------------------------------------------

n_x_bins = 100

x_edges = np.linspace(0.0, 1.0, n_x_bins + 1)
x_grid = 0.5 * (x_edges[:-1] + x_edges[1:])
dx = x_edges[1] - x_edges[0]

density = np.zeros((t_steps, n_x_bins), dtype=float)

for i in range(t_steps):
    density[i], _ = np.histogram(Xt[:, i], bins=x_edges, density=True)


# ------------------------------------------------------------
# 9) probability-density checks
# ------------------------------------------------------------
mass = np.sum(density, axis=1) * dx


# ------------------------------------------------------------
# 10) plots
# ------------------------------------------------------------

fig, (ax_pdf, ax_state, ax_paths, ax_terminal) = plt.subplots(
    nrows=4, figsize=(11, 15), gridspec_kw={"height_ratios": [2.0, 1.0, 1.4, 1.0]}
)


# ------------------------------------------------------------
# Plot 1: empirical probability density p(x,t)
# ------------------------------------------------------------

heatmap_probability = density / np.max(density)

mesh = ax_pdf.pcolormesh(
    em_hours,
    x_grid,
    heatmap_probability.T,
    shading="auto",
    vmin=0.0,
    vmax=1.0,
)

ax_pdf.plot(em_hours, mean, label="Monte-Carlo E[X(t)]")
ax_pdf.axvline(price_drop_hour, linestyle="--", label="Price drop")

ax_pdf.set_ylabel("State of charge x")
ax_pdf.set_title("Euler-Maruyama estimate of BM1 probability density p(x,t)")
ax_pdf.set_ylim(0.0, 1.0)
ax_pdf.legend(loc="best")

fig.colorbar(mesh, ax=ax_pdf, label="Normalized probability density")


# ------------------------------------------------------------
# Plot 2: price signal and distribution moments
# ------------------------------------------------------------

ax_state.step(em_hours, Ut, where="post", label="Price U(t)")
ax_state.plot(em_hours, mean, label="E[X(t)]")

ax_state.fill_between(
    em_hours,
    np.clip(mean - std, 0.0, 1.0),
    np.clip(mean + std, 0.0, 1.0),
    alpha=0.2,
    label="E[X] +/- std",
)

ax_state.axvline(price_drop_hour, linestyle="--")

ax_state.set_ylabel("Price / SoC")
ax_state.set_title("Price drop and Euler-Maruyama distribution moments")
ax_state.set_ylim(0.0, 1.0)
ax_state.grid(True)
ax_state.legend(loc="best")


# ------------------------------------------------------------
# Plot 3: individual Euler-Maruyama trajectories
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Plot 4: terminal probability density
# ------------------------------------------------------------

ax_terminal.plot(x_grid, density[-1], label="p(x, 24 h)")
ax_terminal.axvline(mean[-1], linestyle="--", label="E[X(24 h)]")

ax_terminal.set_xlabel("State of charge x")
ax_terminal.set_ylabel("Probability density")
ax_terminal.set_title("Terminal Euler-Maruyama probability density")
ax_terminal.grid(True)
ax_terminal.legend(loc="best")


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

fig.tight_layout()

# This file lives in src/exe/BM1, so parents[3] is the repository root.
graph_dir = Path(__file__).resolve().parents[3] / "data" / "graphs"
graph_dir.mkdir(parents=True, exist_ok=True)
graph_path = graph_dir / "BM1_EM_price_step.png"

fig.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved graph to {graph_path}")
