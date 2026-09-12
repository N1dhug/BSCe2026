# src/exe/WM2/FP_price_step.py
# WM2 stochastic response to a single dynamic-price step.

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.model.WM2 import WM2
from src.tool.Simulate import Simulate


# 1) Time grid: four days, hourly reporting.
days = 10
t_interval = (0.0, float(days))
t_steps = 24 * days + 1
t = np.linspace(t_interval[0], t_interval[1], t_steps)
hours = 24.0 * t

sim = Simulate()

# 2) Exogenous signals. Keep the synthetic baselines constant so the response
# visible after hour 24 is caused by the single price step.
B_O_level = 0.30
B_I_level = 0.30
PE_level = 0.50
B_O = sim.energy((np.array([0.0, days]), np.array([B_O_level, B_O_level])))
B_I = sim.energy((np.array([0.0, days]), np.array([B_I_level, B_I_level])))
PE = sim.price((np.array([0.0, days]), np.array([PE_level, PE_level])))

# 3) Single price step: high price -> low price at hour 24.
price_before = 0.75
price_after = 0.25
price_step_hour = 24.0
price_step_time = price_step_hour / 24.0
U = sim.price(
    (
        np.array([0.0, price_step_time, days]),
        np.array([price_before, price_after, price_after]),
    )
)

# 4) Synthetic WM2 consumer-group parameters.
#
# The simplex coefficient vectors define
#
#     P_O = [B_O, V, 1-U]
#     P_I = [B_I, 1-V, O, 1-U]
#     P_A = [1-U, I, V, 1-PE]
#
#     O = beta^T  P_O
#     I = alpha^T P_I
#     A = gamma^T P_A.
#
alpha = np.array([0, 0.1, 0.1, 0.8])
beta = np.array([0.1, 0.85, 0.05])
gamma = np.array([0.25, 0.25, 0.25, 0.25])

kappa_R = 3.46
sigma_0 = 0.1
sigma_O = 0.5

parameters = {
    "U": U,
    "B_O": B_O,
    "B_I": B_I,
    "PE": PE,
    "alpha": alpha,
    "beta": beta,
    "gamma": gamma,
    "kappa_R": kappa_R,
    "sigma_0": sigma_0,
    "sigma_O": sigma_O,
}

model = WM2(t_interval, t_steps)

# 5) Start at the expected equilibrium associated with the pre-step price.
V0 = model.expected_equilibrium(B_O_level, B_I_level, price_before, alpha, beta)
R0 = 0.0
init_cond = [V0*2, R0]

# Computational bounds are numerical truncation only; V=0 and V=1 are not
# pathwise boundaries in WM2. The selected V domain also keeps O >= 0 for the
# synthetic signals used in this executable.
v_bounds = (-0.10, 1.10)
r_bounds = (-0.40, 0.40)
v_steps = 101
r_steps = 101
cfl = 0.80
initial_std = (0.025, 0.015)

# 6) Solve deterministic mean dynamics and the full two-dimensional
# Fokker-Planck equation for p(V, R, t).
deterministic = model.deterministic_solver(init_cond, parameters)
fp = model.fokker_planc_solver(init_cond=init_cond, parameters=parameters, initial_std=initial_std, v_bounds=v_bounds, r_bounds=r_bounds, v_steps=v_steps, r_steps=r_steps, cfl=cfl)

# 7) Evaluate expected affine outputs from E[V]. Because O, I and A are affine,
# evaluating them at E[V] is exactly their expectation.
U_values = np.array([float(U(tk)) for tk in t])
B_O_values = np.array([float(B_O(tk)) for tk in t])
B_I_values = np.array([float(B_I(tk)) for tk in t])
PE_values = np.array([float(PE(tk)) for tk in t])
mean_v = fp["mean_v"]
mean_r = fp["mean_r"]
std_v = np.sqrt(np.maximum(fp["var_v"], 0.0))
std_r = np.sqrt(np.maximum(fp["var_r"], 0.0))

mean_O = beta[0] * B_O_values + beta[1] * mean_v + beta[2] * (1.0 - U_values)
mean_I = (alpha[0] * B_I_values+ alpha[1] * (1.0 - mean_v)+ alpha[2] * mean_O+ alpha[3] * (1.0 - U_values))
mean_A = (gamma[0] * (1.0 - U_values) + gamma[1] * mean_I+ gamma[2] * mean_v+ gamma[3] * (1.0 - PE_values))

mean_g = sigma_0 + sigma_O * mean_O

V_star = np.array([model.expected_equilibrium(bo, bi, u, alpha, beta)for bo, bi, u in zip(B_O_values, B_I_values, U_values)])

# 8) Report parameter and numerical checks.
dV = model.reduced_drift_coefficients(B_O_level, B_I_level, alpha, beta)[1]
corner_equilibria = np.array([model.expected_equilibrium(bo, bi, u, alpha, beta) for bo, bi, u in product((0.0, 1.0), repeat=3)])

print("WM2 Fokker-Planck single price-step example")
print("-------------------------------------------")
print(f"d_V:                              {dV:.6f}")
print(f"kappa_R:                          {kappa_R:.6f}")
print(f"Non-oscillatory threshold -4d_V: {-4.0 * dV:.6f}")
print(
    "V* over parameter-domain corners: "
    f"[{np.min(corner_equilibria):.6f}, {np.max(corner_equilibria):.6f}]"
)
print(f"Initial V*(U={price_before:.2f}):         {V0:.6f}")
print(
    f"Post-step V*(U={price_after:.2f}):       "
    f"{model.expected_equilibrium(B_O_level, B_I_level, price_after, alpha, beta):.6f}"
)
print(f"Maximum |FP mass - 1|:            {np.max(np.abs(fp['mass'] - 1.0)):.3e}")
print(
    "Maximum |E_FP[V]-V_deterministic|: "
    f"{np.max(np.abs(mean_v - deterministic['V'])):.3e}"
)
print(f"Minimum expected output E[O]:     {np.min(mean_O):.6f}")
print(f"Minimum diffusion scale g(E[O]): {np.min(mean_g):.6f}")

# 9) Plot the evolving V marginal and response diagnostics.
v_grid = fp["v_grid"]
mass = fp["mass"]
normalised_marginal_v = fp["marginal_v"] / mass[:, None]
heatmap_probability = normalised_marginal_v / np.max(normalised_marginal_v)

fig, axes = plt.subplots(4, 1, figsize=(11, 16), constrained_layout=True)
ax_pdf, ax_state, ax_rate, ax_flow = axes

mesh = ax_pdf.pcolormesh(
    hours,
    v_grid,
    heatmap_probability.T,
    shading="auto",
    vmin=0.0,
    vmax=1.0,
)
fig.colorbar(mesh, ax=ax_pdf, label="Normalized probability density")
ax_pdf.plot(hours, mean_v, linewidth=1.8, label="E[V]")
ax_pdf.plot(hours, V_star, linestyle="--", linewidth=1.4, label="V*(B,U)")
ax_pdf.axhline(0.0, linestyle=":", linewidth=1.0)
ax_pdf.axhline(1.0, linestyle=":", linewidth=1.0)
ax_pdf.axvline(price_step_hour, linestyle="--", linewidth=1.0)
ax_pdf.set_ylabel("Flexibility state V")
ax_pdf.set_title("WM2 Fokker-Planck marginal density of V")
ax_pdf.legend(loc="best")

ax_state.step(hours, U_values, where="post", label="Price U")
ax_state.plot(hours, deterministic["V"], linewidth=1.5, label="Deterministic V")
ax_state.plot(hours, mean_v, linewidth=2.0, label="Fokker-Planck E[V]")
ax_state.plot(hours, V_star, linestyle="--", label="Expected equilibrium V*")
ax_state.fill_between(
    hours,
    mean_v - std_v,
    mean_v + std_v,
    alpha=0.20,
    label="E[V] +/- std(V)",
)
ax_state.axhline(0.0, linestyle=":")
ax_state.axhline(1.0, linestyle=":")
ax_state.axvline(price_step_hour, linestyle="--")
ax_state.set_ylabel("Price / flexibility")
ax_state.set_title("Single price step and inertial flexibility response")
ax_state.grid(True)
ax_state.legend(loc="best")

ax_rate.plot(hours, deterministic["R"], linewidth=1.5, label="Deterministic R")
ax_rate.plot(hours, mean_r, linewidth=2.0, label="Fokker-Planck E[R]")
ax_rate.fill_between(
    hours,
    mean_r - std_r,
    mean_r + std_r,
    alpha=0.20,
    label="E[R] +/- std(R)",
)
ax_rate.axhline(0.0, linestyle=":")
ax_rate.axvline(price_step_hour, linestyle="--")
ax_rate.set_ylabel("Rate R")
ax_rate.set_title("WM2 inertial-rate response")
ax_rate.grid(True)
ax_rate.legend(loc="best")

ax_flow.plot(hours, mean_O, label="E[O]")
ax_flow.plot(hours, mean_I, label="E[I]")
ax_flow.plot(hours, mean_A, label="E[A]")
ax_flow.axvline(price_step_hour, linestyle="--")
ax_flow.set_xlabel("Time [hours]")
ax_flow.set_ylabel("Normalized model output")
ax_flow.set_title("Expected affine model outputs and output-scaled diffusion")
ax_flow.grid(True)
ax_flow.legend(loc="upper left")

ax_diffusion = ax_flow.twinx()
ax_diffusion.plot(hours, mean_g, linestyle="--", label="g(E[O])")
ax_diffusion.set_ylabel("Diffusion scale")
ax_diffusion.legend(loc="upper right")

# This file lives in src/exe/WM2, so parents[3] is the repository root.
graph_dir = Path(__file__).resolve().parents[3] / "files" / "output" / "graphs"
graph_dir.mkdir(parents=True, exist_ok=True)
graph_path = graph_dir / "WM2_FP_price_step.png"
fig.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved graph to {graph_path}")
