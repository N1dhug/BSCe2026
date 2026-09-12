# src/exe/WM1/FP_price_step.py
# Updated WM1 stochastic response to a single dynamic-price step.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.exe.tool.InputBuilder import InputBuilder
from src.model.WM1 import WM1


# 1) Time grid: four days, hourly reporting.
days = 4
T = (0.0, float(days))
t_steps = 24 * days + 1
t = np.linspace(T[0], T[1], t_steps)
hours = 24.0 * t

inputs = InputBuilder()

# 2) Exogenous baselines and grid price.
B_O = inputs.energy((np.array([0.0, days]), np.array([0.30, 0.30])))
B_I = inputs.energy((np.array([0.0, days]), np.array([0.30, 0.30])))
PE = inputs.price((np.array([0.0, days]), np.array([0.50, 0.50])))

# 3) Dynamic-price step U: high price -> low price at hour 12.
price_step_hour = 12.0
price_step_time = price_step_hour / 24.0
U = inputs.price(
    (
        np.array([0.0, price_step_time, days]),
        np.array([0.75, 0.25, 0.25]),
    )
)

# 4) Updated WM1 simplex coefficients.
#
# At each time step WM1 constructs the model-input vectors
#
#     P_O = [B_O, V, 1-U]
#     P_I = [B_I, 1-V, O, 1-U]
#     P_A = [1-U, I, V, 1-PE]
#
# and evaluates
#
#     O = beta^T  P_O
#     I = alpha^T P_I
#     A = gamma^T P_A.
alpha = np.array([0.15, 0.25, 0.15, 0.45])
beta  = np.array([0.30, 0.45, 0.25])
gamma = np.array([0.30, 0.20, 0.10, 0.40])

parameters = {
    "U": U,
    "B_O": B_O,
    "B_I": B_I,
    "PE": PE,
    "alpha": alpha,
    "beta": beta,
    "gamma": gamma,
    "theta": 0.0,
}

# 5) Stochastic settings from the no-inertia WM1 model.
init_cond = [0.50]
sigma = 0.12
degree = 24
x_steps = 301
cfl = 0.45

# Wide centered-error interval. It is deliberately wider than [0,1] because
# additive diffusion is not pathwise confined to the physical volume interval.
e_bounds = (-1.5, 1.5)

model = WM1(T, t_steps)
result = model.fokker_planc_solver(
    init_cond=init_cond,
    parameters=parameters,
    sigma=sigma,
    degree=degree,
    e_bounds=e_bounds,
    x_steps=x_steps,
    cfl=cfl,
)

# 6) Probability checks.
e_grid = result["e_grid"]
density = result["density_e"]
coefficients = result["coefficients"]
de = (e_bounds[1] - e_bounds[0]) / len(e_grid)
mass = np.sum(density, axis=1) * de
coefficient_mass = np.sum(coefficients, axis=1)
std = np.sqrt(result["error_variance"])

print("WM1 Fokker-Planck probability checks")
print("------------------------------------")
print(f"Maximum |density mass - 1|:    {np.max(np.abs(mass - 1.0)):.3e}")
print(f"Maximum |sum(c) - 1|:          {np.max(np.abs(coefficient_mass - 1.0)):.3e}")
print(f"Minimum coefficient:           {np.min(coefficients):.3e}")
print(f"Maximum P(V < 0):              {np.max(result['lower_violation']):.6f}")
print(f"Maximum P(V > 1):              {np.max(result['upper_violation']):.6f}")
print(
    f"Maximum total bound violation: "
    f"{np.max(result['lower_violation'] + result['upper_violation']):.6f}"
)

# 7) Plots.
fig, axes = plt.subplots(4, 1, figsize=(11, 15), constrained_layout=True)
ax_pdf, ax_state, ax_flow, ax_risk = axes

# WM1's centered stochastic density is independent of the dynamic price U.
# Unlike the BM1 visualization, we therefore do not use a time-density heatmap.
# Show the final/representative centered-error probability density directly as
# an ordinary 2D probability-density curve.
ax_pdf.plot(e_grid, density[-1], linewidth=2.0, label="Centered error density")
ax_pdf.axvline(0.0, linestyle="--", linewidth=1.0, label="Zero error")
ax_pdf.set_xlabel("Centered error e")
ax_pdf.set_ylabel("Probability density")
ax_pdf.set_title("WM1 Fokker-Planck centered-error probability density")
ax_pdf.grid(True)
ax_pdf.legend(loc="best")

ax_state.step(hours, result["U"], where="post", label="Dynamic price U")
ax_state.plot(hours, result["V"], linewidth=2.0, label="Mean volume m(t)")
ax_state.fill_between(
    hours,
    result["V"] - std,
    result["V"] + std,
    alpha=0.2,
    label="m +/- std(E)",
)
ax_state.axhline(0.0, linestyle=":")
ax_state.axhline(1.0, linestyle=":")
ax_state.axvline(price_step_hour, linestyle="--")
ax_state.set_ylabel("Price / volume")
ax_state.set_title("Price step and stochastic WM1 volume")
ax_state.grid(True)
ax_state.legend(loc="best")

ax_flow.plot(hours, result["O"], label="Output O")
ax_flow.plot(hours, result["I"], label="Input I")
ax_flow.plot(hours, result["A"], label="Available A")
ax_flow.axvline(price_step_hour, linestyle="--")
ax_flow.set_ylabel("Normalized power")
ax_flow.set_title("Updated causal order: O -> I -> A")
ax_flow.grid(True)
ax_flow.legend(loc="best")

ax_risk.plot(hours, result["lower_violation"], label="P(V < 0)")
ax_risk.plot(hours, result["upper_violation"], label="P(V > 1)")
ax_risk.plot(
    hours,
    result["lower_violation"] + result["upper_violation"],
    linewidth=2.0,
    label="Total violation probability",
)
ax_risk.axvline(price_step_hour, linestyle="--")
ax_risk.set_xlabel("Time [hours]")
ax_risk.set_ylabel("Probability")
ax_risk.set_title("Physical [0,1] chance-constraint diagnostics")
ax_risk.set_ylim(bottom=0.0)
ax_risk.grid(True)
ax_risk.legend(loc="best")

# This file lives in src/exe/WM1, so parents[3] is the repository root.
graph_dir = Path(__file__).resolve().parents[3] / "files" / "output" / "graphs"
graph_dir.mkdir(parents=True, exist_ok=True)
graph_path = graph_dir / "WM1_FP_price_step.png"
fig.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved graph to {graph_path}")
