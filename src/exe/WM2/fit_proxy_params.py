"""Fit WM2 proxy parameters to one day of simple simulated hourly data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.model.WM2 import WM2
from src.simulate.simple import Simple

days = 1
hours = 24 * days
time_hours = np.arange(hours)

# Synthetic baseline and purchased-input data.  The noiseless input relation is
# I = 0.3 + 0.4 B, with B generated from daily sine/cosine harmonics.
baseline = Simple(days=days, mean=0.5, sin_coeff=(0.25,), cos_coeff=(0.10,), std=0.0).data()
simulated_I = Simple(days=days, mean=0.5, sin_coeff=(0.10,), cos_coeff=(0.04,), std=0.01, seed=1).data()

U = np.full(hours, 0.5)
PE = np.full(hours, 0.5)
input_data = {"I": simulated_I, "U": U, "PE": PE}
baseline_feature_vec = {"B_I": baseline[:, None], "B_O": baseline[:, None]}

# Twenty-four hourly measurements span t = 0,...,23 hours.
model = WM2((0.0, (hours - 1) / 24.0), hours)
initial_conditions = np.array([0.5, 0.0])

fit = model.fit_params_proxy(input_data, initial_conditions, baseline_feature_vec)
fitted_I = fit["deterministic_state"]["I"]
mse = float(np.mean((simulated_I - fitted_I) ** 2))

print("WM2 proxy parameter fit")
print("-----------------------")
print(f"MSE:       {mse:.6e}")
print(f"kappa_R:   {fit['kappa_R']:.6f}")
print(f"sigma_0:   {fit['sigma_0']:.6f}")
print(f"sigma_O:   {fit['sigma_O']:.6f}")
print("alpha:     " + np.array2string(fit["alpha"], precision=4))
print("beta:      " + np.array2string(fit["beta"], precision=4))

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(time_hours, simulated_I, marker="o", linewidth=1.5, label="Simulated data")
ax.plot(time_hours, fitted_I, linewidth=2.0, label="WM2 proxy fit")
ax.set_xlabel("Time [hours]")
ax.set_ylabel("Normalized input I")
ax.set_title("WM2 proxy fit against one day of simulated hourly data")
ax.set_xticks(np.arange(0, 24, 3))
ax.grid(True)
ax.legend(loc="best")

graph_dir = Path(__file__).resolve().parents[3] / "files" / "output" / "graphs"
graph_dir.mkdir(parents=True, exist_ok=True)
graph_path = graph_dir / "WM2_fit_proxy_params.png"
fig.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved graph to {graph_path}")
