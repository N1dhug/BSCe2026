"""Plot one day of the synthetic hourly data used for WM2 fitting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.simulate.simple import Simple

days = 1
hours = 24 * days
time_hours = np.arange(hours)

baseline = Simple(days=days, mean=0.5, sin_coeff=(0.25,), cos_coeff=(0.10,), std=0.0).data()
simulated_I = Simple(days=days, mean=0.5, sin_coeff=(0.10,), cos_coeff=(0.04,), std=0.01, seed=1).data()
U = np.array([
    0.35, 0.30, 0.28, 0.25, 0.27, 0.32,
    0.40, 0.52, 0.65, 0.72, 0.68, 0.60,
    0.55, 0.50, 0.48, 0.55, 0.68, 0.80,
    0.75, 0.66, 0.58, 0.50, 0.44, 0.38,
])

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(time_hours, simulated_I, marker="o", linewidth=1.5, label="Simulated input I")
ax.plot(time_hours, baseline, linewidth=2.0, label="Baseline")
ax.plot(time_hours, U, linewidth=1.5, label="Price U")
ax.set_xlabel("Time [hours]")
ax.set_ylabel("Normalized value")
ax.set_title("Simulated WM2 fitting data")
ax.set_xticks(np.arange(0, 24, 3))
ax.set_ylim(0.0, 1.0)
ax.grid(True)
ax.legend(loc="best")

graph_dir = Path(__file__).resolve().parents[3] / "files" / "output" / "graphs"
graph_dir.mkdir(parents=True, exist_ok=True)
graph_path = graph_dir / "WM2_simulated_data.png"
fig.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved graph to {graph_path}")
