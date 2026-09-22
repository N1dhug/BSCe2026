"""Simulate aggregate WM3 input data, fit L(Time), estimate WM3, and plot the fit."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from src.model.WM3 import WM3


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def daily_profile(hour):
    """Smooth consumption shape inspired by the supplied sketch."""
    hour = np.asarray(hour, dtype=float) % 24.0
    morning = 0.17 * sigmoid((hour - 6.5) / 0.7)
    evening = 0.34 * np.exp(-0.5 * ((hour - 19.0) / 2.0) ** 2)
    late_drop = 0.09 * sigmoid((hour - 22.5) / 0.7)
    return 0.16 + morning + evening - late_drop


def fit_linear_time_feature(hour, input_data):
    """Fit the best single wrapped 24-hour line and return normalized L(Time)."""
    hour = np.asarray(hour, dtype=float) % 24.0
    input_data = np.asarray(input_data, dtype=float)

    best = None
    for origin in range(24):
        x = ((hour - origin) % 24.0) / 24.0
        X = np.column_stack((np.ones(x.size), x))
        coeff, *_ = np.linalg.lstsq(X, input_data, rcond=None)
        prediction = X @ coeff
        mse = float(np.mean((input_data - prediction) ** 2))
        if best is None or mse < best["mse"]:
            best = {"origin": origin, "coeff": coeff, "mse": mse}

    origin = best["origin"]
    intercept, slope = best["coeff"]

    def raw_prediction(value):
        x = ((np.asarray(value, dtype=float) % 24.0 - origin) % 24.0) / 24.0
        return intercept + slope * x

    grid = np.arange(24, dtype=float)
    raw_grid = raw_prediction(grid)
    lo, hi = float(np.min(raw_grid)), float(np.max(raw_grid))

    if np.isclose(lo, hi):
        def L(value):
            value = np.asarray(value, dtype=float)
            result = np.full_like(value, 0.5, dtype=float)
            return float(result) if result.ndim == 0 else result
    else:
        def L(value):
            result = (raw_prediction(value) - lo) / (hi - lo)
            result = np.clip(result, 0.0, 1.0)
            return float(result) if np.ndim(result) == 0 else result

    best.update({"raw_min": lo, "raw_max": hi})
    return L, best


rng = np.random.default_rng(7)
days = 5
users = 250
hours = 24 * days
sample = np.arange(hours)
time_of_day = sample % 24

# Feasible, varying normalized price signal.
U = 0.50 + 0.13 * np.sin(2.0 * np.pi * (time_of_day - 15.0) / 24.0) + 0.06 * np.sin(4.0 * np.pi * time_of_day / 24.0)
U += rng.normal(0.0, 0.025, hours)
U = np.clip(U, 0.20, 0.80)
PE = np.full(hours, 0.5)

# Generate many users around the same daily shape with heterogeneous levels,
# price sensitivities and independent Gaussian noise, then aggregate them.
base = daily_profile(time_of_day)
user_scale = rng.normal(1.0, 0.08, (users, 1))
user_offset = rng.normal(0.0, 0.025, (users, 1))
user_price = np.clip(rng.normal(0.10, 0.02, (users, 1)), 0.03, 0.18)
noise = rng.normal(0.0, 0.045, (users, hours))
user_input = user_offset + user_scale * base[None, :] + user_price * (0.5 - U)[None, :] + noise
user_input = np.clip(user_input, 0.0, 1.0)
simulated_I = np.mean(user_input, axis=0)

# L(Time) is one wrapped 24-hour linear OLS fit to the aggregate input data.
L, L_fit = fit_linear_time_feature(time_of_day, simulated_I)

model = WM3((0.0, (hours - 1) / 24.0), hours, L)
input_data = {"I": simulated_I, "U": U, "Time": time_of_day.astype(float), "PE": PE}
initial_conditions = np.array([0.5, 0.0])

start = perf_counter()
fit = model.fit_params_brute(
    input_data,
    initial_conditions,
    grid_points=4,
    alpha_points=4,
    refinements=2,
    c_v_range=(-0.8, 0.8),
    d_v_range=(-0.8, -0.02),
    alpha_o_range=(0.05, 0.95),
    progress_every=20,
)
elapsed = perf_counter() - start

fitted_I = fit["deterministic_state"]["I"]
mse = float(np.mean((simulated_I - fitted_I) ** 2))

print("WM3 simulated-data brute-force fit")
print("----------------------------------")
print(f"Users:          {users}")
print(f"Days:           {days}")
print(f"Runtime:        {elapsed:.3f} s")
print(f"MSE:            {mse:.6e}")
print(f"L origin:       {L_fit['origin']:02d}:00")
print(f"L OLS slope:    {L_fit['coeff'][1]:.6f}")
print(f"L OLS MSE:      {L_fit['mse']:.6e}")
print(f"Branch:         {fit['brute_branch']}")
print(f"Score:          {fit['brute_score']:.6e}")
print(f"c_V:            {fit['brute_c_V']:.6f}")
print(f"d_V:            {fit['brute_d_V']:.6f}")
print(f"alpha_O:        {fit['brute_alpha_O']:.6f}")
print(f"kappa_R:        {fit['kappa_R']:.6f}")
print("alpha:          " + np.array2string(fit["alpha"], precision=4))
print("beta:           " + np.array2string(fit["beta"], precision=4))

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

axes[0].plot(sample, simulated_I, linewidth=1.3, label="Aggregate simulated input")
axes[0].plot(sample, fitted_I, linewidth=2.0, label="WM3 fitted input")
axes[0].set_ylabel("Normalized input I")
axes[0].set_title("WM3 fit to noisy aggregate user consumption")
axes[0].grid(True)
axes[0].legend(loc="best")

axes[1].plot(sample, np.array([L(value) for value in time_of_day]), linewidth=1.8, label="L(Time)")
axes[1].plot(sample, base, linewidth=1.2, label="Underlying daily shape")
axes[1].set_ylabel("Normalized level")
axes[1].set_title(f"Linear time feature, optimized 24-hour origin = {L_fit['origin']:02d}:00")
axes[1].grid(True)
axes[1].legend(loc="best")

axes[2].plot(sample, U, linewidth=1.4)
axes[2].set_xlabel("Hour")
axes[2].set_ylabel("Normalized price U")
axes[2].set_title("Feasible fluctuating price signal")
axes[2].grid(True)

graph_dir = Path(__file__).resolve().parents[3] / "files" / "output" / "graphs"
graph_dir.mkdir(parents=True, exist_ok=True)
graph_path = graph_dir / "WM3_fit_simulated_data.png"
fig.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

data_dir = Path(__file__).resolve().parents[3] / "files" / "output" / "data"
data_dir.mkdir(parents=True, exist_ok=True)
data_path = data_dir / "WM3_simulated_data.npz"
np.savez(data_path, time=sample, time_of_day=time_of_day, I=simulated_I, U=U, PE=PE, L=np.array([L(value) for value in time_of_day]), fitted_I=fitted_I, user_input=user_input)

print(f"Saved graph to {graph_path}")
print(f"Saved simulated data to {data_path}")
