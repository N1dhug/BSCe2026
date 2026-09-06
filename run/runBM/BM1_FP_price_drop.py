# run/runBM/BM1_FP_price_drop.py
# ------------------------------------------------------------
# BM1 stochastic response to a single price drop.
# No optimizer is used. The Fokker-Planck equation is solved directly for
# the probability density p(x, t).
# ------------------------------------------------------------
from models.BatteryModel.BM1 import BM1
from models.Simulate import Simulate
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# Helper: generate monotone-decreasing interpolation values
# ------------------------------------------------------------
def md_vals(n_points, kappa):
    """Return evenly spaced points and positive decrements summing to 2.

    kappa = 0 gives equal decrements and therefore a linear md function.
    Increasing kappa concentrates the decrease around x=0.5, making the
    monotone response progressively steeper/non-linear.
    """
    if int(n_points) != n_points or n_points < 2:
        raise ValueError("n_points must be an integer >= 2.")
    if kappa < 0:
        raise ValueError("kappa must be non-negative.")

    n_points = int(n_points)
    poi = np.linspace(0.0, 1.0, n_points)

    interval_midpoints = 0.5 * (poi[:-1] + poi[1:])
    centred = 2.0 * (interval_midpoints - 0.5)
    weights = np.exp(-float(kappa) * centred**2)
    weights /= np.sum(weights)

    pr = np.zeros(n_points, dtype=float)
    pr[1:] = 2.0 * weights

    return poi, pr


# ------------------------------------------------------------
# 1) problem size and time grid
# ------------------------------------------------------------
days = 1
T = (0.0, float(days))

# 24 one-hour intervals require 25 grid points when both endpoints are kept.
t_steps = 24 * days + 1
t = np.linspace(T[0], T[1], t_steps)


# ------------------------------------------------------------
# 2) baseline consumption
# ------------------------------------------------------------
sim = Simulate()

B_data = (
    np.array([0.0, 1.0]),
    np.array([0.3, 0.3]),
)
B = sim.energy(B_data)

# BM1 still accepts A in its constructor because the optimizer uses it, but
# A is irrelevant in this run. Reuse B as a harmless placeholder.
A = B


# ------------------------------------------------------------
# 3) fixed price signal: 0.75 -> 0.25 at hour 12
# ------------------------------------------------------------
price_drop_hour = 12.0
price_drop_time = price_drop_hour / 24.0

U_data = (
    np.array([0.0, price_drop_time, 1.0]),
    np.array([0.75, 0.25, 0.25]),
)
U = sim.price(U_data)


# ------------------------------------------------------------
# 4) monotone decreasing f and g from md_vals + kappa
# ------------------------------------------------------------
n_md_points = 9
kappa = 4.0

poi_x, pr_x = md_vals(n_md_points, kappa)
poi_u, pr_u = md_vals(n_md_points, kappa)

f = sim.md(poi_x, pr_x)
g = sim.md(poi_u, pr_u)


# ------------------------------------------------------------
# 5) BM1 parameters and initial state
# ------------------------------------------------------------
C = 1.0
k = 2.0
Xi = 1.0
sigma_x = 0.1

# Initial state of charge requested for the price-drop experiment.
init_cond = [0.5]
parameters = [U, f, g, C, k, Xi]


# ------------------------------------------------------------
# 6) solve the Fokker-Planck equation
# ------------------------------------------------------------
model = BM1(B, A, T, t_steps)

print("Solving BM1 Fokker-Planck equation for a 0.75 -> 0.25 price drop at hour 12...")
fp_t, x_grid, density, mean, variance = model.fokker_planck_solver(
    init_cond,
    parameters,
    sigma_x,
    n_x=401,
)

fp_hours = 24.0 * fp_t
std = np.sqrt(variance)

# Probability mass should remain approximately one for every output time.
dx = x_grid[1] - x_grid[0]
mass = np.sum(density, axis=1) * dx
print(f"Initial probability mass:  {mass[0]:.8f}")
print(f"Terminal probability mass: {mass[-1]:.8f}")
print(f"Terminal E[X]:             {mean[-1]:.6f}")
print(f"Terminal Var[X]:           {variance[-1]:.6f}")


# ------------------------------------------------------------
# 7) plots
# ------------------------------------------------------------
fig, (ax_pdf, ax_state, ax_terminal) = plt.subplots(
    nrows=3,
    figsize=(11, 13),
    gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]},
)

# Full probability-density solution p(x, t)
mesh = ax_pdf.pcolormesh(
    fp_hours,
    x_grid,
    density.T,
    shading="auto",
)
ax_pdf.plot(fp_hours, mean, label="Fokker-Planck E[X(t)]")
ax_pdf.axvline(price_drop_hour, linestyle="--", label="Price drop")
ax_pdf.set_ylabel("State of charge x")
ax_pdf.set_title("BM1 probability density p(x,t)")
ax_pdf.set_ylim(0.0, 1.0)
ax_pdf.legend(loc="best")
fig.colorbar(mesh, ax=ax_pdf, label="Probability density")

# Price signal and first two moments of the state distribution
Ut = U(fp_t)
ax_state.step(fp_hours, Ut, where="post", label="Price U(t)")
ax_state.plot(fp_hours, mean, label="E[X(t)]")
ax_state.fill_between(
    fp_hours,
    np.clip(mean - std, 0.0, 1.0),
    np.clip(mean + std, 0.0, 1.0),
    alpha=0.2,
    label="E[X] +/- std",
)
ax_state.axvline(price_drop_hour, linestyle="--")
ax_state.set_ylabel("Price / SoC")
ax_state.set_title("Price drop and probability-distribution moments")
ax_state.set_ylim(0.0, 1.0)
ax_state.grid(True)
ax_state.legend(loc="best")

# Terminal PDF
ax_terminal.plot(x_grid, density[-1], label="p(x, 24 h)")
ax_terminal.axvline(mean[-1], linestyle="--", label="E[X(24 h)]")
ax_terminal.set_xlabel("State of charge x")
ax_terminal.set_ylabel("Probability density")
ax_terminal.set_title("Terminal state-of-charge density")
ax_terminal.grid(True)
ax_terminal.legend(loc="best")

fig.tight_layout()
plt.show()
