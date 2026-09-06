# run/runBM/BM1
# ------------------------------------------------------------
from models.BatteryModel.BM1 import BM1
from models.Simulate import Simulate
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 1) problem size and time grid
# ------------------------------------------------------------
days = 1
T = (0, days)                 # simulate on [0, days]
N = 24 * days                 # hourly samples
t_steps = int(T[1] * N)
t = np.linspace(T[0], T[1], t_steps)

# ------------------------------------------------------------
# 2) simulation data (baseline and energy demand)
# ------------------------------------------------------------
B_data = (
    np.linspace(0, 1, 2),
    np.array([0.3, 0.3]),
)
A_data = (
    t,
    np.array([
        0.125, 0.125, 0.1, 0.1, 0.125, 0.25, 0.3, 0.75,
        0.75, 0.5, 0.4, 0.25, 0.3, 0.4, 0.5, 0.75,
        0.75, 0.8, 0.75, 0.5, 0.35, 0.2, 0.125, 0.125,
    ]),
)

sim = Simulate()
B = sim.energy(B_data)
A = sim.energy(A_data)

# ------------------------------------------------------------
# 3) control and model function definitions
# ------------------------------------------------------------
initial_u = np.full(N, 0.5)

alphas = (
    np.array([0, 0.25, 0.5, 0.75, 1]),
    np.array([0, 0.25, 0.75, 0.75, 0.25]),
)
betas = (
    np.array([0, 0.2, 0.4, 0.45, 0.6, 0.8, 1]),
    np.array([0, 0.2, 0.85, 0.15, 0.6, 0.15, 0.05]),
)

poi_x, pr_x = alphas
poi_u, pr_u = betas

U = sim.price((t, initial_u))
f = sim.md(poi_x, pr_x)
g = sim.md(poi_u, pr_u)

# ------------------------------------------------------------
# 4) constant model parameters and initial state
# ------------------------------------------------------------
C = 1
k = 2
Xi = 1
sigma_x = 0.1
init_cond = [0.75]
parameters = [U, f, g, C, k, Xi]

# ------------------------------------------------------------
# 5) optimise
# ------------------------------------------------------------
print("Optimizing control signal U(t) to minimize the objective function phi(U)...")
model = BM1(B, A, T, t_steps)
solution = model.optimize(initial_u, init_cond, parameters)

# ------------------------------------------------------------
# 6) simulate with optimal control
# ------------------------------------------------------------
print("Simulating one Euler-Maruyama trajectory...")
U_data_opt = (t, solution.x)
U_opt = sim.price(U_data_opt)
parameters[0] = U_opt

t, Xt, Ut, Bt, fXt, gUt, dt, Dt = model.stochastic_solver(
    init_cond,
    parameters,
    sigma_x,
    seed=1,
)

print("Solving the corresponding Fokker-Planck equation...")
fp_t, x_grid, fp_density, fp_mean, fp_variance = model.fokker_planck_solver(
    init_cond,
    parameters,
    sigma_x,
    n_x=401,
)

dx = x_grid[1] - x_grid[0]
terminal_mass = np.sum(fp_density[-1]) * dx
print(f"Terminal Fokker-Planck probability mass: {terminal_mass:.8f}")

# ------------------------------------------------------------
# 7) plots
# ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    sharex=True,
    figsize=(10, 11),
)

# 1) Baseline vs. Demand and Available energy
ax1.plot(t, Dt, label="Demand D(t)", color="C2")
ax1.plot(t, Bt, label="Baseline B(t)", color="C3")
ax1.plot(t, A_data[1], label="Available A(t)", color="C5")
ax1.set_ylabel("energy")
ax1.set_title("Baseline, Demand and Available energy")
ax1.grid(True)
ax1.legend(loc="best")

# 2) Control signal U(t), Euler-Maruyama SoC, and Fokker-Planck mean
ax2.step(
    t,
    Ut,
    where="post",
    label="Control U(t)",
    color="C0",
)
ax2.plot(
    t,
    Xt,
    label="Euler-Maruyama X(t)",
    color="C4",
)
ax2.plot(
    fp_t,
    fp_mean,
    linestyle="--",
    label="Fokker-Planck E[X(t)]",
    color="C1",
)
ax2.set_ylabel("Price signal & SoC")
ax2.set_title("Optimal price signal and stochastic BM1 state")
ax2.grid(True)
ax2.legend(loc="best")

fig.tight_layout()
plt.show()
