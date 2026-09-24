import numpy as np

from src.model.base.wm import wm
from src.method.fokker_planc import bm_1_fokker_planc
from src.validation.model.wm_1 import wm_1_validation

class wm_1(wm):
    """Updated Water Model 1.

    Model equations remain in the model layer; numerical probability-density
    propagation is delegated to the method layer.
    """

    def __init__(self, T, t_steps):
        # Initialize variables
        T, t_steps = wm_1_validation.constructor(T, t_steps)
        self.fp = bm_1_fokker_planc()
        self.T = T
        self.t_steps = t_steps
        self.t = np.linspace(T[0], T[1], self.t_steps)
        self.dt = self.t[1] - self.t[0]

    def I(self, P_I, alpha):
        # Evaluate model map
        P_I, alpha = wm_1_validation.linear_map(P_I, alpha, 4, "P_I", "alpha")
        return alpha @ P_I

    def O(self, P_O, beta):
        # Evaluate model map
        P_O, beta = wm_1_validation.linear_map(P_O, beta, 3, "P_O", "beta")
        return beta @ P_O

    def A(self, P_A, gamma):
        # Evaluate model map
        P_A, gamma = wm_1_validation.linear_map(P_A, gamma, 4, "P_A", "gamma")
        return gamma @ P_A

    @staticmethod
    def reduced_drift_coefficients(BtO, BtI, alpha, beta):
        # Compute reduced drift
        BtO, BtI, alpha, beta = wm_1_validation.reduced_drift_inputs(BtO, BtI, alpha, beta)
        o_bar = beta[0] * BtO + beta[2]
        i_bar = alpha[0] * BtI + alpha[1] + alpha[2] * o_bar + alpha[3]
        d0 = i_bar - o_bar
        dV = -alpha[1] - (1.0 - alpha[2]) * beta[1]
        dU = (1.0 - alpha[2]) * beta[2] - alpha[3]
        return d0, dV, dU

    def _evaluate_step(self, t, V, parameters):
        # Evaluate model step
        U = wm_1_validation.signal_value(parameters["U"](t), "U(t)")
        BO = wm_1_validation.signal_value(parameters["B_O"](t), "B_O(t)")
        BI = wm_1_validation.signal_value(parameters["B_I"](t), "B_I(t)")
        PE = wm_1_validation.signal_value(parameters["PE"](t), "PE(t)")
        alpha = parameters["alpha"]
        beta = parameters["beta"]
        gamma = parameters["gamma"]

        P_O = np.array([BO, V, 1.0 - U])
        O = self.O(P_O, beta)
        P_I = np.array([BI, 1.0 - V, O, 1.0 - U])
        I = self.I(P_I, alpha)
        P_A = np.array([1.0 - U, I, V, 1.0 - PE])
        A = self.A(P_A, gamma)
        return U, BO, BI, PE, P_I, P_O, P_A, O, I, A

    def deterministic_solver(self, init_cond, parameters):
        # Solve deterministic model
        init, parameters = wm_1_validation.deterministic_inputs(init_cond, parameters)
        V0 = init[0]
        R_prev = init[1] if init.size > 1 else 0.0
        theta = parameters["theta"]

        V = np.empty(self.t_steps)
        R = np.empty(self.t_steps)
        U = np.empty(self.t_steps)
        BO = np.empty(self.t_steps)
        BI = np.empty(self.t_steps)
        PE = np.empty(self.t_steps)
        O = np.empty(self.t_steps)
        I = np.empty(self.t_steps)
        A = np.empty(self.t_steps)
        P_I = np.empty((self.t_steps, 4))
        P_O = np.empty((self.t_steps, 3))
        P_A = np.empty((self.t_steps, 4))
        V[0] = V0

        for k, tk in enumerate(self.t):
            (U[k], BO[k], BI[k], PE[k], P_I[k], P_O[k], P_A[k], O[k], I[k], A[k]) = self._evaluate_step(tk, V[k], parameters)
            R[k] = I[k] - O[k] + theta * R_prev
            if k < self.t_steps - 1:
                V[k + 1] = V[k] + self.dt * R[k]
            R_prev = R[k]

        return {"t": self.t.copy(), "V": V, "R": R, "U": U, "B_O": BO, "B_I": BI, "PE": PE, "P_I": P_I, "P_O": P_O, "P_A": P_A, "O": O, "I": I, "A": A,}

    def fokker_planc_solver(self, init_cond, parameters, sigma, degree=20, e_bounds=(-1.5, 1.5), x_steps=301, cfl=0.45):
        # Solve Fokker-Planck model
        init_cond, parameters, sigma, e_bounds = wm_1_validation.fokker_planc_inputs(init_cond, parameters, sigma, e_bounds)
        alpha = parameters["alpha"]
        beta = parameters["beta"]
        e_min, e_max = e_bounds
        span = e_max - e_min

        V0 = init_cond[0]
        mean_result = self.deterministic_solver([V0], parameters)
        mean = mean_result["V"]
        dV = -alpha[1] - (1.0 - alpha[2]) * beta[1]

        def drift_u(t, u):
            # Evaluate centered drift
            e = e_min + span * np.asarray(u)
            return dV * e / span

        def diffusion_u(t, u):
            # Evaluate centered diffusion
            return np.zeros_like(np.asarray(u)) + sigma / span

        u0 = (0.0 - e_min) / span
        u_grid, density_u, coefficients = self.fp.solve(self.t, [u0], drift_u, diffusion_u, degree=degree, x_steps=x_steps, cfl=cfl,)

        e_grid = e_min + span * u_grid
        density_e = density_u / span
        de = span / len(e_grid)
        error_mean = np.sum(density_e * e_grid[None, :], axis=1) * de
        error_second = np.sum(density_e * (e_grid[None, :] ** 2), axis=1) * de
        error_variance = np.maximum(error_second - error_mean**2, 0.0)

        lower_violation = np.empty(self.t_steps)
        upper_violation = np.empty(self.t_steps)
        for k in range(self.t_steps):
            lower_violation[k] = np.sum(density_e[k, e_grid < -mean[k]]) * de
            upper_violation[k] = np.sum(density_e[k, e_grid > 1.0 - mean[k]]) * de

        result = dict(mean_result)
        result.update({"e_grid": e_grid, "density_e": density_e, "coefficients": coefficients, "error_mean": error_mean, "error_variance": error_variance, "lower_violation": lower_violation, "upper_violation": upper_violation, "sigma": sigma, "e_bounds": (e_min, e_max)})
        return result
