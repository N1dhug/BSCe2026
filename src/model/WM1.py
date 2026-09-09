import numpy as np

from src.model.WM import WM
from src.model.Simulate import Simulate
from src.method.FokkerPlanc import FokkerPlanc


class WM1(WM):
    """Updated Water Model 1.

    Model equations remain in the model layer; numerical probability-density
    propagation is delegated to the method layer.
    """

    def __init__(self, T, t_steps):
        self.simulate = Simulate()
        self.fp = FokkerPlanc()
        self.T = T
        self.t_steps = int(t_steps)
        self.t = np.linspace(T[0], T[1], self.t_steps)
        self.dt = self.t[1] - self.t[0]

    @staticmethod
    def _as_vector(values, length, name):
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != length:
            raise ValueError(f"{name} must contain {length} entries.")
        return values

    @classmethod
    def validate_parameters(cls, alpha, beta, gamma, tol=1e-10):
        alpha = cls._as_vector(alpha, 4, "alpha")
        beta = cls._as_vector(beta, 3, "beta")
        gamma = cls._as_vector(gamma, 4, "gamma")
        for name, values in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if np.any(values < -tol) or np.any(values > 1.0 + tol):
                raise ValueError(f"{name} coefficients must lie in [0, 1].")
            if not np.isclose(np.sum(values), 1.0, atol=tol, rtol=0.0):
                raise ValueError(f"{name} coefficients must sum to 1.")
        return alpha, beta, gamma

    def I(self, P_I, alpha):
        return self._as_vector(alpha, 4, "alpha") @ self._as_vector(P_I, 4, "P_I")

    def O(self, P_O, beta):
        return self._as_vector(beta, 3, "beta") @ self._as_vector(P_O, 3, "P_O")

    def A(self, P_A, gamma):
        return self._as_vector(gamma, 4, "gamma") @ self._as_vector(P_A, 4, "P_A")

    @staticmethod
    def reduced_drift_coefficients(BtO, BtI, alpha, beta):
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        o_bar = beta[0] * BtO + beta[2]
        i_bar = alpha[0] * BtI + alpha[1] + alpha[2] * o_bar + alpha[3]
        d0 = i_bar - o_bar
        dV = -alpha[1] - (1.0 - alpha[2]) * beta[1]
        dU = (1.0 - alpha[2]) * beta[2] - alpha[3]
        return d0, dV, dU

    def _evaluate_step(self, t, V, parameters):
        U = parameters["U"](t)
        BO = parameters["B_O"](t)
        BI = parameters["B_I"](t)
        PE = parameters["PE"](t)
        alpha, beta, gamma = self.validate_parameters(
            parameters["alpha"], parameters["beta"], parameters["gamma"]
        )

        P_O = np.array([BO, V, 1.0 - U])
        O = self.O(P_O, beta)
        P_I = np.array([BI, 1.0 - V, O, 1.0 - U])
        I = self.I(P_I, alpha)
        P_A = np.array([1.0 - U, I, V, 1.0 - PE])
        A = self.A(P_A, gamma)
        return U, BO, BI, PE, P_I, P_O, P_A, O, I, A

    def deterministic_solver(self, init_cond, parameters):
        init = np.asarray(init_cond).reshape(-1)
        V0 = init[0]
        R_prev = init[1] if init.size > 1 else 0.0
        theta = parameters.get("theta", 0.0)

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
            (U[k], BO[k], BI[k], PE[k], P_I[k], P_O[k], P_A[k], O[k], I[k], A[k]) = self._evaluate_step(
                tk, V[k], parameters
            )
            R[k] = I[k] - O[k] + theta * R_prev
            if k < self.t_steps - 1:
                V[k + 1] = V[k] + self.dt * R[k]
            R_prev = R[k]

        return {
            "t": self.t.copy(), "V": V, "R": R, "U": U, "B_O": BO,
            "B_I": BI, "PE": PE, "P_I": P_I, "P_O": P_O, "P_A": P_A,
            "O": O, "I": I, "A": A,
        }

    def fokker_planc_solver(self, init_cond, parameters, sigma, degree=20, e_bounds=(-1.5, 1.5), x_steps=301, cfl=0.45):
        alpha, beta, _ = self.validate_parameters(
            parameters["alpha"], parameters["beta"], parameters["gamma"]
        )
        e_min, e_max = e_bounds
        span = e_max - e_min
        if span <= 0.0:
            raise ValueError("e_bounds must satisfy e_min < e_max.")

        V0 = np.asarray(init_cond).reshape(-1)[0]
        mean_result = self.deterministic_solver([V0], parameters)
        mean = mean_result["V"]
        dV = -alpha[1] - (1.0 - alpha[2]) * beta[1]

        def drift_u(t, u):
            e = e_min + span * np.asarray(u)
            return dV * e / span

        def diffusion_u(t, u):
            return np.zeros_like(np.asarray(u)) + sigma / span

        u0 = (0.0 - e_min) / span
        u_grid, density_u, coefficients = self.fp.solve(
            self.t, [u0], drift_u, diffusion_u,
            degree=degree, x_steps=x_steps, cfl=cfl,
        )

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
        result.update({
            "e_grid": e_grid, "density_e": density_e, "coefficients": coefficients,
            "error_mean": error_mean, "error_variance": error_variance,
            "lower_violation": lower_violation, "upper_violation": upper_violation,
            "sigma": sigma, "e_bounds": (e_min, e_max),
        })
        return result
