import numpy as np

from src.model.base.WM import WM
from src.method.FokkerPlanc import WM2FokkerPlanc
from src.validation.model.WM2 import WM2Validation


class WM2(WM):
    """Water Model 2: inertial WM dynamics with output-scaled diffusion."""

    def __init__(self, T, t_steps):
        T, t_steps = WM2Validation.constructor(T, t_steps)
        self.T = T
        self.t_steps = t_steps
        self.t = np.linspace(T[0], T[1], t_steps)
        self.dt = self.t[1] - self.t[0]
        self.fp = WM2FokkerPlanc()

    def I(self, P_I, alpha):
        P_I, alpha = WM2Validation.linear_map(P_I, alpha, 4, "P_I", "alpha")
        return alpha @ P_I

    def O(self, P_O, beta):
        P_O, beta = WM2Validation.linear_map(P_O, beta, 3, "P_O", "beta")
        return beta @ P_O

    def A(self, P_A, gamma):
        P_A, gamma = WM2Validation.linear_map(P_A, gamma, 4, "P_A", "gamma")
        return gamma @ P_A

    @staticmethod
    def reduced_drift_coefficients(B_O, B_I, alpha, beta):
        B_O, B_I, alpha, beta = WM2Validation.reduced_drift_inputs(
            B_O, B_I, alpha, beta
        )
        o_bar = beta[0] * B_O + beta[2]
        i_bar = alpha[0] * B_I + alpha[1] + alpha[2] * o_bar + alpha[3]
        d0 = i_bar - o_bar
        dV = -alpha[1] - (1.0 - alpha[2]) * beta[1]
        dU = (1.0 - alpha[2]) * beta[2] - alpha[3]
        return d0, dV, dU

    @classmethod
    def expected_equilibrium(cls, B_O, B_I, U, alpha, beta):
        U = WM2Validation.signal_value(U, "U")
        d0, dV, dU = cls.reduced_drift_coefficients(B_O, B_I, alpha, beta)
        if dV >= 0.0:
            raise ValueError("Expected equilibrium requires stable d_V < 0.")
        return -(d0 + dU * U) / dV

    def _evaluate_step(self, t, V, parameters):
        U = WM2Validation.signal_value(parameters["U"](t), "U(t)")
        B_O = WM2Validation.signal_value(parameters["B_O"](t), "B_O(t)")
        B_I = WM2Validation.signal_value(parameters["B_I"](t), "B_I(t)")
        PE = WM2Validation.signal_value(parameters["PE"](t), "PE(t)")
        alpha, beta, gamma = parameters["alpha"], parameters["beta"], parameters["gamma"]
        P_O = np.array([B_O, V, 1.0 - U])
        O = self.O(P_O, beta)
        P_I = np.array([B_I, 1.0 - V, O, 1.0 - U])
        I = self.I(P_I, alpha)
        P_A = np.array([1.0 - U, I, V, 1.0 - PE])
        A = self.A(P_A, gamma)
        return U, B_O, B_I, PE, P_I, P_O, P_A, O, I, A

    def deterministic_solver(self, init_cond, parameters):
        init_cond, parameters = WM2Validation.deterministic_inputs(init_cond, parameters)
        V = np.empty(self.t_steps)
        R = np.empty(self.t_steps)
        R_dot = np.empty(self.t_steps)
        U = np.empty(self.t_steps)
        B_O = np.empty(self.t_steps)
        B_I = np.empty(self.t_steps)
        PE = np.empty(self.t_steps)
        O = np.empty(self.t_steps)
        I = np.empty(self.t_steps)
        A = np.empty(self.t_steps)
        P_I = np.empty((self.t_steps, 4))
        P_O = np.empty((self.t_steps, 3))
        P_A = np.empty((self.t_steps, 4))
        V[0], R[0] = init_cond
        kappa_R = parameters["kappa_R"]

        for k, tk in enumerate(self.t):
            U[k], B_O[k], B_I[k], PE[k], P_I[k], P_O[k], P_A[k], O[k], I[k], A[k] = self._evaluate_step(tk, V[k], parameters)
            R_dot[k] = kappa_R * ((I[k] - O[k]) - R[k])
            if k < self.t_steps - 1:
                V[k + 1] = V[k] + self.dt * R[k]
                R[k + 1] = R[k] + self.dt * R_dot[k]

        return {
            "t": self.t.copy(), "V": V, "R": R, "R_dot": R_dot,
            "U": U, "B_O": B_O, "B_I": B_I, "PE": PE,
            "P_I": P_I, "P_O": P_O, "P_A": P_A,
            "O": O, "I": I, "A": A,
        }

    def fokker_planc_solver(self, init_cond, parameters, initial_density, v_bounds=(-0.5, 1.5), r_bounds=(-1.0, 1.0), v_steps=161, r_steps=161, cfl=0.8):
        return self.fp.solve(
            self.t,
            init_cond,
            parameters,
            initial_density=initial_density,
            v_bounds=v_bounds,
            r_bounds=r_bounds,
            v_steps=v_steps,
            r_steps=r_steps,
            cfl=cfl,
        )
