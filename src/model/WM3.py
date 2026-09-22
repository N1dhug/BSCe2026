import numpy as np

from src.model.base.WM import WM
from src.validation.model.WM3 import WM3Validation

class WM3(WM):
    """Water Model 3 with direct exogenous output features.

    P_O = [V, 1-U, L(Time), H(Chill_factor)], O = beta @ P_O
    P_I = [1-V, 1-U, O], I = alpha @ P_I
    P_A = [1-U, I, V, 1-PE], A = gamma @ P_A

    dV/dt = R
    dR/dt = kappa_R * (I - O - R)

    L and H are supplied feature functions. WM3 performs no parameter fitting.
    """

    def __init__(self, T, t_steps, L, H):
        # Initialize variables
        T, t_steps, L, H = WM3Validation.constructor(T, t_steps, L, H)
        self.T = T
        self.t_steps = t_steps
        self.t = np.linspace(T[0], T[1], t_steps)
        self.dt = self.t[1] - self.t[0]
        self.L = L
        self.H = H

    def I(self, P_I, alpha):
        # Evaluate input map
        P_I, alpha = WM3Validation.linear_map(P_I, alpha, 3, "P_I", "alpha")
        return alpha @ P_I

    def O(self, P_O, beta):
        # Evaluate output map
        P_O, beta = WM3Validation.linear_map(P_O, beta, 4, "P_O", "beta")
        return beta @ P_O

    def A(self, P_A, gamma):
        # Evaluate availability map
        P_A, gamma = WM3Validation.linear_map(P_A, gamma, 4, "P_A", "gamma")
        return gamma @ P_A

    @staticmethod
    def reduced_drift_coefficients(L_value, H_value, alpha, beta):
        # Compute reduced drift
        L_value, H_value, alpha, beta = WM3Validation.reduced_drift_inputs(L_value, H_value, alpha, beta)
        a_V, a_U, a_O = alpha
        b_V, b_U, b_L, b_H = beta

        d0 = a_V + a_U + (a_O - 1.0) * (b_U + b_L * L_value + b_H * H_value)
        dV = -a_V + (a_O - 1.0) * b_V
        dU = (1.0 - a_O) * b_U - a_U
        return d0, dV, dU

    @classmethod
    def expected_equilibrium(cls, L_value, H_value, U, alpha, beta):
        # Compute expected equilibrium
        U = WM3Validation.signal_value(U, "U")
        d0, dV, dU = cls.reduced_drift_coefficients(L_value, H_value, alpha, beta)
        WM3Validation.stable_drift_coefficient(dV)
        return -(d0 + dU * U) / dV

    def _evaluate_step(self, t, V, parameters):
        # Evaluate model step
        U = WM3Validation.signal_value(parameters["U"](t), "U(t)")
        PE = WM3Validation.signal_value(parameters["PE"](t), "PE(t)")
        Time = WM3Validation.finite_feature_input(parameters["Time"](t), "Time(t)")
        Chill = WM3Validation.finite_feature_input(parameters["Chill_factor"](t), "Chill_factor(t)")
        Lt = WM3Validation.feature_value(self.L(Time), "L(Time(t))")
        Ht = WM3Validation.feature_value(self.H(Chill), "H(Chill_factor(t))")

        alpha = parameters["alpha"]
        beta = parameters["beta"]
        gamma = parameters["gamma"]

        P_O = np.array([V, 1.0 - U, Lt, Ht], dtype=float)
        O = self.O(P_O, beta)
        P_I = np.array([1.0 - V, 1.0 - U, O], dtype=float)
        I = self.I(P_I, alpha)
        P_A = np.array([1.0 - U, I, V, 1.0 - PE], dtype=float)
        A = self.A(P_A, gamma)
        return U, PE, Time, Chill, Lt, Ht, P_I, P_O, P_A, O, I, A

    def deterministic_solver(self, init_cond, parameters):
        # Solve deterministic model
        init_cond, parameters = WM3Validation.deterministic_inputs(init_cond, parameters)

        V = np.empty(self.t_steps)
        R = np.empty(self.t_steps)
        R_dot = np.empty(self.t_steps)
        U = np.empty(self.t_steps)
        PE = np.empty(self.t_steps)
        Time = np.empty(self.t_steps)
        Chill = np.empty(self.t_steps)
        Lt = np.empty(self.t_steps)
        Ht = np.empty(self.t_steps)
        O = np.empty(self.t_steps)
        I = np.empty(self.t_steps)
        D = np.empty(self.t_steps)
        A = np.empty(self.t_steps)
        P_I = np.empty((self.t_steps, 3))
        P_O = np.empty((self.t_steps, 4))
        P_A = np.empty((self.t_steps, 4))

        V[0], R[0] = init_cond
        kappa_R = parameters["kappa_R"]

        for k, tk in enumerate(self.t):
            U[k], PE[k], Time[k], Chill[k], Lt[k], Ht[k], P_I[k], P_O[k], P_A[k], O[k], I[k], A[k] = self._evaluate_step(tk, V[k], parameters)
            D[k] = I[k] - O[k]
            R_dot[k] = kappa_R * (D[k] - R[k])

            if k < self.t_steps - 1:
                V[k + 1] = V[k] + self.dt * R[k]
                R[k + 1] = R[k] + self.dt * R_dot[k]

        return {"t": self.t.copy(), "V": V, "R": R, "R_dot": R_dot, "U": U, "PE": PE, "Time": Time, "Chill_factor": Chill, "L": Lt, "H": Ht, "P_I": P_I, "P_O": P_O, "P_A": P_A, "O": O, "I": I, "D": D, "A": A}
