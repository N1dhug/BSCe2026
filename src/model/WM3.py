import numpy as np

from src.model.base.WM import WM
from src.params.WM3 import BruteParams
from src.solver.ActiveSet import ActiveSet
from src.validation.model.WM3 import WM3Validation

class WM3(WM):
    """Water Model 3 with direct exogenous output features.

    P_O = [V, 1-U, L(Time), H(Chill_factor)], O = beta @ P_O
    P_I = [1-V, 1-U, O], I = alpha @ P_I
    P_A = [1-U, I, V, 1-PE], A = gamma @ P_A

    dV/dt = R
    dR/dt = kappa_R * (I - O - R)

    L and H are supplied feature functions. Parameter estimation is delegated
    to src.params.WM3.
    """

    _FIT_EPS = 1e-8
    _FIT_RIDGE = 1e-10

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
    def reduced_coefficients(alpha, beta):
        # Reduce alpha/beta to affine input and drift maps
        alpha, beta = WM3Validation.reduction_inputs(alpha, beta)
        a_V, a_U, a_O = alpha
        b_V, b_U, b_L, b_H = beta

        input_coeff = np.array([a_V + a_U + a_O * b_U, a_O * b_L, a_O * b_H, -a_V + a_O * b_V, -a_U - a_O * b_U], dtype=float)
        drift_coeff = np.array([a_V + a_U + (a_O - 1.0) * b_U, (a_O - 1.0) * b_L, (a_O - 1.0) * b_H, -a_V + (a_O - 1.0) * b_V, (1.0 - a_O) * b_U - a_U], dtype=float)
        return input_coeff, drift_coeff

    @classmethod
    def reduced_drift_coefficients(cls, L_value, H_value, alpha, beta):
        # Compute reduced drift
        L_value, H_value = WM3Validation.feature_value(L_value, "L_value"), WM3Validation.feature_value(H_value, "H_value")
        _, d = cls.reduced_coefficients(alpha, beta)
        return d[0] + d[1] * L_value + d[2] * H_value, d[3], d[4]

    @classmethod
    def expected_equilibrium(cls, L_value, H_value, U, alpha, beta):
        # Compute expected equilibrium
        U = WM3Validation.signal_value(U, "U")
        d0, dV, dU = cls.reduced_drift_coefficients(L_value, H_value, alpha, beta)
        WM3Validation.stable_drift_coefficient(dV)
        return -(d0 + dU * U) / dV

    def _sample_signals(self, parameters):
        # Sample model signals
        U = np.empty(self.t_steps)
        PE = np.empty(self.t_steps)
        Time = np.empty(self.t_steps)
        Chill = np.empty(self.t_steps)
        L = np.empty(self.t_steps)
        H = np.empty(self.t_steps)

        for k, tk in enumerate(self.t):
            U[k] = WM3Validation.signal_value(parameters["U"](tk), "U(t)")
            PE[k] = WM3Validation.signal_value(parameters["PE"](tk), "PE(t)")
            Time[k] = WM3Validation.finite_feature_input(parameters["Time"](tk), "Time(t)")
            Chill[k] = WM3Validation.finite_feature_input(parameters["Chill_factor"](tk), "Chill_factor(t)")
            L[k] = WM3Validation.feature_value(self.L(Time[k]), "L(Time(t))")
            H[k] = WM3Validation.feature_value(self.H(Chill[k]), "H(Chill_factor(t))")

        return U, PE, Time, Chill, L, H

    def _solve_deterministic(self, init_cond, U, PE, Time, Chill, L, H, input_coeff, drift_coeff, kappa_R, gamma=None):
        # Solve deterministic dynamics in reduced coordinates
        V = np.empty(self.t_steps)
        R = np.empty(self.t_steps)
        R_dot = np.empty(self.t_steps)
        O = np.empty(self.t_steps)
        I = np.empty(self.t_steps)
        D = np.empty(self.t_steps)
        A = np.full(self.t_steps, np.nan)
        P_I = np.empty((self.t_steps, 3))
        P_O = np.empty((self.t_steps, 4))
        P_A = np.full((self.t_steps, 4), np.nan)

        V[0], R[0] = init_cond

        for k in range(self.t_steps):
            point = np.array([1.0, L[k], H[k], V[k], U[k]], dtype=float)
            I[k] = input_coeff @ point
            D[k] = drift_coeff @ point
            O[k] = I[k] - D[k]

            P_O[k] = np.array([V[k], 1.0 - U[k], L[k], H[k]])
            P_I[k] = np.array([1.0 - V[k], 1.0 - U[k], O[k]])

            if gamma is not None:
                P_A[k] = np.array([1.0 - U[k], I[k], V[k], 1.0 - PE[k]])
                A[k] = gamma @ P_A[k]

            R_dot[k] = kappa_R * (D[k] - R[k])
            if k < self.t_steps - 1:
                V[k + 1] = V[k] + self.dt * R[k]
                R[k + 1] = R[k] + self.dt * R_dot[k]

        return {"t": self.t.copy(), "V": V, "R": R, "R_dot": R_dot, "U": np.asarray(U, dtype=float).copy(), "PE": np.asarray(PE, dtype=float).copy(), "Time": np.asarray(Time, dtype=float).copy(), "Chill_factor": np.asarray(Chill, dtype=float).copy(), "L": np.asarray(L, dtype=float).copy(), "H": np.asarray(H, dtype=float).copy(), "P_I": P_I, "P_O": P_O, "P_A": P_A, "O": O, "I": I, "D": D, "A": A, "input_coeff": np.asarray(input_coeff, dtype=float).copy(), "drift_coeff": np.asarray(drift_coeff, dtype=float).copy()}

    def deterministic_solver(self, init_cond, parameters):
        # Solve deterministic model
        init_cond, parameters = WM3Validation.deterministic_inputs(init_cond, parameters)
        U, PE, Time, Chill, L, H = self._sample_signals(parameters)
        input_coeff, drift_coeff = self.reduced_coefficients(parameters["alpha"], parameters["beta"])
        return self._solve_deterministic(init_cond, U, PE, Time, Chill, L, H, input_coeff, drift_coeff, parameters["kappa_R"], gamma=parameters["gamma"])

    @staticmethod
    def _least_squares_qp(X, y, A_ub=None, b_ub=None, A_eq=None, b_eq=None, lb=None, ub=None, ridge=1e-10, x0=None, max_iter=10000):
        # Build and solve QP
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = X.shape[1]
        scale = max(1, X.shape[0])
        H = 2.0 * (X.T @ X) / scale + ridge * np.eye(n)
        f = -2.0 * (X.T @ y) / scale
        return ActiveSet.solve(H, f, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub, x0=x0, tol=1e-9, max_iter=max_iter)

    def fit_params_brute(self, input_data, initial_conditions, **kwargs):
        # Fit WM3 with the grid-refined latent-state brute-force method
        return BruteParams(self).fit(input_data, initial_conditions, **kwargs)

    def fit_params(self, input_data, initial_conditions, **kwargs):
        # Default to brute-force WM3 fitting
        return self.fit_params_brute(input_data, initial_conditions, **kwargs)
