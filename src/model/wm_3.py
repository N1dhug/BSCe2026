import numpy as np

from src.method.wm_3_fokker_planc import WM3FokkerPlanc
from src.model.base.wm import WM
from src.params.wm_3 import BruteParams
from src.solver.active_set import ActiveSet
from src.validation.model.wm_3 import WM3Validation


class WM3(WM):
    """Complete Water Model 3.

    P_I = [1-V, 1-U, O], I = alpha @ P_I
    P_O = [V, 1-U, L(Time), H(Temp)], O = beta @ P_O

    dV = R dt
    dR = kappa_R (I - O - R) dt + (sigma_0 + sigma_O O) dW

    The legacy L-only formulation remains supported by omitting H and using a
    three-entry beta vector [beta_V, beta_U, beta_L].
    """

    _FIT_EPS = 1e-8
    _FIT_RIDGE = 1e-10

    def __init__(self, T, t_steps, L, H=None):
        T, t_steps, L, H = WM3Validation.constructor(T, t_steps, L, H)
        self.T = T
        self.t_steps = t_steps
        self.t = np.linspace(T[0], T[1], t_steps)
        self.dt = self.t[1] - self.t[0]
        self.L = L
        self.H = (lambda _x: 0.0) if H is None else H
        self.fp = WM3FokkerPlanc()

    def I(self, P_I, alpha):
        P_I, alpha = WM3Validation.linear_map(P_I, alpha, 3, "P_I", "alpha")
        return alpha @ P_I

    def O(self, P_O, beta):
        beta = WM3Validation.beta_coefficients(beta)
        P_O = np.asarray(P_O, dtype=float).reshape(-1)
        if P_O.size == 3: P_O = np.append(P_O, 0.0)
        if P_O.size != 4 or not np.all(np.isfinite(P_O)): raise ValueError("P_O must contain [V, 1-U, L, H].")
        return beta @ P_O

    def A(self, P_A, gamma):
        P_A, gamma = WM3Validation.linear_map(P_A, gamma, 4, "P_A", "gamma")
        return gamma @ P_A

    @staticmethod
    def reduced_coefficients(alpha, beta):
        alpha, beta = WM3Validation.reduction_inputs(alpha, beta)
        a_V, a_U, a_O = alpha
        b_V, b_U, b_L, b_H = beta

        input_coeff = np.array([
            a_V + a_U + a_O * b_U,
            a_O * b_L,
            -a_V + a_O * b_V,
            -a_U - a_O * b_U,
            a_O * b_H,
        ], dtype=float)

        drift_coeff = np.array([
            a_V + a_U + (a_O - 1.0) * b_U,
            (a_O - 1.0) * b_L,
            -a_V + (a_O - 1.0) * b_V,
            (1.0 - a_O) * b_U - a_U,
            (a_O - 1.0) * b_H,
        ], dtype=float)
        return input_coeff, drift_coeff

    @classmethod
    def reduced_drift_coefficients(cls, L_value, alpha, beta, H_value=0.0):
        L_value = WM3Validation.feature_value(L_value, "L_value")
        H_value = WM3Validation.feature_value(H_value, "H_value")
        _, d = cls.reduced_coefficients(alpha, beta)
        return d[0] + d[1] * L_value + d[4] * H_value, d[2], d[3]

    @classmethod
    def expected_equilibrium(cls, L_value, U, alpha, beta, H_value=0.0):
        U = WM3Validation.signal_value(U, "U")
        d0, dV, dU = cls.reduced_drift_coefficients(L_value, alpha, beta, H_value)
        WM3Validation.stable_drift_coefficient(dV)
        return -(d0 + dU * U) / dV

    @staticmethod
    def drift(V, R, U, L, H, alpha, beta, kappa_R):
        V = float(V)
        R = float(R)
        U = WM3Validation.signal_value(U, "U")
        L = WM3Validation.feature_value(L, "L")
        H = WM3Validation.feature_value(H, "H")
        kappa_R = WM3Validation._finite_scalar(kappa_R, "kappa_R")
        if kappa_R <= 0.0: raise ValueError("kappa_R must be positive.")

        _, d = WM3.reduced_coefficients(alpha, beta)
        D = d @ np.array([1.0, L, V, U, H], dtype=float)
        return np.array([R, kappa_R * (D - R)], dtype=float)

    @staticmethod
    def diffusion(V, U, L, H, beta, sigma_0, sigma_O):
        V = float(V)
        U = WM3Validation.signal_value(U, "U")
        L = WM3Validation.feature_value(L, "L")
        H = WM3Validation.feature_value(H, "H")
        beta = WM3Validation.beta_coefficients(beta)
        sigma_0 = WM3Validation._finite_scalar(sigma_0, "sigma_0")
        sigma_O = WM3Validation._finite_scalar(sigma_O, "sigma_O")
        if sigma_0 <= 0.0: raise ValueError("sigma_0 must be positive.")
        if sigma_O < 0.0: raise ValueError("sigma_O must be nonnegative.")

        O = beta @ np.array([V, 1.0 - U, L, H], dtype=float)
        g = sigma_0 + sigma_O * O
        if g < 0.0: raise ValueError("WM3 diffusion scale g(O) must be nonnegative.")
        return np.array([0.0, g], dtype=float)

    def _sample_signals(self, parameters):
        U = np.empty(self.t_steps)
        PE = np.empty(self.t_steps)
        Time = np.empty(self.t_steps)
        Temp = np.empty(self.t_steps)
        L = np.empty(self.t_steps)
        H = np.empty(self.t_steps)

        for k, tk in enumerate(self.t):
            U[k] = WM3Validation.signal_value(parameters["U"](tk), "U(t)")
            PE[k] = WM3Validation.signal_value(parameters["PE"](tk), "PE(t)")
            Time[k] = WM3Validation.finite_feature_input(parameters["Time"](tk), "Time(t)")
            Temp[k] = WM3Validation.finite_feature_input(parameters["Temp"](tk), "Temp(t)")
            L[k] = WM3Validation.feature_value(self.L(Time[k]), "L(Time(t))")
            H[k] = WM3Validation.feature_value(self.H(Temp[k]), "H(Temp(t))")

        return U, PE, Time, Temp, L, H

    def _solve_deterministic(self, init_cond, U, PE, Time, L, input_coeff,
                             drift_coeff, kappa_R, gamma=None, H=None, Temp=None):
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

        H = np.zeros(self.t_steps, dtype=float) if H is None else np.asarray(H, dtype=float)
        Temp = np.zeros(self.t_steps, dtype=float) if Temp is None else np.asarray(Temp, dtype=float)
        V[0], R[0] = init_cond

        for k in range(self.t_steps):
            point = np.array([1.0, L[k], V[k], U[k], H[k]], dtype=float)
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

        return {
            "t": self.t.copy(), "V": V, "R": R, "R_dot": R_dot,
            "U": np.asarray(U, dtype=float).copy(), "PE": np.asarray(PE, dtype=float).copy(),
            "Time": np.asarray(Time, dtype=float).copy(), "Temp": Temp.copy(),
            "L": np.asarray(L, dtype=float).copy(), "H": H.copy(),
            "P_I": P_I, "P_O": P_O, "P_A": P_A,
            "O": O, "I": I, "D": D, "A": A,
            "input_coeff": np.asarray(input_coeff, dtype=float).copy(),
            "drift_coeff": np.asarray(drift_coeff, dtype=float).copy(),
        }

    def deterministic_solver(self, init_cond, parameters):
        init_cond, parameters = WM3Validation.deterministic_inputs(init_cond, parameters)
        U, PE, Time, Temp, L, H = self._sample_signals(parameters)
        input_coeff, drift_coeff = self.reduced_coefficients(parameters["alpha"], parameters["beta"])
        return self._solve_deterministic(
            init_cond, U, PE, Time, L, input_coeff, drift_coeff,
            parameters["kappa_R"], gamma=parameters["gamma"], H=H, Temp=Temp
        )

    def fokker_planc_solver(self, init_cond, parameters, initial_density,
                            v_bounds=(-0.5, 1.5), r_bounds=(-1.0, 1.0),
                            v_steps=161, r_steps=161, cfl=0.8):
        init_cond, validated = WM3Validation.fokker_planc_inputs(init_cond, parameters)
        _, drift_coeff = self.reduced_coefficients(validated["alpha"], validated["beta"])

        fp_parameters = dict(validated)
        fp_parameters["d_coeff"] = drift_coeff
        fp_parameters["L"] = self.L
        fp_parameters["H"] = self.H

        return self.fp.solve(
            self.t, init_cond, fp_parameters, initial_density=initial_density,
            v_bounds=v_bounds, r_bounds=r_bounds,
            v_steps=v_steps, r_steps=r_steps, cfl=cfl
        )

    @staticmethod
    def _least_squares_qp(X, y, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                          lb=None, ub=None, ridge=1e-10, x0=None, max_iter=10000):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = X.shape[1]
        scale = max(1, X.shape[0])
        H = 2.0 * (X.T @ X) / scale + ridge * np.eye(n)
        f = -2.0 * (X.T @ y) / scale
        return ActiveSet.solve(
            H, f, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            lb=lb, ub=ub, x0=x0, tol=1e-9, max_iter=max_iter
        )

    def fit_params_brute(self, input_data, initial_conditions, **kwargs):
        return BruteParams(self).fit(input_data, initial_conditions, **kwargs)

    def fit_params(self, input_data, initial_conditions, **kwargs):
        return self.fit_params_brute(input_data, initial_conditions, **kwargs)
