import numpy as np

from src.method.fokker_planc import wm_fokker_planc
from src.model.base.wm import wm
from src.params.wm_4 import brute_params
from src.solver.active_set import active_set
from src.validation.model.wm_4 import wm_4_validation


class wm_4(wm):
    """Water Model 4.

    P_Q = [1, F, U, L(Time), H(Temp)], Q = q @ P_Q
    P_O = [F, 1-U, L(Time), H(Temp)], O = omega @ P_O

    dF = R dt
    dR = kappa_R (Q - R) dt + (sigma_0 + sigma_O O) dW
    I = O + R
    """

    _FIT_EPS = 1e-8
    _FIT_RIDGE = 1e-10

    def __init__(self, T, t_steps, L, H=None):
        T, t_steps, L, H = wm_4_validation.constructor(T, t_steps, L, H)
        self.T = T
        self.t_steps = t_steps
        self.t = np.linspace(T[0], T[1], t_steps)
        self.dt = self.t[1] - self.t[0]
        self.L = L
        self.H = (lambda _x: 0.0) if H is None else H
        self.fp = wm_fokker_planc()

    def Q(self, P_Q, q):
        P_Q = wm_4_validation.vector(P_Q, 5, "P_Q")
        q = wm_4_validation.q_coefficients(q)
        return q @ P_Q

    def O(self, P_O, omega):
        P_O = wm_4_validation.vector(P_O, 4, "P_O")
        omega = wm_4_validation.omega_coefficients(omega)
        return omega @ P_O

    @staticmethod
    def I(O, R):
        O = wm_4_validation._finite_scalar(O, "O")
        R = wm_4_validation._finite_scalar(R, "R")
        return O + R

    def A(self, P_A, gamma):
        P_A, gamma = wm_4_validation.linear_map(P_A, gamma, 4, "P_A", "gamma")
        return gamma @ P_A

    @classmethod
    def expected_equilibrium(cls, U, L, H, q):
        U = wm_4_validation.signal_value(U, "U")
        L = wm_4_validation.feature_value(L, "L")
        H = wm_4_validation.feature_value(H, "H")
        q = wm_4_validation.q_coefficients(q)
        return -(q[0] + q[2] * U + q[3] * L + q[4] * H) / q[1]

    @staticmethod
    def drift(F, R, U, L, H, q, kappa_R):
        F = wm_4_validation._finite_scalar(F, "F")
        R = wm_4_validation._finite_scalar(R, "R")
        U = wm_4_validation.signal_value(U, "U")
        L = wm_4_validation.feature_value(L, "L")
        H = wm_4_validation.feature_value(H, "H")
        q = wm_4_validation.q_coefficients(q)
        kappa_R = wm_4_validation._finite_scalar(kappa_R, "kappa_R")
        if kappa_R <= 0.0: raise ValueError("kappa_R must be positive.")
        Q = q @ np.array([1.0, F, U, L, H], dtype=float)
        return np.array([R, kappa_R * (Q - R)], dtype=float)

    @staticmethod
    def diffusion(F, U, L, H, omega, sigma_0, sigma_O):
        F = wm_4_validation._finite_scalar(F, "F")
        U = wm_4_validation.signal_value(U, "U")
        L = wm_4_validation.feature_value(L, "L")
        H = wm_4_validation.feature_value(H, "H")
        omega = wm_4_validation.omega_coefficients(omega)
        sigma_0 = wm_4_validation._finite_scalar(sigma_0, "sigma_0")
        sigma_O = wm_4_validation._finite_scalar(sigma_O, "sigma_O")
        if sigma_0 <= 0.0: raise ValueError("sigma_0 must be positive.")
        if sigma_O < 0.0: raise ValueError("sigma_O must be nonnegative.")
        O = omega @ np.array([F, 1.0 - U, L, H], dtype=float)
        g = sigma_0 + sigma_O * O
        if g < 0.0: raise ValueError("wm_4 diffusion scale g(O) must be nonnegative.")
        return np.array([0.0, g], dtype=float)

    def _sample_signals(self, parameters):
        U, PE = np.empty(self.t_steps), np.empty(self.t_steps)
        Time, Temp = np.empty(self.t_steps), np.empty(self.t_steps)
        L, H = np.empty(self.t_steps), np.empty(self.t_steps)
        for k, tk in enumerate(self.t):
            U[k] = wm_4_validation.signal_value(parameters["U"](tk), "U(t)")
            PE[k] = wm_4_validation.signal_value(parameters["PE"](tk), "PE(t)")
            Time[k] = wm_4_validation.finite_feature_input(parameters["Time"](tk), "Time(t)")
            Temp[k] = wm_4_validation.finite_feature_input(parameters["Temp"](tk), "Temp(t)")
            L[k] = wm_4_validation.feature_value(self.L(Time[k]), "L(Time(t))")
            H[k] = wm_4_validation.feature_value(self.H(Temp[k]), "H(Temp(t))")
        return U, PE, Time, Temp, L, H

    def _solve_deterministic(self, init_cond, U, PE, Time, Temp, L, H, q, omega, kappa_R, gamma=None):
        F, R, R_dot = np.empty(self.t_steps), np.empty(self.t_steps), np.empty(self.t_steps)
        Q, O, I = np.empty(self.t_steps), np.empty(self.t_steps), np.empty(self.t_steps)
        A = np.full(self.t_steps, np.nan)
        P_Q, P_O = np.empty((self.t_steps, 5)), np.empty((self.t_steps, 4))
        P_A = np.full((self.t_steps, 4), np.nan)
        F[0], R[0] = init_cond

        for k in range(self.t_steps):
            P_Q[k] = np.array([1.0, F[k], U[k], L[k], H[k]], dtype=float)
            P_O[k] = np.array([F[k], 1.0 - U[k], L[k], H[k]], dtype=float)
            Q[k], O[k] = q @ P_Q[k], omega @ P_O[k]
            I[k] = O[k] + R[k]
            if gamma is not None:
                P_A[k] = np.array([1.0 - U[k], I[k], F[k], 1.0 - PE[k]], dtype=float)
                A[k] = gamma @ P_A[k]
            R_dot[k] = kappa_R * (Q[k] - R[k])
            if k < self.t_steps - 1:
                F[k + 1] = F[k] + self.dt * R[k]
                R[k + 1] = R[k] + self.dt * R_dot[k]

        return {
            "t": self.t.copy(), "F": F, "R": R, "R_dot": R_dot,
            "U": np.asarray(U, dtype=float).copy(), "PE": np.asarray(PE, dtype=float).copy(),
            "Time": np.asarray(Time, dtype=float).copy(), "Temp": np.asarray(Temp, dtype=float).copy(),
            "L": np.asarray(L, dtype=float).copy(), "H": np.asarray(H, dtype=float).copy(),
            "P_Q": P_Q, "P_O": P_O, "P_A": P_A, "Q": Q, "O": O, "I": I, "A": A,
            "q": np.asarray(q, dtype=float).copy(), "omega": np.asarray(omega, dtype=float).copy(),
        }

    def deterministic_solver(self, init_cond, parameters):
        init_cond, parameters = wm_4_validation.deterministic_inputs(init_cond, parameters)
        U, PE, Time, Temp, L, H = self._sample_signals(parameters)
        return self._solve_deterministic(init_cond, U, PE, Time, Temp, L, H, parameters["q"], parameters["omega"], parameters["kappa_R"], gamma=parameters["gamma"])

    def fokker_planc_solver(self, init_cond, parameters, initial_density, f_bounds=(-0.5, 1.5), r_bounds=(-1.0, 1.0), f_steps=161, r_steps=161, cfl=0.8):
        _, parameters = wm_4_validation.fokker_planc_inputs(init_cond, parameters)
        q, omega = parameters["q"], parameters["omega"]
        kappa_R, sigma_0, sigma_O = parameters["kappa_R"], parameters["sigma_0"], parameters["sigma_O"]

        def fields(time, F):
            U = float(parameters["U"](time))
            L = float(self.L(parameters["Time"](time)))
            H = float(self.H(parameters["Temp"](time)))
            Q = q[0] + q[1] * F + q[2] * U + q[3] * L + q[4] * H
            O = omega[0] * F + omega[1] * (1.0 - U) + omega[2] * L + omega[3] * H
            return Q, O

        def drift_f(time, F, R): return R

        def drift_r(time, F, R):
            Q, _ = fields(time, F)
            return kappa_R * (Q - R)

        def diffusion_r(time, F, R):
            _, O = fields(time, F)
            return sigma_0 + sigma_O * O

        result = self.fp.solve(self.t, initial_density, drift_f, drift_r, diffusion_r, v_bounds=f_bounds, r_bounds=r_bounds, v_steps=f_steps, r_steps=r_steps, cfl=cfl)
        result.update({
            "F_grid": result["v_grid"], "marginal_F": result["marginal_v"],
            "mean_F": result["mean_v"], "var_F": result["var_v"],
        })
        return result

    @staticmethod
    def _least_squares_qp(X, y, A_ub=None, b_ub=None, A_eq=None, b_eq=None, lb=None, ub=None, ridge=1e-10, x0=None, max_iter=10000):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, scale = X.shape[1], max(1, X.shape[0])
        H = 2.0 * (X.T @ X) / scale + ridge * np.eye(n)
        f = -2.0 * (X.T @ y) / scale
        return active_set.solve(H, f, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub, x0=x0, tol=1e-9, max_iter=max_iter)

    def fit_params_brute(self, input_data, initial_conditions, **kwargs): return brute_params(self).fit(input_data, initial_conditions, **kwargs)

    def fit_params(self, input_data, initial_conditions, **kwargs): return self.fit_params_brute(input_data, initial_conditions, **kwargs)
