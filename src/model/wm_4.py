import numpy as np

from src.method.fokker_planc import wm_fokker_planc
from src.model.base.wm import wm
from src.params.wm_4 import optimized_qp_params, qp_params
from src.solver.active_set import active_set
from src.validation.model.wm_4 import wm_4_validation


class wm_4(wm):
    """Water Model 4.

    P_R = [1, F, U, R], dR/dt = r @ P_R
    O = O(Time)

    dF = R dt
    dR = (r_0 + r_F F + r_U U + r_R R) dt + (sigma_0 + sigma_I I) dW
    I = O + R
    """

    _FIT_EPS = 1e-8
    _FIT_RIDGE = 1e-10

    def __init__(self, T, t_steps, O):
        T, t_steps, O = wm_4_validation.constructor(T, t_steps, O)
        self.T = T
        self.t_steps = t_steps
        self.t = np.linspace(T[0], T[1], t_steps)
        self.dt = self.t[1] - self.t[0]
        self.O = O
        self.fp = wm_fokker_planc()

    def dR(self, P_R, r):
        P_R = wm_4_validation.vector(P_R, 4, "P_R")
        r = wm_4_validation.r_coefficients(r)
        return r @ P_R

    @staticmethod
    def I(O, R):
        O = wm_4_validation._finite_scalar(O, "O")
        R = wm_4_validation._finite_scalar(R, "R")
        return O + R

    def A(self, P_A, gamma):
        P_A, gamma = wm_4_validation.linear_map(P_A, gamma, 4, "P_A", "gamma")
        return gamma @ P_A

    @staticmethod
    def equivalent_q(r):
        r = wm_4_validation.r_coefficients(r, enforce_equilibrium=False)
        kappa_R = -r[3]
        return r[:3] / kappa_R, kappa_R

    @classmethod
    def expected_equilibrium(cls, U, r):
        U = wm_4_validation.signal_value(U, "U")
        r = wm_4_validation.r_coefficients(r)
        return -(r[0] + r[2] * U) / r[1]

    @staticmethod
    def drift(F, R, U, r):
        F = wm_4_validation._finite_scalar(F, "F")
        R = wm_4_validation._finite_scalar(R, "R")
        U = wm_4_validation.signal_value(U, "U")
        r = wm_4_validation.r_coefficients(r)
        return np.array([R, r @ np.array([1.0, F, U, R], dtype=float)], dtype=float)

    @staticmethod
    def diffusion(I, sigma_0, sigma_I):
        I = wm_4_validation._finite_scalar(I, "I")
        sigma_0 = wm_4_validation._finite_scalar(sigma_0, "sigma_0")
        sigma_I = wm_4_validation._finite_scalar(sigma_I, "sigma_I")
        if I < 0.0: raise ValueError("wm_4 requires I >= 0 when evaluating input-dependent diffusion.")
        if sigma_0 <= 0.0: raise ValueError("sigma_0 must be positive.")
        if sigma_I < 0.0: raise ValueError("sigma_I must be nonnegative.")
        return np.array([0.0, sigma_0 + sigma_I * I], dtype=float)

    def _sample_signals(self, parameters):
        U, PE = np.empty(self.t_steps), np.empty(self.t_steps)
        Time, O = np.empty(self.t_steps), np.empty(self.t_steps)
        for k, tk in enumerate(self.t):
            U[k] = wm_4_validation.signal_value(parameters["U"](tk), "U(t)")
            PE[k] = wm_4_validation.signal_value(parameters["PE"](tk), "PE(t)")
            Time[k] = wm_4_validation.finite_feature_input(parameters["Time"](tk), "Time(t)")
            O[k] = wm_4_validation.feature_value(self.O(Time[k]), "O(Time(t))")
        return U, PE, Time, O

    def _solve_deterministic(self, init_cond, U, PE, Time, O, r, gamma=None, enforce_equilibrium=True):
        F, R, R_dot = np.empty(self.t_steps), np.empty(self.t_steps), np.empty(self.t_steps)
        I, Q = np.empty(self.t_steps), np.empty(self.t_steps)
        A = np.full(self.t_steps, np.nan)
        P_R = np.empty((self.t_steps, 4))
        P_A = np.full((self.t_steps, 4), np.nan)
        r = wm_4_validation.r_coefficients(r, enforce_equilibrium=enforce_equilibrium)
        q, kappa_R = self.equivalent_q(r)
        F[0], R[0] = init_cond

        for k in range(self.t_steps):
            P_R[k] = np.array([1.0, F[k], U[k], R[k]], dtype=float)
            R_dot[k] = r @ P_R[k]
            Q[k] = q @ np.array([1.0, F[k], U[k]], dtype=float)
            I[k] = O[k] + R[k]
            if gamma is not None:
                P_A[k] = np.array([1.0 - U[k], I[k], F[k], 1.0 - PE[k]], dtype=float)
                A[k] = gamma @ P_A[k]
            if k < self.t_steps - 1:
                F[k + 1] = F[k] + self.dt * R[k]
                R[k + 1] = R[k] + self.dt * R_dot[k]

        return {
            "t": self.t.copy(), "F": F, "R": R, "R_dot": R_dot,
            "U": np.asarray(U, dtype=float).copy(), "PE": np.asarray(PE, dtype=float).copy(),
            "Time": np.asarray(Time, dtype=float).copy(), "O": np.asarray(O, dtype=float).copy(),
            "P_R": P_R, "P_A": P_A, "I": I, "A": A, "r": r.copy(),
            "Q": Q, "q": q.copy(), "kappa_R": float(kappa_R),
        }

    def deterministic_solver(self, init_cond, parameters):
        init_cond, parameters = wm_4_validation.deterministic_inputs(init_cond, parameters)
        U, PE, Time, O = self._sample_signals(parameters)
        return self._solve_deterministic(
            init_cond, U, PE, Time, O, parameters["r"], gamma=parameters["gamma"],
            enforce_equilibrium=bool(parameters.get("enforce_equilibrium", True)),
        )

    def fokker_planc_solver(self, init_cond, parameters, initial_density, f_bounds=(-0.5, 1.5), r_bounds=(-1.0, 1.0), f_steps=161, r_steps=161, cfl=0.8):
        _, parameters = wm_4_validation.fokker_planc_inputs(init_cond, parameters)
        r = parameters["r"]
        sigma_0, sigma_I = parameters["sigma_0"], parameters["sigma_I"]

        def fields(time):
            U = float(parameters["U"](time))
            O = float(self.O(parameters["Time"](time)))
            return U, O

        def drift_f(time, F, R): return R

        def drift_r(time, F, R):
            U, _ = fields(time)
            return r[0] + r[1] * F + r[2] * U + r[3] * R

        def diffusion_r(time, F, R):
            _, O = fields(time)
            I = O + R
            if np.any(np.asarray(I) < 0.0): raise ValueError("wm_4 produced negative I on the selected numerical domain.")
            return sigma_0 + sigma_I * I

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

    def fit_params_qp(self, input_data, initial_conditions, **kwargs): return qp_params(self).fit(input_data, initial_conditions, **kwargs)

    def fit_params_optimized_qp(self, input_data, initial_conditions, **kwargs): return optimized_qp_params(self).fit(input_data, initial_conditions, **kwargs)

    def fit_params(self, input_data, initial_conditions, **kwargs): return self.fit_params_qp(input_data, initial_conditions, **kwargs)
