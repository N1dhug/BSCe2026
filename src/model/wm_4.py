import numpy as np

from src.model.base.wm import wm
from src.params.wm_4 import brute_force_params, optimized_qp_params, qp_params
from src.solver.active_set import active_set
from src.validation.model.wm_4 import wm_4_validation


class wm_4(wm):
    """Water Model 4 with signed flexibility and price domains.

    P_R = [F, U, R], dR/dt = r @ P_R
    O = O(Time)

    dF = R dt
    dR = (r_F F + r_U U + r_R R) dt
    I = O + R

    The structural model and parameter fit are deterministic. Constant additive
    process uncertainty is handled separately through the covariance kernel.

    F, U in [-1, 1], with F = U = 0 as the neutral point.
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

    def dR(self, P_R, r):
        P_R = wm_4_validation.vector(P_R, 3, "P_R")
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
        r = wm_4_validation.r_coefficients(r)
        kappa_R = -r[2]
        return r[:2] / kappa_R, kappa_R

    @classmethod
    def expected_equilibrium(cls, U, r):
        U = wm_4_validation.signal_value(U, "U")
        r = wm_4_validation.r_coefficients(r)
        return -(r[1] * U) / r[0]

    @staticmethod
    def drift(F, R, U, r):
        F = wm_4_validation._finite_scalar(F, "F")
        R = wm_4_validation._finite_scalar(R, "R")
        U = wm_4_validation.signal_value(U, "U")
        r = wm_4_validation.r_coefficients(r)
        return np.array([R, r @ np.array([F, U, R], dtype=float)], dtype=float)

    def process_variance_kernel(self, state, sample_indices):
        """Exact Euler kernel mapping fine-grid g^2 to sampled Var(I)=Var(R), with P0=0."""
        if "r" not in state: raise ValueError("state is missing required entry 'r'.")
        r = wm_4_validation.r_coefficients(state["r"])
        indices = np.asarray(sample_indices, dtype=int).reshape(-1)
        if indices.size < 1 or np.any(indices < 0) or np.any(indices >= self.t_steps): raise ValueError("sample_indices must lie in [0, t_steps).")

        transition = np.array([
            [1.0, self.dt],
            [self.dt * r[0], 1.0 + self.dt * r[2]],
        ], dtype=float)
        max_index = int(np.max(indices))
        impulse = np.empty(max_index, dtype=float)
        response = np.array([0.0, 1.0], dtype=float)
        for lag in range(max_index):
            impulse[lag] = response[1]
            response = transition @ response

        kernel = np.zeros((indices.size, self.t_steps), dtype=float)
        for row, index in enumerate(indices):
            if index > 0: kernel[row, :index] = self.dt * impulse[:index][::-1] ** 2
        return kernel

    def sensitivity(self, parameters):
        """Euler sensitivity of [F,R] and I=O+R to the initial state [F0,R0]."""
        r = parameters["r"] if isinstance(parameters, dict) else parameters
        r = wm_4_validation.r_coefficients(r)
        transition = np.array([
            [1.0, self.dt],
            [self.dt * r[0], 1.0 + self.dt * r[2]],
        ], dtype=float)
        state = np.empty((self.t_steps, 2, 2), dtype=float)
        state[0] = np.eye(2, dtype=float)
        for k in range(self.t_steps - 1): state[k + 1] = transition @ state[k]
        return {"state": state, "input": state[:, 1, :].copy(), "transition": transition}

    def _sample_signals(self, parameters):
        U, PE = np.empty(self.t_steps), np.empty(self.t_steps)
        Time, O = np.empty(self.t_steps), np.empty(self.t_steps)
        for k, tk in enumerate(self.t):
            U[k] = wm_4_validation.signal_value(parameters["U"](tk), "U(t)")
            PE[k] = wm_4_validation.signal_value(parameters["PE"](tk), "PE(t)")
            Time[k] = wm_4_validation.finite_feature_input(parameters["Time"](tk), "Time(t)")
            O[k] = wm_4_validation.feature_value(self.O(Time[k]), "O(Time(t))")
        return U, PE, Time, O

    def _solve_deterministic(self, init_cond, U, PE, Time, O, r, gamma=None):
        F, R, R_dot = np.empty(self.t_steps), np.empty(self.t_steps), np.empty(self.t_steps)
        I, Q = np.empty(self.t_steps), np.empty(self.t_steps)
        A = np.full(self.t_steps, np.nan)
        P_R = np.empty((self.t_steps, 3))
        P_A = np.full((self.t_steps, 4), np.nan)
        r = wm_4_validation.r_coefficients(r)
        q, kappa_R = self.equivalent_q(r)
        F[0], R[0] = init_cond

        for k in range(self.t_steps):
            P_R[k] = np.array([F[k], U[k], R[k]], dtype=float)
            R_dot[k] = r @ P_R[k]
            Q[k] = q @ np.array([F[k], U[k]], dtype=float)
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
        return self._solve_deterministic(init_cond, U, PE, Time, O, parameters["r"], gamma=parameters["gamma"])

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

    def fit_params_brute_force(self, input_data, initial_conditions, **kwargs): return brute_force_params(self).fit(input_data, initial_conditions, **kwargs)

    def fit_params(self, input_data, initial_conditions, **kwargs): return self.fit_params_qp(input_data, initial_conditions, **kwargs)
