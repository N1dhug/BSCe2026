import numpy as np

from src.method.fokker_planc import wm_fokker_planc
from src.model.base.wm import wm
from src.params.wm_4 import brute_force_params, optimized_qp_params, qp_params
from src.solver.active_set import active_set
from src.validation.model.wm_4 import wm_4_validation


class wm_4(wm):
    """Water Model 4 with signed flexibility and price domains.

    P_R = [F, U, R], dR/dt = r @ P_R
    O = O(Time)

    dF = R dt
    dR = (r_F F + r_U U + r_R R) dt + g_H(t,I,U) dW
    I = O + R

    The structural fit is deterministic. Stochastic uncertainty is added only
    through explicit H0-H2 diffusion hypotheses after the deterministic fit.

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
        self.fp = wm_fokker_planc()

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

    @staticmethod
    def diffusion_hypothesis(hypothesis, time_hours, I, U, theta):
        """Evaluate exploratory WM4 diffusion hypotheses H0-H2.

        H0: g = sigma_0 + sigma_t * (1 + cos(2*pi*(t-phase)/24)) / 2
        H1: H0 + sigma_I * I
        H2: H1 + sigma_U * |U|

        sigma_0, sigma_t, sigma_I and sigma_U are nonnegative. The periodic
        term gives H0 a simple positive 24-hour uncertainty profile without
        using O(t). phase is measured in physical hours.
        """
        hypothesis = str(hypothesis).strip().upper()
        if hypothesis not in ("H0", "H1", "H2"): raise ValueError("hypothesis must be one of H0, H1 or H2.")
        theta = np.asarray(theta, dtype=float).reshape(-1)
        expected = {"H0": 3, "H1": 4, "H2": 5}[hypothesis]
        if theta.size != expected or not np.all(np.isfinite(theta)): raise ValueError(f"{hypothesis} theta must contain exactly {expected} finite values.")

        sigma_0, sigma_t, phase = map(float, theta[:3])
        if sigma_0 <= 0.0 or sigma_t < 0.0: raise ValueError("sigma_0 must be positive and sigma_t must be nonnegative.")
        if not 0.0 <= phase <= 24.0: raise ValueError("diffusion phase must lie in [0, 24] physical hours.")

        time_hours = np.asarray(time_hours, dtype=float)
        I, U = np.asarray(I, dtype=float), np.asarray(U, dtype=float)
        if not np.all(np.isfinite(time_hours)) or not np.all(np.isfinite(U)): raise ValueError("diffusion time and price inputs must be finite.")
        if np.any(U < -1.0) or np.any(U > 1.0): raise ValueError("wm_4 diffusion hypotheses require U in [-1, 1].")

        periodic = 0.5 * (1.0 + np.cos(2.0 * np.pi * (time_hours - phase) / 24.0))
        g = sigma_0 + sigma_t * periodic
        if hypothesis in ("H1", "H2"):
            if not np.all(np.isfinite(I)): raise ValueError("H1/H2 diffusion input I must be finite.")
            sigma_I = float(theta[3])
            if sigma_I < 0.0: raise ValueError("sigma_I must be nonnegative.")
            g = g + sigma_I * I
        if hypothesis == "H2":
            sigma_U = float(theta[4])
            if sigma_U < 0.0: raise ValueError("sigma_U must be nonnegative.")
            g = g + sigma_U * np.abs(U)
        shape = np.broadcast(time_hours, I, U).shape
        g = np.broadcast_to(np.asarray(g, dtype=float), shape).astype(float, copy=False)
        if not np.all(np.isfinite(g)) or np.any(g <= 0.0): raise ValueError("Diffusion hypothesis must remain finite and strictly positive over the trajectory.")
        return g

    def process_uncertainty(self, state, hypothesis, theta):
        """Propagate SDE process covariance only, with P0=0.

        This deliberately excludes initial-state/parameter sensitivity. For
        I=O+R, the returned input standard deviation is sqrt(P_RR).
        """
        required = ("Time", "I", "U", "r")
        missing = [name for name in required if name not in state]
        if missing: raise ValueError("state is missing required entries: " + ", ".join(missing) + ".")
        Time = np.asarray(state["Time"], dtype=float).reshape(-1)
        I = np.asarray(state["I"], dtype=float).reshape(-1)
        U = np.asarray(state["U"], dtype=float).reshape(-1)
        if Time.size != self.t_steps or I.size != self.t_steps or U.size != self.t_steps: raise ValueError("state signal lengths must equal wm_4.t_steps.")
        r = wm_4_validation.r_coefficients(state["r"])

        g = self.diffusion_hypothesis(hypothesis, Time, I, U, theta)
        transition = np.array([
            [1.0, self.dt],
            [self.dt * r[0], 1.0 + self.dt * r[2]],
        ], dtype=float)
        covariance = np.zeros((self.t_steps, 2, 2), dtype=float)
        for k in range(self.t_steps - 1):
            process = np.array([[0.0, 0.0], [0.0, g[k] * g[k] * self.dt]], dtype=float)
            covariance[k + 1] = transition @ covariance[k] @ transition.T + process
            covariance[k + 1] = 0.5 * (covariance[k + 1] + covariance[k + 1].T)
        input_variance = np.maximum(covariance[:, 1, 1], 0.0)
        return {
            "covariance": covariance,
            "input_variance": input_variance,
            "input_std": np.sqrt(input_variance),
            "diffusion": g,
            "transition": transition,
            "hypothesis": str(hypothesis).strip().upper(),
            "theta": np.asarray(theta, dtype=float).copy(),
        }

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

    def fokker_planc_solver(self, init_cond, parameters, initial_density, hypothesis, theta, f_bounds=(-1.0, 1.0), r_bounds=(-1.0, 1.0), f_steps=161, r_steps=161, cfl=0.8):
        _, parameters = wm_4_validation.deterministic_inputs(init_cond, parameters)
        r = parameters["r"]

        def fields(time):
            U = float(parameters["U"](time))
            physical_time = float(parameters["Time"](time))
            O = float(self.O(physical_time))
            return U, physical_time, O

        def drift_f(time, F, R): return R

        def drift_r(time, F, R):
            U, _, _ = fields(time)
            return r[0] * F + r[1] * U + r[2] * R

        def diffusion_r(time, F, R):
            U, physical_time, O = fields(time)
            I = O + R
            return self.diffusion_hypothesis(hypothesis, physical_time, I, U, theta)

        result = self.fp.solve(self.t, initial_density, drift_f, drift_r, diffusion_r, v_bounds=f_bounds, r_bounds=r_bounds, v_steps=f_steps, r_steps=r_steps, cfl=cfl)
        result.update({
            "F_grid": result["v_grid"], "marginal_F": result["marginal_v"],
            "mean_F": result["mean_v"], "var_F": result["var_v"],
            "diffusion_hypothesis": str(hypothesis).strip().upper(),
            "diffusion_theta": np.asarray(theta, dtype=float).copy(),
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

    def fit_params_brute_force(self, input_data, initial_conditions, **kwargs): return brute_force_params(self).fit(input_data, initial_conditions, **kwargs)

    def fit_params(self, input_data, initial_conditions, **kwargs): return self.fit_params_qp(input_data, initial_conditions, **kwargs)
