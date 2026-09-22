import numpy as np

from src.model.base.WM import WM
from src.method.FokkerPlanc import WM2FokkerPlanc
from src.params.WM2 import BruteParams, ProxyParams
from src.solver.ActiveSet import ActiveSet
from src.validation.model.WM2 import WM2Validation

class WM2(WM):
    """Water Model 2 with reduced internal deterministic dynamics.

    Public model parameters retain alpha, beta and gamma.  Alpha and beta are
    reduced once to affine input and drift coefficients, and all deterministic
    state propagation uses those reduced coefficients.

    The reduced maps are

        I = c0 + c_BI B_I + c_BO B_O + c_V V + c_U U

        D = I - O
          = d0 + d_BI B_I + d_BO B_O + d_V V + d_U U.

    Parameter fitting works directly in these reduced coordinates. After the
    reduced deterministic fit converges, beta and alpha are recovered directly
    from the fitted reduced coefficients and then validated against the exact
    reduced maps. Only then are sigma_0 and sigma_O fitted, because the
    diffusion scale depends on the beta-defined output O.
    """

    _FIT_EPS = 1e-8
    _FIT_RIDGE = 1e-10

    def __init__(self, T, t_steps):
        # Initialize variables
        T, t_steps = WM2Validation.constructor(T, t_steps)
        self.T = T
        self.t_steps = t_steps
        self.t = np.linspace(T[0], T[1], t_steps)
        self.dt = self.t[1] - self.t[0]
        self.fp = WM2FokkerPlanc()

    def I(self, P_I, alpha):
        # Evaluate input map
        P_I, alpha = WM2Validation.linear_map(P_I, alpha, 4, "P_I", "alpha")
        return alpha @ P_I

    def O(self, P_O, beta):
        # Evaluate output map
        P_O, beta = WM2Validation.linear_map(P_O, beta, 3, "P_O", "beta")
        return beta @ P_O

    def A(self, P_A, gamma):
        # Evaluate availability map
        P_A, gamma = WM2Validation.linear_map(P_A, gamma, 4, "P_A", "gamma")
        return gamma @ P_A

    @staticmethod
    def reduced_coefficients(alpha, beta):
        """Reduce alpha/beta to the affine purchased-input and drift maps."""
        # Compute reduced coefficients
        alpha, beta = WM2Validation.reduction_inputs(alpha, beta)

        a_B, a_V, a_O, a_U = alpha
        b_B, b_V, b_U = beta

        input_coeff = np.array([a_V + a_U + a_O * b_U, a_B, a_O * b_B, -a_V + a_O * b_V, -a_U - a_O * b_U], dtype=float)

        drift_coeff = np.array([a_V + a_U + (a_O - 1.0) * b_U, a_B, (a_O - 1.0) * b_B, -a_V + (a_O - 1.0) * b_V, (1.0 - a_O) * b_U - a_U], dtype=float)
        return input_coeff, drift_coeff

    @classmethod
    def reduced_drift_coefficients(cls, B_O, B_I, alpha, beta):
        # Compute reduced drift
        B_O = WM2Validation.signal_value(B_O, "B_O")
        B_I = WM2Validation.signal_value(B_I, "B_I")
        _, d = cls.reduced_coefficients(alpha, beta)
        d0 = d[0] + d[1] * B_I + d[2] * B_O
        return d0, d[3], d[4]

    @classmethod
    def expected_equilibrium(cls, B_O, B_I, U, alpha, beta):
        # Compute expected equilibrium
        B_O = WM2Validation.signal_value(B_O, "B_O")
        B_I = WM2Validation.signal_value(B_I, "B_I")
        U = WM2Validation.signal_value(U, "U")
        _, d = cls.reduced_coefficients(alpha, beta)
        WM2Validation.stable_drift_coefficient(d[3])
        numerator = d[0] + d[1] * B_I + d[2] * B_O + d[4] * U
        return -numerator / d[3]

    def _sample_signals(self, parameters):
        # Sample input signals
        U = np.empty(self.t_steps)
        B_O = np.empty(self.t_steps)
        B_I = np.empty(self.t_steps)
        PE = np.empty(self.t_steps)

        for k, tk in enumerate(self.t):
            U[k] = WM2Validation.signal_value(parameters["U"](tk), "U(t)")
            B_O[k] = WM2Validation.signal_value(parameters["B_O"](tk), "B_O(t)")
            B_I[k] = WM2Validation.signal_value(parameters["B_I"](tk), "B_I(t)")
            PE[k] = WM2Validation.signal_value(parameters["PE"](tk), "PE(t)")

        return U, B_O, B_I, PE

    def _solve_deterministic(self, init_cond, U, B_O, B_I, PE, input_coeff, drift_coeff, kappa_R, gamma=None):
        """Canonical deterministic WM2 propagation in reduced coordinates."""
        # Solve deterministic dynamics
        V = np.empty(self.t_steps)
        R = np.empty(self.t_steps)
        R_dot = np.empty(self.t_steps)
        O = np.empty(self.t_steps)
        I = np.empty(self.t_steps)
        D = np.empty(self.t_steps)
        A = np.full(self.t_steps, np.nan)
        P_I = np.empty((self.t_steps, 4))
        P_O = np.empty((self.t_steps, 3))
        P_A = np.full((self.t_steps, 4), np.nan)

        V[0], R[0] = init_cond

        for k in range(self.t_steps):
            reduced_point = np.array([1.0, B_I[k], B_O[k], V[k], U[k]], dtype=float)
            I[k] = input_coeff @ reduced_point
            D[k] = drift_coeff @ reduced_point
            O[k] = I[k] - D[k]

            P_O[k] = np.array([B_O[k], V[k], 1.0 - U[k]])
            P_I[k] = np.array([B_I[k], 1.0 - V[k], O[k], 1.0 - U[k]])

            if gamma is not None:
                P_A[k] = np.array([1.0 - U[k], I[k], V[k], 1.0 - PE[k]])
                A[k] = gamma @ P_A[k]

            R_dot[k] = kappa_R * (D[k] - R[k])
            if k < self.t_steps - 1:
                V[k + 1] = V[k] + self.dt * R[k]
                R[k + 1] = R[k] + self.dt * R_dot[k]

        return {"t": self.t.copy(), "V": V, "R": R, "R_dot": R_dot, "U": np.asarray(U, dtype=float).copy(), "B_O": np.asarray(B_O, dtype=float).copy(), "B_I": np.asarray(B_I, dtype=float).copy(), "PE": np.asarray(PE, dtype=float).copy(), "P_I": P_I, "P_O": P_O, "P_A": P_A, "O": O, "I": I, "D": D, "A": A, "input_coeff": np.asarray(input_coeff, dtype=float).copy(), "drift_coeff": np.asarray(drift_coeff, dtype=float).copy(),}

    def deterministic_solver(self, init_cond, parameters):
        # Solve deterministic model
        init_cond, parameters = WM2Validation.deterministic_inputs(init_cond, parameters)
        U, B_O, B_I, PE = self._sample_signals(parameters)
        input_coeff, drift_coeff = self.reduced_coefficients(parameters["alpha"], parameters["beta"])
        return self._solve_deterministic(init_cond, U, B_O, B_I, PE, input_coeff, drift_coeff, parameters["kappa_R"], gamma=parameters["gamma"])

    @staticmethod
    def _least_squares_qp(X, y, A_ub=None, b_ub=None, A_eq=None, b_eq=None, lb=None, ub=None, ridge=1e-10, max_iter=10000):
        # Build and solve QP
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = X.shape[1]
        scale = max(1, X.shape[0])
        H = 2.0 * (X.T @ X) / scale + ridge * np.eye(n)
        f = -2.0 * (X.T @ y) / scale
        return ActiveSet.solve(H, f, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub, tol=1e-9, max_iter=max_iter)

    def fit_params_proxy(self, input_data, initial_conditions, baseline_feature_vec):
        """Fit WM2 using the legacy proxy-state reduced-parameter method."""
        return ProxyParams(self).fit(input_data, initial_conditions, baseline_feature_vec)

    def fit_params_brute(self, input_data, initial_conditions, baseline_feature_vec, **kwargs):
        """Fit WM2 using the grid-refined latent-state brute-force method.

        Keyword arguments are forwarded to :class:\`src.params.WM2.BruteParams\`,
        including grid_points, refinements, baseline_iterations, c_v_range,
        d_v_range and ratio_eps.
        """
        return BruteParams(self).fit(input_data, initial_conditions, baseline_feature_vec, **kwargs)

    def fit_params(self, input_data, initial_conditions, baseline_feature_vec):
        """Backward-compatible alias for the proxy parameter estimator."""
        return self.fit_params_proxy(input_data, initial_conditions, baseline_feature_vec)

    def fokker_planc_solver(self, init_cond, parameters, initial_density, v_bounds=(-0.5, 1.5), r_bounds=(-1.0, 1.0), v_steps=161, r_steps=161, cfl=0.8):
        # Solve Fokker-Planck model
        init_cond, validated = WM2Validation.deterministic_inputs(init_cond, parameters)
        _, drift_coeff = self.reduced_coefficients(validated["alpha"], validated["beta"])
        fp_parameters = dict(validated)
        fp_parameters["d_coeff"] = drift_coeff

        return self.fp.solve(self.t, init_cond, fp_parameters, initial_density=initial_density, v_bounds=v_bounds, r_bounds=r_bounds, v_steps=v_steps, r_steps=r_steps, cfl=cfl)
