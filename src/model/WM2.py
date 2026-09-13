import numpy as np

from src.model.base.WM import WM
from src.method.FokkerPlanc import WM2FokkerPlanc
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

    Parameter fitting works directly in these reduced coordinates.  After the
    reduced deterministic fit converges, alpha and beta are selected to best
    reproduce the fitted reduced model and observed purchased input.  Only then
    are sigma_0 and sigma_O fitted, because the diffusion scale depends on the
    beta-defined output O.
    """

    _FIT_EPS = 1e-8
    _FIT_RIDGE = 1e-10

    def __init__(self, T, t_steps):
        T, t_steps = WM2Validation.constructor(T, t_steps)
        self.T = T
        self.t_steps = t_steps
        self.t = np.linspace(T[0], T[1], t_steps)
        self.dt = self.t[1] - self.t[0]
        self.fp = WM2FokkerPlanc()

    def I(self, P_I, alpha):
        P_I, alpha = WM2Validation.linear_map(
            P_I, alpha, 4, "P_I", "alpha"
        )
        return alpha @ P_I

    def O(self, P_O, beta):
        P_O, beta = WM2Validation.linear_map(
            P_O, beta, 3, "P_O", "beta"
        )
        return beta @ P_O

    def A(self, P_A, gamma):
        P_A, gamma = WM2Validation.linear_map(
            P_A, gamma, 4, "P_A", "gamma"
        )
        return gamma @ P_A

    @staticmethod
    def reduced_coefficients(alpha, beta):
        """Reduce alpha/beta to the affine purchased-input and drift maps."""
        alpha, beta = WM2Validation.reduction_inputs(alpha, beta)

        a_B, a_V, a_O, a_U = alpha
        b_B, b_V, b_U = beta

        input_coeff = np.array(
            [
                a_V + a_U + a_O * b_U,
                a_B,
                a_O * b_B,
                -a_V + a_O * b_V,
                -a_U - a_O * b_U,
            ],
            dtype=float,
        )

        drift_coeff = np.array(
            [
                a_V + a_U + (a_O - 1.0) * b_U,
                a_B,
                (a_O - 1.0) * b_B,
                -a_V + (a_O - 1.0) * b_V,
                (1.0 - a_O) * b_U - a_U,
            ],
            dtype=float,
        )
        return input_coeff, drift_coeff

    @classmethod
    def reduced_drift_coefficients(cls, B_O, B_I, alpha, beta):
        B_O = WM2Validation.signal_value(B_O, "B_O")
        B_I = WM2Validation.signal_value(B_I, "B_I")
        _, d = cls.reduced_coefficients(alpha, beta)
        d0 = d[0] + d[1] * B_I + d[2] * B_O
        return d0, d[3], d[4]

    @classmethod
    def expected_equilibrium(cls, B_O, B_I, U, alpha, beta):
        B_O = WM2Validation.signal_value(B_O, "B_O")
        B_I = WM2Validation.signal_value(B_I, "B_I")
        U = WM2Validation.signal_value(U, "U")
        _, d = cls.reduced_coefficients(alpha, beta)
        WM2Validation.stable_drift_coefficient(d[3])
        numerator = d[0] + d[1] * B_I + d[2] * B_O + d[4] * U
        return -numerator / d[3]

    def _sample_signals(self, parameters):
        U = np.empty(self.t_steps)
        B_O = np.empty(self.t_steps)
        B_I = np.empty(self.t_steps)
        PE = np.empty(self.t_steps)

        for k, tk in enumerate(self.t):
            U[k] = WM2Validation.signal_value(
                parameters["U"](tk), "U(t)"
            )
            B_O[k] = WM2Validation.signal_value(
                parameters["B_O"](tk), "B_O(t)"
            )
            B_I[k] = WM2Validation.signal_value(
                parameters["B_I"](tk), "B_I(t)"
            )
            PE[k] = WM2Validation.signal_value(
                parameters["PE"](tk), "PE(t)"
            )

        return U, B_O, B_I, PE

    def _solve_deterministic(
        self,
        init_cond,
        U,
        B_O,
        B_I,
        PE,
        input_coeff,
        drift_coeff,
        kappa_R,
        gamma=None,
    ):
        """Canonical deterministic WM2 propagation in reduced coordinates."""
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
            reduced_point = np.array(
                [1.0, B_I[k], B_O[k], V[k], U[k]], dtype=float
            )
            I[k] = input_coeff @ reduced_point
            D[k] = drift_coeff @ reduced_point
            O[k] = I[k] - D[k]

            P_O[k] = np.array([B_O[k], V[k], 1.0 - U[k]])
            P_I[k] = np.array([B_I[k], 1.0 - V[k], O[k], 1.0 - U[k]])

            if gamma is not None:
                P_A[k] = np.array(
                    [1.0 - U[k], I[k], V[k], 1.0 - PE[k]]
                )
                A[k] = gamma @ P_A[k]

            R_dot[k] = kappa_R * (D[k] - R[k])
            if k < self.t_steps - 1:
                V[k + 1] = V[k] + self.dt * R[k]
                R[k + 1] = R[k] + self.dt * R_dot[k]

        return {
            "t": self.t.copy(),
            "V": V,
            "R": R,
            "R_dot": R_dot,
            "U": np.asarray(U, dtype=float).copy(),
            "B_O": np.asarray(B_O, dtype=float).copy(),
            "B_I": np.asarray(B_I, dtype=float).copy(),
            "PE": np.asarray(PE, dtype=float).copy(),
            "P_I": P_I,
            "P_O": P_O,
            "P_A": P_A,
            "O": O,
            "I": I,
            "D": D,
            "A": A,
            "input_coeff": np.asarray(input_coeff, dtype=float).copy(),
            "drift_coeff": np.asarray(drift_coeff, dtype=float).copy(),
        }

    def deterministic_solver(self, init_cond, parameters):
        init_cond, parameters = WM2Validation.deterministic_inputs(
            init_cond, parameters
        )
        U, B_O, B_I, PE = self._sample_signals(parameters)
        input_coeff, drift_coeff = self.reduced_coefficients(
            parameters["alpha"], parameters["beta"]
        )
        return self._solve_deterministic(
            init_cond,
            U,
            B_O,
            B_I,
            PE,
            input_coeff,
            drift_coeff,
            parameters["kappa_R"],
            gamma=parameters["gamma"],
        )

    @staticmethod
    def _least_squares_qp(
        X,
        y,
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        lb=None,
        ub=None,
        ridge=1e-10,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = X.shape[1]
        scale = max(1, X.shape[0])
        H = 2.0 * (X.T @ X) / scale + ridge * np.eye(n)
        f = -2.0 * (X.T @ y) / scale
        return ActiveSet.solve(
            H,
            f,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            lb=lb,
            ub=ub,
            tol=1e-9,
        )

    def _fit_state(self, current_conditions):
        current = WM2Validation.reduced_fit_conditions(
            current_conditions, self.t_steps
        )
        return self._solve_deterministic(
            current["init_cond"],
            current["U"],
            current["B_O"],
            current["B_I"],
            current["PE"],
            current["I_coeff"],
            current["d_coeff"],
            current["kappa_R"],
        )

    def _optimize_WM2_params(self, current_conditions, input_data):
        """Fit the deterministic reduced WM2 block with convex QPs."""
        data = WM2Validation.fit_data(input_data, self.t_steps)
        current = WM2Validation.reduced_fit_conditions(
            current_conditions, self.t_steps
        )
        state = self._solve_deterministic(
            current["init_cond"],
            current["U"],
            current["B_O"],
            current["B_I"],
            current["PE"],
            current["I_coeff"],
            current["d_coeff"],
            current["kappa_R"],
        )

        B_I = current["B_I"]
        B_O = current["B_O"]
        U = data["U"]

        X_I = np.column_stack(
            (np.ones(self.t_steps), B_I, B_O, state["V"], U)
        )
        A_ub = np.vstack((X_I, -X_I))
        b_ub = np.concatenate(
            (np.ones(self.t_steps), np.zeros(self.t_steps))
        )
        input_coeff = self._least_squares_qp(
            X_I,
            data["I"],
            A_ub=A_ub,
            b_ub=b_ub,
            ridge=self._FIT_RIDGE,
        )

        no_V = (
            input_coeff[0]
            + input_coeff[1] * B_I
            + input_coeff[2] * B_O
            + input_coeff[4] * U
        )
        if abs(input_coeff[3]) > 1e-5:
            V_proxy = (data["I"] - no_V) / input_coeff[3]
            V_proxy += current["init_cond"][0] - V_proxy[0]
            V_proxy = 0.9 * V_proxy + 0.1 * state["V"]
        else:
            V_proxy = state["V"].copy()

        R_proxy = np.gradient(V_proxy, self.dt)
        R_proxy[0] = current["init_cond"][1]
        R_dot_proxy = np.gradient(R_proxy, self.dt)

        X_D = np.column_stack(
            (
                np.ones(self.t_steps),
                B_I,
                B_O,
                V_proxy,
                U,
                -R_proxy,
            )
        )
        A_D, b_D = WM2Validation.scaled_equilibrium_constraints()

        lb = np.array(
            [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, self._FIT_EPS]
        )
        ub = np.array(
            [np.inf, np.inf, np.inf, -self._FIT_EPS, np.inf, np.inf]
        )
        scaled = self._least_squares_qp(
            X_D,
            R_dot_proxy,
            A_ub=A_D,
            b_ub=b_D,
            lb=lb,
            ub=ub,
            ridge=self._FIT_RIDGE,
        )

        updated = dict(current_conditions)
        updated["I_coeff"] = input_coeff
        updated["kappa_R"] = float(scaled[5])
        updated["d_coeff"] = scaled[:5] / scaled[5]
        updated["V_proxy"] = V_proxy
        updated["R_proxy"] = R_proxy
        updated["deterministic_state"] = self._fit_state(updated)
        return updated

    def _optimize_B_params(
        self, current_conditions, baseline_feature_vec, input_data
    ):
        """Fit normalized B_I/B_O feature weights by convex QP."""
        data = WM2Validation.fit_data(input_data, self.t_steps)
        X_BI, X_BO = WM2Validation.baseline_features(
            baseline_feature_vec, self.t_steps
        )
        current = WM2Validation.reduced_fit_conditions(
            current_conditions, self.t_steps
        )
        state = self._solve_deterministic(
            current["init_cond"],
            current["U"],
            current["B_O"],
            current["B_I"],
            current["PE"],
            current["I_coeff"],
            current["d_coeff"],
            current["kappa_R"],
        )

        c = current["I_coeff"]
        n_I = X_BI.shape[1]
        n_O = X_BO.shape[1]

        fixed = c[0] + c[3] * state["V"] + c[4] * data["U"]
        X = np.column_stack((c[1] * X_BI, c[2] * X_BO))
        y = data["I"] - fixed

        A_eq = np.zeros((3, n_I + n_O))
        b_eq = np.array([1.0, 1.0, 0.0])
        A_eq[0, :n_I] = 1.0
        A_eq[1, n_I:] = 1.0
        A_eq[2, :n_I] = np.sum(X_BI, axis=0)
        A_eq[2, n_I:] = -np.sum(X_BO, axis=0)

        weights = self._least_squares_qp(
            X,
            y,
            A_eq=A_eq,
            b_eq=b_eq,
            lb=np.zeros(n_I + n_O),
            ub=np.ones(n_I + n_O),
            ridge=self._FIT_RIDGE,
        )

        updated = dict(current_conditions)
        updated["B_I_weights"] = weights[:n_I]
        updated["B_O_weights"] = weights[n_I:]
        updated["B_I"] = X_BI @ weights[:n_I]
        updated["B_O"] = X_BO @ weights[n_I:]
        updated["deterministic_state"] = self._fit_state(updated)
        return updated

    def _select_alpha_beta(self, current_conditions, input_data):
        """Select simplex alpha/beta vectors after reduced-fit convergence."""
        data = WM2Validation.fit_data(input_data, self.t_steps)
        current = WM2Validation.reduced_fit_conditions(
            current_conditions, self.t_steps
        )
        reduced_state = self._solve_deterministic(
            current["init_cond"],
            current["U"],
            current["B_O"],
            current["B_I"],
            current["PE"],
            current["I_coeff"],
            current["d_coeff"],
            current["kappa_R"],
        )

        V = reduced_state["V"]
        U = current["U"]
        B_I = current["B_I"]
        B_O = current["B_O"]

        # The final reduced drift determines the latent output target through
        # O = I_data - D.  Beta is therefore identified against an output
        # implied jointly by the reduced model and the observed purchases.
        O_target = data["I"] - reduced_state["D"]
        X_beta = np.column_stack((B_O, V, 1.0 - U))
        simplex_beta = np.ones((1, 3))
        beta = self._least_squares_qp(
            X_beta,
            O_target,
            A_eq=simplex_beta,
            b_eq=np.ones(1),
            lb=np.zeros(3),
            ub=np.ones(3),
            ridge=self._FIT_RIDGE,
        )

        alpha = np.array([1.0, 0.0, 0.0, 0.0])
        previous = None

        for iteration in range(8):
            O_beta = X_beta @ beta
            X_alpha = np.column_stack(
                (B_I, 1.0 - V, O_beta, 1.0 - U)
            )
            A_alpha, b_alpha = WM2Validation.alpha_equilibrium_constraints(
                beta
            )

            try:
                alpha = self._least_squares_qp(
                    X_alpha,
                    data["I"],
                    A_ub=A_alpha,
                    b_ub=b_alpha,
                    A_eq=np.ones((1, 4)),
                    b_eq=np.ones(1),
                    lb=np.zeros(4),
                    ub=np.ones(4),
                    ridge=self._FIT_RIDGE,
                )
            except ValueError:
                if iteration != 0:
                    raise
                beta = np.array([0.0, 1.0, 0.0])
                continue

            A_beta, b_beta = WM2Validation.beta_equilibrium_constraints(
                alpha
            )
            beta_new = self._least_squares_qp(
                X_beta,
                O_target,
                A_ub=A_beta,
                b_ub=b_beta,
                A_eq=simplex_beta,
                b_eq=np.ones(1),
                lb=np.zeros(3),
                ub=np.ones(3),
                ridge=self._FIT_RIDGE,
            )

            pair = np.concatenate((alpha, beta_new))
            if previous is not None and np.max(np.abs(pair - previous)) <= 1e-8:
                beta = beta_new
                break
            previous = pair
            beta = beta_new

        O_beta = X_beta @ beta
        X_alpha = np.column_stack((B_I, 1.0 - V, O_beta, 1.0 - U))
        A_alpha, b_alpha = WM2Validation.alpha_equilibrium_constraints(beta)
        alpha = self._least_squares_qp(
            X_alpha,
            data["I"],
            A_ub=A_alpha,
            b_ub=b_alpha,
            A_eq=np.ones((1, 4)),
            b_eq=np.ones(1),
            lb=np.zeros(4),
            ub=np.ones(4),
            ridge=self._FIT_RIDGE,
        )

        selected_input, selected_drift = self.reduced_coefficients(
            alpha, beta
        )
        selected_state = self._solve_deterministic(
            current["init_cond"],
            current["U"],
            current["B_O"],
            current["B_I"],
            current["PE"],
            selected_input,
            selected_drift,
            current["kappa_R"],
        )

        updated = dict(current_conditions)
        updated["alpha"] = alpha
        updated["beta"] = beta
        updated["reduced_fit_input_coeff"] = current["I_coeff"].copy()
        updated["reduced_fit_d_coeff"] = current["d_coeff"].copy()
        updated["selected_input_coeff"] = selected_input
        updated["selected_d_coeff"] = selected_drift
        updated["reduced_fit_state"] = reduced_state
        updated["deterministic_state"] = selected_state
        updated["selected_model_loss"] = float(
            np.mean((data["I"] - selected_state["I"]) ** 2)
        )
        return updated

    @staticmethod
    def _local_std(residual):
        residual = np.asarray(residual, dtype=float).reshape(-1)
        n = residual.size
        window = max(5, n // 20)
        if window % 2 == 0:
            window += 1
        window = min(window, n if n % 2 else n - 1)
        half = window // 2

        result = np.empty(n)
        for k in range(n):
            lo = max(0, k - half)
            hi = min(n, k + half + 1)
            segment = residual[lo:hi]
            result[k] = (
                np.std(segment, ddof=1) if segment.size > 1 else 0.0
            )
        return result

    def _optimize_uncertainty_params(self, current_conditions, input_data):
        """Fit sigma_0 and sigma_O after alpha/beta selection."""
        data = WM2Validation.fit_data(input_data, self.t_steps)
        alpha, beta = WM2Validation.reduction_inputs(
            current_conditions["alpha"], current_conditions["beta"]
        )
        current = WM2Validation.reduced_fit_conditions(
            current_conditions, self.t_steps
        )
        input_coeff, drift_coeff = self.reduced_coefficients(alpha, beta)
        state = self._solve_deterministic(
            current["init_cond"],
            current["U"],
            current["B_O"],
            current["B_I"],
            current["PE"],
            input_coeff,
            drift_coeff,
            current["kappa_R"],
        )

        residual = data["I"] - state["I"]
        local_std = self._local_std(residual)
        X = np.column_stack((np.ones(self.t_steps), state["O"]))

        sigma = self._least_squares_qp(
            X,
            local_std,
            A_ub=-X,
            b_ub=np.zeros(self.t_steps),
            lb=np.array([self._FIT_EPS, self._FIT_EPS]),
            ub=np.array([np.inf, np.inf]),
            ridge=self._FIT_RIDGE,
        )

        updated = dict(current_conditions)
        updated["sigma_0"] = float(sigma[0])
        updated["sigma_O"] = float(sigma[1])
        updated["residual"] = residual
        updated["local_std"] = local_std
        updated["deterministic_state"] = state
        return updated

    def fit_params(self, input_data, initial_conditions, baseline_feature_vec):
        """Fit reduced WM2, baselines, alpha/beta, then uncertainty.

        Gamma is not identified from purchased-input data and is therefore not
        fitted here.  It remains a separate price-optimization parameter.
        """
        data = WM2Validation.fit_data(input_data, self.t_steps)
        init_cond = WM2Validation.initial_condition(initial_conditions)
        X_BI, X_BO = WM2Validation.baseline_features(
            baseline_feature_vec, self.t_steps
        )

        mean_I = float(np.mean(data["I"]))
        current = {
            "init_cond": init_cond.copy(),
            "U": data["U"].copy(),
            "PE": data["PE"].copy(),
            "B_I": np.full(self.t_steps, mean_I),
            "B_O": np.full(self.t_steps, mean_I),
            "I_coeff": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
            "d_coeff": np.array([0.25, 0.0, 0.0, -0.5, 0.0]),
            "kappa_R": 0.5 / self.dt,
            "sigma_0": self._FIT_EPS,
            "sigma_O": self._FIT_EPS,
            "B_I_weights": None,
            "B_O_weights": None,
        }

        state = self._fit_state(current)
        previous_loss = float(
            np.mean((data["I"] - state["I"]) ** 2)
        )
        history = [previous_loss]
        relative_tol = 1e-6

        for _ in range(50):
            candidate = self._optimize_WM2_params(current, data)
            candidate = self._optimize_B_params(
                candidate,
                {"B_I": X_BI, "B_O": X_BO},
                data,
            )
            candidate = self._optimize_WM2_params(candidate, data)

            state = self._fit_state(candidate)
            loss = float(np.mean((data["I"] - state["I"]) ** 2))
            history.append(loss)

            improvement = previous_loss - loss
            threshold = relative_tol * max(1.0, previous_loss)

            if improvement < -threshold:
                break

            current = candidate
            current["deterministic_state"] = state
            previous_loss = loss

            if improvement <= threshold:
                break

        current["reduced_fit_loss"] = previous_loss
        current = self._select_alpha_beta(current, data)
        current = self._optimize_uncertainty_params(current, data)
        current["loss_history"] = np.asarray(history)
        current["deterministic_loss"] = previous_loss
        current["converged_iterations"] = len(history) - 1
        return current

    def fokker_planc_solver(
        self,
        init_cond,
        parameters,
        initial_density,
        v_bounds=(-0.5, 1.5),
        r_bounds=(-1.0, 1.0),
        v_steps=161,
        r_steps=161,
        cfl=0.8,
    ):
        init_cond, validated = WM2Validation.deterministic_inputs(
            init_cond, parameters
        )
        _, drift_coeff = self.reduced_coefficients(
            validated["alpha"], validated["beta"]
        )
        fp_parameters = dict(validated)
        fp_parameters["d_coeff"] = drift_coeff

        return self.fp.solve(
            self.t,
            init_cond,
            fp_parameters,
            initial_density=initial_density,
            v_bounds=v_bounds,
            r_bounds=r_bounds,
            v_steps=v_steps,
            r_steps=r_steps,
            cfl=cfl,
        )
