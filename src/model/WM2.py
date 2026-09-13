import itertools
import numpy as np

from src.model.base.WM import WM
from src.method.FokkerPlanc import WM2FokkerPlanc
from src.solver.ActiveSet import ActiveSet
from src.validation.model.WM2 import WM2Validation


class WM2(WM):
    """Water Model 2: inertial WM dynamics with output-scaled diffusion.

    The fitting API uses reduced deterministic coefficients directly:

        I = c0 + c_BI*B_I + c_BO*B_O + c_V*V + c_U*U
        D = I - O = d0 + d_BI*B_I + d_BO*B_O + d_V*V + d_U*U

    fit_params alternates a deterministic WM2 fit and a statistical-baseline
    fit, then estimates sigma_0 and sigma_O from local residual dispersion.
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
        alpha, beta, gamma = (
            parameters["alpha"],
            parameters["beta"],
            parameters["gamma"],
        )
        P_O = np.array([B_O, V, 1.0 - U])
        O = self.O(P_O, beta)
        P_I = np.array([B_I, 1.0 - V, O, 1.0 - U])
        I = self.I(P_I, alpha)
        P_A = np.array([1.0 - U, I, V, 1.0 - PE])
        A = self.A(P_A, gamma)
        return U, B_O, B_I, PE, P_I, P_O, P_A, O, I, A

    def deterministic_solver(self, init_cond, parameters):
        init_cond, parameters = WM2Validation.deterministic_inputs(
            init_cond, parameters
        )
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
            (
                U[k], B_O[k], B_I[k], PE[k], P_I[k], P_O[k], P_A[k],
                O[k], I[k], A[k],
            ) = self._evaluate_step(tk, V[k], parameters)
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

    def _prepare_fit_data(self, input_data):
        if not isinstance(input_data, dict):
            raise TypeError("input_data must be a dictionary.")
        if "I" not in input_data or "U" not in input_data:
            raise ValueError("input_data must contain normalized I and U arrays.")

        def vec(value, name):
            x = np.asarray(value, dtype=float).reshape(-1)
            if x.size != self.t_steps:
                raise ValueError(
                    f"{name} must contain exactly {self.t_steps} samples."
                )
            if not np.all(np.isfinite(x)):
                raise ValueError(f"{name} must contain finite values.")
            return x

        I_data = vec(input_data["I"], "input_data['I']")
        U = vec(input_data["U"], "input_data['U']")
        PE = vec(
            input_data.get("PE", np.zeros(self.t_steps)),
            "input_data['PE']",
        )
        for x, name in ((I_data, "I"), (U, "U"), (PE, "PE")):
            if np.any(x < 0.0) or np.any(x > 1.0):
                raise ValueError(f"Normalized {name} data must lie in [0, 1].")
        return {"I": I_data, "U": U, "PE": PE}

    def _prepare_baseline_features(self, baseline_feature_vec):
        if isinstance(baseline_feature_vec, dict):
            if "B_I" not in baseline_feature_vec or "B_O" not in baseline_feature_vec:
                raise ValueError(
                    "baseline_feature_vec must contain B_I and B_O matrices."
                )
            X_I = np.asarray(baseline_feature_vec["B_I"], dtype=float)
            X_O = np.asarray(baseline_feature_vec["B_O"], dtype=float)
        else:
            X_I = np.asarray(baseline_feature_vec, dtype=float)
            X_O = X_I.copy()

        if X_I.ndim == 1:
            X_I = X_I[:, None]
        if X_O.ndim == 1:
            X_O = X_O[:, None]
        if X_I.ndim != 2 or X_O.ndim != 2:
            raise ValueError("Baseline features must be one- or two-dimensional.")
        if X_I.shape[0] != self.t_steps or X_O.shape[0] != self.t_steps:
            raise ValueError("Baseline features must have one row per time step.")
        if X_I.shape[1] == 0 or X_O.shape[1] == 0:
            raise ValueError("At least one baseline feature is required.")
        if not np.all(np.isfinite(X_I)) or not np.all(np.isfinite(X_O)):
            raise ValueError("Baseline features must be finite.")
        if (
            np.any(X_I < 0.0) or np.any(X_I > 1.0)
            or np.any(X_O < 0.0) or np.any(X_O > 1.0)
        ):
            raise ValueError(
                "Baseline features must be normalized to [0,1]. "
                "Do not include price, V, R, or functions derived from them."
            )
        return X_I, X_O

    @staticmethod
    def _least_squares_qp(
        X, y, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
        lb=None, ub=None, ridge=1e-10
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = X.shape[1]
        scale = max(1, X.shape[0])
        H = 2.0 * (X.T @ X) / scale + ridge * np.eye(n)
        f = -2.0 * (X.T @ y) / scale
        return ActiveSet.solve(
            H, f,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            lb=lb, ub=ub,
            tol=1e-9,
        )

    def _simulate_reduced(self, current_conditions):
        init_cond = np.asarray(
            current_conditions["init_cond"], dtype=float
        ).reshape(-1)
        if init_cond.size != 2:
            raise ValueError("Reduced fitting requires initial_conditions=[V0,R0].")
        c = np.asarray(current_conditions["I_coeff"], dtype=float).reshape(-1)
        d = np.asarray(current_conditions["d_coeff"], dtype=float).reshape(-1)
        if c.size != 5 or d.size != 5:
            raise ValueError("I_coeff and d_coeff must each have five entries.")

        B_I = np.asarray(current_conditions["B_I"], dtype=float)
        B_O = np.asarray(current_conditions["B_O"], dtype=float)
        U = np.asarray(current_conditions["U"], dtype=float)
        kappa = float(current_conditions["kappa_R"])

        V = np.empty(self.t_steps)
        R = np.empty(self.t_steps)
        R_dot = np.empty(self.t_steps)
        I = np.empty(self.t_steps)
        D = np.empty(self.t_steps)
        O = np.empty(self.t_steps)
        V[0], R[0] = init_cond

        for k in range(self.t_steps):
            x = np.array([1.0, B_I[k], B_O[k], V[k], U[k]])
            I[k] = c @ x
            D[k] = d @ x
            O[k] = I[k] - D[k]
            R_dot[k] = kappa * (D[k] - R[k])
            if k < self.t_steps - 1:
                V[k + 1] = V[k] + self.dt * R[k]
                R[k + 1] = R[k] + self.dt * R_dot[k]

        return {
            "V": V, "R": R, "R_dot": R_dot,
            "I": I, "D": D, "O": O,
        }

    @staticmethod
    def _equilibrium_constraints_scaled():
        rows = []
        rhs = []
        for B_I, B_O, U in itertools.product((0.0, 1.0), repeat=3):
            n = np.array([1.0, B_I, B_O, 0.0, U, 0.0])
            rows.append(-n)
            rhs.append(0.0)
            upper = n.copy()
            upper[3] += 1.0
            rows.append(upper)
            rhs.append(0.0)
        return np.asarray(rows), np.asarray(rhs)

    def _optimize_WM2_params(self, current_conditions, input_data):
        """Solve the deterministic WM2 fitting block using convex QPs."""
        data = self._prepare_fit_data(input_data)
        state = self._simulate_reduced(current_conditions)
        B_I = np.asarray(current_conditions["B_I"])
        B_O = np.asarray(current_conditions["B_O"])
        U = data["U"]

        X_I = np.column_stack(
            (np.ones(self.t_steps), B_I, B_O, state["V"], U)
        )
        A_ub = np.vstack((X_I, -X_I))
        b_ub = np.concatenate(
            (np.ones(self.t_steps), np.zeros(self.t_steps))
        )
        c = self._least_squares_qp(
            X_I, data["I"], A_ub=A_ub, b_ub=b_ub,
            ridge=self._FIT_RIDGE,
        )

        no_V = c[0] + c[1] * B_I + c[2] * B_O + c[4] * U
        if abs(c[3]) > 1e-5:
            V_proxy = (data["I"] - no_V) / c[3]
            V_proxy += current_conditions["init_cond"][0] - V_proxy[0]
            V_proxy = 0.9 * V_proxy + 0.1 * state["V"]
        else:
            V_proxy = state["V"].copy()

        R_proxy = np.gradient(V_proxy, self.dt)
        R_proxy[0] = current_conditions["init_cond"][1]
        R_dot_proxy = np.gradient(R_proxy, self.dt)

        X_D = np.column_stack(
            (
                np.ones(self.t_steps), B_I, B_O, V_proxy, U, -R_proxy
            )
        )
        A_D, b_D = self._equilibrium_constraints_scaled()

        kappa_max = max(10.0 * self._FIT_EPS, 1.0 / self.dt)
        lb = np.array(
            [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, self._FIT_EPS]
        )
        ub = np.array(
            [np.inf, np.inf, np.inf, -self._FIT_EPS, np.inf, kappa_max]
        )
        q = self._least_squares_qp(
            X_D, R_dot_proxy,
            A_ub=A_D, b_ub=b_D,
            lb=lb, ub=ub,
            ridge=self._FIT_RIDGE,
        )

        updated = dict(current_conditions)
        updated["I_coeff"] = c
        updated["kappa_R"] = float(q[5])
        updated["d_coeff"] = q[:5] / q[5]
        updated["V_proxy"] = V_proxy
        updated["R_proxy"] = R_proxy
        updated["deterministic_state"] = self._simulate_reduced(updated)
        return updated

    def _optimize_B_params(
        self, current_conditions, baseline_feature_vec, input_data
    ):
        """Solve normalized B_I/B_O feature weights as one convex QP."""
        data = self._prepare_fit_data(input_data)
        X_BI, X_BO = self._prepare_baseline_features(baseline_feature_vec)
        state = self._simulate_reduced(current_conditions)
        c = np.asarray(current_conditions["I_coeff"])
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

        w = self._least_squares_qp(
            X, y,
            A_eq=A_eq, b_eq=b_eq,
            lb=np.zeros(n_I + n_O),
            ub=np.ones(n_I + n_O),
            ridge=self._FIT_RIDGE,
        )

        updated = dict(current_conditions)
        updated["B_I_weights"] = w[:n_I]
        updated["B_O_weights"] = w[n_I:]
        updated["B_I"] = X_BI @ w[:n_I]
        updated["B_O"] = X_BO @ w[n_I:]
        updated["deterministic_state"] = self._simulate_reduced(updated)
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
            x = residual[lo:hi]
            result[k] = np.std(x, ddof=1) if x.size > 1 else 0.0
        return result

    def _optimize_uncertainty_params(self, current_conditions, input_data):
        """Fit sigma_0 and sigma_O after deterministic convergence."""
        data = self._prepare_fit_data(input_data)
        state = self._simulate_reduced(current_conditions)
        residual = data["I"] - state["I"]
        local_std = self._local_std(residual)

        # The diffusion law is defined for non-negative aggregate output.
        O_fit = np.maximum(state["O"], 0.0)
        X = np.column_stack((np.ones(self.t_steps), O_fit))
        sigma = self._least_squares_qp(
            X, local_std,
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
        """Run the complete blockwise WM2 parameter fit.

        The initial B_I and B_O are constant and equal to mean observed
        purchased input.  Deterministic WM2 and baseline feature parameters
        are alternated until improvement is negligible or a candidate degrades
        the fit.  The uncertainty parameters are fitted only afterwards.
        """
        data = self._prepare_fit_data(input_data)
        init_cond = WM2Validation.initial_condition(initial_conditions)
        X_BI, X_BO = self._prepare_baseline_features(baseline_feature_vec)

        mean_I = float(np.mean(data["I"]))
        current = {
            "init_cond": init_cond.copy(),
            "U": data["U"].copy(),
            "PE": data["PE"].copy(),
            "B_I": np.full(self.t_steps, mean_I),
            "B_O": np.full(self.t_steps, mean_I),
            "I_coeff": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
            "d_coeff": np.array([0.25, 0.0, 0.0, -0.5, 0.0]),
            "kappa_R": min(1.0, 0.5 / self.dt),
            "sigma_0": self._FIT_EPS,
            "sigma_O": self._FIT_EPS,
            "B_I_weights": None,
            "B_O_weights": None,
        }

        state = self._simulate_reduced(current)
        previous_loss = float(np.mean((data["I"] - state["I"]) ** 2))
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

            state = self._simulate_reduced(candidate)
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
