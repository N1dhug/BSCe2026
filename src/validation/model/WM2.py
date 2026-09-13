import itertools
import numpy as np

from src.validation.model.WM1 import WM1Validation


class WM2Validation(WM1Validation):
    """Validation and admissibility rules for Water Model 2."""

    @classmethod
    def parameters(cls, parameters):
        validated = super().parameters(parameters)
        validated.pop("theta", None)
        validated["kappa_R"] = cls._finite_scalar(
            parameters.get("kappa_R"), "kappa_R"
        )
        validated["sigma_0"] = cls._finite_scalar(
            parameters.get("sigma_0"), "sigma_0"
        )
        validated["sigma_O"] = cls._finite_scalar(
            parameters.get("sigma_O"), "sigma_O"
        )
        if validated["kappa_R"] <= 0.0:
            raise ValueError("kappa_R must be positive.")
        if validated["sigma_0"] <= 0.0:
            raise ValueError("sigma_0 must be positive.")
        if validated["sigma_O"] <= 0.0:
            raise ValueError("sigma_O must be positive.")
        cls.expected_equilibrium_parameters(
            validated["alpha"], validated["beta"]
        )
        return validated

    @classmethod
    def initial_condition(cls, init_cond, allow_rate=True):
        values = np.asarray(init_cond, dtype=float).reshape(-1)
        if values.size != 2:
            raise ValueError("WM2 init_cond must contain exactly [V0, R0].")
        if not np.all(np.isfinite(values)):
            raise ValueError("init_cond must contain finite values.")
        if not 0.0 <= values[0] <= 1.0:
            raise ValueError("Initial expected volume V0 must lie in [0, 1].")
        return values

    @classmethod
    def deterministic_inputs(cls, init_cond, parameters):
        return cls.initial_condition(init_cond), cls.parameters(parameters)

    @classmethod
    def reduction_inputs(cls, alpha, beta):
        return (
            cls.simplex(alpha, 4, "alpha"),
            cls.simplex(beta, 3, "beta"),
        )

    @classmethod
    def reduced_coefficients(cls, input_coeff, drift_coeff):
        return (
            cls.vector(input_coeff, 5, "input_coeff"),
            cls.vector(drift_coeff, 5, "drift_coeff"),
        )

    @classmethod
    def expected_equilibrium_parameters(cls, alpha, beta, tol=1e-10):
        alpha, beta = cls.reduction_inputs(alpha, beta)
        dV = -alpha[1] - (1.0 - alpha[2]) * beta[1]
        cls.stable_drift_coefficient(dV, tol=tol)

        equilibria = []
        for B_O, B_I, U in itertools.product((0.0, 1.0), repeat=3):
            o_bar = beta[0] * B_O + beta[2]
            i_bar = alpha[0] * B_I + alpha[1] + alpha[2] * o_bar + alpha[3]
            d0 = i_bar - o_bar
            dU = (1.0 - alpha[2]) * beta[2] - alpha[3]
            V_star = -(d0 + dU * U) / dV
            equilibria.append((B_O, B_I, U, V_star))
            if V_star < -tol or V_star > 1.0 + tol:
                raise ValueError(
                    "WM2 parameterization is inadmissible: expected equilibrium "
                    f"V*(B_O={B_O}, B_I={B_I}, U={U})={V_star:.6g} "
                    "lies outside [0, 1]."
                )
        return dV, equilibria

    @classmethod
    def stable_drift_coefficient(cls, dV, tol=1e-10):
        dV = cls._finite_scalar(dV, "d_V")
        if dV >= -tol:
            raise ValueError(
                "WM2 requires d_V < 0 so the expected equilibrium is stable."
            )
        return dV

    @classmethod
    def fit_data(cls, input_data, t_steps):
        if not isinstance(input_data, dict):
            raise TypeError("input_data must be a dictionary.")
        if "I" not in input_data or "U" not in input_data:
            raise ValueError("input_data must contain normalized I and U arrays.")

        t_steps = cls._integer(t_steps, "t_steps", 2)

        def fit_vector(value, name):
            vector = np.asarray(value, dtype=float).reshape(-1)
            if vector.size != t_steps:
                raise ValueError(
                    f"{name} must contain exactly {t_steps} samples."
                )
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must contain finite values.")
            return vector

        I_data = fit_vector(input_data["I"], "input_data['I']")
        U = fit_vector(input_data["U"], "input_data['U']")
        PE = fit_vector(
            input_data.get("PE", np.zeros(t_steps, dtype=float)),
            "input_data['PE']",
        )

        for vector, name in ((I_data, "I"), (U, "U"), (PE, "PE")):
            if np.any(vector < 0.0) or np.any(vector > 1.0):
                raise ValueError(
                    f"Normalized {name} data must lie in [0, 1]."
                )

        return {"I": I_data, "U": U, "PE": PE}

    @classmethod
    def baseline_features(cls, baseline_feature_vec, t_steps):
        t_steps = cls._integer(t_steps, "t_steps", 2)

        if isinstance(baseline_feature_vec, dict):
            if (
                "B_I" not in baseline_feature_vec
                or "B_O" not in baseline_feature_vec
            ):
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
            raise ValueError(
                "Baseline features must be one- or two-dimensional."
            )
        if X_I.shape[0] != t_steps or X_O.shape[0] != t_steps:
            raise ValueError(
                "Baseline features must have one row per time step."
            )
        if X_I.shape[1] == 0 or X_O.shape[1] == 0:
            raise ValueError("At least one baseline feature is required.")
        if not np.all(np.isfinite(X_I)) or not np.all(np.isfinite(X_O)):
            raise ValueError("Baseline features must be finite.")
        if (
            np.any(X_I < 0.0)
            or np.any(X_I > 1.0)
            or np.any(X_O < 0.0)
            or np.any(X_O > 1.0)
        ):
            raise ValueError(
                "Baseline features must be normalized to [0, 1]. "
                "Do not include price, V, R, or functions derived from them."
            )

        return X_I, X_O

    @classmethod
    def reduced_fit_conditions(cls, current_conditions, t_steps):
        if not isinstance(current_conditions, dict):
            raise TypeError("current_conditions must be a dictionary.")

        required = (
            "init_cond",
            "I_coeff",
            "d_coeff",
            "B_I",
            "B_O",
            "U",
            "PE",
            "kappa_R",
        )
        missing = [name for name in required if name not in current_conditions]
        if missing:
            raise ValueError(
                "current_conditions is missing required entries: "
                + ", ".join(missing)
                + "."
            )

        t_steps = cls._integer(t_steps, "t_steps", 2)
        init_cond = cls.initial_condition(current_conditions["init_cond"])
        I_coeff, d_coeff = cls.reduced_coefficients(
            current_conditions["I_coeff"], current_conditions["d_coeff"]
        )

        def trajectory(value, name, normalized=False):
            vector = np.asarray(value, dtype=float).reshape(-1)
            if vector.size != t_steps:
                raise ValueError(
                    f"{name} must contain exactly {t_steps} samples."
                )
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must contain finite values.")
            if normalized and (
                np.any(vector < 0.0) or np.any(vector > 1.0)
            ):
                raise ValueError(f"{name} must lie in [0, 1].")
            return vector

        B_I = trajectory(current_conditions["B_I"], "B_I", normalized=True)
        B_O = trajectory(current_conditions["B_O"], "B_O", normalized=True)
        U = trajectory(current_conditions["U"], "U", normalized=True)
        PE = trajectory(current_conditions["PE"], "PE", normalized=True)
        kappa_R = cls._finite_scalar(
            current_conditions["kappa_R"], "kappa_R"
        )
        if kappa_R <= 0.0:
            raise ValueError("kappa_R must be positive.")

        return {
            "init_cond": init_cond,
            "I_coeff": I_coeff,
            "d_coeff": d_coeff,
            "B_I": B_I,
            "B_O": B_O,
            "U": U,
            "PE": PE,
            "kappa_R": kappa_R,
        }

    @staticmethod
    def scaled_equilibrium_constraints():
        rows = []
        rhs = []
        for B_I, B_O, U in itertools.product((0.0, 1.0), repeat=3):
            numerator = np.array([1.0, B_I, B_O, 0.0, U, 0.0])
            rows.append(-numerator)
            rhs.append(0.0)

            upper = numerator.copy()
            upper[3] += 1.0
            rows.append(upper)
            rhs.append(0.0)

        return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)

    @classmethod
    def alpha_equilibrium_constraints(cls, beta, stability_eps=1e-8):
        beta = cls.simplex(beta, 3, "beta")
        stability_eps = cls._finite_scalar(stability_eps, "stability_eps")
        if stability_eps <= 0.0:
            raise ValueError("stability_eps must be positive.")
        rows = []
        rhs = []

        for B_I, B_O, U in itertools.product((0.0, 1.0), repeat=3):
            O0 = beta[0] * B_O + beta[2] * (1.0 - U)
            numerator = np.array(
                [B_I, 1.0, O0, 1.0 - U], dtype=float
            )
            dV = np.array([0.0, -1.0, beta[1], 0.0], dtype=float)

            rows.append(-numerator)
            rhs.append(-O0)

            rows.append(numerator + dV)
            rhs.append(O0 + beta[1])

        rows.append(np.array([0.0, -1.0, beta[1], 0.0]))
        rhs.append(beta[1] - stability_eps)

        return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)

    @classmethod
    def beta_equilibrium_constraints(cls, alpha, stability_eps=1e-8):
        alpha = cls.simplex(alpha, 4, "alpha")
        stability_eps = cls._finite_scalar(stability_eps, "stability_eps")
        if stability_eps <= 0.0:
            raise ValueError("stability_eps must be positive.")
        rows = []
        rhs = []
        coupling = alpha[2] - 1.0

        for B_I, B_O, U in itertools.product((0.0, 1.0), repeat=3):
            q = np.array([B_O, 0.0, 1.0 - U], dtype=float)
            constant = (
                alpha[0] * B_I
                + alpha[1]
                + alpha[3] * (1.0 - U)
            )

            rows.append(-coupling * q)
            rhs.append(constant)

            q_with_v = q.copy()
            q_with_v[1] += 1.0
            upper_constant = (
                alpha[0] * B_I
                + alpha[3] * (1.0 - U)
            )
            rows.append(coupling * q_with_v)
            rhs.append(-upper_constant)

        rows.append(np.array([0.0, coupling, 0.0]))
        rhs.append(alpha[1] - stability_eps)

        return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)

    @staticmethod
    def diffusion_scale(values):
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("WM2 diffusion scale must be finite.")
        if np.any(values < 0.0):
            raise ValueError(
                "WM2 produced negative g(O) on the selected numerical domain; "
                "choose an admissible domain/parameterization with O >= 0."
            )
        return values

    @classmethod
    def fp_inputs(cls, init_cond, parameters):
        init_cond = cls.initial_condition(init_cond)
        parameters = cls.parameters(parameters)
        if "d_coeff" not in parameters:
            raise ValueError(
                "WM2 Fokker-Planck parameters must include reduced d_coeff."
            )
        parameters["d_coeff"] = cls.vector(
            parameters["d_coeff"], 5, "d_coeff"
        )
        return init_cond, parameters
