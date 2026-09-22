import numpy as np

from src.validation.model.WM1 import WM1Validation


class WM3Validation(WM1Validation):
    """Validation rules for Water Model 3."""

    @classmethod
    def constructor(cls, T, t_steps, L, H):
        # Verify constructor inputs
        T, t_steps = super().constructor(T, t_steps)
        if not callable(L):
            raise TypeError("L must be callable.")
        if not callable(H):
            raise TypeError("H must be callable.")
        return T, t_steps, L, H

    @classmethod
    def coefficients(cls, alpha, beta, gamma):
        # Verify WM3 coefficient simplices
        return (
            cls.simplex(alpha, 3, "alpha"),
            cls.simplex(beta, 4, "beta"),
            cls.simplex(gamma, 4, "gamma"),
        )

    @classmethod
    def parameters(cls, parameters):
        # Verify model parameters
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dictionary.")

        required = (
            "U",
            "Time",
            "Chill_factor",
            "PE",
            "alpha",
            "beta",
            "gamma",
            "kappa_R",
        )
        missing = [name for name in required if name not in parameters]
        if missing:
            raise ValueError(
                "parameters is missing required entries: "
                + ", ".join(missing)
                + "."
            )

        validated = dict(parameters)
        for name in ("U", "Time", "Chill_factor", "PE"):
            if not callable(validated[name]):
                raise TypeError(f"parameters['{name}'] must be callable.")

        alpha, beta, gamma = cls.coefficients(
            validated["alpha"], validated["beta"], validated["gamma"]
        )
        validated["alpha"] = alpha
        validated["beta"] = beta
        validated["gamma"] = gamma

        validated["kappa_R"] = cls._finite_scalar(
            validated["kappa_R"], "kappa_R"
        )
        if validated["kappa_R"] <= 0.0:
            raise ValueError("kappa_R must be positive.")

        # Stability of the expected inertial dynamics depends only on d_V.
        a_V, _, a_O = alpha
        b_V = beta[0]
        dV = -a_V + (a_O - 1.0) * b_V
        cls.stable_drift_coefficient(dV)

        return validated

    @classmethod
    def initial_condition(cls, init_cond, allow_rate=True):
        # Verify inputs
        values = np.asarray(init_cond, dtype=float).reshape(-1)
        if values.size != 2:
            raise ValueError("WM3 init_cond must contain exactly [V0, R0].")
        if not np.all(np.isfinite(values)):
            raise ValueError("init_cond must contain finite values.")
        if not 0.0 <= values[0] <= 1.0:
            raise ValueError("Initial expected volume V0 must lie in [0, 1].")
        return values

    @classmethod
    def deterministic_inputs(cls, init_cond, parameters):
        # Verify deterministic solver inputs
        return cls.initial_condition(init_cond), cls.parameters(parameters)

    @classmethod
    def finite_feature_input(cls, value, name):
        # Raw Time and Chill_factor values only need to be finite.
        return cls._finite_scalar(value, name)

    @classmethod
    def feature_value(cls, value, name):
        # L(Time) and H(Chill_factor) are normalized WM3 regressors.
        return cls.signal_value(value, name)

    @classmethod
    def reduced_drift_inputs(cls, L_value, H_value, alpha, beta):
        # Verify inputs for affine drift reduction
        L_value = cls.feature_value(L_value, "L_value")
        H_value = cls.feature_value(H_value, "H_value")
        alpha = cls.simplex(alpha, 3, "alpha")
        beta = cls.simplex(beta, 4, "beta")
        return L_value, H_value, alpha, beta

    @classmethod
    def stable_drift_coefficient(cls, dV, tol=1e-10):
        # Verify asymptotic stability of the expected inertial dynamics
        dV = cls._finite_scalar(dV, "d_V")
        if dV >= -tol:
            raise ValueError(
                "WM3 requires d_V < 0 so the expected equilibrium is stable."
            )
        return dV
