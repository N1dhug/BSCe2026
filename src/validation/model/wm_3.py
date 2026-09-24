import numpy as np

from src.validation.model.wm_1 import wm_1_validation


class wm_3_validation(wm_1_validation):
    """Validation rules for Water Model 3."""

    @classmethod
    def constructor(cls, T, t_steps, L, H=None):
        # Verify constructor inputs
        T, t_steps = super().constructor(T, t_steps)
        if not callable(L): raise TypeError("L must be callable.")
        if H is not None and not callable(H): raise TypeError("H must be callable.")
        return T, t_steps, L, H

    @classmethod
    def beta_coefficients(cls, beta):
        # Accept legacy [beta_V, beta_U, beta_L] or complete [beta_V, beta_U, beta_L, beta_H]
        values = np.asarray(beta, dtype=float).reshape(-1)
        if values.size == 3:
            values = cls.simplex(values, 3, "beta")
            return np.array([values[0], values[1], values[2], 0.0], dtype=float)
        return cls.simplex(values, 4, "beta")

    @classmethod
    def coefficients(cls, alpha, beta, gamma):
        # Verify wm_3 coefficient simplices
        return cls.simplex(alpha, 3, "alpha"), cls.beta_coefficients(beta), cls.simplex(gamma, 4, "gamma")

    @classmethod
    def reduction_inputs(cls, alpha, beta):
        # Verify reduced-map coefficients
        return cls.simplex(alpha, 3, "alpha"), cls.beta_coefficients(beta)

    @classmethod
    def parameters(cls, parameters):
        # Verify deterministic model parameters
        if not isinstance(parameters, dict): raise TypeError("parameters must be a dictionary.")

        required = ("U", "Time", "PE", "alpha", "beta", "gamma", "kappa_R")
        missing = [name for name in required if name not in parameters]
        if missing: raise ValueError("parameters is missing required entries: " + ", ".join(missing) + ".")

        validated = dict(parameters)
        for name in ("U", "Time", "PE"):
            if not callable(validated[name]): raise TypeError(f"parameters['{name}'] must be callable.")
        if "Temp" in validated and not callable(validated["Temp"]): raise TypeError("parameters['Temp'] must be callable.")
        validated.setdefault("Temp", lambda _t: 0.0)

        alpha, beta, gamma = cls.coefficients(validated["alpha"], validated["beta"], validated["gamma"])
        validated["alpha"], validated["beta"], validated["gamma"] = alpha, beta, gamma
        validated["kappa_R"] = cls._finite_scalar(validated["kappa_R"], "kappa_R")
        if validated["kappa_R"] <= 0.0: raise ValueError("kappa_R must be positive.")

        a_V, _, a_O = alpha
        cls.stable_drift_coefficient(-a_V + (a_O - 1.0) * beta[0])
        return validated

    @classmethod
    def diffusion_parameters(cls, parameters):
        # Verify stochastic diffusion parameters
        validated = cls.parameters(parameters)
        missing = [name for name in ("sigma_0", "sigma_O") if name not in parameters]
        if missing: raise ValueError("parameters is missing required stochastic entries: " + ", ".join(missing) + ".")
        validated["sigma_0"] = cls._finite_scalar(parameters["sigma_0"], "sigma_0")
        validated["sigma_O"] = cls._finite_scalar(parameters["sigma_O"], "sigma_O")
        if validated["sigma_0"] <= 0.0: raise ValueError("sigma_0 must be positive.")
        if validated["sigma_O"] < 0.0: raise ValueError("sigma_O must be nonnegative.")
        return validated

    @classmethod
    def initial_condition(cls, init_cond, allow_rate=True):
        # Verify inputs
        values = np.asarray(init_cond, dtype=float).reshape(-1)
        if values.size != 2: raise ValueError("wm_3 init_cond must contain exactly [V0, R0].")
        if not np.all(np.isfinite(values)): raise ValueError("init_cond must contain finite values.")
        if not 0.0 <= values[0] <= 1.0: raise ValueError("Initial expected volume V0 must lie in [0, 1].")
        return values

    @classmethod
    def deterministic_inputs(cls, init_cond, parameters):
        # Verify inputs
        return cls.initial_condition(init_cond), cls.parameters(parameters)

    @classmethod
    def fokker_planc_inputs(cls, init_cond, parameters):
        # Verify stochastic solver inputs
        return cls.initial_condition(init_cond), cls.diffusion_parameters(parameters)

    @classmethod
    def finite_feature_input(cls, value, name):
        # Verify raw feature input
        return cls._finite_scalar(value, name)

    @classmethod
    def feature_value(cls, value, name):
        # Verify normalized feature value
        return cls.signal_value(value, name)

    @classmethod
    def stable_drift_coefficient(cls, dV, tol=1e-10):
        # Verify stability
        dV = cls._finite_scalar(dV, "d_V")
        if dV >= -tol: raise ValueError("wm_3 requires d_V < 0 so the expected equilibrium is stable.")
        return dV

    @classmethod
    def recovered_structural_coefficients(cls, alpha, beta, tol=1e-8):
        # Verify directly recovered structural coefficients
        tol = cls._finite_scalar(tol, "tol")
        if tol <= 0.0: raise ValueError("tol must be positive.")

        alpha, beta = cls.vector(alpha, 3, "recovered alpha"), cls.beta_coefficients(beta)
        for values, name in ((alpha, "recovered alpha"), (beta, "recovered beta")):
            if np.any(values < -tol) or np.any(values > 1.0 + tol): raise ValueError(f"{name} coefficients must lie in [0, 1].")
            if not np.isclose(np.sum(values), 1.0, atol=tol, rtol=0.0): raise ValueError(f"{name} coefficients must sum to 1.")

        alpha, beta = np.clip(alpha, 0.0, 1.0), np.clip(beta, 0.0, 1.0)
        return alpha / np.sum(alpha), beta / np.sum(beta)

    @classmethod
    def fit_data(cls, input_data, t_steps):
        # Verify fitting data
        if not isinstance(input_data, dict): raise TypeError("input_data must be a dictionary.")

        required = ("I", "U", "Time")
        missing = [name for name in required if name not in input_data]
        if missing: raise ValueError("input_data is missing required entries: " + ", ".join(missing) + ".")

        t_steps = cls._integer(t_steps, "t_steps", 2)

        def fit_vector(value, name):
            vector = np.asarray(value, dtype=float).reshape(-1)
            if vector.size != t_steps: raise ValueError(f"{name} must contain exactly {t_steps} samples.")
            if not np.all(np.isfinite(vector)): raise ValueError(f"{name} must contain finite values.")
            return vector

        I = fit_vector(input_data["I"], "input_data['I']")
        U = fit_vector(input_data["U"], "input_data['U']")
        Time = fit_vector(input_data["Time"], "input_data['Time']")
        Temp = fit_vector(input_data.get("Temp", np.zeros(t_steps, dtype=float)), "input_data['Temp']")
        PE = fit_vector(input_data.get("PE", np.zeros(t_steps, dtype=float)), "input_data['PE']")

        for vector, name in ((I, "I"), (U, "U"), (PE, "PE")):
            if np.any(vector < 0.0) or np.any(vector > 1.0): raise ValueError(f"Normalized {name} data must lie in [0, 1].")

        return {"I": I, "U": U, "PE": PE, "Time": Time, "Temp": Temp}
