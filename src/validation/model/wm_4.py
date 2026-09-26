import numpy as np

from src.validation.model.wm_1 import wm_1_validation


class wm_4_validation(wm_1_validation):
    """Validation rules for Water Model 4."""

    @classmethod
    def constructor(cls, T, t_steps, O):
        T, t_steps = super().constructor(T, t_steps)
        if not callable(O): raise TypeError("O must be callable.")
        return T, t_steps, O

    @classmethod
    def q_coefficients(cls, q, tol=1e-10):
        q = cls.vector(q, 3, "q")
        if q[1] >= -tol: raise ValueError("wm_4 requires q_F < 0 so the flexibility equilibrium is stable.")
        if q[2] > tol: raise ValueError("wm_4 requires q_U <= 0 so higher prices cannot increase the desired flexibility state.")
        for U in (0.0, 1.0):
            base = q[0] + q[2] * U
            if base < -tol or base > -q[1] + tol: raise ValueError("q must satisfy 0 <= q_0 + q_U U <= -q_F on U in [0, 1].")
        return q

    @classmethod
    def coefficients(cls, q, gamma=None):
        q = cls.q_coefficients(q)
        gamma = None if gamma is None else cls.simplex(gamma, 4, "gamma")
        return q, gamma

    @classmethod
    def parameters(cls, parameters):
        if not isinstance(parameters, dict): raise TypeError("parameters must be a dictionary.")
        required = ("U", "Time", "q", "kappa_R")
        missing = [name for name in required if name not in parameters]
        if missing: raise ValueError("parameters is missing required entries: " + ", ".join(missing) + ".")

        validated = dict(parameters)
        for name in ("U", "Time"):
            if not callable(validated[name]): raise TypeError(f"parameters['{name}'] must be callable.")
        if "PE" in validated and not callable(validated["PE"]): raise TypeError("parameters['PE'] must be callable.")
        validated.setdefault("PE", lambda _t: 0.0)

        q, gamma = cls.coefficients(validated["q"], validated.get("gamma"))
        validated["q"], validated["gamma"] = q, gamma
        validated["kappa_R"] = cls._finite_scalar(validated["kappa_R"], "kappa_R")
        if validated["kappa_R"] <= 0.0: raise ValueError("kappa_R must be positive.")
        return validated

    @classmethod
    def diffusion_parameters(cls, parameters):
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
        values = np.asarray(init_cond, dtype=float).reshape(-1)
        if values.size != 2: raise ValueError("wm_4 init_cond must contain exactly [F0, R0].")
        if not np.all(np.isfinite(values)): raise ValueError("init_cond must contain finite values.")
        if not 0.0 <= values[0] <= 1.0: raise ValueError("Initial flexibility state F0 must lie in [0, 1].")
        return values

    @classmethod
    def deterministic_inputs(cls, init_cond, parameters): return cls.initial_condition(init_cond), cls.parameters(parameters)

    @classmethod
    def fokker_planc_inputs(cls, init_cond, parameters): return cls.initial_condition(init_cond), cls.diffusion_parameters(parameters)

    @classmethod
    def finite_feature_input(cls, value, name): return cls._finite_scalar(value, name)

    @classmethod
    def feature_value(cls, value, name): return cls.signal_value(value, name)

    @classmethod
    def recovered_structural_coefficients(cls, q, tol=1e-8):
        tol = cls._finite_scalar(tol, "tol")
        if tol <= 0.0: raise ValueError("tol must be positive.")
        q = cls.vector(q, 3, "recovered q")
        if q[1] >= -tol: raise ValueError("recovered q_F must be negative.")
        if q[2] > tol: raise ValueError("recovered q_U must be nonpositive.")
        for U in (0.0, 1.0):
            base = q[0] + q[2] * U
            if base < -tol or base > -q[1] + tol: raise ValueError("recovered q violates the nominal flexibility-equilibrium bounds.")
        return q

    @classmethod
    def fit_data(cls, input_data, t_steps):
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
        PE = fit_vector(input_data.get("PE", np.zeros(t_steps, dtype=float)), "input_data['PE']")
        observation_mask = np.asarray(input_data.get("observation_mask", np.ones(t_steps, dtype=bool)), dtype=bool).reshape(-1)
        if observation_mask.size != t_steps: raise ValueError(f"input_data['observation_mask'] must contain exactly {t_steps} samples.")
        if np.count_nonzero(observation_mask) < 2: raise ValueError("input_data['observation_mask'] must select at least two observations.")
        for vector, name in ((U, "U"), (PE, "PE")):
            if np.any(vector < 0.0) or np.any(vector > 1.0): raise ValueError(f"Normalized {name} data must lie in [0, 1].")
        return {"I": I, "U": U, "PE": PE, "Time": Time, "observation_mask": observation_mask}
