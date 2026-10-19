import numpy as np

from src.validation.model.wm_1 import wm_1_validation


class wm_4_validation(wm_1_validation):
    """Validation rules for signed-domain Water Model 4."""

    @classmethod
    def constructor(cls, T, t_steps, O):
        T, t_steps = super().constructor(T, t_steps)
        if not callable(O): raise TypeError("O must be callable.")
        return T, t_steps, O

    @classmethod
    def signal_value(cls, value, name):
        scalar = cls._finite_scalar(value, name)
        if not -1.0 <= scalar <= 1.0: raise ValueError(f"{name} must lie in [-1, 1].")
        return scalar

    @classmethod
    def r_coefficients(cls, r, tol=1e-10):
        r = cls.vector(r, 3, "r")
        r_F, r_U, r_R = map(float, r)
        if r_F >= -tol: raise ValueError("wm_4 requires r_F < 0 so the flexibility dynamics are restoring.")
        if r_U > tol: raise ValueError("wm_4 requires r_U <= 0 so higher signed prices reduce flexibility flow.")
        if r_R >= -tol: raise ValueError("wm_4 requires r_R < 0 so the rate state is damped.")
        if r_U < r_F - tol: raise ValueError("wm_4 requires r_F <= r_U so F*(U) remains in [-1, 1] for U in [-1, 1].")
        if r_F + r_U < r_R - tol: raise ValueError("wm_4 requires r_F + r_U >= r_R so R*(F,U) remains in [-1, 1] for F,U in [-1, 1].")
        return r

    @classmethod
    def q_coefficients(cls, q, tol=1e-10):
        q = cls.vector(q, 2, "q")
        q_F, q_U = map(float, q)
        if q_F >= -tol: raise ValueError("wm_4 requires q_F < 0.")
        if q_U > tol: raise ValueError("wm_4 requires q_U <= 0.")
        if q_U < q_F - tol: raise ValueError("wm_4 requires q_F <= q_U so F*(U) remains in [-1, 1].")
        if q_F + q_U < -1.0 - tol: raise ValueError("wm_4 requires q_F + q_U >= -1 so R*(F,U)=q_F F+q_U U remains in [-1, 1].")
        return q

    @classmethod
    def legacy_r(cls, q, kappa_R):
        q = cls.q_coefficients(q)
        kappa_R = cls._finite_scalar(kappa_R, "kappa_R")
        if kappa_R <= 0.0: raise ValueError("kappa_R must be positive.")
        return cls.r_coefficients(np.array([kappa_R * q[0], kappa_R * q[1], -kappa_R], dtype=float))

    @classmethod
    def coefficients(cls, r, gamma=None):
        r = cls.r_coefficients(r)
        gamma = None if gamma is None else cls.simplex(gamma, 4, "gamma")
        return r, gamma

    @classmethod
    def parameters(cls, parameters):
        if not isinstance(parameters, dict): raise TypeError("parameters must be a dictionary.")
        required = ("U", "Time")
        missing = [name for name in required if name not in parameters]
        if missing: raise ValueError("parameters is missing required entries: " + ", ".join(missing) + ".")
        if "r" not in parameters and not all(name in parameters for name in ("q", "kappa_R")): raise ValueError("parameters must contain 'r' or the legacy pair 'q' and 'kappa_R'.")

        validated = dict(parameters)
        for name in ("U", "Time"):
            if not callable(validated[name]): raise TypeError(f"parameters['{name}'] must be callable.")
        if "PE" in validated and not callable(validated["PE"]): raise TypeError("parameters['PE'] must be callable.")
        validated.setdefault("PE", lambda _t: 0.0)

        r = cls.r_coefficients(validated["r"]) if "r" in validated else cls.legacy_r(validated["q"], validated["kappa_R"])
        r, gamma = cls.coefficients(r, validated.get("gamma"))
        validated["r"], validated["gamma"] = r, gamma
        kappa_R = -r[2]
        validated["kappa_R"] = float(kappa_R)
        validated["q"] = r[:2] / kappa_R
        r_0 = cls._finite_scalar(validated.get("r_0", 0.0), "r_0")
        if abs(r_0) - r[1] > -r[0] + 1e-8: raise ValueError("WM4 intercept violates the [-1,1] flexibility equilibrium restriction.")
        if abs(r_0) - r[0] - r[1] > -r[2] + 1e-8: raise ValueError("WM4 intercept violates the [-1,1] rate equilibrium restriction.")
        validated["r_0"] = r_0
        return validated

    @classmethod
    def initial_condition(cls, init_cond, allow_rate=True):
        values = np.asarray(init_cond, dtype=float).reshape(-1)
        if values.size != 2: raise ValueError("wm_4 init_cond must contain exactly [F0, R0].")
        if not np.all(np.isfinite(values)): raise ValueError("init_cond must contain finite values.")
        if not -1.0 <= values[0] <= 1.0: raise ValueError("Initial flexibility state F0 must lie in [-1, 1].")
        return values

    @classmethod
    def deterministic_inputs(cls, init_cond, parameters): return cls.initial_condition(init_cond), cls.parameters(parameters)

    @classmethod
    def finite_feature_input(cls, value, name): return cls._finite_scalar(value, name)

    @classmethod
    def feature_value(cls, value, name):
        scalar = cls._finite_scalar(value, name)
        if not 0.0 <= scalar <= 1.0: raise ValueError(f"{name} must lie in [0, 1].")
        return scalar

    @classmethod
    def recovered_structural_coefficients(cls, r, tol=1e-8):
        tol = cls._finite_scalar(tol, "tol")
        if tol <= 0.0: raise ValueError("tol must be positive.")
        return cls.r_coefficients(r, tol=tol)

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
        transition_mask = np.asarray(input_data.get("transition_mask", np.ones(t_steps - 1, dtype=bool)), dtype=bool).reshape(-1)
        if observation_mask.size != t_steps: raise ValueError(f"input_data['observation_mask'] must contain exactly {t_steps} samples.")
        if transition_mask.size != t_steps - 1: raise ValueError(f"input_data['transition_mask'] must contain exactly {t_steps - 1} samples.")
        if np.count_nonzero(observation_mask) < 2: raise ValueError("input_data['observation_mask'] must select at least two observations.")
        if np.count_nonzero(transition_mask) < 1: raise ValueError("input_data['transition_mask'] must select at least one state transition.")
        for vector, name in ((U, "U"), (PE, "PE")):
            if np.any(vector < -1.0) or np.any(vector > 1.0): raise ValueError(f"Signed normalized {name} data must lie in [-1, 1].")
        return {"I": I, "U": U, "PE": PE, "Time": Time, "observation_mask": observation_mask, "transition_mask": transition_mask}
