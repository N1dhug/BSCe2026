import numpy as np

class wm_1_validation:
    """Validation rules for the updated Water Model 1."""

    @staticmethod
    def _finite_scalar(value, name):
        # Verify inputs
        array = np.asarray(value, dtype=float)
        if array.size != 1:
            raise ValueError(f"{name} must be a scalar.")
        scalar = float(array.reshape(-1)[0])
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be finite.")
        return scalar

    @staticmethod
    def _integer(value, name, minimum):
        # Verify inputs
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer.")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be an integer.") from exc
        if float(integer) != float(value):
            raise ValueError(f"{name} must be an integer.")
        if integer < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")
        return integer

    @classmethod
    def constructor(cls, T, t_steps):
        # Verify inputs
        interval = np.asarray(T, dtype=float).reshape(-1)
        if interval.size != 2:
            raise ValueError("T must contain exactly [t_start, t_end].")
        if not np.all(np.isfinite(interval)):
            raise ValueError("T must contain finite values.")
        if interval[1] <= interval[0]:
            raise ValueError("T must satisfy t_start < t_end.")
        return (float(interval[0]), float(interval[1])), cls._integer(t_steps, "t_steps", 2)

    @staticmethod
    def vector(values, length, name):
        # Verify inputs
        vector = np.asarray(values, dtype=float).reshape(-1)
        if vector.size != length:
            raise ValueError(f"{name} must contain {length} entries.")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain finite values.")
        return vector

    @classmethod
    def simplex(cls, values, length, name, tol=1e-10):
        # Verify inputs
        vector = cls.vector(values, length, name)
        if np.any(vector < -tol) or np.any(vector > 1.0 + tol):
            raise ValueError(f"{name} coefficients must lie in [0, 1].")
        if not np.isclose(np.sum(vector), 1.0, atol=tol, rtol=0.0):
            raise ValueError(f"{name} coefficients must sum to 1.")
        return vector

    @classmethod
    def coefficients(cls, alpha, beta, gamma):
        # Verify inputs
        return (cls.simplex(alpha, 4, "alpha"), cls.simplex(beta, 3, "beta"), cls.simplex(gamma, 4, "gamma"),)

    @classmethod
    def parameters(cls, parameters):
        # Verify inputs
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dictionary.")

        required = ("U", "B_O", "B_I", "PE", "alpha", "beta", "gamma")
        missing = [name for name in required if name not in parameters]
        if missing:
            raise ValueError(f"parameters is missing required entries: {', '.join(missing)}.")

        validated = dict(parameters)
        for name in ("U", "B_O", "B_I", "PE"):
            if not callable(validated[name]):
                raise TypeError(f"parameters['{name}'] must be callable.")

        alpha, beta, gamma = cls.coefficients(validated["alpha"], validated["beta"], validated["gamma"])
        validated["alpha"] = alpha
        validated["beta"] = beta
        validated["gamma"] = gamma
        validated["theta"] = cls._finite_scalar(validated.get("theta", 0.0), "theta")
        return validated

    @classmethod
    def initial_condition(cls, init_cond, allow_rate=True):
        # Verify inputs
        values = np.asarray(init_cond, dtype=float).reshape(-1)
        allowed_sizes = (1, 2) if allow_rate else (1,)
        if values.size not in allowed_sizes:
            if allow_rate:
                raise ValueError("init_cond must contain [V0] or [V0, R_previous].")
            raise ValueError("init_cond must contain exactly [V0].")
        if not np.all(np.isfinite(values)):
            raise ValueError("init_cond must contain finite values.")
        if not 0.0 <= values[0] <= 1.0:
            raise ValueError("Initial volume V0 must lie in [0, 1].")
        return values

    @classmethod
    def signal_value(cls, value, name):
        # Verify inputs
        scalar = cls._finite_scalar(value, name)
        if not 0.0 <= scalar <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")
        return scalar

    @classmethod
    def linear_map(cls, P, coefficients, length, vector_name, coefficient_name):
        # Verify inputs
        P = cls.vector(P, length, vector_name)
        coefficients = cls.simplex(coefficients, length, coefficient_name)
        return P, coefficients

    @classmethod
    def deterministic_inputs(cls, init_cond, parameters):
        # Verify inputs
        return cls.initial_condition(init_cond, allow_rate=True), cls.parameters(parameters)

    @classmethod
    def fokker_planc_inputs(cls, init_cond, parameters, sigma, e_bounds):
        # Verify inputs
        init_cond = cls.initial_condition(init_cond, allow_rate=False)
        parameters = cls.parameters(parameters)

        if not np.isclose(parameters["theta"], 0.0, atol=1e-12, rtol=0.0):
            raise ValueError("wm_1 Fokker-Planck approximation requires theta = 0.")

        sigma = cls._finite_scalar(sigma, "sigma")
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative.")

        bounds = np.asarray(e_bounds, dtype=float).reshape(-1)
        if bounds.size != 2 or not np.all(np.isfinite(bounds)):
            raise ValueError("e_bounds must contain two finite values.")
        e_min, e_max = float(bounds[0]), float(bounds[1])
        if e_min >= e_max:
            raise ValueError("e_bounds must satisfy e_min < e_max.")
        if not e_min <= 0.0 <= e_max:
            raise ValueError("e_bounds must contain the centered initial error e = 0.")

        return init_cond, parameters, sigma, (e_min, e_max)

    @classmethod
    def reduced_drift_inputs(cls, BtO, BtI, alpha, beta):
        # Verify inputs
        BtO = cls.signal_value(BtO, "BtO")
        BtI = cls.signal_value(BtI, "BtI")
        alpha = cls.simplex(alpha, 4, "alpha")
        beta = cls.simplex(beta, 3, "beta")
        return BtO, BtI, alpha, beta
