import numpy as np


class BMValidation:
    """Validation rules shared by Battery Model implementations."""

    @staticmethod
    def finite_scalar(value, name):
        array = np.asarray(value, dtype=float)
        if array.size != 1:
            raise ValueError(f"{name} must be a scalar.")
        scalar = float(array.reshape(-1)[0])
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be finite.")
        return scalar

    @staticmethod
    def positive_scalar(value, name):
        scalar = BMValidation.finite_scalar(value, name)
        if scalar <= 0.0:
            raise ValueError(f"{name} must be strictly positive.")
        return scalar

    @staticmethod
    def unit_scalar(value, name):
        scalar = BMValidation.finite_scalar(value, name)
        if not 0.0 <= scalar <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")
        return scalar

    @staticmethod
    def callable_input(value, name):
        if not callable(value):
            raise TypeError(f"{name} must be callable.")
        return value

    @classmethod
    def constructor(cls, B, A, T, t_steps):
        B = cls.callable_input(B, "B")
        A = cls.callable_input(A, "A")

        time_interval = np.asarray(T, dtype=float).reshape(-1)
        if time_interval.size != 2:
            raise ValueError("T must contain exactly [t_start, t_end].")
        if not np.all(np.isfinite(time_interval)):
            raise ValueError("T must contain finite values.")
        if time_interval[1] <= time_interval[0]:
            raise ValueError("T must satisfy t_start < t_end.")

        if isinstance(t_steps, (bool, np.bool_)):
            raise TypeError("t_steps must be an integer.")
        try:
            t_steps_int = int(t_steps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("t_steps must be an integer.") from exc
        if float(t_steps_int) != float(t_steps):
            raise ValueError("t_steps must be an integer.")
        if t_steps_int < 2:
            raise ValueError("t_steps must be at least 2.")

        return B, A, (float(time_interval[0]), float(time_interval[1])), t_steps_int

    @classmethod
    def parameters(cls, parameters):
        try:
            values = list(parameters)
        except TypeError as exc:
            raise TypeError("parameters must be an iterable [U, f, g, C, k, Xi].") from exc

        if len(values) != 6:
            raise ValueError("parameters must contain [U, f, g, C, k, Xi].")

        U, f, g, C, k, Xi = values
        values[0] = cls.callable_input(U, "U")
        values[1] = cls.callable_input(f, "f")
        values[2] = cls.callable_input(g, "g")
        values[3] = cls.positive_scalar(C, "C")
        values[4] = cls.positive_scalar(k, "k")
        values[5] = cls.unit_scalar(Xi, "Xi")
        return values

    @classmethod
    def initial_condition(cls, init_cond):
        values = np.asarray(init_cond, dtype=float).reshape(-1)
        if values.size != 1:
            raise ValueError("BM1 init_cond must contain exactly one state value.")
        if not np.all(np.isfinite(values)):
            raise ValueError("init_cond must be finite.")
        if not 0.0 <= values[0] <= 1.0:
            raise ValueError("Initial state of charge must lie in [0, 1].")
        return values

    @classmethod
    def diffusion(cls, X, sigma_x):
        values = np.asarray(X, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("X must contain finite values.")
        sigma_x = cls.finite_scalar(sigma_x, "sigma_x")
        if sigma_x < 0.0:
            raise ValueError("sigma_x must be non-negative.")
        return values, sigma_x

    @classmethod
    def dynamics(cls, U, f, g, C, k, Xi):
        return cls.parameters([U, f, g, C, k, Xi])
