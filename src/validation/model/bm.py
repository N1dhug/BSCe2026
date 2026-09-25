import numpy as np

class bm_validation:
    """Validation rules shared by Battery Model implementations."""

    @staticmethod
    def finite_scalar(value, name):
        # Verify inputs
        array = np.asarray(value, dtype=float)
        if array.size != 1:
            raise ValueError(f"{name} must be a scalar.")
        scalar = float(array.reshape(-1)[0])
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be finite.")
        return scalar

    @staticmethod
    def finite_array(values, name):
        # Verify inputs
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values.")
        return array

    @classmethod
    def bounded_array(cls, values, lower, upper, name, tol=1e-10):
        # Verify inputs
        array = cls.finite_array(values, name)
        if np.any(array < lower - tol) or np.any(array > upper + tol):
            raise ValueError(f"{name} values must lie in [{lower}, {upper}].")
        return array

    @staticmethod
    def positive_scalar(value, name):
        # Verify inputs
        scalar = bm_validation.finite_scalar(value, name)
        if scalar <= 0.0:
            raise ValueError(f"{name} must be strictly positive.")
        return scalar

    @staticmethod
    def unit_scalar(value, name):
        # Verify inputs
        scalar = bm_validation.finite_scalar(value, name)
        if not 0.0 <= scalar <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")
        return scalar

    @staticmethod
    def callable_input(value, name):
        # Verify inputs
        if not callable(value):
            raise TypeError(f"{name} must be callable.")
        return value

    @classmethod
    def constructor(cls, B, A, T, t_steps):
        # Verify inputs
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
        # Verify inputs
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
    def response_endpoints(cls, f, g):
        """Check the canonical F2 endpoint normalization once at entry.

        This is not a proof of monotonicity, interior range or Lipschitz
        regularity for arbitrary callables.  Those remain model assumptions;
        response_values checks the range at evaluated points during a solve.
        The original callables are returned without rescaling or clipping.
        """
        for response, name in ((f, "f"), (g, "g")):
            cls.callable_input(response, name)
            values = np.array([
                cls.finite_scalar(response(0.0), f"{name}(0)"),
                cls.finite_scalar(response(1.0), f"{name}(1)"),
            ])
            if not np.allclose(values, [1.0, -1.0], atol=1.0e-10, rtol=0.0):
                raise ValueError(
                    f"F2 requires {name}(0) = 1 and {name}(1) = -1 "
                    "under the canonical BM1 normalization."
                )
        return f, g

    @classmethod
    def initial_condition(cls, init_cond):
        # Verify inputs
        values = np.asarray(init_cond, dtype=float).reshape(-1)
        if values.size != 1:
            raise ValueError("bm_1 init_cond must contain exactly one state value.")
        if not np.all(np.isfinite(values)):
            raise ValueError("init_cond must be finite.")
        if not 0.0 <= values[0] <= 1.0:
            raise ValueError("Initial state of charge must lie in [0, 1].")
        return values

    @classmethod
    def state(cls, state):
        # Verify inputs
        values = np.asarray(state)
        if values.size == 0:
            raise ValueError("state must contain a state-of-charge value.")
        return values

    @classmethod
    def diffusion(cls, X, sigma_x):
        # Verify inputs
        values = cls.finite_array(X, "X")
        sigma_x = cls.finite_scalar(sigma_x, "sigma_x")
        if sigma_x < 0.0:
            raise ValueError("sigma_x must be non-negative.")
        return values, sigma_x

    @classmethod
    def dynamics(cls, U, f, g, C, k, Xi):
        # Verify inputs
        return cls.parameters([U, f, g, C, k, Xi])

    @classmethod
    def response_values(cls, Ut, Bt, fXt, gUt):
        # Verify inputs
        return (cls.bounded_array(Ut, 0.0, 1.0, "U(t)"), cls.bounded_array(Bt, 0.0, 1.0, "B(t)"), cls.bounded_array(fXt, -1.0, 1.0, "f(X)"), cls.bounded_array(gUt, -1.0, 1.0, "g(U)"),)
