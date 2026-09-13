import numpy as np

class EulerMaruyamaValidation:
    """Input validation for the generic Euler-Maruyama method."""

    BOUNDARIES = {"reflect", "clip", "none"}

    @staticmethod
    def _positive_integer(value, name):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer.")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be an integer.") from exc
        if float(integer) != float(value):
            raise ValueError(f"{name} must be an integer.")
        if integer < 1:
            raise ValueError(f"{name} must be at least one.")
        return integer

    @classmethod
    def boundary(cls, boundary):
        if boundary not in cls.BOUNDARIES:
            allowed = ", ".join(sorted(cls.BOUNDARIES))
            raise ValueError(f"boundary must be one of: {allowed}.")
        return boundary

    @classmethod
    def boundary_values(cls, values, boundary):
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Boundary values must be finite.")
        return values, cls.boundary(boundary)

    @classmethod
    def solve_inputs(cls, t, init_cond, drift, diffusion, n_paths, seed, boundary):
        t = np.asarray(t, dtype=float)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("t must be a one-dimensional grid with at least two points.")
        if not np.all(np.isfinite(t)):
            raise ValueError("t must contain finite values.")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("t must be strictly increasing.")

        init = np.asarray(init_cond, dtype=float).reshape(-1)
        if init.size != 1 or not np.all(np.isfinite(init)):
            raise ValueError("init_cond must contain exactly one finite state value.")

        if not callable(drift):
            raise TypeError("drift must be callable.")
        if not callable(diffusion):
            raise TypeError("diffusion must be callable.")

        n_paths = cls._positive_integer(n_paths, "n_paths")

        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer or None.")

        boundary = cls.boundary(boundary)
        return t, float(init[0]), drift, diffusion, n_paths, seed, boundary

    @staticmethod
    def step_values(drift_values, diffusion_values, shape):
        try:
            drift_values = np.broadcast_to(np.asarray(drift_values, dtype=float), shape)
        except ValueError as exc:
            raise ValueError("drift output is not broadcastable to the path state shape.") from exc
        try:
            diffusion_values = np.broadcast_to(np.asarray(diffusion_values, dtype=float), shape)
        except ValueError as exc:
            raise ValueError("diffusion output is not broadcastable to the path state shape.") from exc

        if not np.all(np.isfinite(drift_values)):
            raise ValueError("drift returned non-finite values.")
        if not np.all(np.isfinite(diffusion_values)):
            raise ValueError("diffusion returned non-finite values.")

        return drift_values, diffusion_values
