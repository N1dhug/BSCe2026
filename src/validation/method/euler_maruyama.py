import numpy as np

class euler_maruyama_validation:
    """Input validation for the generic Euler-Maruyama method."""

    BOUNDARIES = {"reflect", "clip", "none"}

    @staticmethod
    def _positive_integer(value, name):
        # Verify inputs
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
    def path_count(cls, n_paths):
        # Verify path count
        return cls._positive_integer(n_paths, "n_paths")

    @staticmethod
    def seed(seed):
        # Verify random seed
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer or None.")
        return seed

    @staticmethod
    def time_grid(t, name="t"):
        # Verify time grid
        t = np.asarray(t, dtype=float)
        if t.ndim != 1 or t.size < 2:
            raise ValueError(f"{name} must be a one-dimensional grid with at least two points.")
        if not np.all(np.isfinite(t)):
            raise ValueError(f"{name} must contain finite values.")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError(f"{name} must be strictly increasing.")
        return t

    @classmethod
    def integration_grid_inputs(cls, t_eval, breakpoints, max_step):
        # Verify integration-grid controls
        t_eval = cls.time_grid(t_eval, "t_eval")

        if breakpoints is None:
            breakpoints = np.empty(0, dtype=float)
        else:
            breakpoints = np.asarray(breakpoints, dtype=float)
            if breakpoints.ndim != 1:
                raise ValueError("breakpoints must be a one-dimensional array.")
            if not np.all(np.isfinite(breakpoints)):
                raise ValueError("breakpoints must contain finite values.")
            if np.any(np.diff(breakpoints) <= 0.0):
                raise ValueError("breakpoints must be strictly increasing.")
            if np.any(breakpoints <= t_eval[0]) or np.any(breakpoints >= t_eval[-1]):
                raise ValueError("breakpoints must lie strictly inside the output time interval.")

        max_step_array = np.asarray(max_step, dtype=float)
        if max_step_array.size != 1:
            raise ValueError("max_step must be a scalar.")
        max_step = float(max_step_array.reshape(-1)[0])
        if np.isnan(max_step) or max_step <= 0.0:
            raise ValueError("max_step must be positive or np.inf.")

        return t_eval, breakpoints, max_step

    @classmethod
    def boundary(cls, boundary):
        # Verify inputs
        if boundary not in cls.BOUNDARIES:
            allowed = ", ".join(sorted(cls.BOUNDARIES))
            raise ValueError(f"boundary must be one of: {allowed}.")
        return boundary

    @classmethod
    def boundary_values(cls, values, boundary):
        # Verify inputs
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Boundary values must be finite.")
        return values, cls.boundary(boundary)

    @classmethod
    def initial_states(cls, init_cond, n_paths):
        # Verify one shared initial state or one initial state per path
        initial = np.asarray(init_cond, dtype=float).reshape(-1)
        if initial.size not in (1, n_paths) or not np.all(np.isfinite(initial)):
            raise ValueError(
                "init_cond must contain one finite state value or exactly n_paths finite values."
            )
        if initial.size == 1:
            return np.full(n_paths, float(initial[0]), dtype=float)
        return initial.copy()

    @classmethod
    def brownian_increment_inputs(cls, increments, n_paths, n_steps, seed):
        # Verify externally supplied Wiener increments
        if increments is None:
            return None
        if seed is not None:
            raise ValueError("seed must be None when brownian_increments are supplied.")

        increments = np.asarray(increments, dtype=float)
        if increments.ndim == 1 and n_paths == 1:
            increments = increments[None, :]
        if increments.shape != (n_paths, n_steps):
            raise ValueError(
                "brownian_increments must have shape (n_paths, len(t) - 1)."
            )
        if not np.all(np.isfinite(increments)):
            raise ValueError("brownian_increments must contain finite values.")
        return increments.copy()

    @classmethod
    def solve_inputs(
        cls,
        t,
        init_cond,
        drift,
        diffusion,
        n_paths,
        seed,
        boundary,
        brownian_increments,
        return_diagnostics,
    ):
        # Verify inputs
        t = cls.time_grid(t)
        n_paths = cls.path_count(n_paths)
        initial_states = cls.initial_states(init_cond, n_paths)

        if not callable(drift):
            raise TypeError("drift must be callable.")
        if not callable(diffusion):
            raise TypeError("diffusion must be callable.")

        seed = cls.seed(seed)
        boundary = cls.boundary(boundary)
        brownian_increments = cls.brownian_increment_inputs(
            brownian_increments,
            n_paths,
            len(t) - 1,
            seed,
        )

        if not isinstance(return_diagnostics, (bool, np.bool_)):
            raise TypeError("return_diagnostics must be a boolean.")

        return (
            t,
            initial_states,
            drift,
            diffusion,
            n_paths,
            seed,
            boundary,
            brownian_increments,
            bool(return_diagnostics),
        )

    @classmethod
    def coarsen_inputs(cls, increments, factor):
        # Verify Brownian coarsening inputs
        increments = np.asarray(increments, dtype=float)
        if increments.ndim == 1:
            increments = increments[None, :]
        if increments.ndim != 2 or increments.shape[1] == 0:
            raise ValueError("increments must be a non-empty one- or two-dimensional array.")
        if not np.all(np.isfinite(increments)):
            raise ValueError("increments must contain finite values.")

        factor = cls._positive_integer(factor, "factor")
        if increments.shape[1] % factor != 0:
            raise ValueError("The number of fine Brownian increments must be divisible by factor.")
        return increments, factor

    @staticmethod
    def step_values(drift_values, diffusion_values, shape):
        # Verify inputs
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
