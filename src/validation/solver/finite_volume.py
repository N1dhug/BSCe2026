import numpy as np

class finite_volume_2d_validation:
    @staticmethod
    def bounds(values, name):
        # Verify inputs
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain two finite values.")
        lo, hi = map(float, values)
        if lo >= hi:
            raise ValueError(f"{name} must satisfy lower < upper.")
        return lo, hi

    @staticmethod
    def integer(value, name, minimum=3):
        # Verify inputs
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer.")
        ivalue = int(value)
        if float(ivalue) != float(value) or ivalue < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
        return ivalue

    @staticmethod
    def cfl(value):
        # Verify inputs
        value = float(value)
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("cfl must lie in (0, 1].")
        return value

    @staticmethod
    def time_grid(t):
        # Verify inputs
        t = np.asarray(t, dtype=float).reshape(-1)
        if t.size < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
            raise ValueError("t must be a strictly increasing finite time grid.")
        return t

    @staticmethod
    def density(density, shape, dv, dr):
        # Verify inputs
        density = np.asarray(density, dtype=float)
        if density.shape != shape:
            raise ValueError(f"initial density must have shape {shape}.")
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("initial density must be finite and non-negative.")
        mass = float(np.sum(density) * dv * dr)
        if mass <= 0.0:
            raise ValueError("initial density must have positive mass.")
        return density / mass

    @staticmethod
    def field(values, shape, name, nonnegative=False):
        # Verify inputs
        values = np.asarray(values, dtype=float)
        if values.shape != shape:
            try:
                values = np.broadcast_to(values, shape).astype(float, copy=False)
            except ValueError as exc:
                raise ValueError(f"{name} cannot be broadcast to {shape}.") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite values.")
        if nonnegative and np.any(values < 0.0):
            raise ValueError(f"{name} must be non-negative.")
        return values


class finite_volume_1d_validation:
    """Validation for the conservative one-dimensional Fokker-Planck reference."""

    @staticmethod
    def bounds(values, name="bounds"):
        return finite_volume_2d_validation.bounds(values, name)

    @staticmethod
    def integer(value, name="x_steps", minimum=3):
        return finite_volume_2d_validation.integer(value, name, minimum=minimum)

    @staticmethod
    def time_grid(t, name="t_eval"):
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
        t_eval = cls.time_grid(t_eval)

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
                raise ValueError("breakpoints must lie strictly inside the output interval.")

        max_step_array = np.asarray(max_step, dtype=float)
        if max_step_array.size != 1:
            raise ValueError("max_step must be a scalar.")
        max_step = float(max_step_array.reshape(-1)[0])
        if np.isnan(max_step) or max_step <= 0.0:
            raise ValueError("max_step must be positive or np.inf.")

        return t_eval, breakpoints, max_step

    @staticmethod
    def initial_mass(values):
        mass = np.asarray(values, dtype=float)
        if mass.ndim != 1 or mass.size < 3:
            raise ValueError("initial_mass must be a one-dimensional array with at least three cells.")
        if not np.all(np.isfinite(mass)):
            raise ValueError("initial_mass must contain finite values.")
        if np.any(mass < 0.0):
            raise ValueError("initial_mass must be non-negative.")
        total = float(np.sum(mass))
        if not np.isclose(total, 1.0, atol=1.0e-10, rtol=0.0):
            raise ValueError(
                "initial_mass must sum to one; the reference solver does not renormalize it."
            )
        return mass.copy()

    @staticmethod
    def field(values, shape, name):
        try:
            field = np.broadcast_to(np.asarray(values, dtype=float), shape)
        except ValueError as exc:
            raise ValueError(f"{name} cannot be broadcast to {shape}.") from exc
        if not np.all(np.isfinite(field)):
            raise ValueError(f"{name} must contain finite values.")
        return np.asarray(field, dtype=float)

    @classmethod
    def generator_inputs(cls, dx, drift_faces, diffusion_centers):
        dx_array = np.asarray(dx, dtype=float)
        if dx_array.size != 1:
            raise ValueError("dx must be a scalar.")
        dx = float(dx_array.reshape(-1)[0])
        if not np.isfinite(dx) or dx <= 0.0:
            raise ValueError("dx must be finite and strictly positive.")

        diffusion_centers = np.asarray(diffusion_centers, dtype=float)
        if diffusion_centers.ndim != 1 or diffusion_centers.size < 3:
            raise ValueError("diffusion_centers must contain at least three cell values.")
        if not np.all(np.isfinite(diffusion_centers)):
            raise ValueError("diffusion_centers must contain finite values.")

        drift_faces = cls.field(
            drift_faces,
            (diffusion_centers.size - 1,),
            "drift_faces",
        )
        return dx, drift_faces, diffusion_centers

    @staticmethod
    def evolved_mass(values, tolerance=1.0e-12):
        mass = np.asarray(values, dtype=float)
        if mass.ndim != 1 or not np.all(np.isfinite(mass)):
            raise RuntimeError("Finite-volume cell masses became invalid.")
        if np.min(mass) < -tolerance:
            raise RuntimeError(
                "Finite-volume reference produced a negative cell mass; no clipping was applied."
            )
        return mass
