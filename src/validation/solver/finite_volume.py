import numpy as np

class FiniteVolume2DValidation:
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
