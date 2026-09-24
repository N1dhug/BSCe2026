"""Validation for numerical objective-function helpers."""

import numpy as np


class objective_validation:
    """Validate grids and values used by objective functions."""

    @staticmethod
    def tracking_inputs(time, demand, target):
        """Validate an integrated squared tracking-error calculation."""
        time = np.asarray(time, dtype=float)
        demand = np.asarray(demand, dtype=float)
        target = np.asarray(target, dtype=float)

        if time.ndim != 1 or time.size < 2:
            raise ValueError(
                "time must be a one-dimensional grid with two or more points."
            )
        if demand.shape != time.shape:
            raise ValueError("demand must contain one value per time-grid point.")
        try:
            target = np.broadcast_to(target, time.shape)
        except ValueError as exc:
            raise ValueError(
                "target must be scalar or contain one value per time-grid point."
            ) from exc

        if not np.all(np.isfinite(time)):
            raise ValueError("time must contain finite values.")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("time must be strictly increasing.")
        if not np.all(np.isfinite(demand)):
            raise ValueError("demand must contain finite values.")
        if not np.all(np.isfinite(target)):
            raise ValueError("target must contain finite values.")

        return time, demand, np.asarray(target, dtype=float)
