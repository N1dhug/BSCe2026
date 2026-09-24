"""Reusable numerical objective functions."""

import numpy as np
from scipy.integrate import simpson

from src.validation.method.objective import objective_validation


def tracking_error(time, demand, target):
    """Return the integrated squared error between demand and target."""
    time, demand, target = objective_validation.tracking_inputs(
        time, demand, target
    )
    return float(simpson(np.square(demand - target), x=time))
