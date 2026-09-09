import numpy as np

from src.validation.model.BM import BMValidation


class BM1Validation:
    """Validation for BM1 solver and optimizer entry points."""

    @staticmethod
    def model_inputs(init_cond, parameters):
        return BMValidation.initial_condition(init_cond), BMValidation.parameters(parameters)

    @staticmethod
    def stochastic_inputs(init_cond, parameters, sigma_x):
        init_cond, parameters = BM1Validation.model_inputs(init_cond, parameters)
        _, sigma_x = BMValidation.diffusion(init_cond, sigma_x)
        return init_cond, parameters, sigma_x

    @staticmethod
    def optimize_inputs(initial_u, init_cond, parameters):
        initial_u = np.asarray(initial_u, dtype=float)
        if initial_u.ndim != 1 or initial_u.size == 0:
            raise ValueError("initial_u must be a non-empty one-dimensional vector.")
        if not np.all(np.isfinite(initial_u)):
            raise ValueError("initial_u must contain finite values.")
        if np.any(initial_u < 0.0) or np.any(initial_u > 1.0):
            raise ValueError("initial_u values must lie in [0, 1].")

        init_cond, parameters = BM1Validation.model_inputs(init_cond, parameters)
        return initial_u, init_cond, parameters
