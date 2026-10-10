import numpy as np

from src.validation.model.bm import bm_validation


class bm_1_validation:
    """Validation for bm_1 deterministic solver and diffusion entry points."""

    @staticmethod
    def diffusion_inputs(X, sigma_X):
        """Require real states in [0, 1] and a finite nonnegative intensity."""
        if np.iscomplexobj(X) or np.iscomplexobj(sigma_X):
            raise ValueError('X and sigma_X must be real.')
        X = bm_validation.bounded_array(X, 0.0, 1.0, 'X', tol=0.0)
        sigma_X = bm_validation.finite_scalar(sigma_X, 'sigma_X')
        if sigma_X < 0.0:
            raise ValueError('sigma_X must be nonnegative.')
        return X, sigma_X

    @staticmethod
    def model_inputs(init_cond, parameters):
        return (bm_validation.initial_condition(init_cond), bm_validation.parameters(parameters))

    @staticmethod
    def time_controls(t_eval, breakpoints, interval, max_step):
        """Validate time/output controls for deterministic solvers."""
        t_eval = np.asarray(t_eval, dtype=float)
        if t_eval.ndim != 1 or t_eval.size == 0:
            raise ValueError('t_eval must be a non-empty one-dimensional grid.')
        if not np.all(np.isfinite(t_eval)):
            raise ValueError('t_eval must contain finite values.')
        if np.any(np.diff(t_eval) <= 0.0):
            raise ValueError('t_eval must be strictly increasing.')
        t_start, t_end = interval
        if np.any(t_eval < t_start) or np.any(t_eval > t_end):
            raise ValueError('t_eval values must lie inside the model time interval.')
        if breakpoints is None:
            breakpoints = np.empty(0, dtype=float)
        else:
            breakpoints = np.asarray(breakpoints, dtype=float)
            if breakpoints.ndim != 1:
                raise ValueError('breakpoints must be a one-dimensional array.')
            if not np.all(np.isfinite(breakpoints)):
                raise ValueError('breakpoints must contain finite values.')
            if np.any(np.diff(breakpoints) <= 0.0):
                raise ValueError('breakpoints must be strictly increasing.')
            if np.any(breakpoints <= t_start) or np.any(breakpoints >= t_end):
                raise ValueError('breakpoints must lie strictly inside the model time interval.')
        max_step_array = np.asarray(max_step, dtype=float)
        if max_step_array.size != 1:
            raise ValueError('max_step must be a scalar.')
        max_step = float(max_step_array.reshape(-1)[0])
        if np.isnan(max_step) or max_step <= 0.0:
            raise ValueError('max_step must be positive or np.inf.')
        return (t_eval, breakpoints, max_step)

    @staticmethod
    def deterministic_inputs(init_cond, parameters, t_eval, breakpoints, interval, rtol, atol, max_step):
        """Validate deterministic solver inputs and numerical controls."""
        init_cond, parameters = bm_1_validation.model_inputs(init_cond, parameters)
        t_eval, breakpoints, max_step = bm_1_validation.time_controls(
            t_eval,
            breakpoints,
            interval,
            max_step,
        )
        rtol = bm_validation.positive_scalar(rtol, 'rtol')
        atol = bm_validation.positive_scalar(atol, 'atol')
        return (init_cond, parameters, t_eval, breakpoints, rtol, atol, max_step)

    @staticmethod
    def reference_solver_inputs(default_t, t_eval, parameters, sigma_x):
        """Validate density forecasts while retaining the deterministic interface."""
        if t_eval is None:
            t_eval = default_t.copy()
        _, parameters = bm_1_validation.model_inputs([0.5], parameters)
        _, sigma_x = bm_1_validation.diffusion_inputs(0.5, sigma_x)
        for response in parameters[1:3]:
            endpoints = [bm_validation.finite_scalar(response(x), "response endpoint")
                         for x in (0.0, 1.0)]
            if not np.allclose(endpoints, [1.0, -1.0], rtol=0.0, atol=1e-10):
                raise ValueError("BM1 density forecasts require responses from +1 to -1.")
        return t_eval, parameters, sigma_x
