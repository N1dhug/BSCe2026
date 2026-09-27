import numpy as np

from src.validation.model.bm import bm_validation


class bm_1_validation:
    """Validation for bm_1 solver and optimizer entry points."""

    @staticmethod
    def model_inputs(init_cond, parameters):
        return (bm_validation.initial_condition(init_cond), bm_validation.parameters(parameters))

    @staticmethod
    def time_controls(t_eval, breakpoints, interval, max_step):
        """Validate time/output controls shared by deterministic and EM solvers."""
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
    def stochastic_inputs(init_cond, parameters, sigma_x):
        """Validate the common F2 contract for scalar-initial-state entry points.

        sigma_x=0 is allowed for deterministic-limit tests. Endpoint checks
        are kept out of model_inputs so the deterministic F1 API is unchanged.
        """
        init_cond, parameters = bm_1_validation.model_inputs(init_cond, parameters)
        _, sigma_x = bm_validation.diffusion(init_cond, sigma_x)
        bm_validation.response_endpoints(parameters[1], parameters[2])
        return (init_cond, parameters, sigma_x)

    @staticmethod
    def stochastic_path_inputs(init_cond, parameters, sigma_x):
        """Validate F2 EM inputs with one shared or one initial state per path."""
        parameters = bm_validation.parameters(parameters)
        initial_states = np.asarray(init_cond, dtype=float).reshape(-1)
        if initial_states.size == 0 or not np.all(np.isfinite(initial_states)):
            raise ValueError('init_cond must contain at least one finite state value.')
        if np.any(initial_states < 0.0) or np.any(initial_states > 1.0):
            raise ValueError('Initial state of charge values must lie in [0, 1].')
        _, sigma_x = bm_validation.diffusion(initial_states, sigma_x)
        bm_validation.response_endpoints(parameters[1], parameters[2])
        return (initial_states, parameters, sigma_x)

    @staticmethod
    def stochastic_time_inputs(t_eval, breakpoints, interval, max_step):
        """Validate the EM output grid, discontinuities and maximum internal step."""
        t_eval, breakpoints, max_step = bm_1_validation.time_controls(
            t_eval,
            breakpoints,
            interval,
            max_step,
        )
        if t_eval.size < 2:
            raise ValueError('Euler-Maruyama t_eval must contain at least two points.')
        if t_eval[0] != interval[0] or t_eval[-1] != interval[1]:
            raise ValueError('Euler-Maruyama t_eval must span the full model time interval.')
        return (t_eval, breakpoints, max_step)

    @staticmethod
    def optimize_inputs(
        initial_u,
        interval_edges,
        init_cond,
        parameters,
        t_eval,
        interval,
        rtol,
        atol,
        max_step,
    ):
        """Validate the interval controls and deterministic objective grid."""
        initial_u = np.asarray(initial_u, dtype=float)
        if initial_u.ndim != 1 or initial_u.size == 0:
            raise ValueError('initial_u must be a non-empty one-dimensional vector.')
        if not np.all(np.isfinite(initial_u)):
            raise ValueError('initial_u must contain finite values.')
        if np.any(initial_u < 0.0) or np.any(initial_u > 1.0):
            raise ValueError('initial_u values must lie in [0, 1].')
        interval_edges = np.asarray(interval_edges, dtype=float)
        if interval_edges.ndim != 1:
            raise ValueError('interval_edges must be a one-dimensional array.')
        if interval_edges.size != initial_u.size + 1:
            raise ValueError('interval_edges must contain N + 1 values for N price decisions.')
        if not np.all(np.isfinite(interval_edges)):
            raise ValueError('interval_edges must contain finite values.')
        if np.any(np.diff(interval_edges) <= 0.0):
            raise ValueError('interval_edges must be strictly increasing.')
        t_start, t_end = interval
        if interval_edges[0] != t_start or interval_edges[-1] != t_end:
            raise ValueError('interval_edges must start and end at the model time interval.')
        init_cond, parameters, t_eval, _, rtol, atol, max_step = bm_1_validation.deterministic_inputs(
            init_cond,
            parameters,
            t_eval,
            interval_edges[1:-1],
            interval,
            rtol,
            atol,
            max_step,
        )
        if t_eval.size < 2 or t_eval[0] != t_start or t_eval[-1] != t_end:
            raise ValueError('The optimization t_eval grid must span the full model interval.')
        return (
            initial_u.copy(),
            interval_edges.copy(),
            init_cond,
            parameters,
            t_eval,
            rtol,
            atol,
            max_step,
        )
