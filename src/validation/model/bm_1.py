import numpy as np

from src.validation.model.bm import BMValidation

class BM1Validation:
    """Validation for BM1 solver and optimizer entry points."""

    @staticmethod
    def model_inputs(init_cond, parameters):
        # Verify inputs
        return BMValidation.initial_condition(init_cond), BMValidation.parameters(parameters)

    @staticmethod
    def deterministic_inputs(
        init_cond,
        parameters,
        t_eval,
        breakpoints,
        interval,
        rtol,
        atol,
        max_step,
    ):
        """Validate deterministic solver inputs and numerical controls."""
        init_cond, parameters = BM1Validation.model_inputs(init_cond, parameters)

        t_eval = np.asarray(t_eval, dtype=float)
        if t_eval.ndim != 1 or t_eval.size == 0:
            raise ValueError("t_eval must be a non-empty one-dimensional grid.")
        if not np.all(np.isfinite(t_eval)):
            raise ValueError("t_eval must contain finite values.")
        if np.any(np.diff(t_eval) <= 0.0):
            raise ValueError("t_eval must be strictly increasing.")

        t_start, t_end = interval
        if np.any(t_eval < t_start) or np.any(t_eval > t_end):
            raise ValueError("t_eval values must lie inside the model time interval.")

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
            if np.any(breakpoints <= t_start) or np.any(breakpoints >= t_end):
                raise ValueError(
                    "breakpoints must lie strictly inside the model time interval."
                )

        rtol = BMValidation.positive_scalar(rtol, "rtol")
        atol = BMValidation.positive_scalar(atol, "atol")

        max_step_array = np.asarray(max_step, dtype=float)
        if max_step_array.size != 1:
            raise ValueError("max_step must be a scalar.")
        max_step = float(max_step_array.reshape(-1)[0])
        if np.isnan(max_step) or max_step <= 0.0:
            raise ValueError("max_step must be positive or np.inf.")

        return (
            init_cond,
            parameters,
            t_eval,
            breakpoints,
            rtol,
            atol,
            max_step,
        )

    @staticmethod
    def stochastic_inputs(init_cond, parameters, sigma_x):
        # Verify inputs
        init_cond, parameters = BM1Validation.model_inputs(init_cond, parameters)
        _, sigma_x = BMValidation.diffusion(init_cond, sigma_x)
        return init_cond, parameters, sigma_x

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
            raise ValueError("initial_u must be a non-empty one-dimensional vector.")
        if not np.all(np.isfinite(initial_u)):
            raise ValueError("initial_u must contain finite values.")
        if np.any(initial_u < 0.0) or np.any(initial_u > 1.0):
            raise ValueError("initial_u values must lie in [0, 1].")

        interval_edges = np.asarray(interval_edges, dtype=float)
        if interval_edges.ndim != 1:
            raise ValueError("interval_edges must be a one-dimensional array.")
        if interval_edges.size != initial_u.size + 1:
            raise ValueError(
                "interval_edges must contain N + 1 values for N price decisions."
            )
        if not np.all(np.isfinite(interval_edges)):
            raise ValueError("interval_edges must contain finite values.")
        if np.any(np.diff(interval_edges) <= 0.0):
            raise ValueError("interval_edges must be strictly increasing.")

        t_start, t_end = interval
        if interval_edges[0] != t_start or interval_edges[-1] != t_end:
            raise ValueError(
                "interval_edges must start and end at the model time interval."
            )

        (
            init_cond,
            parameters,
            t_eval,
            _,
            rtol,
            atol,
            max_step,
        ) = BM1Validation.deterministic_inputs(
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
            raise ValueError(
                "The optimization t_eval grid must span the full model interval."
            )

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
