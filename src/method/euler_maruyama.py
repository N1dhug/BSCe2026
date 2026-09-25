import numpy as np

from src.validation.method.euler_maruyama import euler_maruyama_validation

class euler_maruyama:
    """Generic Euler-Maruyama path solver for one-dimensional Ito SDEs."""

    @staticmethod
    def apply_boundary(values, boundary):
        # Apply boundary condition
        values, boundary = euler_maruyama_validation.boundary_values(values, boundary)
        if boundary == "reflect":
            reflected = np.mod(values, 2.0)
            return np.where(reflected <= 1.0, reflected, 2.0 - reflected)
        if boundary == "clip":
            return np.clip(values, 0.0, 1.0)
        return values

    @staticmethod
    def build_time_grid(t_eval, breakpoints=None, max_step=np.inf):
        """Build an integration grid containing every output time and breakpoint."""
        # Build integration grid
        t_eval, breakpoints, max_step = euler_maruyama_validation.integration_grid_inputs(
            t_eval, breakpoints, max_step
        )
        knots = np.unique(np.concatenate((t_eval, breakpoints)))
        pieces = []

        for index, (left, right) in enumerate(zip(knots[:-1], knots[1:])):
            width = right - left
            if np.isinf(max_step):
                n_substeps = 1
            else:
                ratio = width / max_step
                n_substeps = max(
                    1,
                    int(np.ceil(np.nextafter(ratio, -np.inf))),
                )

            segment = np.linspace(left, right, n_substeps + 1, dtype=float)
            segment[0], segment[-1] = left, right
            pieces.append(segment if index == 0 else segment[1:])

        integration_t = np.concatenate(pieces)
        output_indices = np.searchsorted(integration_t, t_eval)

        if np.any(output_indices >= integration_t.size) or not np.allclose(
            integration_t[output_indices], t_eval, atol=1.0e-14, rtol=0.0
        ):
            raise RuntimeError("Failed to embed every output time in the integration grid.")

        return integration_t, output_indices

    @staticmethod
    def brownian_increments(t, n_paths=1, seed=None):
        """Generate independent Wiener increments for a supplied integration grid."""
        # Generate Brownian increments
        t = euler_maruyama_validation.time_grid(t)
        n_paths = euler_maruyama_validation.path_count(n_paths)
        seed = euler_maruyama_validation.seed(seed)
        rng = np.random.default_rng(seed)
        return rng.standard_normal((len(t) - 1, n_paths)).T * np.sqrt(np.diff(t))[None, :]

    @staticmethod
    def coarsen_brownian_increments(increments, factor=2):
        """Sum consecutive fine Wiener increments to obtain a coupled coarse path."""
        # Coarsen Brownian increments
        increments, factor = euler_maruyama_validation.coarsen_inputs(increments, factor)
        n_paths, n_steps = increments.shape
        return increments.reshape(n_paths, n_steps // factor, factor).sum(axis=2)

    def solve_paths(
        self,
        t,
        init_cond,
        drift,
        diffusion,
        n_paths=1,
        seed=None,
        boundary="reflect",
        *,
        brownian_increments=None,
        return_diagnostics=False,
    ):
        """Solve dX = drift(t,X)dt + diffusion(t,X)dW on a supplied time grid.

        ``init_cond`` may contain one shared state or one state per path.
        Externally supplied Brownian increments make path coupling reproducible.
        Boundary diagnostics are computed from the raw proposal before any
        optional reflection or clipping.
        """
        # Solve stochastic paths
        (
            t,
            initial_states,
            drift,
            diffusion,
            n_paths,
            seed,
            boundary,
            brownian_increments,
            return_diagnostics,
        ) = euler_maruyama_validation.solve_inputs(
            t,
            init_cond,
            drift,
            diffusion,
            n_paths,
            seed,
            boundary,
            brownian_increments,
            return_diagnostics,
        )

        if brownian_increments is None:
            brownian_increments = self.brownian_increments(t, n_paths=n_paths, seed=seed)

        paths = np.empty((n_paths, len(t)), dtype=float)
        paths[:, 0] = initial_states

        if return_diagnostics:
            lower_by_step = np.zeros(len(t) - 1, dtype=int)
            upper_by_step = np.zeros(len(t) - 1, dtype=int)
            path_violations = np.zeros(n_paths, dtype=bool)
            minimum_proposal = np.inf
            maximum_proposal = -np.inf

        for i in range(len(t) - 1):
            h = t[i + 1] - t[i]
            X_i = paths[:, i]
            drift_i, diffusion_i = euler_maruyama_validation.step_values(
                drift(t[i], X_i), diffusion(t[i], X_i), X_i.shape
            )
            proposal = X_i + drift_i * h + diffusion_i * brownian_increments[:, i]

            if return_diagnostics:
                lower = proposal < 0.0
                upper = proposal > 1.0
                lower_by_step[i] = int(np.sum(lower))
                upper_by_step[i] = int(np.sum(upper))
                path_violations |= lower | upper
                minimum_proposal = min(minimum_proposal, float(np.min(proposal)))
                maximum_proposal = max(maximum_proposal, float(np.max(proposal)))

            paths[:, i + 1] = self.apply_boundary(proposal, boundary)

        if not return_diagnostics:
            return paths

        lower_violations = int(np.sum(lower_by_step))
        upper_violations = int(np.sum(upper_by_step))
        total_violations = lower_violations + upper_violations
        diagnostics = {
            "integration_t": t.copy(),
            "n_paths": n_paths,
            "n_steps": len(t) - 1,
            "boundary": boundary,
            "minimum_raw_proposal": minimum_proposal,
            "maximum_raw_proposal": maximum_proposal,
            "lower_violations": lower_violations,
            "upper_violations": upper_violations,
            "total_violations": total_violations,
            "violating_paths": int(np.sum(path_violations)),
            "lower_violations_by_step": lower_by_step,
            "upper_violations_by_step": upper_by_step,
            "boundary_corrections": total_violations if boundary != "none" else 0,
        }
        return paths, diagnostics
