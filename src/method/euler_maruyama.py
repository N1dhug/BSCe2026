import numpy as np

from src.validation.method.euler_maruyama import EulerMaruyamaValidation

class EulerMaruyama:
    """Generic Euler-Maruyama path solver for one-dimensional Ito SDEs."""

    @staticmethod
    def apply_boundary(values, boundary):
        # Apply boundary condition
        values, boundary = EulerMaruyamaValidation.boundary_values(values, boundary)
        if boundary == "reflect":
            reflected = np.mod(values, 2.0)
            return np.where(reflected <= 1.0, reflected, 2.0 - reflected)
        if boundary == "clip":
            return np.clip(values, 0.0, 1.0)
        return values

    def solve_paths(self, t, init_cond, drift, diffusion, n_paths=1, seed=None, boundary="reflect"):
        """Solve dX = drift(t,X)dt + diffusion(t,X)dW on a supplied time grid."""
        # Solve stochastic paths
        t, x0, drift, diffusion, n_paths, seed, boundary = EulerMaruyamaValidation.solve_inputs(t, init_cond, drift, diffusion, n_paths, seed, boundary)

        rng = np.random.default_rng(seed)
        paths = np.empty((n_paths, len(t)), dtype=float)
        paths[:, 0] = x0

        for i in range(len(t) - 1):
            h = t[i + 1] - t[i]
            X_i = paths[:, i]
            drift_i, diffusion_i = EulerMaruyamaValidation.step_values(drift(t[i], X_i), diffusion(t[i], X_i), X_i.shape)
            dW = np.sqrt(h) * rng.standard_normal(n_paths)
            proposal = X_i + drift_i * h + diffusion_i * dW
            paths[:, i + 1] = self.apply_boundary(proposal, boundary)

        return paths
