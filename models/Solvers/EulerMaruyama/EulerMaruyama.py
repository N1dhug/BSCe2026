import numpy as np


class EulerMaruyama:
    """Generic Euler-Maruyama path solver for one-dimensional Ito SDEs."""

    @staticmethod
    def apply_boundary(values, boundary):
        values = np.asarray(values, dtype=float)

        if boundary == "reflect":
            reflected = np.mod(values, 2.0)
            return np.where(reflected <= 1.0, reflected, 2.0 - reflected)
        if boundary == "clip":
            return np.clip(values, 0.0, 1.0)
        if boundary == "none":
            return values

        raise ValueError("boundary must be 'reflect', 'clip', or 'none'.")

    def solve_paths(self,t,init_cond,drift,diffusion,n_paths=1,seed=None,boundary="reflect",):
        """Solve dX = drift(t, X) dt + diffusion(t, X) dW on a supplied time grid."""
        t = np.asarray(t, dtype=float)
        n_paths = int(n_paths)

        if t.ndim != 1 or len(t) < 2:
            raise ValueError("t must be a one-dimensional array with at least two points.")
        if n_paths < 1:
            raise ValueError("n_paths must be at least 1.")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing.")

        x0 = np.asarray(init_cond, dtype=float)
        if x0.size != 1:
            raise ValueError("EulerMaruyama currently supports one-dimensional initial conditions only.")
        x0 = float(x0.reshape(-1)[0])

        rng = np.random.default_rng(seed)
        paths = np.empty((n_paths, len(t)), dtype=float)
        paths[:, 0] = x0

        for i in range(len(t) - 1):
            t_i = float(t[i])
            h = float(t[i + 1] - t[i])
            X_i = paths[:, i]

            drift_i = np.asarray(drift(t_i, X_i), dtype=float)
            diffusion_i = np.asarray(diffusion(t_i, X_i), dtype=float)
            dW = np.sqrt(h) * rng.standard_normal(n_paths)

            proposal = X_i + drift_i * h + diffusion_i * dW
            paths[:, i + 1] = self.apply_boundary(proposal, boundary)

        return paths
