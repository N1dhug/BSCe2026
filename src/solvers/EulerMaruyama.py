import numpy as np


class EulerMaruyama:
    """Generic Euler-Maruyama path solver for one-dimensional Ito SDEs.

    We approximate solutions of
        dX_t = mu(t, X_t) dt + sigma(t, X_t) dW_t,
    where mu is the drift, sigma is the diffusion, and W_t is Brownian motion.
    The Euler-Maruyama update is
        X_{n+1} = X_n + mu(t_n, X_n) * dt + sigma(t_n, X_n) * sqrt(dt) * Z_n,
    with Z_n ~ N(0, 1).
    """

    @staticmethod
    def apply_boundary(values, boundary):
        """Apply a boundary rule to generated values.

        For a reflected boundary on [0, 1], the map x -> x mod 2 followed by
        folding at 1 is equivalent to the standard reflection operator:
            reflect(x) = 1 - |1 - (x mod 2)|.
        For clipping, values are truncated to [0, 1].
        """
        values = np.asarray(values, dtype=float)
        if boundary == "reflect":
            """Reflection on [0, 1] is achieved by folding x modulo 2:
            x mod 2 in [0, 2), then map [1, 2) to [1, 0] by 2 - x."""
            reflected = np.mod(values, 2.0)
            return np.where(reflected <= 1.0, reflected, 2.0 - reflected)
        if boundary == "clip":
            """Truncate the proposal value to the admissible interval [0, 1]."""
            return np.clip(values, 0.0, 1.0)
        if boundary == "none":
            return values
        return values

    def solve_paths(self,t,init_cond,drift,diffusion,n_paths=1,seed=None,boundary="reflect",):
        """Solve dX = drift(t, X) dt + diffusion(t, X) dW on a supplied time grid."""
        """Time grid t = [t_0, t_1, ..., t_N]. For each interval [t_i, t_{i+1}]
        of length h = t_{i+1} - t_i, the Euler-Maruyama update is used."""
        t = np.asarray(t, dtype=float)
        n_paths = int(n_paths)

        """Initial condition X_0 = x0. In a scalar SDE, x0 is a single value."""
        x0 = np.asarray(init_cond, dtype=float)
        x0 = float(x0.reshape(-1)[0])

        rng = np.random.default_rng(seed)
        """paths[n, i] stores the i-th time point of the n-th simulated path."""
        paths = np.empty((n_paths, len(t)), dtype=float)
        paths[:, 0] = x0

        for i in range(len(t) - 1):
            t_i = t[i]
            h = t[i + 1] - t[i]
            X_i = paths[:, i]

            """Evaluate drift and diffusion at the current time and state:
            mu_i = mu(t_i, X_i), sigma_i = sigma(t_i, X_i)"""
            drift_i = np.asarray(drift(t_i, X_i))
            diffusion_i = np.asarray(diffusion(t_i, X_i))

            """Brownian increment: Delta W_i = W_{t_{i+1}} - W_{t_i}
            has distribution N(0, h). Hence sqrt(h) * Z_i with Z_i ~ N(0,1)."""
            dW = np.sqrt(h) * rng.standard_normal(n_paths)

            """Euler-Maruyama proposal:
            X_{i+1}^* = X_i + mu_i h + sigma_i Delta W_i"""
            proposal = X_i + drift_i * h + diffusion_i * dW

            """Enforce the chosen domain constraint (reflection, clipping, or none)."""
            paths[:, i + 1] = self.apply_boundary(proposal, boundary)

        return paths
