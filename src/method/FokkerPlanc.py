import numpy as np
from scipy.special import gammaln

from src.solver.ActiveSet import ActiveSet


class FokkerPlanc:
    """Generic one-dimensional Fokker-Planck method on x in [0, 1].

    The PDE time integration belongs to the method layer. Constrained least
    squares used to recover Beta-mixture weights is delegated to the generic
    solver layer.
    """

    @staticmethod
    def beta_basis(x, degree):
        x = np.asarray(x, dtype=float)
        degree = int(degree)
        m = np.arange(degree + 1, dtype=float)[:, None]
        x_row = x[None, :]
        log_coefficient = (
            np.log(degree + 1.0)
            + gammaln(degree + 1.0)
            - gammaln(m + 1.0)
            - gammaln(degree - m + 1.0)
        )
        log_basis = (
            log_coefficient
            + m * np.log(x_row)
            + (degree - m) * np.log1p(-x_row)
        )
        return np.exp(log_basis)

    @staticmethod
    def _normalise_basis_discretely(basis, dx):
        basis = np.asarray(basis, dtype=float)
        masses = np.sum(basis, axis=1) * dx
        return basis / masses[:, None]

    @staticmethod
    def initial_coefficients(init_cond, degree):
        x0 = float(init_cond)
        degree = int(degree)
        component_means = (np.arange(degree + 1, dtype=float) + 1.0) / (degree + 2.0)
        coefficients = np.zeros(degree + 1, dtype=float)

        if x0 <= component_means[0]:
            coefficients[0] = 1.0
            return coefficients
        if x0 >= component_means[-1]:
            coefficients[-1] = 1.0
            return coefficients

        right = int(np.searchsorted(component_means, x0, side="right"))
        left = right - 1
        mu_left = component_means[left]
        mu_right = component_means[right]
        weight_right = (x0 - mu_left) / (mu_right - mu_left)
        coefficients[left] = 1.0 - weight_right
        coefficients[right] = weight_right
        return coefficients

    @staticmethod
    def mixture_density(coefficients, basis):
        return np.asarray(coefficients, dtype=float) @ np.asarray(basis, dtype=float)

    @staticmethod
    def project_density(density, basis, dx):
        """Project p onto a nonnegative, unit-mass Beta mixture."""
        density = np.asarray(density, dtype=float)
        basis = np.asarray(basis, dtype=float)
        scale = np.sqrt(dx)
        matrix = basis.T * scale
        target = density * scale
        coefficients = ActiveSet.simplex_least_squares(matrix, target)
        projected = coefficients @ basis
        return coefficients, projected

    @staticmethod
    def _normalise_density(density, dx):
        density = np.maximum(np.asarray(density, dtype=float), 0.0)
        mass = np.sum(density) * dx
        if not np.isfinite(mass) or mass <= 0.0:
            raise RuntimeError("Fokker-Planck density lost positive finite mass.")
        return density / mass

    @staticmethod
    def _flux_rhs(x, density, drift_values, diffusion_values):
        dx = float(x[1] - x[0])
        density = np.asarray(density, dtype=float)
        drift_values = np.broadcast_to(np.asarray(drift_values, dtype=float), density.shape)
        diffusion_values = np.broadcast_to(np.asarray(diffusion_values, dtype=float), density.shape)
        diffusion_squared = np.square(diffusion_values)

        face_drift = 0.5 * (drift_values[:-1] + drift_values[1:])
        upwind_density = np.where(face_drift >= 0.0, density[:-1], density[1:])
        advective_flux = face_drift * upwind_density
        diffusive_quantity = diffusion_squared * density
        diffusive_flux = -0.5 * np.diff(diffusive_quantity) / dx

        flux = np.zeros(len(x) + 1, dtype=float)
        flux[1:-1] = advective_flux + diffusive_flux
        return -np.diff(flux) / dx

    @staticmethod
    def _stable_step(dx, drift_values, diffusion_values, cfl):
        max_drift = float(np.max(np.abs(drift_values)))
        max_diffusion_squared = float(np.max(np.square(diffusion_values)))
        advective_limit = np.inf if max_drift == 0.0 else dx / max_drift
        diffusive_limit = np.inf if max_diffusion_squared == 0.0 else dx * dx / max_diffusion_squared
        return cfl * min(advective_limit, diffusive_limit)

    def solve(self, t, init_cond, drift, diffusion, degree=12, x_steps=201, cfl=0.45):
        """Solve the one-dimensional Fokker-Planck equation."""
        t = np.asarray(t, dtype=float)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("t must be a one-dimensional grid with at least two points.")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("t must be strictly increasing.")
        if int(x_steps) < 3:
            raise ValueError("x_steps must be at least three.")
        if not 0.0 < float(cfl) < 1.0:
            raise ValueError("cfl must lie strictly between zero and one.")

        x0 = float(np.asarray(init_cond).reshape(-1)[0])
        degree = int(degree)
        x_steps = int(x_steps)
        dx = 1.0 / x_steps
        x = (np.arange(x_steps, dtype=float) + 0.5) * dx

        basis = self._normalise_basis_discretely(self.beta_basis(x, degree), dx)
        coefficients = self.initial_coefficients(x0, degree)
        density = self._normalise_density(self.mixture_density(coefficients, basis), dx)

        density_history = np.empty((len(t), x_steps), dtype=float)
        coefficient_history = np.empty((len(t), degree + 1), dtype=float)
        density_history[0] = density
        coefficient_history[0] = coefficients

        for n in range(len(t) - 1):
            current_time = float(t[n])
            target_time = float(t[n + 1])

            while current_time < target_time:
                drift_values = np.broadcast_to(np.asarray(drift(current_time, x), dtype=float), x.shape)
                diffusion_values = np.broadcast_to(np.asarray(diffusion(current_time, x), dtype=float), x.shape)
                stable_h = self._stable_step(dx, drift_values, diffusion_values, cfl)
                remaining = target_time - current_time
                h = remaining if not np.isfinite(stable_h) else min(remaining, stable_h)
                if h <= 0.0:
                    raise RuntimeError("Fokker-Planck method produced a non-positive substep.")

                density = density + h * self._flux_rhs(x, density, drift_values, diffusion_values)
                density = self._normalise_density(density, dx)
                current_time += h

            coefficients, density = self.project_density(density, basis, dx)
            coefficient_history[n + 1] = coefficients
            density_history[n + 1] = density

        return x, density_history, coefficient_history
