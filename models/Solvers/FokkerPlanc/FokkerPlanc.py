import numpy as np
from scipy.optimize import nnls
from scipy.special import gammaln


class FokkerPlanc:
    """Generic one-dimensional Fokker-Planck solver on x in [0, 1].

    The probability density is represented at each reported time by a
    non-negative mixture of normalized Beta basis densities

        p(t, x) ~= sum_m c_m(t) l_m(x),

    where c_m >= 0 and sum_m c_m = 1.

    Between reported times the Fokker-Planck equation is advanced with a
    conservative finite-volume flux discretization and adaptive explicit
    substeps. The resulting density is then projected back onto the Beta
    mixture basis.
    """

    @staticmethod
    def beta_basis(x, degree):
        """Return normalized Beta basis densities l_0, ..., l_degree.

        l_m(x) = (degree + 1) * C(degree, m)
                 * x**m * (1 - x)**(degree - m)

        which is the Beta(m + 1, degree - m + 1) density.
        """
        x = np.asarray(x, dtype=float)
        degree = int(degree)
        m = np.arange(degree + 1, dtype=float)[:, None]
        x_row = x[None, :]
        log_coefficient = (np.log(degree + 1.0)+ gammaln(degree + 1.0)- gammaln(m + 1.0) - gammaln(degree - m + 1.0))
        log_basis = (log_coefficient+ m * np.log(x_row) + (degree - m) * np.log1p(-x_row))
        return np.exp(log_basis)

    @staticmethod
    def _normalise_basis_discretely(basis, dx):
        basis = np.asarray(basis, dtype=float)
        masses = np.sum(basis, axis=1) * dx
        if np.any(masses <= 0.0):
            raise ValueError("Every Beta basis function must have positive discrete mass.")
        return basis / masses[:, None]

    @staticmethod
    def initial_coefficients(init_cond, degree):
        """Build a probability-preserving Beta-mixture approximation of X(0).

        Adjacent Beta components are blended so their mixture mean follows the
        supplied initial state whenever it lies between the available basis
        means. Near x=0 or x=1 the nearest endpoint component is used.
        """
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
        coefficients = np.asarray(coefficients, dtype=float)
        basis = np.asarray(basis, dtype=float)
        return coefficients @ basis

    @staticmethod
    def project_density(density, basis, dx):
        """Project a density onto the non-negative Beta-mixture basis."""
        density = np.asarray(density, dtype=float)
        basis = np.asarray(basis, dtype=float)
        scale = np.sqrt(dx)
        coefficients, _ = nnls(basis.T * scale, density * scale)
        coefficient_mass = np.sum(coefficients)
        coefficients /= coefficient_mass
        projected = coefficients @ basis
        projected_mass = np.sum(projected) * dx
        projected /= projected_mass

        return coefficients, projected

    @staticmethod
    def _normalise_density(density, dx):
        density = np.maximum(np.asarray(density, dtype=float), 0.0)
        mass = np.sum(density) * dx
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
        flux[0] = 0.0
        flux[-1] = 0.0

        return -np.diff(flux) / dx

    @staticmethod
    def _stable_step(dx, drift_values, diffusion_values, cfl):
        max_drift = float(np.max(np.abs(drift_values)))
        max_diffusion_squared = float(np.max(np.square(diffusion_values)))

        advective_limit = np.inf if max_drift == 0.0 else dx / max_drift
        diffusive_limit = (
            np.inf
            if max_diffusion_squared == 0.0
            else dx * dx / max_diffusion_squared
        )

        return cfl * min(advective_limit, diffusive_limit)

    def solve(self,t,init_cond,drift,diffusion,degree=12,x_steps=201,cfl=0.45,max_substeps=100000,):
        """Solve the one-dimensional Fokker-Planck equation.
        Parameters
        ----------
        t : array-like
            Output time grid.
        init_cond : float or one-element array-like
            Initial state in [0, 1].
        drift : callable
            drift(t, x) for dX = drift dt + diffusion dW.
        diffusion : callable
            diffusion(t, x) for dX = drift dt + diffusion dW.
        degree : int
            Beta-series degree M. The mixture contains M + 1 components.
        x_steps : int
            Number of finite-volume cells on [0, 1].
        cfl : float
            Explicit stability factor in (0, 1).
        max_substeps : int
            Safety limit for internal substeps per reported time interval.
        """
        x0 = init_cond[0]
        degree = degree
        x_steps = x_steps

        dx = 1.0 / x_steps
        x = (np.arange(x_steps, dtype=float) + 0.5) * dx

        basis = self.beta_basis(x, degree)
        basis = self._normalise_basis_discretely(basis, dx)

        coefficients = self.initial_coefficients(x0, degree)
        density = self.mixture_density(coefficients, basis)
        density = self._normalise_density(density, dx)

        density_history = np.empty((len(t), x_steps), dtype=float)
        coefficient_history = np.empty((len(t), degree + 1), dtype=float)
        density_history[0] = density
        coefficient_history[0] = coefficients

        for n in range(len(t) - 1):
            current_time = float(t[n])
            target_time = float(t[n + 1])
            substeps = 0

            while current_time < target_time:
                drift_values = np.broadcast_to(
                    np.asarray(drift(current_time, x), dtype=float), x.shape
                )
                diffusion_values = np.broadcast_to(
                    np.asarray(diffusion(current_time, x), dtype=float), x.shape
                )

                stable_h = self._stable_step(dx, drift_values, diffusion_values, cfl)
                remaining = target_time - current_time
                h = remaining if not np.isfinite(stable_h) else min(remaining, stable_h)

                if h <= 0.0 or not np.isfinite(h):
                    raise RuntimeError("Could not determine a valid Fokker-Planck time step.")

                rhs = self._flux_rhs(x, density, drift_values, diffusion_values)
                density = density + h * rhs
                density = self._normalise_density(density, dx)

                current_time += h
                substeps += 1
                if substeps > max_substeps:
                    raise RuntimeError(
                        "Fokker-Planck solver exceeded max_substeps. "
                        "Reduce x_steps, increase the model time resolution, or inspect the model parameters."
                    )

            coefficients, density = self.project_density(density, basis, dx)
            coefficient_history[n + 1] = coefficients
            density_history[n + 1] = density

        return x, density_history, coefficient_history
