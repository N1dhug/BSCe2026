import numpy as np
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
    def _simplex_least_squares_active_set(matrix, target, tol=1e-10, max_iter=None):
        """Solve a simplex-constrained least-squares problem with an active set.

        Solves

            min_c  0.5 * ||matrix @ c - target||_2^2

        subject to

            c >= 0,
            sum(c) = 1.

        The active set contains coefficients currently fixed at zero. For the
        remaining free coefficients, the equality-constrained least-squares
        subproblem is solved from its KKT system. If that solution leaves the
        simplex, a line search moves to the nearest non-negativity boundary.
        Once a feasible face minimizer is reached, the reduced gradients are
        checked and active constraints are released until the KKT conditions
        are satisfied.
        """
        matrix = np.asarray(matrix, dtype=float)
        target = np.asarray(target, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional.")
        if target.ndim != 1 or target.shape[0] != matrix.shape[0]:
            raise ValueError("target must be one-dimensional and match matrix rows.")

        n_coefficients = matrix.shape[1]
        if n_coefficients == 0:
            raise ValueError("matrix must contain at least one coefficient column.")

        if max_iter is None:
            max_iter = max(100, 20 * n_coefficients * n_coefficients)

        # Strictly feasible starting point on the probability simplex.
        coefficients = np.full(n_coefficients, 1.0 / n_coefficients, dtype=float)
        active = np.zeros(n_coefficients, dtype=bool)

        def solve_free_subproblem(active_set):
            free = np.flatnonzero(~active_set)
            if free.size == 0:
                raise RuntimeError("Active-set least squares has no free coefficients.")

            free_matrix = matrix[:, free]
            gram = free_matrix.T @ free_matrix
            rhs = free_matrix.T @ target

            # KKT system for
            #   min 0.5 ||A_F c_F - b||^2
            #   s.t. 1^T c_F = 1.
            kkt = np.block(
                [
                    [gram, np.ones((free.size, 1), dtype=float)],
                    [np.ones((1, free.size), dtype=float), np.zeros((1, 1), dtype=float)],
                ]
            )
            kkt_rhs = np.concatenate((rhs, np.array([1.0], dtype=float)))
            solution, *_ = np.linalg.lstsq(kkt, kkt_rhs, rcond=None)

            trial = np.zeros(n_coefficients, dtype=float)
            trial[free] = solution[:-1]
            return trial

        for _ in range(max_iter):
            trial = solve_free_subproblem(active)
            free = ~active

            # If the minimizer on the current face violates c >= 0, move from
            # the current feasible point toward it only until the first free
            # coefficient reaches zero, then add that constraint to the set.
            if np.any(trial[free] < -tol):
                direction = trial - coefficients
                decreasing = free & (direction < -tol)

                if not np.any(decreasing):
                    raise RuntimeError(
                        "Active-set least squares could not find a feasible boundary step."
                    )

                step_lengths = -coefficients[decreasing] / direction[decreasing]
                step = float(np.min(step_lengths))
                step = min(max(step, 0.0), 1.0)

                coefficients = coefficients + step * direction
                hit_boundary = free & (coefficients <= tol)
                coefficients[hit_boundary] = 0.0

                coefficient_sum = np.sum(coefficients)
                if coefficient_sum <= 0.0 or not np.isfinite(coefficient_sum):
                    raise RuntimeError("Active-set least squares lost simplex feasibility.")
                coefficients /= coefficient_sum

                active |= hit_boundary
                continue

            # The current face minimizer is feasible. Clean tiny numerical
            # negatives, preserve the equality constraint, and test KKT
            # optimality for coefficients fixed at zero.
            coefficients = trial
            coefficients[np.abs(coefficients) <= tol] = 0.0
            coefficients = np.maximum(coefficients, 0.0)
            coefficient_sum = np.sum(coefficients)
            if coefficient_sum <= 0.0 or not np.isfinite(coefficient_sum):
                raise RuntimeError("Active-set least squares produced an invalid simplex point.")
            coefficients /= coefficient_sum

            active = coefficients <= tol
            free_indices = np.flatnonzero(~active)
            if free_indices.size == 0:
                raise RuntimeError("Active-set least squares has no positive coefficient.")

            residual = matrix @ coefficients - target
            gradient = matrix.T @ residual

            # For free variables, stationarity gives grad_i + lambda = 0.
            # Averaging over the free set gives a stable estimate of lambda.
            lagrange_multiplier = -float(np.mean(gradient[free_indices]))
            reduced_gradient = gradient + lagrange_multiplier

            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                return coefficients

            gradient_scale = max(1.0, float(np.max(np.abs(gradient))))
            optimality_tol = tol * gradient_scale
            active_reduced = reduced_gradient[active_indices]

            # For c_j = 0, KKT requires grad_j + lambda >= 0. Release the
            # most violating active constraint and resolve on the larger face.
            worst_position = int(np.argmin(active_reduced))
            if active_reduced[worst_position] >= -optimality_tol:
                return coefficients

            active[active_indices[worst_position]] = False

        raise RuntimeError(
            "Active-set least-squares projection did not converge within max_iter."
        )

    @staticmethod
    def project_density(density, basis, dx):
        """Project a density onto the probability-simplex Beta-mixture basis.

        The coefficients are obtained directly from the constrained problem

            min_c ||B c - p||_2^2
            s.t. c >= 0, sum(c) = 1,

        rather than solving NNLS and normalising the coefficients afterwards.
        """
        density = np.asarray(density, dtype=float)
        basis = np.asarray(basis, dtype=float)

        scale = np.sqrt(dx)
        matrix = basis.T * scale
        target = density * scale
        coefficients = FokkerPlanc._simplex_least_squares_active_set(matrix, target)
        projected = coefficients @ basis

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

    def solve(
        self,
        t,
        init_cond,
        drift,
        diffusion,
        degree=12,
        x_steps=201,
        cfl=0.45,
        max_substeps=100000,
    ):
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
