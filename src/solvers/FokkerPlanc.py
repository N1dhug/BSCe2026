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

        """For the Beta density, the normalizing constant is
          B(m+1, degree-m+1)^{-1} = (degree+1) * C(degree,m).
        This is evaluated in log-space to preserve numerical stability."""
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

        """Each basis function is a density on [0,1], so the cell mass is
          integral_0^1 l_m(x) dx ≈ sum_j l_m(x_j) * dx."""
        masses = np.sum(basis, axis=1) * dx
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

        """The mean of the m-th Beta basis element is
          E[l_m] = (m+1)/(degree+2),
        so the mixture mean is the convex combination of adjacent basis means."""
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

        n_coefficients = matrix.shape[1]
        if max_iter is None:
            max_iter = max(100, 20 * n_coefficients * n_coefficients)

        """We solve a constrained projection onto the probability simplex,
        i.e. find c in R^n such that c_j >= 0 and sum_j c_j = 1 while
        minimizing the squared residual ||B c - p||_2^2."""
        coefficients = np.full(n_coefficients, 1.0 / n_coefficients, dtype=float)
        active = np.zeros(n_coefficients, dtype=bool)

        def solve_free_subproblem(active_set):
            free = np.flatnonzero(~active_set)
            if free.size == 0:
                raise RuntimeError("Active-set least squares has no free coefficients.")

            free_matrix = matrix[:, free]
            gram = free_matrix.T @ free_matrix
            rhs = free_matrix.T @ target

            """On a given face of the simplex, the KKT system is
              [G  1] [c_F] = [A^T b]
              [1^T 0] [λ  ]   [1   ]
            where G = A_F^T A_F and λ is the Lagrange multiplier for the
            equality constraint sum(c_F) = 1."""
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

            """If the unconstrained minimizer on the active face violates
            non-negativity, we move along the segment from the current point to
            the trial point until some component hits zero; this is the usual
            boundary-hit step of an active-set method."""
            if np.any(trial[free] < -tol):
                direction = trial - coefficients
                decreasing = free & (direction < -tol)

                step_lengths = -coefficients[decreasing] / direction[decreasing]
                step = float(np.min(step_lengths))
                step = min(max(step, 0.0), 1.0)

                coefficients = coefficients + step * direction
                hit_boundary = free & (coefficients <= tol)
                coefficients[hit_boundary] = 0.0

                coefficient_sum = np.sum(coefficients)
                coefficients /= coefficient_sum

                active |= hit_boundary
                continue

            """Once the current face minimizer is feasible, we keep the sum fixed,
            remove tiny numerical noise, and check the KKT conditions for active
            variables c_j = 0."""
            coefficients = trial
            coefficients[np.abs(coefficients) <= tol] = 0.0
            coefficients = np.maximum(coefficients, 0.0)
            coefficient_sum = np.sum(coefficients)
            coefficients /= coefficient_sum

            active = coefficients <= tol
            free_indices = np.flatnonzero(~active)
            residual = matrix @ coefficients - target
            gradient = matrix.T @ residual

            """Stationarity on the free set gives grad_j + λ = 0, while for active
            variables c_j = 0 the KKT condition is grad_j + λ >= 0."""
            lagrange_multiplier = -float(np.mean(gradient[free_indices]))
            reduced_gradient = gradient + lagrange_multiplier

            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                return coefficients

            gradient_scale = max(1.0, float(np.max(np.abs(gradient))))
            optimality_tol = tol * gradient_scale
            active_reduced = reduced_gradient[active_indices]

            worst_position = int(np.argmin(active_reduced))
            if active_reduced[worst_position] >= -optimality_tol:
                return coefficients

            active[active_indices[worst_position]] = False


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

        """We solve the discrete least-squares problem with the mass matrix
        weighting sqrt(dx), so that the 2-norm approximates the L^2 norm on the
        grid: ||B c - p||_{L^2}^2 ≈ sum_j (B_j c - p_j)^2 dx."""
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

        """The Fokker-Planck equation is
          ∂_t p = -∂_x (a(t,x) p) + 1/2 ∂_x^2 (σ(t,x)^2 p),
        where a = drift and σ = diffusion.
        The finite-volume form uses fluxes across cell faces:
          p_t = -(F_{j+1/2} - F_{j-1/2}) / dx."""
        face_drift = 0.5 * (drift_values[:-1] + drift_values[1:])
        upwind_density = np.where(face_drift >= 0.0, density[:-1], density[1:])
        advective_flux = face_drift * upwind_density
        diffusive_quantity = diffusion_squared * density

        """The diffusion term is written as a conservative flux of the
        quantity σ^2 p, so the flux is -1/2 * ∂_x (σ^2 p)."""
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

        """For explicit advection-diffusion schemes, the timestep must satisfy
        the CFL condition. The advective bound is roughly dx/|a|, while the
        diffusive bound is dx^2 / |σ|^2, both scaled by cfl in (0,1)."""
        advective_limit = np.inf if max_drift == 0.0 else dx / max_drift
        diffusive_limit = (
            np.inf
            if max_diffusion_squared == 0.0
            else dx * dx / max_diffusion_squared
        )

        return cfl * min(advective_limit, diffusive_limit)

    def solve(self, t, init_cond, drift, diffusion, degree=12, x_steps=201,cfl=0.45):
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
        """

        """The PDE is posed on the interval [0,1].  We discretize it in space on
        a uniform grid x_j = (j+1/2)dx with cell width dx = 1/N and evolve the
        density p_j(t) by an explicit conservative finite-volume update."""
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

        """Advance the Fokker-Planck equation with adaptive explicit substeps."""
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

                rhs = self._flux_rhs(x, density, drift_values, diffusion_values)
                density = density + h * rhs
                density = self._normalise_density(density, dx)

                current_time += h
                substeps += 1

            """After each reporting interval, project the cellwise density back to a
            probability-simplex Beta mixture: p(x) ≈ sum_m c_m l_m(x), c_m >= 0,
            sum_m c_m = 1. This keeps the representation non-negative and
            mass-preserving at each output time."""
            coefficients, density = self.project_density(density, basis, dx)
            coefficient_history[n + 1] = coefficients
            density_history[n + 1] = density

        return x, density_history, coefficient_history
