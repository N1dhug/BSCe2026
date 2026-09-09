import numpy as np


class ActiveSet:
    """Generic active-set solvers for small convex constrained least-squares problems."""

    @staticmethod
    def simplex_least_squares(matrix, target, tol=1e-10, max_iter=None):
        """Solve min 0.5*||A c-b||^2 subject to c >= 0 and sum(c)=1."""
        matrix = np.asarray(matrix, dtype=float)
        target = np.asarray(target, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional.")
        if target.ndim != 1:
            target = target.reshape(-1)
        if matrix.shape[0] != target.size:
            raise ValueError("matrix row count must match target length.")

        n_coefficients = matrix.shape[1]
        if n_coefficients == 0:
            raise ValueError("matrix must contain at least one coefficient column.")
        if max_iter is None:
            max_iter = max(100, 20 * n_coefficients * n_coefficients)

        coefficients = np.full(n_coefficients, 1.0 / n_coefficients, dtype=float)
        active = np.zeros(n_coefficients, dtype=bool)

        def solve_free_subproblem(active_set):
            free = np.flatnonzero(~active_set)
            if free.size == 0:
                raise RuntimeError("Active-set least squares has no free coefficients.")

            free_matrix = matrix[:, free]
            gram = free_matrix.T @ free_matrix
            rhs = free_matrix.T @ target
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

            if np.any(trial[free] < -tol):
                direction = trial - coefficients
                decreasing = free & (direction < -tol)
                step_lengths = -coefficients[decreasing] / direction[decreasing]
                step = float(np.min(step_lengths))
                step = min(max(step, 0.0), 1.0)

                coefficients = coefficients + step * direction
                hit_boundary = free & (coefficients <= tol)
                coefficients[hit_boundary] = 0.0
                coefficients /= np.sum(coefficients)
                active |= hit_boundary
                continue

            coefficients = trial
            coefficients[np.abs(coefficients) <= tol] = 0.0
            coefficients = np.maximum(coefficients, 0.0)
            coefficients /= np.sum(coefficients)

            active = coefficients <= tol
            free_indices = np.flatnonzero(~active)
            residual = matrix @ coefficients - target
            gradient = matrix.T @ residual
            lagrange_multiplier = -float(np.mean(gradient[free_indices]))
            reduced_gradient = gradient + lagrange_multiplier

            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                return coefficients

            gradient_scale = max(1.0, float(np.max(np.abs(gradient))))
            active_reduced = reduced_gradient[active_indices]
            worst_position = int(np.argmin(active_reduced))
            if active_reduced[worst_position] >= -tol * gradient_scale:
                return coefficients

            active[active_indices[worst_position]] = False

        raise RuntimeError("Active-set simplex least-squares solver did not converge.")
