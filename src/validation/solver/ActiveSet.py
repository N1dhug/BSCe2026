import numpy as np


class ActiveSetValidation:
    """Input validation for active-set constrained least-squares solvers."""

    @staticmethod
    def simplex_least_squares(matrix, target, tol, max_iter):
        matrix = np.asarray(matrix, dtype=float)
        target = np.asarray(target, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional.")
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("matrix must contain at least one row and one coefficient column.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must contain finite values.")

        target = target.reshape(-1)
        if target.size != matrix.shape[0]:
            raise ValueError("matrix row count must match target length.")
        if not np.all(np.isfinite(target)):
            raise ValueError("target must contain finite values.")

        tol_array = np.asarray(tol, dtype=float)
        if tol_array.size != 1:
            raise ValueError("tol must be a scalar.")
        tol = float(tol_array.reshape(-1)[0])
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be finite and strictly positive.")

        n_coefficients = matrix.shape[1]
        if max_iter is None:
            max_iter = max(100, 20 * n_coefficients * n_coefficients)
        else:
            if isinstance(max_iter, (bool, np.bool_)):
                raise TypeError("max_iter must be an integer.")
            try:
                max_iter_int = int(max_iter)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("max_iter must be an integer.") from exc
            if float(max_iter_int) != float(max_iter):
                raise ValueError("max_iter must be an integer.")
            if max_iter_int < 1:
                raise ValueError("max_iter must be at least one.")
            max_iter = max_iter_int

        return matrix, target, tol, max_iter
