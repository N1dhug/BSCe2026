import numpy as np

class ActiveSetValidation:
    """Input validation for generic convex quadratic programs."""

    @staticmethod
    def _tolerance(tol):
        tol_array = np.asarray(tol, dtype=float)
        if tol_array.size != 1:
            raise ValueError("tol must be a scalar.")
        tol = float(tol_array.reshape(-1)[0])
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be finite and strictly positive.")
        return tol

    @staticmethod
    def _max_iter(max_iter, n_variables):
        if max_iter is None:
            return max(100, 20 * n_variables * n_variables)

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
        return max_iter_int

    @staticmethod
    def _matrix(matrix, n_variables, name):
        if matrix is None:
            return np.empty((0, n_variables), dtype=float)

        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != n_variables:
            raise ValueError(f"{name} must be a two-dimensional array with {n_variables} columns.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must contain finite values.")
        return matrix

    @staticmethod
    def _rhs(rhs, n_rows, name):
        if n_rows == 0:
            if rhs is None:
                return np.empty(0, dtype=float)
            rhs = np.asarray(rhs, dtype=float).reshape(-1)
            if rhs.size != 0:
                raise ValueError(f"{name} must be empty when its constraint matrix is absent.")
            return rhs

        if rhs is None:
            raise ValueError(f"{name} is required when its constraint matrix is provided.")
        rhs = np.asarray(rhs, dtype=float).reshape(-1)
        if rhs.size != n_rows:
            raise ValueError(f"{name} length must match its constraint row count.")
        if not np.all(np.isfinite(rhs)):
            raise ValueError(f"{name} must contain finite values.")
        return rhs

    @staticmethod
    def _bound(bound, n_variables, default, name):
        if bound is None:
            return np.full(n_variables, default, dtype=float)

        values = np.asarray(bound, dtype=float)
        if values.size == 1:
            values = np.full(n_variables, float(values.reshape(-1)[0]), dtype=float)
        else:
            values = values.reshape(-1)
            if values.size != n_variables:
                raise ValueError(f"{name} must be scalar or have length {n_variables}.")

        if np.any(np.isnan(values)):
            raise ValueError(f"{name} must not contain NaN.")
        return values

    @classmethod
    def quadratic_program(cls, hessian, linear, A_ub, b_ub, A_eq, b_eq, lb, ub, x0, tol, max_iter,):
        hessian = np.asarray(hessian, dtype=float)
        if hessian.ndim != 2 or hessian.shape[0] == 0:
            raise ValueError("hessian must be a non-empty square matrix.")
        if hessian.shape[0] != hessian.shape[1]:
            raise ValueError("hessian must be square.")
        if not np.all(np.isfinite(hessian)):
            raise ValueError("hessian must contain finite values.")

        n_variables = hessian.shape[0]
        linear = np.asarray(linear, dtype=float).reshape(-1)
        if linear.size != n_variables:
            raise ValueError("linear length must match the Hessian dimension.")
        if not np.all(np.isfinite(linear)):
            raise ValueError("linear must contain finite values.")

        tol = cls._tolerance(tol)
        symmetry_scale = max(1.0, float(np.max(np.abs(hessian))))
        if not np.allclose(hessian, hessian.T, atol=tol * symmetry_scale, rtol=0.0):
            raise ValueError("hessian must be symmetric.")

        hessian = 0.5 * (hessian + hessian.T)
        eigenvalues = np.linalg.eigvalsh(hessian)
        psd_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        if float(np.min(eigenvalues)) < -tol * psd_scale:
            raise ValueError("hessian must be positive semidefinite for a convex QP.")

        A_ub = cls._matrix(A_ub, n_variables, "A_ub")
        b_ub = cls._rhs(b_ub, A_ub.shape[0], "b_ub")
        A_eq = cls._matrix(A_eq, n_variables, "A_eq")
        b_eq = cls._rhs(b_eq, A_eq.shape[0], "b_eq")

        lb = cls._bound(lb, n_variables, -np.inf, "lb")
        ub = cls._bound(ub, n_variables, np.inf, "ub")
        if np.any(lb > ub):
            raise ValueError("Every lower bound must be <= its upper bound.")

        if x0 is not None:
            x0 = np.asarray(x0, dtype=float).reshape(-1)
            if x0.size != n_variables:
                raise ValueError("x0 length must match the Hessian dimension.")
            if not np.all(np.isfinite(x0)):
                raise ValueError("x0 must contain finite values.")

        max_iter = cls._max_iter(max_iter, n_variables)

        return (hessian, linear, A_ub, b_ub, A_eq, b_eq, lb, ub, x0, tol, max_iter,)
