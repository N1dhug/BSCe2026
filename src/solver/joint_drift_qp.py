import numpy as np

from src.solver.active_set import active_set
from src.validation.solver.joint_drift_qp import joint_drift_qp_validation


class joint_drift_qp:
    """Joint convex drift / initial-flexibility fit with beta_0=theta_F*F0."""

    @staticmethod
    def solve(rate, price, dt, rate_features, price_features, coefficient_constraints,
              *, transition_mask=None, f_bounds=(-1.0, 1.0), ridge=1e-10,
              enforce_path=True, eps=1e-8):
        R, U, C, G, H, mask, lower, upper = joint_drift_qp_validation.inputs(
            rate, price, dt, rate_features, price_features, transition_mask, f_bounds, ridge, eps)
        n_rate, n_price = G.shape[1], H.shape[1]
        X = np.column_stack((np.ones(len(R)-1), C[:-1], G, H))[mask]
        y = (np.diff(R) / dt)[mask]
        if X.shape[0] < 1: raise ValueError("Joint drift fit requires observed transitions.")
        coefficient_A, coefficient_b, coefficient_lb, coefficient_ub = coefficient_constraints
        coefficient_A = np.asarray(coefficient_A, dtype=float)
        coefficient_b = np.asarray(coefficient_b, dtype=float)
        coefficient_lb = np.asarray(coefficient_lb, dtype=float)
        coefficient_ub = np.asarray(coefficient_ub, dtype=float)
        p = 2 + n_rate + n_price
        if coefficient_A.ndim != 2 or coefficient_A.shape[1] != p-1 or coefficient_b.shape != (coefficient_A.shape[0],):
            raise ValueError("Structural constraint dimensions do not match the response features.")
        if coefficient_lb.shape != (p-1,) or coefficient_ub.shape != (p-1,):
            raise ValueError("Structural coefficient bounds have invalid dimensions.")
        rows = [np.column_stack((np.zeros(coefficient_A.shape[0]), coefficient_A))]
        rhs = [coefficient_b]
        if enforce_path:
            for c in C:
                row_lo = np.zeros(p); row_lo[:2] = [-1.0, upper-c]
                row_hi = np.zeros(p); row_hi[:2] = [1.0, c-lower]
                # (upper-C)*beta_F <= beta_0 <= (lower-C)*beta_F
                # lower: -beta_0 + (upper-C)*beta_F <= 0
                # upper: beta_0 + (C-lower)*beta_F <= 0
                rows.append(np.stack((row_lo, row_hi)))
                rhs.append(np.zeros(2))
        A = np.vstack(rows)
        b = np.concatenate(rhs)
        lb = np.r_[-np.inf, coefficient_lb]
        ub = np.r_[np.inf, coefficient_ub]
        scale = max(1, X.shape[0])
        Q = 2.0 * X.T @ X / scale + float(ridge) * np.eye(p)
        q = -2.0 * X.T @ y / scale
        beta = active_set.solve(Q, q, A_ub=A, b_ub=b, lb=lb, ub=ub, tol=1e-9, max_iter=10000)
        if beta[1] >= -eps: raise RuntimeError("Joint fit did not recover a strictly restoring coefficient.")
        f0 = float(beta[0] / beta[1])
        if enforce_path and (f0 < lower-1e-6 or f0 > upper+1e-6):
            raise RuntimeError("Joint fit violated preferred initial-flexibility bounds.")
        return {"beta": beta, "coefficients": beta[1:].copy(), "init_cond": np.array([f0, R[0]]),
                "qp_loss": float(np.mean((X @ beta-y)**2)), "design": X, "target": y,
                "reconstructed_F": f0+C, "observed_R": R.copy(), "transition_mask": mask.copy()}
