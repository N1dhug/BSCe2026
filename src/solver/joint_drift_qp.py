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


    @staticmethod
    def solve_many(rates, prices, dt, coefficient_constraints, *, transition_masks=None,
                   f_bounds=(-1.0, 1.0), ridge=1e-10, enforce_path=True, eps=1e-8):
        """One convex QP: a separate beta_0=theta_F*F0 for every day, shared theta."""
        prepared, lower, upper, dt, ridge, eps = joint_drift_qp_validation.many(
            rates, prices, dt, transition_masks, f_bounds, ridge, eps)
        n = len(prepared)
        A0, b0, lb0, ub0 = (np.asarray(v, dtype=float) for v in coefficient_constraints)
        if A0.ndim != 2 or A0.shape[1] != 3 or b0.shape != (A0.shape[0],):
            raise ValueError("Shared WM4 coefficient constraints must have three columns.")
        if lb0.shape != (3,) or ub0.shape != (3,):
            raise ValueError("Invalid shared WM4 coefficient bounds.")
        designs, targets, constraints, limits, integrations = [], [], [], [], []
        constraints.append(np.column_stack((np.zeros((len(b0), n)), A0)))
        limits.append(b0)
        for i, (R, U, mask) in enumerate(prepared):
            C = np.r_[0.0, np.cumsum(dt * R[:-1])]
            integrations.append(C)
            X = np.zeros((len(R)-1, n+3), dtype=float)
            X[:, i] = 1.0
            X[:, n:] = np.column_stack((C[:-1], U[:-1], R[:-1]))
            designs.append(X[mask])
            targets.append((np.diff(R)/dt)[mask])
            if enforce_path:
                # theta_F < 0; F0=beta0/theta_F, F_k=F0+C_k.
                # Extrema suffice to constrain the entire reconstructed path.
                lo_row, hi_row = np.zeros(n+3), np.zeros(n+3)
                lo_row[i], lo_row[n] = -1.0, upper - float(np.max(C))
                hi_row[i], hi_row[n] = 1.0, float(np.min(C)) - lower
                constraints.append(np.vstack((lo_row, hi_row)))
                limits.append(np.zeros(2))
        X, y = np.vstack(designs), np.concatenate(targets)
        A, b = np.vstack(constraints), np.concatenate(limits)
        lb, ub = np.r_[np.full(n, -np.inf), lb0], np.r_[np.full(n, np.inf), ub0]
        scale = max(1, len(y))
        H = 2.0 * X.T @ X / scale + ridge * np.eye(n+3)
        f = -2.0 * X.T @ y / scale
        solution = np.asarray(active_set.solve(H, f, A_ub=A, b_ub=b, lb=lb, ub=ub,
                                                tol=1e-9, max_iter=10000), dtype=float)
        theta = solution[n:]
        if theta[0] >= -eps:
            raise RuntimeError("Joint multi-day QP failed to recover theta_F<0.")
        F0 = solution[:n] / theta[0]
        if enforce_path and any(np.min(f0+C) < lower-1e-6 or np.max(f0+C) > upper+1e-6
                                for f0, C in zip(F0, integrations)):
            raise RuntimeError("Multi-day QP violated reconstructed flexibility bounds.")
        states = np.column_stack((F0, [R[0] for R, _, _ in prepared]))
        return {"coefficients": theta.copy(), "beta": solution.copy(), "init_conds": states,
                "qp_loss": float(np.mean((X @ solution-y)**2)),
                "transition_count": len(y), "design_rank": int(np.linalg.matrix_rank(X)),
                "reconstructed_F": [f0+C for f0, C in zip(F0, integrations)]}
