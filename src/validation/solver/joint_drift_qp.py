import numpy as np


class joint_drift_qp_validation:
    @staticmethod
    def inputs(rate, price, dt, rate_features, price_features, mask, f_bounds, ridge, eps):
        R = np.asarray(rate, dtype=float).reshape(-1)
        U = np.asarray(price, dtype=float).reshape(-1)
        G = np.asarray(rate_features, dtype=float)
        H = np.asarray(price_features, dtype=float)
        dt = float(dt)
        if R.size < 2 or U.shape != R.shape or not np.isfinite(np.r_[R, U]).all():
            raise ValueError("Rate and price must be finite vectors of equal length >= 2.")
        if np.any(U < -1.0) or np.any(U > 1.0): raise ValueError("Price must lie in [-1,1].")
        if not np.isfinite(dt) or dt <= 0.0: raise ValueError("dt must be positive and finite.")
        if G.ndim != 2 or H.ndim != 2 or G.shape[0] != R.size-1 or H.shape[0] != R.size-1:
            raise ValueError("Feature matrices require one row per rate transition.")
        if not np.isfinite(G).all() or not np.isfinite(H).all(): raise ValueError("Features must be finite.")
        if mask is None: mask = np.ones(R.size-1, dtype=bool)
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size != R.size-1 or not mask.any(): raise ValueError("Transition mask must select at least one transition.")
        lo, hi = map(float, f_bounds)
        if not np.isfinite([lo, hi]).all() or not -1.0 <= lo < hi <= 1.0:
            raise ValueError("Preferred F0 bounds must be nonempty within [-1,1].")
        if not np.isfinite(ridge) or ridge < 0.0 or not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("ridge must be nonnegative and eps positive.")
        C = np.r_[0.0, np.cumsum(dt*R[:-1])]
        return R, U, C, G, H, mask, lo, hi
