import numpy as np


class input_builder_validation:
    """Validation rules for executable input builders."""

    DISTRIBUTIONS = {"gaussian", "normal", "uniform", "beta", "beta_mixture", "gaussian_mixture", "normal_mixture"}

    @staticmethod
    def price_intervals(price_data):
        try: interval_edges, interval_prices = price_data
        except (TypeError, ValueError) as exc: raise ValueError("price_data must contain (interval_edges, interval_prices).") from exc

        interval_edges, interval_prices = np.asarray(interval_edges, dtype=float), np.asarray(interval_prices, dtype=float)
        if interval_edges.ndim != 1 or interval_edges.size < 2: raise ValueError("interval_edges must be a one-dimensional array with at least two values.")
        if not np.all(np.isfinite(interval_edges)): raise ValueError("interval_edges must contain finite values.")
        if np.any(np.diff(interval_edges) <= 0.0): raise ValueError("interval_edges must be strictly increasing.")
        if interval_prices.ndim != 1 or interval_prices.size == 0: raise ValueError("interval_prices must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(interval_prices)): raise ValueError("interval_prices must contain finite values.")
        if np.any(interval_prices < 0.0) or np.any(interval_prices > 1.0): raise ValueError("interval_prices must lie in [0, 1].")
        if interval_edges.size != interval_prices.size + 1: raise ValueError("price_intervals requires N + 1 interval_edges for N interval_prices.")
        return interval_edges, interval_prices

    @staticmethod
    def price_query(t):
        query = np.asarray(t, dtype=float)
        if not np.all(np.isfinite(query)): raise ValueError("Price evaluation times must be finite.")
        return query

    @classmethod
    def distribution(cls, distribution):
        kind = str(distribution).strip().lower().replace("-", "_")
        if kind not in cls.DISTRIBUTIONS: raise ValueError("Unsupported initial density. Choose one of: gaussian, uniform, beta, beta_mixture, gaussian_mixture.")
        return kind

    @staticmethod
    def pair(values, name):
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size != 2 or not np.all(np.isfinite(array)): raise ValueError(f"{name} must contain exactly two finite values.")
        return float(array[0]), float(array[1])

    @classmethod
    def positive_pair(cls, values, name):
        pair = cls.pair(values, name)
        if pair[0] <= 0.0 or pair[1] <= 0.0: raise ValueError(f"{name} must contain strictly positive values.")
        return pair

    @classmethod
    def ordered_pair(cls, values, name):
        pair = cls.pair(values, name)
        if pair[0] >= pair[1]: raise ValueError(f"{name} must satisfy lower < upper.")
        return pair

    @staticmethod
    def mixture_weights(weights, n_components):
        if weights is None: return np.full(n_components, 1.0 / n_components)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size != n_components or not np.all(np.isfinite(weights)) or np.any(weights < 0.0): raise ValueError("weights must be finite, non-negative, and match the component count.")
        total = float(np.sum(weights))
        if total <= 0.0: raise ValueError("weights must have positive total mass.")
        return weights / total

    @classmethod
    def beta_mixture(cls, v_range, r_range, v_shapes, r_shapes, weights):
        v_range, r_range = cls.ordered_pair(v_range, "v_range"), cls.ordered_pair(r_range, "r_range")
        v_shapes, r_shapes = np.asarray(v_shapes, dtype=float), np.asarray(r_shapes, dtype=float)
        if v_shapes.ndim != 2 or v_shapes.shape[1] != 2 or v_shapes.shape[0] == 0: raise ValueError("v_shapes must have shape (n_components, 2).")
        if r_shapes.shape != v_shapes.shape: raise ValueError("r_shapes must match v_shapes.")
        if not np.all(np.isfinite(v_shapes)) or not np.all(np.isfinite(r_shapes)) or np.any(v_shapes <= 0.0) or np.any(r_shapes <= 0.0): raise ValueError("Beta mixture shapes must contain positive finite values.")
        return v_range, r_range, v_shapes, r_shapes, cls.mixture_weights(weights, v_shapes.shape[0])

    @classmethod
    def gaussian_mixture(cls, means, stds, weights):
        means, stds = np.asarray(means, dtype=float), np.asarray(stds, dtype=float)
        if means.ndim != 2 or means.shape[1] != 2 or means.shape[0] == 0: raise ValueError("means must have shape (n_components, 2).")
        if stds.shape != means.shape or not np.all(np.isfinite(stds)) or np.any(stds <= 0.0): raise ValueError("stds must match means and contain positive finite values.")
        if not np.all(np.isfinite(means)): raise ValueError("means must contain finite values.")
        return means, stds, cls.mixture_weights(weights, means.shape[0])
