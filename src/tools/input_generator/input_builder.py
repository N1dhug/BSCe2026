import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.special import betaln
from src.validation.tools.input_generator.input_builder import input_builder_validation


class input_builder:
    """Build executable input functions from discrete or parametric data."""

    @staticmethod
    def energy(energy_data):
        t_e, e = energy_data
        return interp1d(t_e, e, kind="linear", bounds_error=False, fill_value="extrapolate")

    @staticmethod
    def price(price_data):
        t_p, p = price_data
        return interp1d(t_p, p, kind="previous", bounds_error=False, fill_value="extrapolate")

    @staticmethod
    def price_intervals(price_data):
        """Build a right-continuous price from one value per interval."""
        interval_edges, interval_prices = input_builder_validation.price_intervals(price_data)
        internal_edges, prices = interval_edges[1:-1].copy(), interval_prices.copy()

        def interval_price(t):
            query = input_builder_validation.price_query(t)
            values = prices[np.searchsorted(internal_edges, query, side="right")]
            return float(values) if query.ndim == 0 else values

        return interval_price

    @staticmethod
    def md(poi_x, pr_x, y_interval=(1.0, -1.0)):
        interp_y_vals = np.full(len(pr_x), y_interval[0]) - np.cumsum(pr_x)
        return PchipInterpolator(poi_x, interp_y_vals)

    @staticmethod
    def md_vals(kappa, n):
        poi_x = np.linspace(0.0, 1.0, n)
        F = 0.5 * (1.0 + np.tanh(kappa * (poi_x - 0.5)) / np.tanh(kappa / 2.0))
        F[0], F[-1] = 0.0, 1.0
        pr_x = np.zeros(n)
        pr_x[1:] = 2.0 * np.diff(F)
        pr_x[1:] *= 2.0 / np.sum(pr_x[1:])
        return poi_x, pr_x

    @staticmethod
    def initial_density(distribution, **kwargs):
        """Return a callable two-dimensional initial probability density."""
        kind = input_builder_validation.distribution(distribution)

        if kind in {"gaussian", "normal"}:
            mean = input_builder_validation.pair(kwargs.get("mean"), "mean")
            std = input_builder_validation.positive_pair(kwargs.get("std"), "std")
            norm = 1.0 / (2.0 * np.pi * std[0] * std[1])

            def density(V, R):
                exponent = -0.5 * (np.square((V - mean[0]) / std[0]) + np.square((R - mean[1]) / std[1]))
                return norm * np.exp(exponent)

            return density

        if kind == "uniform":
            v_range = input_builder_validation.ordered_pair(kwargs.get("v_range"), "v_range")
            r_range = input_builder_validation.ordered_pair(kwargs.get("r_range"), "r_range")
            norm = 1.0 / ((v_range[1] - v_range[0]) * (r_range[1] - r_range[0]))

            def density(V, R):
                inside = (V >= v_range[0]) & (V <= v_range[1]) & (R >= r_range[0]) & (R <= r_range[1])
                return norm * inside.astype(float)

            return density

        if kind == "beta":
            v_range = input_builder_validation.ordered_pair(kwargs.get("v_range"), "v_range")
            r_range = input_builder_validation.ordered_pair(kwargs.get("r_range"), "r_range")
            v_shape = input_builder_validation.positive_pair(kwargs.get("v_shape"), "v_shape")
            r_shape = input_builder_validation.positive_pair(kwargs.get("r_shape"), "r_shape")

            def scaled_beta(x, bounds, shape):
                lo, hi = bounds
                z = (x - lo) / (hi - lo)
                inside = (z > 0.0) & (z < 1.0)
                result = np.zeros_like(np.asarray(z, dtype=float))
                if np.any(inside):
                    zi = z[inside]
                    log_pdf = (shape[0] - 1.0) * np.log(zi) + (shape[1] - 1.0) * np.log1p(-zi) - betaln(shape[0], shape[1]) - np.log(hi - lo)
                    result[inside] = np.exp(log_pdf)
                return result

            return lambda V, R: scaled_beta(V, v_range, v_shape) * scaled_beta(R, r_range, r_shape)

        if kind == "beta_mixture":
            v_range, r_range, v_shapes, r_shapes, weights = input_builder_validation.beta_mixture(
                kwargs.get("v_range"), kwargs.get("r_range"), kwargs.get("v_shapes"), kwargs.get("r_shapes"), kwargs.get("weights")
            )

            def scaled_beta(x, bounds, shape):
                lo, hi = bounds
                z = (x - lo) / (hi - lo)
                inside = (z > 0.0) & (z < 1.0)
                result = np.zeros_like(np.asarray(z, dtype=float))
                if np.any(inside):
                    zi = z[inside]
                    log_pdf = (shape[0] - 1.0) * np.log(zi) + (shape[1] - 1.0) * np.log1p(-zi) - betaln(shape[0], shape[1]) - np.log(hi - lo)
                    result[inside] = np.exp(log_pdf)
                return result

            def density(V, R):
                result = np.zeros(np.broadcast(V, R).shape, dtype=float)
                for weight, v_shape, r_shape in zip(weights, v_shapes, r_shapes): result += weight * scaled_beta(V, v_range, v_shape) * scaled_beta(R, r_range, r_shape)
                return result

            return density

        means, stds, weights = input_builder_validation.gaussian_mixture(kwargs.get("means"), kwargs.get("stds"), kwargs.get("weights"))
        norms = 1.0 / (2.0 * np.pi * stds[:, 0] * stds[:, 1])

        def density(V, R):
            result = np.zeros(np.broadcast(V, R).shape, dtype=float)
            for weight, mean, std, norm in zip(weights, means, stds, norms):
                exponent = -0.5 * (np.square((V - mean[0]) / std[0]) + np.square((R - mean[1]) / std[1]))
                result += weight * norm * np.exp(exponent)
            return result

        return density
