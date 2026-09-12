import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.special import betaln


class InputBuilder:
    """Build executable input functions from discrete or parametric data."""

    @staticmethod
    def energy(energy_data):
        t_e, e = energy_data
        return interp1d(
            t_e, e, kind="linear", bounds_error=False, fill_value="extrapolate"
        )

    @staticmethod
    def price(price_data):
        t_p, p = price_data
        return interp1d(
            t_p, p, kind="previous", bounds_error=False, fill_value="extrapolate"
        )

    @staticmethod
    def md(poi_x, pr_x, y_interval=(1.0, -1.0)):
        interp_y_vals = np.full(len(pr_x), y_interval[0]) - np.cumsum(pr_x)
        return PchipInterpolator(poi_x, interp_y_vals)

    @staticmethod
    def md_vals(kappa, n):
        poi_x = np.linspace(0.0, 1.0, n)
        F = 0.5 * (
            1.0
            + np.tanh(kappa * (poi_x - 0.5)) / np.tanh(kappa / 2.0)
        )
        F[0] = 0.0
        F[-1] = 1.0

        pr_x = np.zeros(n)
        pr_x[1:] = 2.0 * np.diff(F)
        pr_x[1:] *= 2.0 / np.sum(pr_x[1:])
        pr_x[0] = 0.0
        return poi_x, pr_x

    @staticmethod
    def initial_density(distribution, **kwargs):
        """Return a callable two-dimensional initial probability density.

        Supported distributions are gaussian, uniform, beta, beta_mixture and
        gaussian_mixture. The finite-volume solver performs the final discrete
        normalization on its computational grid.
        """
        kind = str(distribution).strip().lower().replace("-", "_")

        if kind in {"gaussian", "normal"}:
            mean = InputBuilder._pair(kwargs.get("mean"), "mean")
            std = InputBuilder._positive_pair(kwargs.get("std"), "std")
            norm = 1.0 / (2.0 * np.pi * std[0] * std[1])

            def density(V, R):
                exponent = -0.5 * (
                    np.square((V - mean[0]) / std[0])
                    + np.square((R - mean[1]) / std[1])
                )
                return norm * np.exp(exponent)

            return density

        if kind == "uniform":
            v_range = InputBuilder._ordered_pair(kwargs.get("v_range"), "v_range")
            r_range = InputBuilder._ordered_pair(kwargs.get("r_range"), "r_range")
            norm = 1.0 / (
                (v_range[1] - v_range[0]) * (r_range[1] - r_range[0])
            )

            def density(V, R):
                inside = (
                    (V >= v_range[0])
                    & (V <= v_range[1])
                    & (R >= r_range[0])
                    & (R <= r_range[1])
                )
                return norm * inside.astype(float)

            return density

        if kind == "beta":
            v_range = InputBuilder._ordered_pair(kwargs.get("v_range"), "v_range")
            r_range = InputBuilder._ordered_pair(kwargs.get("r_range"), "r_range")
            v_shape = InputBuilder._positive_pair(kwargs.get("v_shape"), "v_shape")
            r_shape = InputBuilder._positive_pair(kwargs.get("r_shape"), "r_shape")

            def scaled_beta(x, bounds, shape):
                lo, hi = bounds
                z = (x - lo) / (hi - lo)
                inside = (z > 0.0) & (z < 1.0)
                result = np.zeros_like(np.asarray(z, dtype=float))
                if np.any(inside):
                    zi = z[inside]
                    log_pdf = (
                        (shape[0] - 1.0) * np.log(zi)
                        + (shape[1] - 1.0) * np.log1p(-zi)
                        - betaln(shape[0], shape[1])
                        - np.log(hi - lo)
                    )
                    result[inside] = np.exp(log_pdf)
                return result

            def density(V, R):
                return scaled_beta(V, v_range, v_shape) * scaled_beta(
                    R, r_range, r_shape
                )

            return density

        if kind == "beta_mixture":
            v_range = InputBuilder._ordered_pair(kwargs.get("v_range"), "v_range")
            r_range = InputBuilder._ordered_pair(kwargs.get("r_range"), "r_range")
            v_shapes = np.asarray(kwargs.get("v_shapes"), dtype=float)
            r_shapes = np.asarray(kwargs.get("r_shapes"), dtype=float)

            if (
                v_shapes.ndim != 2
                or v_shapes.shape[1] != 2
                or v_shapes.shape[0] == 0
            ):
                raise ValueError("v_shapes must have shape (n_components, 2).")
            if r_shapes.shape != v_shapes.shape:
                raise ValueError("r_shapes must match v_shapes.")
            if (
                not np.all(np.isfinite(v_shapes))
                or not np.all(np.isfinite(r_shapes))
                or np.any(v_shapes <= 0.0)
                or np.any(r_shapes <= 0.0)
            ):
                raise ValueError(
                    "Beta mixture shapes must contain positive finite values."
                )

            weights = kwargs.get("weights")
            if weights is None:
                weights = np.full(v_shapes.shape[0], 1.0 / v_shapes.shape[0])
            else:
                weights = np.asarray(weights, dtype=float).reshape(-1)
                if (
                    weights.size != v_shapes.shape[0]
                    or not np.all(np.isfinite(weights))
                    or np.any(weights < 0.0)
                ):
                    raise ValueError(
                        "weights must be finite, non-negative, and match the "
                        "component count."
                    )
                total = float(np.sum(weights))
                if total <= 0.0:
                    raise ValueError("weights must have positive total mass.")
                weights = weights / total

            def scaled_beta(x, bounds, shape):
                lo, hi = bounds
                z = (x - lo) / (hi - lo)
                inside = (z > 0.0) & (z < 1.0)
                result = np.zeros_like(np.asarray(z, dtype=float))
                if np.any(inside):
                    zi = z[inside]
                    log_pdf = (
                        (shape[0] - 1.0) * np.log(zi)
                        + (shape[1] - 1.0) * np.log1p(-zi)
                        - betaln(shape[0], shape[1])
                        - np.log(hi - lo)
                    )
                    result[inside] = np.exp(log_pdf)
                return result

            def density(V, R):
                result = np.zeros(np.broadcast(V, R).shape, dtype=float)
                for weight, v_shape, r_shape in zip(
                    weights, v_shapes, r_shapes
                ):
                    result += weight * scaled_beta(
                        V, v_range, v_shape
                    ) * scaled_beta(R, r_range, r_shape)
                return result

            return density

        if kind in {"gaussian_mixture", "normal_mixture"}:
            means = np.asarray(kwargs.get("means"), dtype=float)
            stds = np.asarray(kwargs.get("stds"), dtype=float)
            if means.ndim != 2 or means.shape[1] != 2 or means.shape[0] == 0:
                raise ValueError("means must have shape (n_components, 2).")
            if (
                stds.shape != means.shape
                or not np.all(np.isfinite(stds))
                or np.any(stds <= 0.0)
            ):
                raise ValueError(
                    "stds must match means and contain positive finite values."
                )
            if not np.all(np.isfinite(means)):
                raise ValueError("means must contain finite values.")

            weights = kwargs.get("weights")
            if weights is None:
                weights = np.full(means.shape[0], 1.0 / means.shape[0])
            else:
                weights = np.asarray(weights, dtype=float).reshape(-1)
                if (
                    weights.size != means.shape[0]
                    or not np.all(np.isfinite(weights))
                    or np.any(weights < 0.0)
                ):
                    raise ValueError(
                        "weights must be finite, non-negative, and match the "
                        "component count."
                    )
                total = float(np.sum(weights))
                if total <= 0.0:
                    raise ValueError("weights must have positive total mass.")
                weights = weights / total

            norms = 1.0 / (2.0 * np.pi * stds[:, 0] * stds[:, 1])

            def density(V, R):
                result = np.zeros(np.broadcast(V, R).shape, dtype=float)
                for weight, mean, std, norm in zip(weights, means, stds, norms):
                    exponent = -0.5 * (
                        np.square((V - mean[0]) / std[0])
                        + np.square((R - mean[1]) / std[1])
                    )
                    result += weight * norm * np.exp(exponent)
                return result

            return density

        raise ValueError(
            "Unsupported initial density. Choose one of: gaussian, uniform, "
            "beta, beta_mixture, gaussian_mixture."
        )

    @staticmethod
    def _pair(values, name):
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size != 2 or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain exactly two finite values.")
        return float(array[0]), float(array[1])

    @staticmethod
    def _positive_pair(values, name):
        pair = InputBuilder._pair(values, name)
        if pair[0] <= 0.0 or pair[1] <= 0.0:
            raise ValueError(f"{name} must contain strictly positive values.")
        return pair

    @staticmethod
    def _ordered_pair(values, name):
        pair = InputBuilder._pair(values, name)
        if pair[0] >= pair[1]:
            raise ValueError(f"{name} must satisfy lower < upper.")
        return pair
