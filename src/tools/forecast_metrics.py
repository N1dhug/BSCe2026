"""Distribution-based predictive intervals, empirical coverage and error scores."""

import numpy as np

from src.validation.tools.forecast_metrics import forecast_metrics_validation as checks


def weighted_quantiles(values, weights, probabilities):
    """Generalized inverse CDF for discrete weighted outcomes (no KDE smoothing)."""
    values, weights = checks.distribution(values, weights)
    probabilities = np.asarray(probabilities, dtype=float)
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Quantile probabilities must lie in [0, 1].")
    order = np.argsort(values)
    sorted_values, cdf = values[order], np.cumsum(weights[order])
    return sorted_values[np.minimum(np.searchsorted(cdf, probabilities, side="left"), len(values) - 1)]


def predictive_intervals(values, weights):
    """Central equal-tail 68% and 95% intervals for each timestamp."""
    if len(values) != len(weights): raise ValueError("Distributions require one support/weight pair per timestamp.")
    bounds = np.array([
        weighted_quantiles(support, mass, [0.025, 0.16, 0.84, 0.975])
        for support, mass in zip(values, weights)
    ])
    return {68: (bounds[:, 1], bounds[:, 2]), 95: (bounds[:, 0], bounds[:, 3])}


def forecast_metrics(actual, predicted, intervals=None):
    """Scores are on the supplied common demand scale; bias = prediction - actual."""
    actual, predicted, intervals = checks.forecast(actual, predicted, intervals)
    error = predicted - actual
    metrics = {
        "n": int(error.size), "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)),
    }
    if intervals is not None:
        for level, (lower, upper) in intervals.items():
            inside = (actual >= lower - 1e-12) & (actual <= upper + 1e-12)
            metrics[f"coverage_{level}_pct"] = float(100.0 * np.mean(inside))
            metrics[f"coverage_{level}_count"] = int(np.sum(inside))
            metrics[f"width_{level}_mean"] = float(np.mean(upper - lower))
            metrics[f"coverage_{level}_gap_pp"] = metrics[f"coverage_{level}_pct"] - level
    return metrics
