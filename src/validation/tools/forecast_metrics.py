"""Validation for predictive intervals and observed-demand scores."""

import numpy as np


class forecast_metrics_validation:
    @staticmethod
    def forecast(actual, predicted, intervals=None):
        observed, mean = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
        if observed.ndim != 1 or observed.size == 0 or observed.shape != mean.shape or not np.all(np.isfinite(observed)) or not np.all(np.isfinite(mean)):
            raise ValueError("Observed and predicted demand must be equal-length finite 1D vectors.")
        if intervals is None: return observed, mean, None
        cleaned = {}
        for level in (68, 95):
            if level not in intervals: raise ValueError(f"Missing {level}% predictive interval.")
            lower, upper = map(lambda v: np.asarray(v, dtype=float), intervals[level])
            if lower.shape != observed.shape or upper.shape != observed.shape or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower > upper):
                raise ValueError(f"Invalid {level}% predictive interval arrays.")
            cleaned[level] = (lower, upper)
        if np.any(cleaned[95][0] > cleaned[68][0] + 1e-10) or np.any(cleaned[68][1] > cleaned[95][1] + 1e-10):
            raise ValueError("95% interval must enclose the 68% interval.")
        return observed, mean, cleaned

    @staticmethod
    def distribution(values, weights):
        values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
        if values.ndim != 1 or values.size == 0 or weights.shape != values.shape or not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
            raise ValueError("Predictive support and weights must be equal-length finite vectors.")
        if np.any(weights < -1e-12) or np.sum(weights) <= 0.0:
            raise ValueError("Predictive weights must be nonnegative with positive sum.")
        return values, np.maximum(weights, 0.0) / np.sum(np.maximum(weights, 0.0))
