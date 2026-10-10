"""Validation twin for the BM1 five-weekday forecasting executable."""

import numpy as np
import pandas as pd

from src.validation.params.bm_1 import bm_1_params_validation


class bm_1_prediction_validation:
    """Validate historical-day coverage, forecasting inputs, and output distributions."""

    @staticmethod
    def date(day):
        try: start = pd.Timestamp(day).normalize()
        except (TypeError, ValueError) as exc: raise ValueError("day must be a valid calendar date.") from exc
        if pd.isna(start) or start.tz is not None: raise ValueError("day must be a timezone-naive calendar date.")
        return start

    @staticmethod
    def options(X0, sigma_x, x_steps, fp_max_step, *, tag, max_iterations, min_improvement):
        X0, sigma_x = bm_1_params_validation.scalar(X0, "X0"), bm_1_params_validation.scalar(sigma_x, "sigma_x")
        if not 0.0 <= X0 <= 1.0: raise ValueError("X0 must lie in [0, 1].")
        if sigma_x < 0.0: raise ValueError("sigma_x must be nonnegative.")
        if isinstance(x_steps, bool) or not isinstance(x_steps, int) or x_steps < 5: raise ValueError("x_steps must be an integer >= 5.")
        fp_max_step = bm_1_params_validation.scalar(fp_max_step, "fp_max_step")
        if fp_max_step <= 0.0: raise ValueError("fp_max_step must be positive.")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1: raise ValueError("max_iterations must be a positive integer.")
        min_improvement = bm_1_params_validation.scalar(min_improvement, "min_improvement")
        if min_improvement < 0.0: raise ValueError("min_improvement must be nonnegative.")
        if not isinstance(tag, str) or not tag.strip(): raise ValueError("tag must be a nonempty measurement tag.")
        return X0, sigma_x, x_steps, fp_max_step

    @staticmethod
    def frame(frame):
        if not isinstance(frame, pd.DataFrame): raise TypeError("frame must be a pandas DataFrame.")
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None: raise ValueError("frame must have a timezone-naive DatetimeIndex.")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing: raise ValueError("frame index must be sorted and unique.")
        missing = set(("I", "U", "Temp")) - set(frame.columns)
        if missing: raise ValueError("frame is missing columns: " + ", ".join(sorted(missing)))
        return frame

    @classmethod
    def day_signals(cls, frame, day, baseline_model, *, require_demand):
        """Use only known U/temperature for a held-out target day."""
        day = cls.date(day)
        index = pd.date_range(day, periods=24, freq="h")
        columns = ("I", "U", "Temp") if require_demand else ("U", "Temp")
        hourly = frame.reindex(index)[list(columns)]
        if hourly.isna().any().any(): raise ValueError(f"Missing {', '.join(columns)} data for {day:%Y-%m-%d}; day must contain 24 complete hours.")
        U, Temp = hourly["U"].to_numpy(dtype=float), hourly["Temp"].to_numpy(dtype=float)
        baseline = baseline_model.predict(index, Temp)
        if not np.all(np.isfinite(baseline)): raise ValueError("Baseline prediction contains nonfinite values.")
        if np.any((U < 0.0) | (U > 1.0)): raise ValueError("BM1 requires normalized prices U in [0, 1].")
        data = {"Time": np.arange(25, dtype=float), "U": np.r_[U, U[-1]], "B": np.r_[baseline, baseline[-1]]}
        if require_demand:
            I = hourly["I"].to_numpy(dtype=float)
            if np.any((I < 0.0) | (I > 1.0)): raise ValueError("BM1 requires normalized consumption I in [0, 1].")
            data["I"] = I
        return data, index

    @staticmethod
    def initial_distribution(values, x_steps):
        values = np.asarray(values, dtype=float)
        if values.shape != (5,) or not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)): raise ValueError("Exactly five fitted initial states in [0, 1] are required.")
        mass = np.zeros(x_steps)
        np.add.at(mass, np.minimum((values * x_steps).astype(int), x_steps - 1), 1.0 / len(values))
        return mass

    @staticmethod
    def metrics(actual, forecast):
        actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
        if actual.shape != forecast.shape or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(forecast)): raise ValueError("Observed/predicted demand must have equal finite shapes.")
        error = forecast - actual
        return {"mse": float(np.mean(error ** 2)), "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error))}
