"""Validation and restart-file handling for the all-days BM1/WM4 comparison."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.validation.exe.run_model.compare_model_predictions import model_comparison_validation


class model_comparison_all_days_validation:
    """Keep CLI, date-range, saved-result, and output checks out of the executable."""

    REQUIRED = ("I_actual", "BM1", "WM4_MPC", "BM1_lower_68", "BM1_upper_68",
        "BM1_lower_95", "BM1_upper_95", "WM4_lower_68", "WM4_upper_68",
        "WM4_lower_95", "WM4_upper_95")

    @staticmethod
    def dates(start, end):
        start, end = model_comparison_validation.day(start), model_comparison_validation.day(end)
        if end < start: raise ValueError("end-day must be on or after start-day.")
        return start, end

    @staticmethod
    def settings(steps_per_hour, max_state_iter, state_tol, tag, sigma_x, max_iterations, max_nfev, state_maxiter, substeps, x_steps, fp_max_step):
        model_comparison_validation.configuration(steps_per_hour, max_state_iter, state_tol, tag)
        for label, number in (("max_iterations", max_iterations), ("max_nfev", max_nfev),
            ("state_maxiter", state_maxiter), ("substeps", substeps), ("x_steps", x_steps)):
            if isinstance(number, bool) or not isinstance(number, int) or number < (5 if label == "x_steps" else 1):
                raise ValueError(f"{label} must be a positive integer (x_steps >= 5).")
        for label, number in (("sigma_x", sigma_x), ("fp_max_step", fp_max_step)):
            if not np.isfinite(number) or number < (0.0 if label == "sigma_x" else 1e-12):
                raise ValueError(f"{label} must be finite and {'nonnegative' if label == 'sigma_x' else 'positive'}.")

    @classmethod
    def hourly(cls, frame, day):
        day = model_comparison_validation.day(day)
        if not isinstance(frame, pd.DataFrame) or frame.shape[0] != 24: raise ValueError("A comparison day must contain exactly 24 hourly rows.")
        expected = pd.date_range(day, periods=24, freq="h")
        if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.equals(expected):
            raise ValueError(f"Comparison for {day:%Y-%m-%d} must have 24 ordered hourly timestamps.")
        missing = set(cls.REQUIRED) - set(frame.columns)
        if missing: raise ValueError("Comparison is missing columns: " + ", ".join(sorted(missing)))
        for column in cls.REQUIRED:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(values)): raise ValueError(f"Comparison column {column} has missing/nonfinite values.")
        for label in ("BM1", "WM4"):
            for level in (68, 95):
                if np.any(frame[f"{label}_lower_{level}"] > frame[f"{label}_upper_{level}"]):
                    raise ValueError(f"{label} has inverted {level}% interval.")
        return frame

    @classmethod
    def cached(cls, directory, day, settings):
        directory = Path(directory)
        hourly_path, info_path = directory / "predictions_hourly.csv", directory / "comparison.json"
        if not (hourly_path.is_file() and info_path.is_file()): return None
        with info_path.open(encoding="utf-8") as handle: saved = json.load(handle)
        if saved.get("settings") != settings or saved.get("day") != str(day.date()): return None
        hourly = pd.read_csv(hourly_path, index_col=0, parse_dates=True)
        return cls.hourly(hourly, day)
