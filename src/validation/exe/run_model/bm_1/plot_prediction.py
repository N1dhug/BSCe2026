"""Validation and saved-result loading for BM1 prediction heatmaps."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class bm_1_plot_validation:
    """Keep saved-file, numerical and input checks outside the plotting code."""

    REQUIRED = ("I_predicted", "B", "U")
    THETA = ("alpha", "beta", "k", "C", "Xi")

    @staticmethod
    def date(day):
        day = pd.Timestamp(day)
        if pd.isna(day) or day.tz is not None: raise ValueError("day must be a valid timezone-naive date.")
        return day.normalize()

    @classmethod
    def saved(cls, directory, day):
        directory = Path(directory)
        hourly_path, json_path = directory / "hourly.csv", directory / "fit.json"
        if not hourly_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(f"Expected hourly.csv and fit.json in {directory}. Run predict_daily with --save first.")
        hourly = pd.read_csv(hourly_path, index_col=0, parse_dates=True)
        with json_path.open(encoding="utf-8") as handle: metadata = json.load(handle)
        index = pd.DatetimeIndex(hourly.index)
        expected = pd.date_range(cls.date(day), periods=24, freq="h")
        if len(hourly) != 24 or not index.equals(expected): raise ValueError("Saved hourly.csv must contain 24 consecutive hourly timestamps for the requested day.")
        missing = set(cls.REQUIRED) - set(hourly.columns)
        if missing: raise ValueError("Missing forecast columns: " + ", ".join(sorted(missing)))
        for name in (*cls.REQUIRED, "I_observed"):
            if name not in hourly: continue
            values = pd.to_numeric(hourly[name], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(values)) or np.any((values < -1e-8) | (values > 1.0 + 1e-8)):
                raise ValueError(f"Saved {name} must contain 24 finite normalized values in [0, 1].")
            hourly[name] = values
        if not isinstance(metadata, dict) or "theta" not in metadata: raise ValueError("Saved fit.json must contain a theta dictionary.")
        theta = metadata["theta"]
        if not isinstance(theta, dict) or any(name not in theta for name in cls.THETA): raise ValueError("fit.json has incomplete BM1 theta parameters.")
        theta = {name: float(theta[name]) for name in cls.THETA}
        if not all(np.isfinite(list(theta.values()))) or min(theta.values()) <= 0.0 or theta["Xi"] > 1.0:
            raise ValueError("Saved theta parameters must be physically admissible and finite.")
        states = np.asarray(metadata.get("state_X0"), dtype=float)
        if states.shape != (5,) or not np.all(np.isfinite(states)) or np.any((states < 0.0) | (states > 1.0)):
            raise ValueError("Saved fit.json must contain five valid initial-state values.")
        weights = np.asarray(metadata.get("initial_weights", np.full(5, 0.2)), dtype=float)
        if weights.shape != (5,) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Saved initial-state weights must be nonnegative and sum to one.")
        sigma_x = float(metadata.get("sigma_x", 0.0))
        if not np.isfinite(sigma_x) or sigma_x < 0.0: raise ValueError("Saved sigma_x must be finite and nonnegative.")
        if metadata.get("target_day") and cls.date(metadata["target_day"]) != cls.date(day): raise ValueError("Saved target_day differs from the requested day.")
        return hourly, {**metadata, "theta": theta, "state_X0": states, "initial_weights": weights, "sigma_x": sigma_x}

    @staticmethod
    def settings(bins, bandwidth, x_steps, fp_max_step, substeps):
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 20: raise ValueError("bins must be an integer >= 20.")
        if isinstance(x_steps, bool) or not isinstance(x_steps, int) or x_steps < 5: raise ValueError("x_steps must be an integer >= 5.")
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1: raise ValueError("substeps must be a positive integer.")
        bandwidth, fp_max_step = float(bandwidth), float(fp_max_step)
        if not np.isfinite(bandwidth) or bandwidth <= 0.0: raise ValueError("bandwidth must be positive and finite.")
        if not np.isfinite(fp_max_step) or fp_max_step <= 0.0: raise ValueError("fp_max_step must be positive and finite.")
        return bins, bandwidth, x_steps, fp_max_step, substeps

    @staticmethod
    def actual_from_saved_or_import(hourly, metadata, tag, fallback):
        if "I_observed" in hourly: return hourly["I_observed"].to_numpy(dtype=float), "saved I_observed"
        if not fallback:
            raise ValueError("hourly.csv lacks I_observed. Enable --import-actual or rerun predict_daily with --score --save.")
        reference = metadata.get("normalization", {}).get("bounds", {}).get("I")
        if not isinstance(reference, dict) or not {"historical_min", "historical_max"}.issubset(reference):
            raise ValueError("Saved normalization lacks I calibration; rerun predict_daily with --score --save.")
        lo, hi = float(reference["historical_min"]), float(reference["historical_max"])
        if not np.isfinite(lo + hi) or hi <= lo: raise ValueError("Invalid saved demand normalization bounds.")
        from src.exe.run_model.wm_3.fit_weekly import aligned_data
        source, _ = aligned_data(tag, price_domain=(0.0, 1.0), price_year=2022)
        observed = source.reindex(hourly.index)["I"].to_numpy(dtype=float)
        if not np.all(np.isfinite(observed)): raise ValueError("Unable to find complete actual demand for the requested date.")
        return np.clip((observed - lo) / (hi - lo), 0.0, 1.0), "historical input.xlsx (saved normalization)"
