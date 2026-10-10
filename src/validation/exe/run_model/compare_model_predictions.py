"""Validation and common-scaling for a BM1 vs WM4 MPC one-day comparison."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.validation.exe.run_model.bm_1.predict_daily import bm_1_prediction_validation


class model_comparison_validation:
    @staticmethod
    def day(day): return bm_1_prediction_validation.date(day)

    @staticmethod
    def configuration(steps_per_hour, max_state_iter, state_tol, tag):
        if isinstance(steps_per_hour, bool) or not isinstance(steps_per_hour, int) or steps_per_hour < 2:
            raise ValueError("steps_per_hour must be an integer >= 2.")
        if isinstance(max_state_iter, bool) or not isinstance(max_state_iter, int) or max_state_iter < 1:
            raise ValueError("max_state_iter must be a positive integer.")
        if not np.isfinite(state_tol) or state_tol <= 0.0: raise ValueError("state_tol must be finite and positive.")
        if not isinstance(tag, str) or not tag.strip(): raise ValueError("tag must be a nonempty string.")

    @staticmethod
    def common_frame(frame):
        frame = bm_1_prediction_validation.frame(frame)
        if frame["I"].isna().any() or frame["U"].isna().any(): raise ValueError("Missing normalized demand/price values.")
        return frame

    @staticmethod
    def bm1_scale(frame, day):
        """Normalize BM1 input causally while retaining conversion to WM4 demand units."""
        first_training_day = day - pd.Timedelta(weeks=5)
        history = frame.loc[frame.index < first_training_day, ["I", "U"]]
        if history.empty: raise ValueError("No data preceding the oldest same-weekday fit day.")
        normalized = frame.copy()
        scales = {}
        for key in ("I", "U"):
            lo, hi = float(history[key].min()), float(history[key].max())
            if not np.isfinite(lo + hi) or hi - lo <= 0.0: raise ValueError(f"Cannot rescale {key} from pretraining data.")
            normalized[key] = ((frame[key] - lo) / (hi - lo)).clip(0.0, 1.0)
            scales[key] = {"historical_min": lo, "historical_max": hi}
        return normalized, {"reference_end": str(first_training_day), "bounds": scales, "clipping": "values outside the reference range are clipped"}

    @staticmethod
    def bm1_to_wm4(values, normalization):
        try:
            mapping = normalization["bounds"]["I"]
            low, high = float(mapping["historical_min"]), float(mapping["historical_max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("BM1 result is missing the training-only demand normalization bounds.") from exc
        if not np.isfinite(low + high) or high <= low: raise ValueError("Invalid BM1 normalization bounds.")
        return low + (high - low) * np.asarray(values, dtype=float)

    @classmethod
    def hourly_wm4(cls, frame, day, forecasts):
        """Require exactly 24 unique one-hour-ahead MPC targets, including 00:00."""
        timestamps = pd.date_range(day, periods=24, freq="h")
        actual = frame.reindex(timestamps)["I"].to_numpy(dtype=float)
        if not np.all(np.isfinite(actual)): raise ValueError("Target day has incomplete observed consumption.")
        table = forecasts.copy()
        required = ("target_time", "lead_hours", "I_mpc", "lower_68", "upper_68", "lower_95", "upper_95")
        missing = set(required) - set(table.columns)
        if missing: raise ValueError("WM4 MPC rows lack: " + ", ".join(sorted(missing)))
        table["target_time"] = pd.to_datetime(table["target_time"])
        table = table.loc[table["lead_hours"].astype(int) == 1].set_index("target_time")
        if table.index.has_duplicates or not timestamps.isin(table.index).all():
            raise ValueError("Require a complete set of 24 unique WM4 lead-1 predictions (including the previous day's 23:00 origin).")
        table = table.reindex(timestamps)
        for name in ("I_mpc", "lower_68", "upper_68", "lower_95", "upper_95"):
            table[name] = pd.to_numeric(table[name], errors="coerce")
            if not np.all(np.isfinite(table[name])): raise ValueError(f"Invalid WM4 {name} values.")
        return table, actual, timestamps

    @staticmethod
    def save_directory(output_root, day):
        return Path(output_root) / f"{day:%Y-%m-%d}"
