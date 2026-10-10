"""Validation for matched 1-8h BM1 / WM4 MPC evaluation."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.validation.exe.run_model.bm_1.predict_mpc import bm_1_mpc_validation


class compare_mpc_validation:
    @staticmethod
    def date_range(start, end, horizon, steps_per_hour, x_steps, fp_max_step, prior_bandwidth, substeps):
        start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        if pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None or end < start:
            raise ValueError("MPC comparison requires start_day <= end_day, without timezone.")
        bm_1_mpc_validation.options(horizon, x_steps, fp_max_step, prior_bandwidth, substeps, 5)
        if isinstance(steps_per_hour, bool) or not isinstance(steps_per_hour, int) or steps_per_hour < 2:
            raise ValueError("steps_per_hour must be an integer >=2 for WM4.")
        return start, end

    @staticmethod
    def wm4_saved(source):
        if source is None: return None
        frame = pd.read_csv(source) if isinstance(source, (str, Path)) else source.copy()
        if not isinstance(frame, pd.DataFrame): raise TypeError("WM4 saved forecasts must be a DataFrame/CSV path.")
        required = {"origin", "target_time", "lead_hours", "I_mpc", "lower_68", "upper_68", "lower_95", "upper_95"}
        if missing := required - set(frame.columns):
            raise ValueError("WM4 saved forecast missing " + ", ".join(sorted(missing)))
        frame["origin"] = pd.to_datetime(frame["origin"])
        frame["target_time"] = pd.to_datetime(frame["target_time"])
        if frame["origin"].dt.tz is not None or frame["target_time"].dt.tz is not None:
            raise ValueError("WM4 saved timestamps must be timezone-naive.")
        frame["lead_hours"] = pd.to_numeric(frame["lead_hours"]).astype(int)
        if frame.duplicated(subset=("origin", "lead_hours")).any():
            raise ValueError("WM4 saved forecasts contain duplicate origin/lead rows.")
        return frame.set_index(["origin", "lead_hours"], drop=False).sort_index()


    @staticmethod
    def observed(value, timestamp):
        value = float(value)
        if not np.isfinite(value): raise ValueError(f"Missing actual consumption at {timestamp}.")
        return value

    @staticmethod
    def complete_pair(bm_rows, wm_rows, origin, horizon):
        if len(bm_rows) != horizon or len(wm_rows) != horizon: raise ValueError(f"{origin}: incomplete {horizon}-lead predictions.")
        b, w = pd.DataFrame(bm_rows).sort_values("lead_hours"), pd.DataFrame(wm_rows).sort_values("lead_hours")
        expected = np.arange(1, horizon + 1, dtype=int)
        if not np.array_equal(b["lead_hours"].to_numpy(dtype=int), expected) or not np.array_equal(w["lead_hours"].to_numpy(dtype=int), expected):
            raise ValueError(f"{origin}: lead indices do not cover 1..{horizon}.")
        if not pd.to_datetime(b["target_time"]).reset_index(drop=True).equals(pd.to_datetime(w["target_time"]).reset_index(drop=True)):
            raise ValueError(f"{origin}: target timestamps disagree across BM1 and WM4.")
        return b, w

    @staticmethod
    def saved_day(folder, options):
        folder = Path(folder)
        csv, info = folder / "matched_forecasts.csv", folder / "settings.json"
        if not csv.exists() or not info.exists(): return None
        with info.open(encoding="utf-8") as handle: previous = json.load(handle)
        if previous != options: return None
        frame = pd.read_csv(csv, parse_dates=["origin", "target_time", "origin_day"])
        required = {"origin", "target_time", "lead_hours", "I_observed", "I_BM1", "I_WM4",
            "BM1_lower_68", "BM1_upper_68", "BM1_lower_95", "BM1_upper_95",
            "WM4_lower_68", "WM4_upper_68", "WM4_lower_95", "WM4_upper_95"}
        if missing := required - set(frame.columns):
            raise ValueError("Cached matched MPC forecasts missing " + ", ".join(sorted(missing)))
        horizon = int(options["horizon"])
        if frame.duplicated(subset=("origin", "lead_hours")).any():
            raise ValueError("Cached MPC comparison contains duplicate origin/lead combinations.")
        for origin, group in frame.groupby("origin"):
            if not np.array_equal(np.sort(group["lead_hours"].to_numpy(dtype=int)), np.arange(1, horizon + 1)):
                raise ValueError(f"Cached origin {origin} contains incomplete or invalid prediction leads.")
            expected = pd.Timestamp(origin) + pd.to_timedelta(group["lead_hours"], unit="h")
            if not np.array_equal(pd.DatetimeIndex(group["target_time"]).to_numpy(), pd.DatetimeIndex(expected).to_numpy()):
                raise ValueError(f"Cached origin {origin} has inconsistent prediction target times.")
        return frame
