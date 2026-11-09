"""Validation twin for BM1 adaptive beta-mixture methodology executable."""
import numpy as np
import pandas as pd


class bm_1_beta_moving_horizon_validation:
    @staticmethod
    def origin(origin):
        stamp = pd.Timestamp(origin)
        if pd.isna(stamp) or stamp.tz is not None or stamp != stamp.floor("h"):
            raise ValueError("Origin must be a timezone-naive timestamp exactly on an hour.")
        return stamp

    @staticmethod
    def settings(N, k, maxiter, deterministic_cycles, max_nfev):
        for name, value, minimum in (("N", N, 1), ("k", k, 2),
                                     ("maxiter", maxiter, 1),
                                     ("deterministic_cycles", deterministic_cycles, 1),
                                     ("max_nfev", max_nfev, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if k > 168:
            raise ValueError("k must not exceed 168 hours: the latest weekly historical window cannot overlap the prediction interval.")
        return N, k

    @staticmethod
    def frame(frame):
        if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("BM1 data require a DataFrame indexed by timestamps.")
        if frame.index.tz is not None or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("BM1 data must have a sorted, unique, naive hourly DatetimeIndex.")
        missing = {"I", "U", "Temp"} - set(frame.columns)
        if missing: raise ValueError(f"BM1 data missing columns {sorted(missing)}.")
        if not np.isfinite(frame[["I", "U", "Temp"]].to_numpy(dtype=float)).all():
            raise ValueError("BM1 input signals must be finite.")
        return frame

    @classmethod
    def window(cls, frame, origin, k, baseline=None):
        origin = cls.origin(origin)
        index = pd.date_range(origin, periods=k, freq="h")
        selected = frame.reindex(index)
        if selected[["I", "U", "Temp"]].isna().any().any():
            raise ValueError(f"Incomplete {k}-hour BM1 interval starting {origin}.")
        b = selected["I"].to_numpy(dtype=float) if baseline is None else baseline.predict(
            index, selected["Temp"].to_numpy(dtype=float))
        u, y = selected["U"].to_numpy(dtype=float), selected["I"].to_numpy(dtype=float)
        for name, values in (("baseline B", b), ("price U", u), ("demand I", y)):
            if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError(f"BM1 {name} must lie inside [0,1].")
        return {"Time": np.arange(k, dtype=float), "I": y, "U":u, "B":np.asarray(b, dtype=float),
                "timestamps":index}
