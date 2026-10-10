"""Validation and sampling of BM1 shifted 24-hour MPC windows."""

import numpy as np
import pandas as pd

from src.validation.exe.run_model.bm_1.predict_daily import bm_1_prediction_validation as daily_checks


class bm_1_mpc_validation:
    @staticmethod
    def options(max_horizon, x_steps, fp_max_step, prior_bandwidth, substeps, weeks):
        for label, number, least in (("max_horizon", max_horizon, 1), ("x_steps", x_steps, 20),
            ("substeps", substeps, 1), ("weeks", weeks, 5)):
            if isinstance(number, bool) or not isinstance(number, int) or number < least:
                raise ValueError(f"{label} must be an integer >= {least}.")
        if not 1 <= max_horizon <= 8: raise ValueError("BM1 MPC supports prediction leads 1..8.")
        if weeks != 5: raise ValueError("Existing pooled BM1 estimator requires exactly five historical windows.")
        for label, number in (("fp_max_step", fp_max_step), ("prior_bandwidth", prior_bandwidth)):
            if not np.isfinite(number) or number <= 0: raise ValueError(f"{label} must be finite and positive.")
        return max_horizon, x_steps

    @staticmethod
    def origin(origin):
        result = pd.Timestamp(origin)
        if pd.isna(result) or result.tz is not None: raise ValueError("MPC origin must be a timezone-naive timestamp.")
        if result.minute != 0 or result.second != 0 or result.microsecond != 0:
            raise ValueError("BM1 MPC origin must be exactly on an hour boundary.")
        return result

    @staticmethod
    def frame(frame):
        return daily_checks.frame(frame)

    @classmethod
    def window(cls, frame, origin, baseline, *, hours=24, observations=True):
        """Provide strictly held price, temperature-derived baseline and past I.

        For target prediction windows, actual future I is separate and never
        supplied to the parameter fit or state-conditioning operation.
        """
        origin = cls.origin(origin)
        index = pd.date_range(origin, periods=hours, freq="h")
        signals = frame.reindex(index)
        required = ["U", "Temp"] + (["I"] if observations else [])
        if signals[required].isna().any().any():
            raise ValueError(f"Incomplete {hours}-hour BM1 {'training' if observations else 'forecast'} window at {origin}.")
        U = signals["U"].to_numpy(dtype=float)
        B = np.asarray(baseline.predict(index, signals["Temp"].to_numpy(dtype=float)), dtype=float)
        if U.shape != (hours,) or B.shape != (hours,) or not np.all(np.isfinite(B)) or np.any((U < 0) | (U > 1)) or np.any((B < 0) | (B > 1)):
            raise ValueError("BM1 MPC U and B must be finite normalized hourly series.")
        result = {"Time": np.arange(hours + 1, dtype=float), "U": np.r_[U, U[-1]], "B": np.r_[B, B[-1]]}
        if observations:
            I = signals["I"].to_numpy(dtype=float)
            if np.any((I < 0.0) | (I > 1.0)): raise ValueError("BM1 historical demand must be in [0,1].")
            result["I"] = I
        return result

    @staticmethod
    def observation(frame, origin):
        value = float(frame["I"].reindex([origin]).iloc[0])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"At {origin}, BM1 MPC requires one observed normalized I for state updating.")
        return value

    @staticmethod
    def posterior(mass, expected_size):
        mass = np.asarray(mass, dtype=float)
        if mass.shape != (expected_size,) or not np.all(np.isfinite(mass)) or np.any(mass < 0.0) or mass.sum() <= 0.0:
            raise ValueError("BM1 MPC posterior must be finite nonnegative probability mass on the solver grid.")
        return mass / mass.sum()
