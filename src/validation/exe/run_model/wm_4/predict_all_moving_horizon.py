import numpy as np
import pandas as pd


class predict_all_moving_horizon_validation:
    @staticmethod
    def arguments(args):
        if args.training_weekdays != 5:
            raise ValueError("This study requires exactly five historical corresponding weekdays.")
        if args.steps_per_hour < 2 or args.max_days < 0 or args.progress_every < 1:
            raise ValueError("steps_per_hour >=2, max_days >=0 and progress_every >=1 are required.")
        if args.group not in ("MED", "ALL"):
            raise ValueError("group must be MED or ALL.")
        for name in ("start_day", "end_day"):
            value = getattr(args, name)
            if value is not None and (pd.isna(pd.Timestamp(value)) or pd.Timestamp(value) != pd.Timestamp(value).normalize()):
                raise ValueError(f"{name} must be a valid midnight date.")
        if args.start_day is not None and args.end_day is not None and pd.Timestamp(args.start_day) > pd.Timestamp(args.end_day):
            raise ValueError("start_day must not follow end_day.")

    @staticmethod
    def oracle(observed, base_prediction, sensitivity, r0):
        observed = np.asarray(observed, dtype=float).reshape(-1)
        base_prediction = np.asarray(base_prediction, dtype=float).reshape(-1)
        sensitivity = np.asarray(sensitivity, dtype=float).reshape(-1)
        if observed.shape != (24,) or base_prediction.shape != observed.shape or sensitivity.shape != observed.shape:
            raise ValueError("Oracle F0 requires exactly 24 aligned hourly observations.")
        if not np.isfinite(np.r_[observed, base_prediction, sensitivity, r0]).all():
            raise ValueError("Oracle observations, response and historical R0 must be finite.")
        if not -1.0 <= float(r0) <= 1.0:
            raise ValueError("Historical R0 must lie in [-1, 1].")
        if sensitivity @ sensitivity < 1e-14:
            raise ValueError("Oracle sensitivity to F0 is too small to identify F0.")
        return observed, base_prediction, sensitivity
