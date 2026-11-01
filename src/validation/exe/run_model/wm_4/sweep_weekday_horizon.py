import numpy as np
import pandas as pd


class sweep_weekday_horizon_validation:
    @staticmethod
    def arguments(args):
        if args.group not in ("MED", "ALL"):
            raise ValueError("group must be MED or ALL.")
        names = getattr(args, "baselines", ("statistical_output", "weekly_average", "weekly_temperature"))
        choices = {"statistical_output", "weekly_average", "weekly_temperature"}
        if not names or len(set(names)) != len(names) or not set(names) <= choices:
            raise ValueError("Baselines must be unique and selected from the three supported methods.")
        if not 2 <= args.min_weekdays <= args.max_weekdays <= 10:
            raise ValueError("Weekday training counts must satisfy 2 <= min <= max <= 10.")
        if not 6 <= args.min_horizon <= args.max_horizon <= 24:
            raise ValueError("Forecast horizons must satisfy 6 <= min <= max <= 24.")
        if args.steps_per_hour < 2 or args.max_days < 0 or args.progress_every < 1:
            raise ValueError("steps_per_hour >=2, max_days >=0 and progress_every >=1 are required.")
        for label in ("start_day", "end_day"):
            raw = getattr(args, label)
            if raw is None: continue
            value = pd.Timestamp(raw)
            if pd.isna(value) or value != value.normalize():
                raise ValueError(f"{label} must represent a valid midnight calendar day.")
        if args.start_day and args.end_day and pd.Timestamp(args.start_day) > pd.Timestamp(args.end_day):
            raise ValueError("start_day cannot follow end_day.")

    @staticmethod
    def training_data(frame, days):
        if not isinstance(frame, pd.DataFrame) or not {"I", "U", "Temp"} <= set(frame.columns):
            raise ValueError("Input frame needs I, U and Temp columns.")
        if not isinstance(days, (list, tuple)) or not 2 <= len(days) <= 10:
            raise ValueError("Training requires between 2 and 10 preceding matching weekdays.")
        if len(set(pd.Timestamp(x) for x in days)) != len(days):
            raise ValueError("Training dates must be distinct.")
        blocks = []
        for day in days:
            day = pd.Timestamp(day)
            if day != day.normalize():
                raise ValueError("Training days must begin at midnight.")
            index = pd.date_range(day, periods=24, freq="h")
            sample = frame.reindex(index)[["I", "U", "Temp"]]
            if sample.isna().any().any():
                raise ValueError(f"Missing consumption, price or temperature on {day:%Y-%m-%d}.")
            blocks.append(sample["I"].to_numpy(dtype=float))
        values = np.vstack(blocks)
        if not np.isfinite(values).all():
            raise ValueError("Training demand must be finite.")
        return values

    @staticmethod
    def oracle(observed, base, sensitivity, min_horizon, max_horizon):
        observed = np.asarray(observed, dtype=float).reshape(-1)
        base = np.asarray(base, dtype=float).reshape(-1)
        sensitivity = np.asarray(sensitivity, dtype=float).reshape(-1)
        if observed.shape != (24,) or base.shape != (24,) or sensitivity.shape != (24,):
            raise ValueError("Oracle requires 24 hourly observations and matching predictions/sensitivities.")
        if not np.isfinite(np.r_[observed, base, sensitivity]).all():
            raise ValueError("Oracle inputs must be finite.")
        if not 6 <= int(min_horizon) <= int(max_horizon) <= 24:
            raise ValueError("Oracle horizons must lie between 6 and 24 hours.")
        return observed, base, sensitivity

    @staticmethod
    def history(day, days):
        day = pd.Timestamp(day)
        if len(days) < 2 or len(days) > 10 or any(pd.Timestamp(d) >= day for d in days):
            raise ValueError("All training weekdays must precede the forecast origin.")
        if any(pd.Timestamp(d).dayofweek != day.dayofweek for d in days):
            raise ValueError("Historical training dates must match the forecast weekday.")
        return days
