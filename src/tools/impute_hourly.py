"""Within-calendar-day hourly interpolation for historical training data.

An entire missing day is NEVER synthesized. Values at the start/end of an
observed day use that same day's nearest observed hour; interior gaps use
linear time interpolation. This is for historical fitting, not score labels.
"""
import numpy as np
import pandas as pd


def impute_hourly_days(frame, days=None, columns=("I", "Temp"), require_days=False):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Hourly imputation requires a datetime-indexed DataFrame.")
    if frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Hourly index must be sorted, unique and contain no NaT.")
    columns = tuple(columns)
    if not columns or len(set(columns)) != len(columns) or any(column not in frame for column in columns):
        raise ValueError("Imputation columns must be unique existing columns.")
    if days is None:
        days = pd.DatetimeIndex(frame.index.normalize().unique()).sort_values()
    else:
        days = pd.DatetimeIndex(days).normalize().unique().sort_values()
    if len(days) == 0: raise ValueError("Hourly imputation requires at least one calendar day.")

    blocks, unavailable, per_column = [], [], {name: 0 for name in columns}
    changed_slots = 0
    for day in days:
        index = pd.date_range(day, periods=24, freq="h")
        original = frame.reindex(index)[list(columns)].replace([np.inf, -np.inf], np.nan)
        # Any entirely absent required signal means the WHOLE day is unusable.
        if original.notna().sum().eq(0).any():
            unavailable.append(day.strftime("%Y-%m-%d"))
            continue
        missing = original.isna()
        block = original.interpolate(method="time", limit_direction="both")
        if block.isna().any().any(): raise RuntimeError(f"Hourly interpolation failed for {day:%Y-%m-%d}.")
        for name in columns: per_column[name] += int(missing[name].sum())
        changed_slots += int(missing.any(axis=1).sum())
        blocks.append(block)
    if unavailable and require_days:
        raise ValueError("Missing full day(s), which will not be imputed: " + ", ".join(unavailable))
    if not blocks:
        raise ValueError("No complete or partially observed calendar days are available for hourly imputation.")
    filled = pd.concat(blocks).sort_index()
    details = {
        "imputed_hourly_slots": changed_slots,
        "imputed_by_column": per_column,
        "included_days": len(blocks),
        "missing_days": unavailable,
        "method": "linear_within_day_nearest_edge",
    }
    return filled, details
