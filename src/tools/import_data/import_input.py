from __future__ import annotations

from datetime import time as dt_time
from pathlib import Path

import pandas as pd


def interpolate_time_gaps(values):
    """Existing importer policy; callers delimit the permitted time window."""
    return values.interpolate(method="time", limit_direction="both").ffill().bfill()


class input_data:
    """Import/query consumption data from the internal 2022 sheet."""

    SHEET_NAME, DATE_COLUMN, TIME_COLUMN = "2022 med dato (3)", "Dato", "Time"
    META = ("År", "Måned", DATE_COLUMN, TIME_COLUMN)

    def __init__(self, path: str | Path):
        self.path, self.data, self.observed_data = Path(path), None, None
        self.data = self._load()

    def _load(self):
        if not self.path.exists(): raise FileNotFoundError(f"Input workbook not found: {self.path}")
        data = pd.read_excel(self.path, sheet_name=self.SHEET_NAME, header=[0, 1], engine="openpyxl")
        data.columns = self._clean_columns(data.columns)
        data = self._coerce_measurements(self._build_datetime_index(data)).sort_index()
        self.observed_data = data.copy()
        return self._impute_missing(self._complete_hourly_index(data))

    @staticmethod
    def _clean_columns(columns):
        return pd.Index([str(bottom).strip() if str(top).startswith("Unnamed") else f"{str(top).strip()}_{str(bottom).strip()}" for top, bottom in columns])

    @staticmethod
    def _date_series(values):
        numeric = pd.to_numeric(values, errors="coerce")
        result = pd.to_datetime(values, dayfirst=True, errors="coerce")
        excel = numeric.between(20_000, 80_000)
        yyyymmdd = numeric.between(19_000_101, 21_001_231)
        if excel.any(): result.loc[excel] = pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric.loc[excel], unit="D")
        if yyyymmdd.any(): result.loc[yyyymmdd] = pd.to_datetime(numeric.loc[yyyymmdd].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        return result.ffill()

    @staticmethod
    def _time_delta(values):
        delta = pd.Series(pd.NaT, index=values.index, dtype="timedelta64[ns]")
        numeric = pd.to_numeric(values, errors="coerce")
        fraction = numeric.notna() & numeric.ge(0) & numeric.lt(1)
        hour = numeric.notna() & numeric.ge(0) & numeric.le(24) & ~fraction
        hhmm = numeric.notna() & numeric.gt(24) & numeric.le(2359) & numeric.mod(1).eq(0)

        delta.loc[fraction] = pd.to_timedelta(numeric.loc[fraction], unit="D")
        delta.loc[hour] = pd.to_timedelta(numeric.loc[hour], unit="h")
        if hhmm.any():
            n = numeric.loc[hhmm].astype(int)
            valid = n.mod(100).lt(60) & n.floordiv(100).lt(24)
            idx = n.index[valid]
            delta.loc[idx] = pd.to_timedelta(n.loc[idx].floordiv(100), unit="h") + pd.to_timedelta(n.loc[idx].mod(100), unit="m")

        obj_time = values.map(lambda x: isinstance(x, dt_time))
        if obj_time.any():
            t = values.loc[obj_time]
            seconds = t.map(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1e6)
            delta.loc[obj_time] = pd.to_timedelta(seconds, unit="s")

        remaining = delta.isna() & values.notna()
        if remaining.any():
            text = values.loc[remaining].astype(str).str.strip()
            direct = pd.to_timedelta(text.str.replace(",", ".", regex=False), errors="coerce")
            match = text.str.extract(r"(?<!\d)(\d{1,2})(?:[:.](\d{2}))?(?=\s*(?:-|–|—|to|$))")
            hours = pd.to_numeric(match[0], errors="coerce")
            minutes = pd.to_numeric(match[1], errors="coerce").fillna(0)
            parsed = pd.to_timedelta(hours.where(hours.between(0, 23)), unit="h") + pd.to_timedelta(minutes.where(minutes.between(0, 59)), unit="m")
            delta.loc[remaining] = direct.where(direct.notna(), parsed)
        return delta

    def _build_datetime_index(self, data):
        if self.DATE_COLUMN not in data.columns: raise ValueError(f"Worksheet '{self.SHEET_NAME}' does not contain '{self.DATE_COLUMN}'.")
        date = data[self.DATE_COLUMN].ffill()
        if self.TIME_COLUMN in data.columns:
            time = data[self.TIME_COLUMN]
            timestamp = pd.to_datetime(date.astype(str).str.strip() + " " + time.astype(str).str.strip(), dayfirst=True, errors="coerce")
            fallback = timestamp.isna()
            if fallback.any(): timestamp.loc[fallback] = self._date_series(date.loc[fallback]).dt.normalize() + self._time_delta(time.loc[fallback])
        else: timestamp = self._date_series(date)

        valid = timestamp.notna()
        if not valid.all():
            cols = [c for c in data.columns if c not in self.META]
            has_measurement = data.loc[~valid, cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
            if has_measurement.any():
                bad = data.index[~valid][has_measurement]
                rows = list(bad[:10])
                sample = data.loc[rows, [c for c in (self.DATE_COLUMN, self.TIME_COLUMN) if c in data.columns]].to_dict("records")
                raise ValueError(f"Could not parse date/time for {len(bad)} data row(s); first rows={rows}, raw values={sample}.")
            data, timestamp = data.loc[valid].copy(), timestamp.loc[valid]

        data = data.copy()
        data.index = pd.DatetimeIndex(timestamp, name="Datetime")
        return data

    @property
    def metadata_columns(self): return [c for c in self.META if c in self.data.columns]

    @property
    def measurement_columns(self):
        meta = set(self.metadata_columns)
        return [c for c in self.data.columns if c not in meta]

    @staticmethod
    def measurement_id(column): return column.rsplit("_", 1)[0]

    @staticmethod
    def measurement_tag(column):
        parts = column.rsplit("_", 1)
        return parts[1] if len(parts) == 2 else ""

    def _coerce_measurements(self, data):
        data, cols = data.copy(), [c for c in data.columns if c not in self.META]
        data[cols] = data[cols].apply(pd.to_numeric, errors="coerce")
        return data

    def _complete_hourly_index(self, data):
        if data.empty: return data.copy()
        if data.index.has_duplicates:
            duplicates = data.index[data.index.duplicated()].unique()
            raise ValueError(f"Duplicate input timestamps found: {list(duplicates[:10])}")
        index = pd.date_range(data.index.min().floor("h"), data.index.max().ceil("h"), freq="h", name="Datetime")
        return data.reindex(index)

    def _impute_missing(self, data):
        data, cols = data.copy(), [c for c in data.columns if c not in self.META]
        if cols: data[cols] = interpolate_time_gaps(data[cols])
        return data

    def _select_period(self, start, end):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        return self.data.loc[(self.data.index >= start) & (self.data.index < end)].copy()

    def day(self, date):
        start = pd.Timestamp(date).normalize()
        return self._select_period(start, start + pd.Timedelta(days=1))

    def week(self, date):
        date = pd.Timestamp(date).normalize()
        start = date - pd.Timedelta(days=date.weekday())
        return self._select_period(start, start + pd.Timedelta(days=7))

    def month(self, date, month=None):
        if month is None:
            date = pd.Timestamp(date)
            year, month = date.year, date.month
        else: year = int(date)
        start = pd.Timestamp(year=year, month=int(month), day=1)
        return self._select_period(start, start + pd.offsets.MonthBegin(1))

    def by_tag(self, tag):
        tag = tag.strip().upper()
        return self.data[[c for c in self.measurement_columns if self.measurement_tag(c).upper() == tag]].copy()

    def by_id(self, measurement_id):
        measurement_id = str(measurement_id)
        return self.data[[c for c in self.measurement_columns if self.measurement_id(c) == measurement_id]].copy()
