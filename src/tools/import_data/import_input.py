from __future__ import annotations

from pathlib import Path

import pandas as pd


class input_data:
    """Import/query consumption data from the internal 2022 sheet."""

    SHEET_NAME, DATE_COLUMN, TIME_COLUMN = "2022 med dato (3)", "Dato", "Time"
    META = ("År", "Måned", DATE_COLUMN, TIME_COLUMN)

    def __init__(self, path: str | Path):
        self.path, self.data = Path(path), None
        self.data = self._load()

    def _load(self):
        if not self.path.exists(): raise FileNotFoundError(f"Input workbook not found: {self.path}")
        data = pd.read_excel(self.path, sheet_name=self.SHEET_NAME, header=[0, 1], engine="openpyxl")
        data.columns = self._clean_columns(data.columns)
        return self._impute_missing(self._coerce_measurements(self._build_datetime_index(data))).sort_index()

    @staticmethod
    def _clean_columns(columns):
        return pd.Index([str(bottom).strip() if str(top).startswith("Unnamed") else f"{str(top).strip()}_{str(bottom).strip()}" for top, bottom in columns])

    @staticmethod
    def _time_delta(time):
        numeric = pd.to_numeric(time, errors="coerce")
        delta = pd.Series(pd.NaT, index=time.index, dtype="timedelta64[ns]")
        fractional = numeric.notna() & (numeric.abs() < 1)
        hours = numeric.notna() & ~fractional & (numeric >= 0) & (numeric <= 24)
        delta.loc[fractional] = pd.to_timedelta(numeric.loc[fractional], unit="D")
        delta.loc[hours] = pd.to_timedelta(numeric.loc[hours], unit="h")

        remaining = delta.isna() & time.notna()
        if remaining.any():
            text = time.loc[remaining].astype(str).str.strip()
            parsed = pd.to_timedelta(text, errors="coerce")
            range_start = text.str.extract(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(?:-|–|to)")[0]
            range_minute = text.str.extract(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(?:-|–|to)")[1]
            range_hour = pd.to_numeric(range_start, errors="coerce")
            range_minute = pd.to_numeric(range_minute, errors="coerce").fillna(0)
            range_delta = pd.to_timedelta(range_hour, unit="h") + pd.to_timedelta(range_minute, unit="m")
            parsed = parsed.where(parsed.notna(), range_delta)
            delta.loc[remaining] = parsed
        return delta

    def _build_datetime_index(self, data):
        if self.DATE_COLUMN not in data.columns: raise ValueError(f"Worksheet '{self.SHEET_NAME}' does not contain '{self.DATE_COLUMN}'.")
        date = pd.to_datetime(data[self.DATE_COLUMN], dayfirst=True, errors="coerce").ffill()
        if self.TIME_COLUMN in data.columns: timestamp = date.dt.normalize() + self._time_delta(data[self.TIME_COLUMN])
        else: timestamp = date

        valid = timestamp.notna()
        if not valid.all():
            measurement_cols = [c for c in data.columns if c not in self.META]
            has_measurement = data.loc[~valid, measurement_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
            if has_measurement.any():
                rows = list(data.index[~valid][has_measurement][:10])
                raise ValueError(f"Could not parse date/time for {int(has_measurement.sum())} data row(s); first affected Excel data rows: {rows}.")
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

    def _impute_missing(self, data):
        data, cols = data.copy(), [c for c in data.columns if c not in self.META]
        if cols: data[cols] = data[cols].interpolate(method="time", limit_direction="both").ffill().bfill()
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
