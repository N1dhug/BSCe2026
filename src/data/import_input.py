from __future__ import annotations

from pathlib import Path

import pandas as pd


class InputData:
    """Import and query the internal 2022 consumption workbook sheet.

    The expected worksheet has a two-row header:
      - row 1: measurement/customer identifier
      - row 2: metadata tag, for example MED or UDEN

    The first columns contain calendar information such as
    År, Måned, Dato and Time.
    """

    SHEET_NAME = "2022 med dato (3)"
    DATE_COLUMN = "Dato"
    TIME_COLUMN = "Time"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Input workbook not found: {self.path}")

        data = pd.read_excel(
            self.path,
            sheet_name=self.SHEET_NAME,
            header=[0, 1],
            engine="openpyxl",
        )

        data.columns = self._clean_columns(data.columns)
        data = self._build_datetime_index(data)
        data = self._coerce_measurements(data)
        data = self._impute_missing(data)

        return data.sort_index()

    @staticmethod
    def _clean_columns(columns: pd.MultiIndex) -> pd.Index:
        """Flatten calendar columns while preserving measurement tags."""
        cleaned: list[str] = []

        for top, bottom in columns:
            top = str(top).strip()
            bottom = str(bottom).strip()

            if top.startswith("Unnamed"):
                cleaned.append(bottom)
            else:
                cleaned.append(f"{top}_{bottom}")

        return pd.Index(cleaned)

    def _build_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.DATE_COLUMN not in data.columns:
            raise ValueError(
                f"Worksheet '{self.SHEET_NAME}' does not contain "
                f"the expected '{self.DATE_COLUMN}' column."
            )

        date = pd.to_datetime(
            data[self.DATE_COLUMN],
            dayfirst=True,
            errors="coerce",
        )

        if self.TIME_COLUMN in data.columns:
            time = data[self.TIME_COLUMN]

            numeric_time = pd.to_numeric(time, errors="coerce")
            numeric_delta = pd.to_timedelta(numeric_time, unit="D")

            string_delta = pd.to_timedelta(
                time.astype(str),
                errors="coerce",
            )

            time_delta = numeric_delta.fillna(string_delta)
            timestamp = date.dt.normalize() + time_delta
        else:
            timestamp = date

        if timestamp.isna().any():
            bad_rows = int(timestamp.isna().sum())
            raise ValueError(
                f"Could not parse date/time for {bad_rows} row(s) in "
                f"worksheet '{self.SHEET_NAME}'."
            )

        data = data.copy()
        data.index = pd.DatetimeIndex(timestamp, name="Datetime")

        return data

    @property
    def metadata_columns(self) -> list[str]:
        return [
            column
            for column in ("År", "Måned", self.DATE_COLUMN, self.TIME_COLUMN)
            if column in self.data.columns
        ]

    @property
    def measurement_columns(self) -> list[str]:
        metadata = set(self.metadata_columns)
        return [column for column in self.data.columns if column not in metadata]

    @staticmethod
    def measurement_id(column: str) -> str:
        """Return the identifier from a flattened id_tag column."""
        return column.rsplit("_", 1)[0]

    @staticmethod
    def measurement_tag(column: str) -> str:
        """Return the metadata tag from a flattened id_tag column."""
        parts = column.rsplit("_", 1)
        return parts[1] if len(parts) == 2 else ""

    def _coerce_measurements(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        metadata = {
            column
            for column in ("År", "Måned", self.DATE_COLUMN, self.TIME_COLUMN)
            if column in data.columns
        }

        measurement_columns = [
            column for column in data.columns if column not in metadata
        ]

        data[measurement_columns] = data[measurement_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        return data

    def _impute_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing measurements independently for every meter.

        Interior gaps are linearly interpolated in time. Leading/trailing
        gaps are filled from the nearest observed value. Metadata columns
        are left unchanged.
        """
        data = data.copy()

        metadata = {
            column
            for column in ("År", "Måned", self.DATE_COLUMN, self.TIME_COLUMN)
            if column in data.columns
        }
        measurement_columns = [
            column for column in data.columns if column not in metadata
        ]

        if not measurement_columns:
            return data

        measurements = data[measurement_columns].copy()
        measurements = measurements.interpolate(
            method="time",
            axis=0,
            limit_direction="both",
        )
        measurements = measurements.ffill().bfill()

        data[measurement_columns] = measurements
        return data

    def _select_period(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        mask = (self.data.index >= start) & (self.data.index < end)
        return self.data.loc[mask].copy()

    def day(self, date: str | pd.Timestamp) -> pd.DataFrame:
        """Return all samples belonging to one calendar day."""
        start = pd.Timestamp(date).normalize()
        return self._select_period(start, start + pd.Timedelta(days=1))

    def week(self, date: str | pd.Timestamp) -> pd.DataFrame:
        """Return the Monday-Sunday week containing date."""
        date = pd.Timestamp(date).normalize()
        start = date - pd.Timedelta(days=date.weekday())
        return self._select_period(start, start + pd.Timedelta(days=7))

    def month(
        self,
        date: str | pd.Timestamp | int,
        month: int | None = None,
    ) -> pd.DataFrame:
        """Return one calendar month.

        Accepted forms:
            month("2022-01")
            month("2022-01-15")
            month(2022, 1)
        """
        if month is None:
            value = pd.Timestamp(date)
            year = value.year
            month = value.month
        else:
            year = int(date)

        start = pd.Timestamp(year=year, month=int(month), day=1)
        end = start + pd.offsets.MonthBegin(1)
        return self._select_period(start, end)

    def by_tag(self, tag: str) -> pd.DataFrame:
        """Return all measurement columns matching a tag such as MED/UDEN."""
        tag = tag.strip().upper()
        columns = [
            column
            for column in self.measurement_columns
            if self.measurement_tag(column).upper() == tag
        ]
        return self.data[columns].copy()

    def by_id(self, measurement_id: str | int) -> pd.DataFrame:
        """Return every column belonging to one measurement identifier."""
        measurement_id = str(measurement_id)
        columns = [
            column
            for column in self.measurement_columns
            if self.measurement_id(column) == measurement_id
        ]
        return self.data[columns].copy()
