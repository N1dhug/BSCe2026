from __future__ import annotations

from pathlib import Path

import pandas as pd


class SpotPriceData:
    """Import and query hourly electricity spot prices.

    Expected columns:
        HourDK, HourUTC, PriceArea, SpotPriceDKK, SpotPriceEUR

    HourUTC is used as the unique time reference and converted to Danish
    local time for indexing. This avoids ambiguity around daylight-saving
    time changes while keeping day/week/month queries in Danish local time.
    """

    LOCAL_TIMEZONE = "Europe/Copenhagen"
    REQUIRED_COLUMNS = (
        "HourDK",
        "HourUTC",
        "PriceArea",
        "SpotPriceDKK",
        "SpotPriceEUR",
    )
    PRICE_COLUMNS = ("SpotPriceDKK", "SpotPriceEUR")

    def __init__(
        self,
        path: str | Path,
        sheet_name: str | int = 0,
        price_area: str | None = None,
    ):
        self.path = Path(path)
        self.sheet_name = sheet_name
        self.price_area = price_area.upper() if price_area is not None else None
        self.data = self._load()

    def _read(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Spot-price file not found: {self.path}")

        suffix = self.path.suffix.lower()

        if suffix in {".xlsx", ".xlsm"}:
            return pd.read_excel(
                self.path,
                sheet_name=self.sheet_name,
                engine="openpyxl",
            )

        if suffix == ".xls":
            return pd.read_excel(
                self.path,
                sheet_name=self.sheet_name,
            )

        if suffix in {".csv", ".txt"}:
            return pd.read_csv(
                self.path,
                sep=None,
                engine="python",
            )

        raise ValueError(
            "Unsupported spot-price file format. "
            "Expected .xlsx, .xlsm, .xls, .csv or .txt."
        )

    def _load(self) -> pd.DataFrame:
        data = self._read()
        data.columns = data.columns.astype(str).str.strip()

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in data.columns
        ]
        if missing:
            raise ValueError(
                "Spot-price data is missing required column(s): "
                + ", ".join(missing)
            )

        data = data[list(self.REQUIRED_COLUMNS)].copy()

        utc = pd.to_datetime(
            data["HourUTC"],
            errors="coerce",
            utc=True,
        )
        if utc.isna().any():
            bad_rows = int(utc.isna().sum())
            raise ValueError(
                f"Could not parse HourUTC for {bad_rows} row(s)."
            )

        data["_HourUTC"] = utc
        data["PriceArea"] = data["PriceArea"].astype(str).str.strip().str.upper()

        for column in self.PRICE_COLUMNS:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        if self.price_area is not None:
            data = data.loc[data["PriceArea"] == self.price_area].copy()
            if data.empty:
                raise ValueError(
                    f"Price area '{self.price_area}' was not found in the file."
                )

        data = self._impute_missing_hours(data)

        data["HourUTC"] = data["_HourUTC"]
        data["HourDK"] = data["_HourUTC"].dt.tz_convert(self.LOCAL_TIMEZONE)

        data = data.drop(columns="_HourUTC")
        data = data.set_index("HourDK", drop=False)
        data.index.name = "Datetime"

        return data.sort_index()

    def _impute_missing_hours(self, data: pd.DataFrame) -> pd.DataFrame:
        """Insert missing hourly rows and impute missing prices per price area.

        Missing hours are created on the UTC timeline, so daylight-saving
        transitions do not introduce false gaps or duplicate timestamps.
        Price values are linearly interpolated in time. Leading/trailing
        missing values are filled from the nearest observed price.
        """
        groups: list[pd.DataFrame] = []

        for area, area_data in data.groupby("PriceArea", sort=False):
            area_data = area_data.copy()
            area_data = area_data.set_index("_HourUTC").sort_index()

            if area_data.index.has_duplicates:
                duplicates = area_data.index[area_data.index.duplicated()].unique()
                raise ValueError(
                    f"Duplicate HourUTC values found for price area '{area}': "
                    f"{list(duplicates[:5])}"
                )

            hourly_index = pd.date_range(
                start=area_data.index.min(),
                end=area_data.index.max(),
                freq="h",
            )

            area_data = area_data.reindex(hourly_index)
            area_data.index.name = "_HourUTC"
            area_data["PriceArea"] = area

            prices = area_data[list(self.PRICE_COLUMNS)].interpolate(
                method="time",
                limit_direction="both",
            )
            prices = prices.ffill().bfill()
            area_data[list(self.PRICE_COLUMNS)] = prices

            area_data = area_data.reset_index()
            groups.append(area_data)

        if not groups:
            return data

        return pd.concat(groups, ignore_index=True)

    def _local_timestamp(self, value: str | pd.Timestamp) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)

        if timestamp.tzinfo is None:
            return timestamp.tz_localize(self.LOCAL_TIMEZONE)

        return timestamp.tz_convert(self.LOCAL_TIMEZONE)

    def _select_period(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        start = self._local_timestamp(start)
        end = self._local_timestamp(end)

        mask = (self.data.index >= start) & (self.data.index < end)
        return self.data.loc[mask].copy()

    def day(self, date: str | pd.Timestamp) -> pd.DataFrame:
        """Return all spot-price samples belonging to one Danish calendar day."""
        start = self._local_timestamp(date).normalize()
        end = start + pd.DateOffset(days=1)
        return self._select_period(start, end)

    def week(self, date: str | pd.Timestamp) -> pd.DataFrame:
        """Return the Monday-Sunday Danish calendar week containing date."""
        date = self._local_timestamp(date).normalize()
        start = date - pd.DateOffset(days=date.weekday())
        end = start + pd.DateOffset(days=7)
        return self._select_period(start, end)

    def month(
        self,
        date: str | pd.Timestamp | int,
        month: int | None = None,
    ) -> pd.DataFrame:
        """Return one Danish calendar month.

        Accepted forms:
            month("2022-01")
            month("2022-01-15")
            month(2022, 1)
        """
        if month is None:
            value = self._local_timestamp(date)
            year = value.year
            month = value.month
        else:
            year = int(date)

        start = pd.Timestamp(
            year=year,
            month=int(month),
            day=1,
            tz=self.LOCAL_TIMEZONE,
        )
        end = start + pd.offsets.MonthBegin(1)
        return self._select_period(start, end)

    def by_area(self, price_area: str) -> pd.DataFrame:
        """Return all rows belonging to one price area, for example DK1."""
        price_area = price_area.strip().upper()
        return self.data.loc[self.data["PriceArea"] == price_area].copy()

    def prices(
        self,
        currency: str = "DKK",
        price_area: str | None = None,
    ) -> pd.Series:
        """Return the hourly spot-price series in DKK or EUR."""
        currency = currency.strip().upper()

        columns = {
            "DKK": "SpotPriceDKK",
            "EUR": "SpotPriceEUR",
        }
        if currency not in columns:
            raise ValueError("currency must be either 'DKK' or 'EUR'.")

        data = self.data
        if price_area is not None:
            price_area = price_area.strip().upper()
            data = data.loc[data["PriceArea"] == price_area]

        return data[columns[currency]].copy()

