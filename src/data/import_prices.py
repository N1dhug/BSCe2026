from __future__ import annotations

from pathlib import Path

import pandas as pd


class SpotPriceData:
    """Import/query hourly spot-price data."""

    LOCAL_TIMEZONE = "Europe/Copenhagen"
    REQUIRED_COLUMNS = ("HourDK", "HourUTC", "PriceArea", "SpotPriceDKK", "SpotPriceEUR")
    PRICE_COLUMNS = ("SpotPriceDKK", "SpotPriceEUR")

    def __init__(self, path: str | Path, sheet_name: str | int = 0, price_area: str | None = None):
        self.path = Path(path)
        self.sheet_name = sheet_name
        self.price_area = price_area.upper() if price_area else None
        self.data = self._load()

    def _read(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Spot-price file not found: {self.path}")

        suffix = self.path.suffix.lower()
        if suffix in {".xlsx", ".xlsm"}:
            return pd.read_excel(self.path, sheet_name=self.sheet_name, engine="openpyxl")
        if suffix == ".xls":
            return pd.read_excel(self.path, sheet_name=self.sheet_name)
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(self.path, sep=None, engine="python")
        raise ValueError("Expected .xlsx, .xlsm, .xls, .csv or .txt.")

    def _load(self):
        data = self._read()
        data.columns = data.columns.astype(str).str.strip()

        missing = [c for c in self.REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        data = data[list(self.REQUIRED_COLUMNS)].copy()
        data["_HourUTC"] = pd.to_datetime(data["HourUTC"], errors="coerce", utc=True)

        if data["_HourUTC"].isna().any():
            raise ValueError(f"Could not parse HourUTC for {int(data['_HourUTC'].isna().sum())} row(s).")

        data["PriceArea"] = data["PriceArea"].astype(str).str.strip().str.upper()
        data[list(self.PRICE_COLUMNS)] = data[list(self.PRICE_COLUMNS)].apply(pd.to_numeric, errors="coerce")

        if self.price_area:
            data = data.loc[data["PriceArea"] == self.price_area].copy()
            if data.empty:
                raise ValueError(f"Price area '{self.price_area}' was not found.")

        data = self._impute_missing_hours(data)
        data["HourUTC"] = data["_HourUTC"]
        data["HourDK"] = data["_HourUTC"].dt.tz_convert(self.LOCAL_TIMEZONE)
        data = data.drop(columns="_HourUTC").set_index("HourDK", drop=False)
        data.index.name = "Datetime"
        return data.sort_index()

    def _impute_missing_hours(self, data):
        groups = []

        for area, area_data in data.groupby("PriceArea", sort=False):
            area_data = area_data.set_index("_HourUTC").sort_index()

            if area_data.index.has_duplicates:
                duplicates = area_data.index[area_data.index.duplicated()].unique()
                raise ValueError(f"Duplicate HourUTC values for '{area}': {list(duplicates[:5])}")

            index = pd.date_range(area_data.index.min(), area_data.index.max(), freq="h")
            area_data = area_data.reindex(index)
            area_data.index.name = "_HourUTC"
            area_data["PriceArea"] = area
            area_data[list(self.PRICE_COLUMNS)] = (
                area_data[list(self.PRICE_COLUMNS)]
                .interpolate(method="time", limit_direction="both")
                .ffill()
                .bfill()
            )
            groups.append(area_data.reset_index())

        return pd.concat(groups, ignore_index=True) if groups else data

    def _local_timestamp(self, value):
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize(self.LOCAL_TIMEZONE) if timestamp.tzinfo is None else timestamp.tz_convert(self.LOCAL_TIMEZONE)

    def _select_period(self, start, end):
        start, end = self._local_timestamp(start), self._local_timestamp(end)
        return self.data.loc[(self.data.index >= start) & (self.data.index < end)].copy()

    def day(self, date):
        start = self._local_timestamp(date).normalize()
        return self._select_period(start, start + pd.DateOffset(days=1))

    def week(self, date):
        date = self._local_timestamp(date).normalize()
        start = date - pd.DateOffset(days=date.weekday())
        return self._select_period(start, start + pd.DateOffset(days=7))

    def month(self, date, month=None):
        if month is None:
            date = self._local_timestamp(date)
            year, month = date.year, date.month
        else:
            year = int(date)

        start = pd.Timestamp(year=year, month=int(month), day=1, tz=self.LOCAL_TIMEZONE)
        return self._select_period(start, start + pd.offsets.MonthBegin(1))

    def by_area(self, price_area):
        price_area = price_area.strip().upper()
        return self.data.loc[self.data["PriceArea"] == price_area].copy()

    def prices(self, currency="DKK", price_area=None):
        currency = currency.strip().upper()
        columns = {"DKK": "SpotPriceDKK", "EUR": "SpotPriceEUR"}
        if currency not in columns:
            raise ValueError("currency must be 'DKK' or 'EUR'.")

        data = self.data if price_area is None else self.data.loc[self.data["PriceArea"] == price_area.strip().upper()]
        return data[columns[currency]].copy()
