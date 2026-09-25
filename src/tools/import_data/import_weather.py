from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


class weather_data:
    """Download DMI country-level mean temperature for Denmark."""

    BASE_URL = "https://opendataapi.dmi.dk/v2/climateData/collections/countryValue/items"
    LOCAL_TIMEZONE = "Europe/Copenhagen"
    PARAMETER = "mean_temp"
    PAGE_LIMIT = 300_000

    def __init__(self): self.data: pd.DataFrame | None = None

    @staticmethod
    def _utc(value):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None: timestamp = timestamp.tz_localize(weather_data.LOCAL_TIMEZONE)
        return timestamp.tz_convert("UTC")

    @staticmethod
    def _iso(timestamp): return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _request(self, start, end, resolution, offset=0):
        params = {
            "parameterId": self.PARAMETER,
            "timeResolution": resolution,
            "datetime": f"{self._iso(start)}/{self._iso(end - pd.Timedelta(microseconds=1))}",
            "limit": self.PAGE_LIMIT,
            "offset": offset,
        }
        request = Request(f"{self.BASE_URL}?{urlencode(params)}", headers={"Accept": "application/geo+json"})
        with urlopen(request, timeout=120) as response: return json.load(response)

    def _fetch(self, start, end, resolution):
        start, end = self._utc(start), self._utc(end)
        if end <= start: raise ValueError("end must be later than start.")

        rows, offset = [], 0
        while True:
            features = self._request(start, end, resolution, offset).get("features", [])
            for feature in features:
                props = feature.get("properties", {})
                rows.append({
                    "DatetimeUTC": pd.to_datetime(props.get("from"), utc=True, errors="coerce"),
                    "Temperature": pd.to_numeric(props.get("value"), errors="coerce"),
                })
            if len(features) < self.PAGE_LIMIT: break
            offset += len(features)

        data = pd.DataFrame(rows).dropna(subset=["DatetimeUTC", "Temperature"])
        if data.empty: raise ValueError(f"DMI returned no Danish country-level {resolution} temperature data for the requested period.")

        local = data["DatetimeUTC"].dt.tz_convert(self.LOCAL_TIMEZONE)
        data["Datetime"] = local.dt.normalize() if resolution == "day" else local.dt.floor("h")
        data = data.groupby("Datetime", as_index=True)["Temperature"].mean().to_frame().sort_index()
        freq = "D" if resolution == "day" else "h"
        first = start.tz_convert(self.LOCAL_TIMEZONE).normalize() if resolution == "day" else start.tz_convert(self.LOCAL_TIMEZONE).floor("h")
        last = end.tz_convert(self.LOCAL_TIMEZONE).normalize() if resolution == "day" else end.tz_convert(self.LOCAL_TIMEZONE).floor("h")
        data = data.reindex(pd.date_range(first, last, freq=freq, inclusive="left"))
        data.index.name = "Datetime"
        data["Temperature"] = data["Temperature"].interpolate(method="time", limit_direction="both").ffill().bfill()
        self.data = data
        return data.copy()

    def fetch(self, start, end):
        """Fetch daily country-level mean temperature for Denmark."""
        return self._fetch(start, end, "day")

    def fetch_hourly(self, start, end):
        """Fetch hourly country-level mean temperature for Denmark."""
        return self._fetch(start, end, "hour")

    def year(self, year):
        start = pd.Timestamp(year=int(year), month=1, day=1, tz=self.LOCAL_TIMEZONE)
        return self.fetch(start, start + pd.DateOffset(years=1))
