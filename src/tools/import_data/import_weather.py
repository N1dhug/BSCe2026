from __future__ import annotations

import json
from pathlib import Path
from time import sleep
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = ROOT / "files" / "input" / "data" / "weather_cache"


class weather_data:
    """Download and persist DMI country-level mean temperature for Denmark."""

    BASE_URL = "https://opendataapi.dmi.dk/v2/climateData/collections/countryValue/items"
    LOCAL_TIMEZONE = "Europe/Copenhagen"
    PARAMETER = "mean_temp"
    PAGE_LIMIT = 300_000

    def __init__(self, cache_dir=DEFAULT_CACHE, retries=3, retry_delay=2.0):
        self.data: pd.DataFrame | None = None
        self.cache_dir = Path(cache_dir)
        self.retries = max(1, int(retries))
        self.retry_delay = max(0.0, float(retry_delay))

    @staticmethod
    def _utc(value):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None: timestamp = timestamp.tz_localize(weather_data.LOCAL_TIMEZONE)
        return timestamp.tz_convert("UTC")

    @staticmethod
    def _iso(timestamp): return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _cache_path(self, start, end, resolution):
        start_key = start.strftime("%Y%m%dT%H%M%SZ")
        end_key = end.strftime("%Y%m%dT%H%M%SZ")
        return self.cache_dir / f"dmi_{self.PARAMETER}_{resolution}_{start_key}_{end_key}.csv"

    def _read_cache(self, path):
        if not path.exists(): return None
        data = pd.read_csv(path)
        if "DatetimeUTC" not in data or "Temperature" not in data: return None
        index = pd.to_datetime(data["DatetimeUTC"], utc=True, errors="coerce")
        values = pd.to_numeric(data["Temperature"], errors="coerce")
        valid = index.notna() & values.notna()
        if not valid.any(): return None
        frame = pd.DataFrame({"Temperature": values[valid].to_numpy(dtype=float)}, index=index[valid])
        frame.index = frame.index.tz_convert(self.LOCAL_TIMEZONE)
        frame.index.name = "Datetime"
        return frame.sort_index()

    def _write_cache(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        cached = data.copy()
        cached.index = cached.index.tz_convert("UTC")
        cached.index.name = "DatetimeUTC"
        cached.to_csv(path)

    def _request(self, start, end, resolution, offset=0):
        params = {
            "parameterId": self.PARAMETER,
            "timeResolution": resolution,
            "datetime": f"{self._iso(start)}/{self._iso(end - pd.Timedelta(microseconds=1))}",
            "limit": self.PAGE_LIMIT,
            "offset": offset,
        }
        request = Request(f"{self.BASE_URL}?{urlencode(params)}", headers={"Accept": "application/geo+json"})
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                with urlopen(request, timeout=120) as response: return json.load(response)
            except (URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self.retries and self.retry_delay > 0.0: sleep(self.retry_delay * attempt)
        raise RuntimeError(
            f"Could not reach the DMI climate API after {self.retries} attempts. "
            "Check the internet/DNS connection. Once a request succeeds, the weather series is cached locally for later offline runs."
        ) from last_error

    def _fetch(self, start, end, resolution):
        start, end = self._utc(start), self._utc(end)
        if end <= start: raise ValueError("end must be later than start.")

        cache_path = self._cache_path(start, end, resolution)
        cached = self._read_cache(cache_path)
        if cached is not None:
            self.data = cached
            return cached.copy()

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
        data["Datetime"] = local.dt.normalize() if resolution == "day" else data["DatetimeUTC"].dt.floor("h").dt.tz_convert(self.LOCAL_TIMEZONE)
        data = data.groupby("Datetime", as_index=True)["Temperature"].mean().to_frame().sort_index()
        freq = "D" if resolution == "day" else "h"
        first = start.tz_convert(self.LOCAL_TIMEZONE).normalize() if resolution == "day" else start.floor("h").tz_convert(self.LOCAL_TIMEZONE)
        last = end.tz_convert(self.LOCAL_TIMEZONE).normalize() if resolution == "day" else end.floor("h").tz_convert(self.LOCAL_TIMEZONE)
        data = data.reindex(pd.date_range(first, last, freq=freq, inclusive="left"))
        data.index.name = "Datetime"
        data["Temperature"] = data["Temperature"].interpolate(method="time", limit_direction="both").ffill().bfill()
        self._write_cache(cache_path, data)
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
