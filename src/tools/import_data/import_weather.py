from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


class weather_data:
    """Download and aggregate hourly DMI climate data for the DK1 area."""

    BASE_URL = "https://opendataapi.dmi.dk/v2/climateData/collections/10kmGridValue/items"
    LOCAL_TIMEZONE = "Europe/Copenhagen"
    PARAMETERS = {"mean_temp": "Temperature", "mean_wind_speed": "WindSpeed"}
    QUERY_BBOX = (7.5, 54.4, 11.0, 57.9)
    PAGE_LIMIT = 300_000

    def __init__(self): self.data: pd.DataFrame | None = None

    @staticmethod
    def _utc(value):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None: timestamp = timestamp.tz_localize(weather_data.LOCAL_TIMEZONE)
        return timestamp.tz_convert("UTC")

    @staticmethod
    def _iso(timestamp): return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _coordinates(value):
        if isinstance(value, (list, tuple)):
            if len(value) >= 2 and all(isinstance(x, (int, float)) for x in value[:2]): yield float(value[0]), float(value[1])
            else:
                for item in value: yield from weather_data._coordinates(item)

    @classmethod
    def _cell_center(cls, geometry):
        coords = list(cls._coordinates((geometry or {}).get("coordinates", [])))
        if not coords: return np.nan, np.nan
        return (min(x for x, _ in coords) + max(x for x, _ in coords)) / 2, (min(y for _, y in coords) + max(y for _, y in coords)) / 2

    @staticmethod
    def _in_dk1(lon, lat):
        """Approximate DK1 as Jutland, Funen and adjacent western islands."""
        if pd.isna(lon) or pd.isna(lat): return False
        return (7.5 <= lon <= 10.75 and 54.4 <= lat <= 57.9) or (9.65 <= lon <= 10.95 and 54.7 <= lat <= 55.75)

    @staticmethod
    def _month_windows(start, end):
        current = start
        while current < end:
            stop = min((current.tz_convert(None) + pd.offsets.MonthBegin(1)).tz_localize("UTC"), end)
            yield current, stop
            current = stop

    def _request(self, parameter, start, end, offset=0):
        params = {
            "parameterId": parameter,
            "timeResolution": "hour",
            "datetime": f"{self._iso(start)}/{self._iso(end - pd.Timedelta(microseconds=1))}",
            "bbox": ",".join(map(str, self.QUERY_BBOX)),
            "limit": self.PAGE_LIMIT,
            "offset": offset,
        }
        request = Request(f"{self.BASE_URL}?{urlencode(params)}", headers={"Accept": "application/geo+json"})
        with urlopen(request, timeout=120) as response: return json.load(response)

    def _fetch_parameter(self, parameter, start, end):
        rows = []
        for window_start, window_end in self._month_windows(start, end):
            offset = 0
            while True:
                features = self._request(parameter, window_start, window_end, offset).get("features", [])
                for feature in features:
                    props = feature.get("properties", {})
                    lon, lat = self._cell_center(feature.get("geometry"))
                    if not self._in_dk1(lon, lat): continue
                    rows.append({
                        "DatetimeUTC": pd.to_datetime(props.get("from"), utc=True, errors="coerce"),
                        "CellId": props.get("cellId"),
                        "Parameter": parameter,
                        "Value": pd.to_numeric(props.get("value"), errors="coerce"),
                        "Longitude": lon,
                        "Latitude": lat,
                        "QCStatus": props.get("qcStatus"),
                    })
                if len(features) < self.PAGE_LIMIT: break
                offset += len(features)
        return pd.DataFrame(rows)

    def fetch(self, start, end):
        """Fetch hourly temperature/wind data and return the DK1 area mean."""
        start, end = self._utc(start), self._utc(end)
        if end <= start: raise ValueError("end must be later than start.")

        raw = pd.concat([self._fetch_parameter(parameter, start, end) for parameter in self.PARAMETERS], ignore_index=True)
        if raw.empty: raise ValueError("DMI returned no DK1 weather data for the requested period.")
        raw = raw.dropna(subset=["DatetimeUTC", "Value"])

        data = raw.groupby(["DatetimeUTC", "Parameter"], as_index=False)["Value"].mean().pivot(index="DatetimeUTC", columns="Parameter", values="Value").rename(columns=self.PARAMETERS)
        data = data.reindex(pd.date_range(start, end, freq="h", inclusive="left", tz="UTC"))
        data.index.name = "DatetimeUTC"
        data = data.interpolate(method="time", limit_direction="both").ffill().bfill()
        data.index, data.index.name = data.index.tz_convert(self.LOCAL_TIMEZONE), "Datetime"
        self.data = data
        return data.copy()

    def year(self, year):
        start = pd.Timestamp(year=int(year), month=1, day=1, tz=self.LOCAL_TIMEZONE)
        return self.fetch(start, start + pd.DateOffset(years=1))

    @staticmethod
    def chill_factor(data):
        """Compute Environment Canada wind chill from degC and m/s."""
        if not {"Temperature", "WindSpeed"}.issubset(data.columns): raise ValueError("data must contain 'Temperature' and 'WindSpeed'.")

        temp = pd.to_numeric(data["Temperature"], errors="coerce")
        wind = pd.to_numeric(data["WindSpeed"], errors="coerce").clip(lower=0) * 3.6
        chill = temp.copy()
        cold, normal_wind = temp <= 0, (temp <= 0) & (wind >= 5)
        low_wind = cold & (wind > 0) & (wind < 5)
        v016 = wind.loc[normal_wind].pow(0.16)
        chill.loc[normal_wind] = 13.12 + 0.6215 * temp.loc[normal_wind] - 11.37 * v016 + 0.3965 * temp.loc[normal_wind] * v016
        chill.loc[low_wind] = temp.loc[low_wind] + ((-1.59 + 0.1345 * temp.loc[low_wind]) / 5) * wind.loc[low_wind]
        chill.name = "ChillFactor"
        return chill

    def with_chill_factor(self, data=None):
        data = self.data if data is None else data
        if data is None: raise ValueError("No weather data loaded. Call fetch() or year() first.")
        result = data.copy()
        result["ChillFactor"] = self.chill_factor(result)
        return result
