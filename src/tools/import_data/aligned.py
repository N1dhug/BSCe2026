"""Shared hourly alignment and normalization for consumption, DK1 prices and weather.

Identical normalization contracts to the original wm_3 seasonal data loader.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.import_data.import_input import input_data
from src.tools.import_data.import_prices import spot_price_data
from src.tools.import_data.import_weather import weather_data

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "files" / "input" / "data"
START, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-11-01")

def normalize(values, lo=None, hi=None, domain=(0.0, 1.0)):
    values = pd.Series(values, index=getattr(values, "index", None), dtype=float)
    lo, hi = (values.min() if lo is None else float(lo)), (values.max() if hi is None else float(hi))
    d_lo, d_hi = map(float, domain)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo: raise ValueError("Cannot normalize a constant or empty series.")
    if not np.isfinite(d_lo) or not np.isfinite(d_hi) or d_hi <= d_lo: raise ValueError("domain must satisfy lower < upper.")
    return d_lo + (d_hi - d_lo) * (values - lo) / (hi - lo)

def tagged_measurement_columns(data, tag=None):
    cols = data.measurement_columns
    if tag is not None:
        tag = str(tag).strip().upper()
        cols = [column for column in cols if data.measurement_tag(column).upper() == tag]
    if not cols: raise ValueError(f"No consumption measurement columns found{'' if tag is None else f' for tag {tag}'}." )
    return cols

def consumption_series(data, tag=None):
    cols = tagged_measurement_columns(data, tag)
    values = data.observed_data.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1).dropna().sort_index().rename("I")

def raw_measurements(data, tag=None):
    cols = tagged_measurement_columns(data, tag)
    values = data.observed_data.loc[:, cols].apply(pd.to_numeric, errors="coerce").sort_index()
    keep = [column for column in cols if values[column].notna().any()]
    if not keep: raise ValueError("No non-aggregated consumption measurements are available.")
    return values[keep]

def price_series(data):
    values = data.prices("DKK")
    values.index = values.index.tz_localize(None)
    return values.groupby(level=0).mean().sort_index().rename("U")

def temperature_series(data, start, end):
    values = data.fetch_hourly(start, end)["Temperature"]
    values.index = values.index.tz_localize(None)
    return values.groupby(level=0).mean().sort_index().rename("Temp")

def aligned_data(tag=None, price_domain=(0.0, 1.0), price_year=None):
    consumption_data = input_data(DATA / "input.xlsx")
    consumption = consumption_series(consumption_data, tag)
    measurements = raw_measurements(consumption_data, tag)
    prices = price_series(spot_price_data(DATA / "prices.xlsx", price_area="DK1"))

    signal_start = max(consumption.index.min(), prices.index.min(), START)
    signal_end = min(consumption.index.max(), prices.index.max(), END + pd.Timedelta(days=1)) + pd.Timedelta(hours=1)
    temperature = temperature_series(weather_data(), signal_start, signal_end)

    index = consumption.index.intersection(prices.index).intersection(temperature.index)
    frame = pd.concat((consumption.reindex(index), prices.reindex(index), temperature.reindex(index)), axis=1).dropna()
    season = frame.loc[(frame.index >= START) & (frame.index < END)].copy()
    if season.empty: raise ValueError("No aligned Jan-Oct data is available for seasonal fitting.")

    i_lo, i_hi = float(season["I"].min()), float(season["I"].max())
    if price_year is None:
        price_reference = season["U"]
    else:
        year_start, year_end = pd.Timestamp(f"{int(price_year)}-01-01"), pd.Timestamp(f"{int(price_year) + 1}-01-01")
        price_reference = prices.loc[(prices.index >= year_start) & (prices.index < year_end)]
        if price_reference.empty: raise ValueError(f"No spot prices available for price calibration year {price_year}.")
    u_lo, u_hi = float(price_reference.min()), float(price_reference.max())
    frame["I"] = normalize(frame["I"], i_lo, i_hi)
    frame["U"] = normalize(frame["U"], u_lo, u_hi, domain=price_domain)
    frame.attrs["price_min"], frame.attrs["price_max"] = u_lo, u_hi
    frame.attrs["price_domain"], frame.attrs["price_year"] = tuple(map(float, price_domain)), price_year
    return frame, measurements
