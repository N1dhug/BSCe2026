from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.tools.import_data.import_input import input_data
from src.tools.import_data.import_prices import spot_price_data
from src.tools.import_data.import_weather import weather_data


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "files" / "input" / "data"
GRAPHS = ROOT / "files" / "output" / "graphs" / "data" / "weekly"
START, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-11-01")


def normalize(series, minimum=None, maximum=None):
    minimum = series.min() if minimum is None else minimum
    maximum = series.max() if maximum is None else maximum
    if not pd.notna(minimum) or not pd.notna(maximum) or maximum <= minimum: raise ValueError("Cannot normalize a constant or empty series.")
    return (series - minimum) / (maximum - minimum)


def consumption_series(data):
    cols = data.measurement_columns
    if not cols: raise ValueError("No consumption measurement columns found in input data.")
    observed = data.observed_data.loc[(data.observed_data.index >= START) & (data.observed_data.index < END), cols]
    return observed.mean(axis=1).dropna().rename("Consumption")


def price_series(data):
    prices = data.prices("DKK")
    start, end = START.tz_localize(data.LOCAL_TIMEZONE), END.tz_localize(data.LOCAL_TIMEZONE)
    prices = prices.loc[(prices.index >= start) & (prices.index < end)]
    prices.index = prices.index.tz_localize(None)
    return prices.groupby(level=0).mean().sort_index().rename("Price")


def temperature_series(data):
    temperature = data.fetch_hourly(START, END)["Temperature"]
    temperature.index = temperature.index.tz_localize(None)
    return temperature.groupby(level=0).mean().sort_index().rename("Temperature")


def complete_weeks(consumption):
    weeks = []
    for period, values in consumption.groupby(consumption.index.to_period("W-SUN")):
        start, end = period.start_time, period.end_time.normalize() + pd.Timedelta(days=1)
        if start < START or end > END: continue
        index = pd.date_range(start, end, freq="h", inclusive="left")
        values = values.sort_index()
        if len(values) == len(index) and values.index.equals(index): weeks.append((start, end))
    return weeks


def plot_week(consumption, prices, temperature, start, end):
    index = pd.date_range(start, end, freq="h", inclusive="left")
    c = consumption.reindex(index)
    p = prices.reindex(index).interpolate(method="time", limit_direction="both").ffill().bfill()
    t = temperature.reindex(index).interpolate(method="time", limit_direction="both").ffill().bfill()

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(index, c, label="Aggregate consumption")
    ax.plot(index, p, label="DK1 spot price")
    ax.plot(index, t, label="Denmark hourly mean temperature")
    for day in pd.date_range(start, end, freq="D", inclusive="left"): ax.axvline(day, linewidth=0.6, alpha=0.2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(index[0], index[-1])
    ax.set_ylabel("Season-normalized value")
    ax.set_xlabel("Time")
    ax.set_title(f"Week {start:%Y-%m-%d} to {(end - pd.Timedelta(days=1)):%Y-%m-%d}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    GRAPHS.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAPHS / f"week_{start:%Y-%m-%d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    consumption_data = input_data(DATA / "input.xlsx")
    consumption = consumption_series(consumption_data)
    prices = price_series(spot_price_data(DATA / "prices.xlsx", price_area="DK1"))
    temperature = temperature_series(weather_data())

    consumption = normalize(consumption, consumption.min(), consumption.max())
    prices = normalize(prices, prices.min(), prices.max())
    temperature = normalize(temperature, temperature.min(), temperature.max())

    weeks = complete_weeks(consumption)
    for start, end in weeks: plot_week(consumption, prices, temperature, start, end)
    print(f"Saved {len(weeks)} complete non-imputed weekly plots to {GRAPHS}")


if __name__ == "__main__": main()
