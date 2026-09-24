from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.tools.import_data.import_input import input_data
from src.tools.import_data.import_prices import spot_price_data
from src.tools.import_data.import_weather import weather_data


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "files" / "input" / "data"
START, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-01")
PERIODS = {"day": "D", "week": "W-MON", "month": "MS"}


def consumption_series(data):
    cols = data.measurement_columns
    if not cols: raise ValueError("No consumption measurement columns found in input data.")
    period = data.data.loc[(data.data.index >= START) & (data.data.index < END), cols]
    return period.mean(axis=1).rename("AverageConsumerConsumption")


def price_series(data):
    prices = data.prices("DKK")
    start, end = START.tz_localize(data.LOCAL_TIMEZONE), END.tz_localize(data.LOCAL_TIMEZONE)
    return prices.loc[(prices.index >= start) & (prices.index < end)].rename("SpotPriceDKK")


def chill_series(data):
    weather = data.fetch(START, END)
    return data.chill_factor(weather).rename("ChillFactor")


def aggregate(series, freq, total=False):
    return series.resample(freq).sum() if total else series.resample(freq).mean()


def plot_period(consumption, prices, chill, label, freq):
    c, p, h = aggregate(consumption, freq, True), aggregate(prices, freq), aggregate(chill, freq)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"DK1 data overview — {label} aggregation, Jan–Nov 2022")
    axes[0].plot(c.index, c.values)
    axes[0].set_ylabel("Energy consumption")
    axes[0].set_title("Average consumer energy consumption (input)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(p.index, p.values)
    axes[1].set_ylabel("DKK/MWh")
    axes[1].set_title("DK1 spot price")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(h.index, h.values)
    axes[2].set_ylabel("Chill factor [°C]")
    axes[2].set_xlabel("Date")
    axes[2].set_title("Average DK1 chill factor")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    consumption = consumption_series(input_data(DATA / "input.xlsx"))
    prices = price_series(spot_price_data(DATA / "prices.xlsx", price_area="DK1"))
    chill = chill_series(weather_data())
    for label, freq in PERIODS.items(): plot_period(consumption, prices, chill, label, freq)
    plt.show()


if __name__ == "__main__": main()
