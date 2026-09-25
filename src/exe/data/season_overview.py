from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.tools.import_data.import_input import input_data
from src.tools.import_data.import_prices import spot_price_data
from src.tools.import_data.import_weather import weather_data


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "files" / "input" / "data"
GRAPHS = ROOT / "files" / "output" / "graphs" / "data"
START, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-11-01")
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


def temperature_series(data):
    return data.fetch(START, END)["Temperature"].rename("Temperature")


def aggregate(series, freq, total=False):
    return series.resample(freq).sum() if total else series.resample(freq).mean()


def plot_period(consumption, prices, temperature, label, freq):
    c, p, t = aggregate(consumption, freq, True), aggregate(prices, freq), aggregate(temperature, freq)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Data overview — {label} aggregation, Jan–Oct 2022")
    axes[0].plot(c.index, c.values)
    axes[0].set_ylabel("Energy consumption")
    axes[0].set_title("Average consumer energy consumption (input)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(p.index, p.values)
    axes[1].set_ylabel("DKK/MWh")
    axes[1].set_title("DK1 spot price")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t.index, t.values)
    axes[2].set_ylabel("Temperature [°C]")
    axes[2].set_xlabel("Date")
    axes[2].set_title("Average temperature — Denmark")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    GRAPHS.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAPHS / f"data_overview_{label}.png", dpi=300, bbox_inches="tight")
    return fig


def main():
    consumption = consumption_series(input_data(DATA / "input.xlsx"))
    prices = price_series(spot_price_data(DATA / "prices.xlsx", price_area="DK1"))
    temperature = temperature_series(weather_data())
    for label, freq in PERIODS.items(): plot_period(consumption, prices, temperature, label, freq)
    plt.show()


if __name__ == "__main__": main()
