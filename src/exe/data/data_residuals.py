from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tools.import_data.import_input import input_data
from src.tools.import_data.import_prices import spot_price_data
from src.tools.import_data.import_weather import weather_data


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "files" / "input" / "data"
GRAPHS = ROOT / "files" / "output" / "graphs" / "data"
START, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-11-01")


def daily_consumption(data):
    cols = data.measurement_columns
    if not cols: raise ValueError("No consumption measurement columns found in input data.")
    period = data.data.loc[(data.data.index >= START) & (data.data.index < END), cols]
    return period.mean(axis=1).resample("D").sum().rename("Consumption")


def daily_prices(data):
    prices = data.prices("DKK")
    start, end = START.tz_localize(data.LOCAL_TIMEZONE), END.tz_localize(data.LOCAL_TIMEZONE)
    prices = prices.loc[(prices.index >= start) & (prices.index < end)].resample("D").mean()
    prices.index = prices.index.tz_localize(None)
    return prices.rename("Price")


def daily_temperature(data):
    temperature = data.fetch(START, END)["Temperature"].resample("D").mean()
    temperature.index = temperature.index.tz_localize(None)
    return temperature.rename("Temperature")


def linear_fit(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    X = np.column_stack((np.ones(mask.sum()), x[mask]))
    beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
    fitted = beta[0] + beta[1] * x
    ss_res = np.sum((y[mask] - (beta[0] + beta[1] * x[mask])) ** 2)
    ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return fitted, beta, r2


def plot_line(ax, x, fitted):
    order = np.argsort(np.asarray(x))
    ax.plot(np.asarray(x)[order], np.asarray(fitted)[order])


def main():
    consumption = daily_consumption(input_data(DATA / "input.xlsx"))
    prices = daily_prices(spot_price_data(DATA / "prices.xlsx", price_area="DK1"))
    temperature = daily_temperature(weather_data())
    frame = pd.concat([consumption, prices, temperature], axis=1, join="inner").dropna()

    fit_temp, beta_temp, r2_temp = linear_fit(frame["Temperature"], frame["Consumption"])
    frame["TemperatureFit"] = fit_temp
    frame["Residual"] = frame["Consumption"] - frame["TemperatureFit"]

    fit_price, beta_price, r2_price = linear_fit(frame["Price"], frame["Consumption"])
    fit_residual, beta_residual, r2_residual = linear_fit(frame["Price"], frame["Residual"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    fig.suptitle("Daily consumption relationships, Jan–Oct 2022")

    axes[0].scatter(frame["Temperature"], frame["Consumption"], s=18, alpha=0.7)
    plot_line(axes[0], frame["Temperature"], fit_temp)
    axes[0].set_title(f"Consumption vs. temperature — slope={beta_temp[1]:.3f}, R²={r2_temp:.3f}")
    axes[0].set_xlabel("Average temperature [°C]")
    axes[0].set_ylabel("Daily average-consumer consumption")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(frame["Price"], frame["Consumption"], s=18, alpha=0.7)
    plot_line(axes[1], frame["Price"], fit_price)
    axes[1].set_title(f"Consumption vs. DK1 spot price — slope={beta_price[1]:.6f}, R²={r2_price:.3f}")
    axes[1].set_xlabel("Average DK1 spot price [DKK/MWh]")
    axes[1].set_ylabel("Daily average-consumer consumption")
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(frame["Price"], frame["Residual"], s=18, alpha=0.7)
    plot_line(axes[2], frame["Price"], fit_residual)
    axes[2].axhline(0, linewidth=1)
    axes[2].set_title(f"Temperature-adjusted consumption residual vs. price — slope={beta_residual[1]:.6f}, R²={r2_residual:.3f}")
    axes[2].set_xlabel("Average DK1 spot price [DKK/MWh]")
    axes[2].set_ylabel("Consumption residual")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    GRAPHS.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAPHS / "data_residuals.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__": main()
