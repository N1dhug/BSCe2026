from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model.wm_3 import wm_3
from src.tools.import_data.import_input import input_data
from src.tools.import_data.import_prices import spot_price_data
from src.tools.import_data.import_weather import weather_data


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "files" / "input" / "data"
OUTPUT = ROOT / "files" / "output" / "wm_3"
START, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-11-01")
INIT = np.array([0.5, 0.5], dtype=float)
MIN_L_SLOPE = 1e-6


def normalize(values, lo=None, hi=None):
    values = pd.Series(values, index=getattr(values, "index", None), dtype=float)
    lo, hi = (values.min() if lo is None else float(lo)), (values.max() if hi is None else float(hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo: raise ValueError("Cannot normalize a constant or empty series.")
    return (values - lo) / (hi - lo)


def consumption_series(data):
    cols = data.measurement_columns
    if not cols: raise ValueError("No consumption measurement columns found.")
    values = data.observed_data.loc[(data.observed_data.index >= START) & (data.observed_data.index < END), cols]
    return values.mean(axis=1).dropna().sort_index().rename("I")


def normalized_measurements(data):
    cols = data.measurement_columns
    if not cols: raise ValueError("No consumption measurement columns found.")
    values = data.observed_data.loc[(data.observed_data.index >= START) & (data.observed_data.index < END), cols].apply(pd.to_numeric, errors="coerce")
    keep = []
    for column in cols:
        finite = values[column].dropna()
        if finite.empty: continue
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo: continue
        values[column] = (values[column] - lo) / (hi - lo)
        keep.append(column)
    if not keep: raise ValueError("No non-constant non-aggregated consumption measurements are available.")
    return values[keep].sort_index()


def price_series(data):
    values = data.prices("DKK")
    start, end = START.tz_localize(data.LOCAL_TIMEZONE), END.tz_localize(data.LOCAL_TIMEZONE)
    values = values.loc[(values.index >= start) & (values.index < end)]
    values.index = values.index.tz_localize(None)
    return values.groupby(level=0).mean().sort_index().rename("U")


def temperature_series(data):
    values = data.fetch_hourly(START, END)["Temperature"]
    values.index = values.index.tz_localize(None)
    return values.groupby(level=0).mean().sort_index().rename("Temp")


def aligned_data():
    consumption_data = input_data(DATA / "input.xlsx")
    consumption = consumption_series(consumption_data)
    measurements = normalized_measurements(consumption_data)
    prices = price_series(spot_price_data(DATA / "prices.xlsx", price_area="DK1"))
    temperature = temperature_series(weather_data())
    index = consumption.index.intersection(prices.index).intersection(temperature.index)
    frame = pd.concat((consumption.reindex(index), prices.reindex(index), temperature.reindex(index)), axis=1).dropna()
    frame["I"] = normalize(frame["I"])
    frame["U"] = normalize(frame["U"])
    frame["weekday"] = frame.index.weekday
    frame["hour"] = frame.index.hour + frame.index.minute / 60.0
    return frame, measurements.reindex(frame.index)


def fit_seasonal(frame):
    daily = frame[["I", "Temp"]].resample("D").mean().dropna()
    X = np.column_stack((np.ones(len(daily)), daily["Temp"].to_numpy(dtype=float)))
    coef = np.linalg.lstsq(X, daily["I"].to_numpy(dtype=float), rcond=None)[0]
    fitted_daily = X @ coef
    lo, hi = float(np.min(fitted_daily)), float(np.max(fitted_daily))

    def seasonal_prediction(temperature): return float(coef[0] + coef[1] * float(temperature))

    if hi - lo <= 1e-12:
        def H(_temperature): return 0.5
    else:
        def H(temperature): return float(np.clip((seasonal_prediction(temperature) - lo) / (hi - lo), 0.0, 1.0))

    return H, seasonal_prediction, {"intercept": float(coef[0]), "slope": float(coef[1]), "prediction_min": lo, "prediction_max": hi}


def positive_bounded_line(x, y, min_slope=MIN_L_SLOPE):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2: raise ValueError("At least two finite observations are required for an L_d(t) fit.")

    x_max = 23.0
    b_lo, b_hi = float(min_slope), 1.0 / x_max
    if b_lo <= 0.0 or b_lo > b_hi: raise ValueError("min_slope must lie in (0, 1/23].")

    candidates = []
    X = np.column_stack((np.ones(x.size), x))
    a_ols, b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    if a_ols >= 0.0 and b_lo <= b_ols <= b_hi and a_ols + x_max * b_ols <= 1.0: candidates.append((float(a_ols), float(b_ols)))

    b = b_lo
    a = float(np.clip(np.mean(y - b * x), 0.0, 1.0 - x_max * b))
    candidates.append((a, b))

    xx = float(x @ x)
    b = b_lo if xx <= 0.0 else float(np.clip((x @ y) / xx, b_lo, b_hi))
    candidates.append((0.0, b))

    z = x - x_max
    zz = float(z @ z)
    b = b_lo if zz <= 0.0 else float(np.clip((z @ (y - 1.0)) / zz, b_lo, b_hi))
    candidates.append((1.0 - x_max * b, b))

    def loss(pair):
        a, b = pair
        residual = y - (a + b * x)
        return float(residual @ residual)

    a, b = min(candidates, key=loss)
    return a, b, loss((a, b)), int(x.size)


def fit_daily_time_functions(raw_week, week, seasonal_prediction):
    values = raw_week.to_numpy(dtype=float)
    seasonal = np.array([seasonal_prediction(value) for value in week["Temp"].to_numpy(dtype=float)], dtype=float)
    residual = values - seasonal[:, None]
    finite = residual[np.isfinite(residual)]
    if finite.size < 2: raise ValueError("No finite non-aggregated weekly measurements are available for L_d(t).")
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo: raise ValueError("Season-adjusted non-aggregated weekly input is constant; cannot fit L_d(t).")
    target = (residual - lo) / (hi - lo)

    weekday = week.index.weekday.to_numpy(dtype=int)
    hour = week.index.hour.to_numpy(dtype=float) + week.index.minute.to_numpy(dtype=float) / 60.0
    n_measurements, best = raw_week.shape[1], None

    for shift in range(24):
        models, sse, samples = {}, 0.0, 0
        shifted_hour = np.mod(hour - shift, 24.0)
        for day in range(7):
            rows = weekday == day
            x = np.repeat(shifted_hour[rows], n_measurements)
            y = target[rows].reshape(-1)
            intercept, slope, day_sse, day_samples = positive_bounded_line(x, y)
            models[day] = (intercept, slope)
            sse += day_sse
            samples += day_samples
        score = sse / max(samples, 1)
        if best is None or score < best["score"]: best = {"shift": shift, "models": models, "sse": sse, "score": score, "samples": samples}

    shift, models = best["shift"], best["models"]

    def L(time):
        time = float(time) % 168.0
        day, hour_value = int(time // 24.0) % 7, time % 24.0
        x = (hour_value - shift) % 24.0
        intercept, slope = models[day]
        return float(intercept + slope * x)

    best["residual_min"], best["residual_max"] = lo, hi
    best["measurement_columns"] = n_measurements
    return L, best


def complete_week(frame, measurements, week_start):
    start = pd.Timestamp(week_start).normalize()
    start -= pd.Timedelta(days=start.weekday())
    end = start + pd.Timedelta(days=7)
    index = pd.date_range(start, end, freq="h", inclusive="left")
    week, raw_week = frame.reindex(index), measurements.reindex(index)
    if len(week) != 168 or week[["I", "U", "Temp"]].isna().any().any(): raise ValueError(f"{start:%Y-%m-%d} is not a complete observed 168-hour week.")
    if raw_week.notna().sum().sum() == 0: raise ValueError(f"{start:%Y-%m-%d} contains no non-aggregated input observations.")
    return start, week, raw_week


def model_input(week):
    time = week.index.weekday.to_numpy(dtype=float) * 24.0 + week.index.hour.to_numpy(dtype=float)
    return {"I": week["I"].to_numpy(dtype=float), "U": week["U"].to_numpy(dtype=float), "PE": week["U"].to_numpy(dtype=float), "Time": time, "Temp": week["Temp"].to_numpy(dtype=float)}


def save_plot(start, week, fit, daily_fit):
    state = fit["deterministic_state"]
    index = week.index
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(index, week["I"], label="Observed aggregate input I")
    axes[0].plot(index, state["I"], label="Fitted wm_3 input I")
    axes[0].set_ylabel("Normalized input")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(index, state["L"], label="L_d(t), non-aggregated fit")
    axes[1].plot(index, state["H"], label="H(T)")
    axes[1].set_ylabel("Fitted features")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(index, week["U"], label="Normalized DK1 price U")
    axes[2].plot(index, state["V"], label="V")
    axes[2].plot(index, state["R"], label="R")
    axes[2].set_ylabel("State / price")
    axes[2].set_xlabel("Time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"wm_3 weekly fit {start:%Y-%m-%d} | positive L_d slopes | optimized shift={daily_fit['shift']:02d}:00")
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"weekly_fit_{start:%Y-%m-%d}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def print_result(start, seasonal, daily_fit, fit, plot_path):
    names = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    print(f"week={start:%Y-%m-%d}")
    print(f"H(T): daily seasonal regression intercept={seasonal['intercept']:.8f}, slope={seasonal['slope']:.8f}")
    print(f"L_d(t): fit to {daily_fit['samples']} non-aggregated observations from {daily_fit['measurement_columns']} measurement columns")
    print(f"L_d(t): common shifted 24 h slice starts at {daily_fit['shift']:02d}:00, MSE={daily_fit['score']:.8f}")
    for day, name in enumerate(names):
        intercept, slope = daily_fit["models"][day]
        print(f"  {name}: L={intercept:.8f} + {slope:.8f}*((hour-{daily_fit['shift']}) mod 24)")
    print(f"alpha={np.array2string(fit['alpha'], precision=8)}")
    print(f"beta={np.array2string(fit['beta'], precision=8)}")
    print(f"kappa_R={fit['kappa_R']:.8f}")
    print(f"sigma_0={fit['sigma_0']:.8f}, sigma_O={fit['sigma_O']:.8f}")
    print(f"initial condition={np.array2string(fit['init_cond'], precision=8)}")
    print(f"deterministic loss={fit['deterministic_loss']:.10f}")
    print(f"plot={plot_path}")


def main():
    parser = ArgumentParser(description="Fit wm_3 to one weekly response using seasonal H(T) and positive weekday-specific shifted linear L_d(t) fitted from non-aggregated input data.")
    parser.add_argument("--week", default="2022-01-03", help="Any date in the requested Monday-Sunday week.")
    parser.add_argument("--grid-points", type=int, default=4)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    frame, measurements = aligned_data()
    H, seasonal_prediction, seasonal = fit_seasonal(frame)
    start, week, raw_week = complete_week(frame, measurements, args.week)
    L, daily_fit = fit_daily_time_functions(raw_week, week, seasonal_prediction)
    data = model_input(week)

    model = wm_3((0.0, 167.0), 168, L, H)
    fit = model.fit_params(
        data, INIT,
        grid_points=args.grid_points,
        refinements=args.refinements,
        fit_initial_state=False,
        fit_diffusion=True,
        verbose=True,
        progress_every=args.progress_every,
    )
    plot_path = save_plot(start, week, fit, daily_fit)
    print_result(start, seasonal, daily_fit, fit, plot_path)


if __name__ == "__main__": main()
