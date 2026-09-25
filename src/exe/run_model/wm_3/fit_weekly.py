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
MIN_L_SLOPE = 1e-12


def normalize(values, lo=None, hi=None):
    values = pd.Series(values, index=getattr(values, "index", None), dtype=float)
    lo, hi = (values.min() if lo is None else float(lo)), (values.max() if hi is None else float(hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo: raise ValueError("Cannot normalize a constant or empty series.")
    return (values - lo) / (hi - lo)


def consumption_series(data):
    cols = data.measurement_columns
    if not cols: raise ValueError("No consumption measurement columns found.")
    values = data.observed_data.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1).dropna().sort_index().rename("I")


def raw_measurements(data):
    cols = data.measurement_columns
    if not cols: raise ValueError("No consumption measurement columns found.")
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


def aligned_data():
    consumption_data = input_data(DATA / "input.xlsx")
    consumption = consumption_series(consumption_data)
    measurements = raw_measurements(consumption_data)
    prices = price_series(spot_price_data(DATA / "prices.xlsx", price_area="DK1"))

    signal_start = max(consumption.index.min(), prices.index.min(), START)
    signal_end = min(consumption.index.max(), prices.index.max(), END + pd.Timedelta(days=1)) + pd.Timedelta(hours=1)
    temperature = temperature_series(weather_data(), signal_start, signal_end)

    index = consumption.index.intersection(prices.index).intersection(temperature.index)
    frame = pd.concat((consumption.reindex(index), prices.reindex(index), temperature.reindex(index)), axis=1).dropna()
    season = frame.loc[(frame.index >= START) & (frame.index < END)].copy()
    if season.empty: raise ValueError("No aligned Jan-Oct data is available for seasonal fitting.")

    i_lo, i_hi = float(season["I"].min()), float(season["I"].max())
    u_lo, u_hi = float(season["U"].min()), float(season["U"].max())
    frame["I"] = normalize(frame["I"], i_lo, i_hi)
    frame["U"] = normalize(frame["U"], u_lo, u_hi)
    return frame, measurements


def fit_seasonal(frame):
    period = frame.loc[(frame.index >= START) & (frame.index < END), ["I", "Temp"]]
    daily = period.resample("D").mean().dropna()
    X = np.column_stack((np.ones(len(daily)), daily["Temp"].to_numpy(dtype=float)))
    coef = np.linalg.lstsq(X, daily["I"].to_numpy(dtype=float), rcond=None)[0]
    fitted = X @ coef
    lo, hi = float(np.min(fitted)), float(np.max(fitted))

    if hi - lo <= 1e-12:
        def H(_temperature): return 0.5
    else:
        def H(temperature): return float(np.clip((coef[0] + coef[1] * float(temperature) - lo) / (hi - lo), 0.0, 1.0))

    return H, {"intercept": float(coef[0]), "slope": float(coef[1]), "prediction_min": lo, "prediction_max": hi}


def positive_line(x, y, min_slope=MIN_L_SLOPE):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2: raise ValueError("At least two finite observations are required for an L_d(t) fit.")

    b_min = float(min_slope)
    if b_min <= 0.0: raise ValueError("min_slope must be positive.")

    candidates = []
    X = np.column_stack((np.ones(x.size), x))
    a_ols, b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    if a_ols >= 0.0 and b_ols >= b_min: candidates.append((float(a_ols), float(b_ols)))

    b = b_min
    a = max(0.0, float(np.mean(y - b * x)))
    candidates.append((a, b))

    xx = float(x @ x)
    b = b_min if xx <= 0.0 else max(b_min, float((x @ y) / xx))
    candidates.append((0.0, b))

    def loss(pair):
        a, b = pair
        residual = y - (a + b * x)
        return float(residual @ residual)

    a, b = min(candidates, key=loss)
    return a, b, loss((a, b)), int(x.size)


def week_start(value):
    start = pd.Timestamp(value).normalize()
    return start - pd.Timedelta(days=start.weekday())


def fit_daily_time_functions(measurements, start):
    n_measurements, best = measurements.shape[1], None

    for shift in range(24):
        models, sse, samples = {}, 0.0, 0
        shifted_start = start + pd.Timedelta(hours=shift)

        for day in range(7):
            slice_start = shifted_start + pd.Timedelta(days=day)
            index = pd.date_range(slice_start, slice_start + pd.Timedelta(days=1), freq="h", inclusive="left")
            values = measurements.reindex(index).to_numpy(dtype=float)
            x = np.repeat(np.arange(24, dtype=float), n_measurements)
            y = values.reshape(-1)
            intercept, slope, day_sse, day_samples = positive_line(x, y)
            models[day] = (intercept, slope)
            sse += day_sse
            samples += day_samples

        score = sse / max(samples, 1)
        if best is None or score < best["score"]:
            best = {"shift": shift, "raw_models": models, "sse": sse, "score": score, "samples": samples}

    raw_models = best["raw_models"]
    predictions = np.concatenate([
        raw_models[day][0] + raw_models[day][1] * np.arange(24, dtype=float)
        for day in range(7)
    ])
    lo, hi = float(np.min(predictions)), float(np.max(predictions))
    if hi <= lo: raise ValueError("The raw L_d(t) line fits are constant and cannot be normalized for wm_3.")

    models = {
        day: ((raw_models[day][0] - lo) / (hi - lo), raw_models[day][1] / (hi - lo))
        for day in range(7)
    }

    def L(time):
        time = float(time)
        if time < 0.0: time = 0.0
        if time > 167.0: time = 167.0
        day, x = min(int(time // 24.0), 6), time % 24.0
        intercept, slope = models[day]
        return float(np.clip(intercept + slope * x, 0.0, 1.0))

    best["models"] = models
    best["raw_prediction_min"], best["raw_prediction_max"] = lo, hi
    best["measurement_columns"] = n_measurements
    best["slice_start"] = start + pd.Timedelta(hours=best["shift"])
    best["slice_end"] = best["slice_start"] + pd.Timedelta(days=7)
    return L, best


def shifted_week(frame, start, shift):
    shifted_start = start + pd.Timedelta(hours=int(shift))
    shifted_end = shifted_start + pd.Timedelta(days=7)
    index = pd.date_range(shifted_start, shifted_end, freq="h", inclusive="left")
    week = frame.reindex(index)
    if len(week) != 168 or week[["I", "U", "Temp"]].isna().any().any():
        raise ValueError(f"Shifted week {shifted_start} to {shifted_end} is not a complete 168-hour observed interval.")
    return shifted_start, shifted_end, week


def model_input(week):
    time = np.arange(len(week), dtype=float)
    return {"I": week["I"].to_numpy(dtype=float), "U": week["U"].to_numpy(dtype=float), "PE": week["U"].to_numpy(dtype=float), "Time": time, "Temp": week["Temp"].to_numpy(dtype=float)}


def save_plot(start, end, week, fit, daily_fit):
    state = fit["deterministic_state"]
    index = week.index
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(index, week["I"], label="Observed aggregate input I")
    axes[0].plot(index, state["I"], label="Fitted wm_3 input I")
    axes[0].set_ylabel("Normalized input")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(index, state["L"], label="L_d(t), raw non-aggregate fit")
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

    fig.suptitle(f"wm_3 weekly fit {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} | natural L shift={daily_fit['shift']:02d}:00")
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"weekly_fit_{start:%Y-%m-%d_%H%M}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def print_result(base_start, seasonal, daily_fit, fit, plot_path):
    names = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    print(f"base week={base_start:%Y-%m-%d}")
    print(f"H(T): daily seasonal regression intercept={seasonal['intercept']:.8f}, slope={seasonal['slope']:.8f}")
    print("L_d(t): fitted directly to raw, non-aggregated input data; no seasonal subtraction or input normalization is used for the line fit.")
    print(f"L_d(t): natural shifted week {daily_fit['slice_start']} -> {daily_fit['slice_end']}, MSE={daily_fit['score']:.8f}")
    print(f"L_d(t): {daily_fit['samples']} raw observations from {daily_fit['measurement_columns']} measurement columns")
    for day, name in enumerate(names):
        raw_intercept, raw_slope = daily_fit["raw_models"][day]
        intercept, slope = daily_fit["models"][day]
        slice_start = daily_fit["slice_start"] + pd.Timedelta(days=day)
        slice_end = slice_start + pd.Timedelta(days=1)
        print(f"  {name} slice {slice_start:%Y-%m-%d %H:%M} -> {slice_end:%Y-%m-%d %H:%M}: raw={raw_intercept:.8f}+{raw_slope:.8f}*t, normalized L={intercept:.8f}+{slope:.8f}*t")
    print(f"alpha={np.array2string(fit['alpha'], precision=8)}")
    print(f"beta={np.array2string(fit['beta'], precision=8)}")
    print(f"kappa_R={fit['kappa_R']:.8f}")
    print(f"sigma_0={fit['sigma_0']:.8f}, sigma_O={fit['sigma_O']:.8f}")
    print(f"initial condition={np.array2string(fit['init_cond'], precision=8)}")
    print(f"deterministic loss={fit['deterministic_loss']:.10f}")
    print(f"plot={plot_path}")


def main():
    parser = ArgumentParser(description="Fit wm_3 to a naturally shifted weekly response. H(T) is seasonal; L_d(t) is fit directly to raw non-aggregated input observations in seven contiguous 24-hour slices.")
    parser.add_argument("--week", default="2022-01-03", help="Any date in the requested Monday-Sunday base week.")
    parser.add_argument("--grid-points", type=int, default=4)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    frame, measurements = aligned_data()
    H, seasonal = fit_seasonal(frame)
    base_start = week_start(args.week)
    L, daily_fit = fit_daily_time_functions(measurements, base_start)
    start, end, week = shifted_week(frame, base_start, daily_fit["shift"])
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
    plot_path = save_plot(start, end, week, fit, daily_fit)
    print_result(base_start, seasonal, daily_fit, fit, plot_path)


if __name__ == "__main__": main()
