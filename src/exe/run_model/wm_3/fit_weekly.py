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
    consumption = consumption_series(input_data(DATA / "input.xlsx"))
    prices = price_series(spot_price_data(DATA / "prices.xlsx", price_area="DK1"))
    temperature = temperature_series(weather_data())
    index = consumption.index.intersection(prices.index).intersection(temperature.index)
    frame = pd.concat((consumption.reindex(index), prices.reindex(index), temperature.reindex(index)), axis=1).dropna()
    frame["I"] = normalize(frame["I"])
    frame["U"] = normalize(frame["U"])
    frame["weekday"] = frame.index.weekday
    frame["hour"] = frame.index.hour + frame.index.minute / 60.0
    return frame


def fit_seasonal(frame):
    daily = frame[["I", "Temp"]].resample("D").mean().dropna()
    X = np.column_stack((np.ones(len(daily)), daily["Temp"].to_numpy(dtype=float)))
    coef = np.linalg.lstsq(X, daily["I"].to_numpy(dtype=float), rcond=None)[0]
    fitted_daily = X @ coef
    lo, hi = float(np.min(fitted_daily)), float(np.max(fitted_daily))

    if hi - lo <= 1e-12:
        def H(_temperature): return 0.5
    else:
        def H(temperature): return float(np.clip((coef[0] + coef[1] * float(temperature) - lo) / (hi - lo), 0.0, 1.0))

    daily_pred = coef[0] + coef[1] * frame["Temp"].to_numpy(dtype=float)
    residual = frame["I"].to_numpy(dtype=float) - daily_pred
    r_lo, r_hi = float(np.min(residual)), float(np.max(residual))
    if r_hi - r_lo <= 1e-12: raise ValueError("Season-adjusted consumption is constant; cannot fit L_d(t).")
    l_target = (residual - r_lo) / (r_hi - r_lo)
    return H, l_target, {"intercept": float(coef[0]), "slope": float(coef[1]), "prediction_min": lo, "prediction_max": hi, "residual_min": r_lo, "residual_max": r_hi}


def fit_daily_time_functions(frame, target):
    weekday = frame["weekday"].to_numpy(dtype=int)
    hour = frame["hour"].to_numpy(dtype=float)
    target = np.asarray(target, dtype=float)
    best = None

    for shift in range(24):
        models, sse = {}, 0.0
        x = np.mod(hour - shift, 24.0)
        for day in range(7):
            mask = weekday == day
            X = np.column_stack((np.ones(np.sum(mask)), x[mask]))
            coef = np.linalg.lstsq(X, target[mask], rcond=None)[0]
            residual = target[mask] - X @ coef
            models[day] = (float(coef[0]), float(coef[1]))
            sse += float(residual @ residual)
        if best is None or sse < best["sse"]: best = {"shift": shift, "models": models, "sse": sse}

    shift, models = best["shift"], best["models"]

    def L(time):
        time = float(time) % 168.0
        day, hour_value = int(time // 24.0) % 7, time % 24.0
        x = (hour_value - shift) % 24.0
        intercept, slope = models[day]
        return float(np.clip(intercept + slope * x, 0.0, 1.0))

    return L, best


def complete_week(frame, week_start):
    start = pd.Timestamp(week_start).normalize()
    start -= pd.Timedelta(days=start.weekday())
    end = start + pd.Timedelta(days=7)
    index = pd.date_range(start, end, freq="h", inclusive="left")
    week = frame.reindex(index)
    if len(week) != 168 or week[["I", "U", "Temp"]].isna().any().any(): raise ValueError(f"{start:%Y-%m-%d} is not a complete observed 168-hour week.")
    return start, week


def model_input(week):
    time = week.index.weekday.to_numpy(dtype=float) * 24.0 + week.index.hour.to_numpy(dtype=float)
    return {"I": week["I"].to_numpy(dtype=float), "U": week["U"].to_numpy(dtype=float), "PE": week["U"].to_numpy(dtype=float), "Time": time, "Temp": week["Temp"].to_numpy(dtype=float)}


def save_plot(start, week, fit, daily_fit):
    state = fit["deterministic_state"]
    index = week.index
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(index, week["I"], label="Observed input I")
    axes[0].plot(index, state["I"], label="Fitted wm_3 input I")
    axes[0].set_ylabel("Normalized input")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(index, state["L"], label="L_d(t)")
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

    fig.suptitle(f"wm_3 weekly fit {start:%Y-%m-%d} | optimized L shift={daily_fit['shift']:02d}:00")
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
    print(f"L_d(t): common shifted 24 h slice starts at {daily_fit['shift']:02d}:00, total SSE={daily_fit['sse']:.8f}")
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
    parser = ArgumentParser(description="Fit wm_3 to one observed weekly response using fitted seasonal H(T) and weekday-specific shifted linear L_d(t).")
    parser.add_argument("--week", default="2022-01-03", help="Any date in the requested Monday-Sunday week.")
    parser.add_argument("--grid-points", type=int, default=4)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    frame = aligned_data()
    H, l_target, seasonal = fit_seasonal(frame)
    L, daily_fit = fit_daily_time_functions(frame, l_target)
    start, week = complete_week(frame, args.week)
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
