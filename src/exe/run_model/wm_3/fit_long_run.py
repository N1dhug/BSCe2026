from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_3.fit_weekly import aligned_data
from src.model.wm_3 import wm_3
from src.params.wm_3 import brute_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_3" / "long_run"
INIT = np.array([0.5, 0.5], dtype=float)
DAY_HOURS = 24


def day_start(value): return pd.Timestamp(value).normalize()


def linear_time_function():
    def L(time_hours): return float(np.clip(float(time_hours) / DAY_HOURS, 0.0, 1.0))
    return L


def long_run_response(frame, base_start, radius_days=14):
    radius_days = int(radius_days)
    if radius_days < 1: raise ValueError("radius_days must be at least 1.")

    start = base_start - pd.Timedelta(days=radius_days)
    end = base_start + pd.Timedelta(days=radius_days + 1)
    period = frame.loc[(frame.index >= start) & (frame.index < end), ["I", "U", "Temp"]].dropna().copy()
    if len(period) < 48: raise ValueError("The selected long-run window contains too few aligned observations.")

    temp_min, temp_max = float(period["Temp"].min()), float(period["Temp"].max())
    temp_span = temp_max - temp_min
    if temp_span <= 1e-12: raise ValueError("Temperature is effectively constant in the long-run response window.")
    H_values = (temp_max - period["Temp"].to_numpy(dtype=float)) / temp_span

    hours = period.index.hour.to_numpy()
    weekdays = period.index.weekday.to_numpy()
    day_offset = ((period.index - base_start) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    day_offset -= float(np.mean(day_offset))

    hour_dummies = np.column_stack([(hours == hour).astype(float) for hour in range(1, 24)])
    weekday_dummies = np.column_stack([(weekdays == day).astype(float) for day in range(1, 7)])
    X = np.column_stack((
        np.ones(len(period)),
        period["U"].to_numpy(dtype=float),
        H_values,
        day_offset,
        hour_dummies,
        weekday_dummies,
    ))
    y = period["I"].to_numpy(dtype=float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ coef
    residual = y - fitted
    sse = float(residual @ residual)
    centered = y - float(np.mean(y))
    sst = float(centered @ centered)
    r2 = 1.0 - sse / sst if sst > 0.0 else np.nan

    price_response = float(coef[1])
    h_response = float(coef[2])
    raw_temp_response = -h_response / temp_span

    def H(temperature):
        return float(np.clip((temp_max - float(temperature)) / temp_span, 0.0, 1.0))

    return H, {
        "window_start": start, "window_end": end, "samples": len(period),
        "temp_min": temp_min, "temp_max": temp_max, "temp_span": temp_span,
        "price_response": price_response, "h_response": h_response,
        "raw_temp_response": raw_temp_response, "r2": float(r2), "sse": sse,
    }


def selected_day(frame, start):
    end = start + pd.Timedelta(days=1)
    index = pd.date_range(start, end, freq="h", inclusive="left")
    day = frame.reindex(index)
    if len(day) != DAY_HOURS or day[["I", "U", "Temp"]].isna().any().any(): raise ValueError(f"{start:%Y-%m-%d} is not a complete aligned 24-hour day.")
    return end, day


def model_input(frame, day, start, steps_per_hour):
    steps_per_hour = int(steps_per_hour)
    if steps_per_hour < 1: raise ValueError("steps_per_hour must be at least 1.")

    n_steps = DAY_HOURS * steps_per_hour + 1
    tau = np.linspace(0.0, 1.0, n_steps)
    time_hours = tau * DAY_HOURS
    fine_index = start + pd.to_timedelta(time_hours, unit="h")

    source_index = pd.date_range(start, start + pd.Timedelta(days=1), freq="h")
    source = frame.reindex(source_index)
    if source[["U", "Temp"]].iloc[:-1].isna().any().any(): raise ValueError("Hourly U/Temp signals are incomplete inside the selected day.")
    source["U"] = source["U"].ffill().bfill()
    source["Temp"] = source["Temp"].interpolate(method="time", limit_direction="both").ffill().bfill()

    source_hours = np.arange(DAY_HOURS + 1, dtype=float)
    hour_bin = np.minimum(np.floor(time_hours + 1e-12).astype(int), DAY_HOURS)
    U = source["U"].to_numpy(dtype=float)[hour_bin]
    Temp = np.interp(time_hours, source_hours, source["Temp"].to_numpy(dtype=float))

    observed_I = day["I"].to_numpy(dtype=float)
    I = np.interp(time_hours, np.arange(DAY_HOURS, dtype=float), observed_I, left=observed_I[0], right=observed_I[-1])
    observation_mask = np.zeros(n_steps, dtype=bool)
    observation_mask[np.arange(DAY_HOURS) * steps_per_hour] = True
    return {"I": I, "U": U, "PE": U.copy(), "Time": time_hours, "Temp": Temp, "observation_mask": observation_mask}, fine_index


def equilibrium_responses(fit):
    c, d = fit["I_coeff"], fit["d_coeff"]
    ratio = float(c[2] / d[2])
    return float(c[3] - ratio * d[3]), float(c[4] - ratio * d[4])


def save_plot(start, end, day, fine_index, fit, long_run):
    state, obs = fit["deterministic_state"], fit["observation_mask"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(day.index, day["I"], label="Observed hourly aggregate input I")
    axes[0].scatter(day.index, day["I"], s=18, label="Observed I samples")
    axes[0].plot(fine_index, state["I"], label="Fitted wm_3 input I")
    axes[0].scatter(fine_index[obs], state["I"][obs], s=18, label="WM3 prediction at observation times")
    axes[0].set_ylabel("Normalized input")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(fine_index, state["U"], label="Normalized price U")
    axes[1].plot(fine_index, state["L"], label="Daily linear L(t)")
    axes[1].plot(fine_index, state["H"], label="Monthly-scale H(T)")
    axes[1].set_ylabel("Signals / features")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(fine_index, state["V"], label="State V")
    axes[2].plot(fine_index, state["R"], label="Rate R")
    axes[2].set_ylabel("State / rate")
    axes[2].set_xlabel("Physical time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(
        f"wm_3 long-run constrained fit {start:%Y-%m-%d} | "
        f"price response={long_run['price_response']:.4f}, H response={long_run['h_response']:.4f}"
    )
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"long_run_fit_{start:%Y-%m-%d}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def print_result(start, end, long_run, fit, model, plot_path, steps_per_hour):
    price_eq, h_eq = equilibrium_responses(fit)
    print(f"selected day={start} -> {end}")
    print(f"long-run window={long_run['window_start']} -> {long_run['window_end']} ({long_run['samples']} hourly samples)")
    print("monthly statistical fit: I ~ U + H(T) + linear day trend + hour-of-day fixed effects + weekday fixed effects")
    print(f"long-run regression R^2={long_run['r2']:.8f}")
    print(f"long-run normalized price response={long_run['price_response']:.8f}")
    print(f"long-run H(T) response={long_run['h_response']:.8f}")
    print(f"equivalent raw temperature response={long_run['raw_temp_response']:.8f} per degree C")
    print(f"H(T)=clip(({long_run['temp_max']:.8f}-T)/{long_run['temp_span']:.8f}, 0, 1)")
    print("daily time effect: L(t)=t/24 is fixed as a basis, while its wm_3 coefficient is estimated inside the model.")
    print(f"model time: tau in [0, 1], dt={model.dt:.12f}, response grid={3600.0 / steps_per_hour:.4f} physical seconds")
    print(f"alpha={np.array2string(fit['alpha'], precision=8)}")
    print(f"beta={np.array2string(fit['beta'], precision=8)}")
    print(f"kappa_R={fit['kappa_R']:.8f}, branch={fit['brute_branch']}")
    print(f"initial condition={np.array2string(fit['init_cond'], precision=8)}")
    print(f"wm_3 equilibrium price response={price_eq:.8f}")
    print(f"wm_3 equilibrium H response={h_eq:.8f}")
    print(f"sigma_0={fit['sigma_0']:.8f}, sigma_O={fit['sigma_O']:.8f}")
    print(f"deterministic loss={fit['deterministic_loss']:.10f}")
    print(f"brute fit time={fit['brute_fit_time']:.3f}s")
    print(f"plot={plot_path}")


def main():
    parser = ArgumentParser(description="Fit wm_3 for one day while constraining long-run price and temperature responses from a +/- day window.")
    parser.add_argument("--day", default="2022-01-03", help="Selected calendar day to fit.")
    parser.add_argument("--radius-days", type=int, default=14, help="Number of calendar days on each side used for long-run response estimation.")
    parser.add_argument("--steps-per-hour", type=int, default=60, help="Internal wm_3 response steps per physical hour.")
    parser.add_argument("--grid-points", type=int, default=4)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    frame, _ = aligned_data()
    start = day_start(args.day)
    end, day = selected_day(frame, start)

    H, long_run = long_run_response(frame, start, args.radius_days)
    L = linear_time_function()
    data, fine_index = model_input(frame, day, start, args.steps_per_hour)

    model = wm_3((0.0, 1.0), len(fine_index), L, H)
    fit = brute_params(model).fit(
        data, INIT,
        grid_points=args.grid_points,
        refinements=args.refinements,
        fit_initial_state=True,
        fit_diffusion=True,
        long_run_price_response=long_run["price_response"],
        long_run_h_response=long_run["h_response"],
        verbose=True,
        progress_every=args.progress_every,
    )

    plot_path = save_plot(start, end, day, fine_index, fit, long_run)
    print_result(start, end, long_run, fit, model, plot_path, args.steps_per_hour)


if __name__ == "__main__": main()
