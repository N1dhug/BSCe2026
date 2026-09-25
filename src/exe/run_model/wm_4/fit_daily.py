from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_3.fit_weekly import aligned_data, fit_seasonal, positive_line
from src.model.wm_4 import wm_4
from src.params.wm_4 import brute_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "daily"
INIT = np.array([0.5, 0.0], dtype=float)
DAY_HOURS = 24


def day_start(value): return pd.Timestamp(value).normalize()


def fit_time_function(measurements, start):
    n_measurements = measurements.shape[1]
    index = pd.date_range(start, start + pd.Timedelta(hours=DAY_HOURS), freq="h", inclusive="left")
    values = measurements.reindex(index).to_numpy(dtype=float)
    x = np.repeat(np.arange(DAY_HOURS, dtype=float), n_measurements)
    y = values.reshape(-1)
    raw_intercept, raw_slope, sse, samples = positive_line(x, y)
    score = sse / max(samples, 1)

    raw_prediction = raw_intercept + raw_slope * np.arange(DAY_HOURS, dtype=float)
    scale = float(np.max(raw_prediction))
    if not np.isfinite(scale) or scale <= 0.0: raise ValueError("The fitted raw L(t) response cannot be scaled into [0, 1].")

    intercept, slope = raw_intercept / scale, raw_slope / scale

    def L(time_hours):
        time_hours = float(np.clip(time_hours, 0.0, np.nextafter(float(DAY_HOURS), 0.0)))
        return float(np.clip(intercept + slope * time_hours, 0.0, 1.0))

    return L, {
        "slice_start": start, "slice_end": start + pd.Timedelta(hours=DAY_HOURS),
        "raw_model": (raw_intercept, raw_slope), "model": (intercept, slope),
        "raw_scale": scale, "sse": sse, "score": score, "samples": samples,
        "measurement_columns": n_measurements,
    }


def selected_day(frame, start):
    end = start + pd.Timedelta(hours=DAY_HOURS)
    index = pd.date_range(start, end, freq="h", inclusive="left")
    day = frame.reindex(index)
    if len(day) != DAY_HOURS or day[["I", "U", "Temp"]].isna().any().any(): raise ValueError(f"{start:%Y-%m-%d} is not a complete aligned 24-hour day.")
    return end, day


def model_input(frame, day, start, steps_per_hour):
    steps_per_hour = int(steps_per_hour)
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2 for a finer-than-hourly response grid.")

    n_steps = DAY_HOURS * steps_per_hour + 1
    tau = np.linspace(0.0, 1.0, n_steps)
    time_hours = tau * DAY_HOURS
    fine_index = start + pd.to_timedelta(time_hours, unit="h")

    source_index = pd.date_range(start, start + pd.Timedelta(hours=DAY_HOURS), freq="h")
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


def save_plot(start, end, day, fine_index, fit):
    state, obs = fit["deterministic_state"], fit["observation_mask"]
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    axes[0].plot(day.index, day["I"], label="Observed hourly aggregate input I")
    axes[0].scatter(day.index, day["I"], s=18, label="Observed I samples")
    axes[0].plot(fine_index, state["I"], label="Fitted wm_4 input I=O+R")
    axes[0].scatter(fine_index[obs], state["I"][obs], s=18, label="WM4 prediction at observation times")
    axes[0].set_ylabel("Normalized input")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(fine_index, state["L"], label="Day-specific linear L(t)")
    axes[1].plot(fine_index, state["H"], label="Seasonal H(T)")
    axes[1].plot(fine_index, state["U"], label="Normalized DK1 price U")
    axes[1].set_ylabel("Signals / features")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(fine_index, state["Q"], label="Desired flexibility flow Q")
    axes[2].plot(fine_index, state["O"], label="Hidden output O")
    axes[2].set_ylabel("Flow / output")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(fine_index, state["F"], label="Flexibility state F")
    axes[3].plot(fine_index, state["R"], label="Realized flexibility flow R")
    axes[3].set_ylabel("State / rate")
    axes[3].set_xlabel("Physical time")
    axes[3].legend()
    axes[3].grid(True, alpha=0.25)

    fig.suptitle(f"wm_4 daily fit {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} | q_L=omega_L=0, seasonal H(T)")
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"daily_fit_{start:%Y-%m-%d_%H%M}_qL0_omegaL0.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def print_result(start, end, seasonal, time_fit, fit, model, plot_path, steps_per_hour):
    raw_intercept, raw_slope = time_fit["raw_model"]
    intercept, slope = time_fit["model"]
    print(f"selected day={start} -> {end}")
    print(f"model time: tau in [0, 1], dt={model.dt:.12f}, response grid={3600.0 / steps_per_hour:.4f} physical seconds")
    print(f"observations={np.count_nonzero(fit['observation_mask'])} hourly I samples; state grid points={model.t_steps}")
    print("parameter solver: src.params.wm_4.brute_params with q_L=omega_L=0; L(t) is diagnostic only")
    print(f"H(T): seasonal daily regression intercept={seasonal['intercept']:.8f}, slope={seasonal['slope']:.8f}")
    print("L(t): fitted directly to raw non-aggregated measurements for the selected calendar day.")
    print(f"L(t): raw={raw_intercept:.8f}+{raw_slope:.8f}*hour, MSE={time_fit['score']:.8f}, samples={time_fit['samples']}")
    print(f"L(t): normalized={intercept:.8f}+{slope:.8f}*hour, scale={time_fit['raw_scale']:.8f}")
    print(f"q={np.array2string(fit['q'], precision=8)}")
    print(f"omega={np.array2string(fit['omega'], precision=8)}")
    print(f"kappa_R={fit['kappa_R']:.8f}, branch={fit['brute_branch']}")
    print(f"sigma_0={fit['sigma_0']:.8f}, sigma_O={fit['sigma_O']:.8f}")
    print(f"initial condition [F0,R0]={np.array2string(fit['init_cond'], precision=8)}")
    print(f"deterministic loss at hourly observations={fit['deterministic_loss']:.10f}")
    print(f"brute fit time={fit['brute_fit_time']:.3f}s")
    print(f"plot={plot_path}")


def main():
    parser = ArgumentParser(description="Fit wm_4 to one calendar day using seasonal H(T) and a day-specific linear L(t).")
    parser.add_argument("--day", default="2022-01-03", help="Calendar day to fit.")
    parser.add_argument("--steps-per-hour", type=int, default=60, help="Internal wm_4 response substeps per physical hour; default 60 gives a one-minute grid.")
    parser.add_argument("--grid-points", type=int, default=4)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    frame, measurements = aligned_data()
    start = day_start(args.day)
    end, day = selected_day(frame, start)
    H, seasonal = fit_seasonal(frame)
    L, time_fit = fit_time_function(measurements, start)
    data, fine_index = model_input(frame, day, start, args.steps_per_hour)

    model = wm_4((0.0, 1.0), len(fine_index), L, H)
    fit = brute_params(model).fit(
        data, INIT,
        grid_points=args.grid_points,
        refinements=args.refinements,
        fit_initial_state=True,
        q_l_zero=True,
        omega_l_zero=True,
        fit_diffusion=True,
        verbose=True,
        progress_every=args.progress_every,
    )

    plot_path = save_plot(start, end, day, fine_index, fit)
    print_result(start, end, seasonal, time_fit, fit, model, plot_path, args.steps_per_hour)


if __name__ == "__main__": main()
