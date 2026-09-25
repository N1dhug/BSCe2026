from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_3.fit_weekly import aligned_data, fit_seasonal, positive_line
from src.model.wm_3 import wm_3
from src.params.wm_3 import brute_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_3" / "daily"
INIT = np.array([0.5, 0.5], dtype=float)
DAY_HOURS = 24


def day_start(value):
    return pd.Timestamp(value).normalize()


def fit_time_function(measurements, base_start):
    n_measurements, best = measurements.shape[1], None

    for shift in range(24):
        start = base_start + pd.Timedelta(hours=shift)
        index = pd.date_range(start, start + pd.Timedelta(hours=DAY_HOURS), freq="h", inclusive="left")
        values = measurements.reindex(index).to_numpy(dtype=float)
        x = np.repeat(np.arange(DAY_HOURS, dtype=float), n_measurements)
        y = values.reshape(-1)
        intercept, slope, sse, samples = positive_line(x, y)
        score = sse / max(samples, 1)
        if best is None or score < best["score"]:
            best = {
                "shift": shift, "slice_start": start, "slice_end": start + pd.Timedelta(hours=DAY_HOURS),
                "raw_model": (intercept, slope), "sse": sse, "score": score, "samples": samples,
            }

    raw_intercept, raw_slope = best["raw_model"]
    raw_prediction = raw_intercept + raw_slope * np.arange(DAY_HOURS, dtype=float)
    scale = float(np.max(raw_prediction))
    if not np.isfinite(scale) or scale <= 0.0: raise ValueError("The fitted raw L(t) response cannot be scaled into [0, 1].")

    intercept, slope = raw_intercept / scale, raw_slope / scale

    def L(time_hours):
        time_hours = float(np.clip(time_hours, 0.0, np.nextafter(float(DAY_HOURS), 0.0)))
        return float(np.clip(intercept + slope * time_hours, 0.0, 1.0))

    best["model"] = (intercept, slope)
    best["raw_scale"] = scale
    best["measurement_columns"] = n_measurements
    return L, best


def shifted_day(frame, start, shift):
    shifted_start = start + pd.Timedelta(hours=int(shift))
    shifted_end = shifted_start + pd.Timedelta(hours=DAY_HOURS)
    index = pd.date_range(shifted_start, shifted_end, freq="h", inclusive="left")
    day = frame.reindex(index)
    if len(day) != DAY_HOURS or day[["I", "U", "Temp"]].isna().any().any():
        raise ValueError(f"Shifted day {shifted_start} to {shifted_end} is not a complete 24-hour observed interval.")
    return shifted_start, shifted_end, day


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

    observed_hours = np.arange(DAY_HOURS, dtype=float)
    observed_I = day["I"].to_numpy(dtype=float)
    I = np.interp(time_hours, observed_hours, observed_I, left=observed_I[0], right=observed_I[-1])
    observation_mask = np.zeros(n_steps, dtype=bool)
    observation_mask[np.arange(DAY_HOURS) * steps_per_hour] = True

    data = {"I": I, "U": U, "PE": U.copy(), "Time": time_hours, "Temp": Temp, "observation_mask": observation_mask}
    return data, fine_index


def save_plot(start, end, day, fine_index, fit, time_fit):
    state, obs = fit["deterministic_state"], fit["observation_mask"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(day.index, day["I"], label="Observed hourly aggregate input I")
    axes[0].scatter(day.index, day["I"], s=18, label="Observed I samples")
    axes[0].plot(fine_index, state["I"], label="Fitted wm_3 input I")
    axes[0].scatter(fine_index[obs], state["I"][obs], s=18, label="WM3 prediction at observation times")
    axes[0].set_ylabel("Normalized input")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(fine_index, state["L"], label="L(t), raw non-aggregate fit")
    axes[1].plot(fine_index, state["H"], label="H(T)")
    axes[1].set_ylabel("Fitted features")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(fine_index, state["U"], label="Normalized DK1 price U")
    axes[2].plot(fine_index, state["V"], label="V")
    axes[2].plot(fine_index, state["R"], label="R")
    axes[2].set_ylabel("State / price")
    axes[2].set_xlabel("Physical time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"wm_3 daily fit {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} | optimized natural shift={time_fit['shift']:02d}:00")
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"daily_fit_{start:%Y-%m-%d_%H%M}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def print_result(base_start, seasonal, time_fit, fit, plot_path, model, steps_per_hour):
    raw_intercept, raw_slope = time_fit["raw_model"]
    intercept, slope = time_fit["model"]
    print(f"base day={base_start:%Y-%m-%d}")
    print(f"optimized 24 h slice={time_fit['slice_start']} -> {time_fit['slice_end']}")
    print(f"model time: tau in [0, 1], dt={model.dt:.12f}, response grid={3600.0 / steps_per_hour:.4f} physical seconds")
    print(f"observations: {np.count_nonzero(fit['observation_mask'])} hourly I samples; state dynamics use {model.t_steps} grid points")
    print("parameter solver: src.params.wm_3.brute_params with vectorized state maps and fine-grid constraint generation")
    print(f"H(T): daily seasonal regression intercept={seasonal['intercept']:.8f}, slope={seasonal['slope']:.8f}")
    print("L(t): fitted directly to raw non-aggregated hourly measurements in the selected natural 24 h slice.")
    print(f"L(t): raw={raw_intercept:.8f}+{raw_slope:.8f}*hour, MSE={time_fit['score']:.8f}, samples={time_fit['samples']}")
    print(f"L(t): normalized={intercept:.8f}+{slope:.8f}*hour, scale={time_fit['raw_scale']:.8f}")
    print(f"alpha={np.array2string(fit['alpha'], precision=8)}")
    print(f"beta={np.array2string(fit['beta'], precision=8)}")
    print(f"kappa_R={fit['kappa_R']:.8f}")
    print(f"sigma_0={fit['sigma_0']:.8f}, sigma_O={fit['sigma_O']:.8f}")
    print(f"initial condition={np.array2string(fit['init_cond'], precision=8)}")
    print(f"deterministic loss at hourly observations={fit['deterministic_loss']:.10f}")
    print(f"constraint rounds={fit.get('constraint_rounds', 'n/a')}, active fine-grid constraints={fit.get('active_input_constraints', 'n/a')}")
    print(f"brute fit time={fit.get('brute_fit_time', np.nan):.3f}s")
    print(f"plot={plot_path}")


def main():
    parser = ArgumentParser(description="Fit wm_3 to one optimized natural 24-hour slice on a dimensionless fine response grid.")
    parser.add_argument("--day", default="2022-01-03", help="Calendar day whose 00:00-23:00 starting hours are searched for the best contiguous 24 h slice.")
    parser.add_argument("--steps-per-hour", type=int, default=60, help="Internal wm_3 response substeps per physical hour; default 60 gives a one-minute grid.")
    parser.add_argument("--grid-points", type=int, default=4)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    frame, measurements = aligned_data()
    H, seasonal = fit_seasonal(frame)
    base_start = day_start(args.day)
    L, time_fit = fit_time_function(measurements, base_start)
    start, end, day = shifted_day(frame, base_start, time_fit["shift"])
    data, fine_index = model_input(frame, day, start, args.steps_per_hour)

    model = wm_3((0.0, 1.0), len(fine_index), L, H)
    fit = brute_params(model).fit(
        data, INIT,
        grid_points=args.grid_points,
        refinements=args.refinements,
        fit_initial_state=False,
        fit_diffusion=True,
        verbose=True,
        progress_every=args.progress_every,
    )

    plot_path = save_plot(start, end, day, fine_index, fit, time_fit)
    print_result(base_start, seasonal, time_fit, fit, plot_path, model, args.steps_per_hour)


if __name__ == "__main__": main()
