from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from src.exe.run_model.wm_3.fit_weekly import END, START, aligned_data, fit_seasonal
from src.model.wm_4 import wm_4
from src.params.wm_4 import brute_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "daily"
INIT = np.array([0.5, 0.0], dtype=float)
DAY_HOURS = 24
WEEK_HOURS = 168


def day_start(value): return pd.Timestamp(value).normalize()


def hour_of_week(index):
    index = pd.DatetimeIndex(index)
    return (
        index.dayofweek.to_numpy(dtype=float) * DAY_HOURS
        + index.hour.to_numpy(dtype=float)
        + index.minute.to_numpy(dtype=float) / 60.0
        + index.second.to_numpy(dtype=float) / 3600.0
    )


def fit_periodic_time(frame, exclude_start=None, exclude_end=None):
    period = frame.loc[(frame.index >= START) & (frame.index < END), ["I"]].copy()
    if exclude_start is not None and exclude_end is not None:
        period = period.loc[(period.index < exclude_start) | (period.index >= exclude_end)]
    if period.empty: raise ValueError("No long-run data is available for the periodic L(t) fit.")

    bins = period.index.dayofweek * DAY_HOURS + period.index.hour
    profile = period["I"].groupby(bins).mean().reindex(np.arange(WEEK_HOURS))
    if profile.isna().all(): raise ValueError("The hour-of-week profile contains no finite observations.")
    profile = profile.interpolate(limit_direction="both").ffill().bfill()

    raw = profile.to_numpy(dtype=float)
    lo, hi = float(np.min(raw)), float(np.max(raw))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12: raise ValueError("The long-run hour-of-week profile cannot be normalized.")

    normalized = (raw - lo) / (hi - lo)
    x = np.arange(WEEK_HOURS + 1, dtype=float)
    y = np.append(normalized, normalized[0])
    spline = CubicSpline(x, y, bc_type="periodic")

    def L(time_hour_of_week):
        t = float(time_hour_of_week) % WEEK_HOURS
        return float(np.clip(spline(t), 0.0, 1.0))

    fitted = np.array([L(value) for value in hour_of_week(period.index)], dtype=float)
    normalized_observed = np.clip((period["I"].to_numpy(dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    mse = float(np.mean((normalized_observed - fitted) ** 2))
    return L, {
        "period_hours": WEEK_HOURS, "profile_points": WEEK_HOURS,
        "raw_min": lo, "raw_max": hi, "mse": mse,
        "samples": int(len(period)),
    }


def selected_day(frame, start):
    end = start + pd.Timedelta(hours=DAY_HOURS)
    index = pd.date_range(start, end, freq="h", inclusive="left")
    day = frame.reindex(index)
    if len(day) != DAY_HOURS or day[["I", "U", "Temp"]].isna().any().any(): raise ValueError(f"{start:%Y-%m-%d} is not a complete aligned 24-hour day.")
    return end, day


def model_input(frame, start, steps_per_hour, warmup_hours):
    steps_per_hour, warmup_hours = int(steps_per_hour), int(warmup_hours)
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2 for a finer-than-hourly response grid.")
    if warmup_hours < 0: raise ValueError("warmup_hours must be nonnegative.")

    total_start = start - pd.Timedelta(hours=warmup_hours)
    end = start + pd.Timedelta(hours=DAY_HOURS)
    total_hours = warmup_hours + DAY_HOURS
    n_steps = total_hours * steps_per_hour + 1
    model_time = np.linspace(0.0, total_hours / DAY_HOURS, n_steps)
    elapsed_hours = np.linspace(0.0, float(total_hours), n_steps)
    fine_index = total_start + pd.to_timedelta(elapsed_hours, unit="h")

    source_index = pd.date_range(total_start, end, freq="h")
    source = frame.reindex(source_index)
    if source[["I", "U", "Temp"]].iloc[:-1].isna().any().any(): raise ValueError(f"Warm-up/target signals are incomplete between {total_start} and {end}.")
    source["I"] = source["I"].interpolate(method="time", limit_direction="both").ffill().bfill()
    source["U"] = source["U"].ffill().bfill()
    source["Temp"] = source["Temp"].interpolate(method="time", limit_direction="both").ffill().bfill()

    source_hours = np.arange(total_hours + 1, dtype=float)
    hour_bin = np.minimum(np.floor(elapsed_hours + 1e-12).astype(int), total_hours)
    I = np.interp(elapsed_hours, source_hours, source["I"].to_numpy(dtype=float))
    U = source["U"].to_numpy(dtype=float)[hour_bin]
    Temp = np.interp(elapsed_hours, source_hours, source["Temp"].to_numpy(dtype=float))
    Time = hour_of_week(fine_index)

    observation_mask = np.zeros(n_steps, dtype=bool)
    target_offset = warmup_hours * steps_per_hour
    observation_mask[target_offset + np.arange(DAY_HOURS) * steps_per_hour] = True

    data = {"I": I, "U": U, "PE": U.copy(), "Time": Time, "Temp": Temp, "observation_mask": observation_mask}
    return data, fine_index, total_start, (0.0, total_hours / DAY_HOURS)


def save_plot(start, end, day, fine_index, fit, warmup_hours):
    state, obs = fit["deterministic_state"], fit["observation_mask"]
    target = (fine_index >= start) & (fine_index <= end)
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    axes[0].plot(day.index, day["I"], label="Observed hourly aggregate input I")
    axes[0].scatter(day.index, day["I"], s=18, label="Observed I samples")
    axes[0].plot(fine_index[target], state["I"][target], label="Fitted wm_4 input I=O+R")
    axes[0].scatter(fine_index[obs], state["I"][obs], s=18, label="WM4 prediction at observation times")
    axes[0].set_ylabel("Normalized input")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(fine_index[target], state["L"][target], label="Long-run 168 h periodic L(t)")
    axes[1].plot(fine_index[target], state["H"][target], label="Seasonal H(T)")
    axes[1].plot(fine_index[target], state["U"][target], label="Normalized DK1 price U")
    axes[1].set_ylabel("Signals / features")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(fine_index[target], state["Q"][target], label="Desired flexibility flow Q")
    axes[2].plot(fine_index[target], state["O"][target], label="Hidden output O")
    axes[2].set_ylabel("Flow / output")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(fine_index[target], state["F"][target], label="Flexibility state F")
    axes[3].plot(fine_index[target], state["R"][target], label="Realized flexibility flow R")
    axes[3].set_ylabel("State / rate")
    axes[3].set_xlabel("Physical time")
    axes[3].legend()
    axes[3].grid(True, alpha=0.25)

    fig.suptitle(f"wm_4 daily fit {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} | periodic L, seasonal H, {warmup_hours} h warm-up")
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"daily_fit_{start:%Y-%m-%d_%H%M}_periodic_warmup.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def print_result(start, end, total_start, seasonal, time_fit, fit, model, plot_path, steps_per_hour, warmup_hours):
    target_index = warmup_hours * steps_per_hour
    state = fit["deterministic_state"]
    print(f"warm-up={total_start} -> {start} ({warmup_hours} h); target={start} -> {end}")
    print(f"model time measured in days: T={model.T}, dt={model.dt:.12f}, response grid={3600.0 / steps_per_hour:.4f} physical seconds")
    print(f"observations={np.count_nonzero(fit['observation_mask'])} target-day hourly I samples; total state grid points={model.t_steps}")
    print("parameter solver: src.params.wm_4.brute_params | Q=q0+qF*F+qU*U, O=omegaL*L+omegaH*H")
    print(f"H(T): Jan-Oct seasonal daily regression intercept={seasonal['intercept']:.8f}, slope={seasonal['slope']:.8f}")
    print(f"L(t): Jan-Oct 168 h periodic cubic spline, samples={time_fit['samples']}, profile points={time_fit['profile_points']}, normalized MSE={time_fit['mse']:.8f}")
    print(f"L(t): raw profile range=[{time_fit['raw_min']:.8f}, {time_fit['raw_max']:.8f}]")
    print(f"q={np.array2string(fit['q'], precision=8)}")
    print(f"omega=[omega_L, omega_H]={np.array2string(fit['omega'], precision=8)}")
    print(f"kappa_R={fit['kappa_R']:.8f}, branch={fit['brute_branch']}")
    print(f"sigma_0={fit['sigma_0']:.8f}, sigma_O={fit['sigma_O']:.8f}")
    print(f"fixed warm-up initial [F0,R0]={np.array2string(fit['init_cond'], precision=8)}")
    print(f"target-day initial state [F,R]=[{state['F'][target_index]:.8f}, {state['R'][target_index]:.8f}]")
    print(f"deterministic loss at target-day observations={fit['deterministic_loss']:.10f}")
    print(f"brute fit time={fit['brute_fit_time']:.3f}s")
    print(f"plot={plot_path}")


def main():
    parser = ArgumentParser(description="Fit wm_4 to one calendar day using a long-run periodic L(t), seasonal H(T), and a pre-day state warm-up.")
    parser.add_argument("--day", default="2022-01-03", help="Calendar day to score.")
    parser.add_argument("--warmup-hours", type=int, default=48, help="Hours simulated before the target day with a fixed initial state; default 48.")
    parser.add_argument("--steps-per-hour", type=int, default=60, help="Internal wm_4 response substeps per physical hour; default 60 gives a one-minute grid.")
    parser.add_argument("--grid-points", type=int, default=4)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    frame, _ = aligned_data()
    start = day_start(args.day)
    end, day = selected_day(frame, start)
    H, seasonal = fit_seasonal(frame)
    L, time_fit = fit_periodic_time(frame, exclude_start=start, exclude_end=end)
    data, fine_index, total_start, model_T = model_input(frame, start, args.steps_per_hour, args.warmup_hours)

    model = wm_4(model_T, len(fine_index), L, H)
    fit = brute_params(model).fit(
        data, INIT,
        grid_points=args.grid_points,
        refinements=args.refinements,
        fit_initial_state=False,
        fit_diffusion=True,
        verbose=True,
        progress_every=args.progress_every,
    )

    plot_path = save_plot(start, end, day, fine_index, fit, args.warmup_hours)
    print_result(start, end, total_start, seasonal, time_fit, fit, model, plot_path, args.steps_per_hour, args.warmup_hours)


if __name__ == "__main__": main()
