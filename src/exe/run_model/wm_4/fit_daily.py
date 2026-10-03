from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_3.fit_weekly import aligned_data
from src.exe.run_model.wm_4.output_model import output_callable, statistical_output
from src.model.wm_4 import wm_4
from src.params.wm_4 import qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "daily"
INIT_F = 0.0
INIT = np.array([INIT_F, 0.0], dtype=float)
DAY_HOURS = 24
PARAMETER_WEEKS = 5
BASELINE_WEEKS = PARAMETER_WEEKS


def day_start(value): return pd.Timestamp(value).normalize()


def aligned_wm4_data(tag="MED"):
    return aligned_data(tag, price_domain=(-1.0, 1.0), price_year=2022)


def fit_expected_output(frame, start):
    model = statistical_output().fit(frame, day_start(start))
    return model, dict(model.metadata)


def model_input(frame, start, steps_per_hour, warmup_hours, output_model=None):
    start, steps_per_hour, warmup_hours = day_start(start), int(steps_per_hour), int(warmup_hours)
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2 for a finer-than-hourly response grid.")
    if warmup_hours < 0: raise ValueError("warmup_hours must be nonnegative.")

    total_start = start - pd.Timedelta(hours=warmup_hours)
    end = start + pd.Timedelta(hours=DAY_HOURS)
    total_hours = warmup_hours + DAY_HOURS
    n_steps = total_hours * steps_per_hour + 1
    elapsed = np.linspace(0.0, float(total_hours), n_steps)
    fine_index = total_start + pd.to_timedelta(elapsed, unit="h")

    source_index = pd.date_range(total_start, end, freq="h")
    source = frame.reindex(source_index)[["I", "U", "Temp"]].copy()
    if source[["I", "U", "Temp"]].iloc[:-1].isna().any().any(): raise ValueError(f"Warm-up/target signals are incomplete between {total_start} and {end}.")
    source["I"] = source["I"].interpolate(method="time", limit_direction="both").ffill().bfill()
    source["U"] = source["U"].ffill().bfill()
    source["Temp"] = source["Temp"].interpolate(method="time", limit_direction="both").ffill().bfill()

    source_hours = np.arange(total_hours + 1, dtype=float)
    hour_bin = np.minimum(np.floor(elapsed + 1e-12).astype(int), total_hours)
    I = np.interp(elapsed, source_hours, source["I"].to_numpy(dtype=float))
    U = source["U"].to_numpy(dtype=float)[hour_bin]
    Temp = np.interp(elapsed, source_hours, source["Temp"].to_numpy(dtype=float))

    observation_mask = np.zeros(n_steps, dtype=bool)
    transition_mask = np.zeros(n_steps - 1, dtype=bool)
    offset = warmup_hours * steps_per_hour
    observation_mask[offset + np.arange(DAY_HOURS) * steps_per_hour] = True
    transition_mask[offset:offset + DAY_HOURS * steps_per_hour] = True
    data = {
        "I": I, "U": U, "PE": U.copy(), "Temp": Temp, "Time": elapsed,
        "observation_mask": observation_mask, "transition_mask": transition_mask,
    }
    if output_model is not None: data["O"] = output_model.predict(fine_index, Temp)
    return data, fine_index, total_start, (0.0, total_hours / DAY_HOURS)


def fitted_initial_condition(data):
    if "O" not in data: raise ValueError("WM4 fitted initial condition requires exogenous O samples.")
    return np.array([INIT_F, float(data["I"][0] - data["O"][0])], dtype=float)


def model_output(data, fine_index):
    if "O" not in data: raise ValueError("model_output requires precomputed O samples.")
    return output_callable(fine_index, data["O"])


def selected_day(frame, start):
    start = day_start(start)
    end = start + pd.Timedelta(hours=DAY_HOURS)
    index = pd.date_range(start, end, freq="h", inclusive="left")
    day = frame.reindex(index)
    if len(day) != DAY_HOURS or day[["I", "U", "Temp"]].isna().any().any(): raise ValueError(f"{start:%Y-%m-%d} is not a complete aligned 24-hour day.")
    return end, day


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

    axes[1].plot(fine_index[target], state["O"][target], label="Expected output O(time, weekday, temperature)")
    axes[1].plot(fine_index[target], state["U"][target], label="Normalized DK1 price U")
    axes[1].set_ylabel("Signals / output")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(fine_index[target], state["R_dot"][target], label="Flexibility acceleration dR/dt")
    axes[2].plot(fine_index[target], state["O"][target], label="Expected output O")
    axes[2].set_ylabel("Flow / output")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(fine_index[target], state["F"][target], label="Flexibility state F")
    axes[3].plot(fine_index[target], state["R"][target], label="Realized flexibility flow R")
    axes[3].set_ylabel("State / rate")
    axes[3].set_xlabel("Physical time")
    axes[3].legend()
    axes[3].grid(True, alpha=0.25)

    fig.suptitle(f"wm_4 daily fit {start:%Y-%m-%d} | robust weather-aware O, {warmup_hours} h warm-up")
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"daily_fit_{start:%Y-%m-%d_%H%M}_weather_output.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def print_result(start, end, total_start, output_fit, fit, model, plot_path, steps_per_hour, warmup_hours):
    target_index = warmup_hours * steps_per_hour
    state = fit["deterministic_state"]
    print(f"warm-up={total_start} -> {start} ({warmup_hours} h); target={start} -> {end}")
    print(f"model time measured in days: T={model.T}, dt={model.dt:.12f}, response grid={3600.0 / steps_per_hour:.4f} physical seconds")
    print(f"observations={np.count_nonzero(fit['observation_mask'])} target-day hourly I samples; total state grid points={model.t_steps}")
    print("parameter solver: src.params.wm_4.qp_params | dR/dt=rF*F+rU*U+rR*R, I=O+R")
    print(f"O: robust Huber-ridge model using time-of-day Fourier terms, weekday, season and temperature/chill interactions")
    print(f"O training: {output_fit['fit_start']} -> {output_fit['fit_end']}, samples={output_fit['samples']}, features={output_fit['features']}, MSE={output_fit['mse']:.8f}, MAE={output_fit['mae']:.8f}")
    print(f"O temperature range=[{output_fit['temperature_min']:.3f}, {output_fit['temperature_max']:.3f}] C, heating base={output_fit['base_temperature']:.1f} C")
    r = fit["r"]
    print(f"coefficients: r_F={r[0]:.8f}, r_U={r[1]:.8f}, r_R={r[2]:.8f}")
    print(f"damping: branch={fit['damping_branch']}, zeta={fit['damping_ratio']:.8f}")
    print(f"diffusion: sigma_0={fit['sigma_0']:.8f}, sigma_I={fit['sigma_I']:.8f}")
    print(f"fitted initial [F0,R0]={np.array2string(fit['init_cond'], precision=8)}")
    print(f"target-day initial state [F,R]=[{state['F'][target_index]:.8f}, {state['R'][target_index]:.8f}]")
    print(f"QP transition loss={fit['qp_loss']:.10f}, free-run deterministic loss={fit['deterministic_loss']:.10f}")
    print(f"QP fit time={fit['qp_fit_time']:.3f}s, solve={fit['qp_solve_time']:.4f}s")
    print(f"plot={plot_path}")


def fit_day(frame, start, warmup_hours=24, steps_per_hour=60, **kwargs):
    start = day_start(start)
    end, day = selected_day(frame, start)
    output_model, output_fit = fit_expected_output(frame, start)
    data, fine_index, total_start, model_T = model_input(frame, start, steps_per_hour, warmup_hours, output_model)
    model = wm_4(model_T, len(fine_index), model_output(data, fine_index))
    fit = qp_params(model).fit(data, fitted_initial_condition(data), fit_diffusion=True, verbose=False, **kwargs)
    return end, day, output_fit, fit, model, fine_index, total_start


def main():
    parser = ArgumentParser(description="Fit wm_4 to one calendar day using a robust weather-aware exogenous output model O.")
    parser.add_argument("--day", default="2022-04-18", help="Calendar day to score.")
    parser.add_argument("--warmup-hours", type=int, default=24, help="Hours simulated before the target day; default 24.")
    parser.add_argument("--steps-per-hour", type=int, default=60, help="Internal wm_4 response substeps per physical hour; default 60 gives a one-minute grid.")
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    start = day_start(args.day)
    end, day, output_fit, fit, model, fine_index, total_start = fit_day(frame, start, args.warmup_hours, args.steps_per_hour)
    plot_path = save_plot(start, end, day, fine_index, fit, args.warmup_hours)
    print_result(start, end, total_start, output_fit, fit, model, plot_path, args.steps_per_hour, args.warmup_hours)


if __name__ == "__main__": main()
