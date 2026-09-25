from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_3.fit_weekly import aligned_data
from src.model.wm_3 import wm_3
from src.params.wm_3 import brute_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_3" / "linear"
INIT = np.array([0.5, 0.5], dtype=float)
DAY_HOURS = 24
TEMP_REFERENCE = 20.0
TEMP_SPAN = 20.0


def day_start(value): return pd.Timestamp(value).normalize()


def shifted_day(frame, base_start, shift):
    start = base_start + pd.Timedelta(hours=int(shift))
    end = start + pd.Timedelta(hours=DAY_HOURS)
    index = pd.date_range(start, end, freq="h", inclusive="left")
    day = frame.reindex(index)
    if len(day) != DAY_HOURS or day[["I", "U", "Temp"]].isna().any().any(): raise ValueError(f"Shifted day {start} to {end} is not a complete observed 24-hour interval.")
    return start, end, day


def linear_time_function():
    def L(time_hours): return float(np.clip(float(time_hours) / DAY_HOURS, 0.0, 1.0))
    return L, {"intercept": 0.0, "slope": 1.0 / DAY_HOURS}


def linear_temperature_function(reference=TEMP_REFERENCE, span=TEMP_SPAN):
    reference, span = float(reference), float(span)
    if not np.isfinite(reference): raise ValueError("temp_reference must be finite.")
    if not np.isfinite(span) or span <= 0.0: raise ValueError("temp_span must be positive and finite.")
    intercept, slope = reference / span, -1.0 / span

    def H(temperature): return float(np.clip((reference - float(temperature)) / span, 0.0, 1.0))
    return H, {"intercept": intercept, "slope": slope, "reference": reference, "span": span, "cold_limit": reference - span, "warm_limit": reference}


def model_input(frame, day, start, steps_per_hour):
    steps_per_hour = int(steps_per_hour)
    if steps_per_hour < 1: raise ValueError("steps_per_hour must be at least 1.")

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


def candidate_fit(frame, base_start, shift, steps_per_hour, grid_points, refinements, temp_reference, temp_span):
    start, end, day = shifted_day(frame, base_start, shift)
    L, l_params = linear_time_function()
    H, h_params = linear_temperature_function(temp_reference, temp_span)
    data, fine_index = model_input(frame, day, start, steps_per_hour)
    model = wm_3((0.0, 1.0), len(fine_index), L, H)
    fit = brute_params(model).fit(
        data, INIT,
        grid_points=grid_points,
        refinements=refinements,
        fit_initial_state=True,
        fit_diffusion=False,
        verbose=False,
    )
    return {
        "shift": int(shift), "start": start, "end": end, "day": day,
        "L": L, "H": H, "L_params": l_params, "H_params": h_params,
        "data": data, "fine_index": fine_index, "model": model, "fit": fit,
        "score": float(fit["deterministic_loss"]),
    }


def optimize_slice(frame, base_start, steps_per_hour, grid_points, refinements, temp_reference, temp_span):
    best, start_time = None, perf_counter()
    for shift in range(24):
        t0 = perf_counter()
        try: candidate = candidate_fit(frame, base_start, shift, steps_per_hour, grid_points, refinements, temp_reference, temp_span)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            print(f"[linear] shift={shift:02d}:00 status=failed elapsed={perf_counter() - t0:.2f}s error={error}")
            continue
        print(f"[linear] shift={shift:02d}:00 loss={candidate['score']:.10f} elapsed={perf_counter() - t0:.2f}s")
        if best is None or candidate["score"] < best["score"]: best = candidate

    if best is None: raise RuntimeError("No admissible linear-feature wm_3 candidate was found.")
    print(f"[linear] optimized shift={best['shift']:02d}:00 loss={best['score']:.10f} search_time={perf_counter() - start_time:.2f}s")
    return best


def final_fit(frame, best, steps_per_hour, grid_points, refinements, progress_every, temp_reference, temp_span):
    start, end, day = shifted_day(frame, best["start"].normalize(), best["start"].hour)
    L, l_params = linear_time_function()
    H, h_params = linear_temperature_function(temp_reference, temp_span)
    data, fine_index = model_input(frame, day, start, steps_per_hour)
    model = wm_3((0.0, 1.0), len(fine_index), L, H)
    fit = brute_params(model).fit(
        data, INIT,
        grid_points=grid_points,
        refinements=refinements,
        fit_initial_state=True,
        fit_diffusion=True,
        verbose=True,
        progress_every=progress_every,
    )
    return {
        **best,
        "start": start, "end": end, "day": day, "L": L, "H": H,
        "L_params": l_params, "H_params": h_params,
        "data": data, "fine_index": fine_index, "model": model, "fit": fit,
        "score": float(fit["deterministic_loss"]),
    }


def save_plot(result):
    start, end, day = result["start"], result["end"], result["day"]
    fit, fine_index = result["fit"], result["fine_index"]
    state, obs = fit["deterministic_state"], fit["observation_mask"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(day.index, day["I"], label="Observed hourly aggregate input I")
    axes[0].scatter(day.index, day["I"], s=18, label="Observed I samples")
    axes[0].plot(fine_index, state["I"], label="Fitted wm_3 input I")
    axes[0].scatter(fine_index[obs], state["I"][obs], s=18, label="WM3 prediction at observation times")
    axes[0].set_ylabel("Normalized input")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(fine_index, state["L"], label="Linear L(t)")
    axes[1].plot(fine_index, state["H"], label="Fixed-scale cold H(T)")
    axes[1].set_ylabel("Linear features")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(fine_index, state["U"], label="Normalized DK1 price U")
    axes[2].plot(fine_index, state["V"], label="V")
    axes[2].plot(fine_index, state["R"], label="R")
    axes[2].set_ylabel("State / price")
    axes[2].set_xlabel("Physical time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"wm_3 linear-feature fit {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} | fixed cold temperature scale")
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"linear_fit_{start:%Y-%m-%d_%H%M}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def print_result(result, plot_path, steps_per_hour):
    fit, model = result["fit"], result["model"]
    lp, hp = result["L_params"], result["H_params"]
    print(f"optimized slice={result['start']} -> {result['end']}")
    print(f"model time: tau in [0, 1], dt={model.dt:.12f}, response grid={3600.0 / steps_per_hour:.4f} physical seconds")
    print("feature fitting: no long-term L/H regression; L is a fixed linear time basis and H is a fixed physical cold-response scale.")
    print(f"L(t)={lp['intercept']:.8f}+{lp['slope']:.8f}*hour")
    print(f"H(T)=clip(({hp['reference']:.4f}-T)/{hp['span']:.4f}, 0, 1), active physical range=[{hp['cold_limit']:.4f}, {hp['warm_limit']:.4f}] C")
    print(f"observed temperature range=[{result['day']['Temp'].min():.4f}, {result['day']['Temp'].max():.4f}] C")
    print(f"alpha={np.array2string(fit['alpha'], precision=8)}")
    print(f"beta={np.array2string(fit['beta'], precision=8)}")
    print(f"kappa_R={fit['kappa_R']:.8f}")
    print(f"sigma_0={fit['sigma_0']:.8f}, sigma_O={fit['sigma_O']:.8f}")
    print(f"initial condition={np.array2string(fit['init_cond'], precision=8)}")
    print(f"deterministic loss={fit['deterministic_loss']:.10f}")
    print(f"brute d_V={fit['brute_d_V']:.8f}, search range=[-2, 0)")
    print(f"brute fit time={fit['brute_fit_time']:.3f}s")
    print(f"plot={plot_path}")


def main():
    parser = ArgumentParser(description="Fit wm_3 on one optimized natural 24-hour slice with fixed linear L(t) and physically scaled cold H(T).")
    parser.add_argument("--day", default="2022-01-03", help="Calendar day whose 24 starting hours are searched.")
    parser.add_argument("--steps-per-hour", type=int, default=60, help="Final internal response steps per physical hour.")
    parser.add_argument("--shift-steps-per-hour", type=int, default=2, help="Coarse response grid used only while selecting the 24-hour slice.")
    parser.add_argument("--shift-grid-points", type=int, default=3)
    parser.add_argument("--shift-refinements", type=int, default=1)
    parser.add_argument("--grid-points", type=int, default=4)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--temp-reference", type=float, default=TEMP_REFERENCE, help="Temperature in C where the cold-response H(T) reaches 0.")
    parser.add_argument("--temp-span", type=float, default=TEMP_SPAN, help="Temperature span in C over which H(T) rises linearly from 0 to 1 as temperature falls.")
    args = parser.parse_args()

    frame, _ = aligned_data()
    base_start = day_start(args.day)
    best = optimize_slice(frame, base_start, args.shift_steps_per_hour, args.shift_grid_points, args.shift_refinements, args.temp_reference, args.temp_span)
    result = final_fit(frame, best, args.steps_per_hour, args.grid_points, args.refinements, args.progress_every, args.temp_reference, args.temp_span)
    plot_path = save_plot(result)
    print_result(result, plot_path, args.steps_per_hour)


if __name__ == "__main__": main()
