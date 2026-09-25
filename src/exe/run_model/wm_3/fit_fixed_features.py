from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.exe.run_model.wm_3.fit_daily import day_start, fit_time_function, model_input, shifted_day
from src.exe.run_model.wm_3.fit_weekly import aligned_data, fit_seasonal
from src.model.wm_3 import wm_3
from src.params.wm_3 import brute_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_3" / "fixed_features"
INIT = np.array([0.5, 0.5], dtype=float)


def save_plot(start, end, day, fine_index, fit, time_fit, seasonal):
    state, obs = fit["deterministic_state"], fit["observation_mask"]
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    axes[0].plot(day.index, day["I"], label="Observed hourly aggregate input I")
    axes[0].scatter(day.index, day["I"], s=18, label="Observed I samples")
    axes[0].plot(fine_index, state["I"], label="Fitted wm_3 input I")
    axes[0].scatter(fine_index[obs], state["I"][obs], s=18, label="WM3 prediction at observation times")
    axes[0].set_ylabel("Normalized input")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(fine_index, state["L"], label="Fixed daily linear L(t)")
    axes[1].plot(fine_index, state["H"], label="Fixed seasonal H(T)")
    axes[1].set_ylabel("Fixed features")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(fine_index, state["U"], label="Normalized DK1 price U")
    axes[2].plot(fine_index, state["O"], label="Output O")
    axes[2].plot(fine_index, state["D"], label="D=I-O")
    axes[2].set_ylabel("Price / flow")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(fine_index, state["V"], label="V")
    axes[3].plot(fine_index, state["R"], label="R")
    axes[3].set_ylabel("State")
    axes[3].set_xlabel("Physical time")
    axes[3].legend()
    axes[3].grid(True, alpha=0.25)

    fig.suptitle(
        f"wm_3 fixed-feature fit {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} | "
        f"daily linear L(t), seasonal H(T), shift={time_fit['shift']:02d}:00"
    )
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"fixed_features_fit_{start:%Y-%m-%d_%H%M}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def print_result(base_start, start, end, time_fit, seasonal, fit, model, plot_path, steps_per_hour):
    raw_intercept, raw_slope = time_fit["raw_model"]
    intercept, slope = time_fit["model"]
    c_u = float(fit["I_coeff"][3])

    print(f"base day={base_start:%Y-%m-%d}")
    print(f"selected 24 h slice={start} -> {end}")
    print("fit order: estimate H(T) seasonally and L(t) from the single 24 h raw measurement slice first; keep both fixed while fitting wm_3 parameters.")
    print(f"model time: tau in [0, 1], dt={model.dt:.12f}, response grid={3600.0 / steps_per_hour:.4f} physical seconds")
    print(f"observations={np.count_nonzero(fit['observation_mask'])} hourly samples; state grid points={model.t_steps}")
    print(f"H(T) seasonal raw regression={seasonal['intercept']:.8f}+{seasonal['slope']:.8f}*T")
    print(f"H(T) seasonal fitted range=[{seasonal['prediction_min']:.8f}, {seasonal['prediction_max']:.8f}] before normalization")
    print(f"L(t) raw={raw_intercept:.8f}+{raw_slope:.8f}*hour, MSE={time_fit['score']:.8f}, samples={time_fit['samples']}")
    print(f"L(t) fixed normalized={intercept:.8f}+{slope:.8f}*hour, scale={time_fit['raw_scale']:.8f}")
    print(f"alpha={np.array2string(fit['alpha'], precision=8)}")
    print(f"beta={np.array2string(fit['beta'], precision=8)}")
    print(f"effective input price coefficient c_U={c_u:.8f}")
    print(f"kappa_R={fit['kappa_R']:.8f}")
    print(f"sigma_0={fit['sigma_0']:.8f}, sigma_O={fit['sigma_O']:.8f}")
    print(f"initial condition={np.array2string(fit['init_cond'], precision=8)}")
    print(f"deterministic loss={fit['deterministic_loss']:.10f}")
    print(f"brute d_V={fit['brute_d_V']:.8f}")
    print(f"brute fit time={fit['brute_fit_time']:.3f}s")
    print(f"plot={plot_path}")


def main():
    parser = ArgumentParser(description="Fit wm_3 after independently fixing seasonal H(T) and one daily linear L(t).")
    parser.add_argument("--day", default="2022-01-03", help="Base calendar day whose 24 possible start hours are considered for the daily L(t) fit.")
    parser.add_argument("--steps-per-hour", type=int, default=60, help="Internal wm_3 response steps per physical hour.")
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
        fit_initial_state=True,
        fit_diffusion=True,
        verbose=True,
        progress_every=args.progress_every,
    )

    plot_path = save_plot(start, end, day, fine_index, fit, time_fit, seasonal)
    print_result(base_start, start, end, time_fit, seasonal, fit, model, plot_path, args.steps_per_hour)


if __name__ == "__main__": main()
