from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.exe.run_model.wm_4.analyze_observation_uncertainty import DEFAULT_WEEKS
from src.exe.run_model.wm_4.fit_daily import DAY_HOURS, aligned_wm4_data, day_start, selected_day
from src.exe.run_model.wm_4.fit_constant_diffusion_qp import fit_prediction_day


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "uncertainty" / "diffusion_weekdays"


def build_plot_data(frame, prediction_day, weeks=DEFAULT_WEEKS, steps_per_hour=60, max_state_iter=50, state_tol=1e-8):
    result = fit_prediction_day(
        frame, prediction_day, weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )
    hourly_index = np.arange(DAY_HOURS, dtype=int) * int(steps_per_hour)
    days = []
    for day, state in zip(result["training_days"], result["training_states"]):
        _, observed = selected_day(frame, day)
        days.append({
            "day": day,
            "I": observed["I"].to_numpy(dtype=float),
            "U": np.asarray(state["U"], dtype=float)[hourly_index],
            "F": np.asarray(state["F"], dtype=float)[hourly_index],
        })
    return result, days


def save_plot(result, days, path):
    hours = np.arange(DAY_HOURS, dtype=float)
    diffusion_time = np.asarray(result["prediction_state"]["Time"], dtype=float)
    diffusion = np.asarray(result["fine_g"], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

    axes[0].plot(diffusion_time, diffusion, label=f"Estimated constant diffusion $\\sigma$={result['sigma']:.4f}")
    axes[0].plot(hours, result["hourly"]["demand_residual_std"], marker="o", linestyle="--", label="Historical demand residual RMS")
    axes[0].axhline(result["observation_sigma"], linestyle=":", label=f"Constant demand error std={result['observation_sigma']:.4f}")
    axes[0].plot(hours, result["hourly"]["fitted_propagated_std"], marker=".", label="Total propagated demand std")
    axes[0].set_ylabel("Diffusion / observed std")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    for values in days:
        label = values["day"].strftime("%Y-%m-%d")
        axes[1].plot(hours, values["I"], marker="o", label=label)
        axes[2].step(hours, values["U"], where="post", label=label)
        axes[3].plot(hours, values["F"], label=label)

    axes[1].set_ylabel("Observed input I")
    axes[2].set_ylabel("Normalized price U")
    axes[3].set_ylabel("Fitted flexibility F")
    axes[3].set_xlabel("Hour of day")

    for ax in axes[1:]:
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=min(len(days), 5))

    axes[0].set_xlim(0.0, 24.0)
    axes[3].set_xticks(np.arange(0, 25, 2))
    prediction_day = day_start(result["prediction_day"])
    fig.suptitle(
        f"WM4 diffusion context | {len(days)} corresponding {prediction_day.day_name()}s before {prediction_day:%Y-%m-%d}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Plot locally estimated WM4 diffusion together with input, price and fitted flexibility for the corresponding historical weekdays used in the fit.")
    parser.add_argument("--day", default="2022-08-22", help="Prediction day; the previous corresponding weekdays are used for fitting and plotting.")
    parser.add_argument("--weeks", type=int, default=DEFAULT_WEEKS, help="Number of corresponding historical weekdays; default 5.")
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    result, days = build_plot_data(
        frame, args.day, args.weeks, args.steps_per_hour,
        max_state_iter=args.max_state_iter, state_tol=args.state_tol,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{result['prediction_day']:%Y-%m-%d}_{int(args.weeks)}_weekdays_diffusion_context.png"
    save_plot(result, days, path)

    print(f"prediction day={result['prediction_day']:%Y-%m-%d}")
    print("corresponding weekdays=" + ", ".join(values["day"].strftime("%Y-%m-%d") for values in days))
    print(f"constant process sigma={result['sigma']:.10f}, demand error std={result['observation_sigma']:.10f}")
    print(f"plot={path}")


if __name__ == "__main__": main()
