from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, aligned_wm4_data, day_start, fit_expected_output, fitted_initial_condition, model_input, model_output
from src.model.wm_4 import wm_4
from src.params.wm_4 import brute_force_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "brute_force_f0"


def fit_history_brute_force(frame, predict_start, parameter_weeks, steps_per_hour, **kwargs):
    predict_start, parameter_weeks = day_start(predict_start), int(parameter_weeks)
    if parameter_weeks < 1: raise ValueError("parameter_weeks must be at least 1.")
    output_model, output_fit = fit_expected_output(frame, predict_start)
    fit_days = [predict_start - pd.Timedelta(weeks=k) for k in range(1, parameter_weeks + 1)]
    datasets, indices, seeds = [], [], []
    model_T = None

    for day in fit_days:
        data, fine_index, _, day_T = model_input(frame, day, steps_per_hour, 0, output_model)
        if model_T is None: model_T = day_T
        elif tuple(day_T) != tuple(model_T): raise ValueError("Historical wm_4 fitting days must use the same model horizon.")
        datasets.append(data); indices.append(fine_index); seeds.append(fitted_initial_condition(data))

    model = wm_4(model_T, len(indices[0]), model_output(datasets[0], indices[0]))
    fit = brute_force_params(model).fit_many(datasets, np.vstack(seeds), **kwargs)
    fit["training_days"] = fit_days
    return output_model, output_fit, fit, model


def save_profile(fit, output):
    profile = pd.DataFrame(fit["f0_profile"]).sort_values("F0").reset_index(drop=True)
    profile.to_csv(output / "f0_profile.csv", index=False)
    successful = profile.loc[profile["success"].astype(bool)].copy()
    if successful.empty: return profile

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(successful["F0"], successful["qp_loss"])
    ax.axvline(float(fit["F0"]), linestyle="--", label=f"best F0={fit['F0']:.4f}")
    ax.axvline(0.0, linestyle=":", label="neutral F0=0")
    ax.set_xlabel("Shared fixed F0")
    ax.set_ylabel("Rate-transition QP loss")
    ax.set_title("ARM (Affine Revised Model) brute-force profile for shared midnight flexibility state")
    ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "f0_qp_loss_profile.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(successful["F0"], successful["r_F"], label="r_F")
    ax.plot(successful["F0"], successful["r_U"], label="r_U")
    ax.plot(successful["F0"], successful["r_R"], label="r_R")
    ax.axvline(float(fit["F0"]), linestyle="--", label=f"best F0={fit['F0']:.4f}")
    ax.set_xlabel("Shared fixed F0")
    ax.set_ylabel("Fitted coefficient")
    ax.set_title("ARM (Affine Revised Model) coefficients across the F0 profile")
    ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "f0_parameter_profile.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return profile


def save_results(predict_day, output_fit, fit, output):
    output.mkdir(parents=True, exist_ok=True)
    profile = save_profile(fit, output)
    states = np.asarray(fit["init_cond"], dtype=float)
    training = pd.DataFrame({
        "prediction_day": predict_day,
        "training_day": fit["training_days"],
        "F_0": states[:, 0],
        "R_0": states[:, 1],
    })
    training.to_csv(output / "training_states.csv", index=False)

    r = np.asarray(fit["r"], dtype=float)
    summary = pd.DataFrame([{
        "prediction_day": predict_day,
        "training_days": len(fit["training_days"]),
        "best_F0": float(fit["F0"]),
        "qp_loss": float(fit["qp_loss"]),
        "deterministic_loss": float(fit["deterministic_loss"]),
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "R0_mean": float(np.mean(states[:, 1])),
        "R0_std": float(np.std(states[:, 1])),
        "candidates": int(fit["brute_force_candidates"]),
        "feasible_candidates": int(fit["brute_force_feasible_candidates"]),
        "fit_time": float(fit["qp_fit_time"]),
        "O_fit_mse": float(output_fit["mse"]),
    }])
    summary.to_csv(output / "summary.csv", index=False)
    return profile, summary


def print_results(profile, summary):
    row = summary.iloc[0]
    print(f"best shared F0={row['best_F0']:.8f}")
    print(f"QP rate-transition loss={row['qp_loss']:.10f}")
    print(f"free-run deterministic loss={row['deterministic_loss']:.10f}")
    print(f"r=[r_F={row['r_F']:.8f}, r_U={row['r_U']:.8f}, r_R={row['r_R']:.8f}]")
    print(f"R0 mean={row['R0_mean']:.8f}, std={row['R0_std']:.8f}")
    print(f"feasible F0 candidates={int(row['feasible_candidates'])}/{int(row['candidates'])}")
    successful = profile.loc[profile["success"].astype(bool)]
    if not successful.empty:
        neutral = successful.iloc[np.argmin(np.abs(successful["F0"].to_numpy(dtype=float)))]
        delta = float(neutral["qp_loss"] - row["qp_loss"])
        relative = 100.0 * delta / float(row["qp_loss"]) if float(row["qp_loss"]) > 0.0 else np.nan
        print(f"nearest neutral candidate F0={float(neutral['F0']):.8f}, loss={float(neutral['qp_loss']):.10f}")
        print(f"neutral-minus-best QP loss={delta:.10f} ({relative:.4f}% of best loss)")


def main():
    parser = ArgumentParser(description="Profile a shared fixed F0 for historical WM4 rate-dynamics fits.")
    parser.add_argument("--day", default="2022-04-18", help="Prediction day whose previous matching weekdays define the fitting set.")
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--f0-min", type=float, default=-1.0)
    parser.add_argument("--f0-max", type=float, default=1.0)
    parser.add_argument("--f0-steps", type=int, default=81)
    parser.add_argument("--max-state-iter", type=int, default=25)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    predict_day = day_start(args.day)
    _, output_fit, fit, _ = fit_history_brute_force(
        frame, predict_day, args.baseline_weeks, args.steps_per_hour,
        f0_min=args.f0_min, f0_max=args.f0_max, f0_steps=args.f0_steps,
        max_state_iter=args.max_state_iter,
        state_tol=args.state_tol, verbose=args.verbose,
    )
    output = OUTPUT / f"{predict_day:%Y-%m-%d}"
    profile, summary = save_results(predict_day, output_fit, fit, output)
    print_results(profile, summary)
    print(f"output={output}")


if __name__ == "__main__": main()
