from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start, fit_expected_output, model_input, model_output
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup
from src.model.wm_4 import wm_4
from src.params.wm_4 import optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "initial_f0_sensitivity"
DEFAULT_F0_GUESSES = np.linspace(-1.0, 1.0, 9)


def prepare_history(frame, prediction_day, weeks, steps_per_hour):
    prediction_day, weeks, steps_per_hour = day_start(prediction_day), int(weeks), int(steps_per_hour)
    if weeks < 1: raise ValueError("weeks must be at least 1.")
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2.")

    output_model, output_fit = fit_expected_output(frame, prediction_day)
    training_days = [prediction_day - pd.Timedelta(weeks=k) for k in range(1, weeks + 1)]
    datasets, indices, r0_seeds = [], [], []
    model_T = None

    for day in training_days:
        data, fine_index, _, day_T = model_input(frame, day, steps_per_hour, 0, output_model)
        if model_T is None: model_T = day_T
        elif tuple(day_T) != tuple(model_T): raise ValueError("Historical wm_4 fitting days must use the same model horizon.")
        datasets.append(data); indices.append(fine_index)
        r0_seeds.append(float(data["I"][0] - data["O"][0]))

    model = wm_4(model_T, len(indices[0]), model_output(datasets[0], indices[0]))
    return output_model, output_fit, training_days, datasets, np.asarray(r0_seeds, dtype=float), model


def fit_from_f0_guess(model, datasets, r0_seeds, f0_guess, max_state_iter=50, state_tol=1e-8):
    f0_guess = float(f0_guess)
    if not -1.0 <= f0_guess <= 1.0: raise ValueError("Every F0 initial guess must lie in [-1, 1].")
    seeds = np.column_stack((
        np.full(len(datasets), f0_guess, dtype=float),
        np.asarray(r0_seeds, dtype=float),
    ))
    return optimized_qp_params(model).fit_many(
        datasets, seeds, fit_initial_state=True,
        max_state_iter=max_state_iter, state_tol=state_tol, verbose=False,
    )


def evaluate_oracle(frame, prediction_day, output_model, fit, steps_per_hour):
    prediction_day = day_start(prediction_day)
    _, actual, U, Time, model, fine_index = prediction_setup(frame, prediction_day, output_model, steps_per_hour)
    prediction_data, _, _, _ = model_input(frame, prediction_day, steps_per_hour, 0, output_model)
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(prediction_data, fit["r"])

    hourly_index = np.arange(DAY_HOURS, dtype=int) * int(steps_per_hour)
    observed = actual["I"].to_numpy(dtype=float)
    predicted = oracle["deterministic_state"]["I"][hourly_index]
    metrics = deviation_metrics(observed, predicted)
    return {
        "actual": actual, "model": model, "fine_index": fine_index,
        "oracle": oracle, "metrics": metrics,
        "U": U, "Time": Time,
    }


def result_row(f0_guess, fit, oracle_result):
    r = np.asarray(fit["r"], dtype=float)
    q = np.asarray(fit["q"], dtype=float)
    states = np.asarray(fit["init_cond"], dtype=float)
    metrics = oracle_result["metrics"]
    oracle_init = np.asarray(oracle_result["oracle"]["init_cond"], dtype=float)

    row = {
        "F0_guess": float(f0_guess),
        "oracle_mse": metrics["mse"], "oracle_rmse": metrics["rmse"], "oracle_mae": metrics["mae"], "oracle_bias": metrics["bias"],
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "q_F": q[0], "q_U": q[1], "kappa_R": float(fit["kappa_R"]),
        "qp_loss": float(fit["qp_loss"]), "historical_mse": float(fit["deterministic_loss"]),
        "state_iterations": int(fit["state_iterations"]),
        "oracle_F0": oracle_init[0], "oracle_R0": oracle_init[1],
    }
    for i, state in enumerate(states, start=1):
        row[f"fit_F0_{i}"] = float(state[0])
        row[f"fit_R0_{i}"] = float(state[1])
    return row


def save_best_oracle_plot(prediction_day, training_days, best_row, oracle_result, path):
    prediction_day = day_start(prediction_day)
    actual = oracle_result["actual"]
    fine_index = oracle_result["fine_index"]
    state = oracle_result["oracle"]["deterministic_state"]
    metrics = oracle_result["metrics"]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(actual.index, actual["I"], marker="o", label="Observed input I")
    axes[0].plot(fine_index, state["I"], label="Best oracle ARM I")
    axes[0].plot(fine_index, state["O"], label="Expected output O", alpha=0.8)
    axes[0].set_ylabel("Normalized input")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)

    axes[1].plot(fine_index, state["U"], label="Normalized price U")
    axes[1].plot(fine_index, state["R_dot"], label="dR/dt")
    axes[1].set_ylabel("Price / acceleration")
    axes[1].legend(); axes[1].grid(True, alpha=0.25)

    axes[2].plot(fine_index, state["F"], label="Oracle flexibility F")
    axes[2].plot(fine_index, state["R"], label="Oracle rate R")
    axes[2].set_ylabel("State / rate"); axes[2].set_xlabel("Physical time")
    axes[2].legend(); axes[2].grid(True, alpha=0.25)

    fit_days = ", ".join(day.strftime("%Y-%m-%d") for day in training_days)
    fig.suptitle(
        f"ARM (Affine Revised Model) best oracle after historical F0-seed sensitivity | prediction {prediction_day:%Y-%m-%d}\n"
        f"best historical seed F0={best_row['F0_guess']:.3f} | MSE={metrics['mse']:.6f}, RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}\n"
        f"training days: {fit_days}"
    )
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def run(frame, prediction_day, weeks=BASELINE_WEEKS, steps_per_hour=60, f0_guesses=DEFAULT_F0_GUESSES, max_state_iter=50, state_tol=1e-8):
    prediction_day = day_start(prediction_day)
    f0_guesses = np.asarray(f0_guesses, dtype=float).reshape(-1)
    if f0_guesses.size < 1 or not np.all(np.isfinite(f0_guesses)): raise ValueError("f0_guesses must contain at least one finite value.")
    if np.any(f0_guesses < -1.0) or np.any(f0_guesses > 1.0): raise ValueError("Every F0 initial guess must lie in [-1, 1].")

    output_model, output_fit, training_days, datasets, r0_seeds, model = prepare_history(
        frame, prediction_day, weeks, steps_per_hour,
    )

    rows, details = [], []
    for f0_guess in f0_guesses:
        try:
            fit = fit_from_f0_guess(
                model, datasets, r0_seeds, f0_guess,
                max_state_iter=max_state_iter, state_tol=state_tol,
            )
            oracle_result = evaluate_oracle(frame, prediction_day, output_model, fit, steps_per_hour)
            row = result_row(f0_guess, fit, oracle_result)
            row["success"] = True; row["failure"] = ""
            rows.append(row); details.append((float(f0_guess), fit, oracle_result, row))
            print(
                f"F0 seed={f0_guess: .4f} | "
                f"MSE={row['oracle_mse']:.10f} RMSE={row['oracle_rmse']:.10f} MAE={row['oracle_mae']:.10f} | "
                f"r=[{row['r_F']:.8f}, {row['r_U']:.8f}, {row['r_R']:.8f}] | "
                f"q=[{row['q_F']:.8f}, {row['q_U']:.8f}] kappa_R={row['kappa_R']:.8f}"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            rows.append({
                "F0_guess": float(f0_guess), "success": False, "failure": str(exc),
                "oracle_mse": np.nan, "oracle_rmse": np.nan, "oracle_mae": np.nan, "oracle_bias": np.nan,
                "r_F": np.nan, "r_U": np.nan, "r_R": np.nan,
                "q_F": np.nan, "q_U": np.nan, "kappa_R": np.nan,
                "qp_loss": np.nan, "historical_mse": np.nan, "state_iterations": np.nan,
                "oracle_F0": np.nan, "oracle_R0": np.nan,
            })
            print(f"F0 seed={f0_guess: .4f} | FAILED: {exc}")

    if not details: raise RuntimeError("All F0 sensitivity fits failed.")
    summary = pd.DataFrame(rows)
    best = min(details, key=lambda item: item[3]["oracle_mse"])
    return {
        "prediction_day": prediction_day, "training_days": training_days,
        "output_fit": output_fit, "summary": summary,
        "best_f0_guess": best[0], "best_fit": best[1], "best_oracle": best[2],
        "r0_seeds": r0_seeds,
    }


def print_summary(result):
    summary = result["summary"]
    columns = [
        "F0_guess", "oracle_mse", "oracle_rmse", "oracle_mae",
        "r_F", "r_U", "r_R", "q_F", "q_U", "kappa_R",
        "historical_mse", "qp_loss", "state_iterations",
        "oracle_F0", "oracle_R0",
    ]
    print()
    print(f"prediction day={result['prediction_day']:%Y-%m-%d}")
    print("training days=" + ", ".join(day.strftime("%Y-%m-%d") for day in result["training_days"]))
    print("R0 seeds=" + np.array2string(result["r0_seeds"], precision=6, separator=", "))
    print()
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:.8f}"))

    best = summary.loc[summary["oracle_mse"].idxmin()]
    print()
    print(
        f"best oracle initialization: F0 seed={best['F0_guess']:.6f}, "
        f"MSE={best['oracle_mse']:.10f}, RMSE={best['oracle_rmse']:.10f}, MAE={best['oracle_mae']:.10f}"
    )
    print(f"best r=[{best['r_F']:.10f}, {best['r_U']:.10f}, {best['r_R']:.10f}]")
    print(f"best q=[{best['q_F']:.10f}, {best['q_U']:.10f}], kappa_R={best['kappa_R']:.10f}")


def main():
    parser = ArgumentParser(description="Test WM4 optimized-QP sensitivity to the historical F0 initial guess for one prediction day.")
    parser.add_argument("--day", default="2022-08-22", help="Prediction day.")
    parser.add_argument("--weeks", type=int, default=BASELINE_WEEKS, help="Number of previous corresponding weekdays; default 5.")
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--f0-guesses", type=float, nargs="+", default=DEFAULT_F0_GUESSES.tolist(), help="Common F0 seed applied to every historical weekday. Default: -1 to 1 in steps of 0.25.")
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    result = run(
        frame, args.day, args.weeks, args.steps_per_hour, args.f0_guesses,
        max_state_iter=args.max_state_iter, state_tol=args.state_tol,
    )

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    stem = f"{result['prediction_day']:%Y-%m-%d}_{int(args.weeks)}_weekdays"
    csv_path = output / f"{stem}_f0_sensitivity.csv"
    plot_path = output / f"{stem}_best_oracle.png"
    result["summary"].to_csv(csv_path, index=False)
    best_row = result["summary"].loc[result["summary"]["oracle_mse"].idxmin()]
    save_best_oracle_plot(result["prediction_day"], result["training_days"], best_row, result["best_oracle"], plot_path)

    print_summary(result)
    print()
    print(f"summary={csv_path}")
    print(f"best oracle plot={plot_path}")


if __name__ == "__main__": main()
