from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, INIT_F, aligned_wm4_data, day_start, fit_expected_output, fitted_initial_condition, model_input, model_output
from src.exe.run_model.wm_4.output_model import output_callable
from src.model.wm_4 import wm_4
from src.params.wm_4 import qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction"


def fit_history(frame, predict_start, parameter_weeks, warmup_hours, steps_per_hour, fit_diffusion=True, **kwargs):
    predict_start, parameter_weeks = day_start(predict_start), int(parameter_weeks)
    if parameter_weeks < 1: raise ValueError("parameter_weeks must be at least 1.")
    output_model, output_fit = fit_expected_output(frame, predict_start)
    fit_days = [predict_start - pd.Timedelta(weeks=k) for k in range(1, parameter_weeks + 1)]
    datasets, indices, origins, init_conds = [], [], [], []
    model_T = None

    for day in fit_days:
        data, fine_index, origin, day_T = model_input(frame, day, steps_per_hour, warmup_hours, output_model)
        if model_T is None: model_T = day_T
        elif tuple(day_T) != tuple(model_T): raise ValueError("Historical wm_4 fitting days must use the same model horizon.")
        datasets.append(data); indices.append(fine_index); origins.append(origin); init_conds.append(fitted_initial_condition(data))

    model = wm_4(model_T, len(indices[0]), model_output(datasets[0], indices[0]))
    fit = qp_params(model).fit_many(
        datasets, np.vstack(init_conds),
        fit_diffusion=fit_diffusion, verbose=False, **kwargs,
    )
    fit["training_days"] = fit_days
    fit["fit_trajectories"] = len(datasets)
    return output_model, output_fit, fit, model, indices, origins


def prediction_setup(frame, start, output_model, steps_per_hour):
    start = day_start(start)
    end = start + pd.Timedelta(hours=DAY_HOURS)
    steps_per_hour = int(steps_per_hour)
    n_steps = DAY_HOURS * steps_per_hour + 1
    elapsed = np.linspace(0.0, float(DAY_HOURS), n_steps)
    fine_index = start + pd.to_timedelta(elapsed, unit="h")

    source_index = pd.date_range(start, end, freq="h")
    source = frame.reindex(source_index)[["I", "U", "Temp"]].copy()
    if source.iloc[:-1].isna().any().any(): raise ValueError(f"Incomplete prediction I/U/temperature data between {start} and {end}.")
    source["U"] = source["U"].ffill().bfill()
    source["Temp"] = source["Temp"].interpolate(method="time", limit_direction="both").ffill().bfill()

    hour_bin = np.minimum(np.floor(elapsed + 1e-12).astype(int), DAY_HOURS)
    prices = source["U"].to_numpy(dtype=float)
    temperature = np.interp(elapsed, np.arange(DAY_HOURS + 1, dtype=float), source["Temp"].to_numpy(dtype=float))
    O_values = output_model.predict(fine_index, temperature)
    model = wm_4((0.0, 1.0), n_steps, output_callable(fine_index, O_values))

    def U(t):
        hour = min(int(np.floor(float(t) * DAY_HOURS + 1e-12)), DAY_HOURS)
        return float(prices[hour])

    def Time(t): return float(t) * DAY_HOURS

    actual = source.iloc[:DAY_HOURS].copy()
    return end, actual, U, Time, model, fine_index


def historical_prediction_state(fit, warmup_hours, steps_per_hour):
    states = fit.get("deterministic_states")
    if not states: raise ValueError("Historical prediction state requires a multi-trajectory WM4 fit.")
    index = int(warmup_hours) * int(steps_per_hour)
    values = np.array([[state["F"][index], state["R"][index]] for state in states], dtype=float)
    init = np.median(values, axis=0)
    init[0] = np.clip(init[0], -1.0, 1.0)
    return init


def optimized_prediction_state(frame, start, model, r, steps_per_hour, output_model):
    data, fine_index, _, _ = model_input(frame, start, steps_per_hour, 0, output_model)
    helper = qp_params(model)
    data = helper._prepare_data(data)
    base = helper._state_from_parameters(data, np.array([0.0, 0.0]), r)
    f_unit = helper._state_from_parameters(data, np.array([1.0, 0.0]), r)
    r_unit = helper._state_from_parameters(data, np.array([0.0, 1.0]), r)
    obs = np.asarray(data["observation_mask"], dtype=bool)
    X = np.column_stack(((f_unit["I"] - base["I"])[obs], (r_unit["I"] - base["I"])[obs]))
    y = data["I"][obs] - base["I"][obs]
    init = model._least_squares_qp(X, y, lb=np.array([-1.0, -1.0]), ub=np.array([1.0, 1.0]), ridge=model._FIT_RIDGE)
    state = helper._state_from_parameters(data, init, r, full=True)
    loss = float(np.mean((data["I"][obs] - state["I"][obs]) ** 2))
    return np.asarray(init, dtype=float), state, fine_index, loss


def deviation_metrics(observed, predicted):
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    return {
        "mse": float(np.mean(residual ** 2)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def save_plot(predict_start, predict_end, actual, fine_index, state, path, training_weeks):
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(actual.index, actual["I"], label="Observed same-weekday input I")
    axes[0].scatter(actual.index, actual["I"], s=18)
    axes[0].plot(fine_index, state["I"], label="WM4 prediction I=O+R")
    axes[0].plot(fine_index, state["O"], label="Weather-aware expected output O", alpha=0.8)
    axes[0].plot(fine_index, state["U"], label="Normalized DK1 spot price U", alpha=0.8)
    axes[0].set_ylabel("Normalized input / price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(fine_index, state["U"], label="Known prediction-day normalized price U")
    axes[1].plot(fine_index, state["R_dot"], label="Flexibility acceleration dR/dt")
    axes[1].set_ylabel("Price / acceleration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(fine_index, state["F"], label="Predicted flexibility state F")
    axes[2].plot(fine_index, state["R"], label="Predicted realized flexibility flow R")
    axes[2].set_ylabel("State / rate")
    axes[2].set_xlabel("Physical time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"wm_4 same-weekday prediction | joint QP over previous {training_weeks} matching weekdays -> {predict_start:%Y-%m-%d}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Predict one wm_4 calendar day with weather-aware O and a joint multi-day QP parameter fit.")
    parser.add_argument("--day", default="2022-08-22", help="Calendar day to predict.")
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS, help="Previous matching weekdays fitted jointly for WM4 parameters; default 5.")
    parser.add_argument("--warmup-hours", type=int, default=24)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--deterministic", action="store_true", help="Skip diffusion parameter estimation.")
    parser.add_argument("--optimize-prediction-state", action="store_true", help="Diagnostic only: fit prediction-day [F0,R0] from that day's observed I instead of using historical fitted states.")
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    predict_start = day_start(args.day)
    output_model, output_fit, fit, _, _, _ = fit_history(
        frame, predict_start, args.baseline_weeks, args.warmup_hours, args.steps_per_hour,
        fit_diffusion=not args.deterministic,
    )
    predict_end, actual, U, Time, predict_model, fine_index = prediction_setup(frame, predict_start, output_model, args.steps_per_hour)

    if args.optimize_prediction_state:
        init, state, fine_index, init_fit_mse = optimized_prediction_state(
            frame, predict_start, predict_model, fit["r"], args.steps_per_hour, output_model,
        )
    else:
        init = historical_prediction_state(fit, args.warmup_hours, args.steps_per_hour)
        state = predict_model.deterministic_solver(init, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})
        init_fit_mse = np.nan

    hourly = np.arange(DAY_HOURS) * args.steps_per_hour
    observed = actual["I"].to_numpy(dtype=float)
    baseline = state["O"][hourly]
    predicted = state["I"][hourly]
    baseline_metrics, prediction_metrics = deviation_metrics(observed, baseline), deviation_metrics(observed, predicted)

    path = OUTPUT / f"daily_prediction_{predict_start:%Y-%m-%d}.png"
    save_plot(predict_start, predict_end, actual, fine_index, state, path, args.baseline_weeks)

    fit_days = ", ".join(day.strftime("%Y-%m-%d") for day in fit["training_days"])
    print(f"prediction day={predict_start} -> {predict_end}")
    print(f"WM4 parameter trajectories: {fit_days}")
    print(f"O training={output_fit['fit_start']} -> {output_fit['fit_end']}, samples={output_fit['samples']}, MSE={output_fit['mse']:.8f}")
    r = fit["r"]
    print(f"r = [r_F={r[0]:.8f}, r_U={r[1]:.8f}, r_R={r[2]:.8f}]")
    if not args.deterministic: print(f"diffusion: sigma_0={fit['sigma_0']:.8f}, sigma_I={fit['sigma_I']:.8f}")
    print(f"initial_cond = [{init[0]:.8f}, {init[1]:.8f}], optimized_from_prediction_day = {str(args.optimize_prediction_state).lower()}")
    print()
    print(f"{'':<8}{'MSE':>14}{'RMSE':>14}{'MAE':>14}{'bias':>14}")
    print(f"{'WM4':<8}{prediction_metrics['mse']:>14.10f}{prediction_metrics['rmse']:>14.10f}{prediction_metrics['mae']:>14.10f}{prediction_metrics['bias']:>14.10f}")
    print(f"{'O(t)':<8}{baseline_metrics['mse']:>14.10f}{baseline_metrics['rmse']:>14.10f}{baseline_metrics['mae']:>14.10f}{baseline_metrics['bias']:>14.10f}")


if __name__ == "__main__": main()
