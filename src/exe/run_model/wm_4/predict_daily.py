from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_3.fit_weekly import aligned_data
from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, INIT, day_start, fit_expected_output, model_input
from src.model.wm_4 import wm_4
from src.params.wm_4 import brute_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction"


def fit_history(frame, predict_start, baseline_weeks, warmup_hours, steps_per_hour, fit_diffusion=True, **kwargs):
    O, output_fit = fit_expected_output(frame, predict_start, baseline_weeks)
    fit_days = list(output_fit["baseline_days"])
    datasets, indices, origins = [], [], []
    model_T = None

    for day in fit_days:
        data, fine_index, origin, day_T = model_input(frame, day, steps_per_hour, warmup_hours)
        if model_T is None: model_T = day_T
        elif tuple(day_T) != tuple(model_T): raise ValueError("Historical wm_4 fitting days must use the same model horizon.")
        datasets.append(data); indices.append(fine_index); origins.append(origin)

    observation_mask = np.asarray(datasets[0]["observation_mask"], dtype=bool)
    time = np.asarray(datasets[0]["Time"], dtype=float)
    if any(not np.array_equal(np.asarray(data["observation_mask"], dtype=bool), observation_mask) for data in datasets[1:]): raise ValueError("Historical wm_4 fitting days must share the same observation mask.")
    if any(not np.allclose(np.asarray(data["Time"], dtype=float), time) for data in datasets[1:]): raise ValueError("Historical wm_4 fitting days must share the same time grid.")

    average_data = {
        "I": np.mean(np.vstack([data["I"] for data in datasets]), axis=0),
        "U": np.mean(np.vstack([data["U"] for data in datasets]), axis=0),
        "PE": np.mean(np.vstack([data["PE"] for data in datasets]), axis=0),
        "Time": time.copy(),
        "observation_mask": observation_mask.copy(),
    }

    model = wm_4(model_T, len(indices[0]), O)
    fit = brute_params(model).fit(
        average_data, INIT,
        fit_initial_state=warmup_hours == 0, fit_diffusion=fit_diffusion, verbose=True,
        **kwargs,
    )
    fit["average_training_data"] = average_data
    fit["fit_trajectories"] = len(datasets)
    return O, output_fit, fit, model, indices, origins


def forecast_signals(frame, start):
    end = start + pd.Timedelta(hours=DAY_HOURS)
    index = pd.date_range(start, end, freq="h")
    source = frame.reindex(index)[["I", "U"]].copy()
    if source.iloc[:-1].isna().any().any(): raise ValueError(f"Incomplete prediction I/U data between {start} and {end}.")
    source["U"] = source["U"].ffill().bfill()
    prices = source["U"].to_numpy(dtype=float)

    def U(t):
        hour = min(int(np.floor(float(t) * DAY_HOURS + 1e-12)), DAY_HOURS)
        return float(prices[hour])

    def Time(t): return float(t) * DAY_HOURS
    return end, source.iloc[:DAY_HOURS].copy(), U, Time


def optimized_prediction_state(frame, start, model, q, kappa_R, steps_per_hour):
    data, fine_index, _, _ = model_input(frame, start, steps_per_hour, 0)
    helper = brute_params(model)
    data = helper._prepare_data(data)
    base = helper._state_from_parameters(data, np.array([0.0, 0.0]), q, kappa_R)
    f_unit = helper._state_from_parameters(data, np.array([1.0, 0.0]), q, kappa_R)
    r_unit = helper._state_from_parameters(data, np.array([0.0, 1.0]), q, kappa_R)
    obs = np.asarray(data["observation_mask"], dtype=bool)
    X = np.column_stack(((f_unit["I"] - base["I"])[obs], (r_unit["I"] - base["I"])[obs]))
    y = data["I"][obs] - base["I"][obs]
    init = model._least_squares_qp(
        X, y,
        lb=np.array([0.0, -1.0]), ub=np.array([1.0, 1.0]),
        ridge=model._FIT_RIDGE,
    )
    state = helper._state_from_parameters(data, init, q, kappa_R, full=True)
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
    axes[0].plot(fine_index, state["I"], label="WM4 one-week-ahead prediction I=O+R")
    axes[0].plot(fine_index, state["O"], label="Expected output O(t)", alpha=0.8)
    axes[0].plot(fine_index, state["U"], label="Normalized DK1 spot price U", alpha=0.8)
    axes[0].set_ylabel("Normalized input / price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(fine_index, state["U"], label="Known prediction-day normalized price U")
    axes[1].plot(fine_index, state["Q"], label="Desired flexibility flow Q")
    axes[1].set_ylabel("Price / desired flow")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(fine_index, state["F"], label="Predicted flexibility state F")
    axes[2].plot(fine_index, state["R"], label="Predicted realized flexibility flow R")
    axes[2].set_ylabel("State / rate")
    axes[2].set_xlabel("Physical time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"wm_4 same-weekday prediction | fit previous {training_weeks} matching weekdays -> predict {predict_start:%Y-%m-%d}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Predict one wm_4 calendar day using O(t) and parameters fitted from the preceding matching weekdays.")
    parser.add_argument("--day", default="2022-08-22", help="Calendar day to predict. The preceding matching weekdays are used for both O(t) construction and parameter fitting.")
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS, help="Previous matching weekdays used for O(t) and joint parameter fitting; default 5.")
    parser.add_argument("--warmup-hours", type=int, default=24)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--grid-points", type=int, default=4)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--deterministic", action="store_true", help="Run deterministic fitting/prediction only and skip diffusion parameter estimation.")
    parser.add_argument("--optimize-prediction-state", action="store_true", help="Diagnostic mode: fit prediction-day [F0,R0] against that day's observed I while keeping q and kappa_R fixed.")
    args = parser.parse_args()

    frame, _ = aligned_data()
    predict_start = day_start(args.day)
    O_predict, output_fit, fit, _, _, _ = fit_history(
        frame, predict_start, args.baseline_weeks, args.warmup_hours, args.steps_per_hour,
        grid_points=args.grid_points, refinements=args.refinements, progress_every=args.progress_every,
        fit_diffusion=not args.deterministic,
    )

    predict_end, actual, U, Time = forecast_signals(frame, predict_start)
    n_steps = DAY_HOURS * args.steps_per_hour + 1
    predict_model = wm_4((0.0, 1.0), n_steps, O_predict)
    if args.optimize_prediction_state:
        init, state, fine_index, init_fit_mse = optimized_prediction_state(
            frame, predict_start, predict_model, fit["q"], fit["kappa_R"], args.steps_per_hour,
        )
    else:
        recent_state = fit["deterministic_state"]
        init = np.array([recent_state["F"][-1], recent_state["R"][-1]], dtype=float)
        state = predict_model.deterministic_solver(init, {
            "U": U, "PE": U, "Time": Time,
            "q": fit["q"], "kappa_R": fit["kappa_R"],
        })
        fine_index = pd.date_range(predict_start, predict_end, periods=n_steps)
        init_fit_mse = np.nan

    hourly = np.arange(DAY_HOURS) * args.steps_per_hour
    observed = actual["I"].to_numpy(dtype=float)
    baseline = state["O"][hourly]
    predicted = state["I"][hourly]
    baseline_metrics = deviation_metrics(observed, baseline)
    prediction_metrics = deviation_metrics(observed, predicted)

    path = OUTPUT / f"daily_prediction_{predict_start:%Y-%m-%d}.png"
    save_plot(predict_start, predict_end, actual, fine_index, state, path, output_fit["weeks"])

    fit_days = ", ".join(day.strftime("%Y-%m-%d") for day in output_fit["baseline_days"])
    print(f"prediction day={predict_start} -> {predict_end}")
    print(f"O(t): previous {output_fit['weeks']} matching {output_fit['weekday']}s: {fit_days}")
    print(f"O(t): remove hourly min/max, average remaining {output_fit['retained_weeks']}")
    print(f"parameter fit: average I(t) and average U(t) across all {output_fit['weeks']} corresponding days")
    print(f"parameter fit days={fit_days}")
    print(f"mode={'deterministic' if args.deterministic else 'stochastic'}")
    print(f"q={np.array2string(fit['q'], precision=8)}")
    print(f"q_F search range={fit['q_f_range']}, expansions={fit['q_f_expansions']}")
    print(f"kappa_R={fit['kappa_R']:.8f}, damping={fit['brute_branch']}, zeta={fit['damping_ratio']:.8f}")
    if not args.deterministic: print(f"sigma_0={fit['sigma_0']:.8f}, sigma_I={fit['sigma_I']:.8f}")
    print(f"average-signal deterministic loss={fit['deterministic_loss']:.10f}")
    if args.optimize_prediction_state:
        print(f"prediction initial [F0,R0] optimized from prediction-day I={np.array2string(init, precision=8)}")
        print(f"prediction-state optimization MSE={init_fit_mse:.10f} (diagnostic; uses observed prediction-day I)")
    else:
        print(f"forecast initial [F,R] reused from averaged historical fit terminal state={np.array2string(init, precision=8)}")
    print(f"O(t) deviation from actual I: MSE={baseline_metrics['mse']:.10f}, RMSE={baseline_metrics['rmse']:.10f}, MAE={baseline_metrics['mae']:.10f}, bias={baseline_metrics['bias']:.10f}")
    print(f"parametrized I=O+R deviation from actual I: MSE={prediction_metrics['mse']:.10f}, RMSE={prediction_metrics['rmse']:.10f}, MAE={prediction_metrics['mae']:.10f}, bias={prediction_metrics['bias']:.10f}")
    print(f"plot={path}")


if __name__ == "__main__": main()
