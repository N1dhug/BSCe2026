from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_3.fit_weekly import aligned_data
from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, INIT, day_start, fit_expected_output, model_input, selected_day
from src.model.wm_4 import wm_4
from src.params.wm_4 import brute_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction"


def expected_output(frame, origin, end, weeks):
    days = pd.date_range(origin.normalize(), end.normalize(), freq="D")
    profiles, metadata = {}, {}
    for day in days: profiles[day], metadata[day] = fit_expected_output(frame, day, weeks)

    def O(elapsed_hours):
        stamp = origin + pd.Timedelta(hours=float(elapsed_hours))
        day = stamp.normalize()
        hour = stamp.hour + stamp.minute / 60.0 + stamp.second / 3600.0
        return profiles[day](hour)

    return O, metadata


def warmup_state(frame, predict_start, warmup_hours, steps_per_hour, baseline_weeks, q, kappa_R):
    warmup_hours, steps_per_hour = int(warmup_hours), int(steps_per_hour)
    if warmup_hours < 1: raise ValueError("Prediction warm-up requires at least one hour of historical data.")
    start = predict_start - pd.Timedelta(hours=warmup_hours)
    n_steps = warmup_hours * steps_per_hour + 1
    elapsed = np.linspace(0.0, float(warmup_hours), n_steps)
    index = pd.date_range(start, predict_start, freq="h")
    source = frame.reindex(index)[["I", "U"]].copy()
    if source.iloc[:-1].isna().any().any(): raise ValueError(f"Incomplete prediction warm-up I/U data between {start} and {predict_start}.")
    source["I"] = source["I"].interpolate(method="time", limit_direction="both").ffill().bfill()
    source["U"] = source["U"].ffill().bfill()

    observed_hours = np.arange(warmup_hours, dtype=float)
    observed_I = source["I"].iloc[:-1].to_numpy(dtype=float)
    I = np.interp(elapsed, observed_hours, observed_I, left=observed_I[0], right=observed_I[-1])
    hour_bin = np.minimum(np.floor(elapsed + 1e-12).astype(int), warmup_hours)
    U = source["U"].to_numpy(dtype=float)[hour_bin]
    observation_mask = np.zeros(n_steps, dtype=bool)
    observation_mask[np.arange(warmup_hours) * steps_per_hour] = True

    O, _ = expected_output(frame, start, predict_start, baseline_weeks)
    model = wm_4((0.0, warmup_hours / DAY_HOURS), n_steps, O)
    helper = brute_params(model)
    data = helper._prepare_data({"I": I, "U": U, "PE": U.copy(), "Time": elapsed, "observation_mask": observation_mask})
    base = helper._state_from_parameters(data, np.array([0.0, 0.0]), q, kappa_R)
    f_unit = helper._state_from_parameters(data, np.array([1.0, 0.0]), q, kappa_R)
    r_unit = helper._state_from_parameters(data, np.array([0.0, 1.0]), q, kappa_R)
    obs = data["observation_mask"]
    X = np.column_stack(((f_unit["I"] - base["I"])[obs], (r_unit["I"] - base["I"])[obs]))
    y = data["I"][obs] - base["I"][obs]
    initial = model._least_squares_qp(
        X, y, lb=np.array([0.0, -1.0]), ub=np.array([1.0, 1.0]), ridge=model._FIT_RIDGE,
    )
    state = helper._state_from_parameters(data, initial, q, kappa_R)
    mse = float(np.mean((data["I"][obs] - state["I"][obs]) ** 2))
    return np.array([state["F"][-1], state["R"][-1]], dtype=float), initial, mse, start


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


def deviation_metrics(observed, predicted):
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    return {
        "mse": float(np.mean(residual ** 2)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def save_plot(fit_start, predict_start, predict_end, actual, fine_index, state, path):
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

    fig.suptitle(f"wm_4 same-weekday prediction | fit {fit_start:%Y-%m-%d} -> predict {predict_start:%Y-%m-%d}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Fit wm_4 on one calendar day, use historical warm-up data before the same weekday one week later, and predict that day.")
    parser.add_argument("--day", default="2022-08-15", help="Calendar day used for fitting; the same weekday one week later is predicted.")
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS, help="Matching weekdays used for trimmed O(t); default 5.")
    parser.add_argument("--warmup-hours", type=int, default=24)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--grid-points", type=int, default=4)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    frame, _ = aligned_data()
    fit_start = day_start(args.day)
    fit_end, _ = selected_day(frame, fit_start)
    O_fit, _ = fit_expected_output(frame, fit_start, args.baseline_weeks)
    fit_data, fit_index, _, model_T = model_input(frame, fit_start, args.steps_per_hour, args.warmup_hours)

    fit_model = wm_4(model_T, len(fit_index), O_fit)
    fit = brute_params(fit_model).fit(
        fit_data, INIT,
        grid_points=args.grid_points,
        refinements=args.refinements,
        fit_initial_state=False,
        fit_diffusion=True,
        verbose=True,
        progress_every=args.progress_every,
    )

    predict_start = fit_start + pd.Timedelta(weeks=1)
    predict_end, actual, U, Time = forecast_signals(frame, predict_start)
    O_predict, output_fit = fit_expected_output(frame, predict_start, args.baseline_weeks)
    init, warmup_initial, warmup_mse, warmup_start = warmup_state(
        frame, predict_start, args.warmup_hours, args.steps_per_hour, args.baseline_weeks, fit["q"], fit["kappa_R"],
    )
    n_steps = DAY_HOURS * args.steps_per_hour + 1
    predict_model = wm_4((0.0, 1.0), n_steps, O_predict)
    state = predict_model.deterministic_solver(init, {
        "U": U, "PE": U, "Time": Time,
        "q": fit["q"], "kappa_R": fit["kappa_R"],
    })
    fine_index = pd.date_range(predict_start, predict_end, periods=n_steps)

    hourly = np.arange(DAY_HOURS) * args.steps_per_hour
    observed = actual["I"].to_numpy(dtype=float)
    baseline = state["O"][hourly]
    predicted = state["I"][hourly]
    baseline_metrics = deviation_metrics(observed, baseline)
    prediction_metrics = deviation_metrics(observed, predicted)

    path = OUTPUT / f"daily_prediction_{fit_start:%Y-%m-%d}_to_{predict_start:%Y-%m-%d}.png"
    save_plot(fit_start, predict_start, predict_end, actual, fine_index, state, path)

    baseline_days = ", ".join(day.strftime("%Y-%m-%d") for day in output_fit["baseline_days"])
    print(f"fitting day={fit_start} -> {fit_end}")
    print(f"prediction day={predict_start} -> {predict_end}")
    print(f"prediction O(t): previous {output_fit['weeks']} matching {output_fit['weekday']}s: {baseline_days}")
    print(f"prediction O(t): remove hourly min/max, average remaining {output_fit['retained_weeks']}")
    print(f"q={np.array2string(fit['q'], precision=8)}")
    print(f"kappa_R={fit['kappa_R']:.8f}, branch={fit['brute_branch']}")
    print(f"sigma_0={fit['sigma_0']:.8f}, sigma_I={fit['sigma_I']:.8f}")
    print(f"prediction warm-up={warmup_start} -> {predict_start} ({args.warmup_hours} h historical I/U)")
    print(f"warm-up fitted initial [F,R]={np.array2string(warmup_initial, precision=8)}, MSE={warmup_mse:.10f}")
    print(f"forecast initial [F,R] at {predict_start}={np.array2string(init, precision=8)}")
    print(f"O(t) deviation from actual I: MSE={baseline_metrics['mse']:.10f}, RMSE={baseline_metrics['rmse']:.10f}, MAE={baseline_metrics['mae']:.10f}, bias={baseline_metrics['bias']:.10f}")
    print(f"parametrized I=O+R deviation from actual I: MSE={prediction_metrics['mse']:.10f}, RMSE={prediction_metrics['rmse']:.10f}, MAE={prediction_metrics['mae']:.10f}, bias={prediction_metrics['bias']:.10f}")
    print(f"plot={path}")


if __name__ == "__main__": main()
