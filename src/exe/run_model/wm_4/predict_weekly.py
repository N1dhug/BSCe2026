from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import INIT_F, aligned_wm4_data
from src.exe.run_model.wm_4.output_model import output_callable, statistical_output
from src.model.wm_4 import wm_4
from src.params.wm_4 import qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction"
WEEK_HOURS = 168


def week_start(value):
    start = pd.Timestamp(value).normalize()
    return start - pd.Timedelta(days=start.weekday())


def fit_input(frame, start, steps_per_hour, warmup_hours, output_model):
    steps_per_hour, warmup_hours = int(steps_per_hour), int(warmup_hours)
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be at least 2.")
    if warmup_hours < 0: raise ValueError("warmup_hours must be nonnegative.")

    origin = start - pd.Timedelta(hours=warmup_hours)
    end = start + pd.Timedelta(hours=WEEK_HOURS)
    total_hours = warmup_hours + WEEK_HOURS
    n_steps = total_hours * steps_per_hour + 1
    elapsed = np.linspace(0.0, float(total_hours), n_steps)
    fine_index = origin + pd.to_timedelta(elapsed, unit="h")

    source_index = pd.date_range(origin, end, freq="h")
    source = frame.reindex(source_index)[["I", "U", "Temp"]].copy()
    if source.iloc[:-1].isna().any().any(): raise ValueError(f"Incomplete I/U/temperature data between {origin} and {end}.")
    source["I"] = source["I"].interpolate(method="time", limit_direction="both").ffill().bfill()
    source["U"] = source["U"].ffill().bfill()
    source["Temp"] = source["Temp"].interpolate(method="time", limit_direction="both").ffill().bfill()

    source_hours = np.arange(total_hours + 1, dtype=float)
    hour_bin = np.minimum(np.floor(elapsed + 1e-12).astype(int), total_hours)
    I = np.interp(elapsed, source_hours, source["I"].to_numpy(dtype=float))
    U = source["U"].to_numpy(dtype=float)[hour_bin]
    Temp = np.interp(elapsed, source_hours, source["Temp"].to_numpy(dtype=float))
    O = output_model.predict(fine_index, Temp)

    observation_mask = np.zeros(n_steps, dtype=bool)
    transition_mask = np.zeros(n_steps - 1, dtype=bool)
    offset = warmup_hours * steps_per_hour
    observation_mask[offset + np.arange(WEEK_HOURS) * steps_per_hour] = True
    transition_mask[offset:offset + WEEK_HOURS * steps_per_hour] = True
    return {
        "I": I, "U": U, "PE": U.copy(), "Temp": Temp, "O": O,
        "Time": elapsed, "observation_mask": observation_mask, "transition_mask": transition_mask,
    }, fine_index, origin, end


def prediction_setup(frame, start, steps_per_hour, output_model):
    end = start + pd.Timedelta(hours=WEEK_HOURS)
    n_steps = WEEK_HOURS * int(steps_per_hour) + 1
    elapsed = np.linspace(0.0, float(WEEK_HOURS), n_steps)
    fine_index = start + pd.to_timedelta(elapsed, unit="h")
    source_index = pd.date_range(start, end, freq="h")
    source = frame.reindex(source_index)[["I", "U", "Temp"]].copy()
    if source.iloc[:-1].isna().any().any(): raise ValueError(f"Incomplete prediction I/U/temperature data between {start} and {end}.")
    source["U"] = source["U"].ffill().bfill()
    source["Temp"] = source["Temp"].interpolate(method="time", limit_direction="both").ffill().bfill()

    hour_bin = np.minimum(np.floor(elapsed + 1e-12).astype(int), WEEK_HOURS)
    prices = source["U"].to_numpy(dtype=float)
    Temp = np.interp(elapsed, np.arange(WEEK_HOURS + 1, dtype=float), source["Temp"].to_numpy(dtype=float))
    O = output_model.predict(fine_index, Temp)
    model = wm_4((0.0, WEEK_HOURS / 24.0), n_steps, output_callable(fine_index, O))

    def U(t):
        hour = min(int(np.floor(float(t) * 24.0 + 1e-12)), WEEK_HOURS)
        return float(prices[hour])

    def Time(t): return float(t) * 24.0

    return end, source.iloc[:WEEK_HOURS].copy(), U, Time, model, fine_index


def deviation_metrics(observed, predicted):
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    return {
        "mse": float(np.mean(residual ** 2)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
    }


def save_plot(fit_start, predict_start, predict_end, actual, fine_index, state, path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    axes[0].plot(actual.index, actual["I"], label="Observed next-week input I")
    axes[0].scatter(actual.index, actual["I"], s=12)
    axes[0].plot(fine_index, state["I"], label="ARM next-week prediction I=O+R")
    axes[0].plot(fine_index, state["O"], label="Weather-aware expected output O", alpha=0.8)
    axes[0].plot(fine_index, state["U"], label="Normalized DK1 spot price U", alpha=0.8)
    axes[0].set_ylabel("Normalized input / price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(fine_index, state["U"], label="Known next-week normalized price U")
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

    fig.suptitle(f"wm_4 next-week prediction | fit {fit_start:%Y-%m-%d} week -> predict {predict_start:%Y-%m-%d} to {predict_end:%Y-%m-%d}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Fit wm_4 on one week and predict the following week with weather-aware exogenous O.")
    parser.add_argument("--week", default="2022-08-08", help="Any date in the fitting week; the following week is predicted.")
    parser.add_argument("--warmup-hours", type=int, default=24)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    fit_start = week_start(args.week)
    fit_end = fit_start + pd.Timedelta(hours=WEEK_HOURS)
    output_model = statistical_output().fit(frame, fit_end)

    fit_data, fit_index, fit_origin, _ = fit_input(frame, fit_start, args.steps_per_hour, args.warmup_hours, output_model)
    fit_model = wm_4((0.0, (args.warmup_hours + WEEK_HOURS) / 24.0), len(fit_index), output_callable(fit_index, fit_data["O"]))
    fit_init = np.array([INIT_F, fit_data["I"][0] - fit_data["O"][0]], dtype=float)
    fit = qp_params(fit_model).fit(fit_data, fit_init, verbose=False)

    predict_start = fit_end
    predict_end, actual, U, Time, predict_model, fine_index = prediction_setup(frame, predict_start, args.steps_per_hour, output_model)
    init = np.array([fit["deterministic_state"]["F"][-1], fit["deterministic_state"]["R"][-1]], dtype=float)
    state = predict_model.deterministic_solver(init, {"U": U, "PE": U, "Time": Time, "r": fit["r"]})

    hourly = np.arange(WEEK_HOURS) * args.steps_per_hour
    observed = actual["I"].to_numpy(dtype=float)
    baseline = state["O"][hourly]
    predicted = state["I"][hourly]
    baseline_metrics, prediction_metrics = deviation_metrics(observed, baseline), deviation_metrics(observed, predicted)

    path = OUTPUT / f"weekly_prediction_{fit_start:%Y-%m-%d}_to_{predict_start:%Y-%m-%d}.png"
    save_plot(fit_start, predict_start, predict_end, actual, fine_index, state, path)

    print(f"fitting week={fit_start} -> {fit_end}")
    print(f"prediction week={predict_start} -> {predict_end}")
    print(f"O training={output_model.metadata['fit_start']} -> {output_model.metadata['fit_end']}, samples={output_model.metadata['samples']}, MSE={output_model.metadata['mse']:.8f}")
    r = fit["r"]
    print(f"coefficients: r_F={r[0]:.8f}, r_U={r[1]:.8f}, r_R={r[2]:.8f}")
    print(f"damping: branch={fit['damping_branch']}, zeta={fit['damping_ratio']:.8f}")
    print(f"forecast initial [F,R]={np.array2string(init, precision=8)}")
    print(f"O deviation from actual I: MSE={baseline_metrics['mse']:.10f}, RMSE={baseline_metrics['rmse']:.10f}, MAE={baseline_metrics['mae']:.10f}, bias={baseline_metrics['bias']:.10f}")
    print(f"parametrized I=O+R deviation from actual I: MSE={prediction_metrics['mse']:.10f}, RMSE={prediction_metrics['rmse']:.10f}, MAE={prediction_metrics['mae']:.10f}, bias={prediction_metrics['bias']:.10f}")
    print(f"plot={path}")


if __name__ == "__main__": main()
