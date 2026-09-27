from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_3.fit_weekly import aligned_data
from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, INIT, fit_expected_output
from src.model.wm_4 import wm_4
from src.params.wm_4 import brute_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction"
WEEK_HOURS = 168


def week_start(value):
    start = pd.Timestamp(value).normalize()
    return start - pd.Timedelta(days=start.weekday())


def expected_output(frame, origin, end, weeks):
    days = pd.date_range(origin.normalize(), end.normalize(), freq="D")
    profiles, metadata = {}, {}
    for day in days:
        profiles[day], metadata[day] = fit_expected_output(frame, day, weeks)

    def O(elapsed_hours):
        stamp = origin + pd.Timedelta(hours=float(elapsed_hours))
        day = stamp.normalize()
        hour = stamp.hour + stamp.minute / 60.0 + stamp.second / 3600.0
        return profiles[day](hour)

    return O, metadata


def fit_input(frame, start, steps_per_hour, warmup_hours):
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
    source = frame.reindex(source_index)[["I", "U"]].copy()
    if source.iloc[:-1].isna().any().any(): raise ValueError(f"Incomplete I/U data between {origin} and {end}.")
    source["I"] = source["I"].interpolate(method="time", limit_direction="both").ffill().bfill()
    source["U"] = source["U"].ffill().bfill()

    source_hours = np.arange(total_hours + 1, dtype=float)
    hour_bin = np.minimum(np.floor(elapsed + 1e-12).astype(int), total_hours)
    I = np.interp(elapsed, source_hours, source["I"].to_numpy(dtype=float))
    U = source["U"].to_numpy(dtype=float)[hour_bin]

    observation_mask = np.zeros(n_steps, dtype=bool)
    offset = warmup_hours * steps_per_hour
    observation_mask[offset + np.arange(WEEK_HOURS) * steps_per_hour] = True
    return {
        "I": I, "U": U, "PE": U.copy(), "Time": elapsed, "observation_mask": observation_mask,
    }, fine_index, origin, end


def forecast_signals(frame, start):
    end = start + pd.Timedelta(hours=WEEK_HOURS)
    index = pd.date_range(start, end, freq="h")
    source = frame.reindex(index)[["I", "U"]].copy()
    if source.iloc[:-1].isna().any().any(): raise ValueError(f"Incomplete prediction I/U data between {start} and {end}.")
    source["U"] = source["U"].ffill().bfill()
    prices = source["U"].to_numpy(dtype=float)

    def U(t):
        hour = min(int(np.floor(float(t) * 24.0 + 1e-12)), WEEK_HOURS)
        return float(prices[hour])

    def Time(t): return float(t) * 24.0
    return end, source.iloc[:WEEK_HOURS].copy(), U, Time


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
    axes[0].plot(fine_index, state["I"], label="WM4 next-week prediction I=O+R")
    axes[0].plot(fine_index, state["O"], label="Expected output O(t)", alpha=0.8)
    axes[0].plot(fine_index, state["U"], label="Normalized DK1 spot price U", alpha=0.8)
    axes[0].set_ylabel("Normalized input / price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(fine_index, state["U"], label="Known next-week normalized price U")
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

    fig.suptitle(f"wm_4 next-week prediction | fit {fit_start:%Y-%m-%d} week -> predict {predict_start:%Y-%m-%d} to {predict_end:%Y-%m-%d}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Fit wm_4 on one Monday-Sunday week, predict the following week, and compare the prediction with observed data.")
    parser.add_argument("--week", default="2022-08-08", help="Any date in the fitting week; the following week is predicted.")
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS, help="Matching weekdays used for trimmed O(t); default 5.")
    parser.add_argument("--warmup-hours", type=int, default=24)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--grid-points", type=int, default=4)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    frame, _ = aligned_data()
    fit_start = week_start(args.week)
    fit_end = fit_start + pd.Timedelta(hours=WEEK_HOURS)
    fit_data, fit_index, fit_origin, _ = fit_input(frame, fit_start, args.steps_per_hour, args.warmup_hours)
    O_fit, _ = expected_output(frame, fit_origin, fit_end, args.baseline_weeks)
    fit_model = wm_4((0.0, (args.warmup_hours + WEEK_HOURS) / 24.0), len(fit_index), O_fit)
    fit = brute_params(fit_model).fit(
        fit_data, INIT,
        grid_points=args.grid_points,
        refinements=args.refinements,
        fit_initial_state=False,
        fit_diffusion=True,
        verbose=True,
        progress_every=args.progress_every,
    )

    predict_start = fit_end
    predict_end, actual, U, Time = forecast_signals(frame, predict_start)
    O_predict, _ = expected_output(frame, predict_start, predict_end, args.baseline_weeks)
    n_steps = WEEK_HOURS * args.steps_per_hour + 1
    predict_model = wm_4((0.0, WEEK_HOURS / 24.0), n_steps, O_predict)
    init = np.array([fit["deterministic_state"]["F"][-1], fit["deterministic_state"]["R"][-1]], dtype=float)
    state = predict_model.deterministic_solver(init, {
        "U": U, "PE": U, "Time": Time,
        "q": fit["q"], "kappa_R": fit["kappa_R"],
    })
    fine_index = pd.date_range(predict_start, predict_end, periods=n_steps)

    hourly = np.arange(WEEK_HOURS) * args.steps_per_hour
    observed = actual["I"].to_numpy(dtype=float)
    baseline = state["O"][hourly]
    predicted = state["I"][hourly]
    baseline_metrics = deviation_metrics(observed, baseline)
    prediction_metrics = deviation_metrics(observed, predicted)

    path = OUTPUT / f"weekly_prediction_{fit_start:%Y-%m-%d}_to_{predict_start:%Y-%m-%d}.png"
    save_plot(fit_start, predict_start, predict_end, actual, fine_index, state, path)

    print(f"fitting week={fit_start} -> {fit_end}")
    print(f"prediction week={predict_start} -> {predict_end}")
    print(f"q={np.array2string(fit['q'], precision=8)}")
    print(f"kappa_R={fit['kappa_R']:.8f}, branch={fit['brute_branch']}")
    print(f"sigma_0={fit['sigma_0']:.8f}, sigma_I={fit['sigma_I']:.8f}")
    print(f"forecast initial [F,R]={np.array2string(init, precision=8)}")
    print(f"O(t) deviation from actual I: MSE={baseline_metrics['mse']:.10f}, RMSE={baseline_metrics['rmse']:.10f}, MAE={baseline_metrics['mae']:.10f}, bias={baseline_metrics['bias']:.10f}")
    print(f"parametrized I=O+R deviation from actual I: MSE={prediction_metrics['mse']:.10f}, RMSE={prediction_metrics['rmse']:.10f}, MAE={prediction_metrics['mae']:.10f}, bias={prediction_metrics['bias']:.10f}")
    print(f"plot={path}")


if __name__ == "__main__": main()
