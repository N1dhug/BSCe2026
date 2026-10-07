from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY, fit_history_optimized
from src.exe.run_model.wm_4.predict_all_days_shared_initial_state import fit_shared_initial_state
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup
from src.params.wm_4 import optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_recent_r0"
HORIZONS = (1, 2, 3, 4, 6, 8, 12, 24)


def _predict(model, init, U, Time, r):
    return model.deterministic_solver(
        np.asarray(init, dtype=float),
        {"U": U, "PE": U, "Time": Time, "r": r},
    )


def _output_at(output_model, frame, timestamp):
    timestamp = pd.Timestamp(timestamp)
    index = pd.DatetimeIndex([timestamp])
    source = frame.reindex(index)
    if source["Temp"].isna().any(): raise ValueError(f"Missing temperature at {timestamp}.")
    return float(output_model.predict(index, source["Temp"].to_numpy(dtype=float))[0])


def _previous_residual(frame, output_model, start):
    previous = day_start(start) - pd.Timedelta(hours=1)
    source = frame.reindex(pd.DatetimeIndex([previous]))
    if source["I"].isna().any() or source["Temp"].isna().any(): raise ValueError(f"Missing I/temperature at t=-1 ({previous}).")
    I_minus_1 = float(source["I"].iloc[0])
    O_minus_1 = _output_at(output_model, frame, previous)
    return previous, I_minus_1, O_minus_1, I_minus_1 - O_minus_1


def _previous_day_oracle_carryover(frame, start, output_model, r, steps_per_hour):
    previous_day = day_start(start) - pd.Timedelta(days=1)
    _, _, U, Time, model, _ = prediction_setup(
        frame, previous_day, output_model, steps_per_hour,
    )
    data, _, _, _ = model_input(
        frame, previous_day, steps_per_hour, 0, output_model,
    )
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(
        data, r,
    )
    state = oracle["deterministic_state"]
    init = np.array([
        float(np.asarray(state["F"], dtype=float)[-1]),
        float(np.asarray(state["R"], dtype=float)[-1]),
    ], dtype=float)
    return {
        "day": previous_day,
        "oracle_init": np.asarray(oracle["init_cond"], dtype=float),
        "oracle_fit_mse": float(oracle["loss"]),
        "carry_init": init,
    }


def _pseudo_oracle_fixed_r0(model, prediction_data, r, R0, f_bounds=(-1.0, 1.0)):
    helper = optimized_qp_params(model)
    data = helper._prepare_data(prediction_data)
    r = np.asarray(r, dtype=float)
    R0 = float(R0)
    base = helper._state_from_parameters(data, np.array([0.0, R0]), r)
    f_unit = helper._state_from_parameters(data, np.array([1.0, R0]), r)
    obs = np.asarray(data["observation_mask"], dtype=bool)
    X = ((np.asarray(f_unit["I"], dtype=float) - np.asarray(base["I"], dtype=float))[obs])[:, None]
    y = np.asarray(data["I"], dtype=float)[obs] - np.asarray(base["I"], dtype=float)[obs]
    F0 = float(model._least_squares_qp(
        X, y,
        lb=np.array([float(f_bounds[0])], dtype=float),
        ub=np.array([float(f_bounds[1])], dtype=float),
        ridge=model._FIT_RIDGE,
    )[0])
    init = np.array([F0, R0], dtype=float)
    state = helper._state_from_parameters(data, init, r, full=True)
    residual = np.asarray(state["I"], dtype=float)[obs] - np.asarray(data["I"], dtype=float)[obs]
    return {
        "init_cond": init,
        "deterministic_state": state,
        "loss": float(np.mean(residual ** 2)),
        "observation_mask": obs.copy(),
    }


def day_result(frame, start, baseline_weeks, steps_per_hour, **fit_kwargs):
    start = day_start(start)
    output_model, output_fit, fit, fit_model = fit_history_optimized(
        frame, start, baseline_weeks, steps_per_hour, **fit_kwargs,
    )
    shared_fit = fit_shared_initial_state(
        frame, output_model, fit_model, fit, steps_per_hour,
    )
    shared_init = np.asarray(shared_fit["init_cond"], dtype=float)
    historical_init = np.asarray(fit["x_hat"], dtype=float)

    _, actual, U, Time, model, _ = prediction_setup(
        frame, start, output_model, steps_per_hour,
    )
    prediction_data, _, _, _ = model_input(
        frame, start, steps_per_hour, 0, output_model,
    )
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(
        prediction_data, fit["r"],
    )
    oracle_init = np.asarray(oracle["init_cond"], dtype=float)

    shared_state = _predict(model, shared_init, U, Time, fit["r"])
    historical_state = _predict(model, historical_init, U, Time, fit["r"])
    oracle_state = oracle["deterministic_state"]

    previous, I_minus_1, O_minus_1, R_minus_1 = _previous_residual(
        frame, output_model, start,
    )
    hourly_index = np.arange(DAY_HOURS, dtype=int) * int(steps_per_hour)
    observed = actual["I"].to_numpy(dtype=float)
    O = np.asarray(shared_state["O"], dtype=float)[hourly_index]
    I0_observed = float(observed[0])
    O0 = float(O[0])
    R0_observed = I0_observed - O0

    shared_recent_init = np.array([shared_init[0], R_minus_1], dtype=float)
    historical_recent_init = np.array([historical_init[0], R_minus_1], dtype=float)
    shared_observed_r0_init = np.array([shared_init[0], R0_observed], dtype=float)
    historical_observed_r0_init = np.array([historical_init[0], R0_observed], dtype=float)
    previous_oracle = _previous_day_oracle_carryover(
        frame, start, output_model, fit["r"], steps_per_hour,
    )
    previous_oracle_init = np.asarray(previous_oracle["carry_init"], dtype=float)
    previous_oracle_f0_observed_r0_init = np.array([
        previous_oracle_init[0], R0_observed,
    ], dtype=float)
    pseudo_oracle = _pseudo_oracle_fixed_r0(
        model, prediction_data, fit["r"], R0_observed,
    )
    pseudo_oracle_init = np.asarray(pseudo_oracle["init_cond"], dtype=float)

    shared_recent_state = _predict(model, shared_recent_init, U, Time, fit["r"])
    historical_recent_state = _predict(model, historical_recent_init, U, Time, fit["r"])
    shared_observed_r0_state = _predict(model, shared_observed_r0_init, U, Time, fit["r"])
    historical_observed_r0_state = _predict(model, historical_observed_r0_init, U, Time, fit["r"])
    previous_oracle_state = _predict(model, previous_oracle_init, U, Time, fit["r"])
    previous_oracle_f0_observed_r0_state = _predict(
        model, previous_oracle_f0_observed_r0_init, U, Time, fit["r"],
    )
    pseudo_oracle_state = pseudo_oracle["deterministic_state"]

    states = {
        "shared": shared_state,
        "historical": historical_state,
        "shared_recent": shared_recent_state,
        "historical_recent": historical_recent_state,
        "shared_observed_r0": shared_observed_r0_state,
        "historical_observed_r0": historical_observed_r0_state,
        "previous_oracle": previous_oracle_state,
        "previous_oracle_f0_observed_r0": previous_oracle_f0_observed_r0_state,
        "pseudo_oracle": pseudo_oracle_state,
        "oracle": oracle_state,
    }
    predictions = {
        name: np.asarray(state["I"], dtype=float)[hourly_index]
        for name, state in states.items()
    }
    metrics = {
        name: deviation_metrics(observed, prediction)
        for name, prediction in predictions.items()
    }

    r = np.asarray(fit["r"], dtype=float)
    daily = {
        "day": start,
        "weekday": start.day_name(),
        "previous_time": previous,
        "r_F": r[0],
        "r_U": r[1],
        "r_R": r[2],
        "I_minus_1": I_minus_1,
        "O_minus_1": O_minus_1,
        "R_minus_1_observed": R_minus_1,
        "I0_observed": I0_observed,
        "O0": O0,
        "R0_observed": R0_observed,
        "shared_F0": shared_init[0],
        "shared_R0": shared_init[1],
        "historical_F0": historical_init[0],
        "historical_R0": historical_init[1],
        "oracle_F0": oracle_init[0],
        "oracle_R0": oracle_init[1],
        "oracle_fit_mse": float(oracle["loss"]),
        "pseudo_oracle_F0": pseudo_oracle_init[0],
        "pseudo_oracle_R0": pseudo_oracle_init[1],
        "pseudo_oracle_fit_mse": float(pseudo_oracle["loss"]),
        "shared_F0_gap_to_pseudo_oracle": shared_init[0] - pseudo_oracle_init[0],
        "pseudo_oracle_F0_gap_to_oracle": pseudo_oracle_init[0] - oracle_init[0],
        "shared_recent_F0": shared_recent_init[0],
        "shared_recent_R0": shared_recent_init[1],
        "historical_recent_F0": historical_recent_init[0],
        "historical_recent_R0": historical_recent_init[1],
        "shared_observed_r0_F0": shared_observed_r0_init[0],
        "shared_observed_r0_R0": shared_observed_r0_init[1],
        "historical_observed_r0_F0": historical_observed_r0_init[0],
        "historical_observed_r0_R0": historical_observed_r0_init[1],
        "previous_oracle_day": previous_oracle["day"],
        "previous_oracle_fit_F0": previous_oracle["oracle_init"][0],
        "previous_oracle_fit_R0": previous_oracle["oracle_init"][1],
        "previous_oracle_fit_mse": previous_oracle["oracle_fit_mse"],
        "previous_oracle_carry_F0": previous_oracle_init[0],
        "previous_oracle_carry_R0": previous_oracle_init[1],
        "previous_oracle_f0_observed_r0_F0": previous_oracle_f0_observed_r0_init[0],
        "previous_oracle_f0_observed_r0_R0": previous_oracle_f0_observed_r0_init[1],
        "previous_oracle_carry_F0_error_vs_oracle": previous_oracle_init[0] - oracle_init[0],
        "previous_oracle_carry_R0_error_vs_oracle": previous_oracle_init[1] - oracle_init[1],
        "shared_R0_error_vs_oracle": shared_init[1] - oracle_init[1],
        "historical_R0_error_vs_oracle": historical_init[1] - oracle_init[1],
        "recent_R0_error_vs_oracle": R_minus_1 - oracle_init[1],
        "observed_R0_error_vs_oracle": R0_observed - oracle_init[1],
        "shared_F0_error_vs_oracle": shared_init[0] - oracle_init[0],
        "historical_F0_error_vs_oracle": historical_init[0] - oracle_init[0],
        "O_fit_mse": float(output_fit["mse"]),
    }
    for name, values in metrics.items():
        daily.update({
            f"{name}_mse": values["mse"],
            f"{name}_rmse": values["rmse"],
            f"{name}_mae": values["mae"],
            f"{name}_bias": values["bias"],
        })

    hourly = pd.DataFrame({
        "day": start,
        "hour": np.arange(DAY_HOURS, dtype=int),
        "I_observed": observed,
        "O": O,
        "I_shared": predictions["shared"],
        "I_historical": predictions["historical"],
        "I_shared_recent": predictions["shared_recent"],
        "I_historical_recent": predictions["historical_recent"],
        "I_shared_observed_r0": predictions["shared_observed_r0"],
        "I_historical_observed_r0": predictions["historical_observed_r0"],
        "I_previous_oracle": predictions["previous_oracle"],
        "I_previous_oracle_f0_observed_r0": predictions["previous_oracle_f0_observed_r0"],
        "I_pseudo_oracle": predictions["pseudo_oracle"],
        "I_oracle": predictions["oracle"],
        "F_shared_recent": np.asarray(shared_recent_state["F"], dtype=float)[hourly_index],
        "R_shared_recent": np.asarray(shared_recent_state["R"], dtype=float)[hourly_index],
        "F_historical_recent": np.asarray(historical_recent_state["F"], dtype=float)[hourly_index],
        "R_historical_recent": np.asarray(historical_recent_state["R"], dtype=float)[hourly_index],
        "F_historical_observed_r0": np.asarray(historical_observed_r0_state["F"], dtype=float)[hourly_index],
        "R_historical_observed_r0": np.asarray(historical_observed_r0_state["R"], dtype=float)[hourly_index],
        "F_previous_oracle": np.asarray(previous_oracle_state["F"], dtype=float)[hourly_index],
        "R_previous_oracle": np.asarray(previous_oracle_state["R"], dtype=float)[hourly_index],
        "F_previous_oracle_f0_observed_r0": np.asarray(previous_oracle_f0_observed_r0_state["F"], dtype=float)[hourly_index],
        "R_previous_oracle_f0_observed_r0": np.asarray(previous_oracle_f0_observed_r0_state["R"], dtype=float)[hourly_index],
        "F_pseudo_oracle": np.asarray(pseudo_oracle_state["F"], dtype=float)[hourly_index],
        "R_pseudo_oracle": np.asarray(pseudo_oracle_state["R"], dtype=float)[hourly_index],
        "F_oracle": np.asarray(oracle_state["F"], dtype=float)[hourly_index],
        "R_oracle": np.asarray(oracle_state["R"], dtype=float)[hourly_index],
    }, index=actual.index)
    hourly.index.name = "time"
    return daily, hourly


def _series_metrics(target, estimate):
    target = np.asarray(target, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    residual = estimate - target
    if target.size < 2 or np.std(target) <= 0.0 or np.std(estimate) <= 0.0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(target, estimate)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "correlation": correlation,
    }


def r0_summary(daily):
    target = daily["oracle_R0"].to_numpy(dtype=float)
    rows = []
    for name, column in (
        ("observed R0 = I0-O0", "R0_observed"),
        ("recent residual R-1", "R_minus_1_observed"),
        ("shared historical R0", "shared_R0"),
        ("historical direct R0", "historical_R0"),
    ):
        metrics = _series_metrics(target, daily[column].to_numpy(dtype=float))
        rows.append({"estimator": name, "days": int(len(daily)), **metrics})
    return pd.DataFrame(rows)


def cumulative_horizon_summary(hourly, horizons=HORIZONS):
    rows = []
    for horizon in horizons:
        part = hourly[hourly["hour"] < int(horizon)]
        observed = part["I_observed"].to_numpy(dtype=float)
        row = {
            "horizon_hours": int(horizon),
            "days": int(part["day"].nunique()),
            "observations": int(len(part)),
        }
        for name, column in (
            ("shared", "I_shared"),
            ("historical", "I_historical"),
            ("shared_recent", "I_shared_recent"),
            ("historical_recent", "I_historical_recent"),
            ("shared_observed_r0", "I_shared_observed_r0"),
            ("historical_observed_r0", "I_historical_observed_r0"),
            ("previous_oracle", "I_previous_oracle"),
            ("previous_oracle_f0_observed_r0", "I_previous_oracle_f0_observed_r0"),
            ("pseudo_oracle", "I_pseudo_oracle"),
            ("oracle", "I_oracle"),
        ):
            metrics = deviation_metrics(observed, part[column].to_numpy(dtype=float))
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
        row["shared_recent_gain_vs_shared_pct"] = 100.0 * (
            1.0 - row["shared_recent_rmse"] / row["shared_rmse"]
        ) if row["shared_rmse"] > 0.0 else np.nan
        row["historical_recent_gain_vs_historical_pct"] = 100.0 * (
            1.0 - row["historical_recent_rmse"] / row["historical_rmse"]
        ) if row["historical_rmse"] > 0.0 else np.nan
        row["shared_observed_r0_gap_to_pseudo_oracle_pct"] = 100.0 * (
            row["shared_observed_r0_rmse"] / row["pseudo_oracle_rmse"] - 1.0
        ) if row["pseudo_oracle_rmse"] > 0.0 else np.nan
        row["historical_observed_r0_gap_to_pseudo_oracle_pct"] = 100.0 * (
            row["historical_observed_r0_rmse"] / row["pseudo_oracle_rmse"] - 1.0
        ) if row["pseudo_oracle_rmse"] > 0.0 else np.nan
        row["previous_oracle_gain_vs_shared_pct"] = 100.0 * (
            1.0 - row["previous_oracle_rmse"] / row["shared_rmse"]
        ) if row["shared_rmse"] > 0.0 else np.nan
        row["pseudo_oracle_gap_to_oracle_pct"] = 100.0 * (
            row["pseudo_oracle_rmse"] / row["oracle_rmse"] - 1.0
        ) if row["oracle_rmse"] > 0.0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def save_r0_plot(daily, path):
    day = pd.to_datetime(daily["day"])
    fig, axes = plt.subplots(3, 1, figsize=(17, 13), sharex=False)

    axes[0].plot(day, daily["oracle_R0"], label="Oracle R0")
    axes[0].plot(day, daily["R0_observed"], label="Observed residual I0-O0")
    axes[0].plot(day, daily["R_minus_1_observed"], label="Previous residual I-1-O-1")
    axes[0].plot(day, daily["shared_R0"], label="Shared historical R0")
    axes[0].plot(day, daily["historical_R0"], label="Historical-direct R0")
    axes[0].set_ylabel("R")
    axes[0].legend(ncol=2)
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(daily["R0_observed"], daily["oracle_R0"], label="Oracle vs observed R0")
    axes[1].scatter(daily["R_minus_1_observed"], daily["oracle_R0"], label="Oracle vs previous residual")
    limits = [
        min(float(daily["oracle_R0"].min()), float(daily["R0_observed"].min()), float(daily["R_minus_1_observed"].min())),
        max(float(daily["oracle_R0"].max()), float(daily["R0_observed"].max()), float(daily["R_minus_1_observed"].max())),
    ]
    axes[1].plot(limits, limits, linestyle="--", linewidth=0.8, label="Identity")
    axes[1].set_xlabel("Residual-based R estimate")
    axes[1].set_ylabel("Oracle R0")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(day, daily["shared_recent_mse"], label="Shared F0 + previous residual R0")
    axes[2].plot(day, daily["historical_recent_mse"], label="Historical F0 + previous residual R0")
    axes[2].plot(day, daily["shared_mse"], label="Shared state")
    axes[2].plot(day, daily["historical_mse"], label="Historical-direct state")
    axes[2].plot(day, daily["shared_observed_r0_mse"], label="Shared F0 + observed R0")
    axes[2].plot(day, daily["historical_observed_r0_mse"], label="Historical F0 + observed R0")
    axes[2].plot(day, daily["previous_oracle_mse"], label="Previous-day oracle carryover")
    axes[2].plot(day, daily["previous_oracle_f0_observed_r0_mse"], label="Previous-oracle F0 + observed R0")
    axes[2].plot(day, daily["pseudo_oracle_mse"], label="Pseudo-oracle: optimized F0 + observed R0")
    axes[2].plot(day, daily["oracle_mse"], label="Oracle")
    axes[2].set_ylabel("Daily MSE")
    axes[2].set_xlabel("Prediction day")
    axes[2].legend(ncol=2)
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("WM4 recent-residual R0 diagnostic and hybrid initialization")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_horizon_plot(summary, path):
    horizon = summary["horizon_hours"].to_numpy(dtype=int)
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.plot(horizon, summary["shared_rmse"], marker="o", label="Shared state")
    ax.plot(horizon, summary["shared_recent_rmse"], marker="o", label="Shared F0 + recent R")
    ax.plot(horizon, summary["historical_rmse"], marker="o", label="Historical direct")
    ax.plot(horizon, summary["historical_recent_rmse"], marker="o", label="Historical F0 + recent R")
    ax.plot(horizon, summary["shared_observed_r0_rmse"], marker="o", label="Shared F0 + observed R0")
    ax.plot(horizon, summary["historical_observed_r0_rmse"], marker="o", label="Historical F0 + observed R0")
    ax.plot(horizon, summary["previous_oracle_rmse"], marker="o", label="Previous-day oracle carryover")
    ax.plot(horizon, summary["previous_oracle_f0_observed_r0_rmse"], marker="o", label="Previous-oracle F0 + observed R0")
    ax.plot(horizon, summary["pseudo_oracle_rmse"], marker="o", label="Pseudo-oracle: optimized F0 + observed R0")
    ax.plot(horizon, summary["oracle_rmse"], marker="o", label="Oracle")
    ax.set_ylabel("Cumulative RMSE")
    ax.set_xlabel("Open-loop horizon (hours)")
    ax.set_xticks(horizon)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.suptitle("WM4 cumulative RMSE with recent-residual R0 initialization")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(daily, hourly, r0, cumulative, skipped):
    print(f"days fitted={len(daily)}, skipped={len(skipped)}, hourly observations={len(hourly)}")
    print()
    print("R0 estimators relative to oracle R0:")
    print(f"{'estimator':<28}{'RMSE':>12}{'MAE':>12}{'bias':>12}{'corr':>12}")
    for _, row in r0.iterrows():
        print(
            f"{row['estimator']:<28}"
            f"{row['rmse']:>12.8f}"
            f"{row['mae']:>12.8f}"
            f"{row['bias']:>12.8f}"
            f"{row['correlation']:>12.6f}"
        )

    print()
    observed = hourly["I_observed"].to_numpy(dtype=float)
    print(f"{'method':<34}{'MSE':>14}{'RMSE':>14}{'MAE':>14}{'bias':>14}")
    for name, column in (
        ("shared state", "I_shared"),
        ("historical direct", "I_historical"),
        ("shared F0 + recent R", "I_shared_recent"),
        ("historical F0 + recent R", "I_historical_recent"),
        ("shared F0 + observed R0", "I_shared_observed_r0"),
        ("historical F0 + observed R0", "I_historical_observed_r0"),
        ("previous-day oracle carryover", "I_previous_oracle"),
        ("previous-oracle F0 + observed R0", "I_previous_oracle_f0_observed_r0"),
        ("pseudo-oracle F0 + observed R0", "I_pseudo_oracle"),
        ("oracle", "I_oracle"),
    ):
        metrics = deviation_metrics(observed, hourly[column].to_numpy(dtype=float))
        print(f"{name:<34}{metrics['mse']:>14.10f}{metrics['rmse']:>14.10f}{metrics['mae']:>14.10f}{metrics['bias']:>14.10f}")

    print()
    print("cumulative RMSE from forecast start:")
    print(
        f"{'hours':>7}{'shared':>11}{'hist':>11}"
        f"{'sh+R0':>11}{'hist+R0':>11}{'prev-or':>11}{'prevF+R0':>11}"
        f"{'pseudo':>11}{'oracle':>11}"
    )
    for _, row in cumulative.iterrows():
        print(
            f"{int(row['horizon_hours']):>7d}"
            f"{row['shared_rmse']:>11.8f}"
            f"{row['historical_rmse']:>11.8f}"
            f"{row['shared_observed_r0_rmse']:>11.8f}"
            f"{row['historical_observed_r0_rmse']:>11.8f}"
            f"{row['previous_oracle_rmse']:>11.8f}"
            f"{row['previous_oracle_f0_observed_r0_rmse']:>11.8f}"
            f"{row['pseudo_oracle_rmse']:>11.8f}"
            f"{row['oracle_rmse']:>11.8f}"
        )

    if skipped:
        print()
        print("skipped days:")
        for day, reason in skipped: print(f"{day:%Y-%m-%d}: {reason}")


def run_all_days(frame, start=START_DAY, end=END_DAY, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, fail_fast=False, **fit_kwargs):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")
    daily_rows, hourly_frames, skipped = [], [], []

    for day in pd.date_range(start, end, freq="D"):
        try:
            daily, hourly = day_result(
                frame, day, baseline_weeks, steps_per_hour, **fit_kwargs,
            )
            daily_rows.append(daily)
            hourly_frames.append(hourly)
            print(
                f"{day:%Y-%m-%d} | R-1={daily['R_minus_1_observed']:.6f} "
                f"| R0 obs={daily['R0_observed']:.6f} shared={daily['shared_R0']:.6f} "
                f"hist={daily['historical_R0']:.6f} oracle={daily['oracle_R0']:.6f} "
                f"| F0 hist={daily['historical_F0']:.6f} shared={daily['shared_F0']:.6f} "
                f"prev={daily['previous_oracle_carry_F0']:.6f} pseudo={daily['pseudo_oracle_F0']:.6f} oracle={daily['oracle_F0']:.6f} "
                f"| MSE sh+R0={daily['shared_observed_r0_mse']:.8f} "
                f"hist+R0={daily['historical_observed_r0_mse']:.8f} prev={daily['previous_oracle_mse']:.8f} "
                f"pseudo={daily['pseudo_oracle_mse']:.8f} oracle={daily['oracle_mse']:.8f}"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc)))
            print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not daily_rows: raise RuntimeError("No days in the requested range could be fitted.")
    return (
        pd.DataFrame(daily_rows).sort_values("day").reset_index(drop=True),
        pd.concat(hourly_frames).sort_index(),
        skipped,
    )


def main():
    parser = ArgumentParser(description="Diagnose WM4 target-day R0 using the observed residual I-O at hour 0 and the causal previous-hour residual, then test hybrid initial states with historical/shared F0 and recent R.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    daily, hourly, skipped = run_all_days(
        frame, args.start_day, args.end_day, args.baseline_weeks, args.steps_per_hour,
        fail_fast=args.fail_fast,
        max_state_iter=args.max_state_iter,
        state_tol=args.state_tol,
    )

    r0 = r0_summary(daily)
    cumulative = cumulative_horizon_summary(hourly)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_r0_diagnostic.csv", index=False)
    hourly.to_csv(output / "hourly_predictions.csv")
    r0.to_csv(output / "r0_estimator_summary.csv", index=False)
    cumulative.to_csv(output / "cumulative_horizon_summary.csv", index=False)
    daily[[
        "day",
        "shared_F0", "historical_F0", "previous_oracle_carry_F0",
        "pseudo_oracle_F0", "oracle_F0",
        "shared_R0", "historical_R0", "previous_oracle_carry_R0",
        "R0_observed", "oracle_R0",
        "shared_observed_r0_mse", "historical_observed_r0_mse",
        "previous_oracle_mse", "previous_oracle_f0_observed_r0_mse",
        "pseudo_oracle_mse", "oracle_mse",
    ]].to_csv(output / "state_carryover_comparison.csv", index=False)
    if skipped:
        pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)

    save_r0_plot(daily, output / "r0_diagnostic.png")
    save_horizon_plot(cumulative, output / "recent_r0_cumulative_rmse.png")
    print()
    print_summary(daily, hourly, r0, cumulative, skipped)
    print(f"output={output}")


if __name__ == "__main__": main()
