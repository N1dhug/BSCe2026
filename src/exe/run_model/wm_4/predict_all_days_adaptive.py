from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY, fit_history_optimized
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_adaptive"
HORIZONS = (1, 2, 3, 4, 6, 8)


def _predict(model, init, U, Time, r):
    return model.deterministic_solver(
        np.asarray(init, dtype=float),
        {"U": U, "PE": U, "Time": Time, "r": r},
    )


def _simulate_segment(init, U, O, r, dt, start_step, end_step):
    start_step, end_step = int(start_step), int(end_step)
    if end_step < start_step: raise ValueError("end_step must be on or after start_step.")
    init, U, O, r = np.asarray(init, dtype=float), np.asarray(U, dtype=float), np.asarray(O, dtype=float), np.asarray(r, dtype=float)
    n = end_step - start_step + 1
    F, R, I = np.empty(n), np.empty(n), np.empty(n)
    F[0], R[0] = init

    for j in range(n):
        k = start_step + j
        I[j] = O[k] + R[j]
        if j < n - 1:
            F[j + 1] = F[j] + dt * R[j]
            R[j + 1] = R[j] + dt * (r[0] * F[j] + r[1] * U[k] + r[2] * R[j])

    return {"F": F, "R": R, "I": I}


def _propagate_one_hour(init, U, O, r, dt, hour, steps_per_hour):
    start = int(hour - 1) * int(steps_per_hour)
    end = int(hour) * int(steps_per_hour)
    state = _simulate_segment(init, U, O, r, dt, start, end)
    return np.array([state["F"][-1], state["R"][-1]], dtype=float)


def _fit_available_state(model, observed, U, O, r, current_hour, steps_per_hour, f_bounds=(-1.0, 1.0), r_bounds=(-1.0, 1.0)):
    current_hour = int(current_hour)
    obs_hours = np.arange(0, current_hour + 1, dtype=int)
    if len(obs_hours) < 3: raise ValueError("At least three observations are required for adaptive backfitting.")

    start_step = 0
    end_step = current_hour * int(steps_per_hour)
    sample_steps = obs_hours * int(steps_per_hour)

    def response(init):
        state = _simulate_segment(init, U, O, r, model.dt, start_step, end_step)
        return np.asarray(state["I"], dtype=float)[sample_steps], state

    base_I, _ = response(np.array([0.0, 0.0]))
    f_I, _ = response(np.array([1.0, 0.0]))
    r_I, _ = response(np.array([0.0, 1.0]))
    X = np.column_stack((f_I - base_I, r_I - base_I))
    y = np.asarray(observed, dtype=float)[obs_hours] - base_I
    init = model._least_squares_qp(
        X, y,
        lb=np.array([float(f_bounds[0]), float(r_bounds[0])], dtype=float),
        ub=np.array([float(f_bounds[1]), float(r_bounds[1])], dtype=float),
        ridge=model._FIT_RIDGE,
    )
    fitted_I, state = response(init)
    residual = fitted_I - np.asarray(observed, dtype=float)[obs_hours]
    current = np.array([state["F"][-1], state["R"][-1]], dtype=float)
    return {
        "first_hour": 0,
        "observations": int(len(obs_hours)),
        "window_init": np.asarray(init, dtype=float),
        "current_state": current,
        "fit_mse": float(np.mean(residual ** 2)),
    }


def _forecast_from_state(init, U, O, r, dt, origin_hour, max_horizon, steps_per_hour):
    origin_step = int(origin_hour) * int(steps_per_hour)
    end_hour = min(DAY_HOURS - 1, int(origin_hour) + int(max_horizon))
    end_step = end_hour * int(steps_per_hour)
    state = _simulate_segment(init, U, O, r, dt, origin_step, end_step)
    forecasts = {}
    for lead in range(1, int(max_horizon) + 1):
        target = int(origin_hour) + lead
        if target >= DAY_HOURS: break
        forecasts[lead] = float(state["I"][lead * int(steps_per_hour)])
    return forecasts


def day_result(frame, start, baseline_weeks, steps_per_hour, max_horizon=8, **fit_kwargs):
    start = day_start(start)
    output_model, _, fit, _ = fit_history_optimized(
        frame, start, baseline_weeks, steps_per_hour, **fit_kwargs,
    )
    _, actual, U_fn, Time_fn, model, _ = prediction_setup(
        frame, start, output_model, steps_per_hour,
    )

    historical = np.asarray(fit["x_hat"], dtype=float)
    seed_state = _predict(model, historical, U_fn, Time_fn, fit["r"])
    U = np.asarray(seed_state["U"], dtype=float)
    O = np.asarray(seed_state["O"], dtype=float)
    hourly_steps = np.arange(DAY_HOURS, dtype=int) * int(steps_per_hour)
    observed = actual["I"].to_numpy(dtype=float)
    O_hourly = O[hourly_steps]
    R_observed = observed - O_hourly

    initial = np.array([historical[0], R_observed[0]], dtype=float)
    static_state = _predict(model, initial, U_fn, Time_fn, fit["r"])
    static_I = np.asarray(static_state["I"], dtype=float)[hourly_steps]

    r_update_state = initial.copy()
    backfit_state = initial.copy()
    rows, state_rows = [], []

    for hour in range(DAY_HOURS):
        if hour > 0:
            r_update_state = _propagate_one_hour(
                r_update_state, U, O, fit["r"], model.dt, hour, steps_per_hour,
            )
            r_update_state[1] = R_observed[hour]

            if hour >= 2:
                backfit = _fit_available_state(
                    model, observed, U, O, fit["r"], hour, steps_per_hour,
                )
                backfit_state = np.array([
                    backfit["current_state"][0],
                    R_observed[hour],
                ], dtype=float)
                backfit_first_hour = backfit["first_hour"]
                backfit_used = backfit["observations"]
                backfit_mse = backfit["fit_mse"]
                backfit_R_before_observation = backfit["current_state"][1]
            else:
                backfit_state = _propagate_one_hour(
                    backfit_state, U, O, fit["r"], model.dt, hour, steps_per_hour,
                )
                backfit_state[1] = R_observed[hour]
                backfit_first_hour = 0
                backfit_used = hour + 1
                backfit_mse = np.nan
                backfit_R_before_observation = np.nan
        else:
            backfit_first_hour = 0
            backfit_used = 1
            backfit_mse = np.nan
            backfit_R_before_observation = np.nan

        r_forecast = _forecast_from_state(
            r_update_state, U, O, fit["r"], model.dt, hour, max_horizon, steps_per_hour,
        )
        backfit_forecast = _forecast_from_state(
            backfit_state, U, O, fit["r"], model.dt, hour, max_horizon, steps_per_hour,
        )

        state_rows.append({
            "day": start,
            "origin_hour": hour,
            "I_observed": observed[hour],
            "O": O_hourly[hour],
            "R_observed": R_observed[hour],
            "r_update_F": r_update_state[0],
            "r_update_R": r_update_state[1],
            "backfit_F": backfit_state[0],
            "backfit_R": backfit_state[1],
            "backfit_R_before_observation": backfit_R_before_observation,
            "backfit_first_hour": backfit_first_hour,
            "backfit_observations": backfit_used,
            "backfit_mse": backfit_mse,
        })

        for lead in sorted(r_forecast):
            target = hour + lead
            rows.append({
                "day": start,
                "origin_hour": hour,
                "lead_hours": lead,
                "target_hour": target,
                "I_observed": observed[target],
                "O": O_hourly[target],
                "I_static": static_I[target],
                "I_adaptive_r": r_forecast[lead],
                "I_adaptive_backfit": backfit_forecast[lead],
                "origin_R_observed": R_observed[hour],
                "origin_r_update_F": r_update_state[0],
                "origin_backfit_F": backfit_state[0],
            })

    return pd.DataFrame(rows), pd.DataFrame(state_rows)


def summary_by_lead(forecasts):
    rows = []
    for lead, part in forecasts.groupby("lead_hours", sort=True):
        observed = part["I_observed"].to_numpy(dtype=float)
        row = {
            "lead_hours": int(lead),
            "days": int(part["day"].nunique()),
            "forecasts": int(len(part)),
        }
        for name, column in (
            ("static", "I_static"),
            ("adaptive_r", "I_adaptive_r"),
            ("adaptive_backfit", "I_adaptive_backfit"),
            ("O", "O"),
        ):
            metrics = deviation_metrics(observed, part[column].to_numpy(dtype=float))
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
        row["adaptive_r_gain_vs_static_pct"] = 100.0 * (
            1.0 - row["adaptive_r_rmse"] / row["static_rmse"]
        ) if row["static_rmse"] > 0.0 else np.nan
        row["adaptive_backfit_gain_vs_static_pct"] = 100.0 * (
            1.0 - row["adaptive_backfit_rmse"] / row["static_rmse"]
        ) if row["static_rmse"] > 0.0 else np.nan
        row["backfit_gain_vs_r_update_pct"] = 100.0 * (
            1.0 - row["adaptive_backfit_rmse"] / row["adaptive_r_rmse"]
        ) if row["adaptive_r_rmse"] > 0.0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def cumulative_summary(forecasts, horizons=HORIZONS):
    rows = []
    for horizon in horizons:
        part = forecasts[forecasts["lead_hours"] <= int(horizon)]
        if part.empty: continue
        observed = part["I_observed"].to_numpy(dtype=float)
        row = {
            "horizon_hours": int(horizon),
            "days": int(part["day"].nunique()),
            "forecasts": int(len(part)),
        }
        for name, column in (
            ("static", "I_static"),
            ("adaptive_r", "I_adaptive_r"),
            ("adaptive_backfit", "I_adaptive_backfit"),
            ("O", "O"),
        ):
            metrics = deviation_metrics(observed, part[column].to_numpy(dtype=float))
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
        row["adaptive_r_gain_vs_static_pct"] = 100.0 * (
            1.0 - row["adaptive_r_rmse"] / row["static_rmse"]
        ) if row["static_rmse"] > 0.0 else np.nan
        row["adaptive_backfit_gain_vs_static_pct"] = 100.0 * (
            1.0 - row["adaptive_backfit_rmse"] / row["static_rmse"]
        ) if row["static_rmse"] > 0.0 else np.nan
        row["backfit_gain_vs_r_update_pct"] = 100.0 * (
            1.0 - row["adaptive_backfit_rmse"] / row["adaptive_r_rmse"]
        ) if row["adaptive_r_rmse"] > 0.0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def save_plot(summary, path):
    horizon = summary["horizon_hours"].to_numpy(dtype=int)
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.plot(horizon, summary["static_rmse"], marker="o", label="Static historical F0 + observed R0")
    ax.plot(horizon, summary["adaptive_r_rmse"], marker="o", label="Adaptive observed-R update")
    ax.plot(horizon, summary["adaptive_backfit_rmse"], marker="o", label="Adaptive all-history backfit F + observed R")
    ax.plot(horizon, summary["O_rmse"], marker="o", label="O(t)")
    ax.set_ylabel("Cumulative RMSE")
    ax.set_xlabel("Forecast horizon from each update (hours)")
    ax.set_xticks(horizon)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.suptitle("ARM (Affine Revised Model) rolling adaptive forecast performance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(cumulative, lead, skipped):
    print("rolling cumulative RMSE after each hourly observation:")
    print(f"{'hours':>7}{'static':>12}{'adaptive R':>12}{'backfit':>12}{'O(t)':>12}{'R gain':>12}{'fit gain':>12}")
    for _, row in cumulative.iterrows():
        print(
            f"{int(row['horizon_hours']):>7d}"
            f"{row['static_rmse']:>12.8f}"
            f"{row['adaptive_r_rmse']:>12.8f}"
            f"{row['adaptive_backfit_rmse']:>12.8f}"
            f"{row['O_rmse']:>12.8f}"
            f"{row['adaptive_r_gain_vs_static_pct']:>11.2f}%"
            f"{row['adaptive_backfit_gain_vs_static_pct']:>11.2f}%"
        )

    print()
    print("RMSE by exact lead time:")
    print(f"{'lead':>7}{'static':>12}{'adaptive R':>12}{'backfit':>12}{'O(t)':>12}")
    for _, row in lead.iterrows():
        print(
            f"{int(row['lead_hours']):>7d}"
            f"{row['static_rmse']:>12.8f}"
            f"{row['adaptive_r_rmse']:>12.8f}"
            f"{row['adaptive_backfit_rmse']:>12.8f}"
            f"{row['O_rmse']:>12.8f}"
        )

    if skipped:
        print()
        print("skipped days:")
        for day, reason in skipped: print(f"{day:%Y-%m-%d}: {reason}")


def run_all_days(frame, start=START_DAY, end=END_DAY, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, max_horizon=8, fail_fast=False, **fit_kwargs):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")
    forecasts, states, skipped = [], [], []

    for day in pd.date_range(start, end, freq="D"):
        try:
            day_forecasts, day_states = day_result(
                frame, day, baseline_weeks, steps_per_hour,
                max_horizon=max_horizon,
                **fit_kwargs,
            )
            forecasts.append(day_forecasts)
            states.append(day_states)
            print(
                f"{day:%Y-%m-%d} | forecasts={len(day_forecasts)} "
                f"| mean backfit MSE={day_states['backfit_mse'].dropna().mean():.8f}"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc)))
            print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not forecasts: raise RuntimeError("No days in the requested range could be fitted.")
    return pd.concat(forecasts, ignore_index=True), pd.concat(states, ignore_index=True), skipped


def main():
    parser = ArgumentParser(description="Evaluate causal WM4 rolling state updates after each new hourly observation. Compare the static historical-F0 + observed-R0 forecast with an observed-R adaptive update and an expanding-history F backfit + observed-R update.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--max-horizon", type=int, default=8)
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    if args.max_horizon < 1 or args.max_horizon > DAY_HOURS - 1: raise ValueError(f"max-horizon must lie in [1, {DAY_HOURS - 1}].")

    frame, _ = aligned_wm4_data("MED")
    forecasts, states, skipped = run_all_days(
        frame, args.start_day, args.end_day, args.baseline_weeks, args.steps_per_hour,
        max_horizon=args.max_horizon,
        fail_fast=args.fail_fast,
        max_state_iter=args.max_state_iter,
        state_tol=args.state_tol,
    )
    lead = summary_by_lead(forecasts)
    horizons = tuple(h for h in HORIZONS if h <= args.max_horizon)
    cumulative = cumulative_summary(forecasts, horizons)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output / "rolling_forecasts.csv", index=False)
    states.to_csv(output / "adaptive_states.csv", index=False)
    lead.to_csv(output / "lead_time_summary.csv", index=False)
    cumulative.to_csv(output / "cumulative_horizon_summary.csv", index=False)
    if skipped:
        pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)
    save_plot(cumulative, output / "adaptive_cumulative_rmse.png")

    print()
    print(f"days={forecasts['day'].nunique()}, forecasts={len(forecasts)}, backfit=all observations available since day start")
    print_summary(cumulative, lead, skipped)
    print(f"output={output}")


if __name__ == "__main__": main()
