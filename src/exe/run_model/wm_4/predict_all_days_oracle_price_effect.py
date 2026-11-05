from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import BASELINE_WEEKS, DAY_HOURS, aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY, fit_history_optimized
from src.exe.run_model.wm_4.predict_daily import deviation_metrics, prediction_setup
from src.params.wm_4 import optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_oracle_price_effect"
DEFAULT_EFFECT_SHARE = 0.05
COEFFICIENT_TOL = 1e-8


def fit_day(frame, day, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, effect_share=DEFAULT_EFFECT_SHARE, max_state_iter=50, state_tol=1e-8):
    day, baseline_weeks, steps_per_hour = day_start(day), int(baseline_weeks), int(steps_per_hour)
    output_model, output_fit, fit, _ = fit_history_optimized(
        frame, day, baseline_weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )
    _, actual, U, Time, model, _ = prediction_setup(frame, day, output_model, steps_per_hour)

    prediction_data, _, _, _ = model_input(frame, day, steps_per_hour, 0, output_model)
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(prediction_data, fit["r"])
    init = np.asarray(oracle["init_cond"], dtype=float)
    state = oracle["deterministic_state"]

    hourly_index = np.arange(DAY_HOURS, dtype=int) * steps_per_hour
    observed = actual["I"].to_numpy(dtype=float)
    predicted = np.asarray(state["I"], dtype=float)[hourly_index]
    metrics = deviation_metrics(observed, predicted)

    r = np.asarray(fit["r"], dtype=float)
    F = np.asarray(state["F"], dtype=float)[hourly_index]
    R = np.asarray(state["R"], dtype=float)[hourly_index]
    price = np.asarray(state["U"], dtype=float)[hourly_index]

    rF_F = r[0] * F
    rU_U = r[1] * price
    rR_R = r[2] * R
    R_dot = rF_F + rU_U + rR_R

    abs_F = np.abs(rF_F)
    abs_U = np.abs(rU_U)
    abs_R = np.abs(rR_R)
    total_abs = abs_F + abs_U + abs_R
    hourly_share = np.divide(abs_U, total_abs, out=np.zeros_like(abs_U), where=total_abs > 1e-12)

    mean_abs_price = float(np.mean(abs_U))
    mean_abs_state = float(np.mean(abs_F + abs_R))
    contribution_share = float(np.sum(abs_U) / np.sum(total_abs)) if float(np.sum(total_abs)) > 1e-12 else 0.0
    contribution_ratio = mean_abs_price / mean_abs_state if mean_abs_state > 1e-12 else np.inf if mean_abs_price > 0.0 else 0.0
    active_hours = hourly_share >= float(effect_share)
    coefficient_active = bool(abs(r[1]) > COEFFICIENT_TOL)
    observable = bool(coefficient_active and contribution_share >= float(effect_share))

    daily = {
        "day": day, "weekday": day.day_name(),
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "oracle_F0": init[0], "oracle_R0": init[1], "oracle_fit_mse": float(oracle["loss"]),
        "wm4_mse": metrics["mse"], "wm4_rmse": metrics["rmse"], "wm4_mae": metrics["mae"], "wm4_bias": metrics["bias"],
        "O_fit_mse": float(output_fit["mse"]), "state_iterations": int(fit["state_iterations"]),
        "mean_abs_rF_F": float(np.mean(abs_F)),
        "mean_abs_rU_U": mean_abs_price,
        "mean_abs_rR_R": float(np.mean(abs_R)),
        "mean_abs_R_dot": float(np.mean(np.abs(R_dot))),
        "max_abs_rU_U": float(np.max(abs_U)),
        "price_contribution_share": contribution_share,
        "price_to_state_contribution_ratio": contribution_ratio,
        "price_effect_hours": int(np.count_nonzero(active_hours)),
        "price_effect_hour_pct": 100.0 * float(np.mean(active_hours)),
        "price_effect_threshold": float(effect_share),
        "price_coefficient_active": coefficient_active,
        "observable_price_effect": observable,
    }

    hourly = pd.DataFrame({
        "day": day,
        "I_observed": observed,
        "I_oracle": predicted,
        "O": np.asarray(state["O"], dtype=float)[hourly_index],
        "U": price,
        "F": F,
        "R": R,
        "r_F_F": rF_F,
        "r_U_U": rU_U,
        "r_R_R": rR_R,
        "R_dot": R_dot,
        "price_contribution_share": hourly_share,
        "observable_price_effect_hour": active_hours & coefficient_active,
    }, index=actual.index)
    hourly.index.name = "time"
    return daily, hourly


def run_all_days(frame, start=START_DAY, end=END_DAY, baseline_weeks=BASELINE_WEEKS, steps_per_hour=60, effect_share=DEFAULT_EFFECT_SHARE, fail_fast=False, max_state_iter=50, state_tol=1e-8):
    start, end = day_start(start), day_start(end)
    if end < start: raise ValueError("end must be on or after start.")
    if not 0.0 <= float(effect_share) <= 1.0: raise ValueError("effect_share must lie in [0,1].")
    daily_rows, hourly_frames, skipped = [], [], []

    for day in pd.date_range(start, end, freq="D"):
        try:
            daily, hourly = fit_day(
                frame, day, baseline_weeks, steps_per_hour, effect_share,
                max_state_iter=max_state_iter, state_tol=state_tol,
            )
            daily_rows.append(daily); hourly_frames.append(hourly)
            marker = "PRICE" if daily["observable_price_effect"] else "-"
            print(
                f"{day:%Y-%m-%d} | {marker:>5} | r_U={daily['r_U']:.6f} | "
                f"price share={100.0 * daily['price_contribution_share']:.2f}% | "
                f"effect hours={daily['price_effect_hours']:2d}/{DAY_HOURS} | "
                f"oracle MSE={daily['wm4_mse']:.8f}"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc))); print(f"{day:%Y-%m-%d} | skipped | {exc}")

    if not daily_rows: raise RuntimeError("No days in the requested range could be fitted.")
    daily = pd.DataFrame(daily_rows).sort_values("day").reset_index(drop=True)
    hourly = pd.concat(hourly_frames).sort_index()
    return daily, hourly, skipped


def save_summary_plot(daily, path, effect_share):
    fig, axes = plt.subplots(4, 1, figsize=(18, 15), sharex=True)

    axes[0].plot(daily["day"], daily["r_U"], marker=".", label=r"Fitted $r_U$")
    axes[0].axhline(0.0, linewidth=0.8)
    axes[0].set_ylabel(r"$r_U$")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)

    axes[1].plot(daily["day"], 100.0 * daily["price_contribution_share"], marker=".", label=r"$|r_UU|$ share of total absolute drift contributions")
    axes[1].axhline(100.0 * effect_share, linestyle="--", label=f"Observable threshold = {100.0 * effect_share:.1f}%")
    axes[1].set_ylabel("Price share (%)")
    axes[1].legend(); axes[1].grid(True, alpha=0.25)

    axes[2].plot(daily["day"], daily["mean_abs_rF_F"], label=r"mean $|r_FF|$")
    axes[2].plot(daily["day"], daily["mean_abs_rU_U"], label=r"mean $|r_UU|$")
    axes[2].plot(daily["day"], daily["mean_abs_rR_R"], label=r"mean $|r_RR|$")
    axes[2].set_ylabel("Mean absolute contribution")
    axes[2].legend(ncol=3); axes[2].grid(True, alpha=0.25)

    active = daily["observable_price_effect"].to_numpy(dtype=bool)
    axes[3].scatter(daily.loc[~active, "day"], daily.loc[~active, "wm4_mse"], label="No observable price effect", alpha=0.65)
    axes[3].scatter(daily.loc[active, "day"], daily.loc[active, "wm4_mse"], label="Observable price effect", alpha=0.9)
    axes[3].set_ylabel("Oracle daily MSE")
    axes[3].set_xlabel("Prediction day")
    axes[3].legend(); axes[3].grid(True, alpha=0.25)

    fig.suptitle("ARM (Affine Revised Model) oracle-state predictions and fitted price-response contribution")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def save_hourly_plot(hourly, path):
    data = hourly.reset_index().sort_values("time")
    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

    axes[0].plot(data["time"], data["I_observed"], label="Observed I", linewidth=0.8)
    axes[0].plot(data["time"], data["I_oracle"], label="Oracle-state ARM", linewidth=0.8)
    axes[0].set_ylabel("Normalized input")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)

    axes[1].plot(data["time"], data["r_F_F"], label=r"$r_FF$", linewidth=0.8)
    axes[1].plot(data["time"], data["r_U_U"], label=r"$r_UU$ price contribution", linewidth=0.9)
    axes[1].plot(data["time"], data["r_R_R"], label=r"$r_RR$", linewidth=0.8)
    axes[1].plot(data["time"], data["R_dot"], label=r"$\dot R$", linewidth=0.8, alpha=0.8)
    axes[1].set_ylabel("Drift contribution")
    axes[1].legend(ncol=4); axes[1].grid(True, alpha=0.25)

    axes[2].plot(data["time"], data["U"], label="Normalized price U", linewidth=0.8)
    axes[2].plot(data["time"], 100.0 * data["price_contribution_share"], label="Price contribution share (%)", linewidth=0.8)
    active = data["observable_price_effect_hour"].to_numpy(dtype=bool)
    axes[2].scatter(data.loc[active, "time"], 100.0 * data.loc[active, "price_contribution_share"], s=10, label="Observable-price hours")
    axes[2].set_ylabel("Price / share (%)")
    axes[2].set_xlabel("Time")
    axes[2].legend(); axes[2].grid(True, alpha=0.25)

    fig.suptitle("ARM (Affine Revised Model) oracle-state hourly price-response contribution")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def weekday_summary(daily):
    rows = []
    for weekday, part in daily.groupby("weekday", sort=False):
        rows.append({
            "weekday": weekday,
            "days": int(len(part)),
            "observable_price_days": int(part["observable_price_effect"].sum()),
            "observable_price_day_pct": 100.0 * float(part["observable_price_effect"].mean()),
            "mean_r_U": float(part["r_U"].mean()),
            "mean_price_contribution_share": float(part["price_contribution_share"].mean()),
            "mean_price_effect_hours": float(part["price_effect_hours"].mean()),
            "mean_oracle_mse": float(part["wm4_mse"].mean()),
        })
    return pd.DataFrame(rows)


def print_summary(daily, hourly, skipped):
    active_days = daily["observable_price_effect"].to_numpy(dtype=bool)
    active_hours = hourly["observable_price_effect_hour"].to_numpy(dtype=bool)

    print()
    print(f"days fitted={len(daily)}, skipped={len(skipped)}, hourly observations={len(hourly)}")
    print(f"days with observable fitted price effect={np.count_nonzero(active_days)}/{len(active_days)} ({100.0 * np.mean(active_days):.2f}%)")
    print(f"hours with observable fitted price effect={np.count_nonzero(active_hours)}/{len(active_hours)} ({100.0 * np.mean(active_hours):.2f}%)")
    print(f"mean r_U={daily['r_U'].mean():.8f}")
    print(f"median r_U={daily['r_U'].median():.8f}")
    print(f"mean price contribution share={100.0 * daily['price_contribution_share'].mean():.2f}%")
    print(f"median price contribution share={100.0 * daily['price_contribution_share'].median():.2f}%")
    print(f"mean oracle MSE={daily['wm4_mse'].mean():.10f}")

    strongest = daily.sort_values("price_contribution_share", ascending=False).head(10)
    print()
    print("strongest fitted price-effect days")
    print(
        strongest[[
            "day", "weekday", "r_U", "price_contribution_share",
            "price_effect_hours", "mean_abs_rU_U", "wm4_mse",
        ]].to_string(index=False)
    )

    if skipped:
        print(); print("skipped days:")
        for day, reason in skipped: print(f"{day:%Y-%m-%d}: {reason}")


def main():
    parser = ArgumentParser(description="Run all WM4 days with oracle initial-state fits and quantify when the fitted r_U U price contribution is observable relative to total drift contributions.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--baseline-weeks", type=int, default=BASELINE_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--price-effect-share", type=float, default=DEFAULT_EFFECT_SHARE, help="Minimum |r_U U| share of |r_F F|+|r_U U|+|r_R R| used to flag an observable price effect.")
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    frame, _ = aligned_wm4_data("MED")
    daily, hourly, skipped = run_all_days(
        frame, args.start_day, args.end_day, args.baseline_weeks, args.steps_per_hour,
        effect_share=args.price_effect_share,
        fail_fast=args.fail_fast,
        max_state_iter=args.max_state_iter,
        state_tol=args.state_tol,
    )

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_oracle_price_effect.csv", index=False)
    hourly.to_csv(output / "hourly_oracle_price_effect.csv")
    weekday_summary(daily).to_csv(output / "weekday_price_effect.csv", index=False)
    if skipped:
        pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)

    save_summary_plot(daily, output / "oracle_price_effect_summary.png", args.price_effect_share)
    save_hourly_plot(hourly, output / "oracle_price_effect_hourly.png")
    print_summary(daily, hourly, skipped)
    print(); print(f"output={output}")


if __name__ == "__main__": main()
