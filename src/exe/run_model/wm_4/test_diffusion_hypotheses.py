from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm

from src.exe.run_model.wm_4.analyze_observation_uncertainty import DEFAULT_WEEKS, _stack_days, corresponding_days, uncertainty_frame
from src.exe.run_model.wm_4.fit_daily import DAY_HOURS, aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY, fit_history_optimized
from src.exe.run_model.wm_4.predict_daily import prediction_setup
from src.params.wm_4 import optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "uncertainty" / "diffusion_hypotheses"
HYPOTHESES = ("H0", "H1", "H2")
ALPHA = 0.05
PARAMETER_NAMES = {
    "H0": ("sigma_0", "sigma_t", "phase_h"),
    "H1": ("sigma_0", "sigma_t", "phase_h", "sigma_I"),
    "H2": ("sigma_0", "sigma_t", "phase_h", "sigma_I", "sigma_U"),
}


def build_record(frame, prediction_day, weeks, steps_per_hour, max_state_iter, state_tol):
    prediction_day = day_start(prediction_day)
    empirical = uncertainty_frame(_stack_days(frame, corresponding_days(prediction_day, weeks)))
    observed_sigma = empirical["s_I"].to_numpy(dtype=float)

    output_model, _, fit, _ = fit_history_optimized(
        frame, prediction_day, weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )
    _, actual, U, Time, model, _ = prediction_setup(frame, prediction_day, output_model, steps_per_hour)
    prediction_data, _, _, _ = model_input(frame, prediction_day, steps_per_hour, 0, output_model)
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(prediction_data, fit["r"])
    oracle_state = oracle["deterministic_state"]

    hourly_index = np.arange(DAY_HOURS, dtype=int) * int(steps_per_hour)
    return {
        "day": prediction_day, "model": model, "state": oracle_state,
        "hourly_index": hourly_index, "observed_sigma": observed_sigma,
        "actual_I": actual["I"].to_numpy(dtype=float),
        "oracle_I": oracle_state["I"][hourly_index],
        "oracle_x": np.asarray(oracle["init_cond"], dtype=float),
        "oracle_mse": float(oracle["loss"]),
        "r": np.asarray(fit["r"], dtype=float),
    }


def build_records(frame, start, end, weeks, steps_per_hour, max_state_iter, state_tol, fail_fast=False):
    records, skipped = [], []
    for day in pd.date_range(day_start(start), day_start(end), freq="D"):
        try:
            record = build_record(frame, day, weeks, steps_per_hour, max_state_iter, state_tol)
            records.append(record)
            print(
                f"{day:%Y-%m-%d} | observed mean s_I={record['observed_sigma'].mean():.6f} "
                f"| oracle x*=[{record['oracle_x'][0]:.5f}, {record['oracle_x'][1]:.5f}]"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc))); print(f"{day:%Y-%m-%d} | skipped | {exc}")
    if len(records) < 4: raise RuntimeError("At least four usable prediction days are required for diffusion hypothesis testing.")
    return records, skipped


def _initial_starts(hypothesis, records):
    mean_sigma = float(np.mean([np.mean(record["observed_sigma"][1:]) for record in records]))
    peak_hours = [int(np.argmax(record["observed_sigma"])) for record in records]
    phase = float(np.median(peak_hours))
    base = max(1e-4, 0.25 * mean_sigma)
    amplitude = max(1e-4, mean_sigma)

    phases = (phase, 0.0, 6.0, 12.0, 18.0)
    starts = []
    for candidate_phase in phases:
        theta = [base, amplitude, candidate_phase]
        if hypothesis in ("H1", "H2"): theta.append(max(1e-4, 0.5 * mean_sigma))
        if hypothesis == "H2": theta.append(max(1e-4, 0.5 * mean_sigma))
        starts.append(np.asarray(theta, dtype=float))
    return starts


def _bounds(hypothesis):
    lower = [1e-8, 0.0, 0.0]
    upper = [np.inf, np.inf, 24.0]
    if hypothesis in ("H1", "H2"): lower.append(0.0); upper.append(np.inf)
    if hypothesis == "H2": lower.append(0.0); upper.append(np.inf)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def predicted_sigma(record, hypothesis, theta):
    result = record["model"].process_uncertainty(record["state"], hypothesis, theta)
    return np.asarray(result["input_std"], dtype=float)[record["hourly_index"]]


def fit_hypothesis_deterministic(hypothesis, records):
    hypothesis = str(hypothesis).upper()
    if hypothesis not in HYPOTHESES: raise ValueError("Unknown diffusion hypothesis.")
    lower, upper = _bounds(hypothesis)

    def residual_vector(theta):
        residuals = []
        for record in records:
            prediction = predicted_sigma(record, hypothesis, theta)
            residual = prediction[1:] - record["observed_sigma"][1:]
            if not np.all(np.isfinite(residual)): raise ValueError(f"{hypothesis} produced non-finite uncertainty residuals.")
            residuals.append(residual)
        return np.concatenate(residuals)

    candidates, failures = [], []
    for start in _initial_starts(hypothesis, records):
        try:
            initial_residual = residual_vector(start)
            if not np.all(np.isfinite(initial_residual)): raise ValueError("non-finite initial residual")
            fit = least_squares(
                residual_vector, start, bounds=(lower, upper),
                method="trf", x_scale="jac", max_nfev=4000,
                ftol=1e-10, xtol=1e-10, gtol=1e-10,
            )
            if fit.success and np.all(np.isfinite(fit.x)) and np.all(np.isfinite(fit.fun)):
                candidates.append(fit)
            else:
                failures.append(str(fit.message))
        except (ValueError, FloatingPointError, OverflowError) as exc:
            failures.append(str(exc))

    if not candidates:
        detail = failures[0] if failures else "no finite candidate"
        raise RuntimeError(f"{hypothesis} diffusion fit failed: {detail}")
    fit = min(candidates, key=lambda item: float(np.mean(item.fun ** 2)))
    return np.asarray(fit.x, dtype=float), float(np.mean(fit.fun ** 2))


def interval_score(y, lower, upper, alpha=0.05):
    y, lower, upper = map(lambda value: np.asarray(value, dtype=float), (y, lower, upper))
    score = upper - lower
    below, above = y < lower, y > upper
    score = score + (2.0 / alpha) * (lower - y) * below + (2.0 / alpha) * (y - upper) * above
    return score


def evaluate_hypothesis(hypothesis, theta, records, alpha=ALPHA):
    sigma_errors, sigma_errors_all = [], []
    covered, covered_all, widths, scores = [], [], [], []
    by_hour = []

    per_record = []
    z = float(norm.ppf(1.0 - alpha / 2.0))
    for record in records:
        sigma = predicted_sigma(record, hypothesis, theta)
        observed_sigma = record["observed_sigma"]
        sigma_errors.append(sigma[1:] - observed_sigma[1:])
        sigma_errors_all.append(sigma - observed_sigma)

        lower = record["oracle_I"] - z * sigma
        upper = record["oracle_I"] + z * sigma
        inside = (record["actual_I"] >= lower) & (record["actual_I"] <= upper)
        covered.append(inside[1:]); covered_all.append(inside)
        widths.append(upper - lower)
        scores.append(interval_score(record["actual_I"], lower, upper, alpha=alpha))

        per_record.append({
            "day": record["day"], "hypothesis": hypothesis,
            "sigma_rmse_h1_23": float(np.sqrt(np.mean((sigma[1:] - observed_sigma[1:]) ** 2))),
            "sigma_mae_h1_23": float(np.mean(np.abs(sigma[1:] - observed_sigma[1:]))),
            "coverage_h1_23_pct": 100.0 * float(np.mean(inside[1:])),
            "coverage_all_pct": 100.0 * float(np.mean(inside)),
            "mean_interval_width": float(np.mean(upper - lower)),
            "mean_interval_score": float(np.mean(scores[-1])),
        })

        for hour in range(DAY_HOURS):
            by_hour.append({
                "day": record["day"], "hypothesis": hypothesis, "hour": hour,
                "observed_s_I": observed_sigma[hour], "predicted_sde_sigma": sigma[hour],
                "oracle_I": record["oracle_I"][hour], "actual_I": record["actual_I"][hour],
                "lower": lower[hour], "upper": upper[hour], "inside": bool(inside[hour]),
            })

    process_error = np.concatenate(sigma_errors)
    all_error = np.concatenate(sigma_errors_all)
    covered_process = np.concatenate(covered)
    covered_full = np.concatenate(covered_all)
    width = np.concatenate(widths)
    score = np.concatenate(scores)
    observed = np.concatenate([record["observed_sigma"][1:] for record in records])
    predicted = np.concatenate([predicted_sigma(record, hypothesis, theta)[1:] for record in records])
    correlation = np.corrcoef(observed, predicted)[0, 1] if np.std(observed) > 0.0 and np.std(predicted) > 0.0 else np.nan

    summary = {
        "hypothesis": hypothesis, "test_days": len(records), "nominal_coverage_pct": 100.0 * (1.0 - alpha),
        "sigma_rmse_h1_23": float(np.sqrt(np.mean(process_error ** 2))),
        "sigma_mae_h1_23": float(np.mean(np.abs(process_error))),
        "sigma_bias_h1_23": float(np.mean(process_error)),
        "sigma_correlation_h1_23": float(correlation),
        "sigma_rmse_all_hours": float(np.sqrt(np.mean(all_error ** 2))),
        "coverage_h1_23_pct": 100.0 * float(np.mean(covered_process)),
        "coverage_all_pct": 100.0 * float(np.mean(covered_full)),
        "coverage_error_95_h1_23_pct_points": abs(100.0 * float(np.mean(covered_process)) - 95.0),
        "coverage_error_95_all_pct_points": abs(100.0 * float(np.mean(covered_full)) - 95.0),
        "mean_interval_width": float(np.mean(width)),
        "mean_interval_score": float(np.mean(score)),
    }
    for name, value in zip(PARAMETER_NAMES[hypothesis], theta): summary[name] = float(value)
    return summary, pd.DataFrame(per_record), pd.DataFrame(by_hour)


def hourly_comparison(hourly):
    rows = []
    for (hypothesis, hour), part in hourly.groupby(["hypothesis", "hour"], sort=True):
        error = part["predicted_sde_sigma"] - part["observed_s_I"]
        rows.append({
            "hypothesis": hypothesis, "hour": int(hour), "days": int(len(part)),
            "mean_observed_s_I": float(part["observed_s_I"].mean()),
            "mean_predicted_sde_sigma": float(part["predicted_sde_sigma"].mean()),
            "sigma_rmse": float(np.sqrt(np.mean(error ** 2))),
            "coverage_pct": 100.0 * float(part["inside"].mean()),
        })
    return pd.DataFrame(rows)


def save_plot(hourly_summary, path):
    fig, ax = plt.subplots(figsize=(12, 7))
    observed = hourly_summary.groupby("hour", sort=True)["mean_observed_s_I"].mean()
    ax.plot(observed.index, observed.values, marker="o", linewidth=2.0, label="Observed rolling $s_I$")
    for hypothesis in HYPOTHESES:
        part = hourly_summary[hourly_summary["hypothesis"] == hypothesis].sort_values("hour")
        ax.plot(part["hour"], part["mean_predicted_sde_sigma"], marker="o", label=f"{hypothesis} SDE")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Standard deviation")
    ax.set_xticks(np.arange(0, DAY_HOURS, 2))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def print_summary(summary, train_days, test_days, skipped):
    print()
    print("WM4 diffusion hypothesis comparison")
    print("deterministic WM4 coefficient fit + oracle prediction state; initial-state/parameter sensitivity excluded (P0=0)")
    print("H0-H2 coefficients are fitted deterministically by least squares to rolling observed s_I on the training dates.")
    print(f"train days={train_days}, test days={test_days}, skipped={len(skipped)}")
    print("H0 = time-periodic diffusion")
    print("H1 = H0 + sigma_I * I")
    print("H2 = H1 + sigma_U * |U|")
    print()
    columns = [
        "hypothesis", "coverage_h1_23_pct", "coverage_error_95_h1_23_pct_points",
        "coverage_all_pct", "sigma_rmse_h1_23", "sigma_correlation_h1_23",
        "mean_interval_width", "mean_interval_score",
    ]
    ranked = summary[columns].sort_values(["coverage_error_95_h1_23_pct_points", "mean_interval_score"])
    print(ranked.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    best_coverage = summary.sort_values(["coverage_error_95_h1_23_pct_points", "mean_interval_score"]).iloc[0]
    best_sigma = summary.sort_values("sigma_rmse_h1_23").iloc[0]
    best_score = summary.sort_values("mean_interval_score").iloc[0]
    print()
    print(f"best held-out 95% coverage calibration: {best_coverage['hypothesis']} "
          f"(coverage={best_coverage['coverage_h1_23_pct']:.2f}%, "
          f"error={best_coverage['coverage_error_95_h1_23_pct_points']:.2f} percentage points)")
    print(f"lowest held-out diffusion RMSE: {best_sigma['hypothesis']} ({best_sigma['sigma_rmse_h1_23']:.8f})")
    print(f"lowest held-out 95% interval score: {best_score['hypothesis']} ({best_score['mean_interval_score']:.8f})")
    print()
    print("fitted shared diffusion parameters")
    for _, row in summary.iterrows():
        names = PARAMETER_NAMES[row["hypothesis"]]
        values = ", ".join(f"{name}={row[name]:.8f}" for name in names)
        print(f"{row['hypothesis']}: {values}")
    print()
    print("Hour 0 is excluded when fitting/ranking process sigma because P0=0 implies zero SDE variance at the initial instant.")
    print("Coverage remains an oracle-centred diagnostic because x* is optimized from the full prediction day.")


def main():
    parser = ArgumentParser(description="Fit and compare WM4 H0-H2 SDE diffusion hypotheses against rolling observed input uncertainty.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--weeks", type=int, default=DEFAULT_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--train-fraction", type=float, default=0.70, help="Chronological fraction of usable prediction days used to fit shared diffusion parameters.")
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    if not 0.2 <= args.train_fraction <= 0.9: raise ValueError("train-fraction must lie in [0.2, 0.9].")

    frame, _ = aligned_wm4_data("MED")
    records, skipped = build_records(
        frame, args.start_day, args.end_day, args.weeks, args.steps_per_hour,
        args.max_state_iter, args.state_tol, fail_fast=args.fail_fast,
    )
    records = sorted(records, key=lambda record: record["day"])
    split = int(np.floor(args.train_fraction * len(records)))
    split = min(max(split, 2), len(records) - 2)
    train, test = records[:split], records[split:]

    summaries, daily_frames, hourly_frames = [], [], []
    for hypothesis in HYPOTHESES:
        theta, train_loss = fit_hypothesis_deterministic(hypothesis, train)
        summary, daily, hourly = evaluate_hypothesis(hypothesis, theta, test, alpha=ALPHA)
        summary["train_sigma_mse_h1_23"] = train_loss
        summaries.append(summary); daily_frames.append(daily); hourly_frames.append(hourly)

    summary = pd.DataFrame(summaries)
    summary["coverage_rank_95"] = summary["coverage_error_95_h1_23_pct_points"].rank(method="min", ascending=True).astype(int)
    daily = pd.concat(daily_frames, ignore_index=True)
    hourly = pd.concat(hourly_frames, ignore_index=True)
    by_hour = hourly_comparison(hourly)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "hypothesis_summary.csv", index=False)
    daily.to_csv(output / "daily_test_metrics.csv", index=False)
    hourly.to_csv(output / "hourly_test_predictions.csv", index=False)
    by_hour.to_csv(output / "hourly_hypothesis_summary.csv", index=False)
    if skipped: pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)
    save_plot(by_hour, output / "observed_vs_sde_uncertainty.png")
    print_summary(summary, len(train), len(test), skipped)
    print(); print(f"output={output}")


if __name__ == "__main__": main()
