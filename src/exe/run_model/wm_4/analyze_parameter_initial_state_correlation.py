from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.exe.run_model.wm_4.fit_daily import aligned_wm4_data, day_start, model_input
from src.exe.run_model.wm_4.predict_all_days_optimized import END_DAY, START_DAY, fit_history_optimized
from src.exe.run_model.wm_4.predict_daily import prediction_setup
from src.params.wm_4 import optimized_qp_params


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "wm_4" / "initial_state" / "parameter_correlation"
DEFAULT_WEEKS = 8
TARGETS = ("oracle_F0", "oracle_R0")
FEATURE_SETS = {
    "r": ("r_F", "r_U", "r_R"),
    "q_kappa": ("q_F", "q_U", "kappa_R"),
    "r_U0": ("r_F", "r_U", "r_R", "U0"),
    "derived": ("q_F", "q_U", "kappa_R", "damping_ratio", "equilibrium_gain", "U0"),
}
CORRELATION_FEATURES = (
    "r_F", "r_U", "r_R",
    "q_F", "q_U", "kappa_R",
    "damping_ratio", "equilibrium_gain", "U0", "equilibrium_F0",
)


def fit_day(frame, day, weeks=DEFAULT_WEEKS, steps_per_hour=60, max_state_iter=50, state_tol=1e-8):
    day = day_start(day)
    output_model, _, fit, _ = fit_history_optimized(
        frame, day, weeks, steps_per_hour,
        max_state_iter=max_state_iter, state_tol=state_tol,
    )
    _, _, U, _, model, _ = prediction_setup(frame, day, output_model, steps_per_hour)

    estimate = optimized_qp_params.estimate_prediction_initial_state(fit["init_cond"])
    x_hat = np.asarray(estimate["x_hat"], dtype=float)
    P0 = np.asarray(estimate["covariance"], dtype=float)

    prediction_data, _, _, _ = model_input(frame, day, steps_per_hour, 0, output_model)
    oracle = optimized_qp_params(model).optimize_prediction_initial_state(prediction_data, fit["r"])
    x_oracle = np.asarray(oracle["init_cond"], dtype=float)

    r = np.asarray(fit["r"], dtype=float)
    q, kappa_R = model.equivalent_q(r)
    natural_frequency = float(np.sqrt(-r[0]))
    damping_ratio = float(-r[2] / (2.0 * natural_frequency))
    equilibrium_gain = float(-r[1] / r[0])
    U0 = float(U(model.t[0]))
    equilibrium_F0 = equilibrium_gain * U0

    return {
        "day": day, "weekday": day.day_name(), "weeks": int(weeks),
        "r_F": r[0], "r_U": r[1], "r_R": r[2],
        "q_F": q[0], "q_U": q[1], "kappa_R": kappa_R,
        "natural_frequency": natural_frequency, "damping_ratio": damping_ratio,
        "equilibrium_gain": equilibrium_gain, "U0": U0, "equilibrium_F0": equilibrium_F0,
        "estimated_F0": x_hat[0], "estimated_R0": x_hat[1],
        "estimated_F0_std": float(np.sqrt(max(P0[0, 0], 0.0))),
        "estimated_R0_std": float(np.sqrt(max(P0[1, 1], 0.0))),
        "oracle_F0": x_oracle[0], "oracle_R0": x_oracle[1],
        "F0_estimation_error": x_hat[0] - x_oracle[0],
        "R0_estimation_error": x_hat[1] - x_oracle[1],
        "oracle_fit_mse": float(oracle["loss"]),
        "historical_fit_mse": float(fit["deterministic_loss"]),
        "state_iterations": int(fit["state_iterations"]),
    }


def build_frame(frame, start=START_DAY, end=END_DAY, weeks=DEFAULT_WEEKS, steps_per_hour=60, max_state_iter=50, state_tol=1e-8, fail_fast=False):
    start, end = day_start(start), day_start(end)
    rows, skipped = [], []
    for day in pd.date_range(start, end, freq="D"):
        try:
            row = fit_day(
                frame, day, weeks=weeks, steps_per_hour=steps_per_hour,
                max_state_iter=max_state_iter, state_tol=state_tol,
            )
            rows.append(row)
            print(
                f"{day:%Y-%m-%d} | "
                f"r=[{row['r_F']:.5f},{row['r_U']:.5f},{row['r_R']:.5f}] | "
                f"x_hat=[{row['estimated_F0']:.4f},{row['estimated_R0']:.4f}] | "
                f"x*=[{row['oracle_F0']:.4f},{row['oracle_R0']:.4f}]"
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append((day, str(exc))); print(f"{day:%Y-%m-%d} | skipped | {exc}")
    if len(rows) < 10: raise RuntimeError("At least 10 usable days are required for parameter/state correlation analysis.")
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True), skipped


def correlation_table(frame):
    rows = []
    for target in TARGETS:
        for feature in CORRELATION_FEATURES:
            part = frame[[target, feature]].dropna()
            if len(part) < 3 or part[target].nunique() < 2 or part[feature].nunique() < 2:
                pearson_r = pearson_p = spearman_r = spearman_p = np.nan
            else:
                pearson_r, pearson_p = pearsonr(part[feature], part[target])
                spearman_r, spearman_p = spearmanr(part[feature], part[target])
            rows.append({
                "target": target, "feature": feature, "n": int(len(part)),
                "pearson_r": pearson_r, "pearson_p": pearson_p,
                "spearman_r": spearman_r, "spearman_p": spearman_p,
            })
    return pd.DataFrame(rows)


def standardized_ols(frame, target, features):
    part = frame[[target, *features]].dropna()
    X, y = part[list(features)].to_numpy(dtype=float), part[target].to_numpy(dtype=float)
    if len(part) <= len(features) + 1: raise RuntimeError("Insufficient observations for regression.")
    mean, scale = X.mean(axis=0), X.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    Z = (X - mean) / scale
    design = np.column_stack((np.ones(len(Z)), Z))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ beta
    residual = y - fitted
    ss_res = float(residual @ residual)
    centered = y - y.mean()
    ss_tot = float(centered @ centered)
    return {
        "beta": beta, "mean": mean, "scale": scale,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else np.nan,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "n": int(len(part)),
    }


def predict_ols(model, row, features):
    x = row[list(features)].to_numpy(dtype=float)
    z = (x - model["mean"]) / model["scale"]
    return float(model["beta"][0] + z @ model["beta"][1:])


def regression_summary(frame):
    rows, coefficients = [], []
    for target in TARGETS:
        for name, features in FEATURE_SETS.items():
            model = standardized_ols(frame, target, features)
            rows.append({
                "target": target, "feature_set": name, "n": model["n"],
                "in_sample_r2": model["r2"], "in_sample_rmse": model["rmse"],
            })
            coefficients.append({
                "target": target, "feature_set": name, "term": "intercept",
                "standardized_coefficient": model["beta"][0],
            })
            for feature, coefficient in zip(features, model["beta"][1:]):
                coefficients.append({
                    "target": target, "feature_set": name, "term": feature,
                    "standardized_coefficient": coefficient,
                })
    return pd.DataFrame(rows), pd.DataFrame(coefficients)


def walk_forward(frame, min_train=28):
    min_train = int(min_train)
    if min_train < 7: raise ValueError("min_train must be at least 7.")
    if min_train >= len(frame): raise ValueError("min_train must be smaller than the number of usable days.")

    rows = []
    for i in range(min_train, len(frame)):
        train, row = frame.iloc[:i], frame.iloc[i]
        result = {
            "day": row["day"],
            "oracle_F0": row["oracle_F0"], "oracle_R0": row["oracle_R0"],
            "historical_F0": row["estimated_F0"], "historical_R0": row["estimated_R0"],
            "expanding_mean_F0": float(train["oracle_F0"].mean()),
            "expanding_mean_R0": float(train["oracle_R0"].mean()),
        }
        for target in TARGETS:
            short = "F0" if target == "oracle_F0" else "R0"
            for name, features in FEATURE_SETS.items():
                try:
                    model = standardized_ols(train, target, features)
                    result[f"{short}_{name}"] = predict_ols(model, row, features)
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    result[f"{short}_{name}"] = np.nan
        rows.append(result)
    return pd.DataFrame(rows)


def prediction_summary(predictions):
    rows = []
    for short in ("F0", "R0"):
        actual = predictions[f"oracle_{short}"].to_numpy(dtype=float)
        predictors = {
            "historical_state_estimate": predictions[f"historical_{short}"].to_numpy(dtype=float),
            "expanding_oracle_mean": predictions[f"expanding_mean_{short}"].to_numpy(dtype=float),
            **{name: predictions[f"{short}_{name}"].to_numpy(dtype=float) for name in FEATURE_SETS},
        }
        historical = predictors["historical_state_estimate"]
        mask_hist = np.isfinite(actual) & np.isfinite(historical)
        historical_rmse = float(np.sqrt(np.mean((historical[mask_hist] - actual[mask_hist]) ** 2)))

        for name, predicted in predictors.items():
            mask = np.isfinite(actual) & np.isfinite(predicted)
            y, yhat = actual[mask], predicted[mask]
            error = yhat - y
            rmse = float(np.sqrt(np.mean(error ** 2)))
            correlation = float(np.corrcoef(y, yhat)[0, 1]) if len(y) > 1 and np.std(y) > 0.0 and np.std(yhat) > 0.0 else np.nan
            rows.append({
                "target": short, "predictor": name, "n": int(len(y)),
                "rmse": rmse, "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)),
                "correlation": correlation,
                "rmse_improvement_vs_historical_pct":
                    100.0 * (historical_rmse - rmse) / historical_rmse if historical_rmse > 0.0 else np.nan,
            })
    return pd.DataFrame(rows)


def error_correlations(frame):
    rows = []
    for error in ("F0_estimation_error", "R0_estimation_error"):
        for feature in CORRELATION_FEATURES:
            part = frame[[error, feature]].dropna()
            rho = part[error].corr(part[feature], method="pearson") if len(part) >= 3 else np.nan
            rows.append({"error": error, "feature": feature, "n": int(len(part)), "pearson_r": rho})
    return pd.DataFrame(rows)


def save_plots(frame, correlations, predictions, output):
    output.mkdir(parents=True, exist_ok=True)

    raw = correlations[correlations["feature"].isin(("r_F", "r_U", "r_R"))]
    matrix = raw.pivot(index="target", columns="feature", values="pearson_r").reindex(index=TARGETS, columns=("r_F", "r_U", "r_R"))
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(3), matrix.columns)
    ax.set_yticks(np.arange(2), ("oracle F0", "oracle R0"))
    ax.set_title("WM4 fitted r versus oracle initial state")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout(); fig.savefig(output / "r_oracle_state_correlation.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, target in enumerate(TARGETS):
        for j, feature in enumerate(("r_F", "r_U", "r_R")):
            ax = axes[i, j]
            x, y = frame[feature].to_numpy(dtype=float), frame[target].to_numpy(dtype=float)
            ax.scatter(x, y, alpha=0.65)
            if np.std(x) > 1e-12:
                beta = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 100)
                ax.plot(xs, beta[0] * xs + beta[1])
            corr = frame[[feature, target]].corr().iloc[0, 1]
            ax.set_xlabel(feature); ax.set_ylabel(target); ax.set_title(f"r={corr:.3f}"); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "r_vs_oracle_state_scatter.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    for short in ("F0", "R0"):
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(predictions["day"], predictions[f"oracle_{short}"], label=f"oracle {short}")
        ax.plot(predictions["day"], predictions[f"historical_{short}"], label="historical state estimate")
        ax.plot(predictions["day"], predictions[f"{short}_r"], label="walk-forward r regression")
        ax.plot(predictions["day"], predictions[f"{short}_r_U0"], label="walk-forward r + U0 regression")
        ax.set_xlabel("Prediction day"); ax.set_ylabel(short)
        ax.set_title(f"Walk-forward {short} prediction from fitted WM4 parameters")
        ax.legend(); ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(output / f"{short.lower()}_walk_forward.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def print_summary(frame, correlations, regressions, prediction, errors, skipped):
    print()
    print(f"days fitted={len(frame)}, skipped={len(skipped)}")
    print("Oracle [F0,R0] is treated as the target state; distance from the oracle is initial-state estimation error.")
    print()
    ranked = correlations.assign(abs_r=np.abs(correlations["pearson_r"])).sort_values(["target", "abs_r"], ascending=[True, False])
    print("strongest absolute parameter/oracle-state correlations")
    print(ranked.groupby("target", as_index=False).head(6).drop(columns="abs_r").to_string(index=False))
    print()
    print("multivariate in-sample fits")
    print(regressions.to_string(index=False))
    print()
    print("walk-forward state prediction")
    print(prediction.sort_values(["target", "rmse"]).to_string(index=False))
    print()
    print("parameter correlation with current initial-state estimation error")
    ranked_error = errors.assign(abs_r=np.abs(errors["pearson_r"])).sort_values(["error", "abs_r"], ascending=[True, False])
    print(ranked_error.groupby("error", as_index=False).head(5).drop(columns="abs_r").to_string(index=False))


def main():
    parser = ArgumentParser(description="Test whether fitted WM4 dynamics parameters contain predictive information about the oracle prediction-day initial state.")
    parser.add_argument("--start-day", default=START_DAY)
    parser.add_argument("--end-day", default=END_DAY)
    parser.add_argument("--weeks", type=int, default=DEFAULT_WEEKS)
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--min-train", type=int, default=28)
    parser.add_argument("--max-state-iter", type=int, default=50)
    parser.add_argument("--state-tol", type=float, default=1e-8)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    frame_data, _ = aligned_wm4_data("MED")
    frame, skipped = build_frame(
        frame_data, args.start_day, args.end_day, args.weeks, args.steps_per_hour,
        max_state_iter=args.max_state_iter, state_tol=args.state_tol, fail_fast=args.fail_fast,
    )
    correlations = correlation_table(frame)
    regressions, coefficients = regression_summary(frame)
    predictions = walk_forward(frame, args.min_train)
    prediction = prediction_summary(predictions)
    errors = error_correlations(frame)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "parameter_state_data.csv", index=False)
    correlations.to_csv(output / "parameter_oracle_correlations.csv", index=False)
    regressions.to_csv(output / "regression_summary.csv", index=False)
    coefficients.to_csv(output / "regression_coefficients.csv", index=False)
    predictions.to_csv(output / "walk_forward_predictions.csv", index=False)
    prediction.to_csv(output / "walk_forward_summary.csv", index=False)
    errors.to_csv(output / "parameter_estimation_error_correlations.csv", index=False)
    if skipped: pd.DataFrame([{"day": day, "reason": reason} for day, reason in skipped]).to_csv(output / "skipped_days.csv", index=False)
    save_plots(frame, correlations, predictions, output)
    print_summary(frame, correlations, regressions, prediction, errors, skipped)
    print(); print(f"output={output}")


if __name__ == "__main__": main()
