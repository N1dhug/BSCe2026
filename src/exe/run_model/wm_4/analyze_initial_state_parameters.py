from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_optimized_top_end"
OUTPUT = INPUT / "initial_state_parameter_analysis"
TARGETS = ("F_0", "R_0")
PARAMETERS = ("r_F", "r_U", "r_R")
FEATURE_SETS = {
    "raw": ("r_F", "r_U", "r_R"),
    "raw_U0": ("r_F", "r_U", "r_R", "U0"),
    "derived": ("equilibrium_gain", "natural_frequency", "damping_ratio", "U0", "equilibrium_F0"),
}
CORRELATION_FEATURES = (
    "r_F", "r_U", "r_R", "U0",
    "equilibrium_gain", "natural_frequency", "damping_ratio", "equilibrium_F0",
)


def load_results(root):
    root = Path(root)
    daily_path, hourly_path = root / "daily_metrics.csv", root / "hourly_predictions.csv"
    if not daily_path.exists(): raise FileNotFoundError(f"Missing {daily_path}. Run the optimized top-end predictor first.")
    if not hourly_path.exists(): raise FileNotFoundError(f"Missing {hourly_path}. Run the optimized top-end predictor first.")
    daily = pd.read_csv(daily_path, parse_dates=["day"])
    hourly = pd.read_csv(hourly_path)
    required_daily = {"day", "F_0", "R_0", *PARAMETERS}
    missing = sorted(required_daily.difference(daily.columns))
    if missing: raise RuntimeError(f"{daily_path} is missing required columns: {', '.join(missing)}.")
    if "day" not in hourly.columns or "U" not in hourly.columns: raise RuntimeError(f"{hourly_path} must contain day and U.")
    hourly["day"] = pd.to_datetime(hourly["day"])
    if "time" in hourly.columns: hourly["time"] = pd.to_datetime(hourly["time"], errors="coerce")
    sort_cols = ["day", "time"] if "time" in hourly.columns else ["day"]
    u0 = hourly.sort_values(sort_cols).groupby("day", as_index=False).first()[["day", "U"]].rename(columns={"U": "U0"})
    frame = daily.merge(u0, on="day", how="inner").sort_values("day").drop_duplicates("day", keep="last").reset_index(drop=True)
    if len(frame) < 10: raise RuntimeError("At least 10 overlapping optimized top-end days are required.")
    return frame


def add_derived(frame):
    result = frame.copy()
    r_F = result["r_F"].to_numpy(dtype=float)
    r_U = result["r_U"].to_numpy(dtype=float)
    r_R = result["r_R"].to_numpy(dtype=float)
    U0 = result["U0"].to_numpy(dtype=float)
    F0 = result["F_0"].to_numpy(dtype=float)
    R0 = result["R_0"].to_numpy(dtype=float)
    valid = r_F < 0.0
    natural_frequency = np.full(len(result), np.nan)
    natural_frequency[valid] = np.sqrt(-r_F[valid])
    equilibrium_gain = np.divide(-r_U, r_F, out=np.full(len(result), np.nan), where=np.abs(r_F) > 1e-12)
    damping_ratio = np.divide(-r_R, 2.0 * natural_frequency, out=np.full(len(result), np.nan), where=natural_frequency > 1e-12)
    result["equilibrium_gain"] = equilibrium_gain
    result["natural_frequency"] = natural_frequency
    result["damping_ratio"] = damping_ratio
    result["equilibrium_F0"] = equilibrium_gain * U0
    result["initial_dR"] = r_F * F0 + r_U * U0 + r_R * R0
    result["F0_equilibrium_residual"] = F0 - result["equilibrium_F0"]
    return result


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
                "target": target, "feature": feature, "n": len(part),
                "pearson_r": pearson_r, "pearson_p": pearson_p,
                "spearman_r": spearman_r, "spearman_p": spearman_p,
            })
    return pd.DataFrame(rows)


def standardized_ols(train, target, features):
    part = train[[target, *features]].dropna()
    if len(part) <= len(features): raise RuntimeError("Insufficient observations for regression.")
    X = part[list(features)].to_numpy(dtype=float)
    y = part[target].to_numpy(dtype=float)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    Z = (X - mean) / scale
    design = np.column_stack((np.ones(len(Z)), Z))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ beta
    residual = y - fitted
    ss_res = float(residual @ residual)
    centered = y - y.mean()
    ss_tot = float(centered @ centered)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else np.nan
    return {"beta": beta, "mean": mean, "scale": scale, "r2": r2, "n": len(part)}


def predict_ols(model, row, features):
    x = row[list(features)].to_numpy(dtype=float)
    z = (x - model["mean"]) / model["scale"]
    return float(model["beta"][0] + z @ model["beta"][1:])


def regression_summary(frame):
    rows, coefficients = [], []
    for target in TARGETS:
        for name, features in FEATURE_SETS.items():
            model = standardized_ols(frame, target, features)
            rows.append({"target": target, "feature_set": name, "n": model["n"], "in_sample_r2": model["r2"]})
            coefficients.append({"target": target, "feature_set": name, "term": "intercept", "coefficient": model["beta"][0]})
            for feature, coefficient in zip(features, model["beta"][1:]):
                coefficients.append({"target": target, "feature_set": name, "term": feature, "coefficient": coefficient})
    return pd.DataFrame(rows), pd.DataFrame(coefficients)


def walk_forward(frame, min_train=28):
    min_train = int(min_train)
    if min_train < 7: raise ValueError("min_train must be at least 7.")
    if len(frame) <= min_train: raise ValueError("min_train must be smaller than the number of available days.")
    rows = []
    for i in range(min_train, len(frame)):
        train, row = frame.iloc[:i], frame.iloc[i]
        base = {
            "day": row["day"], "F_0": row["F_0"], "R_0": row["R_0"],
            "zero_F0": 0.0, "zero_R0": 0.0,
            "expanding_mean_F0": float(train["F_0"].mean()),
            "expanding_mean_R0": float(train["R_0"].mean()),
        }
        for target in TARGETS:
            for name, features in FEATURE_SETS.items():
                try:
                    model = standardized_ols(train, target, features)
                    base[f"{target}_{name}"] = predict_ols(model, row, features)
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    base[f"{target}_{name}"] = np.nan
        rows.append(base)
    return pd.DataFrame(rows)


def prediction_summary(predictions):
    rows = []
    for target in TARGETS:
        actual = predictions[target].to_numpy(dtype=float)
        candidates = {
            "zero": predictions[f"zero_{'F0' if target == 'F_0' else 'R0'}"].to_numpy(dtype=float),
            "expanding_mean": predictions[f"expanding_mean_{'F0' if target == 'F_0' else 'R0'}"].to_numpy(dtype=float),
            **{name: predictions[f"{target}_{name}"].to_numpy(dtype=float) for name in FEATURE_SETS},
        }
        zero_error = actual - candidates["zero"]
        zero_rmse = float(np.sqrt(np.mean(zero_error ** 2)))
        mean_error = actual - candidates["expanding_mean"]
        mean_rmse = float(np.sqrt(np.mean(mean_error ** 2)))
        for name, predicted in candidates.items():
            mask = np.isfinite(actual) & np.isfinite(predicted)
            y, yhat = actual[mask], predicted[mask]
            error = y - yhat
            rmse = float(np.sqrt(np.mean(error ** 2)))
            mae = float(np.mean(np.abs(error)))
            bias = float(np.mean(yhat - y))
            correlation = float(np.corrcoef(y, yhat)[0, 1]) if len(y) > 1 and np.std(y) > 0.0 and np.std(yhat) > 0.0 else np.nan
            rows.append({
                "target": target, "predictor": name, "n": len(y),
                "rmse": rmse, "mae": mae, "bias": bias, "correlation": correlation,
                "rmse_improvement_vs_zero_pct": 100.0 * (zero_rmse - rmse) / zero_rmse if zero_rmse > 0.0 else np.nan,
                "rmse_improvement_vs_expanding_mean_pct": 100.0 * (mean_rmse - rmse) / mean_rmse if mean_rmse > 0.0 else np.nan,
            })
    return pd.DataFrame(rows)


def equilibrium_summary(frame):
    residual = frame["initial_dR"].to_numpy(dtype=float)
    f_residual = frame["F0_equilibrium_residual"].to_numpy(dtype=float)
    return pd.DataFrame([{
        "days": len(frame),
        "initial_dR_mean": float(np.mean(residual)),
        "initial_dR_std": float(np.std(residual, ddof=1)),
        "initial_dR_rmse_from_zero": float(np.sqrt(np.mean(residual ** 2))),
        "F0_minus_equilibrium_mean": float(np.mean(f_residual)),
        "F0_minus_equilibrium_std": float(np.std(f_residual, ddof=1)),
        "F0_equilibrium_rmse": float(np.sqrt(np.mean(f_residual ** 2))),
        "F0_equilibrium_correlation": float(frame["F_0"].corr(frame["equilibrium_F0"])),
    }])


def save_plots(frame, correlations, predictions, output):
    output.mkdir(parents=True, exist_ok=True)

    matrix = correlations.pivot(index="target", columns="feature", values="pearson_r").reindex(index=TARGETS, columns=CORRELATION_FEATURES)
    fig, ax = plt.subplots(figsize=(12, 4))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    ax.set_title("Pearson correlation: fitted dynamics versus oracle initial state")
    fig.colorbar(image, ax=ax, label="correlation")
    fig.tight_layout(); fig.savefig(output / "parameter_state_correlations.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    for target in TARGETS:
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(predictions["day"], predictions[target], label=f"oracle {target}")
        ax.plot(predictions["day"], predictions[f"expanding_mean_{'F0' if target == 'F_0' else 'R0'}"], label="expanding mean")
        for name in FEATURE_SETS: ax.plot(predictions["day"], predictions[f"{target}_{name}"], label=name)
        ax.axhline(0.0, linestyle="--", label="zero")
        ax.set_xlabel("Prediction day"); ax.set_ylabel(target)
        ax.set_title(f"Walk-forward prediction of {target} from fitted dynamics")
        ax.legend(); ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(output / f"{target.replace('_', '').lower()}_walk_forward.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(frame["equilibrium_F0"], frame["F_0"], alpha=0.7)
    limit = max(float(np.max(np.abs(frame["equilibrium_F0"]))), float(np.max(np.abs(frame["F_0"]))), 1e-6)
    ax.plot([-limit, limit], [-limit, limit], linestyle="--")
    ax.set_xlabel("F0 implied by constant-price equilibrium")
    ax.set_ylabel("Oracle F0")
    ax.set_title("Oracle F0 versus equilibrium-implied F0")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "F0_equilibrium_relation.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def print_summary(correlations, regression, prediction, equilibrium):
    print("strongest absolute parameter-state correlations")
    ranked = correlations.assign(abs_pearson=np.abs(correlations["pearson_r"])).sort_values(["target", "abs_pearson"], ascending=[True, False])
    print(ranked.groupby("target", as_index=False).head(5).drop(columns="abs_pearson").to_string(index=False))
    print()
    print("in-sample linear regressions")
    print(regression.to_string(index=False))
    print()
    print("walk-forward initial-state prediction")
    print(prediction.to_string(index=False))
    print()
    print("initial dynamic-equilibrium diagnostics")
    print(equilibrium.to_string(index=False))


def main():
    parser = ArgumentParser(description="Analyze whether optimized WM4 rate parameters and midnight price explain the oracle prediction-day initial state.")
    parser.add_argument("--input", default=str(INPUT))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--min-train", type=int, default=28, help="Minimum earlier days before walk-forward state prediction begins.")
    args = parser.parse_args()

    frame = add_derived(load_results(args.input))
    correlations = correlation_table(frame)
    regressions, coefficients = regression_summary(frame)
    predictions = walk_forward(frame, args.min_train)
    prediction = prediction_summary(predictions)
    equilibrium = equilibrium_summary(frame)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "state_parameter_data.csv", index=False)
    correlations.to_csv(output / "correlations.csv", index=False)
    regressions.to_csv(output / "regression_summary.csv", index=False)
    coefficients.to_csv(output / "regression_coefficients.csv", index=False)
    predictions.to_csv(output / "walk_forward_predictions.csv", index=False)
    prediction.to_csv(output / "walk_forward_summary.csv", index=False)
    equilibrium.to_csv(output / "equilibrium_summary.csv", index=False)
    save_plots(frame, correlations, predictions, output)
    print_summary(correlations, regressions, prediction, equilibrium)


if __name__ == "__main__": main()
