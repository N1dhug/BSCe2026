from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.calibrate_mpc_sigma import _prepare, fit_variance_components


ROOT = Path(__file__).resolve().parents[4]
MPC_OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_mpc"
DEFAULT_FORECASTS = MPC_OUTPUT / "mpc_forecasts.csv"
DEFAULT_ESTIMATES = MPC_OUTPUT / "sigma_from_mpc" / "variance_component_estimates.csv"
OUTPUT = MPC_OUTPUT / "aggregate_uncertainty"


def _load_estimate(path, mode, prepared):
    path = Path(path)
    if path.exists():
        estimates = pd.read_csv(path)
        selected = estimates[estimates["mode"].astype(str) == str(mode)]
        if selected.empty: raise ValueError(f"No variance estimate with mode={mode!r} in {path}.")
        row = selected.iloc[0]
        return {
            "r0_variance": float(row["R0_variance"]),
            "sigma_R0": float(row["sigma_R0"]),
            "q": float(row["wiener_q"]),
            "sigma_W": float(row["sigma_W"]),
            "validation_start": pd.Timestamp(row["validation_start"]) if pd.notna(row.get("validation_start")) else None,
            "source": str(path),
        }

    if mode != "all": raise FileNotFoundError(f"{path} is required for estimate-mode={mode!r}.")
    fit = fit_variance_components(prepared)
    return {
        "r0_variance": float(fit["r0_variance"]),
        "sigma_R0": float(fit["sigma_R0"]),
        "q": float(fit["q"]),
        "sigma_W": float(fit["sigma"]),
        "validation_start": None,
        "source": "fitted directly from saved MPC forecasts",
    }


def _component_rows(frame, estimate):
    data = frame.copy()
    data["observed_squared_error"] = data["forecast_error"].to_numpy(dtype=float) ** 2
    data["F_variance"] = np.maximum(data["initial_variance"].to_numpy(dtype=float), 0.0)
    data["R0_variance"] = np.maximum(
        float(estimate["r0_variance"]) * data["unit_initial_r_variance"].to_numpy(dtype=float),
        0.0,
    )
    data["W_variance"] = np.maximum(
        float(estimate["q"]) * data["unit_process_variance"].to_numpy(dtype=float),
        0.0,
    )
    data["predicted_variance"] = data["F_variance"] + data["R0_variance"] + data["W_variance"]
    return data


def _complete_origins(frame, horizon):
    horizon = int(horizon)
    valid = []
    for origin, part in frame.groupby("origin", sort=False):
        leads = np.sort(part["lead_hours"].astype(int).unique())
        if np.array_equal(leads, np.arange(1, horizon + 1, dtype=int)): valid.append(origin)
    return frame[frame["origin"].isin(valid)].copy()


def _aggregate_metrics(part):
    observed_mse = float(part["observed_squared_error"].mean())
    F_mse = float(part["F_variance"].mean())
    R0_mse = float(part["R0_variance"].mean())
    W_mse = float(part["W_variance"].mean())
    predicted_mse = F_mse + R0_mse + W_mse
    observed_rmse = float(np.sqrt(max(observed_mse, 0.0)))
    predicted_rmse = float(np.sqrt(max(predicted_mse, 0.0)))
    denom = predicted_mse if predicted_mse > 0.0 else np.nan
    return {
        "forecasts": int(len(part)),
        "origins": int(part["origin"].nunique()),
        "days": int(part["origin_day"].nunique()),
        "observed_mse": observed_mse,
        "observed_rmse": observed_rmse,
        "F_mse": F_mse,
        "R0_mse": R0_mse,
        "W_mse": W_mse,
        "predicted_mse": predicted_mse,
        "predicted_rmse": predicted_rmse,
        "predicted_to_observed_rmse_ratio": predicted_rmse / observed_rmse if observed_rmse > 0.0 else np.nan,
        "rmse_error": predicted_rmse - observed_rmse,
        "rmse_error_pct": 100.0 * (predicted_rmse / observed_rmse - 1.0) if observed_rmse > 0.0 else np.nan,
        "F_variance_share_pct": 100.0 * F_mse / denom if np.isfinite(denom) else np.nan,
        "R0_variance_share_pct": 100.0 * R0_mse / denom if np.isfinite(denom) else np.nan,
        "W_variance_share_pct": 100.0 * W_mse / denom if np.isfinite(denom) else np.nan,
        "F_only_rmse": float(np.sqrt(max(F_mse, 0.0))),
        "F_R0_rmse": float(np.sqrt(max(F_mse + R0_mse, 0.0))),
        "F_R0_W_rmse": predicted_rmse,
    }


def per_origin_summary(frame, horizon):
    rows = []
    for origin, part in frame.groupby("origin", sort=True):
        part = part[part["lead_hours"].astype(int) <= int(horizon)]
        if part["lead_hours"].nunique() != int(horizon): continue
        metrics = _aggregate_metrics(part)
        rows.append({
            "origin": pd.Timestamp(origin),
            "origin_day": pd.Timestamp(part["origin_day"].iloc[0]),
            "origin_hour": int(pd.Timestamp(origin).hour),
            "horizon_hours": int(horizon),
            **metrics,
        })
    return pd.DataFrame(rows)


def per_day_summary(frame, horizon):
    rows = []
    for day, part in frame.groupby("origin_day", sort=True):
        part = part[part["lead_hours"].astype(int) <= int(horizon)]
        if part.empty: continue
        rows.append({
            "origin_day": pd.Timestamp(day),
            "horizon_hours": int(horizon),
            **_aggregate_metrics(part),
        })
    return pd.DataFrame(rows)


def cumulative_horizon_summary(frame, max_horizon):
    rows = []
    for horizon in range(1, int(max_horizon) + 1):
        part = frame[frame["lead_hours"].astype(int) <= horizon]
        metrics = _aggregate_metrics(part)
        rows.append({"horizon_hours": horizon, **metrics})
    return pd.DataFrame(rows)


def _rmse_quality(summary):
    if summary.empty: return {}
    observed = summary["observed_rmse"].to_numpy(dtype=float)
    predicted = summary["predicted_rmse"].to_numpy(dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    if observed.size == 0: return {}
    correlation = float(np.corrcoef(observed, predicted)[0, 1]) if observed.size > 1 and np.std(observed) > 0.0 and np.std(predicted) > 0.0 else np.nan
    return {
        "count": int(observed.size),
        "rmse_prediction_mae": float(np.mean(np.abs(predicted - observed))),
        "rmse_prediction_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "rmse_prediction_bias": float(np.mean(predicted - observed)),
        "rmse_prediction_correlation": correlation,
    }


def save_plot(cumulative, daily, path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    h = cumulative["horizon_hours"].to_numpy(dtype=int)
    axes[0].plot(h, cumulative["observed_rmse"], marker="o", label="Observed MPC RMSE")
    axes[0].plot(h, cumulative["predicted_rmse"], marker="o", label="Predicted stochastic RMSE")
    axes[0].set_xlabel("Cumulative MPC prediction horizon (hours)")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(h)
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].bar(h, cumulative["F_variance_share_pct"], label="Historical F uncertainty")
    axes[1].bar(
        h, cumulative["R0_variance_share_pct"],
        bottom=cumulative["F_variance_share_pct"],
        label="Initial R uncertainty",
    )
    axes[1].bar(
        h, cumulative["W_variance_share_pct"],
        bottom=cumulative["F_variance_share_pct"] + cumulative["R0_variance_share_pct"],
        label="Wiener process",
    )
    axes[1].set_xlabel("Cumulative MPC prediction horizon (hours)")
    axes[1].set_ylabel("Share of predicted variance (%)")
    axes[1].set_xticks(h)
    axes[1].set_ylim(0.0, 100.0)
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("WM4 MPC aggregate stochastic RMSE calibration")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if daily.empty: return
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily["origin_day"], daily["observed_rmse"], label="Observed daily aggregate RMSE")
    ax.plot(daily["origin_day"], daily["predicted_rmse"], label="Predicted daily aggregate RMSE")
    ax.set_xlabel("Target day")
    ax.set_ylabel("8-hour-window aggregate RMSE")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path.with_name("daily_observed_vs_predicted_rmse.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(overall, cumulative, origin_quality, day_quality, estimate, mode):
    print(f"estimate mode={mode}")
    print(f"sigma_R0={estimate['sigma_R0']:.8f}, sigma_W={estimate['sigma_W']:.8f}")
    print()
    print("aggregate MPC uncertainty quality:")
    print(f"observed RMSE          = {overall['observed_rmse']:.8f}")
    print(f"predicted RMSE         = {overall['predicted_rmse']:.8f}")
    print(f"predicted/observed     = {overall['predicted_to_observed_rmse_ratio']:.4f}")
    print(f"RMSE error             = {overall['rmse_error']:.8f} ({overall['rmse_error_pct']:.2f}%)")
    print()
    print("predicted variance decomposition:")
    print(f"F uncertainty          = {overall['F_mse']:.8f} ({overall['F_variance_share_pct']:.2f}%)")
    print(f"R0 uncertainty         = {overall['R0_mse']:.8f} ({overall['R0_variance_share_pct']:.2f}%)")
    print(f"Wiener process         = {overall['W_mse']:.8f} ({overall['W_variance_share_pct']:.2f}%)")
    print(f"total predicted MSE    = {overall['predicted_mse']:.8f}")
    print(f"observed MSE           = {overall['observed_mse']:.8f}")
    print()
    print("cumulative horizon calibration:")
    print(f"{'H':>4}{'obs RMSE':>12}{'pred RMSE':>12}{'ratio':>10}{'F%':>9}{'R0%':>9}{'W%':>9}")
    for _, row in cumulative.iterrows():
        print(
            f"{int(row['horizon_hours']):>4d}"
            f"{row['observed_rmse']:>12.8f}"
            f"{row['predicted_rmse']:>12.8f}"
            f"{row['predicted_to_observed_rmse_ratio']:>10.3f}"
            f"{row['F_variance_share_pct']:>8.2f}%"
            f"{row['R0_variance_share_pct']:>8.2f}%"
            f"{row['W_variance_share_pct']:>8.2f}%"
        )
    print()
    if origin_quality:
        print(
            "per-origin predicted-vs-observed 8h RMSE | "
            f"MAE={origin_quality['rmse_prediction_mae']:.8f}, "
            f"RMSE={origin_quality['rmse_prediction_rmse']:.8f}, "
            f"bias={origin_quality['rmse_prediction_bias']:.8f}, "
            f"corr={origin_quality['rmse_prediction_correlation']:.4f}"
        )
    if day_quality:
        print(
            "per-day predicted-vs-observed aggregate RMSE | "
            f"MAE={day_quality['rmse_prediction_mae']:.8f}, "
            f"RMSE={day_quality['rmse_prediction_rmse']:.8f}, "
            f"bias={day_quality['rmse_prediction_bias']:.8f}, "
            f"corr={day_quality['rmse_prediction_correlation']:.4f}"
        )


def main():
    parser = ArgumentParser(description="Aggregate WM4 MPC stochastic uncertainty over complete receding 1-8 hour prediction windows using saved MPC forecasts. Compares observed squared forecast error with the predicted F-state, initial-R and Wiener-process variance contributions without rerunning WM4.")
    parser.add_argument("--forecasts", default=str(DEFAULT_FORECASTS))
    parser.add_argument("--estimates", default=str(DEFAULT_ESTIMATES))
    parser.add_argument("--estimate-mode", choices=("all", "chronological_holdout"), default="all")
    parser.add_argument("--steps-per-hour", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--include-calibration-period", action="store_true", help="With chronological_holdout estimates, evaluate all saved days instead of only the held-out later days.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    if args.horizon < 1 or args.horizon > 8: raise ValueError("horizon must lie in [1, 8].")

    forecast_path = Path(args.forecasts)
    if not forecast_path.exists(): raise FileNotFoundError(forecast_path)

    prepared = _prepare(pd.read_csv(forecast_path), args.steps_per_hour)
    estimate = _load_estimate(args.estimates, args.estimate_mode, prepared)

    if args.estimate_mode == "chronological_holdout" and not args.include_calibration_period:
        if estimate["validation_start"] is None: raise ValueError("chronological_holdout estimate is missing validation_start.")
        prepared = prepared[prepared["origin_day"] >= estimate["validation_start"]].copy()

    data = _component_rows(prepared, estimate)
    data = data[data["lead_hours"].astype(int) <= int(args.horizon)].copy()
    data = _complete_origins(data, args.horizon)
    if data.empty: raise RuntimeError("No complete MPC prediction windows remain after filtering.")

    overall = _aggregate_metrics(data)
    origin = per_origin_summary(data, args.horizon)
    daily = per_day_summary(data, args.horizon)
    cumulative = cumulative_horizon_summary(data, args.horizon)
    origin_quality = _rmse_quality(origin)
    day_quality = _rmse_quality(daily)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mode = args.estimate_mode
    suffix = f"{mode}_{int(args.horizon)}h"

    data.to_csv(output / f"forecast_variance_rows_{suffix}.csv", index=False)
    origin.to_csv(output / f"per_origin_rmse_{suffix}.csv", index=False)
    daily.to_csv(output / f"per_day_rmse_{suffix}.csv", index=False)
    cumulative.to_csv(output / f"cumulative_horizon_{suffix}.csv", index=False)

    summary = {
        "estimate_mode": mode,
        "horizon_hours": int(args.horizon),
        "sigma_R0": estimate["sigma_R0"],
        "sigma_W": estimate["sigma_W"],
        **overall,
        **{f"per_origin_{key}": value for key, value in origin_quality.items()},
        **{f"per_day_{key}": value for key, value in day_quality.items()},
    }
    pd.DataFrame([summary]).to_csv(output / f"aggregate_summary_{suffix}.csv", index=False)

    save_plot(cumulative, daily, output / f"aggregate_uncertainty_{suffix}.png")
    print_summary(overall, cumulative, origin_quality, day_quality, estimate, mode)
    print()
    print(f"complete origins={overall['origins']}, days={overall['days']}, forecasts={overall['forecasts']}")
    print(f"estimate source={estimate['source']}")
    print(f"output={output}")


if __name__ == "__main__": main()
