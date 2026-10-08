from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
MPC_OUTPUT = ROOT / "files" / "output" / "wm_4" / "prediction" / "all_days_mpc"
DEFAULT_INPUT = MPC_OUTPUT / "mpc_forecasts.csv"
OUTPUT = MPC_OUTPUT / "sigma_from_mpc"


def _unit_process_variance(r_F, r_R, lead_hours, steps_per_hour):
    steps_per_hour, lead_hours = int(steps_per_hour), int(lead_hours)
    dt = 1.0 / (24.0 * steps_per_hour)
    transition = np.array([
        [1.0, dt],
        [dt * float(r_F), 1.0 + dt * float(r_R)],
    ], dtype=float)
    response = np.array([0.0, 1.0], dtype=float)
    total = 0.0
    for _ in range(lead_hours * steps_per_hour):
        total += dt * response[1] ** 2
        response = transition @ response
    return float(total)


def _prepare(frame, steps_per_hour):
    required = {
        "origin", "origin_day", "lead_hours", "I_observed", "I_mpc",
        "initial_variance", "process_variance", "diffusion_q", "r_F", "r_R",
    }
    missing = sorted(required.difference(frame.columns))
    if missing: raise ValueError(f"MPC forecast file is missing required columns: {missing}")

    data = frame.copy()
    data["origin"] = pd.to_datetime(data["origin"])
    data["origin_day"] = pd.to_datetime(data["origin_day"])
    data["forecast_error"] = data["I_observed"].to_numpy(dtype=float) - data["I_mpc"].to_numpy(dtype=float)

    q_old = data["diffusion_q"].to_numpy(dtype=float)
    process = data["process_variance"].to_numpy(dtype=float)
    unit = np.full(len(data), np.nan, dtype=float)
    valid = np.isfinite(q_old) & np.isfinite(process) & (q_old > 1e-12)
    unit[valid] = process[valid] / q_old[valid]

    missing_unit = ~np.isfinite(unit)
    if np.any(missing_unit):
        for idx in np.flatnonzero(missing_unit):
            row = data.iloc[idx]
            unit[idx] = _unit_process_variance(
                row["r_F"], row["r_R"], row["lead_hours"], steps_per_hour,
            )

    if not np.all(np.isfinite(unit)) or np.any(unit < 0.0):
        raise ValueError("Could not recover finite nonnegative unit process-variance responses from saved MPC data.")
    data["unit_process_variance"] = unit
    return data


def _lead_targets(frame):
    rows = []
    for lead, part in frame.groupby("lead_hours", sort=True):
        error = part["forecast_error"].to_numpy(dtype=float)
        empirical_bias = float(np.mean(error))
        empirical_variance = float(np.var(error, ddof=1)) if len(error) > 1 else 0.0
        empirical_rmse = float(np.sqrt(np.mean(error ** 2)))
        initial_variance = float(part["initial_variance"].mean())
        unit_process_variance = float(part["unit_process_variance"].mean())
        residual_variance_target = empirical_variance - initial_variance
        q_by_lead = max(
            0.0,
            residual_variance_target / unit_process_variance,
        ) if unit_process_variance > 0.0 else np.nan
        rows.append({
            "lead_hours": int(lead),
            "observations": int(len(part)),
            "empirical_bias": empirical_bias,
            "empirical_rmse": empirical_rmse,
            "empirical_variance": empirical_variance,
            "empirical_std": float(np.sqrt(max(empirical_variance, 0.0))),
            "mean_initial_variance": initial_variance,
            "mean_unit_process_variance": unit_process_variance,
            "process_variance_target": residual_variance_target,
            "q_by_lead": q_by_lead,
            "sigma_by_lead": float(np.sqrt(q_by_lead)) if np.isfinite(q_by_lead) else np.nan,
        })
    return pd.DataFrame(rows)


def fit_sigma(frame):
    lead = _lead_targets(frame)
    X = lead["mean_unit_process_variance"].to_numpy(dtype=float)
    y = lead["process_variance_target"].to_numpy(dtype=float)
    denominator = float(X @ X)
    q = max(0.0, float((X @ y) / denominator)) if denominator > 0.0 else 0.0
    return {
        "q": q,
        "sigma": float(np.sqrt(q)),
        "lead_targets": lead,
    }


def evaluate(frame, q):
    data = frame.copy()
    q = float(q)
    data["calibrated_process_variance"] = q * data["unit_process_variance"].to_numpy(dtype=float)
    data["calibrated_variance"] = np.maximum(
        data["initial_variance"].to_numpy(dtype=float)
        + data["calibrated_process_variance"].to_numpy(dtype=float),
        0.0,
    )
    data["calibrated_std"] = np.sqrt(data["calibrated_variance"])
    std = data["calibrated_std"].to_numpy(dtype=float)
    error = data["forecast_error"].to_numpy(dtype=float)
    data["calibrated_z"] = np.divide(
        error, std,
        out=np.full_like(error, np.nan),
        where=std > 1e-12,
    )
    data["within_1sigma_calibrated"] = np.abs(error) <= std
    data["within_95_calibrated"] = np.abs(error) <= 1.96 * std

    rows = []
    for lead, part in data.groupby("lead_hours", sort=True):
        error = part["forecast_error"].to_numpy(dtype=float)
        variance = part["calibrated_variance"].to_numpy(dtype=float)
        z = part["calibrated_z"].to_numpy(dtype=float)
        z = z[np.isfinite(z)]
        bias = float(np.mean(error))
        rmse = float(np.sqrt(np.mean(error ** 2)))
        empirical_std = float(np.std(error, ddof=1)) if len(error) > 1 else 0.0
        predicted_rms_std = float(np.sqrt(np.mean(variance)))
        z_mean = float(np.mean(z)) if z.size else np.nan
        z_std = float(np.std(z, ddof=1)) if z.size > 1 else np.nan
        if z.size > 2 and z_std > 0.0:
            centered = (z - z_mean) / z_std
            skew = float(np.mean(centered ** 3))
            excess_kurtosis = float(np.mean(centered ** 4) - 3.0)
        else:
            skew, excess_kurtosis = np.nan, np.nan

        rows.append({
            "lead_hours": int(lead),
            "observations": int(len(part)),
            "bias": bias,
            "rmse": rmse,
            "empirical_std": empirical_std,
            "predicted_rms_std": predicted_rms_std,
            "std_to_rmse_ratio": predicted_rms_std / rmse if rmse > 0.0 else np.nan,
            "coverage_68_pct": 100.0 * float(part["within_1sigma_calibrated"].mean()),
            "coverage_95_pct": 100.0 * float(part["within_95_calibrated"].mean()),
            "z_mean": z_mean,
            "z_std": z_std,
            "z_skew": skew,
            "z_excess_kurtosis": excess_kurtosis,
        })
    return data, pd.DataFrame(rows)


def _split_days(frame, calibration_fraction):
    days = np.array(sorted(pd.unique(frame["origin_day"])))
    if len(days) < 2: raise ValueError("At least two origin days are required for chronological calibration/validation.")
    fraction = float(calibration_fraction)
    if not 0.0 < fraction < 1.0: raise ValueError("calibration_fraction must lie strictly between 0 and 1.")
    split = min(max(1, int(np.floor(len(days) * fraction))), len(days) - 1)
    calibration_days = set(days[:split])
    mask = frame["origin_day"].isin(calibration_days)
    return frame.loc[mask].copy(), frame.loc[~mask].copy(), days[split]


def save_plot(summary, path, title):
    lead = summary["lead_hours"].to_numpy(dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(lead, summary["rmse"], marker="o", label="Empirical MPC RMSE")
    axes[0].plot(lead, summary["predicted_rms_std"], marker="o", label="MPC-calibrated stochastic RMS std")
    axes[0].set_ylabel("Error / predicted standard deviation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(lead, summary["coverage_68_pct"], marker="o", label="Observed ±1σ coverage")
    axes[1].plot(lead, summary["coverage_95_pct"], marker="o", label="Observed 95% coverage")
    axes[1].axhline(68.27, linestyle="--", linewidth=0.8, label="Gaussian 68.27%")
    axes[1].axhline(95.0, linestyle="--", linewidth=0.8, label="Gaussian 95%")
    axes[1].set_ylabel("Coverage (%)")
    axes[1].set_xlabel("Lead time (hours)")
    axes[1].set_xticks(lead)
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(label, fit, summary):
    print(f"{label}: q={fit['q']:.10f}, sigma={fit['sigma']:.10f}")
    print(f"{'lead':>7}{'RMSE':>12}{'pred std':>12}{'ratio':>10}{'z mean':>10}{'z std':>10}{'cov68':>10}{'cov95':>10}")
    for _, row in summary.iterrows():
        print(
            f"{int(row['lead_hours']):>7d}"
            f"{row['rmse']:>12.8f}"
            f"{row['predicted_rms_std']:>12.8f}"
            f"{row['std_to_rmse_ratio']:>10.3f}"
            f"{row['z_mean']:>10.3f}"
            f"{row['z_std']:>10.3f}"
            f"{row['coverage_68_pct']:>9.2f}%"
            f"{row['coverage_95_pct']:>9.2f}%"
        )


def main():
    parser = ArgumentParser(description="Estimate the WM4 constant Wiener diffusion sigma directly from saved receding-horizon MPC forecast errors. No WM4 refitting is performed.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Saved mpc_forecasts.csv from predict_all_days_mpc.")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--steps-per-hour", type=int, default=60, help="Only used as a fallback if saved process_variance/diffusion_q cannot recover the unit variance response.")
    parser.add_argument("--calibration-fraction", type=float, default=0.7, help="Chronological fraction of days used to fit sigma for the holdout diagnostic.")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists(): raise FileNotFoundError(f"Saved MPC forecast file not found: {path}")
    frame = _prepare(pd.read_csv(path), args.steps_per_hour)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    global_fit = fit_sigma(frame)
    global_rows, global_summary = evaluate(frame, global_fit["q"])
    global_fit["lead_targets"].to_csv(output / "sigma_by_lead_all.csv", index=False)
    global_summary.to_csv(output / "calibration_all.csv", index=False)
    global_rows.to_csv(output / "mpc_forecasts_sigma_calibrated_all.csv", index=False)
    save_plot(global_summary, output / "calibration_all.png", "WM4 MPC variance with sigma estimated from all MPC errors")

    calibration, validation, validation_start = _split_days(frame, args.calibration_fraction)
    holdout_fit = fit_sigma(calibration)
    validation_rows, validation_summary = evaluate(validation, holdout_fit["q"])
    holdout_fit["lead_targets"].to_csv(output / "sigma_by_lead_calibration.csv", index=False)
    validation_summary.to_csv(output / "calibration_holdout.csv", index=False)
    validation_rows.to_csv(output / "mpc_forecasts_sigma_calibrated_holdout.csv", index=False)
    save_plot(validation_summary, output / "calibration_holdout.png", "WM4 MPC variance: sigma fitted on earlier MPC days, evaluated on holdout")

    pd.DataFrame([
        {
            "mode": "all",
            "q": global_fit["q"],
            "sigma": global_fit["sigma"],
            "calibration_days": int(frame["origin_day"].nunique()),
            "validation_days": int(frame["origin_day"].nunique()),
            "validation_start": frame["origin_day"].min(),
        },
        {
            "mode": "chronological_holdout",
            "q": holdout_fit["q"],
            "sigma": holdout_fit["sigma"],
            "calibration_days": int(calibration["origin_day"].nunique()),
            "validation_days": int(validation["origin_day"].nunique()),
            "validation_start": validation_start,
        },
    ]).to_csv(output / "sigma_estimates.csv", index=False)

    print(f"input={path}")
    print(f"rows={len(frame)}, days={frame['origin_day'].nunique()}")
    print()
    print_summary("all MPC data", global_fit, global_summary)
    print()
    print(
        f"chronological holdout: calibration days={calibration['origin_day'].nunique()}, "
        f"validation days={validation['origin_day'].nunique()}, validation starts={pd.Timestamp(validation_start):%Y-%m-%d}"
    )
    print_summary("holdout fit", holdout_fit, validation_summary)
    print()
    print("lead-specific sigma diagnostic (all MPC data):")
    print(global_fit["lead_targets"][["lead_hours", "sigma_by_lead"]].to_string(index=False))
    print(f"output={output}")


if __name__ == "__main__": main()
