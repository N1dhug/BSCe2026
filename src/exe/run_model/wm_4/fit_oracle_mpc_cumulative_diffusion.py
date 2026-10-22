from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_daily import DAY_HOURS, day_start
from src.exe.run_model.wm_4.fit_oracle_mpc_diffusion import OUTPUT as ORACLE_OUTPUT, Z_95


INPUT = ORACLE_OUTPUT / "oracle_24h_by_origin_and_lead.csv"
OUTPUT = ORACLE_OUTPUT / "cumulative_24h"
Z_68 = 1.0
NOMINAL_68_PCT = 68.26894921370859  # Two-sided Gaussian +/-1 sigma
NOMINAL_95_PCT = 95.0
REQUIRED = {"origin", "lead_hours", "I_observed", "I_oracle", "oracle_error", "unit_process_variance"}


def load_predictions(path, start_day=None, end_day=None):
    """Read previously fitted oracle trajectories and fixed WM4 process kernels; fit nothing here."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing cached oracle trajectories: {path}\n"
            "First run: python -m src.exe.run_model.wm_4.fit_oracle_mpc_diffusion"
        )
    frame = pd.read_csv(path, parse_dates=["origin"])
    missing = sorted(REQUIRED.difference(frame.columns))
    if missing: raise ValueError("Oracle prediction file is missing: " + ", ".join(missing))
    if frame["origin"].isna().any(): raise ValueError("Oracle origin timestamps must be valid.")
    if start_day is not None: frame = frame.loc[frame["origin"] >= day_start(start_day)]
    if end_day is not None: frame = frame.loc[frame["origin"] < day_start(end_day) + pd.Timedelta(days=1)]
    if frame.empty: raise ValueError("No oracle trajectories for the requested period.")
    if frame.duplicated(["origin", "lead_hours"]).any(): raise ValueError("Duplicate oracle origin / lead-hour rows.")
    if not np.isfinite(frame[["I_observed", "I_oracle", "oracle_error", "unit_process_variance"]].to_numpy(dtype=float)).all():
        raise ValueError("Observed/oracle values, residuals and WM4 unit-variance kernels must be finite.")
    if (frame["unit_process_variance"] < -1e-12).any(): raise ValueError("WM4 unit-process variance cannot be negative.")
    if not np.allclose(frame["oracle_error"], frame["I_observed"] - frame["I_oracle"], atol=1e-8, rtol=1e-6):
        raise ValueError("Saved oracle_error does not equal I_observed - I_oracle.")
    frame = frame.copy()
    frame["unit_process_variance"] = frame["unit_process_variance"].clip(lower=0.0)
    return frame.sort_values(["origin", "lead_hours"]).reset_index(drop=True)


def fit_nonnegative_variances(target, process_cumulative):
    """Exact nonnegative two-parameter least-squares solution for q_0, q_p."""
    y, b = np.asarray(target, dtype=float), np.asarray(process_cumulative, dtype=float)
    h = np.arange(1, y.size + 1, dtype=float)
    if y.size != DAY_HOURS or b.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(b).all():
        raise ValueError("Expected 24 finite cumulative error and kernel samples.")
    if np.any(b < 0.0) or np.all(b == 0.0): raise ValueError("WM4 process kernel is zero or invalid.")
    A = np.column_stack((h, b))
    if np.linalg.svd(A / np.linalg.norm(A, axis=0), compute_uv=False)[-1] < 1e-10:
        raise ValueError("Observation and process variances cannot be separately identified on this kernel.")
    candidates = [np.zeros(2)]
    for j in (0, 1):
        v = np.zeros(2)
        v[j] = max(0.0, float(A[:, j] @ y) / float(A[:, j] @ A[:, j]))
        candidates.append(v)
    interior = np.linalg.lstsq(A, y, rcond=None)[0]
    if np.all(interior >= -1e-12): candidates.append(np.maximum(interior, 0.0))
    q = min(candidates, key=lambda v: float(np.sum((A @ v - y) ** 2)))
    return float(q[0]), float(q[1])


def fit_cumulative_origin(group):
    """Fit q_0=sigma_0**2 and q_p=sigma_p**2 for each oracle 24-hour trajectory.

    Observation errors epsilon_h iid N(0,1) enter I_h=O_h+R_h+sigma_0*epsilon_h.
    Independent Wiener noise enters dR=drift*dt+sigma_p*dW.
    With known oracle x0, Var(I_h)=q_0+q_p*K_h and expected cumulative
    squared error at horizon H is H*q_0+q_p*sum(K[:H]).
    Match realized cumulative squared errors with nonnegative two-variable QP.
    This is an in-sample, one-path moment proxy, not empirical ensemble variance.
    """
    group = group.sort_values("lead_hours").copy()
    hours = group["lead_hours"].to_numpy(dtype=int)
    if len(hours) != DAY_HOURS or not np.array_equal(hours, np.arange(1, DAY_HOURS + 1)):
        raise ValueError(f"Origin {group['origin'].iloc[0]} must have exactly the 24 leads 1..24.")

    error = group["oracle_error"].to_numpy(dtype=float)
    K = group["unit_process_variance"].to_numpy(dtype=float)
    if not np.isfinite(error).all() or not np.isfinite(K).all() or np.any(K < 0.0):
        raise ValueError("Oracle residuals / WM4 variance kernels must be finite and nonnegative.")
    observed_cumulative = np.cumsum(error ** 2)
    unit_cumulative = np.cumsum(K)
    denominator = float(unit_cumulative @ unit_cumulative)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("The cumulative WM4 process-variance kernel is zero or invalid.")

    q_0, q_p = fit_nonnegative_variances(observed_cumulative, unit_cumulative)
    sigma_0, sigma_p = float(np.sqrt(q_0)), float(np.sqrt(q_p))
    observation_variance = np.full(DAY_HOURS, q_0)
    process_variance = q_p * K
    total_variance = observation_variance + process_variance
    process_std = np.sqrt(np.maximum(process_variance, 0.0))
    total_std = np.sqrt(np.maximum(total_variance, 0.0))
    fitted_cumulative_process = q_p * unit_cumulative
    fitted_cumulative_observation = q_0 * hours
    fitted_cumulative = fitted_cumulative_observation + fitted_cumulative_process
    group["error_squared"] = error ** 2
    group["cumulative_error_squared"] = observed_cumulative
    group["cumulative_unit_variance"] = unit_cumulative
    group["fitted_cumulative_process_variance"] = fitted_cumulative_process
    group["fitted_cumulative_observation_variance"] = fitted_cumulative_observation
    group["fitted_cumulative_total_variance"] = fitted_cumulative
    group["cumulative_variance_residual"] = fitted_cumulative - observed_cumulative
    group["cumulative_error_mse"] = observed_cumulative / hours
    group["cumulative_error_rmse"] = np.sqrt(group["cumulative_error_mse"])
    group["cumulative_model_mean_variance"] = fitted_cumulative / hours
    group["cumulative_model_rms_std"] = np.sqrt(group["cumulative_model_mean_variance"])
    group["observation_variance"] = observation_variance
    group["process_variance"] = process_variance
    group["total_variance"] = total_variance
    group["process_std"] = process_std
    group["total_std"] = total_std
    group["lower_68"] = group["I_oracle"] - Z_68 * total_std
    group["upper_68"] = group["I_oracle"] + Z_68 * total_std
    group["covered_68"] = np.abs(error) <= Z_68 * total_std
    group["lower_95"] = group["I_oracle"] - Z_95 * total_std
    group["upper_95"] = group["I_oracle"] + Z_95 * total_std
    group["covered_95"] = np.abs(error) <= Z_95 * total_std
    group["q_0"], group["q_p"] = q_0, q_p
    group["sigma_0"], group["sigma_p"] = sigma_0, sigma_p
    group["q"], group["sigma"] = q_p, sigma_p  # Backward-compatible process-only aliases

    fit = {
        "origin": group["origin"].iloc[0],
        "origin_day": group["origin"].iloc[0].normalize(),
        "q_0": q_0, "q_p": q_p, "sigma_0": sigma_0, "sigma_p": sigma_p,
        "q": q_p, "sigma": sigma_p,
        "horizon_hours": DAY_HOURS,
        "cumulative_curve_mse": float(np.mean((fitted_cumulative - observed_cumulative) ** 2)),
        "observed_24h_error_squared_sum": float(observed_cumulative[-1]),
        "fitted_24h_variance_sum": float(fitted_cumulative[-1]),
        "observed_24h_rmse": float(np.sqrt(observed_cumulative[-1] / DAY_HOURS)),
        "model_24h_rms_std": float(np.sqrt(fitted_cumulative[-1] / DAY_HOURS)),
        "mean_oracle_residual": float(np.mean(error)),
        "total_68_coverage_pct": 100.0 * float(np.mean(group["covered_68"])),
        "total_95_coverage_pct": 100.0 * float(np.mean(group["covered_95"])),
        "process_68_coverage_pct": 100.0 * float(np.mean(group["covered_68"])),  # Legacy name: total coverage
        "process_95_coverage_pct": 100.0 * float(np.mean(group["covered_95"])),  # Legacy name: total coverage
    }
    for key in ("r_F", "r_U", "r_R", "F0_oracle", "R0_oracle"):
        if key in group.columns: fit[key] = float(group[key].iloc[0])
    return fit, group


def fit_all_origins(forecasts):
    """Fit an independent diffusion for each hourly MPC origin without refitting model parameters."""
    fits, trajectories = [], []
    for _, group in forecasts.groupby("origin", sort=True):
        fit, result = fit_cumulative_origin(group)
        fits.append(fit); trajectories.append(result)
    if not fits: raise ValueError("No complete oracle MPC origins were provided.")
    origins = pd.DataFrame(fits)
    hourly = pd.concat(trajectories, ignore_index=True)
    return origins, hourly


def horizon_summary(hourly):
    """Aggregate empirical residual variance and nominal interval coverage at each 1..24h lead.

    Compare empirical residual variance across origins to the sum of mean
    observation and Wiener process variance. Intervals use total variance.
    Each origin fits both sigmas on these same errors (in-sample diagnostic).
    """
    rows = []
    for hour, group in hourly.groupby("lead_hours", sort=True):
        residual = group["oracle_error"].to_numpy(dtype=float)
        n = residual.size
        observed_cum = group["cumulative_error_squared"].to_numpy(dtype=float)
        model_cum = group["fitted_cumulative_total_variance"].to_numpy(dtype=float)
        covered_68 = group["covered_68"].to_numpy(dtype=bool)
        covered_95 = group["covered_95"].to_numpy(dtype=bool)
        empirical_var = float(np.var(residual, ddof=1)) if n > 1 else np.nan
        observation_var = float(np.mean(group["observation_variance"]))
        process_var = float(np.mean(group["process_variance"]))
        predicted_var = float(np.mean(group["total_variance"]))
        coverage_68 = 100.0 * float(np.mean(covered_68))
        coverage_95 = 100.0 * float(np.mean(covered_95))
        rows.append({
            "lead_hours": int(hour), "origins": int(n),
            "hourly_oracle_bias": float(np.mean(residual)),
            "hourly_oracle_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "hourly_oracle_mse": float(np.mean(residual ** 2)),
            "empirical_oracle_variance": empirical_var,
            "empirical_oracle_std": float(np.sqrt(empirical_var)),
            "mean_predicted_observation_variance": observation_var,
            "mean_predicted_process_variance": process_var,
            "mean_predicted_total_variance": predicted_var,
            "mean_predicted_process_std": float(np.sqrt(process_var)),
            "mean_predicted_total_std": float(np.sqrt(predicted_var)),
            "predicted_to_empirical_variance_ratio": (
                predicted_var / empirical_var if empirical_var > 0.0 else np.nan
            ),
            "mean_cumulative_squared_error": float(np.mean(observed_cum)),
            "mean_cumulative_model_variance": float(np.mean(model_cum)),
            "cumulative_error_rmse": float(np.sqrt(np.mean(observed_cum) / hour)),
            "cumulative_model_rms_std": float(np.sqrt(np.mean(model_cum) / hour)),
            "mean_cumulative_difference": float(np.mean(model_cum - observed_cum)),
            "covered_68_origins": int(np.count_nonzero(covered_68)),
            "covered_95_origins": int(np.count_nonzero(covered_95)),
            "nominal_68_coverage_pct": NOMINAL_68_PCT,
            "nominal_95_coverage_pct": NOMINAL_95_PCT,
            "total_68_coverage_pct": coverage_68,
            "total_95_coverage_pct": coverage_95,
            "process_68_coverage_pct": coverage_68,
            "process_95_coverage_pct": coverage_95,
            "coverage_68_error_pp": coverage_68 - NOMINAL_68_PCT,
            "coverage_95_error_pp": coverage_95 - NOMINAL_95_PCT,
            "coverage_68_ratio": coverage_68 / NOMINAL_68_PCT,
            "coverage_95_ratio": coverage_95 / NOMINAL_95_PCT,
        })
    return pd.DataFrame(rows).sort_values("lead_hours").reset_index(drop=True)


def save_calibration_plot(summary, output):
    """Per-lead variance, 68%/95% interval coverage and 1:1 coverage ratios."""
    h = summary["lead_hours"].to_numpy(dtype=int)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(h, summary["empirical_oracle_variance"], marker="o",
        label="Empirical oracle-residual variance across MPC origins")
    axes[0].plot(h, summary["mean_predicted_total_variance"], marker="o",
        label="Mean WM4 fitted total variance")
    axes[0].plot(h, summary["mean_predicted_observation_variance"], linestyle=":",
        label="Observation variance sigma_0^2")
    axes[0].plot(h, summary["mean_predicted_process_variance"], linestyle="--",
        label="Propagated process variance sigma_p^2 K_h")
    axes[0].set_ylabel("Marginal variance")
    axes[0].legend(ncol=2); axes[0].grid(True, alpha=0.25)

    axes[1].plot(h, summary["total_68_coverage_pct"], marker="o", label="Observed +/-1 total sigma coverage")
    axes[1].plot(h, summary["total_95_coverage_pct"], marker="o", label="Observed 95% total sigma coverage")
    axes[1].axhline(NOMINAL_68_PCT, linestyle="--", label="Gaussian 68.27% nominal")
    axes[1].axhline(NOMINAL_95_PCT, linestyle="--", label="95% nominal")
    axes[1].set_ylabel("Empirical coverage (%)")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].legend(ncol=2); axes[1].grid(True, alpha=0.25)

    axes[2].plot(h, summary["coverage_68_ratio"], marker="o", label="68.27% empirical / nominal")
    axes[2].plot(h, summary["coverage_95_ratio"], marker="o", label="95% empirical / nominal")
    axes[2].axhline(1.0, linestyle="--", label="Perfect calibration (1:1)")
    axes[2].set_ylabel("Coverage calibration ratio")
    axes[2].set_xlabel("Hours since hourly MPC origin")
    axes[2].legend(); axes[2].grid(True, alpha=0.25)
    axes[2].set_xticks(np.arange(1, DAY_HOURS + 1, 2))
    axes[2].set_xlim(1, DAY_HOURS)

    fig.suptitle("WM4 observation + Wiener process variance and 68% / 95% coverage (in-sample)")
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def save_plot(origins, summary, output):
    fig, axes = plt.subplots(3, 1, figsize=(15, 13))
    axes[0].scatter(origins["origin"], origins["sigma_0"], s=7, alpha=0.75, label="Observation sigma_0")
    axes[0].scatter(origins["origin"], origins["sigma_p"], s=7, alpha=0.75, label="Process sigma_p")
    axes[0].axhline(float(np.mean(origins["sigma_0"])), linestyle=":", label="Mean sigma_0")
    axes[0].axhline(float(np.mean(origins["sigma_p"])), linestyle="--", label="Mean sigma_p")
    axes[0].set_ylabel("Uncertainty sigma")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)

    h = summary["lead_hours"].to_numpy(dtype=int)
    axes[1].plot(h, summary["mean_cumulative_squared_error"], marker="o", label="Observed cumulative squared oracle error")
    axes[1].plot(h, summary["mean_cumulative_model_variance"], marker="o", label="Fitted cumulative observation + process variance")
    axes[1].set_ylabel("Cumulative squared error / variance")
    axes[1].legend(); axes[1].grid(True, alpha=0.25)

    axes[2].plot(h, summary["cumulative_error_rmse"], marker="o", label="Cumulative oracle RMSE")
    axes[2].plot(h, summary["cumulative_model_rms_std"], marker="o", label="Cumulative model total RMS std")
    axes[2].plot(h, summary["hourly_oracle_rmse"], linestyle=":", label="Exact-hour oracle RMSE")
    axes[2].set_xlabel("Hours since MPC origin")
    axes[2].set_ylabel("Normalized input")
    axes[2].legend(); axes[2].grid(True, alpha=0.25)
    for ax in axes[1:]: ax.set_xticks(np.arange(1, DAY_HOURS + 1, 2))

    fig.suptitle("WM4 observation + process uncertainty fitted to cumulative 24h squared oracle errors")
    fig.tight_layout(); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    parser = ArgumentParser(description="Fit observation sigma_0 and process sigma_p at each MPC origin to cumulative 24h squared oracle errors, using saved deterministic parameters and oracle states.")
    parser.add_argument("--input", default=str(INPUT), help="Saved oracle_24h_by_origin_and_lead.csv.")
    parser.add_argument("--start-day", default=None)
    parser.add_argument("--end-day", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    if args.start_day and args.end_day and day_start(args.end_day) < day_start(args.start_day):
        parser.error("--end-day must be on/after --start-day.")

    forecasts = load_predictions(args.input, args.start_day, args.end_day)
    origins, hourly = fit_all_origins(forecasts)
    horizons = horizon_summary(hourly)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)

    origins.to_csv(output / "cumulative_diffusion_by_mpc_origin.csv", index=False)
    hourly.to_csv(output / "cumulative_diffusion_by_origin_and_lead.csv", index=False)
    horizons.to_csv(output / "cumulative_diffusion_by_horizon.csv", index=False)
    horizons.to_csv(output / "variance_coverage_by_horizon.csv", index=False)
    save_plot(origins, horizons, output / "cumulative_24h_diffusion.png")
    save_calibration_plot(horizons, output / "variance_coverage_by_horizon.png")

    print(f"origins={len(origins)}, fitted horizon={DAY_HOURS} hours")
    print(f"observation sigma_0 mean={origins['sigma_0'].mean():.9f}, median={origins['sigma_0'].median():.9f}")
    print(f"process sigma_p mean={origins['sigma_p'].mean():.9f}, median={origins['sigma_p'].median():.9f}")
    print(f"mean final observed RMSE={origins['observed_24h_rmse'].mean():.9f}")
    print(f"mean final predicted RMS std={origins['model_24h_rms_std'].mean():.9f}")
    print("Nonnegative q_0=sigma_0^2 and q_p=sigma_p^2 jointly fitted per origin to 1..24h cumulative squared errors.")
    print("No WM4 drift coefficients, robust baseline or oracle initial states are refitted.")
    print("Cumulative squared errors are one-path proxies for expected variances, NOT empirical ensemble variances.")
    print("Empirical variance and Gaussian 68.27% / 95% interval coverage by lead:")
    print(f"{'lead':>6} {'n':>6} {'var(obs)':>12} {'var(total)':>12} {'cov68':>10} {'cov95':>10} {'ratio68':>10} {'ratio95':>10}")
    for row in horizons.itertuples(index=False):
        print(f"{row.lead_hours:6d} {row.origins:6d} {row.empirical_oracle_variance:12.7f} "
            f"{row.mean_predicted_total_variance:12.7f} {row.total_68_coverage_pct:9.2f}% "
            f"{row.total_95_coverage_pct:9.2f}% {row.coverage_68_ratio:10.3f} "
            f"{row.coverage_95_ratio:10.3f}")
    print("Coverage ratios are observed / nominal: 1.0 indicates ideal calibration.")
    print("Intervals use total_std=sqrt(sigma_0^2 + sigma_p^2*K_h), fitted on the SAME origin residuals.")
    print("Overlapping hourly origins are dependent; these coverages are NOT independent holdout estimates.")
    print(f"output={output}")


if __name__ == "__main__": main()
