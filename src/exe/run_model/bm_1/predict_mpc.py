"""BM1 MPC: fit five shifted 24h windows and forecast up to eight hours.

The pooled logistic theta and each historical X0 are refitted at EVERY
hourly origin, using five previous matching weekday/hour windows.
Historical demand residuals determine the nonnegative sigma_x² estimate.
Only the latest observed consumption I(origin) is assimilated. Future I is
never used by fitting, diffusion calibration or state initialization.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.exe.run_model.bm_1.predict_daily import held_signal
from src.model.bm_1 import bm_1
from src.params.bm_1 import lsq_params, historical_diffusion_params
from src.solver.finite_volume import finite_volume_1d_cached
from src.tools.baseline import statistical_output
from src.tools.forecast_metrics import predictive_intervals, weighted_quantiles
from src.validation.exe.run_model.bm_1.predict_mpc import bm_1_mpc_validation as checks


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "bm_1" / "prediction" / "mpc"


def posterior_mass(states, x_grid, prediction, observed, observation_sd, bandwidth):
    """Smooth five empirical X0 points, then assimilate I(origin) via Bayes rule.

    The KDE bandwidth and observation SD are explicit approximations, not
    additional state-process diffusion. Posterior normalization is exact.
    """
    states = np.asarray(states, dtype=float)
    smooth = max(float(bandwidth), float(np.std(states)) / 2.0)
    prior = np.exp(-0.5 * ((x_grid[:, None] - states[None, :]) / smooth) ** 2).mean(axis=1)
    prior = checks.posterior(prior, len(x_grid))
    z = (prediction - observed) / observation_sd
    log_prob = np.log(np.maximum(prior, 1e-300)) - 0.5 * z * z
    weights = np.exp(log_prob - np.max(log_prob))
    return checks.posterior(weights, len(x_grid))


def predict_origin(frame, origin, output_model, *, X0=0.5, weeks=5, horizon=8, x_steps=101,
    max_iterations=3, max_nfev=120, state_maxiter=60, substeps=2,
    fp_max_step=0.25, prior_bandwidth=0.025, C=None, Xi=None, min_capacity=0.05,
    fitted=None, joint_max_nfev=0, joint_max_seconds=20.0, joint_gtol=1e-5):
    """Return 1..horizon predictive demand distributions for one MPC origin.

    frame must be normalized to BM1's [0,1] I and U before calling.
    output_model must be trained using only observations preceding the origin.
    """
    origin = checks.origin(origin)
    checks.options(horizon, x_steps, fp_max_step, prior_bandwidth, substeps, weeks)
    if not np.isfinite(min_capacity) or not 0.001 <= min_capacity < lsq_params.BOUNDS["C"][1]:
        raise ValueError("min_capacity must be finite and in [0.001, 100).")
    if C is not None and C < min_capacity:
        raise ValueError(f"Fixed BM1 C={C} is smaller than min_capacity={min_capacity}; lower --min-capacity explicitly.")
    training_origins = [origin - pd.Timedelta(weeks=k) for k in range(weeks, 0, -1)]
    training = [checks.window(frame, t, output_model, hours=24) for t in training_origins]
    model = bm_1(held_signal(training[0]["B"]), lambda t: 0.5, (0.0, 24.0), 25)
    if fitted is None:
        fitted = lsq_params(model).fit_many(training, X0, C=C, Xi=Xi, max_iterations=max_iterations,
            max_nfev=max_nfev, state_maxiter=state_maxiter, substeps=substeps,
            fit_bounds={"C": (float(min_capacity), lsq_params.BOUNDS["C"][1])},
            joint_max_nfev=joint_max_nfev,joint_max_seconds=joint_max_seconds,joint_gtol=joint_gtol)
    if fitted["warning"]:
        print(f"[BM1 MPC {origin:%Y-%m-%d %H:%M}] WARNING: {fitted['warning']}", flush=True)
    print(f"[BM1 MPC {origin:%Y-%m-%d %H:%M}] Historical diffusion calibration "
        f"(C={fitted['C']:.5f}, Xi={fitted['Xi']:.5f})", flush=True)
    diffusion = historical_diffusion_params().fit_many(training, fitted, substeps=substeps)
    sigma = float(diffusion["sigma_x"])
    print(f"[BM1 MPC {origin:%Y-%m-%d %H:%M}] Diffusion fitted sigma_x={sigma:.5f}; "
        f"state posterior / Fokker-Planck forecast starting", flush=True)

    future = checks.window(frame, origin, output_model, hours=horizon + 1, observations=False)
    observed = checks.observation(frame, origin)
    x_grid = (np.arange(x_steps, dtype=float) + 0.5) / x_steps
    f, g = fitted["f"], fitted["g"]
    delta0 = np.tanh(0.5 * fitted["k"] * (f(x_grid) + g(future["U"][0])))
    output0 = future["B"][0] + fitted["Xi"] * delta0 * np.where(delta0 >= 0.0, 1.0 - future["B"][0], future["B"][0])
    masses0 = posterior_mass(fitted["initial_states"], x_grid, output0, observed,
        diffusion["observation_sd"], prior_bandwidth)

    supports, masses = [], []
    if sigma > 0.0:
        price = held_signal(future["U"])
        model = bm_1(held_signal(future["B"]), lambda t: 0.5, (0.0, float(horizon)), horizon + 1)
        model.fokker_planc_reference = finite_volume_1d_cached()
        parameters = [price, f, g, fitted["C"], fitted["k"], fitted["Xi"]]
        t, x, density, cell_mass, ex, vx, u, b, expected_d, diagnostics = model.fokker_planc_reference_solver(
            masses0, parameters, sigma, t_eval=np.arange(horizon + 1, dtype=float), max_step=fp_max_step,
        )
        for lead in range(1, horizon + 1):
            delta = np.tanh(0.5 * fitted["k"] * (f(x) + g(future["U"][lead])))
            D = future["B"][lead] + fitted["Xi"] * delta * np.where(delta >= 0, 1.0 - future["B"][lead], future["B"][lead])
            supports.append(D)
            masses.append(cell_mass[lead])
    else:
        trajectories = []
        for x0 in x_grid:
            _, demand = lsq_params._predict(future, float(x0), fitted, substeps)
            trajectories.append(demand)
        predictions = np.asarray(trajectories)
        for lead in range(1, horizon + 1):
            supports.append(predictions[:, lead])
            masses.append(masses0)

    intervals = predictive_intervals(supports, masses)
    rows = []
    for lead, (D, weights) in enumerate(zip(supports, masses), start=1):
        rows.append({
            "origin": origin, "origin_day": origin.normalize(), "origin_hour": origin.hour,
            "target_time": origin + pd.Timedelta(hours=lead), "lead_hours": lead,
            "I_BM1": float(np.dot(D, weights)),
            "lower_68": float(intervals[68][0][lead - 1]), "upper_68": float(intervals[68][1][lead - 1]),
            "lower_95": float(intervals[95][0][lead - 1]), "upper_95": float(intervals[95][1][lead - 1]),
            "sigma_x": sigma, "diffusion_q": float(diffusion["q"]),
            "observation_sd": float(diffusion["observation_sd"]),
            "historical_fit_rmse": float(diffusion["historical_rmse"]),
            "historical_diffusion_variance_rmse": float(diffusion["variance_rmse"]),
            "prior_state_mean": float(np.mean(fitted["initial_states"])),
            "posterior_state_mean": float(np.dot(x_grid, masses0)),
            "posterior_state_std": float(np.sqrt(np.dot(np.square(x_grid - np.dot(x_grid, masses0)), masses0))),
            "pooled_training_mse": float(fitted["mse"]), "fit_iterations": int(fitted["iterations"]),
            "fit_converged": bool(fitted["converged"]), "fit_stop_reason": fitted["stop_reason"],
            "fit_message": fitted["message"], "theta_nfev": int(fitted["theta_nfev"]),
        })
    return rows, {"fit": fitted, "diffusion": diffusion, "posterior": masses0,
        "training_origins": training_origins, "origin_observed_I": observed}


def predict_day(frame, day, *, X0=0.5, horizon=8, x_steps=101, max_iterations=3,
    max_nfev=120, state_maxiter=60, substeps=2, fp_max_step=0.25, prior_bandwidth=0.025,
    C=None, Xi=None, skip_errors=False, min_capacity=0.05):
    """Apply the same 24h five-week historical fitting at all 24 hourly origins."""
    day = checks.origin(pd.Timestamp(day).normalize())
    frame = checks.frame(frame)
    baseline = statistical_output().fit(frame, day)
    forecasts, origins, skipped = [], [], []
    for hour in range(24):
        origin = day + pd.Timedelta(hours=hour)
        try:
            predicted, diagnostics = predict_origin(frame, origin, baseline, X0=X0,
                horizon=horizon, x_steps=x_steps, max_iterations=max_iterations,
                max_nfev=max_nfev, state_maxiter=state_maxiter, substeps=substeps,
                fp_max_step=fp_max_step, prior_bandwidth=prior_bandwidth, C=C, Xi=Xi,
                min_capacity=min_capacity)
            forecasts.extend(predicted)
            origins.append({"origin": origin, **diagnostics["diffusion"],
                "fit_converged": diagnostics["fit"]["converged"],
                "fit_stop_reason": diagnostics["fit"]["stop_reason"],
                "fit_message": diagnostics["fit"]["message"],
                "theta_nfev": diagnostics["fit"]["theta_nfev"]})
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            if not skip_errors: raise
            skipped.append((origin, str(exc)))
    return pd.DataFrame(forecasts), pd.DataFrame(origins), skipped

def main():
    """Run standalone BM1 MPC, or evaluate BM1 and WM4 on one identical day."""
    from argparse import ArgumentParser
    from src.exe.run_model.wm_4.fit_daily import aligned_wm4_data
    from src.tools.forecast_metrics import forecast_metrics
    from src.validation.exe.run_model.compare_model_predictions import model_comparison_validation as scale

    parser = ArgumentParser(description="BM1 24-hour-window rolling MPC: estimate sigma_x historically and predict 1-8 hours each hour.")
    parser.add_argument("--day", default="2022-08-22")
    parser.add_argument("--tag", default="MED")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--X0", type=float, default=0.5)
    parser.add_argument("--x-steps", type=int, default=101)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--state-maxiter", type=int, default=60)
    parser.add_argument("--fp-max-step", type=float, default=0.25)
    parser.add_argument("--prior-bandwidth", type=float, default=0.025)
    parser.add_argument("--min-capacity", type=float, default=0.05,
        help="Minimum BM1 fitted capacity C; prevents extremely expensive stiff RK4 candidate solves (default 0.05).")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--compare-wm4", action="store_true", help="Compare BM1 against WM4 MPC using identical 1-8h leads on this single day.")
    parser.add_argument("--wm4-forecasts", type=Path, help="Optional saved WM4 mpc_forecasts.csv with all requested leads.")
    parser.add_argument("--steps-per-hour", type=int, default=2, help="WM4 numerical steps per hour, when comparing.")
    parser.add_argument("--wm4-max-state-iter", type=int, default=12, help="WM4 alternating state-fit iterations, when comparing.")
    parser.add_argument("--wm4-state-tol", type=float, default=1e-8)
    parser.add_argument("--no-resume", action="store_true", help="Refit comparison even if a matching daily checkpoint exists.")
    parser.add_argument("--output", type=Path, help="Output root; defaults to the BM1 or comparison output directory.")
    args = parser.parse_args()
    day = checks.origin(pd.Timestamp(args.day).normalize())

    if args.compare_wm4:
        from src.exe.run_model.compare_mpc_predictions import OUTPUT as COMPARISON_OUTPUT, run_all_days
        return run_all_days(day, day, tag=args.tag, horizon=args.horizon, X0=args.X0,
            x_steps=args.x_steps, fp_max_step=args.fp_max_step,
            prior_bandwidth=args.prior_bandwidth, max_iterations=args.max_iterations,
            max_nfev=args.max_nfev, state_maxiter=args.state_maxiter,
            steps_per_hour=args.steps_per_hour, max_state_iter=args.wm4_max_state_iter,
            state_tol=args.wm4_state_tol, wm4_forecasts=args.wm4_forecasts,
            min_capacity=args.min_capacity,
            output_root=args.output if args.output is not None else COMPARISON_OUTPUT,
            resume=not args.no_resume, fail_fast=not args.skip_errors)
    raw, _ = aligned_wm4_data(args.tag)
    normalized, calibration = scale.bm1_scale(raw, day)
    prediction, fit_diagnostics, skipped = predict_day(normalized, day, X0=args.X0, horizon=args.horizon,
        x_steps=args.x_steps, max_iterations=args.max_iterations, max_nfev=args.max_nfev,
        state_maxiter=args.state_maxiter, fp_max_step=args.fp_max_step,
        prior_bandwidth=args.prior_bandwidth, skip_errors=args.skip_errors,
        min_capacity=args.min_capacity)
    if prediction.empty: raise RuntimeError("BM1 MPC did not complete any forecast origins.")
    for col in ("I_BM1", "lower_68", "upper_68", "lower_95", "upper_95"):
        prediction[col] = scale.bm1_to_wm4(prediction[col].to_numpy(dtype=float), calibration)
    prediction["I_observed"] = raw["I"].reindex(pd.DatetimeIndex(prediction["target_time"])).to_numpy(dtype=float)
    output = (args.output if args.output is not None else OUTPUT) / f"{day:%Y-%m-%d}"
    output.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output / "bm1_mpc_forecasts.csv", index=False)
    fit_diagnostics.to_csv(output / "historical_diffusion.csv", index=False)
    pd.DataFrame(skipped, columns=["origin", "reason"]).to_csv(output / "skipped_origins.csv", index=False)
    results = []
    for lead, group in prediction.groupby("lead_hours"):
        bounds = {p: (group[f"lower_{p}"].to_numpy(dtype=float),
            group[f"upper_{p}"].to_numpy(dtype=float)) for p in (68, 95)}
        result = forecast_metrics(group["I_observed"].to_numpy(dtype=float), group["I_BM1"].to_numpy(dtype=float), bounds)
        results.append({"lead_hours": lead, **result})
        print(f"lead {lead}h: RMSE={result['rmse']:.6f}, MSE={result['mse']:.6f}, "
              f"MAE={result['mae']:.6f}, bias={result['bias']:+.6f}, "
              f"coverage68={result['coverage_68_pct']:.1f}%, coverage95={result['coverage_95_pct']:.1f}%")
    pd.DataFrame(results).to_csv(output / "rmse_coverage_by_lead.csv", index=False)
    nonconverged = int((~fit_diagnostics["fit_converged"]).sum())
    print(f"BM1 MPC fitted {prediction['origin'].nunique()}/24 hourly origins; "
        f"nonconverged={nonconverged}; "
        f"sigma_x median={fit_diagnostics['sigma_x'].median():.6f}; output={output}")
    if nonconverged: print(f"WARNING: {nonconverged} origin(s) hit an optimizer cap; "
        "their predictions were retained and marked fit_converged=False in the output.")
    return prediction


if __name__ == "__main__": main()
