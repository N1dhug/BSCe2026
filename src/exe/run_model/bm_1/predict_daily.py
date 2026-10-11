"""BM1 next-day demand forecast from five matching weekdays and latent-state uncertainty.

Run: python -m src.exe.run_model.bm_1.predict_daily --day 2022-08-22 --save
Import: predict_day("2022-08-22", X0=0.5, max_iterations=6, frame=...)
"""

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.exe.run_model.wm_3.fit_weekly import aligned_data
from src.model.bm_1 import bm_1
from src.params.bm_1 import lsq_params
from src.tools.baseline import statistical_output
from src.tools.forecast_metrics import predictive_intervals, forecast_metrics
from src.validation.exe.run_model.bm_1.predict_daily import bm_1_prediction_validation as checks


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "bm_1" / "prediction"
WEEKS = 5
HOURS = 24


def aligned_bm1_data(day, tag="MED"):
    """Use the existing importers, but recalibrate affine scales before the oldest fit day.

    Re-normalizing the already affine-normalized importer values cancels its
    full-season min/max constants. Out-of-reference-range later data are
    clipped to BM1's physical [0, 1] normalization and counted.
    """
    frame, _ = aligned_data(tag, price_domain=(0.0, 1.0), price_year=2022)
    first_day = checks.date(day) - pd.Timedelta(weeks=WEEKS)
    historical = frame.loc[frame.index < first_day, ["I", "U"]]
    frame = frame.copy()
    bounds, clipped = {}, {}
    for field in ("I", "U"):
        lo, hi = float(historical[field].min()), float(historical[field].max())
        if not np.isfinite(lo + hi) or hi <= lo: raise ValueError(f"Cannot calibrate normalized {field} from dates earlier than {first_day}.")
        transformed = (frame[field] - lo) / (hi - lo)
        clipped[field] = int(((transformed < 0.0) | (transformed > 1.0)).sum())
        frame[field] = transformed.clip(0.0, 1.0)
        bounds[field] = {"historical_min": lo, "historical_max": hi}
    return checks.frame(frame), {"reference_end": str(first_day), "bounds": bounds, "clipped_rows": clipped}


def matching_days(day):
    start = checks.date(day)
    return [start - pd.Timedelta(weeks=k) for k in range(WEEKS, 0, -1)]


def held_signal(values):
    """Vectorized right-continuous zero-order hold on the hourly grid."""
    values = np.asarray(values, dtype=float).copy()
    def signal(t):
        t = np.asarray(t, dtype=float)
        k = np.clip(np.floor(t + 1e-12).astype(int), 0, len(values) - 1)
        result = values[k]
        return float(result) if t.ndim == 0 else result
    return signal


def prepare_days(frame, dates):
    """Refit each training-day baseline only on information from earlier dates."""
    datasets, metadata = [], []
    for start in dates:
        baseline = statistical_output().fit(frame, start)
        data, _ = checks.day_signals(frame, start, baseline, require_demand=True)
        datasets.append(data)
        metadata.append({"day": str(start.date()), **baseline.metadata})
    return datasets, metadata


def forecast_from_distribution(data, fitted, *, sigma_x=0.0, x_steps=101, fp_max_step=0.25, substeps=2):
    """Forecast the five-point empirical state mixture, optionally through BM1 FP.

    sigma_x=0 propagates each state deterministically without numerical density
    diffusion; sigma_x>0 evolves the discrete initial mass using finite volume.
    This is conditional on supplied target-day prices and temperatures.
    """
    states = np.asarray(fitted["initial_states"], dtype=float)
    model_input = {**data, "dt": np.diff(data["Time"])}
    paths = np.array([lsq_params._predict(model_input, float(x), fitted, substeps)[1][:HOURS] for x in states])
    state_paths = np.array([lsq_params._predict(model_input, float(x), fitted, substeps)[0][:HOURS] for x in states])
    mass = checks.initial_distribution(states, x_steps)
    model = bm_1(held_signal(data["B"]), lambda t: 0.0 * np.asarray(t, dtype=float) + 0.5, (0.0, HOURS), HOURS + 1)
    forecast = paths.mean(axis=0)
    lower, upper = np.quantile(paths, [0.1, 0.9], axis=0)
    support = [paths[:, i] for i in range(HOURS)]
    probabilities = [np.full(states.size, 1.0 / states.size) for _ in range(HOURS)]
    fp = None

    if sigma_x > 0.0:
        parameters = [held_signal(data["U"]), fitted["f"], fitted["g"], fitted["C"], fitted["k"], fitted["Xi"]]
        result = model.fokker_planc_reference_solver(mass, parameters, sigma_x, t_eval=data["Time"], max_step=fp_max_step)
        t, x, density, cell_mass, EX, VX, price, baseline, expected_demand, diagnostics = result
        forecast = expected_demand[:HOURS]
        lower, upper = np.empty(HOURS), np.empty(HOURS)
        for i in range(HOURS):
            delta = model.l(fitted["f"](x), fitted["g"](data["U"][i]), fitted["k"])
            demand = model.D(data["B"][i], fitted["Xi"], delta)
            order = np.argsort(demand)
            cdf = np.cumsum(cell_mass[i, order])
            q = np.minimum(np.searchsorted(cdf, [0.1, 0.9], side="left"), len(x) - 1)
            lower[i], upper[i] = demand[order[q]]
        fp = {"x": x, "cell_mass": cell_mass, "density": density, "state_mean": EX, "state_variance": VX, "diagnostics": diagnostics}
        support = []
        for i in range(HOURS):
            delta = model.l(fitted["f"](x), fitted["g"](data["U"][i]), fitted["k"])
            support.append(model.D(data["B"][i], fitted["Xi"], delta))
        probabilities = list(cell_mass[:HOURS])

    intervals = predictive_intervals(support, probabilities)
    return {"forecast": forecast, "lower": lower, "upper": upper, "intervals": intervals,
        "ensemble": paths, "state_paths": state_paths, "initial_mass": mass,
        "initial_grid": model.fokker_planc_reference.grid(x_steps)[1], "fp": fp}


def predict_day(day="2022-08-22", *, frame=None, tag="MED", X0=0.5, C=None, Xi=None, alpha0=2.0, beta0=2.0, k0=1.0, C0=1.0, Xi0=0.5, max_iterations=6, min_improvement=1e-6, max_nfev=150, state_maxiter=100, substeps=2, sigma_x=0.0, x_steps=101, fp_max_step=0.25, score=False, joint_max_nfev=0, joint_max_seconds=20., joint_gtol=1e-5):
    """Fit one shared theta over five previous same weekdays, then forecast a day.

    First stage: a pooled theta LSQ with one shared X0 guess. Second stage:
    optimize five daily X0 values holding theta fixed. Alternating cycles stop
    at max_iterations or when pooled SSE stops improving. The fitted X0 values
    define a five-atom empirical initial distribution for the prediction day.
    """
    start = checks.date(day)
    X0, sigma_x, x_steps, fp_max_step = checks.options(X0, sigma_x, x_steps, fp_max_step, tag=tag, max_iterations=max_iterations, min_improvement=min_improvement)
    if frame is None: frame, normalization = aligned_bm1_data(start, tag)
    else: frame, normalization = checks.frame(frame), {"source": "supplied frame; user is responsible for causal normalization"}
    days = matching_days(start)
    training, training_baselines = prepare_days(frame, days)
    base = statistical_output().fit(frame, start)
    target, index = checks.day_signals(frame, start, base, require_demand=False)

    model = bm_1(held_signal(target["B"]), lambda t: 0.0 * np.asarray(t, dtype=float) + 0.5, (0.0, HOURS), HOURS + 1)
    fitter = lsq_params(model)
    fitted = fitter.fit_many(training, X0, C=C, Xi=Xi, alpha0=alpha0, beta0=beta0, k0=k0, C0=C0, Xi0=Xi0,
        max_iterations=max_iterations, min_improvement=min_improvement, max_nfev=max_nfev, state_maxiter=state_maxiter, substeps=substeps,
        joint_max_nfev=joint_max_nfev,joint_max_seconds=joint_max_seconds,joint_gtol=joint_gtol)
    ensemble = forecast_from_distribution(target, fitted, sigma_x=sigma_x, x_steps=x_steps, fp_max_step=fp_max_step, substeps=substeps)
    hourly = pd.DataFrame({
        "I_predicted": ensemble["forecast"], "I_lower": ensemble["lower"], "I_upper": ensemble["upper"],
        "I_ensemble_mean": ensemble["ensemble"].mean(axis=0), "B": target["B"][:HOURS], "U": target["U"][:HOURS],
        "X_ensemble_mean": ensemble["state_paths"].mean(axis=0),
    }, index=index)
    for level, (lo, hi) in ensemble["intervals"].items():
        hourly[f"lower_{level}"], hourly[f"upper_{level}"] = lo, hi
    training_rows = []
    for day_index, (date, data, state) in enumerate(zip(days, training, fitted["initial_states"])):
        predicted = fitter._predict(data, float(state), fitted, substeps)[1][:HOURS]
        training_rows.append({"day": date, "X0": state, **checks.metrics(data["I"], predicted),
            "baseline_mse": checks.metrics(data["I"], data["B"][:HOURS])["mse"]})
    summary = pd.DataFrame(training_rows).set_index("day")
    if fitted["warning"]: print(f"[BM1 day-ahead {start:%Y-%m-%d}] WARNING: {fitted['warning']}", flush=True)
    metrics = None
    if score:
        actual = frame.reindex(index)["I"].to_numpy(dtype=float)
        metrics = {"model": forecast_metrics(actual, hourly["I_predicted"], ensemble["intervals"]),
            "baseline": forecast_metrics(actual, hourly["B"])}
        hourly["I_observed"] = actual
    return {
        "day": start, "tag": tag, "training_days": days, "training": summary, "hourly": hourly, "fit": fitted,
        "initial_states": fitted["initial_states"].copy(), "initial_weights": fitted["initial_weights"].copy(),
        "initial_mass": ensemble["initial_mass"], "initial_grid": ensemble["initial_grid"],
        "ensemble": ensemble, "baseline_fit": dict(base.metadata), "training_baselines": training_baselines,
        "normalization": normalization, "sigma_x": sigma_x, "metrics": metrics,
    }


def save_result(result, output_root=OUTPUT, plot=True):
    """Save the forecast, five fitted state atoms, fit history and optional plot."""
    directory = Path(output_root) / f"{result['day']:%Y-%m-%d}"
    directory.mkdir(parents=True, exist_ok=True)
    result["hourly"].to_csv(directory / "hourly.csv", index_label="time")
    result["training"].to_csv(directory / "training_days.csv", index_label="day")
    pd.DataFrame(result["fit"]["history"]).to_csv(directory / "iterations.csv", index=False)
    fit = result["fit"]
    metadata = {
        "target_day": str(result["day"].date()), "tag": result["tag"], "training_days": [str(d.date()) for d in result["training_days"]],
        "theta": {key: float(fit[key]) for key in ("alpha", "beta", "k", "Xi", "C")},
        "state_X0": [float(v) for v in result["initial_states"]], "initial_weights": [float(v) for v in result["initial_weights"]],
        "pooled_mse": float(fit["mse"]), "iterations": int(fit["iterations"]),
        "fit_converged": bool(fit["converged"]), "fit_stop_reason": fit["stop_reason"],
        "fit_message": fit["message"], "theta_nfev": int(fit["theta_nfev"]),
        "sigma_x": float(result["sigma_x"]), "baseline_fit_start": str(result["baseline_fit"]["fit_start"]),
        "baseline_fit_end": str(result["baseline_fit"]["fit_end"]), "normalization": result["normalization"], "metrics": result["metrics"],
    }
    if 'joint_fit' in fit:metadata['joint_fit']=fit['joint_fit']
    (directory / "fit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        hourly = result["hourly"]
        axes[0].plot(hourly.index, hourly["I_predicted"], label="BM1 forecast")
        axes[0].fill_between(hourly.index, hourly["I_lower"], hourly["I_upper"], alpha=0.2, label="Initial-state uncertainty" if result["sigma_x"] == 0 else "BM1 FP uncertainty")
        axes[0].plot(hourly.index, hourly["B"], label="Baseline B(t)", linestyle="--")
        if "I_observed" in hourly: axes[0].plot(hourly.index, hourly["I_observed"], label="Observed demand", alpha=0.8)
        axes[0].set_ylabel("Normalized demand")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(hourly.index, hourly["U"], label="Price U(t)")
        axes[1].set_ylabel("Normalized price"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel("Time")
        fig.suptitle(f"BM1 forecast {result['day']:%Y-%m-%d} | pooled five matching weekdays")
        fig.tight_layout()
        fig.savefig(directory / "forecast.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    return directory


def main():
    parser = ArgumentParser(description="Forecast normalized BM1 electricity demand using five matching weekdays.")
    parser.add_argument("--day", default="2022-08-22", help="Target calendar date, YYYY-MM-DD.")
    parser.add_argument("--tag", default="MED", help="Consumption group.")
    parser.add_argument("--X0", type=float, default=0.5, help="Shared initial-state guess used for pooled fitting.")
    parser.add_argument("--C", type=float, default=None, help="Fix capacity C instead of fitting it.")
    parser.add_argument("--Xi", type=float, default=None, help="Fix flexibility fraction Xi instead of fitting it.")
    parser.add_argument("--max-iterations", type=int, default=6, help="Maximum alternating theta/state fitting cycles.")
    parser.add_argument("--min-improvement", type=float, default=1e-6, help="Stop when relative SSE gain drops below this.")
    parser.add_argument("--max-nfev", type=int, default=150, help="Maximum evaluations per joint theta least-squares step.")
    parser.add_argument("--state-maxiter", type=int, default=100, help="Maximum bounded scalar optimizer iterations per day.")
    parser.add_argument('--joint-max-nfev',type=int,default=0,help='Optional final joint refinement budget; 0 keeps the original fit.')
    parser.add_argument('--joint-max-seconds',type=float,default=20.)
    parser.add_argument('--joint-gtol',type=float,default=1e-5)
    parser.add_argument("--sigma-x", type=float, default=0.0, help="Known diffusion amplitude; 0 uses an empirical deterministic ensemble.")
    parser.add_argument("--x-steps", type=int, default=101, help="Finite-volume cells if sigma-x > 0.")
    parser.add_argument("--fp-max-step", type=float, default=0.25, help="Maximum internal FV time step in hours.")
    parser.add_argument("--score", action="store_true", help="Compare with held-out target I only after forecasting.")
    parser.add_argument("--save", action="store_true", help="Save CSV/JSON and forecast plot to files/output/bm_1/prediction.")
    args = parser.parse_args()
    forecast = predict_day(args.day, tag=args.tag, X0=args.X0, C=args.C, Xi=args.Xi, max_iterations=args.max_iterations,
        min_improvement=args.min_improvement, max_nfev=args.max_nfev, state_maxiter=args.state_maxiter, sigma_x=args.sigma_x,
        x_steps=args.x_steps, fp_max_step=args.fp_max_step, score=args.score,
        joint_max_nfev=args.joint_max_nfev,joint_max_seconds=args.joint_max_seconds,joint_gtol=args.joint_gtol)
    fit = forecast["fit"]
    print(f"Target: {forecast['day']:%Y-%m-%d}; training days: {', '.join(str(d.date()) for d in forecast['training_days'])}")
    print("Joint fit:", {key: round(float(fit[key]), 6) for key in ("alpha", "beta", "k", "Xi", "C")})
    print(f"Five fitted initial states: {np.array2string(forecast['initial_states'], precision=5)}")
    print(f"Pooled training MSE: {fit['mse']:.8f}; accepted/attempted cycles: {sum(v['accepted'] for v in fit['history'])}/{fit['iterations']}")
    print(f"Fit converged: {fit['converged']}; stop_reason={fit['stop_reason']}; {fit['message']}")
    if forecast["metrics"] is not None:
        print("Held-out target metrics:", forecast["metrics"])
        model_metrics = forecast["metrics"]["model"]
        for level in (68, 95):
            print(f"BM1 central {level}% coverage: {model_metrics[f'coverage_{level}_count']}/24 "
                f"({model_metrics[f'coverage_{level}_pct']:.2f}%), nominal gap "
                f"{model_metrics[f'coverage_{level}_gap_pp']:+.2f} pp")
        if args.sigma_x == 0: print("BM1 sigma_x=0: coverage reflects initial-state variability only.")
    if args.save: print("Saved:", save_result(forecast))
    return forecast


if __name__ == "__main__": main()
