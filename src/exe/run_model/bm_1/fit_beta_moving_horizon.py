"""Fit BM1 by the chapter-four historical/oracle methodology, using adaptive beta mixtures.

Run from repository root:
    python -m src.exe.run_model.bm_1.fit_beta_moving_horizon --N 5 --k 24 --origin "2022-08-22 06:00" --tag ALL

Both weekly-average and weekly-temperature baselines are fitted on exactly the
N corresponding historical intervals; stochastic parameters and N latent X0
are jointly optimized by adaptive beta-mixture maximum likelihood. The target
interval is reserved for an oracle initial-state fit and retrospective scoring.
"""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd

from src.exe.run_model.wm_3.fit_weekly import aligned_data
from src.method.bm_1_beta import bm_1_beta
from src.params.bm_1_beta import bm_1_beta_params
from src.tools.baseline import weekly_average, weekly_temperature
from src.validation.exe.run_model.bm_1.fit_beta_moving_horizon import bm_1_beta_moving_horizon_validation as checks


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "files" / "output" / "bm_1" / "methodology"
BASELINES = {"weekly_average": weekly_average, "weekly_temperature": weekly_temperature}


def score(truth, predicted):
    residual = np.asarray(predicted, dtype=float) - np.asarray(truth, dtype=float)
    if residual.shape != np.asarray(truth).shape or not np.isfinite(residual).all():
        raise ValueError("Cannot evaluate non-finite or differently shaped forecasts.")
    return {"rmse":float(np.sqrt(np.mean(residual**2))), "mae":float(np.mean(np.abs(residual))),
            "bias":float(np.mean(residual)), "mse":float(np.mean(residual**2))}


def timed(timings, label, function):
    started = perf_counter()
    print(f"[BM1 beta] {label} starting", flush=True)
    try:
        result = function()
        return result
    finally:
        elapsed = perf_counter() - started
        timings[label] = float(elapsed)
        print(f"[BM1 beta] {label}: {elapsed:.3f}s", flush=True)


def json_serializable(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, default=json_serializable, allow_nan=False)


def historical_origins(origin, N):
    # Chronologically ascending, so the most recent fitting interval is last.
    return [origin - pd.Timedelta(weeks=week) for week in range(N, 0, -1)]


def historical_training(frame, dates, k):
    times = pd.DatetimeIndex(np.concatenate([pd.date_range(d, periods=k, freq="h").values for d in dates]))
    selected = frame.reindex(times)
    if selected[["I", "Temp", "U"]].isna().any().any():
        raise ValueError("One or more corresponding historical intervals have missing measurements.")
    return selected[["I", "Temp"]].sort_index()


def rescale_past_only(frame, origin):
    """Undo prior affine full-season scaling using strictly pre-origin history.

    BM1 demands physically normalized I and U. Calibrating only on pre-origin
    records prevents forecast demand observations affecting scale selection.
    """
    past = frame.loc[frame.index < origin]
    if past.empty: raise ValueError("No historical data available before the prediction origin.")
    out, metadata = frame.copy(), {}
    for field in ("I", "U"):
        lo, hi = float(past[field].min()), float(past[field].max())
        if not np.isfinite([lo, hi]).all() or hi <= lo: raise ValueError(f"No valid historical range for {field}.")
        transformed = (out[field]-lo)/(hi-lo)
        metadata[field] = {"historical_min":lo, "historical_max":hi,
                           "clipped_below_zero":int((transformed < 0.0).sum()),
                           "clipped_above_one":int((transformed > 1.0).sum())}
        out[field] = transformed.clip(0.0, 1.0)
    return checks.frame(out), metadata


def refined_beta_checks(training, target, fitted, oracle, historical, *, mean_tol=0.002, nll_per_obs_tol=0.02):
    """Check historical posterior NLL and open demand means at a finer mixture resolution."""
    refined = bm_1_beta(32, 24, 96, 0.0625)
    historical_diffs, nll_diffs = [], []
    for data, state, original in zip(training, fitted["initial_states"], historical):
        control = refined.simulate(data, fitted, float(state))
        pilot_open = bm_1_beta(8, 8, 48, 0.25).simulate(data, fitted, float(state))
        historical_diffs.append(float(max(np.max(np.abs(control["mean"]-original["mean"])),
                                           np.max(np.abs(control["mean"]-pilot_open["mean"])))))
        pilot_nll = bm_1_beta(8, 8, 48, 0.25).simulate(data, fitted, float(state), observations=data["I"])["nll"]
        control_nll = refined.simulate(data, fitted, float(state), observations=data["I"])["nll"]
        nll_diffs.append(float(abs(pilot_nll-control_nll)/len(data["I"])))
    checked = refined.simulate(target, fitted, float(oracle["initial_state"]))
    checked.update(refined.intervals(checked, fitted["sigma_y"]))
    normal = oracle["prediction"]
    demand_diff = float(np.max(np.abs(checked["mean"]-normal["mean"])))
    quantile_diff = max(float(np.max(np.abs(checked[key]-normal[key])))
                        for key in ("lower_68", "upper_68", "lower_95", "upper_95"))
    return {"historical_max_mean_difference":max(historical_diffs),
            "historical_max_nll_difference_per_observation":max(nll_diffs),
            "oracle_max_mean_difference":demand_diff, "oracle_max_quantile_difference":quantile_diff,
            "mean_tolerance":mean_tol, "nll_per_observation_tolerance":nll_per_obs_tol,
            "passed":bool(max(historical_diffs + [demand_diff,quantile_diff]) <= mean_tol
                          and max(nll_diffs) <= nll_per_obs_tol)}


def finite_volume_checks(training, target, fitted, oracle, *, mean_tol=0.002, nll_per_obs_tol=0.02):
    """Independent FV beta accuracy check at two spatial/time resolutions."""
    from src.method.bm_1_fv_reference import bm_1_fv_reference
    beta = bm_1_beta(32, 24, 96, 0.0625)
    checks = {}
    for label, cells, step in (("reference",1631,1.0/32.0), ("refined",3263,1.0/64.0)):
        reference = bm_1_fv_reference(cells,step)
        max_mean, max_nll = 0.0, 0.0
        for data, state in zip(training, fitted["initial_states"]):
            same = beta.simulate(data, fitted, float(state))
            control = reference.simulate(data, fitted, float(state))
            max_mean = max(max_mean, float(np.max(np.abs(same["mean"]-control["mean"]))))
            beta_nll = beta.simulate(data, fitted, float(state), observations=data["I"])["nll"]
            fv_nll = reference.simulate(data, fitted, float(state), observations=data["I"])["nll"]
            max_nll = max(max_nll,float(abs(beta_nll-fv_nll)/len(data["I"])))
        x0 = float(oracle["initial_state"])
        target_beta = beta.simulate(target, fitted, x0)
        target_beta.update(beta.intervals(target_beta, fitted["sigma_y"]))
        target_fv = reference.simulate(target, fitted, x0)
        target_fv.update(reference.intervals(target_fv, fitted["sigma_y"]))
        target_mean = float(np.max(np.abs(target_beta["mean"]-target_fv["mean"])))
        target_quantile = max(float(np.max(np.abs(target_beta[key]-target_fv[key])))
                              for key in ("lower_68","upper_68","lower_95","upper_95"))
        checks[label] = {"historical_max_mean_difference":max_mean,
                         "historical_max_nll_difference_per_observation":max_nll,
                         "oracle_max_mean_difference":target_mean,
                         "oracle_max_quantile_difference":target_quantile,
                         "passed":bool(max(max_mean,target_mean,target_quantile)<=mean_tol
                                       and max_nll<=nll_per_obs_tol)}
    checks["passed"] = bool(all(value["passed"] for value in checks.values()))
    checks["mean_tolerance"] = mean_tol
    checks["nll_per_observation_tolerance"] = nll_per_obs_tol
    return checks


def run_baseline(frame, origin, dates, k, name, args, root):
    timings = {}
    started = perf_counter()
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    base = timed(timings, "baseline_fit", lambda: BASELINES[name](min_samples=2).fit(
        historical_training(frame, dates, k), origin, exact_historical=True))
    historical = [checks.window(frame, t, k, base) for t in dates]
    prediction = checks.window(frame, origin, k, base)
    solver = bm_1_beta_params()
    deterministic = timed(timings, "deterministic_fit", lambda: solver.fit_deterministic(
        historical, initial_state=args.x0, cycles=args.deterministic_cycles,
        max_nfev=args.max_nfev, state_maxiter=args.state_maxiter, substeps=args.substeps))
    print(f"[BM1 beta/{name}] deterministic MSE={deterministic['mse']:.8f}", flush=True)
    try:
        fitted = timed(timings, "stochastic_beta_mle", lambda: solver.fit_mle(
            historical, deterministic, maxiter=args.maxiter, maxfun=args.maxfun,
            progress=args.progress))
    except Exception as exc:
        timings["total_until_failure"] = float(perf_counter() - started)
        save_json(directory/"timings.json", timings)
        save_json(directory/"failure.json", {"stage":"stochastic_beta_mle",
            "error_type":type(exc).__name__, "message":str(exc),
            "N":len(dates), "k":k, "origin":origin, "baseline":name,
            "deterministic_fit":{"mse":deterministic["mse"],
                                 "parameters":{key:deterministic[key] for key in solver.NAMES},
                                 "historical_states":deterministic["initial_states"]},
            "timings_seconds":timings})
        raise
    if not fitted["converged"]:
        print(f"[BM1 beta/{name}] WARNING: MLE did not converge: {fitted['message']}", flush=True)
        if args.strict: raise RuntimeError(f"BM1 beta MLE failed to converge for {name}: {fitted['message']}")
    oracle = timed(timings, "oracle_initial_state", lambda: solver.oracle(
        prediction, fitted, maxiter=args.oracle_maxiter))
    baseline_score = score(prediction["I"], prediction["B"])
    predicted_score = score(prediction["I"], oracle["prediction"]["mean"])
    stochastic = bm_1_beta(24, 12, 48, 0.125)
    historical_curves = timed(timings, "historical_open_propagation", lambda: [
        stochastic.simulate(data, fitted, float(state))
        for data, state in zip(historical, fitted["initial_states"])])
    fit_truth = np.concatenate([data["I"] for data in historical])
    fit_prediction = np.concatenate([r["mean"] for r in historical_curves])
    fit_score = score(fit_truth, fit_prediction)
    verification = None
    if args.verify_refined or args.verify_fv:
        verification = timed(timings, "refined_beta_validation", lambda: refined_beta_checks(
            historical, prediction, fitted, oracle, historical_curves))
        print(f"[BM1 beta/{name}] refined numerical check passed={verification['passed']}", flush=True)
        if args.strict and not verification["passed"]: raise RuntimeError(f"BM1 beta resolution check failed for {name}.")
    if args.verify_fv:
        fv_verification = timed(timings, "finite_volume_validation", lambda: finite_volume_checks(
            historical, prediction, fitted, oracle))
        verification = {"beta":verification,"finite_volume":fv_verification,
                        "passed":bool(verification["passed"] and fv_verification["passed"])}
        print(f"[BM1 beta/{name}] independent FV check passed={fv_verification['passed']}", flush=True)
        if args.strict and not fv_verification["passed"]: raise RuntimeError(f"BM1 finite-volume verification failed for {name}.")
    rows = []
    for data, curve, date in zip(historical, historical_curves, dates):
        for j, (time, actual, nominal, estimate) in enumerate(zip(data["timestamps"], data["I"], data["B"], curve["mean"])):
            rows.append({"historical_origin":date, "timestamp":time, "lead_hour":j,
                         "observed":actual, "baseline":nominal, "beta_mean":estimate})
    pd.DataFrame(rows).to_csv(directory/"historical_fit.csv", index=False)
    p = oracle["prediction"]
    forecast = pd.DataFrame({"timestamp":prediction["timestamps"],"lead_hour":np.arange(k),
                             "observed":prediction["I"],"baseline":prediction["B"],"price":prediction["U"],
                             "beta_mean":p["mean"],"lower_68":p["lower_68"],"upper_68":p["upper_68"],
                             "lower_95":p["lower_95"],"upper_95":p["upper_95"]})
    forecast.to_csv(directory/"oracle_prediction.csv",index=False)
    coverage = {f"coverage_{level}":float(np.mean(
        (prediction["I"] >= p[f"lower_{level}"]) & (prediction["I"] <= p[f"upper_{level}"])))
                for level in (68,95)}
    timings["total"] = float(perf_counter()-started)
    save_json(directory/"timings.json", timings)
    numerically_verified = bool(verification is not None and (verification["passed"] if args.verify_fv else False))
    accepted = bool(fitted["converged"] and numerically_verified)
    report = {"baseline":name, "N":len(dates), "k":k, "origin":origin,
              "numerically_verified":numerically_verified,"methodology_accepted":accepted,
              "historical_origins":dates, "historical_fit":fit_score, "oracle":predicted_score,
              "baseline_score":baseline_score, "coverage":coverage,
              "oracle_initial_state":oracle["initial_state"],
              "baseline_metadata":base.metadata,
              "deterministic_parameters":{key:deterministic[key] for key in solver.NAMES},
              "historical_deterministic_states":deterministic["initial_states"],
              "stochastic_parameters":{key:fitted[key] for key in (*solver.NAMES,"sigma_x","sigma_y")},
              "historical_stochastic_states":fitted["initial_states"],
              "historical_nll":fitted["nll"], "mle_converged":fitted["converged"],
              "mle_iterations":fitted["iterations"],"mle_evaluations":fitted["evaluations"],
              "mle_message":fitted["message"], "resolutions":{"estimation":fitted["resolution"],
              "prediction":oracle["resolution"]}, "verification":verification, "timings_seconds":timings}
    save_json(directory/"result.json", report)
    print(f"[BM1 beta/{name}] historical RMSE={fit_score['rmse']:.6f} "
          f"oracle RMSE={predicted_score['rmse']:.6f} baseline RMSE={baseline_score['rmse']:.6f} "
          f"total={timings['total']:.3f}s", flush=True)
    return report


def run(N=5, k=24, origin="2022-08-22 06:00", *, tag="ALL", baseline="both",
        output=None, x0=0.5, maxiter=45, maxfun=700, deterministic_cycles=4,
        max_nfev=120, state_maxiter=60, substeps=2, oracle_maxiter=70,
        verify_refined=False, verify_fv=False, strict=False, progress=False):
    """Execute one historical-fit/oracle experiment for each selected baseline."""
    N, k = checks.settings(N, k, maxiter, deterministic_cycles, max_nfev)
    if strict and not (verify_refined and verify_fv):
        raise ValueError("--strict requires --verify-refined and --verify-fv for methodology acceptance.")
    origin = checks.origin(origin)
    tag = str(tag).upper()
    if tag not in ("ALL", "MED", "UDEN"): raise ValueError("tag must be ALL, MED, or UDEN.")
    names = list(BASELINES) if baseline == "both" else [baseline]
    if any(name not in BASELINES for name in names): raise ValueError("Unknown baseline selection.")
    root = Path(output) if output is not None else OUTPUT/f"{origin:%Y%m%d_%H%M}_N{N}_k{k}_{tag}"
    root.mkdir(parents=True, exist_ok=True)
    global_timings = {}
    started = perf_counter()
    frame, _ = timed(global_timings, "all_data_loading", lambda: aligned_data(
        None if tag == "ALL" else tag, price_domain=(0.0,1.0), price_year=None))
    frame, normalization = timed(global_timings, "historical_normalization", lambda: rescale_past_only(
        checks.frame(frame), origin))
    dates = historical_origins(origin, N)
    if dates[-1] + pd.Timedelta(hours=k) > origin:
        raise ValueError("Latest historical interval overlaps the prediction interval.")
    reports = {}
    for name in names:
        reports[name] = run_baseline(frame, origin, dates, k, name, argparse_namespace(
            x0=x0,maxiter=maxiter,maxfun=maxfun,deterministic_cycles=deterministic_cycles,
            max_nfev=max_nfev,state_maxiter=state_maxiter,substeps=substeps,
            oracle_maxiter=oracle_maxiter,verify_refined=verify_refined,verify_fv=verify_fv,strict=strict,progress=progress),root)
    global_timings["full_run"] = float(perf_counter()-started)
    save_json(root/"run_summary.json", {"origin":origin,"N":N,"k":k,"tag":tag,"baselines":names,
                                      "normalization":normalization,"timings_seconds":global_timings,
                                      "results":{name:{"historical_fit":r["historical_fit"],"oracle":r["oracle"],
                                                       "baseline_score":r["baseline_score"],"mle_converged":r["mle_converged"],
                                                       "methodology_accepted":r["methodology_accepted"],
                                                       "timings_seconds":r["timings_seconds"]} for name,r in reports.items()}})
    print(f"[BM1 beta] Finished. Full run {global_timings['full_run']:.3f}s. Output: {root}",flush=True)
    return reports


def argparse_namespace(**kwargs):
    from argparse import Namespace
    return Namespace(**kwargs)


def main():
    parser = ArgumentParser(description="BM1 adaptive beta-mixture historical fitting / oracle evaluation (single moving-horizon origin).")
    parser.add_argument("--N",type=int,default=5,help="Number of corresponding historical weekly intervals (default 5).")
    parser.add_argument("--k",type=int,default=24,help="Historical and oracle-prediction horizon, in hours (default 24).")
    parser.add_argument("--origin",default="2022-08-22 06:00",help='Origin timestamp, e.g. "2022-08-22 06:00".')
    parser.add_argument("--tag",choices=["ALL","MED","UDEN"],default="ALL")
    parser.add_argument("--baseline",choices=["both",*BASELINES],default="both")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--x0",type=float,default=0.5)
    parser.add_argument("--maxiter",type=int,default=45,help="Maximum stochastic MLE L-BFGS-B iterations.")
    parser.add_argument("--maxfun",type=int,default=700,help="Maximum stochastic MLE objective evaluations.")
    parser.add_argument("--deterministic-cycles",type=int,default=4)
    parser.add_argument("--max-nfev",type=int,default=120)
    parser.add_argument("--state-maxiter",type=int,default=60)
    parser.add_argument("--substeps",type=int,default=2)
    parser.add_argument("--oracle-maxiter",type=int,default=70)
    parser.add_argument("--verify-refined",action="store_true",help="Check against finer adaptive beta-mixture discretization.")
    parser.add_argument("--verify-fv",action="store_true",help="Check beta against independent K=1631 and K=3263 finite-volume references (slow).")
    parser.add_argument("--strict",action="store_true",help="Reject non-converged MLE or failed requested numerical verification.")
    parser.add_argument("--progress",action="store_true",help="Print MLE objective-evaluation progress and elapsed time.")
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__": main()
