"""N=6, k=24 stochastic BM1 fixed-beta historical MLE and target oracle.

Run from repository root:
    python -m src.exe.run_model.bm_1.fit_fixed_beta_day --day 2022-08-22 --tag ALL

The six matching historical weekdays fit shared BM1 dynamics and independent
initial beta densities. Target observations select ONLY the oracle initial
density location. Reported target NLL and intervals are OPEN-LOOP marginal
quantities, not filtered sequential likelihoods or genuine day-ahead scores.
"""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.moving_horizon_common import group_frame
from src.exe.run_model.wm_4.sweep_weekday_horizon import baseline_from_history, historical_calendar_days
from src.tools.impute_hourly import impute_hourly_days
from src.validation.exe.run_model.bm_1.fit_beta_moving_horizon import bm_1_beta_moving_horizon_validation as checks
from src.model.bm_1 import bm_1

N, K, BASELINE = 6, 24, "weekly_temperature"
ROOT = Path(__file__).resolve().parents[4]


def prepare(day, tag="ALL"):
    day = pd.Timestamp(day)
    if day.tz is not None or day != day.floor("D"): raise ValueError("Target origin must be a naive midnight.")
    history = [day-pd.Timedelta(weeks=n) for n in range(N,0,-1)]
    raw = group_frame(tag).copy()
    # Preserve the common WM normalization; BM1 price uses [0,1] instead of [-1,1].
    raw["U"] = (raw["U"] + 1.) / 2.
    days = pd.DatetimeIndex([d for origin in history for d in historical_calendar_days([origin],K)]).unique().sort_values()
    training, imputation = impute_hourly_days(raw, days=days, columns=("I","U","Temp"), require_days=True)
    baseline = baseline_from_history(training,history,BASELINE,K)
    hist = [checks.window(training,d,K,baseline) for d in history]
    target = checks.window(raw,day,K,baseline)
    return history,hist,target,imputation


def run(day="2022-08-22",tag="ALL",degree=6,quadrature=48,steps_per_hour=2,
        maxiter=12,maxfun=400,progress=True,output=None):
    started = perf_counter()
    history, hist, target, imputation = prepare(day,tag)
    input_seconds = perf_counter()-started
    # BM1 supplies the actual response and demand functions to the numerical solver.
    bm = bm_1(lambda t: .5,lambda t: .5,(0.,float(K)),K+1)
    estimator = bm.fixed_beta_estimator(degree,quadrature,steps_per_hour)
    fitted = estimator.fit(hist,maxiter=maxiter,maxfun=maxfun,progress=progress)
    oracle = estimator.oracle(target,fitted)
    total = perf_counter()-started
    report = {
        "day":str(pd.Timestamp(day).date()),"tag":tag,"N":N,"k":K,
        "historical_days":[str(d.date()) for d in history],
        "historical_fitted_nll":float(fitted["nll"]),
        "target_marginal_nll":oracle["target_marginal_nll"],
        "target_mean_nll_per_hour":oracle["target_mean_nll"],
        "expected_demand_rmse":oracle["expected_demand_rmse"],
        "empirical_95_coverage_pct":100.*oracle["empirical_95_coverage"],
        "covered_hours":int(round(K*oracle["empirical_95_coverage"])),
        "input_preparation_seconds":float(input_seconds),
        "parameter_estimation_seconds":float(fitted["runtime_seconds"]),
        "total_runtime_seconds":float(total),
        "optimizer_converged":bool(fitted["converged"]),
        "optimizer_message":fitted["message"],"optimizer_evaluations":fitted["evaluations"],
        "invalid_evaluations":fitted["invalid_evaluations"],
        "parameters":{key:float(fitted[key]) for key in estimator.NAMES},
        "historical_initial_distributions":fitted["initial_distributions"].tolist(),
        "oracle_initial_distribution":oracle["initial_distribution"].tolist(),
        "oracle_converged":oracle["oracle_converged"],
        "oracle_message":oracle["oracle_message"],
        "density_method":"Nielsen (2026) fixed beta basis + explicit Euler simplex active-set QP",
        "evaluation_contract":"oracle-conditioned target, open-loop marginal densities; no target filtering",
    }
    print(f"[BM1 fixed beta] day={report['day']} N={N} k={K} hours")
    print(f"  runtime: {total:.3f}s (fit: {fitted['runtime_seconds']:.3f}s)")
    print(f"  historical filtered NLL: {fitted['nll']:.6f}")
    print(f"  target marginal NLL: {oracle['target_marginal_nll']:.6f} (mean: {oracle['target_mean_nll']:.6f})")
    print(f"  target expected-demand RMSE: {oracle['expected_demand_rmse']:.6f}")
    print(f"  target empirical 95% coverage: {100*oracle['empirical_95_coverage']:.2f}%")
    print(f"  optimizer converged: {fitted['converged']} ({fitted['message']})")
    if output:
        output = Path(output)
        output.mkdir(parents=True,exist_ok=True)
        (output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        forecast = oracle["prediction"]
        pd.DataFrame({
            "timestamp":target["timestamps"],"observed":target["I"],
            "baseline":target["B"],"price":target["U"],"expected_demand":forecast["mean"],
            "lower_95":oracle["lower_95"],"upper_95":oracle["upper_95"],
            "marginal_nll":-forecast["loglik"],
        }).to_csv(output/"target_hourly.csv",index=False)
    return report


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--day",default="2022-08-22")
    parser.add_argument("--tag",default="ALL")
    parser.add_argument("--degree",type=int,default=6)
    parser.add_argument("--quadrature",type=int,default=48)
    parser.add_argument("--steps-per-hour",type=int,default=2)
    parser.add_argument("--maxiter",type=int,default=12)
    parser.add_argument("--maxfun",type=int,default=400)
    parser.add_argument("--output",default=None)
    parser.add_argument("--quiet",action="store_true")
    args=parser.parse_args()
    run(args.day,args.tag,args.degree,args.quadrature,args.steps_per_hour,
        args.maxiter,args.maxfun,not args.quiet,args.output)


if __name__=="__main__": main()
