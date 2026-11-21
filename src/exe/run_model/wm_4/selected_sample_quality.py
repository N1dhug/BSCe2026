"""Historical/target RMSE and empirical 95% coverage for the 12 ARM sample days.

Use exactly the N=6, k=24, weekly-temperature, ALL-group historical QP,
stochastic variance fit and full-24-hour target oracle in selected_sample_nll.
The primary scores exclude hour zero, exactly as the marginal NLL does
(hours 1..23). Full 24-hour metrics are additionally reported.

Run from the repository root:
    python -m src.exe.run_model.wm_4.selected_sample_quality
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.method.arm_variance import variance_kernel
from src.method.moving_horizon import group_frame
from src.exe.run_model.wm_4.selected_sample_nll import (
    ROOT, DAYS, N, K, BASELINE, GROUP, DEFAULT_MIN_TAU, score_day,
)


OUTPUT=ROOT/"files"/"output"/"wm_4"/"selected_sample_quality"
Z95=float(norm.ppf(0.975))
WINDOWS=(("lead_1_23",1),("full_24",0))


def evaluate(observed,predicted,variance,first_hour):
    """Pool residuals and marginal Gaussian intervals over one or N days."""
    observed=np.asarray(observed,dtype=float)
    predicted=np.asarray(predicted,dtype=float)
    variance=np.asarray(variance,dtype=float)
    if observed.ndim not in (1,2) or observed.shape[-1]!=K or predicted.shape!=observed.shape:
        raise ValueError("Expected matching 24-hour observed/predicted trajectories.")
    if variance.shape!=(K,) or not np.isfinite(
            np.r_[observed.ravel(),predicted.ravel(),variance]).all():
        raise ValueError("Expected one finite, common 24-hour variance vector.")
    if np.any(variance < -1e-12):
        raise ValueError("Negative model-implied demand variance.")
    residual=(predicted-observed)[...,first_hour:]
    variances=np.broadcast_to(np.maximum(variance[first_hour:],0.),residual.shape)
    included=np.abs(residual)<=Z95*np.sqrt(variances)
    count=int(residual.size)
    return {"observations":count,
            "squared_error_sum":float(np.square(residual).sum()),
            "variance_sum":float(variances.sum()),
            "covered_95_count":int(included.sum())}


def statistics(totals):
    count=totals["observations"]
    if count<=0: raise ValueError("Empty quality statistics.")
    return {"rmse":float(np.sqrt(totals["squared_error_sum"]/count)),
            "model_implied_rmse":float(np.sqrt(totals["variance_sum"]/count)),
            "coverage_95_percent":float(100.*totals["covered_95_count"]/count),
            "covered_95_count":totals["covered_95_count"],
            "observations":count}


def add_totals(pool,key,score):
    target=pool.setdefault(key,{"observations":0,"squared_error_sum":0.,
                                "variance_sum":0.,"covered_95_count":0})
    for name in target: target[name]+=score[name]


def run(year=2022,steps_per_hour=60,output=OUTPUT,
        fail_fast=False,min_tau=DEFAULT_MIN_TAU):
    if steps_per_hour<2: raise ValueError("--steps-per-hour must be >= 2.")
    if not np.isfinite(min_tau) or min_tau<0:
        raise ValueError("--min-observation-sigma must be finite and nonnegative.")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    frame=group_frame(GROUP)
    daily,pooled,skipped=[],{},[]
    print(f"ARM sample quality | {GROUP} | {BASELINE} | N={N}, k={K} | "
          f"year={year} | oracle initial states",flush=True)
    print("Day          | historical RMSE | historical cover | target RMSE | target cover "
          "(hours 1..23)",flush=True)
    for number,month_day in enumerate(DAYS,1):
        day=pd.Timestamp(f"{year}-{month_day}")
        try:
            (result,history,_,historical_observed,historical_predicted,_,
             _,scored_variance,tau2,sigma2,_)=score_day(
                frame,day,steps_per_hour,min_tau)
            theta=np.asarray(result["theta"],dtype=float)
            kernel=variance_kernel(float(theta[0]),float(theta[2]),
                                   steps_per_hour,K)
            variance=np.asarray(tau2+sigma2*kernel,dtype=float)
            if not np.allclose(variance[1:],scored_variance,rtol=1e-12,atol=1e-12):
                raise RuntimeError("Quality and NLL use different variances.")
            observed=np.asarray(result["observed"],dtype=float)
            predicted=np.asarray(result["predicted"],dtype=float)
            row={"day":str(day.date()),"baseline":BASELINE,
                 "training_weekdays":N,"horizon_hours":K,
                 "historical_days":";".join(str(d.date()) for d in history)}
            for label,first_hour in WINDOWS:
                for scope,y,mu in (
                        ("historical",historical_observed,historical_predicted),
                        ("target",observed,predicted)):
                    score=evaluate(y,mu,variance,first_hour)
                    add_totals(pooled,(scope,label),score)
                    for metric,value in statistics(score).items():
                        row[f"{scope}_{metric}_{label}"]=value
            daily.append(row)
            print(f"{day:%Y-%m-%d}   | "
                  f"{row['historical_rmse_lead_1_23']:.6f}        | "
                  f"{row['historical_coverage_95_percent_lead_1_23']:6.2f}%          | "
                  f"{row['target_rmse_lead_1_23']:.6f}    | "
                  f"{row['target_coverage_95_percent_lead_1_23']:6.2f}%",flush=True)
        except (ValueError,RuntimeError,KeyError,np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append({"day":str(day.date()),"reason":str(exc)})
            print(f"[{number:02d}/{len(DAYS)}] SKIPPED {day:%Y-%m-%d}: {exc}",flush=True)
    if not daily: raise RuntimeError("No days were evaluated.")
    day_frame=pd.DataFrame(daily)
    pooled_frame=pd.DataFrame([
        {"evaluation":scope,"window":label,**statistics(pooled[(scope,label)])}
        for label,_ in WINDOWS for scope in ("historical","target")
    ])
    day_frame.to_csv(output/"per_day_quality.csv",index=False)
    pooled_frame.to_csv(output/"pooled_quality.csv",index=False)
    pd.DataFrame(skipped,columns=("day","reason")).to_csv(
        output/"skipped_days.csv",index=False)
    (output/"configuration.json").write_text(json.dumps({
        "year":year,"days":list(DAYS),"group":GROUP,"baseline":BASELINE,
        "training_weekdays":N,"horizon_hours":K,"origin_hour":0,
        "steps_per_hour":steps_per_hour,"minimum_observation_sigma":min_tau,
        "method":"selected_sample_nll.score_day: historical drift QP, historical "
                 "variance least squares, full-24-hour target oracle",
        "primary_window":"lead hours 1..23, matching NLL and variance fitting",
        "additional_window":"hours 0..23, reported separately",
        "historical_observations_per_day":N*(K-1),
        "target_observations_per_day":K-1,
        "coverage":"pointwise central 95% Gaussian observation intervals, "
                   "including observation and process variance",
        "pooling":"sum squared errors then square root of mean; "
                  "sum coverage counts divided by total observation count",
        "caution":"Retrospectively initialized oracle, not operational forecast; "
                  "historical dates overlap between sample windows",
    },indent=2),encoding="utf-8")
    print("\nPooled quality (all completed sample days):",flush=True)
    for _,row in pooled_frame.iterrows():
        print(f"{row['evaluation']:>10} | {row['window']:>9} | "
              f"RMSE={row['rmse']:.6f} | model RMSE={row['model_implied_rmse']:.6f} | "
              f"95% coverage={row['coverage_95_percent']:.2f}% "
              f"({row['covered_95_count']}/{row['observations']})",flush=True)
    print(f"Completed {len(daily)}/{len(DAYS)}; skipped {len(skipped)}. "
          f"Results: {output}",flush=True)
    return day_frame,pooled_frame


def main():
    parser=ArgumentParser(description="ARM historical/target RMSE and 95% coverage "
                          "on the twelve NLL sample dates, without plotting.")
    parser.add_argument("--year",type=int,default=2022)
    parser.add_argument("--steps-per-hour",type=int,default=60)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--min-observation-sigma",type=float,default=DEFAULT_MIN_TAU)
    parser.add_argument("--fail-fast",action="store_true")
    args=parser.parse_args()
    run(args.year,args.steps_per_hour,args.output,args.fail_fast,
        args.min_observation_sigma)


if __name__=="__main__": main()
