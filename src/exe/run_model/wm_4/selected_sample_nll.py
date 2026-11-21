"""ARM/WM4 historical and target marginal NLL on twelve selected 2022 dates.

N=6 corresponding historical weekdays, k=24 hours, weekly-temperature
baseline, 00:00 origin, group ALL. Historical drift and target full-state
oracle use the existing ARM joint QP. Stochastic observation/process
variances are re-estimated by the existing nonnegative squared-residual QP
using only historical hours 1,...,23 (the observed t=0 rate initializes
each historical model). Both historical and target likelihoods use only
hours 1,...,23. No sequential/filtered likelihood is evaluated.

Historical fit NLL: six free-running historical trajectories, 6*23=138
marginal Gaussian log densities per target date, computed with the
historically fitted drift, initial states, and noise parameters.
Target oracle NLL: 23 unfiltered Gaussian log densities per target date,
using the unchanged target-day full-24-hour [F0,R0] retrospective oracle.

The same historical dates can appear in multiple overlapping training
windows; pooled historical NLL counts each fitted window separately.
The oracle uses target observations to choose its initial state, so
this is not an operational out-of-sample forecast score.

Run from repository root:
    python -m src.exe.run_model.wm_4.selected_sample_nll

Produces timing_per_day.csv, timing_average.csv and timing_average.png for
all successfully scored sample dates, in addition to the NLL tables.
"""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.method.arm_variance import fit_variances, variance_kernel
from src.method.moving_horizon import group_frame
from src.method.moving_horizon import historical_weekdays
from src.exe.run_model.wm_4.selected_bm_dates_arm import fit_origin
from src.method.sample_timing import timing_report
from src.exe.run_model.wm_4.sweep_weekday_horizon import historical_calendar_days
from src.tools.impute_hourly import impute_hourly_days


ROOT=Path(__file__).resolve().parents[4]
OUTPUT=ROOT/"files"/"output"/"wm_4"/"selected_sample_nll"
N,K,BASELINE,GROUP=6,24,"weekly_temperature","ALL"
DEFAULT_MIN_TAU=0.0
DAYS=("04-11","04-19","04-27","05-05","05-13","05-21",
      "05-29","05-30","06-07","08-10","08-18","08-26")


def require_target(frame,day):
    """Target I, U, Temp must be observed on every hour; never impute targets."""
    block=frame.reindex(pd.date_range(day,periods=K,freq="h"))[["I","U","Temp"]]
    if block.isna().any().any():
        missing=block.isna().sum()
        raise ValueError(f"Incomplete target day: missing hourly I={missing['I']}, "
                         f"U={missing['U']}, Temp={missing['Temp']}.")


def historical_trajectories(result,steps_per_hour):
    """Return the actual six observed and historically fitted free-run curves."""
    sph=int(steps_per_hour)
    idx=np.arange(K,dtype=int)*sph
    inputs,states=result["historical_inputs"],result["historical_states"]
    if len(inputs)!=N or len(states)!=N:
        raise ValueError("Expected exactly six historical training intervals.")
    observed=np.stack([np.asarray(item["I"],dtype=float)[idx] for item in inputs])
    fitted=np.stack([np.asarray(state["I"],dtype=float)[idx] for state in states])
    if observed.shape!=(N,K) or fitted.shape!=(N,K) or not np.isfinite(
            np.r_[observed.ravel(),fitted.ravel()]).all():
        raise ValueError("Historical trajectories require six complete finite 24-hour curves.")
    return observed,fitted


def historical_errors(result,steps_per_hour):
    """Residuals for the original fitted historical trajectories, including t=0."""
    observed,fitted=historical_trajectories(result,steps_per_hour)
    return observed-fitted


def positive_variances(result,steps_per_hour,min_tau=DEFAULT_MIN_TAU):
    """Refit historical nonnegative variance QP using ONLY hours 1,...,23.

    Original fit_origin's variance output includes t=0, so never use those
    estimated tau2/sigma2 values for marginal likelihoods. By default the
    new fit is purely nonnegative (tau>=0); --min-observation-sigma optionally
    enforces a positive tau lower bound, using exact active-face QP solutions.
    """
    if not np.isfinite(min_tau) or min_tau<0:
        raise ValueError("The minimum observation-noise sigma must be finite and nonnegative.")
    errors=historical_errors(result,steps_per_hour)[:,1:].reshape(-1)
    kernel=variance_kernel(float(result["theta"][0]),float(result["theta"][2]),
                           steps_per_hour,K)[1:]
    x=np.tile(kernel,N)
    if errors.shape!=x.shape or errors.size!=N*(K-1):
        raise ValueError("Stochastic fit requires six historical 23-hour trajectories.")
    tau2,sigma2,_=fit_variances(errors,x)
    floor=float(min_tau**2)
    if tau2>=floor: return float(tau2),float(sigma2),False

    y=errors**2
    xx=float(np.dot(x,x))
    q_on_tau_face=max(0.,float(np.dot(x,y-floor)/xx)) if xx>0 else 0.
    candidates=((floor,q_on_tau_face),(max(floor,float(np.mean(y))),0.))
    tau2,sigma2=min(candidates,
                    key=lambda pair:float(np.mean((y-pair[0]-pair[1]*x)**2)))
    if not (np.isfinite([tau2,sigma2]).all() and tau2>=floor and sigma2>=0):
        raise RuntimeError("Constrained variance QP returned an invalid solution.")
    return float(tau2),float(sigma2),True


def score_marginal(observed,open_mean,theta,tau2,sigma2,steps_per_hour):
    """Return marginal NLL for hours 1,...,23, for one or six trajectories.

    Accept arrays shaped (24,) for a target or (6,24) for the historical
    fit. Both share the same fitted variance kernel and exclude t=0.
    """
    y=np.asarray(observed,dtype=float)
    mu=np.asarray(open_mean,dtype=float)
    r=np.asarray(theta,dtype=float)
    sph=int(steps_per_hour)
    if y.ndim not in (1,2) or y.shape[-1]!=K or mu.shape!=y.shape or r.shape!=(3,):
        raise ValueError("NLL needs matching complete 24-hour curves and three drift coefficients.")
    if sph<2 or not np.isfinite(np.r_[y.ravel(),mu.ravel(),r,tau2,sigma2]).all():
        raise ValueError("Nonfinite data or invalid time resolution in marginal score.")
    if tau2<0 or sigma2<0:
        raise ValueError("Marginal Gaussian noise variances must be nonnegative.")
    kernel=variance_kernel(float(r[0]),float(r[2]),sph,K)[1:]
    variance=tau2+sigma2*kernel
    if not np.isfinite(variance).all() or np.any(variance<=0):
        raise ValueError("Zero/nonfinite Gaussian variance at a scored hour (1..23).")
    nll=-norm.logpdf(y[...,1:],loc=mu[...,1:],scale=np.sqrt(variance))
    if not np.isfinite(nll).all():
        raise ValueError("Nonfinite marginal negative log-likelihood.")
    return np.asarray(nll,dtype=float),variance


def score_day(frame,day,steps_per_hour,min_tau):
    """Score one ARM sample and measure actual computational stage runtimes.

    The timings include both the variance fit within fit_origin and the
    hours-1..23 variance refit used for the reported marginal likelihood.
    """
    sample_started=perf_counter()
    history=historical_weekdays(day,N)
    required=historical_calendar_days(history,K)
    training,imputation=impute_hourly_days(
        frame,days=required,columns=("I","U","Temp"),require_days=True)
    require_target(frame,day)
    result=fit_origin(frame,training,day,history,BASELINE,steps_per_hour)
    variance_started=perf_counter()
    tau2,sigma2,regularized=positive_variances(result,steps_per_hour,min_tau)
    variance_refit_seconds=perf_counter()-variance_started
    historical_observed,historical_predicted=historical_trajectories(result,steps_per_hour)
    historical_nll,variance=score_marginal(
        historical_observed,historical_predicted,result["theta"],
        tau2,sigma2,steps_per_hour)
    target_nll,target_variance=score_marginal(
        result["observed"],result["predicted"],result["theta"],
        tau2,sigma2,steps_per_hour)
    if historical_nll.shape!=(N,K-1) or target_nll.shape!=(K-1,):
        raise RuntimeError("Historical and target marginal NLL sample counts are incorrect.")
    if not np.allclose(variance,target_variance,rtol=1e-13,atol=1e-13):
        raise RuntimeError("Historical and target scores used different variance kernels.")
    result["sample_timing_seconds"]={
        "drift_fit_seconds":float(result["drift_seconds"]),
        "variance_fit_seconds":float(result["diffusion_seconds"]+variance_refit_seconds),
        "variance_initial_fit_seconds":float(result["diffusion_seconds"]),
        "variance_scoring_refit_seconds":float(variance_refit_seconds),
        "oracle_optimization_seconds":float(result["oracle_seconds"]),
        "complete_compute_seconds":float(perf_counter()-sample_started),
    }
    return (result,history,imputation,historical_observed,historical_predicted,
            historical_nll,target_nll,variance,tau2,sigma2,regularized)


def run(year=2022,steps_per_hour=60,output=OUTPUT,fail_fast=False,min_tau=DEFAULT_MIN_TAU,show=False):
    if steps_per_hour<2: raise ValueError("steps_per_hour must be >= 2.")
    if not np.isfinite(min_tau) or min_tau<0: raise ValueError("min_tau must be nonnegative.")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    frame=group_frame(GROUP)
    daily,target_hourly,historical_hourly,skipped=[],[],[],[]
    print(f"ARM / WM4 marginal NLL | group={GROUP} | baseline={BASELINE} "
          f"| N={N} historical weekdays | k={K}h | origin=00:00",flush=True)
    print(f"Scored hours=1..23; historical fit={N*(K-1)} observations/date; "
          f"target oracle={K-1} observations/date; minimum tau={min_tau:g}.",flush=True)
    for number,md in enumerate(DAYS,1):
        day=pd.Timestamp(f"{year}-{md}")
        try:
            (result,history,imputation,hist_obs,hist_pred,hist_nll,target_nll,
             variance,tau2,sigma2,regularized)=score_day(frame,day,steps_per_hour,min_tau)
            historical_total=float(hist_nll.sum())
            target_total=float(target_nll.sum())
            row={
                "day":str(day.date()),"origin_hour":0,"baseline":BASELINE,
                "training_weekdays":N,"horizon_hours":K,
                "historical_observations":N*(K-1),
                "historical_fit_marginal_nll":historical_total,
                "historical_fit_marginal_nll_per_observation":historical_total/(N*(K-1)),
                "target_observations":K-1,
                "target_marginal_nll":target_total,
                "target_marginal_nll_per_observation":target_total/(K-1),
                "historical_rmse":float(np.sqrt(np.mean((hist_obs[:,1:]-hist_pred[:,1:])**2))),
                "target_rmse":float(np.sqrt(np.mean(
                    (np.asarray(result["observed"][1:])-np.asarray(result["predicted"][1:]))**2))),
                "oracle_F0":float(result["initial"][0]),
                "oracle_R0":float(result["initial"][1]),
                "theta_F":float(result["theta"][0]),
                "theta_U":float(result["theta"][1]),
                "theta_R":float(result["theta"][2]),
                "tau":float(np.sqrt(tau2)),
                "sigma":float(np.sqrt(sigma2)),
                "observation_variance_regularized":bool(regularized),
                "historical_days":";".join(str(d.date()) for d in history),
                "imputed_historical_hourly_slots":int(imputation["imputed_hourly_slots"]),
                **result["sample_timing_seconds"],
            }
            daily.append(row)
            for hour in range(1,K):
                j=hour-1
                target_hourly.append({
                    "day":str(day.date()),"hour":hour,
                    "observed":float(result["observed"][hour]),
                    "open_loop_mean":float(result["predicted"][hour]),
                    "predicted_variance":float(variance[j]),
                    "marginal_nll":float(target_nll[j]),
                })
                for slot,past in enumerate(history):
                    historical_hourly.append({
                        "target_day":str(day.date()),"historical_day":str(past.date()),
                        "historical_number":slot+1,"hour":hour,
                        "observed":float(hist_obs[slot,hour]),
                        "fitted_mean":float(hist_pred[slot,hour]),
                        "predicted_variance":float(variance[j]),
                        "marginal_nll":float(hist_nll[slot,j]),
                    })
            adjusted=" [tau lower-bound active]" if regularized else ""
            print(f"[{number:02d}/{len(DAYS)}] {day:%Y-%m-%d} | "
                  f"historical marginal NLL={historical_total:+.8f} "
                  f"(mean/hour={row['historical_fit_marginal_nll_per_observation']:+.8f}) "
                  f"| target marginal NLL={target_total:+.8f} "
                  f"(mean/hour={row['target_marginal_nll_per_observation']:+.8f})"
                  f"{adjusted}",flush=True)
        except (ValueError,RuntimeError,KeyError,np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append({"day":str(day.date()),"reason":str(exc)})
            print(f"[{number:02d}/{len(DAYS)}] {day:%Y-%m-%d} | SKIPPED: {exc}",flush=True)
    pd.DataFrame(daily).to_csv(output/"per_day_nll.csv",index=False)
    pd.DataFrame(target_hourly).to_csv(output/"hourly_nll.csv",index=False)
    pd.DataFrame(historical_hourly).to_csv(output/"historical_hourly_nll.csv",index=False)
    pd.DataFrame(skipped,columns=("day","reason")).to_csv(output/"skipped_days.csv",index=False)
    (output/"configuration.json").write_text(json.dumps({
        "year":year,"days":list(DAYS),"group":GROUP,"baseline":BASELINE,
        "N":N,"k":K,"origin_hour":0,"steps_per_hour":steps_per_hour,
        "minimum_observation_sigma":min_tau,
        "historical_variance_fit":"joint drift QP followed by nonnegative variance LS QP on six historical days, hours 1..23",
        "historical_nll":"six fitted free-running 23-hour Gaussian marginal likelihoods per target date",
        "target_nll":"one oracle-conditioned open-loop 23-hour Gaussian marginal likelihood per target date",
        "historical_observations_per_date":N*(K-1),
        "target_observations_per_date":K-1,
        "variance_regularization":"optional tau lower bound, disabled by default",
        "target_initial_state":"[F0,R0] optimized over full observed 24-hour target (unchanged)",
        "warning":"Retrospective oracle, not a causal forecast; historical windows may overlap across sample dates",
    },indent=2),encoding="utf-8")
    print(f"\nCompleted {len(daily)}/{len(DAYS)} days; skipped {len(skipped)}.")
    if daily:
        nh=N*(K-1)*len(daily)
        nt=(K-1)*len(daily)
        htotal=float(sum(row["historical_fit_marginal_nll"] for row in daily))
        ttotal=float(sum(row["target_marginal_nll"] for row in daily))
        print(f"Pooled historical marginal NLL={htotal:+.8f}; "
              f"mean/observation={htotal/nh:+.8f} ({nh} observations).")
        print(f"Pooled target marginal NLL={ttotal:+.8f}; "
              f"mean/observation={ttotal/nt:+.8f} ({nt} observations).")
    print(f"Results: {output}",flush=True)
    if not daily: raise RuntimeError("No sample day could be scored; see skipped_days.csv.")
    # The shared timing plot uses the actual stages measured in score_day.
    # Complete time is inclusive, not the sum of the other three bars.
    timing_report(pd.DataFrame(daily),"ARM",output,(
        ("Deterministic drift QP","drift_fit_seconds"),
        ("Variance fit (initial + NLL refit)","variance_fit_seconds"),
        ("Oracle initial-state QP","oracle_optimization_seconds"),
        ("Complete process","complete_compute_seconds"),
    ),show=show,expected_days=len(daily))
    return pd.DataFrame(daily)


def main():
    parser=ArgumentParser(description="ARM historical-fit and target-oracle marginal NLL on 12 selected dates.")
    parser.add_argument("--year",type=int,default=2022)
    parser.add_argument("--steps-per-hour",type=int,default=60)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--min-observation-sigma",type=float,default=DEFAULT_MIN_TAU,
                        help="Optional tau lower bound (default 0).")
    parser.add_argument("--fail-fast",action="store_true")
    parser.add_argument("--show",action="store_true",help="Display the saved computational timing plot.")
    args=parser.parse_args()
    return run(args.year,args.steps_per_hour,args.output,args.fail_fast,args.min_observation_sigma,
               show=args.show)


if __name__=="__main__": main()
