"""ARM/WM4 NLL for twelve 2022 sample days, N=6, k=24, 00:00 origins.

The observation at hour 0 is treated as conditioning/initialization information:
it is excluded from BOTH historical stochastic variance estimation and target
likelihood scores. N=6 historical weekdays each contribute hours 1,...,23
(138 variance observations); each target date contributes 23 likelihood terms.

Historical drift parameters and the full-target [F0,R0] retrospective oracle
remain unchanged. Variance is re-estimated from the same historical free-running
ARM trajectories using only residuals at hours 1,...,23. Nonnegative tau^2 and
sigma^2 remain admissible. An optional observation-noise floor can be requested
with --min-observation-sigma 0.001 but is disabled by default.

Sequential score: Gaussian Kalman innovations with observation updates from
hour 1 onward (analogous to IM's filtered MLE likelihood).
Marginal score: unfiltered open-loop Gaussian likelihood for hours 1,...,23.
No update or density evaluation is made using the observation at hour 0.

Note: the oracle target state uses the target observations, so reported scores
are retrospective oracle-conditioned, not operational forecast performance.

Run:
    python -m src.exe.run_model.wm_4.selected_sample_nll
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.exe.run_model.wm_4.fit_moving_horizon_diffusion import fit_variances, variance_kernel
from src.exe.run_model.wm_4.moving_horizon_common import group_frame
from src.exe.run_model.wm_4.predict_all_moving_horizon import historical_weekdays
from src.exe.run_model.wm_4.selected_bm_dates_arm import fit_origin
from src.exe.run_model.wm_4.sweep_weekday_horizon import historical_calendar_days
from src.tools.impute_hourly import impute_hourly_days


ROOT=Path(__file__).resolve().parents[4]
OUTPUT=ROOT/"files"/"output"/"wm_4"/"selected_sample_nll"
N,K,BASELINE,GROUP=6,24,"weekly_temperature","ALL"
DEFAULT_MIN_TAU=0.0  # No variance floor needed when scoring begins at hour 1.
DAYS=("04-11","04-19","04-27","05-05","05-13","05-21",
      "05-29","05-30","06-07","08-10","08-18","08-26")


def require_target(frame,day):
    """Target observations must be measured, not imputed."""
    block=frame.reindex(pd.date_range(day,periods=K,freq="h"))[["I","U","Temp"]]
    if block.isna().any().any():
        missing=block.isna().sum()
        raise ValueError(f"Incomplete target day: missing hourly I={missing['I']}, "
                         f"U={missing['U']}, Temp={missing['Temp']}.")


def historical_errors(result,steps_per_hour):
    """Reuse the exact six historical free-running trajectories fitted by ARM."""
    idx=np.arange(K,dtype=int)*int(steps_per_hour)
    inputs,states=result["historical_inputs"],result["historical_states"]
    if len(inputs)!=N or len(states)!=N:
        raise ValueError("Expected exactly six historical training intervals.")
    errors=np.stack([np.asarray(d["I"],dtype=float)[idx]-
                     np.asarray(s["I"],dtype=float)[idx]
                     for d,s in zip(inputs,states)])
    if errors.shape!=(N,K) or not np.isfinite(errors).all():
        raise ValueError("Historical training residuals must have shape (6,24) and be finite.")
    return errors


def positive_variances(result,steps_per_hour,min_tau=DEFAULT_MIN_TAU):
    """Refit historical variance QP from hours 1,...,23 only.

    Retain the existing nonnegative least-squares estimator, but do NOT use
    the six mechanically zero hour-0 residuals. If the optional observation
    standard-deviation lower bound is active, solve the same convex QP on
    its feasible boundary. Never use the old hour-0-inclusive estimates.
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


def score_likelihoods(observed,open_mean,theta,tau2,sigma2,steps_per_hour):
    """Score hours 1,...,23 conditional on a fixed initial state at hour 0.

    The hourly Kalman filter propagates the unobserved state from t=0 to
    t=1 BEFORE its first score and update. It never scores or assimilates
    the initialization observation at t=0. Return 23-element arrays.
    """
    y=np.asarray(observed,dtype=float)
    mu=np.asarray(open_mean,dtype=float)
    theta=np.asarray(theta,dtype=float)
    sph=int(steps_per_hour)
    if y.shape!=(K,) or mu.shape!=(K,) or theta.shape!=(3,):
        raise ValueError("Expected 24 observations/means and three ARM drift parameters.")
    if sph<2 or not np.isfinite(np.r_[y,mu,theta,tau2,sigma2]).all():
        raise ValueError("Invalid ARM input data or parameters for NLL.")
    if tau2<0 or sigma2<0:
        raise ValueError("Gaussian NLL requires nonnegative observation and process variances.")

    kernel=variance_kernel(float(theta[0]),float(theta[2]),sph,K)
    variance=tau2+sigma2*kernel[1:]
    if not np.isfinite(variance).all() or np.any(variance<=0):
        raise ValueError("Zero/nonfinite predictive variance at a scored hour (1..23); "
                         "historical fit cannot define a Gaussian likelihood.")
    marginal=-norm.logpdf(y[1:],loc=mu[1:],scale=np.sqrt(variance))
    dt=1./(24*sph)
    a=np.array([[1.,dt],[dt*float(theta[0]),1.+dt*float(theta[2])]])
    h=np.array([0.,1.])
    ident=np.eye(2)
    process=np.array([[0.,0.],[0.,sigma2*dt]])
    shift=np.zeros(2)
    covariance=np.zeros((2,2))
    filtered=np.empty(K-1,dtype=float)
    sequential=np.empty(K-1,dtype=float)
    for hour in range(1,K):
        # Propagate one physical hour from the initial state / previous
        # posterior. The initialization observation is never assimilated.
        for _ in range(sph):
            shift=a@shift
            covariance=a@covariance@a.T+process
        covariance=0.5*(covariance+covariance.T)
        filtered[hour-1]=mu[hour]+shift[1]
        s=float(covariance[1,1]+tau2)
        if not np.isfinite(s) or s<=0:
            raise ValueError(f"Invalid Kalman innovation variance at hour {hour}.")
        innovation=float(y[hour]-filtered[hour-1])
        sequential[hour-1]=0.5*(np.log(2.*np.pi*s)+innovation**2/s)
        gain=covariance[:,1]/s
        shift=shift+gain*innovation
        update=ident-np.outer(gain,h)
        covariance=update@covariance@update.T+tau2*np.outer(gain,gain)
    if not np.isfinite(marginal).all() or not np.isfinite(sequential).all():
        raise ValueError("Nonfinite ARM likelihood on hours 1..23.")
    if sigma2==0 and not np.allclose(sequential,marginal,atol=1e-9,rtol=1e-9):
        raise RuntimeError("Kalman and marginal scores differ with zero process noise.")
    return marginal,sequential,variance,filtered


def score_day(frame,day,steps_per_hour,min_tau):
    history=historical_weekdays(day,N)
    required=historical_calendar_days(history,K)
    training,imputation=impute_hourly_days(
        frame,days=required,columns=("I","U","Temp"),require_days=True)
    require_target(frame,day)
    result=fit_origin(frame,training,day,history,BASELINE,steps_per_hour)
    tau2,sigma2,regularized=positive_variances(result,steps_per_hour,min_tau)
    marginal,sequential,variance,filtered=score_likelihoods(
        result["observed"],result["predicted"],result["theta"],
        tau2,sigma2,steps_per_hour)
    return result,history,imputation,marginal,sequential,variance,filtered,tau2,sigma2,regularized


def run(year=2022,steps_per_hour=60,output=OUTPUT,fail_fast=False,min_tau=DEFAULT_MIN_TAU):
    if steps_per_hour<2: raise ValueError("steps_per_hour must be >= 2.")
    if not np.isfinite(min_tau) or min_tau<0: raise ValueError("min_tau must be nonnegative.")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    frame=group_frame(GROUP)
    daily,hourly,skipped=[],[],[]
    print(f"ARM / WM4 | group={GROUP} | baseline={BASELINE} | N={N} historical weekdays "
          f"| k={K} hourly observations | 00:00 origins",flush=True)
    print(f"Scored hours: 01:00..23:00 only (23/day); hour 00:00 conditions the initial state. "
          f"Historical variance fit: {N*(K-1)} observations (6 x 23). "
          f"Optional minimum tau={min_tau:g}.",flush=True)
    for number,md in enumerate(DAYS,1):
        day=pd.Timestamp(f"{year}-{md}")
        try:
            result,history,imputation,marginal,sequential,variance,filtered,tau2,sigma2,regularized=score_day(
                frame,day,steps_per_hour,min_tau)
            total=float(np.sum(sequential))
            open_total=float(np.sum(marginal))
            row={
                "day":str(day.date()),"origin_hour":0,"baseline":BASELINE,
                "training_weekdays":N,"horizon_hours":K,"observations":K-1,
                "target_sequential_nll":total,"target_sequential_nll_per_observation":total/(K-1),
                "target_marginal_nll":open_total,"target_marginal_nll_per_observation":open_total/(K-1),
                "rmse":float(np.sqrt(np.mean(np.square(
                    np.asarray(result["observed"][1:])-np.asarray(result["predicted"][1:]))))),
                "rmse_full_day":float(result["demand_metrics"]["rmse"]),
                "oracle_F0":float(result["initial"][0]),
                "oracle_R0":float(result["initial"][1]),
                "theta_F":float(result["theta"][0]),
                "theta_U":float(result["theta"][1]),
                "theta_R":float(result["theta"][2]),
                "original_tau":float(np.sqrt(result["tau2"])),
                "original_sigma":float(np.sqrt(result["sigma2"])),
                "tau":float(np.sqrt(tau2)),
                "sigma":float(np.sqrt(sigma2)),
                "observation_variance_regularized":bool(regularized),
                "historical_days":";".join(str(d.date()) for d in history),
                "imputed_historical_hourly_slots":int(imputation["imputed_hourly_slots"]),
            }
            daily.append(row)
            for hour in range(1,K):
                j=hour-1
                hourly.append({
                    "day":str(day.date()),"hour":hour,
                    "observed":float(result["observed"][hour]),
                    "open_loop_mean":float(result["predicted"][hour]),
                    "filtered_one_step_mean":float(filtered[j]),
                    "open_loop_variance":float(variance[j]),
                    "marginal_nll":float(marginal[j]),
                    "sequential_nll":float(sequential[j]),
                })
            adjusted=" [tau lower-bound active]" if regularized else ""
            print(f"[{number:02d}/{len(DAYS)}] {day:%Y-%m-%d} | "
                  f"sequential NLL={total:+.8f} | marginal NLL={open_total:+.8f} | "
                  f"RMSE={row['rmse']:.6f} | tau={row['tau']:.6g} | sigma={row['sigma']:.6g}"
                  f"{adjusted}",flush=True)
        except (ValueError,RuntimeError,KeyError,np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append({"day":str(day.date()),"reason":str(exc)})
            print(f"[{number:02d}/{len(DAYS)}] {day:%Y-%m-%d} | SKIPPED: {exc}",flush=True)
    pd.DataFrame(daily).to_csv(output/"per_day_nll.csv",index=False)
    pd.DataFrame(hourly).to_csv(output/"hourly_nll.csv",index=False)
    pd.DataFrame(skipped,columns=("day","reason")).to_csv(output/"skipped_days.csv",index=False)
    (output/"configuration.json").write_text(json.dumps({
        "year":year,"days":list(DAYS),"group":GROUP,"baseline":BASELINE,
        "N":N,"k":K,"origin_hour":0,"steps_per_hour":steps_per_hour,
        "minimum_observation_sigma":min_tau,
        "fit":"historical joint drift QP; variance QP refitted from only historical hours 1..23",
        "historical_variance_observations":N*(K-1),
        "scored_hours":list(range(1,K)),
        "variance_regularization":"optional tau2 >= minimum_observation_sigma^2; disabled by default",
        "target_state":"full-target [F0,R0] oracle, unchanged",
        "sequential_score":"Gaussian observation-updating Kalman innovations; t0 neither scored nor filtered",
        "marginal_score":"sum of unfiltered Gaussian marginal negative log densities for hours 1..23; compare with IM on identical hours",
        "warning":"Target initial state uses future target observations; retrospective oracle, not out-of-sample",
    },indent=2),encoding="utf-8")
    print(f"\nCompleted {len(daily)}/{len(DAYS)} days; skipped {len(skipped)}.")
    if daily:
        seq=float(sum(row["target_sequential_nll"] for row in daily))
        mar=float(sum(row["target_marginal_nll"] for row in daily))
        print(f"Pooled sequential NLL={seq:+.8f}; "
              f"mean/observation={seq/((K-1)*len(daily)):+.8f}.")
        print(f"Pooled marginal NLL={mar:+.8f}; "
              f"mean/observation={mar/((K-1)*len(daily)):+.8f}.")
        print(f"Observation-noise floor active on {sum(r['observation_variance_regularized'] for r in daily)} day(s).")
    print(f"Results: {output}",flush=True)
    if not daily: raise RuntimeError("No sample day could be scored; see skipped_days.csv.")
    return pd.DataFrame(daily)


def main():
    parser=ArgumentParser(description="Compare ARM target marginal and IM-MLE-style sequential NLL on 12 selected days.")
    parser.add_argument("--year",type=int,default=2022)
    parser.add_argument("--steps-per-hour",type=int,default=60)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--min-observation-sigma",type=float,default=DEFAULT_MIN_TAU,
                        help="Optional tau lower bound; default 0 (original nonnegative QP).")
    parser.add_argument("--fail-fast",action="store_true")
    args=parser.parse_args()
    return run(args.year,args.steps_per_hour,args.output,args.fail_fast,args.min_observation_sigma)


if __name__=="__main__": main()
