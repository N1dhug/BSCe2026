"""Fit ARM (WM4) at 00:00 on selected 2022 sample days and score IM-style marginal NLL.

Historical fit: six preceding corresponding weekdays, weekly-temperature baseline,
joint deterministic QP, and historical free-running variance QP.
Target state: full-24-hour [F0,R0] oracle, as in selected_bm_dates_arm.
The target marginal composite NLL uses unfiltered Gaussian observation densities,
not a sequential conditional likelihood.

Run from repository root:
    python -m src.exe.run_model.wm_4.selected_sample_nll
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.exe.run_model.wm_4.moving_horizon_common import group_frame
from src.exe.run_model.wm_4.predict_all_moving_horizon import historical_weekdays
from src.exe.run_model.wm_4.selected_bm_dates_arm import fit_origin
from src.exe.run_model.wm_4.sweep_weekday_horizon import historical_calendar_days
from src.tools.impute_hourly import impute_hourly_days


ROOT=Path(__file__).resolve().parents[4]
OUTPUT=ROOT/"files"/"output"/"wm_4"/"selected_sample_nll"
N,K,BASELINE,GROUP=6,24,"weekly_temperature","ALL"
DAYS=("04-11","04-19","04-27","05-05","05-13","05-21",
      "05-29","05-30","06-07","08-10","08-18","08-26")


def require_target(frame,day):
    """Never impute target observations; all 24 actual hourly samples must exist."""
    block=frame.reindex(pd.date_range(day,periods=K,freq="h"))[["I","U","Temp"]]
    if block.isna().any().any():
        missing=block.isna().sum()
        raise ValueError(f"Incomplete target day: missing hourly I={missing['I']}, "
                         f"U={missing['U']}, Temp={missing['Temp']}.")


def score_day(frame,day,steps_per_hour):
    history=historical_weekdays(day,N)
    required=pd.DatetimeIndex([date for previous in history
                               for date in historical_calendar_days([previous],K)]).unique().sort_values()
    # Historical isolated missing hours may be imputed using the project's existing
    # procedure, but completely absent historical days are forbidden.
    training,imputation=impute_hourly_days(
        frame,days=required,columns=("I","U","Temp"),require_days=True)
    require_target(frame,day)
    result=fit_origin(frame,training,day,history,BASELINE,steps_per_hour)
    y=np.asarray(result["observed"],dtype=float)
    mu=np.asarray(result["predicted"],dtype=float)
    var=np.asarray(result["variance"],dtype=float)
    if any(v.shape!=(K,) or not np.isfinite(v).all() for v in (y,mu,var)):
        raise ValueError("Expected 24 finite observations, predictions, and variances.")
    if np.any(var<=0):
        raise ValueError("Gaussian NLL is undefined for zero fitted predictive variance; "
                         "do not silently replace fitted noise parameters.")
    # IM's MLE target score is the sum of NEGATIVE marginal log densities.
    # ARM's latent state is Gaussian and observations have variance
    # tau^2 + sigma^2 K(t), so no Fokker--Planck integration is necessary.
    hourly_nll=-norm.logpdf(y,loc=mu,scale=np.sqrt(var))
    if not np.isfinite(hourly_nll).all():
        raise ValueError("Nonfinite marginal NLL.")
    return result,history,imputation,hourly_nll


def run(year=2022,steps_per_hour=60,output=OUTPUT,fail_fast=False):
    if steps_per_hour<2: raise ValueError("steps_per_hour must be >= 2.")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    frame=group_frame(GROUP)
    daily,hourly,skipped=[],[],[]
    print(f"ARM / WM4 | group={GROUP} | baseline={BASELINE} | N={N} historical weekdays "
          f"| k={K} hourly observations | 00:00 origins",flush=True)
    for number,md in enumerate(DAYS,1):
        day=pd.Timestamp(f"{year}-{md}")
        try:
            result,history,imputation,nll=score_day(frame,day,steps_per_hour)
            total=float(np.sum(nll))
            mean=float(np.mean(nll))
            row={
                "day":str(day.date()),"origin_hour":0,"baseline":BASELINE,
                "training_weekdays":N,"horizon_hours":K,"observations":K,
                "target_marginal_nll":total,"target_marginal_nll_per_observation":mean,
                "rmse":float(result["demand_metrics"]["rmse"]),
                "oracle_F0":float(result["initial"][0]),
                "oracle_R0":float(result["initial"][1]),
                "theta_F":float(result["theta"][0]),
                "theta_U":float(result["theta"][1]),
                "theta_R":float(result["theta"][2]),
                "tau":float(np.sqrt(result["tau2"])),
                "sigma":float(np.sqrt(result["sigma2"])),
                "historical_days":";".join(str(d.date()) for d in history),
                "imputed_historical_hourly_slots":int(imputation["imputed_hourly_slots"]),
            }
            daily.append(row)
            for hour in range(K):
                hourly.append({
                    "day":str(day.date()),"hour":hour,
                    "observed":float(result["observed"][hour]),
                    "predicted":float(result["predicted"][hour]),
                    "predicted_variance":float(result["variance"][hour]),
                    "marginal_nll":float(nll[hour]),
                })
            print(f"[{number:02d}/{len(DAYS)}] {day:%Y-%m-%d} | "
                  f"NLL={total:+.8f} | NLL/hour={mean:+.8f} | "
                  f"RMSE={row['rmse']:.6f}",flush=True)
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
        "fit":"historical joint drift QP and historical free-running variance QP",
        "target_state":"full-target [F0,R0] oracle",
        "score":"negative sum of unfiltered Gaussian marginal observation log densities",
        "warning":"oracle conditions on target observations; not an out-of-sample forecast",
    },indent=2),encoding="utf-8")
    print(f"\nCompleted {len(daily)}/{len(DAYS)} days; skipped {len(skipped)}.")
    if daily:
        total=float(sum(row["target_marginal_nll"] for row in daily))
        print(f"Pooled marginal NLL={total:+.8f}; "
              f"mean per observation={total/(K*len(daily)):+.8f} "
              f"({K*len(daily)} hourly observations).")
    print(f"Results: {output}",flush=True)
    if not daily: raise RuntimeError("No sample day could be scored; see skipped_days.csv.")
    return pd.DataFrame(daily)


def main():
    parser=ArgumentParser(description="Score 12 selected ARM sample days with IM-compatible marginal NLL.")
    parser.add_argument("--year",type=int,default=2022)
    parser.add_argument("--steps-per-hour",type=int,default=60)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--fail-fast",action="store_true")
    args=parser.parse_args()
    return run(args.year,args.steps_per_hour,args.output,args.fail_fast)


if __name__=="__main__": main()
