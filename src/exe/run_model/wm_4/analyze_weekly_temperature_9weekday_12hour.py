"""WM4 N=9/H=12 weekly-temperature: hour-by-hour pooled oracle RMSE.

Historical theta: joint QP on nine preceding corresponding weekdays with
one fitted F0 and observed R0 per historical day. For the target day,
jointly optimize BOTH oracle F0 and R0 over the entire 12-hour horizon,
ONCE. Accumulate prefix RMSE for hours [00,01),...,[11,12) from that
single fixed 12-hour oracle, alongside the historical temperature baseline.

This is retrospective state fitting, not out-of-sample forecasting.
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.moving_horizon_common import group_frame
from src.exe.run_model.wm_4.sweep_weekday_horizon import evaluate_configuration


ROOT=Path(__file__).resolve().parents[4]
OUTPUT=ROOT/"files"/"output"/"wm_4"/"weekly_temperature_9weekday_12hour"
BASELINE,N,H="weekly_temperature",9,12
VERSION="wm4_weekly_temp_N9_H12_full_state_fixed_oracle_v1"
FILES={"daily":"daily_metrics.csv","hourly":"hourly_predictions.csv",
       "skip":"skipped_days.csv"}


def convert_day(score,components):
    """Rebuild the exact 12-hour oracle using saved zero-state sensitivities."""
    if str(score["baseline"])!=BASELINE or int(score["training_weekdays"])!=N or int(score["horizon_hours"])!=H:
        raise ValueError("Expected the weekly_temperature N=9, H=12 full-state oracle.")
    if not all(key in score for key in ("oracle_F0","oracle_R0","mse","baseline_mse")):
        raise ValueError("Source score lacks full-state oracle parameters.")
    day=pd.Timestamp(score["day"]).normalize()
    hours=pd.DataFrame(components).sort_values("hour")
    hours=hours.loc[hours["hour"].isin(range(H))].copy()
    if len(hours)!=H or not np.array_equal(hours["hour"].to_numpy(dtype=int),np.arange(H)):
        raise ValueError(f"{day:%Y-%m-%d}: incomplete 00–12 hourly oracle components.")
    needed=("observed","baseline_value","prediction_at_initial_zero","sensitivity_F0","sensitivity_R0")
    if any(name not in hours for name in needed):
        raise ValueError("Source components lack full-state sensitivities.")
    if not np.isfinite(hours[list(needed)].to_numpy(dtype=float)).all():
        raise ValueError("Source components have nonfinite inputs.")
    F0,R0=float(score["oracle_F0"]),float(score["oracle_R0"])
    predicted=(hours["prediction_at_initial_zero"].to_numpy(dtype=float)
               +F0*hours["sensitivity_F0"].to_numpy(dtype=float)
               +R0*hours["sensitivity_R0"].to_numpy(dtype=float))
    observed=hours["observed"].to_numpy(dtype=float)
    baseline=hours["baseline_value"].to_numpy(dtype=float)
    error,base_error=predicted-observed,baseline-observed
    mse,base_mse=float(np.mean(error**2)),float(np.mean(base_error**2))
    if not np.isclose(mse,float(score["mse"]),atol=1e-9,rtol=1e-6):
        raise ValueError(f"{day:%Y-%m-%d}: reconstructed 12-hour oracle disagrees with saved MSE.")
    if not np.isclose(base_mse,float(score["baseline_mse"]),atol=1e-9,rtol=1e-6):
        raise ValueError(f"{day:%Y-%m-%d}: reconstructed baseline disagrees with saved MSE.")
    summary={
        "day":day,"baseline":BASELINE,"training_weekdays":N,"horizon_hours":H,
        "oracle_F0":F0,"oracle_R0":R0,
        "theta_F":float(score["theta_F"]),"theta_R":float(score["theta_R"]),
        "theta_U":float(score["theta_U"]),
        "rmse":float(np.sqrt(mse)),"mse":mse,
        "baseline_rmse":float(np.sqrt(base_mse)),"baseline_mse":base_mse,
        "rmse_gain":float(np.sqrt(base_mse)-np.sqrt(mse)),
    }
    if "historical_R0" in score: summary["historical_R0"]=float(score["historical_R0"])
    hourly=[{"day":day,"hour":h,"hour_end":h+1,
             "observed":float(observed[h]),"predicted":float(predicted[h]),
             "baseline_prediction":float(baseline[h]),
             "error":float(error[h]),"baseline_error":float(base_error[h]),
             "oracle_F0":F0,"oracle_R0":R0}
            for h in range(H)]
    return summary,hourly


def summarize(daily,hourly):
    """Fixed-state prefix RMSE pooled over identical days for all 12 lead hours."""
    d,h=daily.copy(),hourly.copy()
    if d.empty or h.empty: raise ValueError("No complete N9/H12 predictions are available.")
    if not {"day","mse","baseline_mse","oracle_F0","oracle_R0"}.issubset(d):
        raise ValueError("Missing daily full-state oracle fields.")
    if not {"day","hour","error","baseline_error"}.issubset(h):
        raise ValueError("Missing hourly forecast residual fields.")
    d["day"],h["day"]=pd.to_datetime(d["day"]),pd.to_datetime(h["day"])
    if d["day"].duplicated().any() or h.duplicated(["day","hour"]).any():
        raise ValueError("Duplicate prediction day or hour.")
    if set(d["day"])!=set(h["day"]): raise ValueError("Daily and hourly cohorts differ.")
    if len(h)!=len(d)*H: raise ValueError("Every day must have exactly twelve forecast observations.")
    if not all(set(g["hour"].astype(int))==set(range(H)) for _,g in h.groupby("day")):
        raise ValueError("Prediction-hour coverage must be 00 through 11 on every date.")
    if not np.isfinite(h[["error","baseline_error"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite hourly prediction errors.")
    errors=h.pivot(index="day",columns="hour",values="error").reindex(index=d["day"],columns=range(H)).to_numpy(dtype=float)
    baseline=h.pivot(index="day",columns="hour",values="baseline_error").reindex(index=d["day"],columns=range(H)).to_numpy(dtype=float)
    if not np.isfinite(errors).all() or not np.isfinite(baseline).all():
        raise ValueError("Incomplete hourly error matrix.")
    mse=np.mean(errors**2,axis=1)
    bmse=np.mean(baseline**2,axis=1)
    if not np.allclose(mse,d["mse"].to_numpy(dtype=float),atol=1e-9,rtol=1e-6):
        raise ValueError("Stored daily WM4 MSE is inconsistent with hourly residuals.")
    if not np.allclose(bmse,d["baseline_mse"].to_numpy(dtype=float),atol=1e-9,rtol=1e-6):
        raise ValueError("Stored daily baseline MSE is inconsistent with hourly residuals.")
    n=len(d)
    cumulative=np.sqrt(np.cumsum(np.sum(errors**2,axis=0))/(n*np.arange(1,H+1)))
    base_cumulative=np.sqrt(np.cumsum(np.sum(baseline**2,axis=0))/(n*np.arange(1,H+1)))
    hourly_rmse=np.sqrt(np.mean(errors**2,axis=0))
    base_hourly_rmse=np.sqrt(np.mean(baseline**2,axis=0))
    result=pd.DataFrame({
        "start_hour":np.zeros(H,dtype=int),"hour_end":np.arange(1,H+1,dtype=int),
        "elapsed_hours":np.arange(1,H+1,dtype=int),
        "days":np.full(H,n,dtype=int),"samples":n*np.arange(1,H+1),
        "cumulative_rmse":cumulative,"baseline_cumulative_rmse":base_cumulative,
        "cumulative_rmse_gain":base_cumulative-cumulative,
        "cumulative_improvement_pct":np.divide(
            100*(base_cumulative-cumulative),base_cumulative,
            out=np.full(H,np.nan),where=base_cumulative>0),
        "hourly_rmse":hourly_rmse,"baseline_hourly_rmse":base_hourly_rmse,
        "hourly_rmse_gain":base_hourly_rmse-hourly_rmse,
        "cumulative_squared_error":np.cumsum(np.sum(errors**2,axis=0)),
        "baseline_cumulative_squared_error":np.cumsum(np.sum(baseline**2,axis=0)),
    })
    if not np.isclose(result.iloc[-1]["cumulative_rmse"],np.sqrt(np.mean(d["mse"])),atol=1e-12):
        raise RuntimeError("Final cumulative RMSE does not equal pooled twelve-hour RMSE.")
    return result


def plots(prefix,output):
    output=Path(output)
    end=prefix["hour_end"].to_numpy(dtype=int)
    wm=prefix["cumulative_rmse"].to_numpy(dtype=float)
    base=prefix["baseline_cumulative_rmse"].to_numpy(dtype=float)
    ymax=max(1e-6,float(max(np.max(wm),np.max(base)))*1.12)
    fig,ax=plt.subplots(figsize=(13,6))
    ax.plot(end,wm,marker="o",lw=2.2,label="WM4 — joint oracle [F0,R0]")
    ax.plot(end,base,marker="s",lw=1.7,ls="--",label="Weekly-temperature baseline")
    ax.fill_between(end,wm,base,where=(wm<=base),alpha=.14,interpolate=True,
                    label="WM4 improvement")
    ax.set(xlim=(0,H),ylim=(0,ymax),xlabel="End of accumulated interval (hours after 00:00)",
           ylabel="Pooled RMSE (normalized demand)",
           title=f"WM4: N={N} corresponding weekdays, H={H}h | fixed full-state oracle")
    ax.set_xticks(range(H+1))
    ax.grid(alpha=.22);ax.legend()
    fig.tight_layout();fig.savefig(output/"cumulative_rmse_vs_baseline.png",dpi=190);plt.close(fig)

    fig,ax=plt.subplots(figsize=(13,5))
    ax.plot(end,prefix["hourly_rmse"],marker="o",lw=2,label="WM4 hourly RMSE")
    ax.plot(end,prefix["baseline_hourly_rmse"],marker="s",ls="--",label="Baseline hourly RMSE")
    ax.set(xlim=(0,H),ylim=(0,ymax),xlabel="Prediction hour ending at (00:00 + hours)",
           ylabel="Pooled RMSE in individual hour",title="Individual lead-hour RMSE (not cumulative)")
    ax.set_xticks(range(H+1))
    ax.grid(alpha=.22);ax.legend()
    fig.tight_layout();fig.savefig(output/"hourly_rmse_vs_baseline.png",dpi=190);plt.close(fig)

    fig,ax=plt.subplots(figsize=(13,4.5))
    ax.plot(end,prefix["cumulative_improvement_pct"],marker="o",lw=2)
    ax.axhline(0,lw=1,color="black")
    ax.set(xlim=(0,H),xlabel="End of accumulated interval (hours after 00:00)",
           ylabel="RMSE improvement (%)",title="WM4 improvement over weekly-temperature baseline")
    ax.set_xticks(range(H+1));ax.grid(alpha=.22)
    fig.tight_layout();fig.savefig(output/"cumulative_improvement_pct.png",dpi=190);plt.close(fig)


def report(output):
    output=Path(output)
    daily=pd.read_csv(output/FILES["daily"],parse_dates=["day"])
    hourly=pd.read_csv(output/FILES["hourly"],parse_dates=["day"])
    prefix=summarize(daily,hourly)
    prefix.to_csv(output/"cumulative_rmse.csv",index=False)
    plots(prefix,output)
    print(f"\nWM4 | baseline={BASELINE} | N={N} historical matching weekdays | H={H}h")
    print(f"Eligible prediction days: {daily['day'].nunique()}; twelve-hour observations: {len(hourly)}")
    print("Joint oracle [F0,R0] fitted once per date using all 12 target observations.")
    print("Prefix values reuse that one optimized state; they are NOT independently reoptimized.")
    print(prefix[["hour_end","days","samples","cumulative_rmse",
                  "baseline_cumulative_rmse","cumulative_rmse_gain",
                  "cumulative_improvement_pct"]].to_string(index=False,
                  float_format=lambda v:f"{v:.8f}"))
    final=prefix.iloc[-1]
    print(f"\nFull 12-hour pooled RMSE: WM4={final.cumulative_rmse:.8f}; "
          f"baseline={final.baseline_cumulative_rmse:.8f}; "
          f"improvement={final.cumulative_improvement_pct:.2f}%")
    print(f"Saved CSV tables and three figures to {output}")
    print("WARNING: oracle states use target observations; results are retrospective, not out-of-sample forecasts.")
    return prefix


def from_sweep(source,start_day=None,end_day=None,max_days=0):
    """Reuse existing full-state N9/H12 sweep results without refitting theta."""
    folder=Path(source)
    if folder.is_file(): folder=folder.parent
    scores=pd.read_csv(folder/"oracle_horizon_scores.csv",parse_dates=["day"])
    components=pd.read_csv(folder/"forecast_components.csv",parse_dates=["day"])
    conditions=((scores["baseline"]==BASELINE)&(scores["training_weekdays"]==N)
                &(scores["horizon_hours"]==H))
    selected=scores.loc[conditions].sort_values("day").copy()
    if start_day: selected=selected.loc[selected["day"]>=pd.Timestamp(start_day)]
    if end_day: selected=selected.loc[selected["day"]<=pd.Timestamp(end_day)]
    if max_days: selected=selected.head(max_days)
    if selected.empty: raise ValueError("No saved weekly_temperature N9/H12 oracle scores found.")
    if selected["day"].duplicated().any(): raise ValueError("Duplicate N9/H12 scores by date.")
    if "oracle_R0" not in selected or "sensitivity_R0" not in components:
        raise ValueError("Saved sweep is obsolete F0-only output; rerun the full-state sweep.")
    subset=components.loc[(components["baseline"]==BASELINE)&
                          (components["training_weekdays"]==N)&
                          (components["day"].isin(selected["day"]))].copy()
    daily,hourly=[],[]
    for record in selected.to_dict("records"):
        matching=subset.loc[subset["day"]==record["day"]].to_dict("records")
        d,h=convert_day(record,matching)
        daily.append(d);hourly.extend(h)
    return daily,hourly


def fresh(group="MED",steps_per_hour=4,start_day=None,end_day=None,
          max_days=0,progress_every=10,fail_fast=False):
    frame=group_frame(group)
    first=max(pd.Timestamp("2022-01-01")+pd.Timedelta(weeks=N),
              frame.index.min().normalize()+pd.Timedelta(weeks=N))
    last=min(pd.Timestamp("2022-10-31"),frame.index.max().normalize())
    if start_day: first=max(first,pd.Timestamp(start_day))
    if end_day: last=min(last,pd.Timestamp(end_day))
    days=list(pd.date_range(first,last,freq="D"))
    if max_days: days=days[:max_days]
    if not days: raise ValueError("No dates available after the nine-week warmup.")
    daily,hourly,skipped=[],[],[]
    for i,day in enumerate(days,start=1):
        try:
            fit,scores,components,_=evaluate_configuration(
                frame,day,N,steps_per_hour,H,H,BASELINE)
            record,observations=convert_day(scores[0],components)
            record["training_days"]=fit["training_days"]
            record["fit_qp_loss"]=float(fit["fit_qp_loss"])
            daily.append(record);hourly.extend(observations)
        except (RuntimeError,ValueError,KeyError,np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append({"day":day,"reason":str(exc)})
        if i%progress_every==0:
            print(f"[WM4 N9/H12] {i}/{len(days)} days, "
                  f"{len(daily)} successful, {len(skipped)} skipped",flush=True)
    return daily,hourly,skipped


def main():
    parser=ArgumentParser(description="WM4 weekly_temperature N9/H12 cumulative oracle and baseline RMSE.")
    parser.add_argument("--group",choices=("MED","ALL"),default="MED")
    parser.add_argument("--from-sweep",default=None,
                        help="Path to existing expanded weekday-horizon sweep folder (no refitting).")
    parser.add_argument("--steps-per-hour",type=int,default=4)
    parser.add_argument("--start-day",default=None)
    parser.add_argument("--end-day",default=None)
    parser.add_argument("--max-days",type=int,default=0)
    parser.add_argument("--progress-every",type=int,default=10)
    parser.add_argument("--output",default=None)
    parser.add_argument("--report-only",action="store_true")
    parser.add_argument("--fail-fast",action="store_true")
    args=parser.parse_args()
    if args.steps_per_hour<2 or args.progress_every<1 or args.max_days<0:
        parser.error("Require steps-per-hour >=2, progress-every>=1 and max-days>=0.")
    output=Path(args.output) if args.output else OUTPUT/args.group.lower()
    if args.report_only:
        return report(output)
    output.mkdir(parents=True,exist_ok=True)
    if args.from_sweep:
        daily,hourly=from_sweep(args.from_sweep,args.start_day,args.end_day,args.max_days)
        skipped=[]
    else:
        daily,hourly,skipped=fresh(args.group,args.steps_per_hour,args.start_day,
                                   args.end_day,args.max_days,args.progress_every,args.fail_fast)
    if not daily:
        raise RuntimeError("No successful full-state oracle fits for this N9/H12 experiment.")
    pd.DataFrame(daily).to_csv(output/FILES["daily"],index=False)
    pd.DataFrame(hourly).to_csv(output/FILES["hourly"],index=False)
    pd.DataFrame(skipped,columns=["day","reason"]).to_csv(output/FILES["skip"],index=False)
    (output/"configuration.json").write_text(json.dumps({
        "version":VERSION,"group":args.group,"baseline":BASELINE,
        "historical_corresponding_weekdays":N,"horizon_hours":H,
        "source_sweep":args.from_sweep,"steps_per_hour":args.steps_per_hour,
        "start_day":args.start_day,"end_day":args.end_day,
        "max_days":args.max_days,"oracle":"joint F0,R0 optimized once over all twelve target observations",
    },indent=2),encoding="utf-8")
    return report(output)


if __name__=="__main__": main()
