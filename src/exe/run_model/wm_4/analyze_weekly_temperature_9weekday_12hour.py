"""WM4 N=9, H=12: pooled prefix RMSE at TWELVE distinct forecast origins.

Origins 00,...,11 predict [origin,origin+12) on the same calendar day.
A single historical joint QP estimates theta from nine corresponding
weekdays; each origin independently fits BOTH F0 and R0 against its
entire twelve-hour target window by the existing convex state oracle.

Every subplot shares the absolute 00–24 time axis and y-axis scale.
The prefix RMSE on one subplot is computed WITHOUT reoptimizing the
initial state at each successive lead hour.

WARNING: this is an in-sample retrospective full-state oracle.
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.moving_horizon_common import HOURS, group_frame, window_data, metrics
from src.exe.run_model.wm_4.predict_all_moving_horizon import historical_weekdays
from src.exe.run_model.wm_4.sweep_weekday_horizon import baseline_from_history
from src.model.wm_4 import wm_4
from src.params.wm_4 import joint_qp_params, optimized_qp_params
from src.tools.impute_hourly import impute_hourly_days


ROOT=Path(__file__).resolve().parents[4]
OUTPUT=ROOT/"files"/"output"/"wm_4"/"weekly_temperature_9weekday_12hour"
BASELINE,N,H="weekly_temperature",9,12
ORIGINS=tuple(range(12))  # Exactly TWELVE panels: [00,12), ... [11,23).
VERSION="wm4_weekly_temp_N9_H12_twelve_rolling_origins_full_state_v2"
FILES={"daily":"origin_daily_metrics.csv","hourly":"origin_hourly_predictions.csv",
       "skip":"skipped_days.csv"}


def fit_origin(model, data, observed, theta, steps_per_hour):
    """Solve one [F0,R0] convex oracle QP over ALL twelve hourly observations."""
    sph=int(steps_per_hour)
    indices=np.arange(H)*sph
    target=np.asarray(observed,dtype=float).reshape(-1)
    if target.shape!=(H,) or not np.isfinite(target).all():
        raise ValueError("An origin must contain twelve finite target observations.")
    subset={key:np.asarray(data[key],dtype=float).copy()
            for key in ("I","U","PE","Time","O")}
    if any(v.shape!=(model.t_steps,) or not np.isfinite(v).all() for v in subset.values()):
        raise ValueError("Origin numerical data must contain finite values at the model resolution.")
    if not np.allclose(subset["I"][indices],target,atol=1e-10,rtol=1e-10):
        raise ValueError("Origin hourly observations do not match the underlying numerical data.")
    mask=np.zeros(model.t_steps,dtype=bool);mask[indices]=True
    subset["observation_mask"]=mask
    subset["transition_mask"]=np.ones(model.t_steps-1,dtype=bool)
    result=optimized_qp_params(model).optimize_prediction_initial_state(
        subset,theta,f_bounds=(-1.0,1.0),r_bounds=(-1.0,1.0))
    initial=np.asarray(result["init_cond"],dtype=float)
    predicted=np.asarray(result["deterministic_state"]["I"],dtype=float)[indices]
    baseline=np.asarray(subset["O"],dtype=float)[indices]
    if initial.shape!=(2,) or not np.isfinite(initial).all():
        raise RuntimeError("The joint-state oracle did not return both F0 and R0.")
    if not np.isclose(float(np.mean((predicted-target)**2)),result["loss"],
                      atol=1e-12,rtol=1e-9):
        raise RuntimeError("Oracle trajectory and fitted QP loss disagree.")
    return initial,predicted,baseline


def evaluate_day(frame,day,steps_per_hour=4,reference=None):
    """Fit theta ONCE, then evaluate ALL twelve 12h origins for one day."""
    day=pd.Timestamp(day).normalize();sph=int(steps_per_hour)
    if sph<2: raise ValueError("steps_per_hour must be >=2.")
    history=historical_weekdays(day,N)
    training,imputation=impute_hourly_days(frame,days=history,
                                           columns=("I","U","Temp"),require_days=True)
    baseline=baseline_from_history(training,history,BASELINE)
    full_model,target,_,observed=window_data(frame,day,baseline,sph)
    history_data=[window_data(training,past,baseline,sph)[1] for past in history]
    if reference is None:
        fitted=joint_qp_params(full_model).fit_many(history_data)
        theta=np.asarray(fitted["r"],dtype=float)
        fit_loss=float(fitted["qp_loss"])
    else:
        if (int(reference["training_weekdays"])!=N or str(reference["baseline"])!=BASELINE):
            raise ValueError("Sweep parameter record does not match the N=9 weekly-temperature experiment.")
        theta=np.array([reference["theta_F"],reference["theta_U"],reference["theta_R"]],dtype=float)
        fit_loss=float(reference.get("fit_qp_loss",np.nan))
        if "baseline_hour_00" in reference and pd.notna(reference["baseline_hour_00"]):
            original=float(reference["baseline_hour_00"])
            current=float(target["O"][0])
            if not np.isclose(original,current,atol=1e-7,rtol=1e-7):
                raise ValueError("Saved baseline differs from recomputed baseline; verify cohort and input data.")
    if theta.shape!=(3,) or not np.isfinite(theta).all():
        raise ValueError("Nonfinite or incomplete historical theta parameters.")
    model=wm_4((0.0,H/HOURS),H*sph+1,lambda t:0.0)
    if not np.isclose(model.dt,full_model.dt,rtol=1e-12,atol=1e-14):
        raise RuntimeError("Origin simulation does not match the training Euler time step.")

    daily,hourly=[],[]
    for origin in ORIGINS:
        start,end=origin*sph,(origin+H)*sph+1
        subset={key:np.asarray(target[key])[start:end].copy()
                for key in ("I","U","PE","Time","O")}
        subset["Time"]-=float(origin)  # Local elapsed hours from the origin.
        if any(len(value)!=H*sph+1 for value in subset.values()):
            raise RuntimeError(f"Origin {origin} has an incomplete model input window.")
        actual=np.asarray(observed[origin:origin+H],dtype=float)
        initial,pred,base=fit_origin(model,subset,actual,theta,sph)
        s,b=metrics(actual,pred),metrics(actual,base)
        historical_r0=float(np.median([
            float(item["I"][start]-item["O"][start]) for item in history_data]))
        daily.append({
            "day":day,"weekday":day.day_name(),"origin_hour":origin,
            "interval_end_hour":origin+H,"baseline":BASELINE,
            "training_weekdays":N,"horizon_hours":H,
            "training_days":";".join(f"{d:%Y-%m-%d}" for d in history),
            "theta_F":float(theta[0]),"theta_U":float(theta[1]),"theta_R":float(theta[2]),
            "oracle_F0":float(initial[0]),"oracle_R0":float(initial[1]),
            "historical_R0":historical_r0,
            "rmse":s["rmse"],"mse":s["mse"],"mae":s["mae"],"bias":s["bias"],
            "baseline_rmse":b["rmse"],"baseline_mse":b["mse"],
            "rmse_gain":b["rmse"]-s["rmse"],
            "fit_qp_loss":fit_loss,
            "imputed_training_hours":int(imputation["imputed_hourly_slots"]),
        })
        for lead in range(H):
            hour=origin+lead
            hourly.append({
                "day":day,"origin_hour":origin,"lead_hour":lead,
                "absolute_hour":hour,"hour_end":hour+1,
                "timestamp":day+pd.Timedelta(hours=hour),
                "observed":float(actual[lead]),"predicted":float(pred[lead]),
                "baseline_prediction":float(base[lead]),
                "error":float(pred[lead]-actual[lead]),
                "baseline_error":float(base[lead]-actual[lead]),
                "oracle_F0":float(initial[0]),"oracle_R0":float(initial[1]),
            })
    if len(daily)!=len(ORIGINS) or len(hourly)!=len(ORIGINS)*H:
        raise RuntimeError("Not every forecast origin was evaluated.")
    return daily,hourly


def summarize(daily,hourly):
    """Pooled 1..12h prefix RMSE separately at each of 12 origins on matched days."""
    d,h=daily.copy(),hourly.copy()
    required_daily={"day","origin_hour","mse","baseline_mse","oracle_F0","oracle_R0"}
    required_hourly={"day","origin_hour","lead_hour","error","baseline_error"}
    if not required_daily.issubset(d) or not required_hourly.issubset(h):
        raise ValueError("Saved output uses the old midnight-only format or lacks required state/lead fields.")
    if d.empty or h.empty: raise ValueError("No complete rolling-origin observations.")
    d["day"],h["day"]=pd.to_datetime(d["day"]),pd.to_datetime(h["day"])
    if d.duplicated(["day","origin_hour"]).any() or h.duplicated(["day","origin_hour","lead_hour"]).any():
        raise ValueError("Duplicate day/origin/lead records.")
    days=set(d["day"])
    if days!=set(h["day"]) or len(d)!=len(days)*len(ORIGINS) or len(h)!=len(d)*H:
        raise ValueError("Every origin must be evaluated for the same complete prediction-day cohort.")
    if not np.isfinite(d[["mse","baseline_mse","oracle_F0","oracle_R0"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite daily scores or oracle states.")
    if not np.isfinite(h[["error","baseline_error"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite hourly residuals.")

    rows,finals=[],[]
    for origin in ORIGINS:
        values=h.loc[h["origin_hour"]==origin]
        state=d.loc[d["origin_hour"]==origin].sort_values("day")
        if len(state)!=len(days) or len(values)!=len(days)*H:
            raise ValueError(f"Origin {origin} has a different number of evaluation days.")
        if set(state["day"])!=days or set(values["day"])!=days:
            raise ValueError(f"Origin {origin} does not match the common evaluation days.")
        w=values.pivot(index="day",columns="lead_hour",values="error").reindex(
            index=state["day"],columns=range(H))
        b=values.pivot(index="day",columns="lead_hour",values="baseline_error").reindex(
            index=state["day"],columns=range(H))
        if w.isna().any().any() or b.isna().any().any():
            raise ValueError(f"Origin {origin} is missing one or more lead-hour observations.")
        errors=w.to_numpy(dtype=float);reference=b.to_numpy(dtype=float)
        mse=np.mean(errors**2,axis=1)
        base_mse=np.mean(reference**2,axis=1)
        if not np.allclose(mse,state["mse"],atol=1e-10,rtol=1e-8):
            raise ValueError(f"Origin {origin} daily MSE differs from individual hourly errors.")
        if not np.allclose(base_mse,state["baseline_mse"],atol=1e-10,rtol=1e-8):
            raise ValueError(f"Origin {origin} daily baseline MSE differs from individual hourly errors.")
        n=len(days)
        sq=np.sum(errors**2,axis=0)
        bsq=np.sum(reference**2,axis=0)
        prefix=np.sqrt(np.cumsum(sq)/(n*np.arange(1,H+1)))
        bprefix=np.sqrt(np.cumsum(bsq)/(n*np.arange(1,H+1)))
        for lead in range(H):
            hour_end=origin+lead+1
            rows.append({
                "origin_hour":origin,"interval_end_hour":origin+H,
                "lead_hour":lead,"elapsed_hours":lead+1,
                "hour_end":hour_end,"days":n,"samples":n*(lead+1),
                "cumulative_rmse":float(prefix[lead]),
                "baseline_cumulative_rmse":float(bprefix[lead]),
                "rmse_gain":float(bprefix[lead]-prefix[lead]),
                "hourly_rmse":float(np.sqrt(sq[lead]/n)),
                "baseline_hourly_rmse":float(np.sqrt(bsq[lead]/n)),
                "cumulative_squared_error":float(np.sum(sq[:lead+1])),
                "baseline_cumulative_squared_error":float(np.sum(bsq[:lead+1])),
            })
        finals.append({
            "origin_hour":origin,"interval_end_hour":origin+H,
            "days":n,"samples":n*H,"aggregate_rmse":float(prefix[-1]),
            "baseline_rmse":float(bprefix[-1]),
            "rmse_gain":float(bprefix[-1]-prefix[-1]),
            "median_daily_rmse":float(np.median(state["rmse"])),
        })
    return pd.DataFrame(rows),pd.DataFrame(finals)


def plot_origins(cumulative,output):
    """Twelve shared-scale subplots aligned to the absolute 00–24 clock."""
    if len(cumulative)!=len(ORIGINS)*H: raise ValueError("Expected 12 origins × 12 lead hours.")
    vals=cumulative[["cumulative_rmse","baseline_cumulative_rmse"]].to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals<0).any():
        raise ValueError("Invalid pooled cumulative RMSE.")
    ymax=max(1e-6,1.10*float(np.max(vals)))
    fig,axes=plt.subplots(4,3,figsize=(19,16),sharex=True,sharey=True)
    for ax,origin in zip(axes.flat,ORIGINS):
        parts=cumulative.loc[cumulative["origin_hour"]==origin].sort_values("elapsed_hours")
        if not np.array_equal(parts["elapsed_hours"].to_numpy(dtype=int),np.arange(1,H+1)):
            raise ValueError(f"Origin {origin} is missing a cumulative step.")
        x=np.arange(origin,origin+H+1,dtype=float)
        wm=parts["cumulative_rmse"].to_numpy(dtype=float)
        base=parts["baseline_cumulative_rmse"].to_numpy(dtype=float)
        step_wm=np.r_[wm[0],wm]
        step_base=np.r_[base[0],base]
        ax.axvspan(origin,origin+H,color="grey",alpha=.08)
        ax.fill_between(x,0,step_wm,step="pre",color="tab:blue",alpha=.24)
        ax.step(x,step_wm,where="pre",color="tab:blue",lw=1.8,
                label="WM4 full-state oracle (pooled prefix RMSE)")
        ax.step(x,step_base,where="pre",color="tab:orange",ls="--",lw=1.5,
                label="Weekly-temperature baseline")
        ax.plot(x[1:],wm,"o",color="tab:blue",markersize=2.5)
        ax.set(xlim=(0,HOURS),ylim=(0,ymax))
        ax.set_xticks(np.arange(0,HOURS+1,3))
        ax.tick_params(labelbottom=True,labelleft=True,labelsize=8)
        ax.grid(alpha=.17)
        ax.set_title(f"Origin {origin:02d}:00 | interval {origin:02d}–{origin+H:02d}:00",
                     fontsize=10)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.968),ncol=2)
    fig.suptitle("WM4: N=9 corresponding weekdays, H=12h | pooled cumulative RMSE by forecast origin",
                 fontsize=15,y=.994)
    fig.supxlabel("Hour of day (all panels 00–24)")
    fig.supylabel("Pooled cumulative RMSE (normalized demand)")
    fig.tight_layout(rect=(.025,.035,.99,.946))
    fig.savefig(Path(output)/"origin_cumulative_rmse_panels.png",dpi=190,bbox_inches="tight")
    plt.close(fig)


def report(output):
    output=Path(output)
    daily=pd.read_csv(output/FILES["daily"],parse_dates=["day"])
    hourly=pd.read_csv(output/FILES["hourly"],parse_dates=["day"])
    cumulative,finals=summarize(daily,hourly)
    cumulative.to_csv(output/"origin_cumulative_rmse.csv",index=False)
    finals.to_csv(output/"origin_aggregate_rmse.csv",index=False)
    plot_origins(cumulative,output)
    print(f"\nWM4 | {BASELINE} | N={N} historical matching weekdays | H={H} hours")
    print(f"Matched evaluation days: {daily['day'].nunique()}; forecast origins: {len(ORIGINS)}.")
    print("Each origin independently optimizes BOTH [F0,R0] against its entire twelve-hour target interval.")
    print("Each plotted cumulative curve retains that origin's fixed oracle state over all twelve prefix steps.")
    print(finals.to_string(index=False,float_format=lambda x:f"{x:.8f}"))
    print(f"\nSaved 12-panel figure and per-origin cumulative/final RMSE CSVs to {output}")
    print("WARNING: Full-state oracle uses future observations; performance is retrospective, not operational.")
    return cumulative,finals


def load_sweep_params(source,group):
    """Read ONLY historical theta fits; old midnight oracle states CANNOT be shifted."""
    folder=Path(source)
    if folder.is_file(): folder=folder.parent
    configuration=folder/"configuration.json"
    if configuration.is_file():
        settings=json.loads(configuration.read_text(encoding="utf-8"))
        if settings.get("group")!=group:
            raise ValueError("Source sweep group does not match this study's group.")
    table=pd.read_csv(folder/"parameter_estimates.csv",parse_dates=["day"])
    needed={"day","baseline","training_weekdays","theta_F","theta_R","theta_U"}
    if not needed.issubset(table):
        raise ValueError("Saved historical parameter estimates lack required columns.")
    subset=table.loc[(table["baseline"]==BASELINE)&(table["training_weekdays"]==N)].copy()
    if subset.empty: raise ValueError("No weekly_temperature N=9 historical theta estimates in the sweep.")
    if subset["day"].duplicated().any():
        raise ValueError("Source sweep contains duplicate parameter fits for the same day.")
    return subset.sort_values("day").set_index("day")


def run(group="MED",steps_per_hour=4,start_day=None,end_day=None,max_days=0,
        progress_every=10,from_sweep=None,output=None,fail_fast=False):
    dest=Path(output) if output else OUTPUT/group.lower()
    dest.mkdir(parents=True,exist_ok=True)
    frame=group_frame(group)
    first=max(pd.Timestamp("2022-01-01")+pd.Timedelta(weeks=N),
              frame.index.min().normalize()+pd.Timedelta(weeks=N))
    last=min(pd.Timestamp("2022-10-31"),frame.index.max().normalize())
    if start_day: first=max(first,pd.Timestamp(start_day))
    if end_day: last=min(last,pd.Timestamp(end_day))
    if first>last: raise ValueError("No eligible prediction dates after warm-up and date filters.")
    saved=load_sweep_params(from_sweep,group) if from_sweep else None
    dates=[day for day in (saved.index if saved is not None else pd.date_range(first,last,freq="D"))
           if first<=day<=last]
    if max_days: dates=dates[:int(max_days)]
    if not dates: raise ValueError("No target days in the requested period.")
    daily,hourly,skipped=[],[],[]
    for i,day in enumerate(dates,start=1):
        try:
            ref=saved.loc[day] if saved is not None else None
            rows,points=evaluate_day(frame,day,steps_per_hour,ref)
            daily.extend(rows);hourly.extend(points)
        except (ValueError,RuntimeError,KeyError,np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            skipped.append({"day":day,"reason":str(exc)})
        if i%progress_every==0:
            print(f"[WM4 N9/H12 origins] {i}/{len(dates)} dates: "
                  f"{len(daily)//len(ORIGINS)} complete, {len(skipped)} skipped",flush=True)
    if not daily:
        raise RuntimeError("No successful rolling-origin forecasts. Check the historical data and skipped dates.")
    pd.DataFrame(daily).to_csv(dest/FILES["daily"],index=False)
    pd.DataFrame(hourly).to_csv(dest/FILES["hourly"],index=False)
    pd.DataFrame(skipped,columns=["day","reason"]).to_csv(dest/FILES["skip"],index=False)
    (dest/"configuration.json").write_text(json.dumps({
        "version":VERSION,"group":group,"baseline":BASELINE,"training_weekdays":N,
        "horizon_hours":H,"origin_hours":list(ORIGINS),"steps_per_hour":steps_per_hour,
        "source_sweep":str(from_sweep) if from_sweep else None,
        "oracle":"F0 and R0 jointly optimized separately over each origin's full H=12 interval",
        "start_day":start_day,"end_day":end_day,"max_days":max_days,
    },indent=2),encoding="utf-8")
    return report(dest)


def main():
    parser=ArgumentParser(description="WM4 weekly-temperature N9/H12: twelve-origin pooled cumulative RMSE.")
    parser.add_argument("--group",choices=("MED","ALL"),default="MED")
    parser.add_argument("--from-sweep",default=None,
                        help="Reuse saved N9 historical theta/day cohort; each shifted origin refits both oracle states.")
    parser.add_argument("--steps-per-hour",type=int,default=4)
    parser.add_argument("--start-day",default=None)
    parser.add_argument("--end-day",default=None)
    parser.add_argument("--max-days",type=int,default=0)
    parser.add_argument("--progress-every",type=int,default=10)
    parser.add_argument("--output",default=None)
    parser.add_argument("--report-only",action="store_true")
    parser.add_argument("--fail-fast",action="store_true")
    args=parser.parse_args()
    if args.steps_per_hour<2 or args.max_days<0 or args.progress_every<1:
        parser.error("Require steps-per-hour>=2, max-days>=0, progress-every>=1.")
    dest=Path(args.output) if args.output else OUTPUT/args.group.lower()
    if args.report_only:
        config=dest/"configuration.json"
        if not config.is_file() or json.loads(config.read_text(encoding="utf-8")).get("version")!=VERSION:
            parser.error("Saved results are obsolete single-origin output; rerun the twelve-origin experiment.")
        return report(dest)
    return run(args.group,args.steps_per_hour,args.start_day,args.end_day,
               args.max_days,args.progress_every,args.from_sweep,dest,args.fail_fast)


if __name__=="__main__":main()
