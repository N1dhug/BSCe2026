"""Preliminary data tests and rolling-origin baseline comparisons.

No WM4 states or oracle initial conditions are used here. All baseline
parameters are fitted exclusively on N corresponding historical k-hour
windows. Target demand is reserved for evaluation; target temperature is
observed exogenous data (retrospective, not an operational temperature
forecast). Missing target values are never imputed.

Usage:
  python -m src.exe.data.baseline_diagnostics --groups MED ALL --N 4 --k 12
  python -m src.exe.data.baseline_diagnostics --groups MED --N 2 4 8 --k 6 12
"""
from argparse import ArgumentParser
from math import ceil, erfc, sqrt
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.moving_horizon_common import group_frame
from src.exe.run_model.wm_4.plot_corresponding_interval_fit import BASELINES, _fit_sparse_baseline


ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/"files"/"output"/"data"/"baseline_diagnostics"
WEEKDAYS=("Mon","Tue","Wed","Thu","Fri","Sat","Sun")


def _validate(frame):
    if not isinstance(frame,pd.DataFrame) or not isinstance(frame.index,pd.DatetimeIndex):
        raise TypeError("Aligned data must be a datetime-indexed DataFrame.")
    if frame.index.has_duplicates or frame.index.hasnans or frame.index.tz is not None:
        raise ValueError("Data index must be unique, timezone-naive, and contain no NaT.")
    if not {"I","Temp","U"}<=set(frame):
        raise ValueError("Aligned data must provide I, Temp and U.")
    frame=frame[["I","Temp","U"]].sort_index().astype(float).replace([np.inf,-np.inf],np.nan)
    if frame.empty: raise ValueError("Data is empty.")
    if np.any((frame.index.minute!=0)|(frame.index.second!=0)):
        raise ValueError("This analysis requires hour-aligned observations.")
    return frame


def _period(frame,start=None,end=None):
    begin=pd.Timestamp(start) if start is not None else frame.index.min()
    finish=pd.Timestamp(end) if end is not None else frame.index.max()+pd.Timedelta(hours=1)
    if pd.isna(begin) or pd.isna(finish) or begin>=finish:
        raise ValueError("Require a valid start earlier than the exclusive end.")
    return frame.loc[(frame.index>=begin)&(frame.index<finish)]


def _metrics(observed,predicted):
    e=np.asarray(predicted,dtype=float)-np.asarray(observed,dtype=float)
    if e.size<1 or not np.isfinite(e).all():
        raise ValueError("Metrics require at least one finite pair.")
    return {"mse":float(np.mean(e*e)),"rmse":float(np.sqrt(np.mean(e*e))),
            "mae":float(np.mean(np.abs(e))),"bias":float(np.mean(e))}


def _lagged(series,hours):
    """Exact clock-time alignment, never row-shift across data gaps."""
    return series.reindex(series.index-pd.Timedelta(hours=int(hours))).to_numpy(dtype=float)


def acf(series,max_lag=336):
    if not isinstance(series,pd.Series) or not isinstance(series.index,pd.DatetimeIndex):
        raise TypeError("acf expects a datetime-indexed Series.")
    if max_lag<1: raise ValueError("max_lag must be positive.")
    series=series.sort_index().replace([np.inf,-np.inf],np.nan)
    x=series.to_numpy(dtype=float)
    rows=[]
    for lag in range(int(max_lag)+1):
        past=x if lag==0 else _lagged(series,lag)
        good=np.isfinite(x)&np.isfinite(past)
        a,b=x[good],past[good]
        corr=float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>1e-12 and np.std(b)>1e-12 else np.nan
        if lag==0 and len(a)>0: corr=1.0
        rows.append({"lag_hours":lag,"pair_count":int(len(a)),"autocorrelation":corr})
    return pd.DataFrame(rows)


def weekday_profile(frame):
    valid=frame["I"].dropna()
    groups=valid.groupby([valid.index.dayofweek,valid.index.hour])
    result=groups.agg(["mean","std","count"]).reset_index()
    result.columns=["weekday","hour","mean","std","count"]
    result["weekday_name"]=result["weekday"].map(dict(enumerate(WEEKDAYS)))
    return result


def corresponding_similarity(frame,max_weeks=8):
    """Paired comparison of 7n days with neighboring 7n +/- 1 day controls."""
    if not isinstance(max_weeks,(int,np.integer)) or max_weeks<1:
        raise ValueError("max_weeks must be a positive integer.")
    demand=frame["I"].sort_index()
    current=demand.to_numpy(dtype=float)
    rows=[]
    for n in range(1,int(max_weeks)+1):
        weekly=_lagged(demand,168*n)
        before=_lagged(demand,168*n-24)
        after=_lagged(demand,168*n+24)
        valid=np.isfinite(current)&np.isfinite(weekly)&np.isfinite(before)&np.isfinite(after)
        if not np.any(valid): continue
        actual=current[valid]
        w=_metrics(actual,weekly[valid])
        b=_metrics(actual,before[valid])
        a=_metrics(actual,after[valid])
        rows.append({"weeks":n,"matched_pairs":int(np.sum(valid)),
                     "weekly_mse":w["mse"],"preceding_weekday_mse":b["mse"],
                     "following_weekday_mse":a["mse"],
                     "weekly_rmse":w["rmse"],"preceding_weekday_rmse":b["rmse"],
                     "following_weekday_rmse":a["rmse"],
                     "gain_vs_neighbor_mean_mse":.5*(b["mse"]+a["mse"])-w["mse"]})
    return pd.DataFrame(rows)


def _demean(values,groups):
    counts=np.bincount(groups,minlength=168)
    sums=np.bincount(groups,weights=values,minlength=168)
    mean=np.divide(sums,counts,out=np.zeros(168,dtype=float),where=counts>0)
    return values-mean[groups]


def _hac(X,residual,index,lags):
    """Bartlett/Newey-West covariance using only exact hourly lag matches."""
    gram=np.linalg.pinv(X.T@X)
    scores=X*residual[:,None]
    meat=scores.T@scores
    for lag in range(1,int(lags)+1):
        previous=index.get_indexer(index-pd.Timedelta(hours=lag))
        valid=previous>=0
        if not np.any(valid): continue
        cross=scores[valid].T@scores[previous[valid]]
        meat+=(1.0-lag/(lags+1.0))*(cross+cross.T)
    covariance=gram@meat@gram
    return np.sqrt(np.maximum(0.0,np.diag(covariance)))


def _coefficient(value,se,identified):
    value=float(value)
    se=float(se)
    if not identified or not np.isfinite(se) or se<1e-14:
        return {"coefficient":value,"hac_standard_error":None,"z":None,"p_approx":None}
    z=value/se
    return {"coefficient":value,"hac_standard_error":se,"z":float(z),
            "p_approx":float(erfc(abs(z)/sqrt(2)))}


def dependence(frame,base_temperature=18.0,hac_lags=24):
    """Weekday-hour fixed-effects: chill alone, then chill+price.

    HAC z/p values are asymptotic *associations*, not causal effects.
    """
    if not np.isfinite(base_temperature) or hac_lags<0:
        raise ValueError("base_temperature must be finite and hac_lags nonnegative.")
    data=frame[["I","Temp","U"]].dropna().sort_index()
    if len(data)<32: raise ValueError("Insufficient finite hourly data for dependence tests.")
    idx=data.index
    groups=idx.dayofweek.to_numpy(dtype=int)*24+idx.hour.to_numpy(dtype=int)
    y=data["I"].to_numpy(dtype=float)
    C=np.maximum(0.0,float(base_temperature)-data["Temp"].to_numpy(dtype=float))
    U=data["U"].to_numpy(dtype=float)
    yr,cr,ur=(_demean(v,groups) for v in (y,C,U))
    Xc=cr[:,None]
    Xcu=np.column_stack((cr,ur))
    beta_c=np.linalg.lstsq(Xc,yr,rcond=None)[0]
    beta_cu=np.linalg.lstsq(Xcu,yr,rcond=None)[0]
    resid_time=yr
    resid_temp=yr-Xc@beta_c
    resid_full=yr-Xcu@beta_cu
    identified_c=bool(np.linalg.matrix_rank(Xc)==1)
    identified_full=bool(np.linalg.matrix_rank(Xcu)==2)
    se_c=_hac(Xc,resid_temp,idx,hac_lags)
    se_full=_hac(Xcu,resid_full,idx,hac_lags)

    # Frisch-Waugh-Lovell: price association after time and chill effects.
    factor=float(np.dot(cr,cr))
    price_adj=ur-cr*(float(np.dot(cr,ur))/factor) if factor>1e-12 else ur.copy()
    y_adj=yr-cr*(float(np.dot(cr,yr))/factor) if factor>1e-12 else yr.copy()
    mse_time=float(np.mean(resid_time**2))
    mse_temp=float(np.mean(resid_temp**2))
    mse_full=float(np.mean(resid_full**2))
    report={
        "samples":len(data),"weekday_hour_groups":int(np.unique(groups).size),
        "base_temperature_c":float(base_temperature),
        "hac_lags_hours":int(hac_lags),
        "temperature_only":_coefficient(beta_c[0],se_c[0],identified_c),
        "temperature_conditional":_coefficient(beta_cu[0],se_full[0],identified_full),
        "price_conditional":_coefficient(beta_cu[1],se_full[1],identified_full),
        "mse_weekday_hour":mse_time,
        "mse_with_temperature":mse_temp,
        "mse_with_temperature_and_price":mse_full,
        "temperature_mse_gain":mse_time-mse_temp,
        "price_conditional_mse_gain":mse_temp-mse_full,
        "identified_temperature":identified_c,
        "identified_joint":identified_full,
        "interpretation":"Conditional associations; HAC normal p-values are approximate, not causal.",
    }
    residuals=pd.DataFrame({"consumption":y,"chill":C,"price":U,
                            "time_adjusted_demand":yr,"time_adjusted_chill":cr,
                            "price_adjusted":price_adj,"demand_adjusted":y_adj,
                            "residual_time":resid_time,"residual_temperature":resid_temp,
                            "residual_temperature_price":resid_full},index=idx)
    daily=pd.DataFrame({"consumption":y,"chill":C},index=idx).resample("D").agg(
        {"consumption":["mean","count"],"chill":"mean"})
    daily.columns=["consumption","hour_count","chill"]
    daily=daily[daily["hour_count"]>=18]
    return report,residuals,daily


def _save(fig,path):
    fig.tight_layout()
    fig.savefig(path,dpi=190,bbox_inches="tight")
    plt.close(fig)


def _plot_profiles(profile,path):
    fig,ax=plt.subplots(figsize=(11,5))
    for d,label in enumerate(WEEKDAYS):
        week=profile[profile["weekday"]==d].sort_values("hour")
        ax.plot(week["hour"],week["mean"],label=label,lw=1.7)
    ax.set(xlabel="Hour of day",ylabel="Normalized consumption",
           title="Average hourly demand by weekday")
    ax.set_xticks(np.arange(0,24,2));ax.grid(alpha=.2);ax.legend(ncol=7,fontsize=9)
    _save(fig,path)


def _plot_acf(values,path,title):
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(values["lag_hours"],values["autocorrelation"],lw=1.15)
    for lag in (24,168):
        if lag<=values["lag_hours"].max():
            ax.axvline(lag,color="0.5",ls="--",lw=.8)
            ax.annotate(f"{lag} h",xy=(lag,.85),fontsize=9)
    ax.axhline(0,color="0.45",lw=.7)
    ax.set(xlabel="Exact time lag (hours)",ylabel="Pairwise autocorrelation",title=title)
    ax.grid(alpha=.2)
    _save(fig,path)


def _plot_similarity(sim,path):
    fig,ax=plt.subplots(figsize=(9,4.5))
    if not sim.empty:
        for name,label in (("weekly_rmse","Same weekday/hour"),
                           ("preceding_weekday_rmse","One weekday earlier"),
                           ("following_weekday_rmse","One weekday later")):
            ax.plot(sim["weeks"],sim[name],marker="o",label=label)
    ax.set(xlabel="Historical lag (weeks)",ylabel="RMSE of matched observations",
           title="Corresponding intervals versus adjacent weekdays")
    ax.grid(alpha=.2);ax.legend()
    _save(fig,path)


def _plot_dependence(report,residuals,daily,path):
    fig,ax=plt.subplots(figsize=(9,5))
    ax.scatter(daily["chill"],daily["consumption"],s=13,alpha=.5)
    ax.set(xlabel="Daily average heating-degree factor (degrees C)",
           ylabel="Daily mean normalized consumption",
           title="Descriptive daily consumption versus heating-degree factor")
    ax.grid(alpha=.2)
    _save(fig,path/"daily_temperature.png")

    fig,ax=plt.subplots(figsize=(9,5))
    sub=residuals.iloc[::max(1,len(residuals)//3000)]
    ax.scatter(sub["time_adjusted_chill"],sub["time_adjusted_demand"],s=5,alpha=.2)
    x=np.linspace(residuals["time_adjusted_chill"].min(),residuals["time_adjusted_chill"].max(),80)
    beta=report["temperature_only"]["coefficient"]
    ax.plot(x,beta*x,color="tab:red",lw=2,label=f"FE temperature slope = {beta:.4g}")
    ax.set(xlabel="Chill after weekday-hour adjustment",
           ylabel="Consumption after weekday-hour adjustment",
           title="Temperature association controlling for weekday and hour")
    ax.grid(alpha=.2);ax.legend()
    _save(fig,path/"temperature_partial.png")

    fig,ax=plt.subplots(figsize=(9,5))
    ax.scatter(sub["price_adjusted"],sub["demand_adjusted"],s=5,alpha=.2)
    x=np.linspace(residuals["price_adjusted"].min(),residuals["price_adjusted"].max(),80)
    beta=report["price_conditional"]["coefficient"]
    ax.plot(x,beta*x,color="tab:red",lw=2,label=f"Conditional price slope = {beta:.4g}")
    ax.set(xlabel="Price after weekday-hour and chill adjustment",
           ylabel="Consumption after weekday-hour and chill adjustment",
           title="Price association conditional on time and temperature")
    ax.grid(alpha=.2);ax.legend()
    _save(fig,path/"price_partial.png")


def _rolling_origins(frame,period,backlogs,horizons,stride,max_origins):
    start=max(period.index.min(),frame.index.min()+pd.Timedelta(weeks=max(backlogs)))
    end=min(period.index.max()+pd.Timedelta(hours=1),
            frame.index.max()+pd.Timedelta(hours=1))
    final=end-pd.Timedelta(hours=max(horizons))
    if start>final: return []
    candidates=pd.date_range(start,final,freq=f"{stride}h")
    if len(candidates)>max_origins:
        chosen=np.linspace(0,len(candidates)-1,max_origins).round().astype(int)
        candidates=candidates[np.unique(chosen)]
    return list(candidates)


def rolling_baselines(frame,period,backlogs=(4,),horizons=(12,),
                      stride=24,max_origins=64):
    """Matched-origin, target-held-out comparison of the three baselines.

    To preserve the fitting assumption, no out-of-window historical rows
    are used and incomplete historical/target windows are skipped, not
    imputed. All three models get identical training/target observations.
    """
    if any(not isinstance(n,(int,np.integer)) or not 1<=n<=52 for n in backlogs):
        raise ValueError("Each N must be an integer in [1,52].")
    if any(not isinstance(k,(int,np.integer)) or not 2<=k<=24 for k in horizons):
        raise ValueError("Each k must be an integer in [2,24].")
    if stride<1 or max_origins<1: raise ValueError("stride and max_origins must be positive.")
    backlogs=tuple(dict.fromkeys(int(n) for n in backlogs))
    horizons=tuple(dict.fromkeys(int(k) for k in horizons))
    origins=_rolling_origins(frame,period,backlogs,horizons,stride,max_origins)
    scores,predictions=[],[]
    skipped={"missing_training":0,"missing_target":0,"baseline_fit":0}
    for k in horizons:
        for N in backlogs:
            for origin in origins:
                target_idx=pd.date_range(origin,periods=k,freq="h")
                target=period.reindex(target_idx)[["I","Temp"]]
                if not np.isfinite(target.to_numpy(dtype=float)).all():
                    skipped["missing_target"]+=1
                    continue
                train_idx=[pd.date_range(origin-pd.Timedelta(weeks=i),
                                         periods=k,freq="h") for i in range(N,0,-1)]
                training=pd.concat([frame.reindex(index)[["I","Temp"]]
                                    for index in train_idx]).sort_index()
                if not np.isfinite(training.to_numpy(dtype=float)).all():
                    skipped["missing_training"]+=1
                    continue
                if not training.index.max()<origin:
                    raise RuntimeError("Baseline training must precede the target origin.")
                all_predictions={}
                for B in BASELINES:
                    try:
                        estimator=_fit_sparse_baseline(training,B)
                        yhat=np.asarray(estimator.predict(target_idx,target["Temp"].to_numpy(dtype=float)),
                                        dtype=float)
                        if yhat.shape!=(k,) or not np.isfinite(yhat).all():
                            raise ValueError("Nonfinite or malformed baseline prediction.")
                        all_predictions[B]=yhat
                    except (ValueError,np.linalg.LinAlgError,FloatingPointError):
                        skipped["baseline_fit"]+=1
                        break
                if len(all_predictions)!=len(BASELINES): continue
                actual=target["I"].to_numpy(dtype=float)
                for B,pred in all_predictions.items():
                    score=_metrics(actual,pred)
                    scores.append({"origin":origin,"N":N,"k":k,"baseline":B,
                                   "train_samples":len(training),**score})
                    for h in range(k):
                        predictions.append({"origin":origin,"timestamp":target_idx[h],
                                            "lead_hour":h,"N":N,"k":k,"baseline":B,
                                            "observed":float(actual[h]),"predicted":float(pred[h]),
                                            "error":float(pred[h]-actual[h])})
    return pd.DataFrame(scores),pd.DataFrame(predictions),{
        "candidate_origins":len(origins),"skipped_windows":skipped,
        "historical_policy":"Exact N corresponding k-hour intervals; no historical imputation.",
        "target_policy":"Observed demand is evaluation-only; observed temperature is used as an exogenous input.",
    }


def summarize_baselines(scores,predictions):
    if scores.empty: return pd.DataFrame()
    rows=[]
    for (N,k,B),group in predictions.groupby(["N","k","baseline"],sort=True):
        values=group["error"].to_numpy(dtype=float)
        origins=scores.loc[(scores["N"]==N)&(scores["k"]==k)&(scores["baseline"]==B)]
        rows.append({"N":int(N),"k":int(k),"baseline":B,
                     "origins":int(origins["origin"].nunique()),"samples":len(values),
                     "mse":float(np.mean(values**2)),
                     "rmse":float(np.sqrt(np.mean(values**2))),
                     "mae":float(np.mean(np.abs(values))),
                     "bias":float(np.mean(values)),
                     "mean_origin_rmse":float(origins["rmse"].mean())})
    return pd.DataFrame(rows)


def _plot_baselines(summary,path):
    for metric,label in (("rmse","Pooled out-of-sample RMSE"),
                         ("bias","Pooled prediction bias")):
        fig,ax=plt.subplots(figsize=(max(9,1.6*summary[["N","k"]].drop_duplicates().shape[0]+3),5))
        configs=sorted(set(zip(summary["N"],summary["k"])))
        xx=np.arange(len(configs),dtype=float)
        width=.8/len(BASELINES)
        for j,B in enumerate(BASELINES):
            subset=summary[summary["baseline"]==B].set_index(["N","k"])
            vals=[subset.at[(N,k),metric] if (N,k) in subset.index else np.nan
                  for N,k in configs]
            ax.bar(xx-.4+width*(j+.5),vals,width=width,label=B)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"N={N}, k={k}h" for N,k in configs])
        ax.set(ylabel=label,xlabel="Historical backlog and prediction horizon",
               title=f"Matched-origin baseline comparison: {label}")
        if metric=="bias": ax.axhline(0,color="0.3",lw=.8)
        ax.grid(axis="y",alpha=.2)
        ax.legend()
        _save(fig,path/f"baseline_{metric}.png")


def _write_exploratory(frame,directory,max_weeks,max_acf_lag,base_temperature,hac_lags):
    directory.mkdir(parents=True,exist_ok=True)
    profile=weekday_profile(frame)
    raw_acf=acf(frame["I"],max_acf_lag)
    similarity=corresponding_similarity(frame,max_weeks)
    report,residuals,daily=dependence(frame,base_temperature,hac_lags)
    profile.to_csv(directory/"weekday_hour_profile.csv",index=False)
    raw_acf.to_csv(directory/"consumption_acf.csv",index=False)
    similarity.to_csv(directory/"corresponding_similarity.csv",index=False)
    residuals.to_csv(directory/"adjusted_residuals.csv",index_label="timestamp")
    daily.to_csv(directory/"daily_temperature.csv",index_label="date")
    (directory/"dependence.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    _plot_profiles(profile,directory/"weekday_profiles.png")
    _plot_acf(raw_acf,directory/"consumption_acf.png","Demand autocorrelation")
    _plot_similarity(similarity,directory/"corresponding_similarity.png")
    _plot_dependence(report,residuals,daily,directory)

    # Diagnose remaining weekly structure after removing fixed effects,
    # temperature and price; only exact hourly pairs enter the ACF.
    for name in ("residual_time","residual_temperature","residual_temperature_price"):
        acf(residuals[name],max_acf_lag).to_csv(
            directory/f"{name}_acf.csv",index=False)
    remaining=acf(residuals["residual_temperature_price"],max_acf_lag)
    _plot_acf(remaining,directory/"adjusted_residual_acf.png",
              "Residual autocorrelation after time, temperature and price")
    return report


def run(*,groups=("MED","ALL"),start=None,end=None,backlogs=(4,),horizons=(12,),
        max_weeks=8,max_acf_lag=336,base_temperature=18.0,hac_lags=24,
        stride_hours=24,max_origins=64,output=None):
    """Save reproducible CSV, JSON and PNG results for every requested group."""
    if isinstance(groups,str): groups=(groups,)
    if not groups or any(group not in ("MED","ALL") for group in groups):
        raise ValueError("groups must contain MED and/or ALL.")
    if max_acf_lag<1 or max_weeks<1: raise ValueError("ACF and weekly lag maxima must be positive.")
    dest=Path(output) if output is not None else OUTPUT
    answers={}
    for group in dict.fromkeys(groups):
        frame=_validate(group_frame(group))
        period=_period(frame,start,end)
        if period.empty: raise ValueError(f"No hourly observations in selected period for {group}.")
        directory=dest/group.lower()
        diagnostics=_write_exploratory(
            period,directory,max_weeks,max_acf_lag,base_temperature,hac_lags)
        scores,predictions,coverage=rolling_baselines(
            frame,period,backlogs,horizons,stride_hours,max_origins)
        if scores.empty:
            raise ValueError(f"No admissible rolling-origin windows for {group}; "
                             "reduce N/k or widen the date range.")
        summary=summarize_baselines(scores,predictions)
        scores.to_csv(directory/"baseline_origin_scores.csv",index=False)
        predictions.to_csv(directory/"baseline_hourly_predictions.csv",index=False)
        summary.to_csv(directory/"baseline_summary.csv",index=False)
        _plot_baselines(summary,directory)
        evaluation={"group":group,"start":str(period.index.min()),
                    "end_inclusive":str(period.index.max()),
                    "observed_hourly_rows":len(period),
                    "backlogs":list(backlogs),"horizons":list(horizons),
                    "stride_hours":int(stride_hours),
                    "max_origins":int(max_origins),"coverage":coverage,
                    "regression":diagnostics,
                    "baseline_summary":summary.to_dict(orient="records"),
                    "notice":"Conditional associations are not causal. "
                             "Observed target temperature is not a real-time forecast. "
                             "Overlapping prediction windows are dependent, not independent trials."}
        (directory/"summary.json").write_text(json.dumps(evaluation,indent=2),encoding="utf-8")
        answers[group]={"directory":directory,"profile":directory/"weekday_hour_profile.csv",
                        "similarity":directory/"corresponding_similarity.csv",
                        "dependence":diagnostics,"baseline_summary":summary,
                        "origin_scores":scores,"predictions":predictions}
        print(f"[{group}] saved diagnostics to {directory}")
        print(summary.to_string(index=False,float_format=lambda v:f"{v:.5f}"))
    return answers


def main():
    parser=ArgumentParser(description="Preliminary time/temperature/price tests and rolling-origin baseline comparison.")
    parser.add_argument("--groups",nargs="+",choices=("MED","ALL"),default=("MED","ALL"))
    parser.add_argument("--start",default=None,help="Inclusive statistical/evaluation start (YYYY-MM-DD).")
    parser.add_argument("--end",default=None,help="Exclusive statistical/evaluation end (YYYY-MM-DD).")
    parser.add_argument("--N",type=int,nargs="+",default=[4],help="Weekly-backlog choices.")
    parser.add_argument("--k",type=int,nargs="+",default=[12],help="Horizon choices in hours (2..24).")
    parser.add_argument("--max-week-lag",type=int,default=8)
    parser.add_argument("--max-acf-lag",type=int,default=336)
    parser.add_argument("--base-temperature",type=float,default=18.0)
    parser.add_argument("--hac-lags",type=int,default=24)
    parser.add_argument("--stride-hours",type=int,default=24)
    parser.add_argument("--max-origins",type=int,default=64)
    parser.add_argument("--output",default=None,help="Output directory containing group subfolders.")
    args=parser.parse_args()
    return run(groups=args.groups,start=args.start,end=args.end,
               backlogs=args.N,horizons=args.k,max_weeks=args.max_week_lag,
               max_acf_lag=args.max_acf_lag,base_temperature=args.base_temperature,
               hac_lags=args.hac_lags,stride_hours=args.stride_hours,
               max_origins=args.max_origins,output=args.output)


if __name__=="__main__": main()
