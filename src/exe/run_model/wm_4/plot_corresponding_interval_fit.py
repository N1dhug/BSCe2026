"""Illustrate WM4 historical fitting on N corresponding k-hour intervals.

For prediction origin (day, hour), select N preceding weekly-offset
intervals. Fit the selected baseline using ONLY those N x k hourly
observations, then fit one common WM4 theta and one F0 per interval
(R0 is observed as D0-B0) with the historical joint convex QP.

The historical grid PNG contains N subplots, each displaying 24 or 25
hours of observed demand while baseline and WM4 fit appear ONLY
inside the k-hour fitting interval. A second PNG displays the prediction
interval with an independently optimized oracle initial state in red.
The prediction-day observations are used ONLY to optimize that oracle state
and report retrospective prediction error, never to fit baseline or theta.
"""
from argparse import ArgumentParser
from math import ceil, sqrt
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.moving_horizon_common import group_frame, metrics
from src.exe.run_model.wm_4.methodology_uncertainty import demand_interval, fit_historical_uncertainty
from src.model.wm_4 import wm_4
from src.params.wm_4 import joint_qp_params, optimized_qp_params
from src.tools.baseline import weekly_average, weekly_temperature, statistical_output
from src.tools.impute_hourly import impute_hourly_days


ROOT=Path(__file__).resolve().parents[4]
OUTPUT=ROOT/"files"/"output"/"wm_4"/"corresponding_interval_fit"
BASELINES={"weekly_average":weekly_average,"weekly_temperature":weekly_temperature,
           "statistical_output":statistical_output}


def _origin(day,hour):
    date=pd.Timestamp(day)
    if pd.isna(date) or date!=date.normalize():
        raise ValueError("day must be a valid calendar date at midnight (YYYY-MM-DD).")
    if not isinstance(hour,(int,np.integer)) or not 0<=hour<=23:
        raise ValueError("hour must be an integer from 0 through 23.")
    return date+pd.Timedelta(hours=int(hour))


def _intervals(origin,k,N):
    """Chronological weekly-offset starts (oldest through most recent)."""
    if not isinstance(k,(int,np.integer)) or not 2<=k<=24:
        raise ValueError("k must be an integer number of hours between 2 and 24.")
    if not isinstance(N,(int,np.integer)) or N<1:
        raise ValueError("N must be a positive integer.")
    return [origin-pd.Timedelta(weeks=i) for i in range(N,0,-1)]


def _fit_sparse_baseline(samples,B):
    """Fit one of the existing baseline formulations on EXACT interval rows.

    Baseline.fit() ordinarily imputes complete calendar days. For this
    illustration it is important not to add out-of-window observations,
    so use the same model mathematics directly on the selected samples.
    """
    if B not in BASELINES: raise ValueError(f"B must be one of {tuple(BASELINES)}.")
    if not isinstance(samples.index,pd.DatetimeIndex) or samples.index.has_duplicates:
        raise ValueError("Baseline training requires distinct datetime-indexed samples.")
    if not {"I","Temp"}<=set(samples):
        raise ValueError("Baseline data must contain I and Temp.")
    samples=samples[["I","Temp"]].sort_index()
    if samples.empty or not np.isfinite(samples.to_numpy(dtype=float)).all():
        raise ValueError("Baseline training intervals must contain finite demand and temperature.")
    model=BASELINES[B](min_samples=2)
    index=samples.index
    demand=samples["I"].to_numpy(dtype=float)
    temp=samples["Temp"].to_numpy(dtype=float)

    if isinstance(model,weekly_temperature):
        groups=model._groups(index)
        chill=np.maximum(0.0,model.base_temperature-temp)
        mean_d=model._profile(groups,demand)
        mean_c=model._profile(groups,chill)
        centered_d=demand-mean_d[groups]
        centered_c=chill-mean_c[groups]
        denom=float(centered_c @ centered_c)
        model.beta_C=float((centered_c @ centered_d)/denom) if denom>1e-12 else 0.0
        model.chill_center=float(np.mean(chill))
        model.levels=mean_d-model.beta_C*(mean_c-model.chill_center)
        model.coef=np.r_[model.levels,model.beta_C]
    elif isinstance(model,weekly_average):
        groups=model._groups(index)
        model.levels=model._profile(groups,demand)
        model.coef=model.levels.copy()
    else:
        raw=model._features(index,temp)
        X=model._standardize_fit(raw)
        penalty=model.ridge*np.eye(X.shape[1],dtype=float)
        penalty[0,0]=0.0
        weights=np.ones(len(demand),dtype=float)
        beta=np.zeros(X.shape[1],dtype=float)
        for _ in range(model.max_iter):
            root=np.sqrt(weights)
            Xw,yw=X*root[:,None],demand*root
            beta=np.linalg.solve(Xw.T@Xw+penalty,Xw.T@yw)
            residual=demand-X@beta
            center=np.median(residual)
            scale=1.4826*np.median(np.abs(residual-center))
            if not np.isfinite(scale) or scale<1e-10: break
            limit=model.huber_delta*scale
            absolute=np.abs(residual)
            next_weights=np.ones_like(weights)
            outside=absolute>limit
            next_weights[outside]=limit/absolute[outside]
            if np.max(np.abs(next_weights-weights))<1e-6: break
            weights=next_weights
        model.coef=beta

    model.metadata={"samples":len(samples),"fitting_mode":"exact_corresponding_intervals",
                    "start":str(index.min()),"end":str(index.max())}
    return model


def _window(frame,start,k,baseline,sph):
    """Create exactly k hourly WM4 observations on the normalized-day grid."""
    hourly=pd.date_range(start,periods=k,freq="h")
    source=frame.reindex(hourly)[["I","U","Temp"]]
    if source.isna().any().any() or not np.isfinite(source.to_numpy(dtype=float)).all():
        raise ValueError(f"Missing historical inputs in [{start}, {start+pd.Timedelta(hours=k)}).")
    elapsed=np.arange(k*sph+1,dtype=float)/sph
    times=start+pd.to_timedelta(elapsed,unit="h")
    sample_hours=np.arange(k+1,dtype=float)
    D=np.interp(elapsed,sample_hours,np.r_[source["I"].to_numpy(),source["I"].iloc[-1]])
    T=np.interp(elapsed,sample_hours,np.r_[source["Temp"].to_numpy(),source["Temp"].iloc[-1]])
    U=np.r_[source["U"].to_numpy(),source["U"].iloc[-1]][
        np.minimum(np.floor(elapsed+1e-10).astype(int),k)]
    B=np.asarray(baseline.predict(times,T),dtype=float)
    if not np.isfinite(B).all() or not np.isfinite(U).all():
        raise ValueError("Baseline or price contains nonfinite samples.")
    mask=np.zeros(len(elapsed),dtype=bool)
    mask[np.arange(k)*sph]=True
    transitions=np.zeros(len(elapsed)-1,dtype=bool)
    transitions[:(k-1)*sph]=True
    data={"I":D,"U":U,"PE":U.copy(),"Time":elapsed,"O":B,
          "observation_mask":mask,"transition_mask":transitions}
    return data


def _context(frame,start,k,edge):
    """Raw demand for a 24- or 25-hour display, without imputing plot values."""
    length=k+2*edge
    timestamps=pd.date_range(start-pd.Timedelta(hours=edge),periods=length,freq="h")
    if timestamps[0]<frame.index.min() or timestamps[-1]>frame.index.max():
        raise ValueError(f"Insufficient demand context around {start:%Y-%m-%d %H:%M}.")
    values=frame.reindex(timestamps)["I"].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        raise ValueError(f"No observed demand in context around {start}.")
    return np.arange(-edge,k+edge+1,dtype=float),np.r_[values,values[-1]]


def _clock_axis(start,k,edge):
    """Continuous clock-hour coordinates with labels wrapped to 00..23."""
    hour=int(pd.Timestamp(start).hour)
    left,right=hour-edge,hour+k+edge
    ticks=np.unique(np.r_[np.arange(left,right+1,3,dtype=int),hour,hour+k])
    return hour,left,right,ticks,[f"{int(tick)%24:02d}" for tick in ticks]


def _plot(starts,frame,k,N,B,sph,fit,model,inputs,output,uncertainty):
    """Exactly ONE grid figure, with one panel per historical interval."""
    edge=int(ceil((24-k)/2))
    ncols=min(3,int(ceil(sqrt(N))))
    nrows=int(ceil(N/ncols))
    fig,axes=plt.subplots(nrows,ncols,figsize=(6*ncols,4.15*nrows),
                          sharex=True,sharey=True,squeeze=False)
    all_values=[]
    fine=np.arange(k*sph+1,dtype=float)/sph
    hour,left,right,xticks,labels=_clock_axis(starts[0],k,edge)
    rows=[]
    for m,(start,data,state) in enumerate(zip(starts,inputs,fit["deterministic_states"])):
        ax=axes.flat[m]
        offset,context=_context(frame,start,k,edge)
        actual=np.asarray(data["I"],dtype=float)[np.arange(k)*sph]
        fitted=np.asarray(state["I"],dtype=float)
        baseline=np.asarray(data["O"],dtype=float)
        if fitted.shape!=(k*sph+1,) or not np.isfinite(fitted).all():
            raise ValueError(f"WM4 fit at interval {m+1} has invalid output.")
        score=metrics(actual,fitted[np.arange(k)*sph])
        bscore=metrics(actual,baseline[np.arange(k)*sph])
        init=np.asarray(fit["init_cond"],dtype=float)[m]
        rows.append({"interval":m+1,"start":start,"end":start+pd.Timedelta(hours=k),
                     "F0":float(init[0]),"R0":float(init[1]),
                     "wm4_rmse":score["rmse"],"baseline_rmse":bscore["rmse"],
                     "process_sigma":uncertainty["sigma"],"observation_tau":uncertainty["tau"],
                     "historical_coverage_95":uncertainty["historical_coverage_95"]})
        # Keep continuous hours for cross-midnight windows, but label actual
        # clock time, so a fitting origin at 06:00 appears at x=6.
        clock_hour=int(start.hour)
        ax.axvspan(clock_hour,clock_hour+k,color="tab:blue",alpha=.08)
        ax.axvline(clock_hour,color="0.55",lw=.8,ls=":")
        ax.axvline(clock_hour+k,color="0.55",lw=.8,ls=":")
        ax.step(offset+clock_hour,context,where="post",color="0.18",lw=1.8,
                label="Observed demand (full context)")
        ax.plot(fine+clock_hour,baseline,lw=2,color="tab:orange",ls="--",
                label="Baseline (fitting interval)")
        lower,upper=demand_interval(fitted,uncertainty)
        ax.fill_between(fine+clock_hour,lower,upper,color="tab:blue",alpha=.20,
                        linewidth=0,label="95% demand interval (historical fit)")
        ax.plot(fine+clock_hour,fitted,lw=2,color="tab:blue",
                label="ARM fit (fitting interval)")
        ax.set_title(f"Week -{N-m}: {start:%a %d %b %Y, %H:%M}  |  "
                     f"RMSE {score['rmse']:.4f}",fontsize=10)
        ax.grid(alpha=.18)
        all_values.extend(context[np.isfinite(context)])
        all_values.extend(baseline[np.isfinite(baseline)])
        all_values.extend(fitted[np.isfinite(fitted)])
        all_values.extend(lower[np.isfinite(lower)])
        all_values.extend(upper[np.isfinite(upper)])
    for ax in axes.flat[N:]: ax.set_visible(False)
    if all_values:
        low,high=float(np.min(all_values)),float(np.max(all_values))
        pad=max(.025,.08*(high-low))
        for ax in axes.flat[:N]: ax.set_ylim(low-pad,high+pad)
    for ax in axes.flat[:N]:
        ax.set_xlim(left,right)
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels)
        ax.tick_params(labelbottom=True,labelleft=True,labelsize=8)
    fig.supxlabel("Hour of day (24-hour clock; wraps at midnight)")
    fig.supylabel("Normalized aggregate electricity demand")
    fig.suptitle(f"ARM (Affine Revised Model) historical joint fit | {N} corresponding weekdays | "
                 f"k={k}h | B={B}\n"
                 f"Shared parameters: "
                 f"θF={float(fit['r'][0]):.4g}, "
                 f"θU={float(fit['r'][1]):.4g}, "
                 f"θR={float(fit['r'][2]):.4g} | "
                 f"σ={uncertainty['sigma']:.4g}, τ={uncertainty['tau']:.4g}",fontsize=13)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.927),
               ncol=4,fontsize=9)
    fig.tight_layout(rect=(.02,.045,.99,.895))
    output=Path(output)
    fig.savefig(output,dpi=190,bbox_inches="tight")
    plt.close(fig)
    return rows


def _plot_oracle(origin,frame,k,N,B,sph,input_data,oracle,output,uncertainty):
    """One independent oracle prediction figure; no historical-fit subplots."""
    edge=int(ceil((24-k)/2))
    hour,left,right,ticks,labels=_clock_axis(origin,k,edge)
    offset,context=_context(frame,origin,k,edge)
    fine=np.arange(k*sph+1,dtype=float)/sph
    observed=np.asarray(input_data["I"],dtype=float)[np.arange(k)*sph]
    baseline=np.asarray(input_data["O"],dtype=float)
    prediction=np.asarray(oracle["deterministic_state"]["I"],dtype=float)
    if prediction.shape!=(k*sph+1,) or not np.isfinite(prediction).all():
        raise ValueError("WM4 oracle returned invalid prediction.")
    score=metrics(observed,prediction[np.arange(k)*sph])
    bscore=metrics(observed,baseline[np.arange(k)*sph])
    init=np.asarray(oracle["init_cond"],dtype=float)
    if init.shape!=(2,) or not np.isfinite(init).all():
        raise ValueError("WM4 oracle returned invalid initial state.")

    fig,ax=plt.subplots(figsize=(11.8,5.0))
    ax.axvspan(hour,hour+k,color="tab:red",alpha=.07)
    ax.axvline(hour,color="0.55",lw=.8,ls=":")
    ax.axvline(hour+k,color="0.55",lw=.8,ls=":")
    ax.step(offset+hour,context,where="post",color="0.18",lw=1.8,
            label="Observed demand (full context)")
    ax.plot(fine+hour,baseline,color="tab:orange",lw=2,ls="--",
            label="Fitted historical baseline (prediction interval)")
    lower,upper=demand_interval(prediction,uncertainty)
    ax.fill_between(fine+hour,lower,upper,color="tab:red",alpha=.20,
                    linewidth=0,label="95% oracle-conditioned demand interval")
    ax.plot(fine+hour,prediction,color="red",lw=2.25,
            label="ARM oracle prediction (prediction interval)")
    ax.set_xlim(left,right)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Hour of day (24-hour clock; wraps at midnight)")
    ax.set_ylabel("Normalized aggregate electricity demand")
    ax.set_title(f"ARM (Affine Revised Model) oracle prediction | {origin:%a %d %b %Y, %H:%M} | "
                 f"k={k}h | N={N} | B={B}\n"
                 f"Oracle RMSE {score['rmse']:.4f} | Baseline RMSE {bscore['rmse']:.4f} | "
                 f"σ={uncertainty['sigma']:.4g}, τ={uncertainty['tau']:.4g}")
    ax.grid(alpha=.18)
    ax.legend(loc="best",fontsize=9)
    fig.tight_layout()
    fig.savefig(output,dpi=190,bbox_inches="tight")
    plt.close(fig)

    hourly=np.arange(k,dtype=int)
    rows=pd.DataFrame({
        "timestamp":origin+pd.to_timedelta(hourly,unit="h"),
        "lead_hour":hourly,
        "observed":observed,
        "baseline":baseline[hourly*sph],
        "wm4_oracle":prediction[hourly*sph],
        "demand_std":uncertainty["sd_hourly"],
        "lower_95":lower[hourly*sph],"upper_95":upper[hourly*sph],
    })
    rows.to_csv(output.with_suffix(".csv"),index=False)
    coverage=float(np.mean((observed>=lower[hourly*sph])&
                           (observed<=upper[hourly*sph])))
    return {"oracle_F0":float(init[0]),"oracle_R0":float(init[1]),
            "oracle_coverage_95":coverage,"process_sigma":uncertainty["sigma"],
            "observation_tau":uncertainty["tau"],
            "oracle_rmse":score["rmse"],"oracle_mse":score["mse"],
            "oracle_mae":score["mae"],"oracle_bias":score["bias"],
            "baseline_rmse":bscore["rmse"],"baseline_mse":bscore["mse"],
            "oracle_plot":output,"oracle_hourly":rows}


def run(*,day,hour,k,N,B,group="MED",steps_per_hour=4,output=None):
    """Fit N corresponding weekly-offset WM4 histories and plot the target-day oracle.

    Parameters
    ----------
    day : str or date-like
        Prediction origin date, e.g. '2022-08-20'.
    hour : int
        Prediction origin hour, 0..23.
    k : int
        Future horizon in hours (2..24).
    N : int
        Number of previous corresponding weekdays.
    B : str
        One of 'weekly_average', 'weekly_temperature', 'statistical_output'.

    Returns the output path and the fitted per-interval/WM4 diagnostics.
    """
    origin=_origin(day,hour)
    starts=_intervals(origin,k,N)
    if B not in BASELINES: raise ValueError(f"Unknown B={B!r}; choose from {tuple(BASELINES)}.")
    if group not in ("ALL","MED"): raise ValueError("group must be ALL or MED.")
    sph=int(steps_per_hour)
    if sph<2: raise ValueError("steps_per_hour must be at least 2.")
    frame=group_frame(group)
    if not isinstance(frame.index,pd.DatetimeIndex) or not frame.index.is_unique:
        raise ValueError("Source data must have a unique hourly datetime index.")
    frame=frame.sort_index()

    # Within-day interpolation is restricted to historical days. The plotted
    # context comes from the raw source and is never used for fitting.
    indices=[pd.date_range(start,periods=k,freq="h") for start in starts]
    needed=pd.DatetimeIndex(np.concatenate([idx.to_numpy() for idx in indices])).normalize().unique()
    history,imputation=impute_hourly_days(
        frame,days=needed,columns=("I","U","Temp"),require_days=True)
    selected=pd.concat([history.reindex(idx)[["I","Temp"]] for idx in indices]).sort_index()
    baseline=_fit_sparse_baseline(selected,B)
    inputs=[_window(history,start,k,baseline,sph) for start in starts]
    model=wm_4((0.0,k/24.0),k*sph+1,lambda t:0.0)
    fit=joint_qp_params(model).fit_many(inputs)
    if len(fit["deterministic_states"])!=N or np.asarray(fit["init_cond"]).shape!=(N,2):
        raise RuntimeError("Joint fit failed to produce one initial state per history.")
    if not np.all(np.isfinite(fit["r"])):
        raise RuntimeError("Joint fit returned nonfinite parameters.")

    uncertainty=fit_historical_uncertainty(fit,inputs,sph)
    target=Path(output) if output else (
        OUTPUT/group.lower()/
        f"{origin:%Y-%m-%d_%H}h_k{k}_N{N}_{B}.png")
    target.parent.mkdir(parents=True,exist_ok=True)
    rows=_plot(starts,frame,k,N,B,sph,fit,model,inputs,target,uncertainty)
    # The baseline and shared theta are historical; future observations enter
    # ONLY this independent two-state oracle optimization and its evaluation.
    prediction_data=_window(frame,origin,k,baseline,sph)
    oracle=optimized_qp_params(model).optimize_prediction_initial_state(
        prediction_data,np.asarray(fit["r"],dtype=float))
    oracle_path=target.with_name(target.stem+"_oracle_prediction.png")
    oracle_result=_plot_oracle(origin,frame,k,N,B,sph,prediction_data,oracle,oracle_path,uncertainty)
    report=pd.DataFrame(rows)
    report.to_csv(target.with_suffix(".csv"),index=False)
    target.with_suffix(".json").write_text(json.dumps({
        "prediction_origin":origin.isoformat(),"horizon_hours":k,
        "historical_weekdays":N,"baseline":B,"group":group,
        "steps_per_hour":sph,"fitting_mode":"exact N corresponding k-hour intervals",
        "theta_F":float(fit["r"][0]),"theta_U":float(fit["r"][1]),
        "theta_R":float(fit["r"][2]),"qp_loss":float(fit["qp_loss"]),
        "process_sigma":uncertainty["sigma"],"observation_tau":uncertainty["tau"],
        "process_variance_rate":uncertainty["sigma2"],
        "observation_variance":uncertainty["tau2"],
        "process_identifiable":uncertainty["identifiable"],
        "historical_coverage_95":uncertainty["historical_coverage_95"],
        "oracle_coverage_95":oracle_result["oracle_coverage_95"],
        "uncertainty":"95% pointwise Gaussian; historical-only variance fit, oracle-conditioned target",
        "oracle_F0":oracle_result["oracle_F0"],
        "oracle_R0":oracle_result["oracle_R0"],
        "oracle_rmse":oracle_result["oracle_rmse"],
        "oracle_mse":oracle_result["oracle_mse"],
        "oracle_mae":oracle_result["oracle_mae"],
        "oracle_bias":oracle_result["oracle_bias"],
        "prediction_baseline_rmse":oracle_result["baseline_rmse"],
        "oracle_prediction_plot":str(oracle_path),
        "oracle_prediction_csv":str(oracle_path.with_suffix(".csv")),
        "imputed_historical_hours":int(imputation["imputed_hourly_slots"]),
        "historical_intervals":[{"start":x.isoformat(),
                                 "end":(x+pd.Timedelta(hours=k)).isoformat()} for x in starts],
    },indent=2),encoding="utf-8")
    print(f"WM4 historical fitting example: {origin:%Y-%m-%d %H:%M} | "
          f"k={k} | N={N} | B={B} | one shared theta, N initial states")
    print(report.to_string(index=False,float_format=lambda v:f"{v:.6f}"))
    print(f"\nHistorical fitting grid: {target}")
    print(f"Oracle prediction plot: {oracle_path}")
    print(f"Oracle F0={oracle_result['oracle_F0']:.6f}, "
          f"R0={oracle_result['oracle_R0']:.6f}, "
          f"RMSE={oracle_result['oracle_rmse']:.6f}")
    print(f"Historical noise fit: σ={uncertainty['sigma']:.6f}, "
          f"τ={uncertainty['tau']:.6f}, "
          f"historical_coverage_95={uncertainty['historical_coverage_95']:.3f}, "
          f"oracle_coverage_95={oracle_result['oracle_coverage_95']:.3f}")
    return {"plot":target,"oracle_plot":oracle_path,
            "summary":report,"oracle":oracle_result,"fit":fit,
            "uncertainty":uncertainty,
            "baseline":baseline,"historical_intervals":starts,
            "imputation":imputation}


def main():
    parser=ArgumentParser(description="Plot N weekly-offset WM4 historical fits in one grid and the target oracle in a second PNG.")
    parser.add_argument("--day",required=True,help="Prediction day YYYY-MM-DD (e.g. 2022-08-20).")
    parser.add_argument("--hour",type=int,required=True,help="Origin hour 0..23.")
    parser.add_argument("--k",type=int,default=12,help="Prediction horizon in hourly observations.")
    parser.add_argument("--N",type=int,default=4,help="Number of historical corresponding weekdays.")
    parser.add_argument("--B","--baseline",dest="B",choices=tuple(BASELINES),
                        default="weekly_temperature",help="Baseline formulation.")
    parser.add_argument("--group",choices=("MED","ALL"),default="MED")
    parser.add_argument("--steps-per-hour",type=int,default=4)
    parser.add_argument("--output",default=None,help="Historical grid PNG path (oracle PNG and companion CSV/JSON alongside).")
    args=parser.parse_args()
    if args.steps_per_hour<2: parser.error("--steps-per-hour must be >= 2.")
    return run(day=args.day,hour=args.hour,k=args.k,N=args.N,B=args.B,
               group=args.group,steps_per_hour=args.steps_per_hour,output=args.output)


if __name__=="__main__": main()
