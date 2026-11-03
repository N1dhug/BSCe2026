"""One grid of M successive hourly WM4 oracle predictions.

For each origin t_m=t_0+m hours, independently:
  1. Fit B on N exact weekly-offset k-hour historical intervals.
  2. Jointly fit one WM4 theta across the N historical intervals.
  3. Optimize [F0,R0] for this ONE target interval (retrospective oracle).
  4. Collect the prediction and its error, then advance the origin one hour.

Plots contain only the M target predictions (no historical-fit panels).
All panels use ONE fixed time axis and the same observed-demand context,
covering the union of the 24/25-hour windows. The moving origin and
k-hour interval visibly shift right by one hour per subplot. Historical
baseline is dashed orange and oracle WM4 prediction red, both restricted
to the moving k-hour interval. Target demand
is never used to estimate B or theta, only the oracle initial state.
"""
from argparse import ArgumentParser
from math import ceil, sqrt
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.moving_horizon_common import group_frame, metrics
from src.exe.run_model.wm_4.plot_corresponding_interval_fit import (
    BASELINES, _fit_sparse_baseline, _intervals, _origin, _window,
)
from src.model.wm_4 import wm_4
from src.params.wm_4 import joint_qp_params, optimized_qp_params
from src.tools.impute_hourly import impute_hourly_days


ROOT=Path(__file__).resolve().parents[4]
OUTPUT=ROOT/"files"/"output"/"wm_4"/"moving_horizon_predictions"


def _origins(day,hour,M):
    """M prediction origins in hourly steps, including the initial origin."""
    if isinstance(M,(bool,np.bool_)) or not isinstance(M,(int,np.integer)) or M<1:
        raise ValueError("M must be a positive integer number of prediction origins.")
    start=_origin(day,hour)
    return [start+pd.Timedelta(hours=m) for m in range(M)]


def _predict_origin(frame,origin,k,N,B,sph,model):
    """Independent historical refit and ONE target-interval oracle prediction."""
    starts=_intervals(origin,k,N)
    indices=[pd.date_range(start,periods=k,freq="h") for start in starts]
    needed=pd.DatetimeIndex(np.concatenate([index.to_numpy() for index in indices]))
    days=needed.normalize().unique()
    history,imputation=impute_hourly_days(
        frame,days=days,columns=("I","U","Temp"),require_days=True)
    selected=pd.concat([history.reindex(index)[["I","Temp"]]
                        for index in indices]).sort_index()
    baseline=_fit_sparse_baseline(selected,B)
    historical_inputs=[_window(history,start,k,baseline,sph) for start in starts]

    # One shared theta across N histories, newly fitted at this origin.
    fit=joint_qp_params(model).fit_many(historical_inputs)
    theta=np.asarray(fit["r"],dtype=float)
    if theta.shape!=(3,) or not np.isfinite(theta).all():
        raise ValueError(f"Invalid WM4 historical parameters at {origin}.")
    if len(fit["deterministic_states"])!=N or np.asarray(fit["init_cond"]).shape!=(N,2):
        raise RuntimeError(f"Historical joint fit did not return N states at {origin}.")

    # The oracle is deliberately the ONLY operation that uses target demand.
    target=_window(frame,origin,k,baseline,sph)
    oracle=optimized_qp_params(model).optimize_prediction_initial_state(target,theta)
    init=np.asarray(oracle["init_cond"],dtype=float)
    predicted=np.asarray(oracle["deterministic_state"]["I"],dtype=float)
    expected=k*sph+1
    if init.shape!=(2,) or not np.isfinite(init).all():
        raise ValueError(f"Invalid oracle state at {origin}.")
    if predicted.shape!=(expected,) or not np.isfinite(predicted).all():
        raise ValueError(f"Invalid oracle trajectory at {origin}.")
    obs_idx=np.arange(k)*sph
    observed=np.asarray(target["I"],dtype=float)[obs_idx]
    base=np.asarray(target["O"],dtype=float)
    score=metrics(observed,predicted[obs_idx])
    baseline_score=metrics(observed,base[obs_idx])

    return {
        "origin":origin,"theta":theta,"initial_state":init,
        "baseline":base,"prediction":predicted,
        "score":score,"baseline_score":baseline_score,
        "historical_starts":starts,
        "historical_samples":int(baseline.metadata["samples"]),
        "imputed_historical_slots":int(imputation["imputed_hourly_slots"]),
        "fit_qp_loss":float(fit["qp_loss"]),
        "observed_hourly":observed,
    }


def _common_axis(frame,origins,k):
    """One absolute clock axis and the same raw demand on every subplot.

    Use the union of all per-origin 24/25-hour contexts, so moving the
    prediction never moves the axes or clips a later prediction.
    """
    edge=int(ceil((24-k)/2))
    reference=origins[0].normalize()
    left=origins[0]-pd.Timedelta(hours=edge)
    right=origins[-1]+pd.Timedelta(hours=k+edge)
    if left<frame.index.min() or right-pd.Timedelta(hours=1)>frame.index.max():
        raise ValueError("Insufficient observed demand for the common moving-horizon context.")
    index=pd.date_range(left,right-pd.Timedelta(hours=1),freq="h")
    demand=frame.reindex(index)["I"].to_numpy(dtype=float)
    if not np.isfinite(demand).any():
        raise ValueError("No observed demand in the common moving-horizon context.")
    xmin=(left-reference).total_seconds()/3600.0
    xmax=(right-reference).total_seconds()/3600.0
    context_x=np.arange(int(xmin),int(xmax)+1,dtype=float)
    context_y=np.r_[demand,demand[-1]]

    ticks=np.arange(3*int(ceil(xmin/3)),int(xmax)+1,3,dtype=int)
    labels=[]
    for tick in ticks:
        clock=int(tick)%24
        if clock==0:
            date=reference+pd.Timedelta(hours=int(tick))
            labels.append(f"00\n{date:%d %b}")
        else: labels.append(f"{clock:02d}")
    return reference,(xmin,xmax),ticks,labels,context_x,context_y


def _plot_grid(results,frame,k,N,B,sph,output):
    """M predictions on a single fixed x-axis; interval advances per panel."""
    M=len(results)
    cols=min(3,int(ceil(sqrt(M))))
    rows=int(ceil(M/cols))
    fig,axes=plt.subplots(rows,cols,figsize=(6.1*cols,4.1*rows),
                          sharex=True,sharey=True,squeeze=False)
    origins=[item["origin"] for item in results]
    reference,limits,ticks,ticklabels,context_x,context_y=_common_axis(frame,origins,k)
    fine=np.arange(k*sph+1,dtype=float)/sph
    extrema=context_y[np.isfinite(context_y)].tolist()
    for m,item in enumerate(results):
        ax=axes.flat[m]
        start=item["origin"]
        # Coordinates refer to the FIRST origin's calendar day, not to the
        # local prediction origin, so the red interval moves in the grid.
        x0=(start-reference).total_seconds()/3600.0
        x=fine+x0
        ax.axvspan(x0,x0+k,color="tab:red",alpha=.07)
        ax.axvline(x0,color="0.5",lw=.8,ls=":")
        ax.axvline(x0+k,color="0.5",lw=.8,ls=":")
        ax.step(context_x,context_y,where="post",
                color="0.18",lw=1.65,label="Observed demand")
        ax.plot(x,item["baseline"],color="tab:orange",ls="--",lw=1.8,
                label="Historical baseline")
        ax.plot(x,item["prediction"],color="red",lw=2.0,
                label="WM4 oracle prediction")
        ax.set_title(f"Origin {m+1}/{M}: {start:%a %d %b %Y, %H:%M}  |  "
                     f"RMSE {item['score']['rmse']:.4f}",fontsize=10)
        ax.grid(alpha=.18)
        for values in (item["baseline"],item["prediction"]):
            extrema.extend(np.asarray(values)[np.isfinite(values)].tolist())
    for ax in axes.flat[M:]: ax.set_visible(False)

    lower,upper=float(np.min(extrema)),float(np.max(extrema))
    pad=max(.025,.08*(upper-lower))
    for ax in axes.flat[:M]:
        ax.set_xlim(*limits)
        ax.set_ylim(lower-pad,upper+pad)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.tick_params(labelbottom=True,labelleft=True,labelsize=8)

    mean_mse=float(np.mean([item["score"]["mse"] for item in results]))
    fig.suptitle(
        f"WM4 moving-horizon oracle predictions | M={M}, k={k}h, N={N}, B={B}\n"
        f"Hourly origin shift | Aggregate RMSE={sqrt(mean_mse):.4f}",
        fontsize=13)
    fig.supxlabel("Shared timeline: hour of day (dates at midnight)")
    fig.supylabel("Normalized aggregate electricity demand")
    handles,legend_labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="upper center",bbox_to_anchor=(.5,.932),
               ncol=3,fontsize=9)
    fig.tight_layout(rect=(.02,.04,.99,.895))
    fig.savefig(output,dpi=190,bbox_inches="tight")
    plt.close(fig)


def _reports(results,k,N,B,group,sph):
    summary_rows,hourly_rows=[],[]
    for m,item in enumerate(results):
        origin=item["origin"]
        theta=item["theta"]
        init=item["initial_state"]
        score=item["score"]
        bscore=item["baseline_score"]
        summary_rows.append({
            "step":m,"origin":origin,"horizon_hours":k,"backlog":N,
            "baseline":B,"group":group,
            "theta_F":float(theta[0]),"theta_U":float(theta[1]),
            "theta_R":float(theta[2]),
            "oracle_F0":float(init[0]),"oracle_R0":float(init[1]),
            "mse":score["mse"],"rmse":score["rmse"],
            "mae":score["mae"],"bias":score["bias"],
            "baseline_mse":bscore["mse"],"baseline_rmse":bscore["rmse"],
            "qp_loss":item["fit_qp_loss"],
            "historical_samples":item["historical_samples"],
            "imputed_historical_slots":item["imputed_historical_slots"],
        })
        idx=np.arange(k,dtype=int)
        observed=item["observed_hourly"]
        baseline=item["baseline"][idx*sph]
        prediction=item["prediction"][idx*sph]
        hourly_rows.extend({
            "step":m,"origin":origin,
            "timestamp":origin+pd.Timedelta(hours=int(h)),
            "lead_hour":int(h),"observed":float(observed[h]),
            "baseline":float(baseline[h]),"wm4_oracle":float(prediction[h]),
        } for h in idx)

    summary=pd.DataFrame(summary_rows)
    hourly=pd.DataFrame(hourly_rows)
    errors=hourly["wm4_oracle"].to_numpy()-hourly["observed"].to_numpy()
    base_errors=hourly["baseline"].to_numpy()-hourly["observed"].to_numpy()
    aggregate={
        "origins":len(results),"prediction_horizon_hours":k,
        "historical_backlog_weeks":N,"baseline":B,"group":group,
        "steps_per_hour":sph,"origin_stride_hours":1,
        "start_origin":results[0]["origin"].isoformat(),
        "last_origin":results[-1]["origin"].isoformat(),
        "total_forecast_samples":len(hourly),
        "mse":float(np.mean(errors**2)),
        "rmse":float(np.sqrt(np.mean(errors**2))),
        "mae":float(np.mean(np.abs(errors))),
        "bias":float(np.mean(errors)),
        "mean_interval_rmse":float(summary["rmse"].mean()),
        "baseline_mse":float(np.mean(base_errors**2)),
        "baseline_rmse":float(np.sqrt(np.mean(base_errors**2))),
        "baseline_mae":float(np.mean(np.abs(base_errors))),
        "baseline_bias":float(np.mean(base_errors)),
        "evaluation":"retrospective oracle; prediction data optimize initial state only",
        "overlapping_intervals":"hourly-shifted windows overlap when k>1",
        "historical_fitting":"baseline and WM4 theta refitted at every origin",
    }
    return summary,hourly,aggregate


def run(*,day,hour,k,N,B,M,group="MED",steps_per_hour=4,output=None):
    """Fit and plot M independently refitted, hourly-shifted WM4 oracles.

    day, hour : start origin; k : horizon (2..24 hours);
    N : historical weekly backlog; B : baseline name; M : origins.
    """
    origins=_origins(day,hour,M)
    _intervals(origins[0],k,N)  # validate once before loading source data
    if B not in BASELINES: raise ValueError(f"B must be one of {tuple(BASELINES)}.")
    if group not in ("ALL","MED"): raise ValueError("group must be ALL or MED.")
    if isinstance(steps_per_hour,(bool,np.bool_)) or not isinstance(steps_per_hour,(int,np.integer)) or steps_per_hour<2:
        raise ValueError("steps_per_hour must be an integer >= 2.")
    sph=int(steps_per_hour)
    frame=group_frame(group)
    if not isinstance(frame.index,pd.DatetimeIndex) or frame.index.has_duplicates:
        raise ValueError("Source data must have a unique hourly DatetimeIndex.")
    frame=frame.sort_index()
    model=wm_4((0.0,k/24.0),k*sph+1,lambda t:0.0)

    results=[]
    for m,origin in enumerate(origins):
        try: item=_predict_origin(frame,origin,k,N,B,sph,model)
        except Exception as exc:
            raise RuntimeError(f"WM4 moving-horizon step {m+1}/{M} at {origin} failed: {exc}") from exc
        results.append(item)
        print(f"[moving-horizon] {m+1}/{M} {origin:%Y-%m-%d %H:%M}  "
              f"RMSE={item['score']['rmse']:.6f}  "
              f"baseline_RMSE={item['baseline_score']['rmse']:.6f}")

    target=Path(output) if output is not None else (
        OUTPUT/group.lower()/f"{origins[0]:%Y-%m-%d_%H}h_M{M}_k{k}_N{N}_{B}.png")
    if target.suffix.lower()!=".png": raise ValueError("output must be a PNG file path.")
    target.parent.mkdir(parents=True,exist_ok=True)
    _plot_grid(results,frame,k,N,B,sph,target)
    summary,hourly,aggregate=_reports(results,k,N,B,group,sph)
    summary.to_csv(target.with_suffix(".csv"),index=False)
    hourly.to_csv(target.with_name(target.stem+"_hourly.csv"),index=False)
    target.with_suffix(".json").write_text(json.dumps(aggregate,indent=2),encoding="utf-8")
    print(f"\nMoving-horizon prediction grid: {target}")
    print(f"Aggregated oracle RMSE: {aggregate['rmse']:.6f}")
    return {"plot":target,"summary":summary,"hourly":hourly,
            "aggregate":aggregate,"predictions":results}


def main():
    parser=ArgumentParser(
        description="Plot M successive hourly WM4 oracle predictions in one grid; refit baseline and theta each origin.")
    parser.add_argument("--day",required=True,help="Starting origin date YYYY-MM-DD.")
    parser.add_argument("--hour",type=int,required=True,help="Starting origin hour, 0..23.")
    parser.add_argument("--k",type=int,default=12,help="Prediction horizon in hours.")
    parser.add_argument("--N",type=int,default=4,help="Number of preceding corresponding weekdays.")
    parser.add_argument("--B","--baseline",dest="B",choices=tuple(BASELINES),
                        default="weekly_temperature")
    parser.add_argument("--M",type=int,required=True,
                        help="Number of successive hourly origins, including the first origin.")
    parser.add_argument("--group",choices=("MED","ALL"),default="MED")
    parser.add_argument("--steps-per-hour",type=int,default=4)
    parser.add_argument("--output",default=None,help="Path for the one grid PNG (CSV/JSON alongside).")
    args=parser.parse_args()
    return run(day=args.day,hour=args.hour,k=args.k,N=args.N,B=args.B,
               M=args.M,group=args.group,steps_per_hour=args.steps_per_hour,
               output=args.output)


if __name__=="__main__": main()
