"""BM1 single-day fit: historical fixed-beta MLE and retrospective target oracle.

Fit shared stochastic parameters and N independent initial beta densities on
N corresponding weekly-offset k-hour windows. The target uses a separately
optimized initial density, with open-loop (unfiltered) marginal predictions.
Target observations never enter historical parameter estimation.
"""
from argparse import ArgumentParser
from math import ceil, sqrt
from pathlib import Path
from time import perf_counter
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.method.moving_horizon import BASELINES
from src.method.single_day import prepare, interval_score, pooled_scores
from src.model.bm_1 import bm_1
from src.validation.exe.run_model.bm_1.fit_beta_moving_horizon import bm_1_beta_moving_horizon_validation as checks


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/"files"/"output"/"bm_1"/"single_day"


def _save_plots(origin, history, k, N, baseline, group, hist, forecasts, target, oracle, scores, output):
    x = origin.hour + np.arange(k)
    ticks = np.unique(np.r_[x[::max(1,ceil(k/6))],x[-1]])
    fig,ax = plt.subplots(figsize=(12,5.2))
    mean = np.asarray(oracle["prediction"]["mean"],dtype=float)
    ax.fill_between(x,oracle["lower_95"],oracle["upper_95"],color="tab:red",alpha=.18,
                    label="95% observation interval")
    ax.plot(x,target["I"],color="0.15",lw=1.7,marker="o",markersize=3.5,label="Observed demand")
    ax.plot(x,target["B"],color="tab:orange",lw=1.7,ls="--",label="Historical baseline")
    ax.plot(x,mean,color="tab:red",lw=2.1,label="BM1 open-loop oracle expected demand")
    ax.set_xticks(ticks,[f"{int(hour)%24:02d}" for hour in ticks])
    ax.set_xlabel("Clock hour (wraps at midnight)")
    ax.set_ylabel("Normalized demand")
    ax.set_title(f"BM1 target | {origin:%Y-%m-%d %H:%M} | k={k}, N={N}, {baseline}, {group}\n"
                 f"Oracle RMSE={scores['rmse']:.4f}, marginal NLL/h={scores['nll_mean']:.4f}, "
                 f"coverage={scores['coverage_95_pct']:.1f}%")
    ax.grid(alpha=.2); ax.legend(fontsize=9); fig.tight_layout()
    target_plot=output/"target.png"
    fig.savefig(target_plot,dpi=180,bbox_inches="tight"); plt.close(fig)

    columns=min(3,int(ceil(sqrt(N))))
    rows=int(ceil(N/columns))
    fig,axes=plt.subplots(rows,columns,figsize=(5.7*columns,3.8*rows),squeeze=False)
    for i,(origin_i,data,forecast) in enumerate(zip(history,hist,forecasts)):
        ax=axes.flat[i]
        lower,upper=forecast["interval"]
        hours=origin_i.hour+np.arange(k)
        ax.fill_between(hours,lower,upper,alpha=.15,color="tab:blue")
        ax.plot(hours,data["I"],color="0.2",marker="o",ms=2.5,lw=1.3,label="Observed")
        ax.plot(hours,data["B"],color="tab:orange",lw=1.5,ls="--",label="Baseline")
        ax.plot(hours,forecast["mean"],color="tab:blue",lw=1.7,
                label="BM1 conditional mean")
        ax.set_title(f"{origin_i:%Y-%m-%d %H:%M}",fontsize=10)
        ax.set_xticks(hours[::max(1,ceil(k/5))],
                      [f"{int(h)%24:02d}" for h in hours[::max(1,ceil(k/5))]])
        ax.grid(alpha=.16)
    for ax in axes.flat[N:]: ax.axis("off")
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.96),ncol=3)
    fig.suptitle(f"BM1 historical fixed-beta fit | k={k}, N={N}, {baseline}\n"
                 "Historical prediction means are one-step-ahead (filtered)",y=.998)
    fig.tight_layout(rect=(0,0,1,.90))
    historical_plot=output/"historical.png"
    fig.savefig(historical_plot,dpi=180,bbox_inches="tight"); plt.close(fig)
    return {"target_plot":str(target_plot),"historical_plot":str(historical_plot)}


def run(day, origin_point=0, k=24, N=6, baseline="weekly_temperature", *,
        group="ALL", degree=6, quadrature=48, steps_per_hour=2,
        maxiter=12, maxfun=400, oracle_maxiter=35, strict=False, quiet=False,
        output=None):
    """Fit one target interval with shared historical BM1 dynamics.

    day: YYYY-MM-DD; origin_point: hour 0..23; k: horizon 2..24 hours;
    N: number of corresponding prior weekdays; baseline: name in BASELINES.
    """
    if any(int(v)<1 for v in (maxiter,maxfun,oracle_maxiter)):
        raise ValueError("Optimizer iteration limits must be positive.")
    prepared=prepare(day,origin_point,k,N,baseline,group=group,bm_price=True)
    origin,history=prepared["origin"],prepared["history"]
    training,raw=prepared["training"],prepared["raw"]
    B=prepared["baseline"]
    hist=[checks.window(training,start,k,B) for start in history]
    target=checks.window(raw,origin,k,B)
    model=bm_1(lambda t:.5,lambda t:.5,(0.,float(k)),k+1)
    estimator=model.fixed_beta_estimator(degree,quadrature,steps_per_hour)

    start=perf_counter()
    seed=estimator.initial_guess(hist)
    fitted=estimator.fit(hist,seed=seed["parameters"],
                         initial_distributions=seed["initial_distributions"],
                         maxiter=maxiter,maxfun=maxfun,progress=not quiet)
    if not fitted["converged"]:
        if strict: raise RuntimeError(f"BM1 stochastic MLE did not converge: {fitted['message']}")
        print(f"[BM1] MLE warning: {fitted['message']}")
    oracle=estimator.oracle(target,fitted,maxiter=oracle_maxiter)
    if not oracle["oracle_converged"]:
        if strict: raise RuntimeError(f"BM1 oracle did not converge: {oracle['oracle_message']}")
        print(f"[BM1] Oracle warning: {oracle['oracle_message']}")

    hist_rows, forecasts, scores=[],[],[]
    for date,data,initial in zip(history,hist,fitted["initial_distributions"]):
        forecast=estimator.solver.simulate(data,fitted,initial,observations=True)
        lower,upper=estimator.solver.intervals(forecast,fitted["sigma_y"])
        scores.append(interval_score(data["I"],forecast,lower,upper))
        forecasts.append({"mean":forecast["mean"],"interval":(lower,upper)})
        for h in range(k):
            hist_rows.append({"start":date.isoformat(),"timestamp":str(data["timestamps"][h]),
                              "hour":h,"observed":data["I"][h],"baseline":data["B"][h],
                              "conditional_mean":forecast["mean"][h],"lower_95":lower[h],
                              "upper_95":upper[h],"conditional_nll":-forecast["loglik"][h]})
    historical=pooled_scores(scores)
    if not np.isclose(historical["nll_total"],fitted["nll"],rtol=1e-9,atol=1e-5):
        raise RuntimeError("Reconstructed historical conditional NLL disagrees with the MLE objective.")
    target_score=interval_score(target["I"],oracle["prediction"],
                                oracle["lower_95"],oracle["upper_95"])
    combined=pooled_scores(scores+[target_score])
    target_rows=[{"timestamp":str(target["timestamps"][h]),"lead_hour":h,
                  "observed":target["I"][h],"baseline":target["B"][h],
                  "price":target["U"][h],"oracle_mean":oracle["prediction"]["mean"][h],
                  "lower_95":oracle["lower_95"][h],"upper_95":oracle["upper_95"][h],
                  "marginal_nll":-oracle["prediction"]["loglik"][h]}
                 for h in range(k)]

    output=Path(output) if output is not None else OUTPUT/group.lower()/(
        f"{origin:%Y-%m-%d_%H}h_k{k}_N{N}_{baseline}")
    output.mkdir(parents=True,exist_ok=True)
    plots=_save_plots(origin,history,k,N,baseline,group,hist,forecasts,target,oracle,
                      target_score,output)
    pd.DataFrame(hist_rows).to_csv(output/"historical_hourly.csv",index=False)
    pd.DataFrame(target_rows).to_csv(output/"target_hourly.csv",index=False)
    summary={"model":"bm_1","day":str(pd.Timestamp(day).date()),"origin_point":int(origin_point),
             "prediction_origin":origin.isoformat(),"k":int(k),"N":int(N),
             "baseline":baseline,"group":group,
             "historical_intervals":[{"start":t.isoformat(),
                                      "end":(t+pd.Timedelta(hours=k)).isoformat()}
                                     for t in history],
             "historical_fit":"N independent initial beta densities; shared seven-parameter stochastic MLE",
             "target_fit":"retrospective oracle initial density; open-loop marginal prediction",
             "parameters":{name:float(fitted[name]) for name in estimator.NAMES},
             "historical_initial_distributions":np.asarray(fitted["initial_distributions"]).tolist(),
             "oracle_initial_distribution":np.asarray(oracle["initial_distribution"]).tolist(),
             "historical":historical,"target":target_score,"combined":combined,
             "historical_mle_converged":bool(fitted["converged"]),
             "oracle_converged":bool(oracle["oracle_converged"]),
             "mle_evaluations":int(fitted["evaluations"]),
             "oracle_evaluations":int(oracle["oracle_evaluations"]),
             "runtime_seconds":float(perf_counter()-start),
             "imputation":prepared["imputation"],**plots}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),
                                       encoding="utf-8")
    print(f"[BM1] {origin:%Y-%m-%d %H:%M} | k={k}, N={N}, B={baseline} "
          f"| historical NLL/h={historical['nll_mean']:.5f} "
          f"| oracle RMSE={target_score['rmse']:.5f}, NLL/h={target_score['nll_mean']:.5f}")
    print(f"[BM1] Output: {output}")
    return {"summary":summary,"fit":fitted,"oracle":oracle,"baseline":B,
            "historical":hist,"target":target,"output":output}


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--day",required=True,help="Target calendar day YYYY-MM-DD.")
    parser.add_argument("--origin-point",type=int,default=0,help="Hour of day, 0..23.")
    parser.add_argument("--k",type=int,default=24,help="Interval length in hours (2..24).")
    parser.add_argument("--N",type=int,default=6,help="Number of historical weekly intervals.")
    parser.add_argument("--baseline",choices=tuple(BASELINES),default="weekly_temperature")
    parser.add_argument("--group",choices=("ALL","MED"),default="ALL")
    parser.add_argument("--degree",type=int,default=6)
    parser.add_argument("--quadrature",type=int,default=48)
    parser.add_argument("--steps-per-hour",type=int,default=2)
    parser.add_argument("--maxiter",type=int,default=12)
    parser.add_argument("--maxfun",type=int,default=400)
    parser.add_argument("--oracle-maxiter",type=int,default=35)
    parser.add_argument("--strict",action="store_true")
    parser.add_argument("--quiet",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    return run(args.day,args.origin_point,args.k,args.N,args.baseline,group=args.group,
               degree=args.degree,quadrature=args.quadrature,
               steps_per_hour=args.steps_per_hour,maxiter=args.maxiter,maxfun=args.maxfun,
               oracle_maxiter=args.oracle_maxiter,strict=args.strict,
               quiet=args.quiet,output=args.output)


if __name__=="__main__": main()
