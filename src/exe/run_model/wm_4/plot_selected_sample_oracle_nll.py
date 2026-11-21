"""Plot the best/worst ARM oracle target days from the 12-day IM comparison.

Rank by mean marginal Gaussian NLL over lead hours 1..23, exactly as in
selected_sample_nll.py. All 24 observed/expected demand points and the
central 95% marginal observation intervals are plotted. Historical drift,
variance, and target oracle states are refitted by the existing sample
routine; there is no alternative estimator or target variance fitting.

Run from repository root:
    python -m src.exe.run_model.wm_4.plot_selected_sample_oracle_nll
    python -m src.exe.run_model.wm_4.plot_selected_sample_oracle_nll --rank-by rmse
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.method.arm_variance import variance_kernel
from src.method.sample_timing import timing_report
from src.method.moving_horizon import group_frame
from src.exe.run_model.wm_4.selected_sample_nll import (
    ROOT, DAYS, N, K, BASELINE, GROUP, DEFAULT_MIN_TAU, score_day,
)


OUTPUT=ROOT/"files"/"output"/"wm_4"/"selected_sample_oracle_plots"
Z95=float(norm.ppf(0.975))
ORANGE="tab:orange"


def evaluate_days(year,steps_per_hour,min_tau):
    """Use the exact historical variance fitting and oracle scoring of the NLL sample."""
    frame=group_frame(GROUP)
    results=[]
    for number,month_day in enumerate(DAYS,1):
        day=pd.Timestamp(f"{year}-{month_day}")
        (fit,history,_,_,_,_,target_nll,scored_variance,tau2,sigma2,_)=score_day(
            frame,day,steps_per_hour,min_tau)
        observed=np.asarray(fit["observed"],dtype=float)
        predicted=np.asarray(fit["predicted"],dtype=float)
        theta=np.asarray(fit["theta"],dtype=float)
        kernel=variance_kernel(float(theta[0]),float(theta[2]),steps_per_hour,K)
        variance=np.asarray(tau2+sigma2*kernel,dtype=float)
        if observed.shape!=(K,) or predicted.shape!=(K,) or variance.shape!=(K,):
            raise ValueError(f"Invalid 24-hour sample trajectory for {day.date()}.")
        if not np.isfinite(np.r_[observed,predicted,variance]).all() or np.any(variance<0):
            raise ValueError(f"Invalid demand or variance for {day.date()}.")
        if not np.allclose(variance[1:],scored_variance,rtol=1e-12,atol=1e-12):
            raise RuntimeError(f"Plotted and likelihood-scored variances disagree: {day.date()}.")
        score=float(np.mean(target_nll))
        result={"day":day,"observed":observed,"predicted":predicted,
                "variance":variance,"nll_hourly":np.r_[np.nan,target_nll],
                "mean_nll":score,"total_nll":float(np.sum(target_nll)),
                "rmse":float(np.sqrt(np.mean((predicted-observed)**2))),
                "historical_days":history,"oracle_initial":np.asarray(fit["initial"],dtype=float),
                "timing":dict(fit["sample_timing_seconds"])}
        results.append(result)
        print(f"[{number:02d}/{len(DAYS)}] {day:%Y-%m-%d} | "
              f"mean marginal NLL={score:+.5f} | oracle RMSE={result['rmse']:.5f}",flush=True)
    return results


def rank_days(results,metric,count):
    if count<1 or 2*count>len(results):
        raise ValueError(f"--count must be in 1..{len(results)//2} for nonoverlapping groups.")
    ordered=sorted(results,key=lambda item:(item["mean_nll"] if metric=="nll"
                                           else item["rmse"],item["day"]))
    return ordered[:count],list(reversed(ordered[-count:])),ordered


def plot_ranked(best,worst,metric,path):
    """Arrange three lowest and highest scoring target days as in the IM figure."""
    count=len(best)
    fig,axes=plt.subplots(count,2,figsize=(16,3.35*count+2.3),sharex=True,sharey=True)
    axes=np.asarray(axes,dtype=object).reshape(count,2)
    hours=np.arange(K)
    items=best+worst
    bounds=np.concatenate([np.r_[item["observed"],item["predicted"]+Z95*np.sqrt(item["variance"]),
                                    item["predicted"]-Z95*np.sqrt(item["variance"])] for item in items])
    bottom=min(0.,float(np.min(bounds)))-0.02
    top=max(0.7,float(np.max(bounds))+0.04)
    score_name="Mean NLL" if metric=="nll" else "RMSE"
    for col,group in enumerate((best,worst)):
        for row,item in enumerate(group):
            ax=axes[row,col]
            mu=item["predicted"]
            sd=np.sqrt(item["variance"])
            ax.fill_between(hours,mu-Z95*sd,mu+Z95*sd,color=ORANGE,alpha=0.17,
                            linewidth=0.5,edgecolor=ORANGE,zorder=1)
            ax.plot(hours,mu,color=ORANGE,lw=2.2,zorder=3)
            ax.plot(hours,item["observed"],color="0.12",lw=1.7,marker="o",
                    markersize=3.2,zorder=4)
            ax.set_title(f"{item['day']:%a %d %b %Y}",loc="left",fontsize=11,pad=13)
            ax.set_title(f"{score_name} = "
                         f"{(item['mean_nll'] if metric=='nll' else item['rmse']):+.5f}",
                         loc="right",fontsize=10,color="0.32",pad=13)
            ax.set_xlim(0,K-1)
            ax.set_ylim(bottom,top)
            ax.set_xticks([0,4,8,12,16,20,23])
            ax.grid(alpha=0.23)
            if col==0: ax.set_ylabel("Normalized demand")
            if row==count-1: ax.set_xlabel("Hour of day")
    fig.suptitle("ARM | 24-hour oracle predictions\n"
                 f"{GROUP}  |  weekly temperature  |  N={N} historical days  |  "
                 f"ranked by {('mean marginal NLL' if metric=='nll' else 'oracle RMSE')}",
                 fontsize=15,y=0.995)
    fig.legend(handles=[Line2D([0],[0],color="0.12",marker="o",markersize=4,lw=1.7,
                               label="Observed demand"),
                        Line2D([0],[0],color=ORANGE,lw=2.2,label="ARM expected demand"),
                        Patch(facecolor=ORANGE,edgecolor=ORANGE,alpha=0.17,
                              label="Central 95% observation interval")],
               loc="upper center",ncol=3,bbox_to_anchor=(0.5,0.921),frameon=False,fontsize=11)
    fig.text(0.27,0.875,f"{count} best days (lowest {metric.upper()})",ha="center",fontsize=12)
    fig.text(0.74,0.875,f"{count} worst days (highest {metric.upper()})",ha="center",fontsize=12)
    fig.text(0.07,0.018,
             f"Selected from {len(DAYS)} target dates. NLL scores hours 1..23; "
             "bands include observation noise; predictions use retrospectively optimized oracle states.",
             fontsize=9.5,color="0.38")
    fig.subplots_adjust(left=0.07,right=0.975,bottom=0.07,top=0.845,hspace=0.46,wspace=0.12)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=190,bbox_inches="tight")
    plt.close(fig)


def save_tables(results,best,worst,metric,output):
    ordered=sorted(results,key=lambda x:(x["mean_nll"] if metric=="nll" else x["rmse"],x["day"]))
    best_days={item["day"] for item in best}
    worst_days={item["day"] for item in worst}
    labels={item["day"]:("best" if item["day"] in best_days else
                         "worst" if item["day"] in worst_days else "other")
            for item in ordered}
    rows=[]
    hourly=[]
    for rank,item in enumerate(ordered,1):
        day=item["day"]
        rows.append({"day":str(day.date()),"rank":rank,"group":labels[day],
                     "mean_marginal_nll_lead_1_23":item["mean_nll"],
                     "total_marginal_nll_lead_1_23":item["total_nll"],
                     "oracle_rmse_24h":item["rmse"],"historical_days":
                     ";".join(str(d.date()) for d in item["historical_days"]),
                     "oracle_F0":float(item["oracle_initial"][0]),
                     "oracle_R0":float(item["oracle_initial"][1])})
        sd=np.sqrt(item["variance"])
        for h in range(K):
            hourly.append({"day":str(day.date()),"hour":h,
                           "observed":float(item["observed"][h]),
                           "expected":float(item["predicted"][h]),
                           "variance":float(item["variance"][h]),
                           "lower_95":float(item["predicted"][h]-Z95*sd[h]),
                           "upper_95":float(item["predicted"][h]+Z95*sd[h]),
                           "marginal_nll":float(item["nll_hourly"][h])})
    pd.DataFrame(rows).to_csv(output/"per_day_ranked.csv",index=False)
    pd.DataFrame(hourly).to_csv(output/"hourly_predictions.csv",index=False)


def run(year=2022,steps_per_hour=60,min_tau=DEFAULT_MIN_TAU,
        metric="nll",count=3,output=OUTPUT):
    if steps_per_hour<2: raise ValueError("--steps-per-hour must be >= 2.")
    if not np.isfinite(min_tau) or min_tau<0: raise ValueError("--min-observation-sigma must be >= 0.")
    if metric not in ("nll","rmse"): raise ValueError("--rank-by must be 'nll' or 'rmse'.")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    results=evaluate_days(year,steps_per_hour,min_tau)
    best,worst,_=rank_days(results,metric,count)
    image=output/f"arm_best_worst_oracle_{metric}.png"
    plot_ranked(best,worst,metric,image)
    save_tables(results,best,worst,metric,output)
    # Reuse the timings measured during each evaluated sample; no second fit.
    timing_report(pd.DataFrame([{"day":str(item["day"].date()),**item["timing"]}
                                for item in results]),"ARM",output,(
        ("Deterministic drift QP","drift_fit_seconds"),
        ("Variance fit (initial + NLL refit)","variance_fit_seconds"),
        ("Oracle initial-state QP","oracle_optimization_seconds"),
        ("Complete process","complete_compute_seconds"),
    ),expected_days=len(results))
    (output/"configuration.json").write_text(json.dumps({
        "year":year,"days":list(DAYS),"group":GROUP,"baseline":BASELINE,
        "N":N,"k":K,"origin_hour":0,"steps_per_hour":steps_per_hour,
        "minimum_observation_sigma":min_tau,"ranking_metric":metric,"group_size":count,
        "nll_scored_hours":"1..23 only; hour 0 omitted from NLL and variance estimation",
        "intervals":"central marginal Gaussian 95%; includes observation noise",
        "oracle":"target initial state optimized retrospectively over 24 observed hours",
    },indent=2),encoding="utf-8")
    print("\nBest:",[(str(x["day"].date()),round(x["mean_nll"] if metric=="nll" else x["rmse"],5)) for x in best])
    print("Worst:",[(str(x["day"].date()),round(x["mean_nll"] if metric=="nll" else x["rmse"],5)) for x in worst])
    print(f"Saved plot: {image}\nSaved ranking: {output/'per_day_ranked.csv'}\n"
          f"Saved hourly data: {output/'hourly_predictions.csv'}",flush=True)
    return image


def main():
    parser=ArgumentParser(description="Plot best/worst 12-day ARM oracle samples, ranked by marginal NLL or RMSE.")
    parser.add_argument("--year",type=int,default=2022)
    parser.add_argument("--steps-per-hour",type=int,default=60)
    parser.add_argument("--min-observation-sigma",type=float,default=DEFAULT_MIN_TAU)
    parser.add_argument("--rank-by",choices=("nll","rmse"),default="nll")
    parser.add_argument("--count",type=int,default=3)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    run(args.year,args.steps_per_hour,args.min_observation_sigma,args.rank_by,args.count,args.output)


if __name__=="__main__": main()
