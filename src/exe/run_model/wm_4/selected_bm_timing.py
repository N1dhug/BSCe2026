"""Seven-origin computational timing reports shared by IM and ARM sample fitters.

Reads only timings produced during the actual fitting run; no reruns or
estimated/cached timings. Complete time is inclusive of the preceding
stages and is not a fourth additive stage.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def timing_report(daily, model, output, stages, show=False):
    """Save per-day stage costs, mean costs and a four-stage horizontal bar plot.

    stages is ordered (display_name, per_day_metrics_column) tuples. Last
    entry is the inclusive complete process. All seven days must be present.
    """
    output = Path(output)
    if len(daily)!=7 or pd.to_datetime(daily["day"]).dt.normalize().nunique()!=7:
        raise ValueError(f"{model} timing report requires seven distinct prediction dates.")
    if len(stages)!=4 or stages[-1][1] not in ("total_compute_seconds","complete_compute_seconds"):
        raise ValueError("Expected three optimization stages followed by complete process.")
    required = {"day",*(key for _,key in stages)}
    absent = required.difference(daily.columns)
    if absent: raise ValueError(f"{model} timing report missing columns: {sorted(absent)}")
    details = []
    for row in daily.itertuples(index=False):
        day = str(pd.Timestamp(row.day).date())
        for label, key in stages:
            seconds = float(getattr(row,key))
            if not np.isfinite(seconds) or seconds<0.:
                raise ValueError(f"Nonfinite or negative timing for {model}, {day}, {key}.")
            details.append({"model":model,"day":day,"stage":label,"key":key,"seconds":seconds})
    per_day = pd.DataFrame(details)
    group = per_day.groupby(["stage","key"],sort=False)["seconds"]
    summary = group.agg(mean_seconds="mean",std_seconds=lambda x: x.std(ddof=0),
                        min_seconds="min",max_seconds="max").reset_index()
    summary.insert(0,"model",model)
    summary.insert(1,"prediction_days",len(daily))
    complete = per_day.loc[per_day["key"]==stages[-1][1]].set_index("day")["seconds"]
    partial = per_day.loc[per_day["key"]!=stages[-1][1]].groupby("day")["seconds"].sum()
    if (partial>complete+1e-5).any():
        raise ValueError(f"{model} complete compute time is smaller than summed measured stages.")

    output.mkdir(parents=True,exist_ok=True)
    per_day.to_csv(output/"timing_per_day.csv",index=False)
    summary.to_csv(output/"timing_average.csv",index=False)
    labels = summary["stage"].tolist()
    means = summary["mean_seconds"].to_numpy(dtype=float)
    fig,ax=plt.subplots(figsize=(11,5.2))
    color = "tab:orange" if model=="IM" else "tab:blue"
    bars = ax.barh(np.arange(4),means,color=[color]*3+["0.40"],height=0.62)
    ax.set_yticks(np.arange(4),labels)
    ax.invert_yaxis()
    ax.set_xlim(0.,max(means.max()*1.22,1e-9))
    ax.set_xlabel("Mean wall-clock seconds per prediction date (seven dates)")
    ax.set_title(f"{model} | Mean computation cost by stage | seven weekday samples")
    ax.grid(axis="x",alpha=0.2)
    ax.set_axisbelow(True)
    for rect,value in zip(bars,means):
        ax.text(value+max(means.max(),1e-9)*0.015,rect.get_y()+rect.get_height()/2,
                f"{value:.4f} s",va="center",fontsize=10)
    fig.text(0.5,0.01,
             "Complete process includes the three stages plus other data preparation/scoring. "
             "Bars are not additive.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,0.06,1,1))
    image = output/"timing_average.png"
    fig.savefig(image,dpi=190,bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)
    print(f"\n{model} mean per-origin computational time across {len(daily)} dates:")
    for label,value in zip(labels,means): print(f"  {label:30s} {value:.6f} s")
    print(f"Timing plot: {image}")
    print(f"Timing data: {output/'timing_per_day.csv'}, {output/'timing_average.csv'}")
    return summary
