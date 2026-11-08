"""Baseline-only in-sample and out-of-sample N,k demand-quality heatmaps.

Replays the EXACT weekly-average and weekly-temperature baseline parameters
saved by sweep_weekday_horizon.py. Uses the same successful hourly origins,
N corresponding historical k-hour intervals, historical-day imputation, and
k target observations as the ARM historical-fit and oracle quality figures.

This executable never fits a baseline, state, or ARM parameter. For each saved
fit row it scores the N historical intervals (in-sample) and the subsequent
target interval (out-of-sample, no oracle). Metrics are pooled over individual
observations before computing RMSE, MAE and model-minus-observed bias.
"""
from argparse import ArgumentParser
from collections import defaultdict
from csv import DictReader
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.moving_horizon_common import group_frame
from src.exe.run_model.wm_4.plot_demand_metric_grid import draw_demand_metric_grid
from src.exe.run_model.wm_4.plot_moving_horizon_quality import (
    bounds, common_origins, configuration, saved_baseline, selected,
)
from src.exe.run_model.wm_4.sweep_weekday_horizon import OUTPUT as SWEEP_OUTPUT
from src.tools.impute_hourly import impute_hourly_days


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/"files"/"output"/"wm_4"/"baseline_moving_horizon_quality"
BASELINES = ("weekly_average", "weekly_temperature")
TITLES = {"weekly_average":"Weekly average",
          "weekly_temperature":"Temperature-adjusted weekly average"}
METRICS = {"rmse":"rmse", "mae":"mae", "bias":"bias"}


def pool_record():
    return {"square":0., "absolute":0., "signed":0., "count":0, "origins":0}


def accumulate(pool, key, error):
    e = np.asarray(error, dtype=float).reshape(-1)
    if not e.size or not np.isfinite(e).all():
        raise ValueError(f"Empty or nonfinite demand residuals for {key}.")
    item = pool[key]
    item["square"] += float(np.dot(e,e))
    item["absolute"] += float(np.abs(e).sum())
    item["signed"] += float(e.sum())
    item["count"] += len(e)
    item["origins"] += 1


def summarize(pool):
    rows = []
    for (baseline,n,k),value in sorted(pool.items()):
        count = value["count"]
        if count<=0: raise ValueError("Cannot summarize an empty group.")
        rows.append({
            "baseline":baseline, "training_weekdays":n, "horizon_hours":k,
            "origins":value["origins"], "observations":count,
            "rmse":float(np.sqrt(value["square"]/count)),
            "mae":value["absolute"]/count, "bias":value["signed"]/count,
        })
    return pd.DataFrame(rows)


def historical_index(row):
    """Exactly N corresponding historical starts and each hour 0..k-1."""
    origin = pd.Timestamp(row["day"])
    n, k = int(row["training_weekdays"]), int(row["horizon_hours"])
    starts = pd.DatetimeIndex([pd.Timestamp(x) for x in row["training_days"].split(";") if x])
    if len(starts)!=n or starts.has_duplicates:
        raise ValueError(f"Historical starts do not match N={n} for origin {origin}.")
    if any(t>=origin or t.dayofweek!=origin.dayofweek or t.hour!=origin.hour
           for t in starts):
        raise ValueError(f"Historical starts do not match corresponding weekdays for {origin}.")
    return pd.DatetimeIndex(
        (starts.to_numpy(dtype="datetime64[ns]")[:,None]+
         np.arange(k).astype("timedelta64[h]")[None,:]).reshape(-1))


def baseline_errors(row, historical, target):
    """Replay a SAVED baseline on precisely the ARM historical/target samples."""
    n, k = int(row["training_weekdays"]), int(row["horizon_hours"])
    model = saved_baseline(row)
    history_index = historical_index(row)
    history = historical.reindex(history_index)[["I","Temp"]]
    target_index = pd.date_range(pd.Timestamp(row["day"]),periods=k,freq="h")
    prediction = target.reindex(target_index)[["I","Temp"]]
    if history.isna().any().any() or prediction.isna().any().any():
        raise ValueError("Incomplete historical or target baseline observations.")
    historical_demand = np.asarray(history["I"],dtype=float)
    target_demand = np.asarray(prediction["I"],dtype=float)
    historical_baseline = np.asarray(
        model.predict(history_index,history["Temp"].to_numpy(dtype=float)),dtype=float)
    target_baseline = np.asarray(
        model.predict(target_index,prediction["Temp"].to_numpy(dtype=float)),dtype=float)
    if historical_baseline.shape!=(n*k,) or target_baseline.shape!=(k,):
        raise ValueError("Saved baseline predictions have unexpected shapes.")
    if not np.isfinite(np.r_[historical_baseline,target_baseline]).all():
        raise ValueError("Saved baseline produced a nonfinite demand prediction.")
    return historical_baseline-historical_demand, target_baseline-target_demand


def evaluate(source, group, selection, cohort=None, progress_every=10000):
    """Stream original successful N,k fits, without re-estimating their baseline."""
    fits = Path(source)/"parameter_estimates.csv"
    if not fits.is_file(): raise FileNotFoundError(fits)
    target = group_frame(group)
    historical,_ = impute_hourly_days(target,columns=("I","U","Temp"))
    fitting = defaultdict(pool_record)
    validation = defaultdict(pool_record)
    scanned, included = 0, 0
    with fits.open(newline="",encoding="utf-8-sig") as file:
        for row in DictReader(file):
            scanned += 1
            if not selected(row,*selection): continue
            origin = pd.Timestamp(row["day"])
            if cohort is not None and origin not in cohort: continue
            key = (row["baseline"],int(row["training_weekdays"]),int(row["horizon_hours"]))
            history_error, target_error = baseline_errors(row,historical,target)
            accumulate(fitting,key,history_error)
            accumulate(validation,key,target_error)
            included += 1
            if included%progress_every==0:
                print(f"[baseline quality] selected={included:,}; scanned={scanned:,}",flush=True)
    if included==0: raise ValueError("No saved N,k baseline fits satisfy the selection.")
    fit, prediction = summarize(fitting), summarize(validation)
    joined = fit.merge(prediction,on=["baseline","training_weekdays","horizon_hours"],
                       suffixes=("_fit","_prediction"),validate="one_to_one")
    if (joined["origins_fit"]!=joined["origins_prediction"]).any():
        raise RuntimeError("Historical and target origin cohorts differ.")
    if not (joined["observations_fit"]==
            joined["origins_fit"]*joined["training_weekdays"]*joined["horizon_hours"]).all():
        raise RuntimeError("In-sample observations are not N*k per origin.")
    if not (joined["observations_prediction"]==
            joined["origins_prediction"]*joined["horizon_hours"]).all():
        raise RuntimeError("Out-of-sample observations are not k per origin.")
    print(f"[baseline quality] selected={included:,}; scanned={scanned:,}; "
          f"QP fits performed=0; baseline fits performed=0",flush=True)
    return fit,prediction


def plot(fit,prediction,output,group,cohort):
    """Use the same three-row ARM renderer and identical in/out color scales."""
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    scale = {name:max(float(fit[name].abs().max()),
                      float(prediction[name].abs().max()),1e-12)
             for name in METRICS}
    common = {"baselines":BASELINES, "titles":TITLES, "group":group,
              "period":"same successful hourly origin sweep as ARM",
              "subtitle":f"Cohort: {cohort} | same saved N,k-specific historical baseline fits",
              "horizon_label":"Historical fitting / prediction horizon k (hours)",
              "rmse_vmax":scale["rmse"], "mae_vmax":scale["mae"],
              "bias_vmax":scale["bias"]}
    results = {}
    for name,frame,title in (
        ("fit",fit,"Baseline-only | historical fit (in-sample) | RMSE / MAE / BIAS"),
        ("prediction",prediction,"Baseline-only | fit validation (out-of-sample) | RMSE / MAE / BIAS"),
    ):
        destination = output/f"{name}_cumulative_quality.png"
        fig = draw_demand_metric_grid(
            frame,METRICS,evaluation=name,output=destination,
            title=f"{title} | {group}",**common)
        plt.close(fig)
        frame.to_csv(output/f"{name}_cumulative_quality.csv",index=False)
        results[name] = destination
    return results


def run(args):
    source = Path(args.source) if args.source else SWEEP_OUTPUT/args.group.lower()
    output = Path(args.output) if args.output else OUTPUT/args.group.lower()
    config = configuration(source)
    if config["group"]!=args.group:
        raise ValueError(f"Saved sweep uses group {config['group']}, not {args.group}.")
    selection = bounds(config,args)
    if set(selection[0])!=set(BASELINES) or len(selection[0])!=2:
        raise ValueError("Comparable baseline heatmaps require weekly_average and weekly_temperature.")
    cohort = common_origins(source,selection) if args.cohort=="matched" else None
    if cohort is not None:
        print(f"[baseline quality] matched hourly origins={len(cohort):,}",flush=True)
    fit,prediction = evaluate(source,args.group,selection,cohort,args.progress_every)
    figures = plot(fit,prediction,output,args.group,args.cohort)
    metadata = {
        "group":args.group, "source":str(source), "cohort":args.cohort,
        "baselines":list(BASELINES), "historical_backlogs":list(selection[1]),
        "horizons":list(selection[2]), "start_day":args.start_day,
        "end_day":args.end_day, "fit":"same N historical k-hour observations as ARM",
        "validation":"same next k target observations as ARM; no oracle state",
        "historical_imputation":"identical within-day hourly imputation as ARM",
        "saved_parameters":"original N,k-specific baseline coefficients; no refitting",
        "aggregation":"pooled observation errors; bias=baseline-observed",
        "overlapping_origins":"retained separately; not independent",
    }
    output.mkdir(parents=True,exist_ok=True)
    (output/"configuration.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    for key,path in figures.items(): print(f"[baseline quality] {key}: {path}",flush=True)
    return {"fit":fit,"prediction":prediction,"plots":figures,"configuration":metadata}


def main():
    parser = ArgumentParser(
        description="Baseline-only matched N,k historical fitting and validation RMSE/MAE/bias heatmaps from saved ARM sweep.")
    parser.add_argument("--group",choices=("ALL","MED"),default="ALL")
    parser.add_argument("--source",default=None,help="Completed ARM N,k sweep directory.")
    parser.add_argument("--output",default=None,help="Destination directory.")
    parser.add_argument("--baselines",nargs="+",choices=BASELINES,default=list(BASELINES))
    parser.add_argument("--min-weekdays",type=int,default=None)
    parser.add_argument("--max-weekdays",type=int,default=None)
    parser.add_argument("--min-horizon",type=int,default=None)
    parser.add_argument("--max-horizon",type=int,default=None)
    parser.add_argument("--start-day",default=None)
    parser.add_argument("--end-day",default=None)
    parser.add_argument("--cohort",choices=("all","matched"),default="all",
                        help="all: each (N,k)'s successful origins; matched: same origins across every selected N,k and baseline.")
    parser.add_argument("--progress-every",type=int,default=10000)
    args = parser.parse_args()
    if args.progress_every<1: parser.error("--progress-every must be positive.")
    return run(args)


if __name__=="__main__": main()
