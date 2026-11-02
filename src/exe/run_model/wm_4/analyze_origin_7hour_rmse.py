"""WM4 N=8 matching weekdays / H=7h: pooled oracle RMSE by origin hour 0..17.

Each forecast origin o uses observed evaluation hours o,...,o+6.
The historical joint QP fits ONCE per day using eight earlier matching
weekdays, regardless of origin. At each origin, R0 is the median of the
eight historical residuals at that hour; oracle F0 is optimized afresh
using only the seven target-hour observations in that origin's window.

This is a retrospective oracle comparison, NOT a deployable forecast.
"""
from argparse import ArgumentParser
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.moving_horizon_common import HOURS, group_frame, window_data, metrics
from src.exe.run_model.wm_4.predict_all_moving_horizon import historical_weekdays
from src.exe.run_model.wm_4.sweep_weekday_horizon import BASELINE_NAMES, baseline_from_history
from src.model.wm_4 import wm_4
from src.params.wm_4 import joint_qp_params
from src.tools.impute_hourly import impute_hourly_days


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/"files"/"output"/"wm_4"/"origin_7hour_rmse"
TRAINING_WEEKDAYS, HORIZON = 8, 7
ORIGINS = tuple(range(HOURS-HORIZON+1))
FILES = {"daily": "origin_daily_metrics.csv", "hourly": "origin_hourly_predictions.csv",
         "fail": "skipped_days.csv"}


def oracle_prediction(model, data, observed, theta, historical_r0, steps_per_hour):
    """Exactly minimize the seven-observation SSE over F0 in [-1,1]."""
    sph = int(steps_per_hour)
    positions = np.arange(HORIZON, dtype=int)*sph
    zero = model._solve_deterministic(
        [0.0, historical_r0], data["U"], data["PE"], data["Time"], data["O"], theta)
    base = np.asarray(zero["I"], dtype=float)[positions]
    sensitivity = np.asarray(model.sensitivity({"r":theta})["input"], dtype=float)[positions, 0]
    observed = np.asarray(observed, dtype=float)
    if observed.shape != (HORIZON,) or not np.isfinite(np.r_[observed,base,sensitivity]).all():
        raise ValueError("Seven-hour oracle requires seven finite observed and modeled demand values.")
    denominator = float(sensitivity@sensitivity)
    f0 = float(np.clip((sensitivity@(observed-base))/denominator,-1.0,1.0)) if denominator>1e-28 else 0.0
    return f0, base+f0*sensitivity, np.asarray(data["O"])[positions], denominator


def evaluate_day(frame, day, steps_per_hour=4, baseline_name="weekly_temperature",
                 reference=None):
    """Produce one *complete* day of 18 equal-length seven-hour origin forecasts."""
    day, sph = pd.Timestamp(day).normalize(), int(steps_per_hour)
    if sph < 2: raise ValueError("At least two Euler steps per hour are required.")
    history = historical_weekdays(day, TRAINING_WEEKDAYS)
    training, imputed = impute_hourly_days(
        frame, days=history, columns=("I","U","Temp"), require_days=True)
    baseline = baseline_from_history(training, history, baseline_name)
    history_data = [window_data(training, previous, baseline, sph)[1] for previous in history]
    full_model, target, _, observed = window_data(frame, day, baseline, sph)
    if reference is None:
        fit = joint_qp_params(full_model).fit_many(history_data)
        theta = np.asarray(fit["r"], dtype=float)
        fit_loss = float(fit["qp_loss"])
    else:
        if (int(reference["training_weekdays"]) != TRAINING_WEEKDAYS
                or int(reference["horizon_hours"]) != HORIZON
                or reference["baseline"] != baseline_name):
            raise ValueError("Cached oracle row does not match the N=8/H=7/baseline configuration.")
        theta = np.array([reference["theta_F"],reference["theta_U"],reference["theta_R"]],dtype=float)
        fit_loss = float(reference.get("fit_qp_loss",np.nan))
    origin_model = wm_4((0.0,HORIZON/HOURS), HORIZON*sph+1, lambda t: 0.0)
    if not np.isclose(origin_model.dt,full_model.dt,atol=1e-14,rtol=1e-13):
        raise RuntimeError("Origin model and full-day Euler time grids differ.")

    rows, hourly = [], []
    for origin in ORIGINS:
        start, stop = origin*sph, (origin+HORIZON)*sph+1
        at = origin*sph
        historic_r0 = float(np.median([
            data["I"][at]-data["O"][at] for data in history_data]))
        segment = {name:np.asarray(target[name])[start:stop]
                   for name in ("I","U","PE","Time","O")}
        if any(len(value) != HORIZON*sph+1 for value in segment.values()):
            raise RuntimeError(f"Incomplete subday inputs at origin {origin}.")
        labels = np.asarray(observed[origin:origin+HORIZON],dtype=float)
        f0, pred, baseline_pred, sensitivity_sq = oracle_prediction(
            origin_model,segment,labels,theta,historic_r0,sph)
        score, bscore = metrics(labels,pred), metrics(labels,baseline_pred)
        record = {
            "day":day, "weekday":day.day_name(), "origin_hour":origin,
            "training_weekdays":TRAINING_WEEKDAYS, "horizon_hours":HORIZON,
            "baseline":baseline_name, "training_days":";".join(f"{d:%Y-%m-%d}" for d in history),
            "theta_F":float(theta[0]), "theta_U":float(theta[1]), "theta_R":float(theta[2]),
            "historical_R0":historic_r0, "oracle_F0":f0, "oracle_sensitivity_norm":float(np.sqrt(sensitivity_sq)),
            "rmse":score["rmse"],"mse":score["mse"],"mae":score["mae"],"bias":score["bias"],
            "baseline_rmse":bscore["rmse"],"baseline_mse":bscore["mse"],
            "rmse_gain":bscore["rmse"]-score["rmse"],"fit_qp_loss":fit_loss,
            "imputed_training_hours":int(imputed["imputed_hourly_slots"]),
        }
        rows.append(record)
        for lead in range(HORIZON):
            hour = origin+lead
            hourly.append({
                "day":day,"origin_hour":origin,"lead_hour":lead,
                "absolute_hour":hour,"timestamp":day+pd.Timedelta(hours=hour),
                "observed":float(labels[lead]),"predicted":float(pred[lead]),
                "baseline_prediction":float(baseline_pred[lead]),
                "error":float(pred[lead]-labels[lead]),
                "baseline_error":float(baseline_pred[lead]-labels[lead]),
                "oracle_F0":f0,"historical_R0":historic_r0,
            })
    if len(rows)!=len(ORIGINS) or len(hourly)!=len(ORIGINS)*HORIZON:
        raise RuntimeError("Origin evaluation did not return every required forecast window.")
    if reference is not None:
        origin0 = rows[0]
        if not np.isclose(origin0["rmse"],float(reference["rmse"]),atol=2e-7,rtol=1e-7):
            raise RuntimeError("Origin 00 RMSE differs from saved seven-hour oracle: "
                               f"{origin0['rmse']:.8f} vs {float(reference['rmse']):.8f}.")
    return rows,hourly


def check_scores(daily, hourly):
    d,h = daily.copy(),hourly.copy()
    required_d = {"day","origin_hour","rmse","mse","baseline_mse","oracle_F0","historical_R0"}
    required_h = {"day","origin_hour","lead_hour","error","baseline_error"}
    if not required_d.issubset(d) or not required_h.issubset(h):
        raise ValueError(f"Missing daily columns {sorted(required_d-set(d))} "
                         f"or hourly columns {sorted(required_h-set(h))}.")
    d["day"],h["day"]=pd.to_datetime(d["day"]),pd.to_datetime(h["day"])
    if d.duplicated(["day","origin_hour"]).any() or h.duplicated(["day","origin_hour","lead_hour"]).any():
        raise ValueError("Duplicate origin windows or hourly forecast errors.")
    if d.empty or h.empty: raise ValueError("No saved origin predictions.")
    if set(d["day"]) != set(h["day"]): raise ValueError("Daily and hourly dates differ.")
    for day,g in d.groupby("day"):
        if set(g["origin_hour"].astype(int)) != set(ORIGINS):
            raise ValueError(f"Incomplete origin set for {day}.")
    if len(h) != len(d)*HORIZON or h.groupby(["day","origin_hour"])["lead_hour"].nunique().ne(HORIZON).any():
        raise ValueError("Every origin needs seven complete hourly prediction errors.")
    expected_h = set(range(HORIZON))
    if any(set(group["lead_hour"].astype(int))!=expected_h
           for _,group in h.groupby(["day","origin_hour"])):
        raise ValueError("Invalid lead hours.")
    if not np.isfinite(d[["rmse","mse","baseline_mse"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite saved performance.")
    if not np.isfinite(h[["error","baseline_error"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite hourly residuals.")
    for (day,origin),grp in h.groupby(["day","origin_hour"]):
        mse = float(np.mean(grp["error"].to_numpy(dtype=float)**2))
        ref = d.loc[(d["day"]==day)&(d["origin_hour"]==origin),"mse"].iloc[0]
        if not np.isclose(mse,ref,atol=1e-10,rtol=1e-8):
            raise ValueError(f"Saved MSE does not match individual errors at {day}, origin={origin}.")
    return d,h


def summarize(daily, hourly):
    """Pooled RMSE = sqrt(mean(error^2)) over matched days and seven lead hours."""
    daily,hourly=check_scores(daily,hourly)
    rows,lead_rows=[],[]
    for origin in ORIGINS:
        scores=daily.loc[daily["origin_hour"]==origin]
        part=hourly.loc[hourly["origin_hour"]==origin]
        error=part["error"].to_numpy(dtype=float)
        base=part["baseline_error"].to_numpy(dtype=float)
        rmses=scores["rmse"].to_numpy(dtype=float)
        row={"origin_hour":origin,"last_prediction_hour":origin+HORIZON-1,
             "days":len(scores),"samples":len(error),
             "aggregate_rmse":float(np.sqrt(np.mean(error**2))),
             "baseline_rmse":float(np.sqrt(np.mean(base**2))),
             "aggregate_mse":float(np.mean(error**2)),
             "bias":float(np.mean(error)),
             "mean_daily_rmse":float(np.mean(rmses)),
             "median_daily_rmse":float(np.median(rmses))}
        row["rmse_gain"]=row["baseline_rmse"]-row["aggregate_rmse"]
        for q in (20,40,50,60,80):
            row[f"daily_rmse_q{q}"]=float(np.percentile(rmses,q))
        rows.append(row)
        for lead,g in part.groupby("lead_hour"):
            lead_rows.append({"origin_hour":origin,"lead_hour":int(lead),
                              "absolute_hour":origin+int(lead),"days":len(g),
                              "rmse":float(np.sqrt(np.mean(g["error"]**2))),
                              "baseline_rmse":float(np.sqrt(np.mean(g["baseline_error"]**2)))})
    return pd.DataFrame(rows).sort_values("origin_hour"),pd.DataFrame(lead_rows).sort_values(
        ["origin_hour","lead_hour"])


def plot_results(aggregate,leads,output):
    fig,ax=plt.subplots(figsize=(13,6))
    ax.plot(aggregate["origin_hour"],aggregate["aggregate_rmse"],marker="o",lw=2.2,
            label="WM4 seven-hour oracle, pooled RMSE")
    ax.plot(aggregate["origin_hour"],aggregate["baseline_rmse"],marker=".",ls="--",lw=1.4,
            label="Historical weekly-temperature baseline")
    ax.set_xticks(ORIGINS);ax.set_xlabel("Forecast origin hour (00–17)")
    ax.set_ylabel("Aggregate RMSE (normalized demand)")
    ax.set_title("WM4 N=8, H=7: pooled oracle prediction RMSE by forecast origin")
    ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(output/"origin_aggregate_rmse.png",dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(13,6))
    x=aggregate["origin_hour"].to_numpy(dtype=float)
    ax.plot(x,aggregate["aggregate_rmse"],marker="o",lw=2,label="Pooled RMSE")
    ax.plot(x,aggregate["daily_rmse_q50"],marker=".",label="Median daily seven-hour RMSE")
    ax.fill_between(x,aggregate["daily_rmse_q20"].to_numpy(dtype=float),
                    aggregate["daily_rmse_q80"].to_numpy(dtype=float),alpha=.2,
                    label="Daily RMSE 20th–80th percentile")
    ax.set_xticks(ORIGINS);ax.set_xlabel("Forecast origin hour (00–17)")
    ax.set_ylabel("Seven-hour RMSE");ax.set_title("Day-to-day spread in oracle prediction error")
    ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(output/"origin_rmse_quantiles.png",dpi=180);plt.close(fig)

    matrix=leads.pivot(index="origin_hour",columns="lead_hour",values="rmse").reindex(
        index=ORIGINS,columns=range(HORIZON))
    fig,ax=plt.subplots(figsize=(11,8))
    heat=ax.imshow(matrix.to_numpy(dtype=float),origin="lower",aspect="auto",interpolation="nearest")
    fig.colorbar(heat,ax=ax,label="Pooled RMSE (normalized demand)")
    ax.set_yticks(np.arange(len(ORIGINS)),[f"{i:02d}:00" for i in ORIGINS])
    ax.set_xticks(np.arange(HORIZON),[f"+{i} h" for i in range(HORIZON)])
    ax.set_ylabel("Forecast origin");ax.set_xlabel("Lead hour")
    ax.set_title("WM4 error by origin and forecast lead hour (same day)")
    fig.tight_layout();fig.savefig(output/"origin_lead_rmse_heatmap.png",dpi=180);plt.close(fig)


def report(output):
    output=Path(output)
    daily=pd.read_csv(output/FILES["daily"],parse_dates=["day"])
    hourly=pd.read_csv(output/FILES["hourly"],parse_dates=["day","timestamp"])
    aggregate,leads=summarize(daily,hourly)
    aggregate.to_csv(output/"origin_aggregate_rmse.csv",index=False)
    leads.to_csv(output/"origin_lead_rmse.csv",index=False)
    plot_results(aggregate,leads,output)
    print(f"\nWM4: eight preceding corresponding weekdays, seven-hour oracle F0; "
          f"{daily['day'].nunique()} matched days.")
    print("Each origin 00–17 predicts origin,...,origin+6; 18 origins per day.")
    print(aggregate[["origin_hour","last_prediction_hour","days","samples",
                     "aggregate_rmse","baseline_rmse","rmse_gain","median_daily_rmse"]]
          .to_string(index=False,float_format=lambda x:f"{x:.8f}"))
    print(f"\nBest origin by pooled oracle RMSE: "
          f"{int(aggregate.loc[aggregate['aggregate_rmse'].idxmin(),'origin_hour']):02d}:00")
    print(f"Saved daily/origin/lead CSVs and three plots: {output}")
    print("WARNING: Each 7-hour F0 oracle uses all seven target observations; "
          "these are NOT out-of-sample operational forecasts.")
    return aggregate


def save_rows(output,daily,hourly,skipped):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if daily: pd.DataFrame(daily).to_csv(output/FILES["daily"],index=False)
    if hourly: pd.DataFrame(hourly).to_csv(output/FILES["hourly"],index=False)
    pd.DataFrame(skipped,columns=["day","reason"]).to_csv(output/FILES["fail"],index=False)


def run(args):
    if args.steps_per_hour<2 or args.progress_every<1 or args.max_days<0:
        raise ValueError("Require steps-per-hour >=2, progress-every >=1 and max-days >=0.")
    output=Path(args.output) if args.output else OUTPUT/args.group.lower()/args.baseline
    if args.report_only: return report(output)
    config={"group":args.group,"baseline":args.baseline,"steps_per_hour":args.steps_per_hour,
            "N":TRAINING_WEEKDAYS,"H":HORIZON,"min_origin":ORIGINS[0],"max_origin":ORIGINS[-1],
            "start_day":args.start_day,"end_day":args.end_day,
            "from_oracle":str(args.from_oracle) if args.from_oracle else None}
    output.mkdir(parents=True,exist_ok=True)
    configuration=output/"configuration.json"
    reference=None
    if args.from_oracle:
        source=Path(args.from_oracle)
        if source.is_dir(): source=source/"daily_metrics.csv"
        reference=pd.read_csv(source,parse_dates=["day"])
        if reference.duplicated("day").any():
            raise ValueError("Cached oracle parameter file has duplicate days.")
        if reference.empty: raise ValueError("Cached oracle parameter file is empty.")
        if not ((reference["training_weekdays"]==TRAINING_WEEKDAYS) &
                (reference["horizon_hours"]==HORIZON) &
                (reference["baseline"]==args.baseline)).all():
            raise ValueError("Saved oracle parameter file contains inconsistent settings.")
        reference=reference.sort_values("day").set_index("day")

    if args.resume:
        if not configuration.is_file() or json.loads(configuration.read_text(encoding="utf-8"))!=config:
            raise ValueError("Resume configuration differs from saved settings.")
        dpath,hpath=output/FILES["daily"],output/FILES["hourly"]
        if dpath.is_file() and hpath.is_file():
            previous=pd.read_csv(dpath,parse_dates=["day"])
            previous_hourly=pd.read_csv(hpath,parse_dates=["day","timestamp"])
            check_scores(previous,previous_hourly)
            daily, hourly=previous.to_dict("records"),previous_hourly.to_dict("records")
            done=set(previous["day"])
        else: daily,hourly,done=[],[],set()
        skipped=pd.read_csv(output/FILES["fail"],parse_dates=["day"]).to_dict("records") if (output/FILES["fail"]).is_file() else []
    else:
        configuration.write_text(json.dumps(config,indent=2),encoding="utf-8")
        daily,hourly,done,skipped=[],[],set(),[]

    frame=group_frame(args.group)
    first=max(pd.Timestamp("2022-01-01")+pd.Timedelta(weeks=TRAINING_WEEKDAYS),
              frame.index.min().normalize()+pd.Timedelta(weeks=TRAINING_WEEKDAYS))
    last=min(pd.Timestamp("2022-10-31"),frame.index.max().normalize())
    if args.start_day: first=max(first,pd.Timestamp(args.start_day))
    if args.end_day: last=min(last,pd.Timestamp(args.end_day))
    if reference is not None:
        days=[day for day in reference.index if first<=day<=last]
    else:
        days=list(pd.date_range(first,last,freq="D"))
    if args.max_days: days=days[:args.max_days]
    if not days: raise ValueError("No dates available after warm-up and requested filters.")
    for k,day in enumerate(days,start=1):
        day=pd.Timestamp(day)
        if day in done: continue
        try:
            cached=reference.loc[day] if reference is not None else None
            daily_rows,hourly_rows=evaluate_day(frame,day,args.steps_per_hour,args.baseline,cached)
            daily.extend(daily_rows);hourly.extend(hourly_rows)
        except (RuntimeError,ValueError,KeyError,np.linalg.LinAlgError) as exc:
            if args.fail_fast: raise
            skipped.append({"day":day,"reason":str(exc)})
        if k%args.progress_every==0:
            save_rows(output,daily,hourly,skipped)
            print(f"[WM4 origin RMSE] {k}/{len(days)} days: "
                  f"{len(daily)//len(ORIGINS)} complete, {len(skipped)} skipped",flush=True)
    save_rows(output,daily,hourly,skipped)
    if not daily: raise RuntimeError(f"No complete origin forecasts. Inspect {output/FILES['fail']}.")
    return report(output)


def main():
    parser=ArgumentParser(description="WM4 N8/H7: aggregate seven-hour oracle RMSE by origin 0..17.")
    parser.add_argument("--group",choices=("MED","ALL"),default="MED")
    parser.add_argument("--baseline",choices=BASELINE_NAMES,default="weekly_temperature")
    parser.add_argument("--steps-per-hour",type=int,default=4)
    parser.add_argument("--from-oracle",default=None,
                        help="Optional saved N8/H7 oracle folder or daily_metrics.csv; reuse fitted theta/day cohort.")
    parser.add_argument("--start-day",default=None)
    parser.add_argument("--end-day",default=None)
    parser.add_argument("--max-days",type=int,default=0)
    parser.add_argument("--progress-every",type=int,default=10)
    parser.add_argument("--output",default=None)
    parser.add_argument("--report-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--fail-fast",action="store_true")
    run(parser.parse_args())


if __name__=="__main__": main()
