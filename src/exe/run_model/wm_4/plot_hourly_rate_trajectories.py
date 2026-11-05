"""Plot every historical hourly R-dot observation used in WM4 rate diagnostics.

One PNG per (baseline, matching-weekday backlog N, horizon k), using the
SAVED constrained joint-QP theta and historical initial states. The observed
derivative is (R[h+1]-R[h])/(1/24), where R=I-O. The comparison curve
averages the fitted WM4 drift over the matching hour. No unconstrained
OLS or new parameter fitting is performed.

Run from project root:
    python -m src.exe.run_model.wm_4.plot_hourly_rate_trajectories
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.moving_horizon_common import group_frame
from src.exe.run_model.wm_4.plot_linear_rate_r2 import _one_configuration, _values, BASELINES, INPUT, ROOT
from src.exe.run_model.wm_4.sweep_weekday_horizon import load_tables


OUTPUT = ROOT/"files"/"output"/"wm_4"/"hourly_rate_trajectories"
TITLES = {"weekly_average":"Weekly average","weekly_temperature":"Weekly temperature"}


def _plot(observed, fitted, target_days, historical_days, baseline, n, k, group, path, dpi=170):
    """Show every observed hourly derivative without connecting separate days."""
    hours = min(int(k), 23)
    actual, predicted = observed[:,:hours], fitted[:,:hours]
    residual = actual-predicted
    y, p = actual.ravel(), predicted.ravel()
    sse = float(np.dot(y-p,y-p))
    sst = float(np.sum((y-np.mean(y))**2))
    r2 = 1.0-sse/sst if sst>1e-12 else np.nan
    rmse = float(np.sqrt(np.mean((y-p)**2)))

    # One NaN gap after each historical-day segment: no invented transitions
    # between different weekdays, even if their dates are consecutive.
    count = len(actual)
    step = hours+1
    x = np.arange(count*step,dtype=float)
    observed_line = np.c_[actual,np.full(count,np.nan)].ravel()
    fitted_line = np.c_[predicted,np.full(count,np.nan)].ravel()
    residual_line = np.c_[residual,np.full(count,np.nan)].ravel()

    fig,(ax,err) = plt.subplots(2,1,figsize=(18,8),sharex=True,
                                gridspec_kw={"height_ratios":[3,1],"hspace":0.12})
    ax.plot(x,observed_line,lw=0.75,alpha=0.85,label="Observed hourly R-dot")
    ax.plot(x,fitted_line,lw=0.75,alpha=0.75,label="WM4 constrained fitted R-dot")
    ax.axhline(0.0,color="black",lw=0.7,alpha=0.4)
    ax.set_ylabel("R-dot (normalized-day time)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.18)
    err.plot(x,residual_line,lw=0.65,alpha=0.8)
    err.axhline(0.0,color="black",lw=0.7,alpha=0.5)
    err.set_ylabel("Observed - fitted")
    err.grid(alpha=0.18)

    # Segments are chronological by target fitting day then historical weekday.
    # The labels denote the target-day origin, not a continuous physical time axis.
    ticks = np.unique(np.linspace(0,count-1,min(count,8),dtype=int))
    err.set_xticks(ticks*step+hours//2,[f"{target_days[i]:%Y-%m-%d}" for i in ticks],
                   rotation=25,ha="right")
    err.set_xlabel("Historical intervals grouped by moving-horizon target day (gaps between intervals)")
    first,last = min(target_days),max(target_days)
    fig.suptitle(f"WM4 {group} | {TITLES[baseline]} | N={n} | k={k}h "
                 f"({hours} derivative hours per historical interval)\n"
                 f"Target origins {first:%Y-%m-%d} to {last:%Y-%m-%d} | "
                 f"intervals={count:,} | observations={len(y):,} | R²={r2:.4f} | RMSE={rmse:.4f}")
    fig.subplots_adjust(left=0.065,right=0.985,bottom=0.14,top=0.85)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=dpi,bbox_inches="tight")
    plt.close(fig)
    return {"baseline":baseline,"training_weekdays":n,"horizon_hours":k,
            "rate_hours_per_interval":hours,"intervals":count,"observations":len(y),
            "target_start":first,"target_end":last,"r2":r2,"rmse":rmse,
            "sse":sse,"sst":sst,"file":str(path)}


def run(*, group="MED", n_values=range(2,15), horizons=range(1,25),
        start_day=None, end_day=None, steps_per_hour=4, input_dir=None,
        output_dir=None, fail_fast=False, progress_every=25, save_csv=False,
        dpi=170, **kwargs):
    """Create all observed/fitted hourly R-dot plots from stored QP estimates.

    start_day and end_day filter the SAVED target fitting origins, inclusive.
    A horizon k uses min(k,23) real hourly differences from each historical
    24-point interval. Thus k=23 and k=24 contain the same observations.
    """
    if kwargs: raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")
    if group not in ("MED","ALL"): raise ValueError("group must be MED or ALL.")
    ns = _values(n_values,2,14,"n_values")
    ks = _values(horizons,1,24,"horizons")
    sph = int(steps_per_hour)
    if sph<2: raise ValueError("steps_per_hour must be at least 2.")
    input_dir = Path(input_dir) if input_dir is not None else INPUT/group.lower()
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT/group.lower()
    config_path = input_dir/"configuration.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if str(config.get("group",group)) != group:
            raise ValueError("Saved fitting group does not match requested group.")
        if int(config.get("steps_per_hour",sph)) != sph:
            raise ValueError("steps_per_hour must equal the existing theta sweep configuration.")

    tables = load_tables(input_dir)
    fits, states = tables["fit"], tables["state"]
    required_fit = {"day","baseline","training_weekdays","theta_F","theta_U","theta_R"}
    required_state = {"day","baseline","training_weekdays","historical_day","historical_F0"}
    if fits.empty or states.empty or not required_fit.issubset(fits) or not required_state.issubset(states):
        raise FileNotFoundError(f"Saved parameter_estimates.csv and historical_initial_states.csv required in {input_dir}.")

    begin = pd.Timestamp(start_day).normalize() if start_day is not None else None
    finish = pd.Timestamp(end_day).normalize() if end_day is not None else None
    if begin is not None and finish is not None and finish<begin:
        raise ValueError("end_day must not precede start_day.")
    fits = fits.loc[fits["baseline"].isin(BASELINES)&fits["training_weekdays"].isin(ns)].copy()
    if begin is not None: fits = fits.loc[fits["day"]>=begin]
    if finish is not None: fits = fits.loc[fits["day"]<=finish]
    fits = fits.drop_duplicates(["day","baseline","training_weekdays"],keep="last")
    if fits.empty: raise ValueError("No saved fit rows match the selected range.")

    frame = group_frame(group)
    state_lookup = {(pd.Timestamp(day).normalize(),name,int(n)):data
                    for (day,name,n),data in states.groupby(["day","baseline","training_weekdays"])}
    output_dir.mkdir(parents=True,exist_ok=True)
    rows,failures = [],[]

    for (baseline,n),subset in fits.groupby(["baseline","training_weekdays"],sort=True):
        actual_segments,predicted_segments,target_days,historical_days = [],[],[],[]
        for i,row in enumerate(subset.sort_values("day").itertuples(index=False),1):
            day = pd.Timestamp(row.day).normalize()
            key = (day,baseline,int(n))
            try:
                history_states = state_lookup.get(key)
                if history_states is None or len(history_states)!=n:
                    raise ValueError("Historical initial states are missing or incomplete.")
                theta = np.array([row.theta_F,row.theta_U,row.theta_R],dtype=float)
                if not np.isfinite(theta).all(): raise ValueError("Nonfinite saved theta.")
                y,p = _one_configuration(frame,day,int(n),baseline,theta,history_states,sph)
                if y.shape != (n,23) or p.shape != y.shape:
                    raise RuntimeError(f"Unexpected hourly rate array shape: {y.shape}, {p.shape}")
                if not np.isfinite(y).all() or not np.isfinite(p).all():
                    raise ValueError("Nonfinite hourly derivative samples.")
                history = [day-pd.Timedelta(weeks=j) for j in range(n,0,-1)]
                actual_segments.extend(y)
                predicted_segments.extend(p)
                target_days.extend([day]*n)
                historical_days.extend(history)
            except (ValueError,RuntimeError,KeyError,np.linalg.LinAlgError) as exc:
                if fail_fast: raise
                failures.append({"day":day,"baseline":baseline,"training_weekdays":n,"error":str(exc)})
            if progress_every and i%progress_every==0:
                print(f"[R-dot plots] {baseline} N={n} | {i}/{len(subset)} windows | skipped={len(failures)}",flush=True)
        if not actual_segments:
            print(f"[R-dot plots] No valid windows for {baseline}, N={n}.",flush=True)
            continue
        actual,predicted = np.vstack(actual_segments),np.vstack(predicted_segments)
        target_days = pd.DatetimeIndex(target_days)
        historical_days = pd.DatetimeIndex(historical_days)
        folder = output_dir/baseline/f"N_{int(n):02d}"
        for k in ks:
            target = folder/f"hourly_r_dot_k_{k:02d}.png"
            rows.append(_plot(actual,predicted,target_days,historical_days,baseline,int(n),k,group,target,dpi=dpi))
            if save_csv:
                h = min(k,23)
                samples = pd.DataFrame({
                    "target_day":np.repeat(target_days.to_numpy(),h),
                    "historical_day":np.repeat(historical_days.to_numpy(),h),
                    "hour":np.tile(np.arange(h),len(actual)),
                    "observed_r_dot":actual[:,:h].ravel(),
                    "fitted_r_dot":predicted[:,:h].ravel(),
                })
                samples["residual"] = samples["observed_r_dot"]-samples["fitted_r_dot"]
                samples.to_csv(folder/f"hourly_r_dot_k_{k:02d}.csv",index=False)
        print(f"[R-dot plots] {baseline} N={n} | {len(actual)} historical intervals | {len(ks)} plots",flush=True)

    pd.DataFrame(rows).to_csv(output_dir/"plot_summary.csv",index=False)
    pd.DataFrame(failures,columns=["day","baseline","training_weekdays","error"]).to_csv(
        output_dir/"skipped_windows.csv",index=False)
    if not rows: raise RuntimeError("No plots generated; inspect skipped_windows.csv for failures.")
    print(f"Saved {len(rows)} WM4 hourly R-dot plots in: {output_dir}")
    return pd.DataFrame(rows)


def main(**kwargs):
    defaults = dict(group="MED",n_values=range(2,15),horizons=range(1,25),
                    start_day=None,end_day=None,steps_per_hour=4,
                    input_dir=None,output_dir=None,fail_fast=False,
                    progress_every=25,save_csv=False,dpi=170)
    return run(**(defaults|kwargs))


if __name__=="__main__": main()
