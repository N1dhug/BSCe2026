"""Pooled WM4 HISTORICAL DEMAND FIT RMSE across backlog N and scored horizon k.

Read the constrained theta and fitted historical initial states produced by
sweep_weekday_horizon.py. Reconstruct free-running WM4 demand on the SAME
historical weekdays used for those fits (never on forecast/target days).
At each (N,k), pool all squared demand errors from the first k hourly
observations of every historical fitting trajectory.

IMPORTANT: Saved theta and [F0,R0] were fitted to complete 24-hour
historical windows. k changes the subset SCORED, not the parameter fit.
This is in-sample fitted-demand quality, not oracle prediction quality,
not derivative error and not a separate k-specific parameter estimation.

Run from the project root:
    python -m src.exe.run_model.wm_4.plot_pooled_fit_rmse
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from src.exe.run_model.wm_4.moving_horizon_common import group_frame, window_data
from src.exe.run_model.wm_4.plot_linear_rate_r2 import _values
from src.exe.run_model.wm_4.sweep_weekday_horizon import baseline_from_history, load_tables
from src.tools.impute_hourly import impute_hourly_days


ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT/"files"/"output"/"wm_4"/"weekday_horizon_sweep_expanded"
OUTPUT = ROOT/"files"/"output"/"wm_4"/"pooled_fit_rmse"
BASELINES = ("weekly_average","weekly_temperature")
TITLES = {"weekly_average":"Weekly average","weekly_temperature":"Weekly temperature"}


def _fit_observations(frame, day, n, name, theta, states, sph):
    """Reproduce the original joint-QP historical baseline and demand fits."""
    day = pd.Timestamp(day).normalize()
    history = [day-pd.Timedelta(weeks=i) for i in range(n,0,-1)]
    training, _ = impute_hourly_days(frame,days=history,columns=("I","U","Temp"),require_days=True)
    baseline = baseline_from_history(training,history,name)
    states = states.drop_duplicates("historical_day",keep="last")
    state_lookup = {pd.Timestamp(row.historical_day).normalize():
                    np.array([row.historical_F0,row.historical_R0],dtype=float)
                    for row in states.itertuples(index=False)}
    if set(history) != set(state_lookup):
        raise ValueError("Saved historical states do not match the N preceding weekdays.")
    if not np.isfinite(theta).all(): raise ValueError("Saved theta contains nonfinite values.")

    observed, fitted, reference = [],[],[]
    for earlier in history:
        model,data,_,hourly_I = window_data(training,earlier,baseline,sph)
        initial = state_lookup[earlier]
        if not np.isfinite(initial).all():
            raise ValueError(f"Nonfinite saved initial state for {earlier:%Y-%m-%d}.")
        # Joint WM4 fitting fixes R0 to the observed I0 - O0 and optimizes F0
        # through beta_i = theta_F * F0_i. This check catches incompatible
        # stored states or changes to the baseline reconstruction.
        observed_r0 = float(data["I"][0]-data["O"][0])
        if not np.isclose(initial[1],observed_r0,rtol=1e-6,atol=1e-7):
            raise ValueError(f"Saved R0 differs from reconstructed I0-O0 on {earlier:%Y-%m-%d}.")
        state = model._solve_deterministic(initial,data["U"],data["PE"],
                                           data["Time"],data["O"],theta)
        hours = np.arange(24,dtype=int)*sph
        predicted = np.asarray(state["I"],dtype=float)[hours]
        original = np.asarray(hourly_I,dtype=float)
        baseline_demand = np.asarray(data["O"],dtype=float)[hours]
        if not (np.isfinite(predicted).all() and np.isfinite(original).all() and
                np.isfinite(baseline_demand).all()):
            raise ValueError(f"Nonfinite reconstructed historical demand on {earlier:%Y-%m-%d}.")
        observed.append(original)
        fitted.append(predicted)
        reference.append(baseline_demand)
    return np.stack(observed),np.stack(fitted),np.stack(reference)


def _plot(summary, ns, ks, group, output, start, end, cmap="viridis", dpi=190):
    finite = summary["pooled_fit_rmse"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise RuntimeError("No valid historical fit RMSE values; inspect skipped_windows.csv.")
    vmax = max(float(np.max(finite)),1e-12)
    norm = Normalize(vmin=0.0,vmax=vmax)
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad("#e5e7eb")
    fig,axes = plt.subplots(1,2,figsize=(19,6.8),sharex=True,sharey=True)
    for ax,name in zip(axes,BASELINES):
        grid = (summary.loc[summary["baseline"]==name]
                .pivot(index="training_weekdays",columns="horizon_hours",values="pooled_fit_rmse")
                .reindex(index=ns,columns=ks).to_numpy(dtype=float))
        image = ax.imshow(np.ma.masked_invalid(grid),cmap=colors,norm=norm,
                          origin="lower",aspect="auto",interpolation="nearest")
        ax.set_title(TITLES[name],fontsize=12)
        ax.set_xticks(np.arange(len(ks)),[str(k) for k in ks],fontsize=8)
        ax.set_yticks(np.arange(len(ns)),[str(n) for n in ns])
        ax.set_xlabel("Historical fit horizon k (hours scored)")
        ax.tick_params(axis="both",length=0)
    axes[0].set_ylabel("Backlog N (preceding matching weekdays)")
    dates = (f"{start:%Y-%m-%d} to {end:%Y-%m-%d}" if start is not None and end is not None
             else "all saved fitting origins")
    fig.suptitle(f"WM4 | {group} | pooled historical DEMAND FIT RMSE | {dates}\n"
                 "Saved constrained 24h theta and fitted historical initial states; k selects scoring hours",
                 fontsize=12)
    fig.subplots_adjust(left=0.055,right=0.90,bottom=0.13,top=0.83,wspace=0.07)
    cax = fig.add_axes([0.916,0.19,0.015,0.57])
    fig.colorbar(image,cax=cax,label="Pooled demand fit RMSE (normalized demand)")
    fig.savefig(output,dpi=dpi,bbox_inches="tight")
    plt.close(fig)


def evaluate(*, group="MED", n_values=range(2,15), horizons=range(1,25),
             start_day=None, end_day=None, steps_per_hour=4, input_dir=None,
             output_dir=None, fail_fast=False, progress_every=25,
             verify_saved_rmse=True, verification_atol=1e-6, dpi=190, **kwargs):
    """Pooled in-sample RMSE of free-running WM4 demand on training histories.

    Dates are target origins whose N PRECEDING same-weekday histories were used
    to estimate theta. The historical fits themselves are what is scored.
    RMSE = sqrt(total SSE / total hourly observations), not mean daily RMSE.
    """
    if kwargs: raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")
    if group not in ("MED","ALL"): raise ValueError("group must be MED or ALL.")
    ns = _values(n_values,2,14,"n_values")
    ks = _values(horizons,1,24,"horizons")
    sph = int(steps_per_hour)
    if sph<2: raise ValueError("steps_per_hour must be at least 2.")
    if verification_atol<0: raise ValueError("verification_atol must be nonnegative.")
    input_dir = Path(input_dir) if input_dir is not None else INPUT/group.lower()
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT/group.lower()

    info = input_dir/"configuration.json"
    if info.is_file():
        config = json.loads(info.read_text(encoding="utf-8"))
        if str(config.get("group",group)) != group: raise ValueError("Saved parameter group differs from requested group.")
        if int(config.get("steps_per_hour",sph)) != sph:
            raise ValueError("steps_per_hour must match the step count used to estimate saved theta.")

    tables = load_tables(input_dir)
    fits, states = tables["fit"],tables["state"]
    required_fit = {"day","baseline","training_weekdays","theta_F","theta_U","theta_R"}
    required_states = {"day","baseline","training_weekdays","historical_day","historical_F0","historical_R0"}
    if fits.empty or states.empty or not required_fit.issubset(fits) or not required_states.issubset(states):
        raise FileNotFoundError(f"Saved parameter_estimates.csv and historical_initial_states.csv are required under {input_dir}.")

    start = pd.Timestamp(start_day).normalize() if start_day is not None else None
    end = pd.Timestamp(end_day).normalize() if end_day is not None else None
    if start is not None and end is not None and end<start: raise ValueError("end_day precedes start_day.")
    fits = fits.loc[fits["baseline"].isin(BASELINES)&fits["training_weekdays"].isin(ns)].copy()
    if start is not None: fits = fits.loc[fits["day"]>=start]
    if end is not None: fits = fits.loc[fits["day"]<=end]
    fits = fits.drop_duplicates(["day","baseline","training_weekdays"],keep="last")
    if fits.empty: raise ValueError("No saved fitted theta records match the selected dataset and intervals.")
    states_by_key = {(pd.Timestamp(day).normalize(),name,int(n)):data
                     for (day,name,n),data in states.groupby(["day","baseline","training_weekdays"])}
    frame = group_frame(group)

    # Store sum of squared errors and sample counts, to get genuine pooled
    # RMSE rather than a mean of day-level RMSE values.
    totals = {(name,n):np.zeros((len(ks),4),dtype=float) for name in BASELINES for n in ns}
    failures,records = [],[]
    for index,row in enumerate(fits.itertuples(index=False),1):
        key = (pd.Timestamp(row.day).normalize(),row.baseline,int(row.training_weekdays))
        try:
            previous = states_by_key.get(key)
            if previous is None or len(previous)!=key[2]:
                raise ValueError("Historical fitted initial-state records are incomplete.")
            theta = np.array([row.theta_F,row.theta_U,row.theta_R],dtype=float)
            observed,fitted,baseline = _fit_observations(frame,key[0],key[2],key[1],theta,previous,sph)
            if observed.shape!=(key[2],24) or fitted.shape!=observed.shape or baseline.shape!=observed.shape:
                raise RuntimeError("Historical model fit returned an invalid N-by-24 hourly array.")

            residual = fitted-observed
            baseline_residual = baseline-observed
            sq = np.cumsum(residual**2,axis=1).sum(axis=0)
            baseline_sq = np.cumsum(baseline_residual**2,axis=1).sum(axis=0)
            exact_rmse = float(np.sqrt(sq[-1]/(key[2]*24)))
            saved_rmse = float(getattr(row,"fit_rmse",np.nan))
            if verify_saved_rmse and np.isfinite(saved_rmse):
                if not np.isclose(exact_rmse,saved_rmse,atol=verification_atol,rtol=1e-5):
                    raise ValueError(f"Historical demand reconstruction RMSE {exact_rmse:.8g} "
                                     f"does not match saved fit_rmse {saved_rmse:.8g}; "
                                     "check source data, saved states and theta version.")

            records.append({"day":key[0],"baseline":key[1],"training_weekdays":key[2],
                            "fitted_24h_rmse":exact_rmse,"saved_fit_rmse":saved_rmse,
                            "baseline_24h_rmse":float(np.sqrt(baseline_sq[-1]/(key[2]*24)))})
            for i,k in enumerate(ks):
                totals[(key[1],key[2])][i] += (sq[k-1],baseline_sq[k-1],key[2]*k,1)
        except (ValueError,RuntimeError,KeyError,np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            failures.append({"day":key[0],"baseline":key[1],"training_weekdays":key[2],"error":str(exc)})
        if progress_every and index%progress_every==0:
            print(f"[demand fit] {index}/{len(fits)} saved fits | skipped={len(failures)}",flush=True)

    summary_rows = []
    for name in BASELINES:
        for n in ns:
            for i,k in enumerate(ks):
                sse,baseline_sse,count,windows = totals[(name,n)][i]
                rmse = float(np.sqrt(sse/count)) if count else np.nan
                base_rmse = float(np.sqrt(baseline_sse/count)) if count else np.nan
                summary_rows.append({"baseline":name,"training_weekdays":n,"horizon_hours":k,
                                     "fitting_windows":int(windows),"historical_intervals":int(windows*n),
                                     "observations":int(count),"sse":sse if count else np.nan,
                                     "pooled_fit_rmse":rmse,"baseline_fit_rmse":base_rmse,
                                     "fit_rmse_gain":base_rmse-rmse})
    summary = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True,exist_ok=True)
    summary.to_csv(output_dir/"pooled_fit_rmse.csv",index=False)
    pd.DataFrame(records,columns=["day","baseline","training_weekdays","fitted_24h_rmse",
                                  "saved_fit_rmse","baseline_24h_rmse"]).to_csv(
                                  output_dir/"per_window_fit_rmse.csv",index=False)
    pd.DataFrame(failures,columns=["day","baseline","training_weekdays","error"]).to_csv(
                                  output_dir/"skipped_windows.csv",index=False)
    evaluated = pd.DataFrame(records)
    first,last = ((pd.Timestamp(evaluated["day"].min()),pd.Timestamp(evaluated["day"].max()))
                  if not evaluated.empty else (None,None))
    _plot(summary,ns,ks,group,output_dir/"pooled_fit_rmse.png",first,last,dpi=dpi)
    print(f"Saved pooled historical demand FIT RMSE: {output_dir} | "
          f"fitting windows={len(records)} | skipped={len(failures)}")
    print("These are in-sample fits to historical weekdays; target-day predictions and oracle states are not used.")
    return summary


def main(**kwargs):
    """Adjust ranges, group, and evaluation period through keyword arguments."""
    defaults = dict(group="MED",n_values=range(2,15),horizons=range(1,25),
                    start_day=None,end_day=None,steps_per_hour=4,
                    input_dir=None,output_dir=None,fail_fast=False,
                    progress_every=25,verify_saved_rmse=True,verification_atol=1e-6,dpi=190)
    return evaluate(**(defaults|kwargs))


if __name__=="__main__": main()
