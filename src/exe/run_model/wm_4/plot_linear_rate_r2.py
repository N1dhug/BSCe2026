"""Pooled WM4 linear-rate R² on historical intervals using SAVED theta estimates.

No theta refitting. Each fitted target (day, baseline, N) supplies one theta
and N historical F0 values from the completed weekday-horizon sweep.
Observed derivatives are formed from ORIGINAL hourly endpoints:
    (R[h+1] - R[h]) / (1/24),  R[h] = I[h] - O[h].
Predicted hourly derivatives average theta @ [F,U,R] over the corresponding
hour using the existing fine-grid state reconstruction. Thus there is exactly
ONE derivative observation per hour, not steps_per_hour pseudo-observations.
Only the first 23 hourly transitions are observable in each 24-sample day;
k=24 uses the same 23 transitions as k=23. Both baselines share one R² scale.

Run as: python -m src.exe.run_model.wm_4.plot_linear_rate_r2
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from src.method.moving_horizon import group_frame, window_data
from src.exe.run_model.wm_4.sweep_weekday_horizon import baseline_from_history, load_tables
from src.tools.impute_hourly import impute_hourly_days


ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT/"files"/"output"/"wm_4"/"weekday_horizon_sweep_expanded"
OUTPUT = ROOT/"files"/"output"/"wm_4"/"linear_rate_r2"
BASELINES = ("weekly_average", "weekly_temperature")
TITLES = ("Weekly average", "Weekly temperature")


def _values(values, low, high, name):
    result = tuple(sorted(set(int(v) for v in values)))
    if not result or any(v < low or v > high for v in result): raise ValueError(f"{name} must contain integers in [{low},{high}].")
    return result


def _rate_components(frame, start, baseline, f0, sph):
    """Compute 23 HOURLY derivatives and matching mean hourly WM4 drift.

    The existing WM4 model uses normalized-day time dt=1/(24*sph).
    We retain the fitted historical F0 and fine-grid F reconstruction so the
    saved theta stays on the correct time scale. Only the rate TARGET is
    changed from sub-hourly forward differences to genuine hourly
    differences; model drift is averaged over each complete hour.
    """
    model, data, _, _ = window_data(frame, start, baseline, sph)
    R = np.asarray(data["I"]-data["O"], dtype=float)
    F = np.r_[float(f0), float(f0)+np.cumsum(model.dt*R[:-1])]
    X_fine = np.column_stack((F[:-1], data["U"][:-1], R[:-1]))
    valid = np.asarray(data["transition_mask"], dtype=bool)
    if np.count_nonzero(valid) != 23*sph or not np.all(valid[:23*sph]) or np.any(valid[23*sph:]):
        raise ValueError("Historical 24-hour WM4 window has an unexpected transition mask.")

    # Anchor differences use the 24 ACTUAL hourly observations (0,...,23).
    # Time is normalized to one day, so one hour is delta_t = 1/24.
    hourly_R = R[np.arange(24)*sph]
    y_hourly = np.diff(hourly_R)/(sph*model.dt)

    # Match an hourly average derivative to the hourly average model drift:
    # (1/delta_t_hour) integral_h^(h+1) theta @ [F,U,R] dt.
    # The left-rectangle approximation exactly uses the same fine-grid
    # reconstruction as the stored theta estimation, without pretending that
    # interpolation creates independent sub-hourly observations.
    X_hourly = X_fine[:23*sph].reshape(23,sph,3).mean(axis=1)
    if X_hourly.shape != (23,3) or y_hourly.shape != (23,):
        raise RuntimeError("Hourly rate-regression shapes are inconsistent.")
    if not np.isfinite(X_hourly).all() or not np.isfinite(y_hourly).all():
        raise ValueError("Nonfinite hourly rate observations.")
    return X_hourly, y_hourly


def _one_configuration(frame, day, n, baseline_name, theta, states, sph):
    history = [pd.Timestamp(day)-pd.Timedelta(weeks=i) for i in range(n,0,-1)]
    # Reproduce the historical-only imputation convention from the theta sweep.
    training, _ = impute_hourly_days(frame, days=history, columns=("I","U","Temp"), require_days=True)
    baseline = baseline_from_history(training, history, baseline_name)
    state_map = {pd.Timestamp(row.historical_day).normalize(): float(row.historical_F0)
                 for row in states.itertuples(index=False)}
    if set(history) != set(state_map): raise ValueError("Incomplete historical F0 estimates.")
    xx, yy = [], []
    for start in history:
        X, y = _rate_components(training, start, baseline, state_map[start], sph)
        xx.append(X); yy.append(y)
    X, y = np.stack(xx), np.stack(yy)
    predicted = X@np.asarray(theta, dtype=float)
    return y, predicted


def evaluate(*, group="ALL", n_values=range(2,15), horizons=range(1,25),
             start_day=None, end_day=None, steps_per_hour=4, input_dir=None,
             output_dir=None, fail_fast=False, progress_every=25, **kwargs):
    """Compute pooled derivative R² for saved moving-horizon thetas.

    Dates filter TARGET days (whose theta is saved), inclusive.
    The historical N weekdays preceding each target day supply the rate
    observations. The selected theta is fixed across all k values.
    Returned R² is in-sample historical derivative fit quality, NOT
    prediction R² and NOT a new unrestricted linear-regression fit.
    """
    if kwargs: raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")
    if group not in ("MED","ALL"): raise ValueError("group must be MED or ALL.")
    ns = _values(n_values, 2, 14, "n_values")
    ks = _values(horizons, 1, 24, "horizons")
    sph = int(steps_per_hour)
    if sph < 2: raise ValueError("steps_per_hour must be at least 2.")
    input_dir = Path(input_dir) if input_dir is not None else INPUT/group.lower()
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT/group.lower()
    tables = load_tables(input_dir)
    fits, states = tables["fit"], tables["state"]
    required_fit = {"day","baseline","training_weekdays","theta_F","theta_U","theta_R"}
    required_state = {"day","baseline","training_weekdays","historical_day","historical_F0"}
    if fits.empty or states.empty or not required_fit.issubset(fits) or not required_state.issubset(states):
        raise FileNotFoundError(f"Saved parameter_estimates.csv and historical_initial_states.csv are required in {input_dir}. Run the WM4 weekday horizon sweep first.")
    start = pd.Timestamp(start_day).normalize() if start_day is not None else None
    end = pd.Timestamp(end_day).normalize() if end_day is not None else None
    if start is not None and end is not None and end < start: raise ValueError("end_day precedes start_day.")
    fits = fits.loc[fits["baseline"].isin(BASELINES)&fits["training_weekdays"].isin(ns)].copy()
    if start is not None: fits = fits.loc[fits["day"] >= start]
    if end is not None: fits = fits.loc[fits["day"] <= end]
    fits = fits.drop_duplicates(["day","baseline","training_weekdays"], keep="last")
    if fits.empty: raise ValueError("No saved theta estimates match the requested dates, baselines, and N values.")
    frame = group_frame(group)
    grouped_states = {(pd.Timestamp(day).normalize(),name,int(n)): data
                      for (day,name,n),data in states.groupby(["day","baseline","training_weekdays"])}
    # Each (baseline,N,k) has pooled sufficient statistics; no large arrays
    # or repeated computations for all 24 horizons.
    totals = {(name,n,k):np.zeros(5,dtype=float) for name in BASELINES for n in ns for k in ks}
    failures = []
    for count,row in enumerate(fits.itertuples(index=False),1):
        key = (pd.Timestamp(row.day).normalize(),row.baseline,int(row.training_weekdays))
        try:
            history_states = grouped_states.get(key)
            if history_states is None or len(history_states) != key[2]:
                raise ValueError("Saved historical states are missing or incomplete.")
            theta = [row.theta_F,row.theta_U,row.theta_R]
            actual, predicted = _one_configuration(frame,key[0],key[2],key[1],theta,history_states,sph)
            for k in ks:
                # Exactly one observed derivative per complete hourly transition.
                # 24 hourly data points have 23 observable hourly transitions.
                hours = min(k,23)
                y, p = actual[:,:hours].reshape(-1),predicted[:,:hours].reshape(-1)
                if not np.isfinite(y).all() or not np.isfinite(p).all():
                    raise ValueError("Nonfinite rate observations or predictions.")
                totals[(key[1],key[2],k)] += (len(y),np.sum(y),np.dot(y,y),np.dot(y-p,y-p),1)
        except (ValueError,RuntimeError,KeyError,np.linalg.LinAlgError) as exc:
            if fail_fast: raise
            failures.append({"day":key[0],"baseline":key[1],"training_weekdays":key[2],"error":str(exc)})
        if progress_every and count%progress_every == 0:
            print(f"[rate R²] {count}/{len(fits)} fitted windows | skipped={len(failures)}",flush=True)
    rows = []
    for name in BASELINES:
        for n in ns:
            for k in ks:
                count,total,squares,sse,windows = totals[(name,n,k)]
                sst = squares-total**2/count if count else np.nan
                r2 = 1.0-sse/sst if count>1 and sst>1e-12 else np.nan
                rows.append({"baseline":name,"training_weekdays":n,"horizon_hours":k,
                             "windows":int(windows),"observations":int(count),
                             "rate_hours_per_window":min(k,23),
                             "sse":sse if count else np.nan,"sst":sst,
                             "r2":r2})
    summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True,exist_ok=True)
    summary.to_csv(output_dir/"pooled_linear_rate_r2.csv",index=False)
    pd.DataFrame(failures,columns=["day","baseline","training_weekdays","error"]).to_csv(output_dir/"skipped_windows.csv",index=False)
    plot(summary,ns,ks,group,output_dir/"pooled_linear_rate_r2.png")
    print(f"Saved pooled rate R²: {output_dir} | windows={len(fits)-len(failures)} | skipped={len(failures)}")
    return summary


def plot(summary,ns,ks,group,output,cmap="viridis",dpi=190):
    finite = summary["r2"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise RuntimeError("No usable pooled R² values. Review skipped_windows.csv (error column) and confirm saved theta/states correspond to this dataset.")
    # A common [min(R²,0),1] scale preserves negative R² rather than clipping
    # it to zero, with an identical color for identical scores in both panels.
    vmin = min(0.0,float(np.min(finite)))
    if vmin >= 1.0: vmin = 0.0
    norm = Normalize(vmin=vmin,vmax=1.0)
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad("#e6e8ea")
    fig,axes = plt.subplots(1,2,figsize=(18,6.8),sharex=True,sharey=True)
    for ax,name,title in zip(axes,BASELINES,TITLES):
        grid = (summary.loc[summary["baseline"]==name]
                .pivot(index="training_weekdays",columns="horizon_hours",values="r2")
                .reindex(index=ns,columns=ks).to_numpy(dtype=float))
        image = ax.imshow(np.ma.masked_invalid(grid),cmap=colors,norm=norm,
                          origin="lower",aspect="auto",interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks(range(len(ks)),[str(k) for k in ks],fontsize=8)
        ax.set_yticks(range(len(ns)),[str(n) for n in ns])
        ax.set_xlabel("Horizon k (hours)")
        ax.tick_params(length=0)
    axes[0].set_ylabel("Backlog N (preceding matching weekdays)")
    fig.suptitle(f"ARM (Affine Revised Model) | {group} | pooled HOURLY rate-dynamics R² (saved θ) | k=24 uses 23 rate hours")
    fig.subplots_adjust(left=0.06,right=0.90,bottom=0.13,top=0.87,wspace=0.08)
    cax=fig.add_axes([0.92,0.18,0.015,0.63])
    fig.colorbar(image,cax=cax,label="Pooled hourly rate-dynamics R²")
    fig.savefig(output,dpi=dpi,bbox_inches="tight")
    plt.close(fig)


def main(**kwargs):
    """Editable kwargs are the intended configuration interface."""
    defaults = dict(group="ALL",n_values=range(2,15),horizons=range(1,25),
                    start_day=None,end_day=None,steps_per_hour=4,
                    input_dir=None,output_dir=None,fail_fast=False,progress_every=25)
    return evaluate(**(defaults|kwargs))


if __name__=="__main__": main()
