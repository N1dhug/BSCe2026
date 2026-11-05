"""ARM hourly prediction-error quintiles using each day's fixed 24h oracle state.

For each baseline choose the best historical backlog N by the pooled 24-hour
oracle RMSE (or optionally MAE). Reconstruct exactly ONE 24-hour trajectory
per prediction day using that day's k=24 oracle F0 and R0, and evaluate the
hourly errors (not the cumulative errors from separately optimized k-hour
oracles). At each hour, rank days by absolute hourly error into five equally
sized quintiles. RMSE is sqrt(mean(squared hourly errors)) within each
quintile; MAE is mean(abs(hourly errors)). Both reflect dispersion across
days at THAT hour. The pooled dashed curve uses all days.

Run from the repository root:
    python -m src.exe.run_model.wm_4.plot_prediction_error_quintiles
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.plot_weekday_horizon_heatmaps import (
    BASELINES, TITLES, INPUT, ROOT, KEY, SCORE_KEY, COMPONENT_KEY,
    COMPONENT_COLUMNS, load_saved, load_fitted_cohort, aggregate_rmse,
)

OUTPUT = ROOT/"files"/"output"/"wm_4"/"prediction_error_quintiles"
METRICS = ("rmse","mae")
HOURS = tuple(range(1,25))


def _eligible_24h(scores, components, eligible, min_days):
    """Use historical-fit-eligible days with a 24h oracle and all 24 observations."""
    scores = scores.copy()
    components = components.copy()
    for frame in (scores,components):
        frame["day"] = pd.to_datetime(frame["day"]).dt.normalize()
    scores = scores.loc[scores["horizon_hours"]==24].drop_duplicates(SCORE_KEY,keep="last")
    components = components.drop_duplicates(COMPONENT_KEY,keep="last")
    scores = scores.merge(eligible[KEY].drop_duplicates(),on=KEY,how="inner",
                          validate="many_to_one")
    components = components.merge(eligible[KEY].drop_duplicates(),on=KEY,how="inner",
                                  validate="many_to_one")
    numeric = ["oracle_F0","oracle_R0"]
    for field in numeric:
        scores[field] = pd.to_numeric(scores[field],errors="coerce")
    for field in COMPONENT_COLUMNS:
        components[field] = pd.to_numeric(components[field],errors="coerce")
    scores = scores.loc[np.isfinite(scores[numeric]).all(axis=1)].copy()
    components = components.loc[
        components["hour"].isin(range(24)) &
        np.isfinite(components[list(COMPONENT_COLUMNS)]).all(axis=1)].copy()
    present = components.groupby(KEY)["hour"].agg(
        lambda s: set(s.astype(int)) == set(range(24)))
    valid_keys = pd.DataFrame(present.index[present].tolist(),columns=KEY)
    if valid_keys.empty:
        raise ValueError("No historical-fit-eligible prediction days have all 24 hourly components.")
    scores = scores.merge(valid_keys,on=KEY,how="inner",validate="many_to_one")
    components = components.merge(scores[KEY],on=KEY,how="inner",validate="many_to_one")
    counts = scores.groupby(["baseline","training_weekdays"])["day"].nunique()
    allowed = set(counts.index[counts>=min_days])
    scores = scores.loc[[tuple(pair) in allowed for pair in
                        scores[["baseline","training_weekdays"]].itertuples(index=False,name=None)]].copy()
    components = components.loc[[tuple(pair) in allowed for pair in
                                components[["baseline","training_weekdays"]].itertuples(index=False,name=None)]].copy()
    if scores.empty:
        raise ValueError(f"No baseline/backlog has at least {min_days} complete 24h oracle days.")
    return scores,components


def _select_backlogs(scores, components, eligible, selection, min_days):
    """Match the prediction heatmap's POOLED k=24 metric, not minimum day error."""
    summary = aggregate_rmse(scores,components,cohort="fitted",fitted_cohort=eligible,
                             baselines=BASELINES,horizons=(24,),
                             n_values=sorted(scores["training_weekdays"].unique()))
    summary = summary.loc[summary["days"]>=min_days].copy()
    choices = []
    for baseline in BASELINES:
        subset = summary.loc[summary["baseline"]==baseline]
        if subset.empty: raise ValueError(f"No complete 24-hour scores for {baseline}.")
        for metric in METRICS:
            criterion = metric if selection=="per_metric" else selection
            field = "aggregate_rmse" if criterion=="rmse" else "aggregate_mae"
            winner = subset.sort_values([field,"training_weekdays"]).iloc[0]
            choices.append({
                "baseline":baseline,"metric":metric,
                "best_n":int(winner["training_weekdays"]),"selected_by":criterion,
                "selection_24h_pooled_score":float(winner[field]),
                "selection_24h_rmse":float(winner["aggregate_rmse"]),
                "selection_24h_mae":float(winner["aggregate_mae"]),
                "days":int(winner["days"]),
                "first_day":winner["start_day"],"last_day":winner["end_day"],
            })
    return pd.DataFrame(choices),summary


def _hourly_residuals(scores, components, choices):
    """Reconstruct errors for a FIXED 24h oracle F0/R0 across every hour."""
    pairs = choices[["baseline","best_n"]].drop_duplicates().rename(
        columns={"best_n":"training_weekdays"})
    scores = scores.merge(pairs,on=["baseline","training_weekdays"],how="inner",
                          validate="many_to_one")
    components = components.merge(scores[KEY+["oracle_F0","oracle_R0"]],
                                  on=KEY,how="inner",validate="many_to_one")
    actual = components["observed"].to_numpy(dtype=float)
    modeled = (components["prediction_at_initial_zero"].to_numpy(dtype=float)+
               components["oracle_F0"].to_numpy(dtype=float)*
               components["sensitivity_F0"].to_numpy(dtype=float)+
               components["oracle_R0"].to_numpy(dtype=float)*
               components["sensitivity_R0"].to_numpy(dtype=float))
    components["prediction"] = modeled
    components["residual"] = modeled-actual
    components["abs_error"] = np.abs(components["residual"])
    components["squared_error"] = components["residual"]**2
    components["hour"] = components["hour"].astype(int)+1
    if not np.isfinite(components["residual"].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite hourly 24-hour-oracle prediction errors.")
    return components


def _quintiles(errors, choices):
    """Rank per-hour absolute error, then compute actual quintile RMSE and MAE."""
    rows = []
    for item in choices.itertuples(index=False):
        subset = errors.loc[(errors["baseline"]==item.baseline)&
                            (errors["training_weekdays"]==item.best_n)]
        count = subset["day"].nunique()
        if count<5: raise ValueError("At least five complete prediction days are required.")
        for hour in HOURS:
            hourly = subset.loc[subset["hour"]==hour].sort_values(
                ["abs_error","day"],kind="stable").reset_index(drop=True)
            if len(hourly)!=count or hourly["day"].nunique()!=count:
                raise ValueError("Hourly components do not form a fixed complete-day cohort.")
            bins = np.array_split(np.arange(count),5)
            pooled = (np.sqrt(np.mean(hourly["squared_error"]))
                      if item.metric=="rmse" else np.mean(hourly["abs_error"]))
            for quintile,idx in enumerate(bins,1):
                group = hourly.iloc[idx]
                value = (np.sqrt(np.mean(group["squared_error"]))
                         if item.metric=="rmse" else np.mean(group["abs_error"]))
                rows.append({
                    "baseline":item.baseline,"metric":item.metric,"best_n":item.best_n,
                    "hour":hour,"quintile":quintile,"days":len(group),
                    "quintile_error":float(value),
                    "mean_hourly_absolute_error":float(group["abs_error"].mean()),
                    "pooled_error_all_days":float(pooled),"cohort_days":count,
                })
    return pd.DataFrame(rows)


def _plot(quantiles, choices, group, output, *, dpi=190, figsize=(18,11)):
    fig,axes=plt.subplots(2,2,figsize=figsize,sharex=True,sharey="row")
    fig.subplots_adjust(left=0.07,right=0.98,bottom=0.14,top=0.83,
                        wspace=0.10,hspace=0.27)
    for ri,metric in enumerate(METRICS):
        for ci,baseline in enumerate(BASELINES):
            ax = axes[ri,ci]
            selected = choices.loc[(choices["baseline"]==baseline)&
                                   (choices["metric"]==metric)].iloc[0]
            data = quantiles.loc[(quantiles["baseline"]==baseline)&
                                 (quantiles["metric"]==metric)]
            for q in range(1,6):
                part = data.loc[data["quintile"]==q].sort_values("hour")
                ax.plot(part["hour"],part["quintile_error"],lw=1.8,
                        label=f"Q{q}: {(q-1)*20}–{q*20}%")
            pooled = data.loc[data["quintile"]==1].sort_values("hour")
            ax.plot(pooled["hour"],pooled["pooled_error_all_days"],
                    color="black",ls="--",lw=1.4,label="Pooled")
            ax.set_title(f"{metric.upper()} | {TITLES[baseline]} | best N={int(selected.best_n)} "
                         f"(24h {selected.selected_by.upper()} minimum)",fontsize=12)
            ax.set_xlim(1,24)
            ax.set_xticks([1,4,8,12,16,20,24])
            ax.set_xlabel("Hour of 24-hour prediction (1–24)")
            if ci==0: ax.set_ylabel(f"Hourly {metric.upper()} (normalized demand)")
            ax.tick_params(labelbottom=True)
            ax.grid(alpha=0.22)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=6,frameon=False,
               bbox_to_anchor=(0.52,0.04))
    earliest=pd.to_datetime(choices["first_day"]).min()
    latest=pd.to_datetime(choices["last_day"]).max()
    fig.suptitle(
        f"ARM (Affine Revised Model) | {group} | historical DEMAND PREDICTION "
        f"hourly RMSE / MAE quintiles | {earliest:%Y-%m-%d} to {latest:%Y-%m-%d}\n"
        "Fixed k=24 oracle initial state per day; hourly errors ranked into "
        "five 20% day groups at each hour (Q1 lowest, Q5 highest)",
        fontsize=12,y=0.975)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=dpi,bbox_inches="tight")
    plt.close(fig)


def run(*, group="ALL", n_values=range(2,15), start_day=None, end_day=None,
        input_dir=None, output_dir=None, fit_manifest=None, selection="rmse",
        steps_per_hour=4, complete_missing=True, completion_fail_fast=False,
        min_days=5, dpi=190, figsize=(18,11)):
    if group not in ("ALL","MED"): raise ValueError("group must be ALL or MED.")
    if selection not in ("rmse","mae","per_metric"):
        raise ValueError("selection must be rmse, mae or per_metric.")
    ns=tuple(sorted(set(int(n) for n in n_values)))
    if not ns or min(ns)<2 or max(ns)>14: raise ValueError("n_values must be within 2..14.")
    if min_days<5: raise ValueError("min_days must be at least 5.")
    if start_day is not None and end_day is not None and pd.Timestamp(end_day)<pd.Timestamp(start_day):
        raise ValueError("end_day precedes start_day.")
    input_dir=Path(input_dir) if input_dir is not None else INPUT/group.lower()
    output_dir=Path(output_dir) if output_dir is not None else OUTPUT/group.lower()
    eligible=load_fitted_cohort(input_dir,baselines=BASELINES,n_values=ns,
                                fit_manifest=fit_manifest,start_day=start_day,end_day=end_day)
    scores,components=load_saved(input_dir)
    if complete_missing:
        from src.exe.run_model.wm_4.complete_oracle_from_fitted_days import complete
        scores,components=complete(scores,components,eligible,input_dir=input_dir,
                                   horizons=(24,),steps_per_hour=steps_per_hour,
                                   fail_fast=completion_fail_fast)
    if start_day is not None:
        date=pd.Timestamp(start_day).normalize()
        scores=scores.loc[pd.to_datetime(scores["day"])>=date]
        components=components.loc[pd.to_datetime(components["day"])>=date]
    if end_day is not None:
        date=pd.Timestamp(end_day).normalize()
        scores=scores.loc[pd.to_datetime(scores["day"])<=date]
        components=components.loc[pd.to_datetime(components["day"])<=date]
    scores,components=_eligible_24h(scores,components,eligible,min_days)
    choices,backlogs=_select_backlogs(scores,components,eligible,selection,min_days)
    errors=_hourly_residuals(scores,components,choices)
    quintiles=_quintiles(errors,choices)
    output_dir.mkdir(parents=True,exist_ok=True)
    target=output_dir/"prediction_error_quintiles.png"
    _plot(quintiles,choices,group,target,dpi=dpi,figsize=figsize)
    quintiles.to_csv(output_dir/"quintile_curves.csv",index=False)
    choices.to_csv(output_dir/"best_backlogs_k24.csv",index=False)
    backlogs.to_csv(output_dir/"all_backlogs_k24.csv",index=False)
    errors.to_csv(output_dir/"hourly_prediction_errors.csv",index=False)
    print(f"[ARM quintiles] Saved hourly 24h-oracle error plots: {target}")
    print(choices[["baseline","metric","best_n","selected_by",
                   "selection_24h_pooled_score","days"]].to_string(index=False))
    print("Each plotted hour uses the SAME fixed 24h oracle state per day; "
          "daily errors are re-ranked by absolute error at that hour.")
    return quintiles,choices


def main(**kwargs):
    defaults=dict(group="ALL",n_values=range(2,15),start_day=None,end_day=None,
                  input_dir=None,output_dir=None,fit_manifest=None,selection="rmse",
                  steps_per_hour=4,complete_missing=True,completion_fail_fast=False,
                  min_days=5,dpi=190,figsize=(18,11))
    return run(**(defaults|kwargs))


if __name__=="__main__":
    parser=ArgumentParser(description="ARM hourly RMSE/MAE quintiles from fixed k=24 oracle states.")
    parser.add_argument("--group",choices=("ALL","MED"),default="ALL")
    parser.add_argument("--min-weekdays",type=int,default=2)
    parser.add_argument("--max-weekdays",type=int,default=14)
    parser.add_argument("--start-day",default=None)
    parser.add_argument("--end-day",default=None)
    parser.add_argument("--input",default=None)
    parser.add_argument("--output-dir",default=None)
    parser.add_argument("--fit-manifest",default=None)
    parser.add_argument("--selection",choices=("rmse","mae","per_metric"),default="rmse")
    parser.add_argument("--steps-per-hour",type=int,default=4)
    parser.add_argument("--min-days",type=int,default=5)
    parser.add_argument("--no-complete-missing",action="store_true")
    parser.add_argument("--fail-fast",action="store_true")
    args=parser.parse_args()
    main(group=args.group,n_values=range(args.min_weekdays,args.max_weekdays+1),
         start_day=args.start_day,end_day=args.end_day,input_dir=args.input,
         output_dir=args.output_dir,fit_manifest=args.fit_manifest,
         selection=args.selection,steps_per_hour=args.steps_per_hour,
         min_days=args.min_days,complete_missing=not args.no_complete_missing,
         completion_fail_fast=args.fail_fast)
