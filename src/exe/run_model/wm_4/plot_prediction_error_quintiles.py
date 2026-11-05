"""ARM prediction-error quintiles for the best 24h historical backlog.

Creates a 2x2 plot: RMSE and MAE rows, weekly-average and
weekly-temperature columns. For each (metric,baseline), select the backlog N
with the lowest POOLED error at k=24, using the same eligible historical-fit
days as the ARM prediction heatmap. Each prediction horizon k has its own
constrained oracle initial state, exactly as in sweep_weekday_horizon.

For each k, rank complete target days by their k-hour RMSE / MAE, partition
them into five approximately equal 20% groups, and plot the MEAN daily error
within each quintile. Quintile 1 contains the lowest-error 20% of days;
quintile 5 the highest-error 20%. Days are re-ranked at every k, so the five
curves describe error distributions, NOT fixed sets of days.

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
    load_saved, load_fitted_cohort, aggregate_rmse,
)


OUTPUT = ROOT/"files"/"output"/"wm_4"/"prediction_error_quintiles"
METRICS = ("rmse","mae")
HORIZONS = tuple(range(1,25))


def _complete_days(scores, components, eligible, *, min_days):
    """Require all k=1..24 scores and 24 hourly components for each target day."""
    scores = scores.copy()
    components = components.copy()
    scores["day"] = pd.to_datetime(scores["day"]).dt.normalize()
    components["day"] = pd.to_datetime(components["day"]).dt.normalize()
    scores = scores.drop_duplicates(SCORE_KEY,keep="last")
    components = components.drop_duplicates(COMPONENT_KEY,keep="last")
    for field in ("rmse","mae"):
        if field not in scores: raise ValueError(f"Saved oracle scores lack {field}.")
        scores[field] = pd.to_numeric(scores[field],errors="coerce")
    scores = scores.merge(eligible[KEY].drop_duplicates(),on=KEY,how="inner",
                          validate="many_to_one")
    components = components.merge(eligible[KEY].drop_duplicates(),on=KEY,how="inner",
                                  validate="many_to_one")
    valid = (scores["horizon_hours"].isin(HORIZONS)
             & np.isfinite(scores[["rmse","mae"]]).all(axis=1))
    scores = scores.loc[valid].copy()
    complete_scores = (scores.groupby(KEY)["horizon_hours"].nunique()==len(HORIZONS))
    complete_components = (components.groupby(KEY)["hour"].agg(
                           lambda hours:set(hours.astype(int))>=set(range(24))))
    valid_keys = complete_scores.index[complete_scores].intersection(
                 complete_components.index[complete_components])
    if not len(valid_keys):
        raise ValueError("No fitted target days have all 24 prediction horizons "
                         "and complete hourly components.")
    keys = pd.DataFrame(valid_keys.tolist(),columns=KEY)
    scores = scores.merge(keys,on=KEY,how="inner",validate="many_to_one")
    components = components.merge(keys,on=KEY,how="inner",validate="many_to_one")
    counts = scores.loc[scores["horizon_hours"]==24].groupby(["baseline","training_weekdays"]).size()
    valid_pair = set(counts.index[counts>=min_days])
    scores = scores.loc[[tuple(row) in valid_pair for row in
                        scores[["baseline","training_weekdays"]].itertuples(index=False,name=None)]]
    components = components.loc[[tuple(row) in valid_pair for row in
                                components[["baseline","training_weekdays"]].itertuples(index=False,name=None)]]
    if scores.empty:
        raise ValueError(f"No complete (baseline,N) prediction cohort has at least {min_days} target days.")
    return scores,components


def _best_backlogs(scores, components, eligible, baselines, n_values, *, selection, min_days):
    """Select by the same pooled 24h RMSE/MAE as the six-panel prediction heatmap."""
    summary = aggregate_rmse(scores,components,cohort="fitted",fitted_cohort=eligible,
                             baselines=baselines,n_values=n_values,horizons=(24,))
    summary = summary.loc[summary["days"]>=min_days].copy()
    if summary.empty: raise ValueError("No complete eligible k=24 prediction scores to rank.")
    selections = []
    for baseline in baselines:
        sample = summary.loc[summary["baseline"]==baseline].copy()
        if sample.empty: raise ValueError(f"No complete k=24 predictions for {baseline}.")
        for metric in METRICS:
            # Default: pick the k=24 RMSE best N for the RMSE row, and the
            # k=24 MAE best N for the MAE row (one per plotted metric/baseline).
            rank_metric = metric if selection=="per_metric" else selection
            column = {"rmse":"aggregate_rmse","mae":"aggregate_mae"}[rank_metric]
            sorted_rows = sample.sort_values([column,"training_weekdays"],kind="stable")
            winner = sorted_rows.iloc[0]
            selections.append({
                "baseline":baseline,"metric":metric,
                "selected_by":rank_metric,"best_n":int(winner["training_weekdays"]),
                "selection_24h_pooled_score":float(winner[column]),
                "selection_24h_rmse":float(winner["aggregate_rmse"]),
                "selection_24h_mae":float(winner["aggregate_mae"]),
                "days":int(winner["days"]),
                "first_day":pd.NaT,"last_day":pd.NaT,
            })
    return pd.DataFrame(selections),summary


def _quintiles(scores, choices):
    """Return per-hour 20% rank-bin means and the true pooled reference curve."""
    rows, cohorts = [],[]
    for item in choices.itertuples(index=False):
        subset = scores.loc[(scores["baseline"]==item.baseline)&
                            (scores["training_weekdays"]==item.best_n)].copy()
        assert subset["day"].nunique()>=5
        # Complete-day filter guarantees the same set of days for all 24 k.
        count = int(subset["day"].nunique())
        first,last = subset["day"].min(),subset["day"].max()
        cohorts.append({
            "baseline":item.baseline,"metric":item.metric,
            "best_n":item.best_n,"days":count,
            "first_day":first,"last_day":last,
        })
        for k in HORIZONS:
            part = subset.loc[subset["horizon_hours"]==k,["day","rmse","mae"]]
            if len(part)!=count or part["day"].nunique()!=count:
                raise ValueError("Unexpected mismatch in complete daily k-horizon cohort.")
            part = part.sort_values([item.metric,"day"],kind="stable").reset_index(drop=True)
            # Five disjoint rank bins; sizes differ by at most one.
            buckets = np.array_split(np.arange(count),5)
            pooled = (float(np.sqrt(np.mean(part["rmse"].to_numpy(dtype=float)**2)))
                      if item.metric=="rmse" else float(part["mae"].mean()))
            for quintile,indices in enumerate(buckets,1):
                block = part.iloc[indices]
                values = block[item.metric].to_numpy(dtype=float)
                rows.append({
                    "baseline":item.baseline,"metric":item.metric,"best_n":item.best_n,
                    "horizon_hours":k,"quintile":quintile,"days":len(values),
                    "mean_daily_error":float(np.mean(values)),
                    "median_daily_error":float(np.median(values)),
                    "min_daily_error":float(np.min(values)),
                    "max_daily_error":float(np.max(values)),
                    "pooled_error_all_days":pooled,
                    "cohort_days":count,
                })
    return pd.DataFrame(rows),pd.DataFrame(cohorts)


def _plot(quantiles, selections, *, group, output, dpi=190, figsize=(18,11)):
    fig,axes = plt.subplots(2,2,figsize=figsize,sharex=True,sharey="row")
    fig.subplots_adjust(left=0.07,right=0.98,bottom=0.14,top=0.83,
                        wspace=0.10,hspace=0.27)
    x = np.array(HORIZONS)
    for ri,metric in enumerate(METRICS):
        for ci,baseline in enumerate(BASELINES):
            ax = axes[ri,ci]
            chosen = selections.loc[(selections["baseline"]==baseline)&
                                    (selections["metric"]==metric)].iloc[0]
            part = quantiles.loc[(quantiles["baseline"]==baseline)&
                                 (quantiles["metric"]==metric)]
            for q in range(1,6):
                subset = part.loc[part["quintile"]==q].sort_values("horizon_hours")
                ax.plot(subset["horizon_hours"],subset["mean_daily_error"],
                        lw=1.8,label=f"Q{q}: {(q-1)*20}–{q*20}%")
            ref = part.loc[part["quintile"]==1].sort_values("horizon_hours")
            ax.plot(ref["horizon_hours"],ref["pooled_error_all_days"],
                    color="black",linestyle="--",linewidth=1.35,label="Pooled")
            ax.set_title(f"{metric.upper()} | {TITLES[baseline]} | best N={int(chosen.best_n)} "
                         f"(24h {chosen.selected_by.upper()} minimum)",fontsize=12)
            ax.set_xticks([1,4,8,12,16,20,24])
            ax.set_xlim(1,24)
            ax.set_xlabel("Prediction horizon k (hours scored)")
            if ci==0: ax.set_ylabel(f"Daily {metric.upper()} (normalized demand)")
            ax.grid(alpha=0.22)
            ax.tick_params(labelbottom=True)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=6,frameon=False,bbox_to_anchor=(0.52,0.04))
    first,last = quantiles["horizon_hours"].min(),quantiles["horizon_hours"].max()
    ranges=selections[["first_day","last_day"]].dropna()
    from_day=pd.Timestamp(ranges["first_day"].min())
    to_day=pd.Timestamp(ranges["last_day"].max())
    title=(f"ARM (Affine Revised Model) | {group} | historical DEMAND PREDICTION "
           f"RMSE / MAE quintiles | {from_day:%Y-%m-%d} to {to_day:%Y-%m-%d}\n"
           "Saved constrained 24h theta and fitted oracle initial states; "
           "five 20% day-ranked groups at each horizon (Q1 lowest, Q5 highest)")
    fig.suptitle(title,fontsize=12,y=0.975)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=dpi,bbox_inches="tight")
    plt.close(fig)


def run(*, group="ALL", n_values=range(2,15), start_day=None, end_day=None,
        input_dir=None, output_dir=None, fit_manifest=None, selection="per_metric",
        steps_per_hour=4, complete_missing=True, completion_fail_fast=False,
        min_days=5, dpi=190, figsize=(18,11)):
    """Generate the 2x2 quintile curves, selected backlogs, and numerical CSVs."""
    if group not in ("ALL","MED"): raise ValueError("group must be ALL or MED.")
    if selection not in ("per_metric","rmse","mae"):
        raise ValueError("selection must be per_metric, rmse or mae.")
    min_days = int(min_days)
    if min_days<5: raise ValueError("min_days must be at least five to form quintiles.")
    ns = tuple(sorted(set(int(n) for n in n_values)))
    if not ns or any(n<2 or n>14 for n in ns):
        raise ValueError("n_values must contain integers between 2 and 14.")
    if start_day is not None and end_day is not None:
        if pd.Timestamp(end_day) < pd.Timestamp(start_day):
            raise ValueError("end_day precedes start_day.")
    input_dir = Path(input_dir) if input_dir is not None else INPUT/group.lower()
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT/group.lower()
    eligible = load_fitted_cohort(input_dir,baselines=BASELINES,n_values=ns,
                                  fit_manifest=fit_manifest,start_day=start_day,end_day=end_day)
    scores,components = load_saved(input_dir)
    if complete_missing:
        from src.exe.run_model.wm_4.complete_oracle_from_fitted_days import complete
        scores,components = complete(scores,components,eligible,input_dir=input_dir,
                                     horizons=HORIZONS,steps_per_hour=steps_per_hour,
                                     fail_fast=completion_fail_fast)
    if start_day is not None:
        start = pd.Timestamp(start_day).normalize()
        scores=scores.loc[scores["day"]>=start]
        components=components.loc[components["day"]>=start]
    if end_day is not None:
        end = pd.Timestamp(end_day).normalize()
        scores=scores.loc[scores["day"]<=end]
        components=components.loc[components["day"]<=end]
    scores,components = _complete_days(scores,components,eligible,min_days=min_days)
    choices,backlogs = _best_backlogs(scores,components,eligible,BASELINES,ns,
                                     selection=selection,min_days=min_days)
    quantiles,cohorts = _quintiles(scores,choices)
    choices = choices.drop(columns=["first_day","last_day"]).merge(
              cohorts,on=["baseline","metric","best_n"],how="left",
              validate="one_to_one",suffixes=("","_observed"))
    output_dir.mkdir(parents=True,exist_ok=True)
    target=output_dir/"prediction_error_quintiles.png"
    _plot(quantiles,choices,group=group,output=target,dpi=dpi,figsize=figsize)
    quantiles.to_csv(output_dir/"quintile_curves.csv",index=False)
    choices.to_csv(output_dir/"best_backlogs_k24.csv",index=False)
    backlogs.to_csv(output_dir/"all_backlogs_k24.csv",index=False)
    print(f"[ARM quintiles] Saved: {target}")
    print(choices[["baseline","metric","best_n","selected_by","selection_24h_pooled_score","days"]]
          .to_string(index=False))
    print("Quintiles are re-ranked by daily k-hour error at every horizon; "
          "these are retrospective oracle fits of the target initial state.")
    return quantiles,choices


def main(**kwargs):
    defaults = dict(group="ALL",n_values=range(2,15),start_day=None,end_day=None,
                    input_dir=None,output_dir=None,fit_manifest=None,
                    selection="per_metric",steps_per_hour=4,complete_missing=True,
                    completion_fail_fast=False,min_days=5,dpi=190,figsize=(18,11))
    return run(**(defaults|kwargs))


if __name__=="__main__":
    parser=ArgumentParser(description="ARM 2x2 best-N prediction RMSE/MAE quintile curves.")
    parser.add_argument("--group",choices=("MED","ALL"),default="ALL")
    parser.add_argument("--min-weekdays",type=int,default=2)
    parser.add_argument("--max-weekdays",type=int,default=14)
    parser.add_argument("--start-day",default=None)
    parser.add_argument("--end-day",default=None)
    parser.add_argument("--input",default=None)
    parser.add_argument("--output-dir",default=None)
    parser.add_argument("--fit-manifest",default=None)
    parser.add_argument("--selection",choices=("per_metric","rmse","mae"),default="per_metric")
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
