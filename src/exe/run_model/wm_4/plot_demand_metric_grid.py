"""Shared 3-row x 2-column ARM demand-error heatmap renderer.

Each row (RMSE, MAE, BIAS) uses one scale and one colorbar shared by the
weekly-average and weekly-temperature panels. Positive bias means that
fitted/predicted demand exceeds observed demand.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


METRICS = (("rmse", "RMSE"), ("mae", "MAE"), ("bias", "BIAS"))


def draw_demand_metric_grid(summary, metric_columns, *, baselines, titles,
                            group, period, evaluation, subtitle, horizon_label,
                            output=None, figsize=(19, 15), dpi=190, cmap="viridis",
                            bias_cmap="coolwarm", rmse_vmax=None):
    """Render two baselines across columns and RMSE/MAE/signed BIAS down rows.

    The metric_columns mapping supplies the three pooled statistic columns.
    Within each metric row both baselines have an identical normalization;
    the BIAS row is centered on zero with symmetric limits.
    """
    baselines = tuple(baselines)
    if len(baselines)!=2 or len(set(baselines))!=2:
        raise ValueError("The ARM metric grid requires exactly two distinct baselines.")
    missing = set(metric_columns[k] for k,_ in METRICS)-set(summary.columns)
    if missing: raise ValueError(f"Missing pooled metric columns: {sorted(missing)}")
    ns = sorted(summary["training_weekdays"].unique())
    ks = sorted(summary["horizon_hours"].unique())
    if not ns or not ks: raise ValueError("No backlog/horizon values are available.")

    normalizations, cmaps = {}, {}
    for key,label in METRICS:
        data = summary[metric_columns[key]].to_numpy(dtype=float)
        finite = data[np.isfinite(data)]
        if not finite.size: raise ValueError(f"No finite {label} values are available for the metric grid.")
        if key=="bias":
            extent = max(float(np.max(np.abs(finite))),1e-12)
            normalizations[key] = TwoSlopeNorm(vmin=-extent,vcenter=0.0,vmax=extent)
            palette = plt.get_cmap(bias_cmap).copy()
        else:
            limit = (float(rmse_vmax) if key=="rmse" and rmse_vmax is not None
                     else float(np.max(finite)))
            if not np.isfinite(limit) or limit<=0:
                if key=="rmse" and rmse_vmax is not None:
                    raise ValueError("rmse_vmax must be positive and finite.")
                limit = 1e-12
            normalizations[key] = Normalize(vmin=0.0,vmax=limit)
            palette = plt.get_cmap(cmap).copy()
        palette.set_bad("#e5e7eb")
        cmaps[key] = palette

    fig,axes = plt.subplots(3,2,figsize=figsize,sharex=True,sharey=True)
    fig.subplots_adjust(left=0.06,right=0.89,bottom=0.07,top=0.89,hspace=0.30,wspace=0.08)
    for ri,(key,label) in enumerate(METRICS):
        mappable = None
        for ci,baseline in enumerate(baselines):
            ax = axes[ri,ci]
            subset = summary.loc[summary["baseline"]==baseline]
            grid = (subset.pivot(index="training_weekdays",columns="horizon_hours",
                                 values=metric_columns[key])
                    .reindex(index=ns,columns=ks).to_numpy(dtype=float))
            mappable = ax.imshow(np.ma.masked_invalid(grid),origin="lower",
                                  aspect="auto",interpolation="nearest",
                                  norm=normalizations[key],cmap=cmaps[key])
            ax.set_title(f"{label} | {titles.get(baseline,baseline)}",fontsize=12)
            ax.set_xticks(range(len(ks)),[str(k) for k in ks],fontsize=8)
            ax.set_yticks(range(len(ns)),[str(n) for n in ns],fontsize=9)
            ax.tick_params(axis="both",length=0,labelbottom=True,labelleft=(ci==0))
            ax.set_xlabel(horizon_label)
            if ci==0: ax.set_ylabel("Backlog N (preceding matching weekdays)")

        # One independent colorbar per metric, aligned with this pair of plots.
        pos = axes[ri,1].get_position()
        cax = fig.add_axes((0.915,pos.y0+0.04*pos.height,0.014,0.92*pos.height))
        meaning = (f"Pooled demand {evaluation.lower()} {label} (normalized demand)"
                   if key!="bias"
                   else f"Pooled demand {evaluation.lower()} BIAS (model − observed)")
        fig.colorbar(mappable,cax=cax,label=meaning)

    fig.suptitle(f"ARM (Affine Revised Model) | {group} | pooled historical "
                 f"DEMAND {evaluation.upper()} RMSE / MAE / BIAS | {period}\n"
                 f"{subtitle}",fontsize=12,y=0.975)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=dpi,bbox_inches="tight")
    return fig
