"""Two-panel pooled oracle-RMSE heatmaps from a completed WM4 N-k sweep.

Uses saved oracle_horizon_scores.csv and forecast_components.csv, never refits.
Each (N, k, day, baseline) uses the *k-specific fitted oracle state*;
sliced errors are reconstructed from the saved affine demand sensitivities.
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT / "files" / "output" / "wm_4" / "weekday_horizon_sweep_expanded"
BASELINES = ("weekly_average", "weekly_temperature")
TITLES = {"weekly_average": "Weekly average",
          "weekly_temperature": "Weekly temperature"}
KEY = ["day", "baseline", "training_weekdays"]
SCORE_KEY = KEY + ["horizon_hours"]
COMPONENT_KEY = KEY + ["hour"]
COMPONENT_COLUMNS = ("observed", "prediction_at_initial_zero", "sensitivity_F0", "sensitivity_R0")


def load_saved(input_dir):
    """Read saved estimates and independent hourly components, never fit a model."""
    input_dir = Path(input_dir)
    paths = (input_dir/"oracle_horizon_scores.csv", input_dir/"forecast_components.csv")
    for path in paths:
        if not path.is_file(): raise FileNotFoundError(f"Required completed-sweep data missing: {path}")
    scores, components = (pd.read_csv(path) for path in paths)
    needed_scores = set(SCORE_KEY + ["oracle_F0", "oracle_R0"])
    needed_components = set(COMPONENT_KEY + list(COMPONENT_COLUMNS))
    for name, frame, required in (("oracle_horizon_scores.csv", scores, needed_scores),
                                   ("forecast_components.csv", components, needed_components)):
        absent = required.difference(frame.columns)
        if absent: raise ValueError(f"{name} is missing columns: {', '.join(sorted(absent))}")
        if frame.empty: raise ValueError(f"{name} contains no saved observations.")
        frame["day"] = pd.to_datetime(frame["day"]).dt.normalize()
        for col in required.difference(("day", "baseline")):
            frame[col] = pd.to_numeric(frame[col], errors="raise")
    scores = scores.drop_duplicates(SCORE_KEY, keep="last").copy()
    components = components.drop_duplicates(COMPONENT_KEY, keep="last").copy()
    return scores, components


def selected_hours(horizon, *, hour_start=0, hour_end=None, last_hours=None, require_full_window=False):
    """Return zero-based hours scored for a k-hour oracle prediction."""
    horizon = int(horizon)
    if last_hours is not None:
        if require_full_window and horizon < last_hours: return []
        return list(range(max(0, horizon-last_hours), horizon))
    if require_full_window and hour_end is not None and horizon < hour_end: return []
    return list(range(hour_start, min(horizon, hour_end if hour_end is not None else horizon)))


def aggregate_rmse(scores, components, *, hour_start=0, hour_end=None, last_hours=None,
                   require_full_window=False, baselines=BASELINES, n_values=None, horizons=None,
                   cohort="global"):
    """Reconstruct *pooled* RMSE from each horizon's independently fitted oracle.

    hour_start inclusive, hour_end exclusive; e.g. 6:12 scores hours 7--12.
    last_hours=4 scores the last four hours of each k (mutually exclusive).
    Shorter k values have their slice truncated unless require_full_window=True.
    cohort='global' matches days across *all* B,N,k configurations;
    cohort='cell' matches days across the baselines at each (N,k).
    """
    baselines = tuple(baselines)
    if not baselines or len(set(baselines)) != len(baselines): raise ValueError("Select unique baselines.")
    if cohort not in ("global", "cell"): raise ValueError("cohort must be 'global' or 'cell'.")
    if not isinstance(hour_start, int) or not 0 <= hour_start < 24:
        raise ValueError("hour_start must be an integer from 0 through 23.")
    if hour_end is not None and (not isinstance(hour_end, int) or not hour_start < hour_end <= 24):
        raise ValueError("hour_end must exceed hour_start and be at most 24.")
    if last_hours is not None and (not isinstance(last_hours, int) or not 1 <= last_hours <= 24 or hour_start != 0 or hour_end is not None):
        raise ValueError("last_hours requires 1..24 and cannot be combined with hour_start/hour_end.")
    if not isinstance(require_full_window, bool): raise TypeError("require_full_window must be bool.")
    if not isinstance(scores, pd.DataFrame) or not isinstance(components, pd.DataFrame):
        raise TypeError("scores and components must be pandas DataFrames.")
    for name, frame, cols in (("scores", scores, SCORE_KEY+["oracle_F0", "oracle_R0"]),
                              ("components", components, COMPONENT_KEY+list(COMPONENT_COLUMNS))):
        if not set(cols).issubset(frame.columns): raise ValueError(f"{name} lacks {sorted(set(cols)-set(frame.columns))}")
    scores, components = scores.copy(), components.copy()
    scores["day"], components["day"] = (pd.to_datetime(f["day"]).dt.normalize() for f in (scores, components))
    scores = scores.drop_duplicates(SCORE_KEY, keep="last")
    components = components.drop_duplicates(COMPONENT_KEY, keep="last")
    n_values = tuple(sorted({int(x) for x in (n_values if n_values is not None else scores["training_weekdays"].unique())}))
    horizons = tuple(sorted({int(x) for x in (horizons if horizons is not None else scores["horizon_hours"].unique())}))
    if not n_values or not horizons or min(n_values) < 1 or min(horizons) < 1 or max(horizons) > 24:
        raise ValueError("Provide nonempty N and horizon selections, with horizons in 1..24.")
    scores = scores.loc[scores["baseline"].isin(baselines) & scores["training_weekdays"].isin(n_values) &
                         scores["horizon_hours"].isin(horizons)].copy()
    components = components.loc[components["baseline"].isin(baselines) &
                                 components["training_weekdays"].isin(n_values)].copy()
    if scores.empty or components.empty: raise ValueError("No saved rows match the selected baselines or N/k range.")
    present = set(scores["baseline"].unique()) & set(components["baseline"].unique())
    missing = set(baselines)-present
    if missing: raise ValueError(f"Saved sweep is missing baselines: {sorted(missing)}")

    # Only complete hourly-component trajectories are used, preventing hidden
    # loss of individual hourly errors when computing pooled RMSE.
    valid_components = components.groupby(KEY)["hour"].agg(lambda s: set(s.astype(int)) >= set(range(24)))
    valid_keys = set(valid_components.index[valid_components])
    scores = scores.loc[[tuple(row) in valid_keys for row in scores[KEY].itertuples(index=False, name=None)]].copy()
    components = components.loc[[tuple(row) in valid_keys for row in components[KEY].itertuples(index=False, name=None)]].copy()
    useful = {h: selected_hours(h, hour_start=hour_start, hour_end=hour_end,
                                last_hours=last_hours, require_full_window=require_full_window) for h in horizons}
    included = [h for h in horizons if useful[h]]
    if not included: raise ValueError("No horizons contain the selected scoring hours.")
    scores = scores.loc[scores["horizon_hours"].isin(included)]
    if scores.empty: raise ValueError("No complete score/component pairs remain after filtering.")

    if cohort == "global":
        expected = len(baselines)*len(n_values)*len(included)
        counts = scores.groupby("day").size()
        common = set(counts.index[counts == expected])
        if not common: raise ValueError("No prediction days match across all baselines, N values and scored horizons.")
        scores = scores.loc[scores["day"].isin(common)]
    else:
        # This matches baselines for each (N,k); the count can vary by cell.
        count = scores.groupby(["day", "training_weekdays", "horizon_hours"])["baseline"].nunique()
        matched = set(count.index[count == len(baselines)])
        scores = scores.loc[[tuple(row) in matched for row in
                             scores[["day", "training_weekdays", "horizon_hours"]].itertuples(index=False, name=None)]]
        if scores.empty: raise ValueError("No (N,k) cells contain matched days across all baselines.")

    grouped_components = {(b, int(n)): frame for (b, n), frame in components.groupby(["baseline", "training_weekdays"])}
    grouped_scores = {(b, int(n), int(h)): frame for (b, n, h), frame in scores.groupby(["baseline", "training_weekdays", "horizon_hours"])}
    rows = []
    for b in baselines:
        for n in n_values:
            for h in horizons:
                hours = useful[h]
                entry = {"baseline": b, "training_weekdays": n, "horizon_hours": h,
                         "hour_start": min(hours) if hours else np.nan,
                         "hour_end": max(hours)+1 if hours else np.nan,
                         "hours_scored": len(hours), "days": 0, "observations": 0, "aggregate_rmse": np.nan}
                scored = grouped_scores.get((b, n, h))
                trajectory = grouped_components.get((b, n))
                if hours and scored is not None and trajectory is not None:
                    part = trajectory.loc[trajectory["hour"].isin(hours), COMPONENT_KEY+list(COMPONENT_COLUMNS)]
                    joined = scored[["day", "oracle_F0", "oracle_R0"]].merge(part, on="day", how="left", validate="one_to_many")
                    expected = len(scored)*len(hours)
                    if len(joined) != expected or joined[list(COMPONENT_COLUMNS)].isna().any().any():
                        raise ValueError(f"Incomplete saved hourly components: baseline={b}, N={n}, k={h}.")
                    residual = (joined["prediction_at_initial_zero"].to_numpy(dtype=float)+
                                joined["oracle_F0"].to_numpy(dtype=float)*joined["sensitivity_F0"].to_numpy(dtype=float)+
                                joined["oracle_R0"].to_numpy(dtype=float)*joined["sensitivity_R0"].to_numpy(dtype=float)-
                                joined["observed"].to_numpy(dtype=float))
                    if not np.isfinite(residual).all(): raise ValueError(f"Nonfinite reconstructed oracle errors: {b}, N={n}, k={h}.")
                    entry.update(days=len(scored), observations=len(residual),
                                 aggregate_rmse=float(np.sqrt(np.mean(residual**2))))
                rows.append(entry)
    return pd.DataFrame(rows)


def draw_heatmaps(summary, *, baselines=BASELINES, output=None, cmap="viridis", vmin=0.0,
                  vmax=None, dpi=190, figsize=(19, 6.8), title=None):
    """Exactly two baseline panes, one common normalization and one colorbar."""
    baselines = tuple(baselines)
    if len(baselines) != 2: raise ValueError("Two-panel figure requires exactly two baselines.")
    ns = sorted(summary["training_weekdays"].unique())
    ks = sorted(summary["horizon_hours"].unique())
    finite = summary["aggregate_rmse"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite): raise ValueError("No finite aggregate RMSE values to plot.")
    if vmax is None:
        vmax = float(np.max(finite))
        if vmax == vmin: vmax = vmin + 1e-12  # E.g. exactly fitted k=1 oracle results.
    else: vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Color normalization must satisfy finite vmin < vmax.")
    norm = Normalize(vmin=vmin, vmax=vmax)
    color = plt.get_cmap(cmap).copy()
    color.set_bad("#e6e8ea")
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, baseline in zip(axes, baselines):
        subset = summary.loc[summary["baseline"] == baseline]
        pivot = subset.pivot(index="training_weekdays", columns="horizon_hours", values="aggregate_rmse")
        values = pivot.reindex(index=ns, columns=ks).to_numpy(dtype=float)
        im = ax.imshow(np.ma.masked_invalid(values), origin="lower", aspect="auto",
                       interpolation="nearest", cmap=color, norm=norm)
        ax.set_title(TITLES.get(baseline, baseline), fontsize=12)
        ax.set_xticks(np.arange(len(ks)), [str(v) for v in ks], fontsize=8)
        ax.set_yticks(np.arange(len(ns)), [str(v) for v in ns])
        ax.set_xlabel("Prediction horizon $k$ (hours)")
        ax.tick_params(axis="both", length=0)
    axes[0].set_ylabel("Historical matching weekdays $N$")
    subtitle = title if title else "WM4 full-state oracle | pooled RMSE | weekly average vs weekly temperature"
    fig.suptitle(subtitle, fontsize=13)
    fig.subplots_adjust(left=0.055, right=0.90, bottom=0.13, top=0.87, wspace=0.08)
    cax = fig.add_axes((0.915, 0.18, 0.015, 0.63))
    fig.colorbar(im, cax=cax, label="Aggregate RMSE (normalized demand)")
    if output is not None:
        output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
    return fig


def plot_heatmaps(*, input_dir=INPUT/"med", output=None, hour_start=0, hour_end=None,
                  last_hours=None, require_full_window=False, cohort="global",
                  baselines=BASELINES, n_values=None, horizons=None, **plot_kwargs):
    """Plot saved two-baseline N-k RMSE with a selectable score window.

    Examples: plot_heatmaps(hour_start=6, hour_end=12)
              plot_heatmaps(last_hours=4)
              plot_heatmaps(hour_start=0, hour_end=12, require_full_window=True)
    """
    input_dir = Path(input_dir)
    scores, components = load_saved(input_dir)
    summary = aggregate_rmse(scores, components, hour_start=hour_start, hour_end=hour_end,
                             last_hours=last_hours, require_full_window=require_full_window,
                             cohort=cohort, baselines=baselines, n_values=n_values, horizons=horizons)
    selected = (f"last_{last_hours:02d}h" if last_hours is not None else
                f"hours_{hour_start:02d}_{hour_end if hour_end is not None else 'end'}")
    stem = f"aggregate_rmse_two_baselines_{selected}_{cohort}"
    output = Path(output) if output is not None else input_dir/(stem+".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output.with_suffix(".csv"), index=False)
    note = (f"last {last_hours} h of each horizon" if last_hours is not None else
            f"hours {hour_start+1}--{hour_end if hour_end is not None else 'k'} (end exclusive in zero-based indexing)")
    fig = draw_heatmaps(summary, baselines=baselines, output=output,
                        title=f"WM4 oracle RMSE by baseline, N and k | {note}", **plot_kwargs)
    plt.close(fig)
    print(f"Saved figure: {output}")
    print(f"Saved pooled metrics: {output.with_suffix('.csv')}")
    print(f"Cohort: {cohort}; matched days per scored cell: "
          f"{int(summary.loc[summary['days']>0, 'days'].min())}--{int(summary['days'].max())}")
    return summary


def main():
    parser = ArgumentParser(description="Plot 2-panel pooled N-k oracle RMSE (weekly average/temperature) from completed WM4 sweep CSVs, without refitting.")
    parser.add_argument("--group", choices=("MED", "ALL"), default="MED")
    parser.add_argument("--input", default=None, help="Directory with oracle_horizon_scores.csv and forecast_components.csv.")
    parser.add_argument("--output", default=None, help="Output PNG path; a same-name CSV is written alongside.")
    parser.add_argument("--hour-start", type=int, default=0, help="First scored prediction hour, zero-based inclusive.")
    parser.add_argument("--hour-end", type=int, default=None, help="Last scored hour, zero-based exclusive; default k.")
    parser.add_argument("--last-hours", type=int, default=None, help="Score the final L hours of each k instead.")
    parser.add_argument("--require-full-window", action="store_true", help="Leave cells blank where k is too short for the requested slice.")
    parser.add_argument("--cohort", choices=("global", "cell"), default="global")
    parser.add_argument("--min-weekdays", type=int, default=None)
    parser.add_argument("--max-weekdays", type=int, default=None)
    parser.add_argument("--min-horizon", type=int, default=None)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--vmax", type=float, default=None, help="Optional common upper color limit.")
    args = parser.parse_args()
    folder = Path(args.input) if args.input is not None else INPUT/args.group.lower()
    scores, _ = load_saved(folder)
    ns = sorted(set(scores["training_weekdays"].astype(int)))
    ks = sorted(set(scores["horizon_hours"].astype(int)))
    ns = [n for n in ns if (args.min_weekdays is None or n >= args.min_weekdays) and
          (args.max_weekdays is None or n <= args.max_weekdays)]
    ks = [k for k in ks if (args.min_horizon is None or k >= args.min_horizon) and
          (args.max_horizon is None or k <= args.max_horizon)]
    plot_heatmaps(input_dir=folder, output=args.output, hour_start=args.hour_start,
                  hour_end=args.hour_end, last_hours=args.last_hours,
                  require_full_window=args.require_full_window, cohort=args.cohort,
                  n_values=ns, horizons=ks, vmax=args.vmax)


if __name__ == "__main__": main()
