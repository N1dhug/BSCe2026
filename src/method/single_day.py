"""Single-day historical fitting preparation shared by BM1 and ARM workflows.

The historical fit uses only N corresponding weekly-offset k-hour windows.
Missing historical hours may be imputed within their calendar days; target
observations are neither imputed nor used for historical fitting.
"""
import numpy as np
import pandas as pd

from src.method.moving_horizon import BASELINES, _origin, _intervals, baseline_from_history, group_frame
from src.tools.impute_hourly import impute_hourly_days


def selection(day, origin_point, k, N, baseline):
    """Return an hourly prediction origin and N chronological weekly histories."""
    if isinstance(origin_point, (bool, np.bool_)): raise ValueError("origin_point must be an hour in [0,23].")
    origin = _origin(day, origin_point)
    if origin.tzinfo is not None: raise ValueError("The prediction day must be timezone-naive.")
    history = _intervals(origin, k, N)
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {tuple(BASELINES)}.")
    return origin, history


def prepare(day, origin_point, k, N, baseline, *, group="ALL", bm_price=False):
    """Load hourly data and fit an exogenous baseline on historical windows only.

    BM1 expects price U in [0,1]; ARM uses the signed price domain [-1,1].
    The baseline itself is fitted against historical demand and temperature,
    independent of the price convention.
    """
    origin, history = selection(day, origin_point, k, N, baseline)
    raw = group_frame(group).copy().sort_index()
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.has_duplicates or raw.index.tz is not None:
        raise ValueError("Source data must have a unique timezone-naive hourly index.")
    if not {"I", "U", "Temp"} <= set(raw): raise ValueError("Missing I, U, or Temp in source data.")
    if bm_price: raw["U"] = .5 * (raw["U"] + 1.)
    target_index = pd.date_range(origin, periods=k, freq="h")
    target_frame = raw.reindex(target_index)[["I", "U", "Temp"]]
    if not np.isfinite(target_frame.to_numpy(dtype=float)).all():
        raise ValueError(f"The target interval starting {origin} must have complete observed I, U, and Temp.")

    calendar_days = pd.DatetimeIndex([
        timestamp.normalize() for past in history
        for timestamp in pd.date_range(past, periods=k, freq="h")
    ]).unique().sort_values()
    training, imputation = impute_hourly_days(raw, days=calendar_days,
                                               columns=("I", "U", "Temp"), require_days=True)
    fitted_baseline = baseline_from_history(training, history, baseline, k)
    return {"origin":origin, "history":history, "raw":raw, "training":training,
            "baseline":fitted_baseline, "imputation":imputation}


def interval_score(observed, simulation, lower, upper):
    """Score one conditional or marginal BM1 density trajectory."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(simulation["mean"], dtype=float)
    logpdf = np.asarray(simulation["loglik"], dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if not (observed.shape == predicted.shape == logpdf.shape == lower.shape == upper.shape):
        raise ValueError("Observed, expected, NLL and interval arrays must agree.")
    if observed.size < 1 or not np.isfinite(np.r_[observed,predicted,logpdf,lower,upper]).all():
        raise ValueError("Scores require nonempty finite arrays.")
    if np.any(lower > upper): raise ValueError("Reversed observation interval limits.")
    residual = predicted-observed
    covered = (observed >= lower) & (observed <= upper)
    return {"observations":int(observed.size),"nll_total":float(-logpdf.sum()),
            "nll_mean":float(-logpdf.mean()),"sse":float(residual@residual),
            "rmse":float(np.sqrt(np.mean(residual**2))),
            "mae":float(np.mean(np.abs(residual))),"bias":float(np.mean(residual)),
            "covered":int(covered.sum()),"coverage_95_pct":float(100.*covered.mean())}


def pooled_scores(scores):
    if not scores: raise ValueError("At least one scored interval is required.")
    n = sum(item["observations"] for item in scores)
    if n <= 0: raise ValueError("The pooled observation count must be positive.")
    return {"observations":int(n),
            "nll_total":float(sum(item["nll_total"] for item in scores)),
            "nll_mean":float(sum(item["nll_total"] for item in scores)/n),
            "rmse":float(np.sqrt(sum(item["sse"] for item in scores)/n)),
            "covered":int(sum(item["covered"] for item in scores)),
            "coverage_95_pct":float(100.*sum(item["covered"] for item in scores)/n)}
