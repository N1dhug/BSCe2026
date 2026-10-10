"""Contracts and physical-hour inputs for deterministic G; no studies or point-fit runner."""
from copy import deepcopy
import hashlib
import numpy as np
import pandas as pd
from src.method.bm_1_deterministic_response import BM1ResponseInputs

ZONE='Europe/Copenhagen'
TOLERANCE=1e-5

def calendar_inputs(frame, baseline, day, *, history=False, steps=12):
    """Use physical elapsed hours and local WM features; spring day has 23 h.

    The declared exogenous baseline grid stays fixed at 60 samples/h when the
    ODE grid is refined. No demand values are inspected for forecast creation.
    """
    day = pd.Timestamp(day).normalize()
    local_day = day.tz_localize(ZONE)
    start = local_day-pd.Timedelta(hours=24) if history else local_day
    end = (day+pd.Timedelta(days=1)).tz_localize(ZONE)
    physical = pd.date_range(start, end, freq='h', inclusive='left')
    index = physical.tz_localize(None)
    source = frame.reindex(index)[['U', 'Temp']]
    if not np.isfinite(source.to_numpy()).all() or np.any(abs(source.U.to_numpy()) > 1):
        raise ValueError(f'Incomplete/domain-invalid scenario inputs for {day}.')
    n = len(index)
    elapsed = np.arange(n*60+1, dtype=float)/60
    fine_index = (start.tz_convert('UTC')+pd.to_timedelta(elapsed, unit='h')).tz_convert(ZONE)
    temp = np.interp(elapsed, np.arange(n), source.Temp.to_numpy())
    B = baseline.predict(fine_index, temp)
    obs = frame.reindex(index).I.to_numpy() if history else None
    inp = BM1ResponseInputs(elapsed, B, (source.U.to_numpy()+1)/2, obs,
        observation_mask=np.isfinite(obs) if history else None,
        warmup_hours=24 if history else 0, steps_per_hour=steps)
    return inp, index

def _plain(value):
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, pd.Timestamp): return value.isoformat()
    return value

def _origin(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tz is not None or t != t.normalize() or t.weekday() != 0:
        raise ValueError('S3 cutoff must be a local-naive Monday midnight.')
    return t

def _frame(raw, columns):
    if (not isinstance(raw, pd.DataFrame) or not isinstance(raw.index, pd.DatetimeIndex)
            or raw.index.tz is not None or raw.index.has_duplicates
            or not raw.index.is_monotonic_increasing or raw.index.hasnans
            or not set(columns).issubset(raw)):
        raise ValueError('Require sorted unique local-naive DatetimeIndex and '+','.join(columns)+'.')
    if np.any(raw.index != raw.index.floor('h')):
        raise ValueError('S3 inputs must be hourly.')

def _day_inputs(frame, baseline, day, *, history=False, steps=12):
    # calendar_inputs uses physical hours, but returns naive local timestamps.
    # Fall-back duplication cannot be represented unambiguously in that contract.
    begin = day.tz_localize(ZONE)
    if history: begin -= pd.Timedelta(hours=24)
    end = (day+pd.Timedelta(days=1)).tz_localize(ZONE)
    local = pd.date_range(begin, end, freq='h', inclusive='left').tz_localize(None)
    if local.has_duplicates:
        raise ValueError('Autumn DST with repeated local hours is unsupported by S3 v1; supply another window.')
    return calendar_inputs(frame, baseline, day, history=history, steps=steps)

def _fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()

def score_forecast(forecast,observations):
    """Attach measured-only raw-unit scores without changing a frozen forecast."""
    if not isinstance(observations,pd.Series) or observations.index.has_duplicates:
        raise ValueError('Scoring requires a uniquely indexed raw-demand Series.')
    result=deepcopy(forecast);table=result['predictions']
    actual=pd.to_numeric(observations.reindex(table.index),errors='coerce').to_numpy(dtype=float, copy=True)
    actual[~np.isfinite(actual)]=np.nan;mask=np.isfinite(actual)
    table['observed']=actual;table['scored']=mask
    scales=result['manifest']['scales']
    result['normalized_predictions']['observed']=(actual-scales['demand_min'])/scales['demand_scale']
    result['normalized_predictions']['scored']=mask
    count=int(mask.sum())
    def metrics(prediction):
        if not count:return dict(samples=0,mse=None,rmse=None,mae=None,bias=None)
        residual=prediction.to_numpy()[mask]-actual[mask]
        mse=float(np.mean(residual**2))
        return dict(samples=count,mse=mse,rmse=float(np.sqrt(mse)),mae=float(np.mean(abs(residual))),bias=float(np.mean(residual)))
    result['scoring']=dict(status='complete' if count==len(mask) else 'partial' if count else 'not-scored',
        expected_samples=len(mask),scored_samples=count,
        metrics={name:metrics(table[name]) for name in ('prediction','baseline')},
        policy='raw units; same finite measured-demand mask; no demand filling')
    return result
