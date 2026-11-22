"""Canonical construction of hourly and corresponding-weekday moving-horizon datasets.

This module owns historical-calendar selection, isolated baseline fits, fine-grid
model inputs and reusable evaluation helpers. Executables only orchestrate runs.
"""
import numpy as np
import pandas as pd

from src.tools.import_data.aligned import aligned_data
from src.model.wm_4 import wm_4
from src.tools.baseline import output_callable, statistical_output, weekly_average, weekly_temperature


GROUPS = ("ALL", "MED")
HOURS = 24

DEFAULT_TRAINING_WEEKDAYS = 5
BASELINES = {"statistical_output":statistical_output,
             "weekly_average":weekly_average,
             "weekly_temperature":weekly_temperature}


def group_frame(group):
    if group not in GROUPS: raise ValueError(f"group must be one of {GROUPS}.")
    frame, _ = aligned_data(None if group == "ALL" else "MED", price_domain=(-1.0, 1.0), price_year=2022)
    return frame


def origins(start, end, stride=1):
    start, end, stride = pd.Timestamp(start), pd.Timestamp(end), int(stride)
    if end < start or stride < 1: raise ValueError("end must be >= start and stride must be positive.")
    return pd.date_range(start, end, freq=f"{stride}h")


def metrics(observed, predicted):
    error = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    if error.size < 1 or not np.all(np.isfinite(error)): raise ValueError("Metrics require finite observations.")
    return {
        "mse": float(np.mean(error ** 2)), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)),
    }


def historical_weekdays(day, count=DEFAULT_TRAINING_WEEKDAYS):
    return [pd.Timestamp(day) - pd.Timedelta(weeks=k) for k in range(count, 0, -1)]


def baseline_from_history(frame, history, name, horizon_hours=HOURS):
    """Fit the baseline ONLY to N corresponding historical intervals of k hours."""
    if name not in BASELINES: raise ValueError(f"Unknown baseline: {name}.")
    horizon_hours = int(horizon_hours)
    if not 1 <= horizon_hours <= HOURS: raise ValueError("horizon_hours must lie in [1,24].")
    samples = [frame.reindex(pd.date_range(past, periods=horizon_hours, freq="h"))[["I", "Temp"]]
               for past in history]
    train = pd.concat(samples).sort_index()
    if train[["I", "Temp"]].isna().any().any():
        raise ValueError("All selected historical baseline samples must be complete.")
    return BASELINES[name](min_samples=max(2, min(48, len(train)))).fit(
        train, max(history)+pd.Timedelta(hours=horizon_hours), exact_historical=True)

def interval_window(frame, start, baseline, steps_per_hour, horizon_hours):
    """Build only the requested k-hour window, not a full-day data window.

    The k hourly observations at hours 0..k-1 are scored. For k<24,
    the observation at hour k closes the k-hour historical fitting
    interval, providing the last actual rate transition. At k=24 the
    available 24 observations span 23 observed hourly transitions.
    """
    horizon_hours, steps_per_hour = int(horizon_hours), int(steps_per_hour)
    if not 1 <= horizon_hours <= HOURS: raise ValueError("horizon_hours must lie in [1,24].")
    if steps_per_hour < 2: raise ValueError("steps_per_hour must be >= 2.")
    start = pd.Timestamp(start)
    points = min(horizon_hours+1, HOURS)
    index = pd.date_range(start, periods=points, freq="h")
    source = frame.reindex(index)[["I", "U", "Temp"]]
    if source.isna().any().any():
        raise ValueError(f"Missing consumption, price or temperature in historical/prediction interval {start} (k={horizon_hours}).")
    observed = source["I"].to_numpy(dtype=float)[:horizon_hours]
    elapsed = np.arange(horizon_hours*steps_per_hour+1, dtype=float)/steps_per_hour
    fine_index = start+pd.to_timedelta(elapsed, unit="h")
    I_hour = source["I"].to_numpy(dtype=float)
    T_hour = source["Temp"].to_numpy(dtype=float)
    U_hour = source["U"].to_numpy(dtype=float)
    if points == horizon_hours:
        I_hour, T_hour, U_hour = np.r_[I_hour, I_hour[-1]], np.r_[T_hour, T_hour[-1]], np.r_[U_hour, U_hour[-1]]
    I = np.interp(elapsed, np.arange(horizon_hours+1), I_hour)
    Temp = np.interp(elapsed, np.arange(horizon_hours+1), T_hour)
    U = U_hour[np.minimum(np.floor(elapsed+1e-10).astype(int), horizon_hours)]
    O = np.asarray(baseline.predict(fine_index, Temp), dtype=float).reshape(-1)
    count = len(elapsed)
    if O.size != count or not np.isfinite(O).all(): raise ValueError("Baseline must be finite throughout the requested horizon.")
    observation_mask = np.zeros(count, dtype=bool)
    observation_mask[np.arange(min(horizon_hours+1, HOURS))*steps_per_hour] = True
    transition_mask = np.zeros(count-1, dtype=bool)
    transition_mask[:min(horizon_hours, HOURS-1)*steps_per_hour] = True
    data = {"I":I, "U":U, "PE":U.copy(), "Time":elapsed, "O":O,
            "observation_mask":observation_mask, "transition_mask":transition_mask}
    model = wm_4((0.0, horizon_hours/HOURS), count, output_callable(fine_index, O))
    return model, data, fine_index, observed

def historical_calendar_days(history, horizon_hours):
    """Calendar dates touched by the corresponding historical k-hour intervals."""
    k = int(horizon_hours)
    if not 1 <= k <= HOURS: raise ValueError("Historical horizon must lie in [1,24].")
    points = k+1 if k < HOURS else HOURS
    return pd.DatetimeIndex([
        timestamp.normalize() for origin in history
        for timestamp in pd.date_range(pd.Timestamp(origin), periods=points, freq="h")
    ]).unique().sort_values()

def hourly_origins(frame, *, max_weekdays, max_horizon, start_day=None, end_day=None,
                   max_days=0, max_origins=0, stride_hours=1):
    """Every eligible hourly origin, with complete data for all evaluated horizons."""
    stride_hours, max_days, max_origins = int(stride_hours), int(max_days), int(max_origins)
    if stride_hours < 1 or max_days < 0 or max_origins < 0:
        raise ValueError("stride_hours must be positive; max_days and max_origins must be nonnegative.")
    if not 1 <= int(max_horizon) <= HOURS or int(max_weekdays) < 2:
        raise ValueError("Invalid maximum horizon or historical weekday backlog.")
    earliest = max(frame.index.min().normalize(), pd.Timestamp("2022-01-01"))
    latest = min(frame.index.max().floor("h"), pd.Timestamp("2022-10-31 23:00"))
    # An endpoint observation at hour k is needed for fitting when k<24.
    # At k=24 the 24 available observations end at hour 23.
    forward = int(max_horizon) if int(max_horizon) < HOURS else HOURS-1
    first = earliest+pd.Timedelta(weeks=int(max_weekdays))
    last = latest-pd.Timedelta(hours=forward)
    if start_day is not None: first = max(first, pd.Timestamp(start_day))
    if end_day is not None:
        chosen_end = pd.Timestamp(end_day)
        last = min(last, chosen_end+pd.Timedelta(hours=23) if chosen_end == chosen_end.normalize() else chosen_end)
    if max_days: last = min(last, first.normalize()+pd.Timedelta(days=max_days-1, hours=23))
    if first > last: raise ValueError("No hourly prediction origins have sufficient history and a complete target horizon.")
    origins = pd.date_range(first, last, freq=f"{stride_hours}h")
    return origins[:max_origins] if max_origins else origins

def _origin(day,hour):
    date=pd.Timestamp(day)
    if pd.isna(date) or date!=date.normalize():
        raise ValueError("day must be a valid calendar date at midnight (YYYY-MM-DD).")
    if not isinstance(hour,(int,np.integer)) or not 0<=hour<=23:
        raise ValueError("hour must be an integer from 0 through 23.")
    return date+pd.Timedelta(hours=int(hour))

def _intervals(origin,k,N):
    """Chronological weekly-offset starts (oldest through most recent)."""
    if not isinstance(k,(int,np.integer)) or not 2<=k<=24:
        raise ValueError("k must be an integer number of hours between 2 and 24.")
    if not isinstance(N,(int,np.integer)) or N<1:
        raise ValueError("N must be a positive integer.")
    return [origin-pd.Timedelta(weeks=i) for i in range(N,0,-1)]

def _fit_sparse_baseline(samples,B):
    """Fit one of the existing baseline formulations on EXACT interval rows.

    Baseline.fit() ordinarily imputes complete calendar days. For this
    illustration it is important not to add out-of-window observations,
    so use the same model mathematics directly on the selected samples.
    """
    if B not in BASELINES: raise ValueError(f"B must be one of {tuple(BASELINES)}.")
    if not isinstance(samples.index,pd.DatetimeIndex) or samples.index.has_duplicates:
        raise ValueError("Baseline training requires distinct datetime-indexed samples.")
    if not {"I","Temp"}<=set(samples):
        raise ValueError("Baseline data must contain I and Temp.")
    samples=samples[["I","Temp"]].sort_index()
    if samples.empty or not np.isfinite(samples.to_numpy(dtype=float)).all():
        raise ValueError("Baseline training intervals must contain finite demand and temperature.")
    model=BASELINES[B](min_samples=2)
    index=samples.index
    demand=samples["I"].to_numpy(dtype=float)
    temp=samples["Temp"].to_numpy(dtype=float)

    if isinstance(model,weekly_temperature):
        groups=model._groups(index)
        chill=np.maximum(0.0,model.base_temperature-temp)
        mean_d=model._profile(groups,demand)
        mean_c=model._profile(groups,chill)
        centered_d=demand-mean_d[groups]
        centered_c=chill-mean_c[groups]
        denom=float(centered_c @ centered_c)
        model.beta_C=float((centered_c @ centered_d)/denom) if denom>1e-12 else 0.0
        model.chill_center=float(np.mean(chill))
        model.levels=mean_d-model.beta_C*(mean_c-model.chill_center)
        model.coef=np.r_[model.levels,model.beta_C]
    elif isinstance(model,weekly_average):
        groups=model._groups(index)
        model.levels=model._profile(groups,demand)
        model.coef=model.levels.copy()
    else:
        raw=model._features(index,temp)
        X=model._standardize_fit(raw)
        penalty=model.ridge*np.eye(X.shape[1],dtype=float)
        penalty[0,0]=0.0
        weights=np.ones(len(demand),dtype=float)
        beta=np.zeros(X.shape[1],dtype=float)
        for _ in range(model.max_iter):
            root=np.sqrt(weights)
            Xw,yw=X*root[:,None],demand*root
            beta=np.linalg.solve(Xw.T@Xw+penalty,Xw.T@yw)
            residual=demand-X@beta
            center=np.median(residual)
            scale=1.4826*np.median(np.abs(residual-center))
            if not np.isfinite(scale) or scale<1e-10: break
            limit=model.huber_delta*scale
            absolute=np.abs(residual)
            next_weights=np.ones_like(weights)
            outside=absolute>limit
            next_weights[outside]=limit/absolute[outside]
            if np.max(np.abs(next_weights-weights))<1e-6: break
            weights=next_weights
        model.coef=beta

    model.metadata={"samples":len(samples),"fitting_mode":"exact_corresponding_intervals",
                    "start":str(index.min()),"end":str(index.max())}
    return model

def _window(frame,start,k,baseline,sph):
    """Create exactly k hourly WM4 observations on the normalized-day grid."""
    hourly=pd.date_range(start,periods=k,freq="h")
    source=frame.reindex(hourly)[["I","U","Temp"]]
    if source.isna().any().any() or not np.isfinite(source.to_numpy(dtype=float)).all():
        raise ValueError(f"Missing historical inputs in [{start}, {start+pd.Timedelta(hours=k)}).")
    elapsed=np.arange(k*sph+1,dtype=float)/sph
    times=start+pd.to_timedelta(elapsed,unit="h")
    sample_hours=np.arange(k+1,dtype=float)
    D=np.interp(elapsed,sample_hours,np.r_[source["I"].to_numpy(),source["I"].iloc[-1]])
    T=np.interp(elapsed,sample_hours,np.r_[source["Temp"].to_numpy(),source["Temp"].iloc[-1]])
    U=np.r_[source["U"].to_numpy(),source["U"].iloc[-1]][
        np.minimum(np.floor(elapsed+1e-10).astype(int),k)]
    B=np.asarray(baseline.predict(times,T),dtype=float)
    if not np.isfinite(B).all() or not np.isfinite(U).all():
        raise ValueError("Baseline or price contains nonfinite samples.")
    mask=np.zeros(len(elapsed),dtype=bool)
    mask[np.arange(k)*sph]=True
    transitions=np.zeros(len(elapsed)-1,dtype=bool)
    transitions[:(k-1)*sph]=True
    data={"I":D,"U":U,"PE":U.copy(),"Time":elapsed,"O":B,
          "observation_mask":mask,"transition_mask":transitions}
    return data
