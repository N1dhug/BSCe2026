"""Historical BM1 data loading and signal preparation; no fitting or optimization."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.import_data.import_input import input_data
from src.tools.import_data.import_prices import spot_price_data
from src.tools.input_generator.input_builder import input_builder
from src.validation.control import positive_integer


ROOT = Path(__file__).resolve().parents[2]
SEASON_START = pd.Timestamp('2022-01-01')
SEASON_END = pd.Timestamp('2022-11-01')
HOURLY_EDGES = np.linspace(0.0, 1.0, 25, dtype=float)
DEFAULT_BASELINE_WEEKS = 5
DEFAULT_DAYS = 3
DATA_DIR = ROOT / 'files' / 'input' / 'data'


@dataclass(frozen=True)
class Normalization:
    """Fixed consumption/price scaling used by the retrospective dataset."""
    consumption_min: float
    consumption_max: float
    price_min: float
    price_max: float


@dataclass(frozen=True)
class DaySignals:
    """One complete day of target, baseline and observed prices, with source history."""
    day: pd.Timestamp
    target: np.ndarray
    baseline: np.ndarray
    observed_price: np.ndarray
    raw_target: np.ndarray
    raw_price: np.ndarray
    baseline_days: tuple


def _aware_series(values, name):
    """H inputs retain timezone, duplicate-time and missing-value information."""
    if not isinstance(values, pd.Series) or not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError(f'{name} must be one explicitly selected timestamped Series.')
    if values.index.tz is None or values.index.hasnans or not values.index.is_unique or not values.index.is_monotonic_increasing:
        raise ValueError(f'{name} requires timezone-aware, unique, increasing timestamps.')
    if np.iscomplexobj(values.to_numpy()): raise ValueError(f'{name} must be real.')
    array = pd.to_numeric(values, errors='raise').to_numpy(dtype=float)
    if np.any(np.isinf(array)): raise ValueError(f'{name} must contain finite values or explicit NaN missing values.')
    return pd.Series(array, index=values.index.copy(), name=values.name)


def previous_weekday_mean(history, target_times, *, origin):
    """H/WM reference: arithmetic mean of five previous same-weekday/hour values.

    All five values must precede the fixed forecast origin. Local calendar
    weeks preserve the hour across DST; ambiguous/nonexistent history hours
    require an explicit data decision and are rejected. No trimmed mean,
    nearest-time fill, target-day values, or future history is substituted.
    F3's separate retrospective trimmed-mean experiment remains unchanged.
    """
    history = _aware_series(history, 'history')
    targets = pd.DatetimeIndex(target_times)
    origin = pd.Timestamp(origin)
    if targets.tz is None or origin.tzinfo is None:
        raise ValueError('Target times and origin require an explicit timezone.')
    if targets.hasnans or not targets.is_unique or not targets.is_monotonic_increasing or np.any(targets < origin):
        raise ValueError('Target times must be unique, increasing and at or after origin.')
    local = targets.tz_convert(history.index.tz)
    source_times, values = [], []
    for target in local:
        candidates = []
        week = 1
        while len(candidates)<5:
            # DateOffset is a local calendar week; Timedelta would shift the
            # selected local hour by one at a daylight-saving transition.
            try:
                stamp = target - pd.DateOffset(weeks=week)
            except (ValueError, TypeError) as exc:
                raise ValueError('Ambiguous/nonexistent local history hour; resolve the observation contract.') from exc
            week += 1
            if stamp < origin: candidates.append(stamp)
        samples = history.reindex(pd.DatetimeIndex(candidates)).to_numpy()
        if not np.all(np.isfinite(samples)):
            raise ValueError('Five complete prior occurrences of the same weekday/hour are required.')
        source_times.append(tuple(candidates)); values.append(float(np.mean(samples)))
    return pd.Series(values, index=targets, name='previous_five_weekday_mean'), tuple(source_times)


@dataclass(frozen=True)
class EstimationPowerScale:
    """Positive physical power scale, optionally fitted strictly before a cutoff."""
    power: float
    unit: str
    fitted_before: pd.Timestamp | None = None

    def __post_init__(self):
        if isinstance(self.power, (bool, np.bool_)) or not np.isfinite(self.power) or self.power<=0:
            raise ValueError('Power scale must be finite and positive.')
        if self.unit not in ('W','kW','MW'):
            raise ValueError('Declare the power unit as W, kW or MW; energy requires an interval observation model.')
        if self.fitted_before is not None:
            cutoff = pd.Timestamp(self.fitted_before)
            if cutoff.tzinfo is None: raise ValueError('Training cutoff requires a timezone.')
            object.__setattr__(self, 'fitted_before', cutoff)

    @classmethod
    def from_training(cls, history, *, cutoff, unit):
        history = _aware_series(history, 'training history')
        cutoff = pd.Timestamp(cutoff)
        if cutoff.tzinfo is None: raise ValueError('Training cutoff requires a timezone.')
        training = history.loc[history.index < cutoff].dropna()
        if training.empty or np.any(training.to_numpy()<0):
            raise ValueError('A nonempty nonnegative training history is required.')
        return cls(float(training.max()), unit, cutoff)


def historical_point_observations(measurements, *, origin, scale, measurement_id,
                                  observation_kind, price_side='right'):
    """H adapter only: selected point power readings, no fit or price assumption.

    Callers must establish that rows are point measurements for the declared
    population. Hourly energy/average power is deliberately not relabeled as a
    point observation. Missing rows must be represented explicitly in the input
    index as NaN; model time is actual elapsed UTC time in days, including DST.
    """
    from src.params.bm_1 import BM1ObservationSeries
    if observation_kind!='point':
        raise NotImplementedError('Interval power/energy needs a separate observation likelihood before H fitting.')
    if not isinstance(scale, EstimationPowerScale): raise TypeError('An explicit frozen power scale is required.')
    if not isinstance(measurement_id,str) or not measurement_id.strip():
        raise ValueError('Declare the selected measurement/population identifier.')
    data = _aware_series(measurements, 'measurements')
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None or data.empty or data.index[0]<=origin:
        raise ValueError('Timezone-aware observations must lie strictly after origin.')
    if scale.fitted_before is not None and scale.fitted_before>origin:
        raise ValueError('Training scale used data after the estimation origin.')
    days = (data.index.tz_convert('UTC')-origin.tz_convert('UTC')).total_seconds().to_numpy()/86400.
    observed = data.notna().to_numpy()
    observations = BM1ObservationSeries(days, data.to_numpy()/scale.power, 'day', observed, 'point', price_side)
    metadata = dict(measurement_id=measurement_id, timestamps=[t.isoformat() for t in data.index],
        origin=origin.isoformat(), power_scale=scale.power, power_unit=scale.unit,
        scale_fitted_before=None if scale.fitted_before is None else scale.fitted_before.isoformat(),
        missing_policy='explicit mask; no imputation', baseline_reference='five previous same-weekday/hour arithmetic mean')
    return observations, metadata


def _normalize(series, lower, upper, name):
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f'{name} normalization range must be finite with upper > lower.')
    return (series.astype(float) - lower) / (upper - lower)


def _consumption_series(data):
    columns = data.measurement_columns
    if not columns:
        raise ValueError('No consumption measurement columns found in input.xlsx.')
    values = data.observed_data.loc[:, columns].apply(pd.to_numeric, errors='coerce').mean(axis=1).dropna().sort_index()
    values.name = 'Consumption'
    return values


def _price_series(data):
    values = data.prices('DKK')
    values.index = values.index.tz_localize(None)
    values = values.groupby(level=0).mean().sort_index()
    values.name = 'SpotPriceDKK'
    return values


def load_real_signals(price_area='DK1'):
    """Load and season-normalize real aggregate consumption and spot price."""
    consumption_import = input_data(DATA_DIR / 'input.xlsx')
    price_import = spot_price_data(DATA_DIR / 'prices.xlsx', price_area=price_area)
    consumption_raw = _consumption_series(consumption_import)
    price_raw = _price_series(price_import)
    consumption_season = consumption_raw.loc[(consumption_raw.index >= SEASON_START) & (consumption_raw.index < SEASON_END)]
    price_season = price_raw.loc[(price_raw.index >= SEASON_START) & (price_raw.index < SEASON_END)]
    if consumption_season.empty or price_season.empty:
        raise ValueError('No Jan-Oct 2022 real data are available for normalization.')
    normalization = Normalization(
        consumption_min=float(consumption_season.min()),
        consumption_max=float(consumption_season.max()),
        price_min=float(price_season.min()),
        price_max=float(price_season.max()),
    )
    consumption = _normalize(
        consumption_raw,
        normalization.consumption_min,
        normalization.consumption_max,
        'consumption',
    )
    price = _normalize(price_raw, normalization.price_min, normalization.price_max, 'price')
    return (consumption_raw, price_raw, consumption, price, normalization)


def _day_index(day):
    day = pd.Timestamp(day).normalize()
    return pd.date_range(day, day + pd.Timedelta(days=1), freq='h', inclusive='left')


def _complete_day_values(series, day, name):
    index = _day_index(day)
    values = series.reindex(index)
    if values.isna().any():
        missing = list(values.index[values.isna()])
        raise ValueError(f'{pd.Timestamp(day):%Y-%m-%d} has incomplete {name}; missing={missing[:5]}.')
    array = values.to_numpy(dtype=float)
    if array.shape != (24,) or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain 24 finite hourly values.')
    return array


def _representative_quantiles(n_days):
    n_days = positive_integer(n_days, 'n_days', 1)
    if n_days == 1:
        return np.array([0.5], dtype=float)
    if n_days == 2:
        return np.array([0.25, 0.75], dtype=float)
    return np.linspace(0.15, 0.85, n_days)


def _hourly_hold(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape != (24,):
        raise ValueError('Hourly held signals must contain 24 values.')
    return input_builder.price_intervals((HOURLY_EDGES, values))


def _objective_dict(evaluation):
    objective = evaluation.objective
    return {
        'total': float(objective.total),
        'bias': float(objective.bias),
        'variance': float(objective.variance),
    }


def _mc_summary(item):
    """Serialize one MonteCarloCandidateResult using its complete control cost."""
    return {'total': float(item.total), 'standard_error': float(item.standard_error)}


def _price_correlation(observed, optimized):
    observed = np.asarray(observed, dtype=float)
    optimized = np.asarray(optimized, dtype=float)
    if np.std(observed) <= 1e-15 or np.std(optimized) <= 1e-15:
        return float('nan')
    return float(np.corrcoef(observed, optimized)[0, 1])


def _json_safe(value):
    """Recursively convert diagnostics to strict JSON (undefined values become null)."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f'Cannot serialize {type(value).__name__}.')


def build_day_signals(
    consumption_raw,
    price_raw,
    consumption,
    price,
    day,
    *,
    baseline_weeks=DEFAULT_BASELINE_WEEKS,
):
    """Build a real target day and history-only matching-weekday baseline."""
    day = pd.Timestamp(day).normalize()
    baseline_weeks = positive_integer(baseline_weeks, 'baseline_weeks', 3)
    target = _complete_day_values(consumption, day, 'normalized consumption')
    observed_price = _complete_day_values(price, day, 'normalized DK1 price')
    raw_target = _complete_day_values(consumption_raw, day, 'raw consumption')
    raw_price = _complete_day_values(price_raw, day, 'raw DK1 price')
    if np.min(target) < -1e-12 or np.max(target) > 1.0 + 1e-12:
        raise ValueError('Selected target day lies outside the Jan-Oct normalization range.')
    if np.min(observed_price) < -1e-12 or np.max(observed_price) > 1.0 + 1e-12:
        raise ValueError('Selected price day lies outside the Jan-Oct normalization range.')
    history_days = tuple((day - pd.Timedelta(weeks=week) for week in range(1, baseline_weeks + 1)))
    history = np.vstack(
        [
            _complete_day_values(consumption, history_day, 'historical normalized consumption')
             for history_day in history_days
        ],
    )
    ordered = np.sort(history, axis=0)
    baseline = np.mean(ordered[1:-1], axis=0)
    if not np.all(np.isfinite(baseline)) or np.min(baseline) < -1e-12 or np.max(baseline) > 1.0 + 1e-12:
        raise ValueError('History-derived baseline is invalid after normalization.')
    return DaySignals(
        day=day,
        target=target.copy(),
        baseline=baseline.copy(),
        observed_price=observed_price.copy(),
        raw_target=raw_target.copy(),
        raw_price=raw_price.copy(),
        baseline_days=history_days,
    )


def select_representative_days(
    consumption_raw,
    price_raw,
    consumption,
    price,
    *,
    n_days=DEFAULT_DAYS,
    baseline_weeks=DEFAULT_BASELINE_WEEKS,
):
    """Select complete low/typical/high-consumption days reproducibly."""
    n_days = positive_integer(n_days, 'n_days', 1)
    baseline_weeks = positive_integer(baseline_weeks, 'baseline_weeks', 3)
    earliest = SEASON_START + pd.Timedelta(weeks=baseline_weeks)
    latest = SEASON_END - pd.Timedelta(days=1)
    candidates = []
    for day in pd.date_range(earliest, latest, freq='D'):
        try:
            signals = build_day_signals(
                consumption_raw,
                price_raw,
                consumption,
                price,
                day,
                baseline_weeks=baseline_weeks,
            )
        except ValueError:
            continue
        candidates.append({'day': day, 'mean_consumption': float(np.mean(signals.target))})
    if len(candidates) < n_days:
        raise ValueError(
            (
                f'Only {len(candidates)}'
                f' complete real-data days satisfy the history requirements; requested {n_days}.'
            ),
        )
    candidates = sorted(candidates, key=lambda row: row['mean_consumption'])
    means = np.array([row['mean_consumption'] for row in candidates], dtype=float)
    quantiles = _representative_quantiles(n_days)
    selected = []
    used = set()
    for quantile in quantiles:
        target_mean = float(np.quantile(means, quantile))
        order = np.argsort(np.abs(means - target_mean))
        for index in order:
            day = candidates[int(index)]['day']
            if day not in used:
                selected.append(day)
                used.add(day)
                break
    return selected


def prepare_historical_curve_training(measurements, baseline, prices, *, contract,
                                      parameters, sigma_y, initial_mass_builder, origin):
    """Build the entire TRAIN likelihood from an explicitly earlier prior time.

    Every declared TRAIN point is scored (unless explicitly missing). Baseline
    and prices must also cover the caller-declared pre-observation interval;
    the adapter never invents an input value or discards the first TRAIN point.
    """
    from src.control.bm_1 import BM1Parameters
    from src.method.bm_1_response import CurveTrainingRecord
    from src.params.bm_1 import BM1ObservationSeries
    from src.params.bm_1_history import BM1HistoricalCurveContract
    if not isinstance(contract, BM1HistoricalCurveContract):
        raise TypeError('Declare the historical curve contract before preparing data.')
    if not isinstance(parameters, BM1Parameters):
        raise TypeError('Physical parameters must be BM1Parameters in day units.')
    if not isinstance(measurements, pd.DataFrame) or not isinstance(measurements.index, pd.DatetimeIndex):
        raise TypeError('Measurements must be an explicitly selected timestamped DataFrame.')
    if tuple(measurements.columns) != contract.measurement_columns:
        raise ValueError('Measurement columns/order differ from the frozen population.')
    split = contract.split
    expected = pd.date_range(split.train_start, split.selection_start, freq='h', inclusive='left')
    if len(expected) < 1 or not measurements.index.equals(expected):
        raise ValueError('Supply the whole hourly TRAIN index, with explicit NaN for missing values and no other roles.')
    origin = pd.Timestamp(origin)
    if pd.isna(origin) or origin.tzinfo is None or origin >= split.train_start:
        raise ValueError('Declare a timezone-aware prior origin strictly before the first TRAIN observation.')
    origin = origin.tz_convert(expected.tz)
    lead_hours = (split.train_start.tz_convert('UTC')-origin.tz_convert('UTC')).total_seconds()/3600.
    if lead_hours != int(lead_hours):
        raise ValueError('Prior origin must align with the hourly input grid.')
    input_times = pd.date_range(origin, split.selection_start, freq='h', inclusive='left')
    seconds = (input_times.tz_convert('UTC')-origin.tz_convert('UTC')).total_seconds().to_numpy()
    end = (split.selection_start.tz_convert('UTC')-origin.tz_convert('UTC')).total_seconds()
    if end-seconds[-1] != 3600.:
        raise ValueError('TRAIN boundaries must align with the hourly grid.')
    selected = pd.concat([_aware_series(measurements[c], c) for c in contract.measurement_columns], axis=1)
    aggregate = selected.mean(axis=1).where(selected.notna().all(axis=1))
    baseline = _aware_series(baseline, 'baseline')
    prices = _aware_series(prices, 'prices')
    if not baseline.index.equals(input_times) or not prices.index.equals(input_times):
        raise ValueError('Baseline and prices must explicitly cover the full prior-origin-to-TRAIN-end index.')
    b = baseline.to_numpy()/contract.power_scale
    if np.any(~np.isfinite(b)) or np.any(b < 0.) or np.any(b > 1.):
        raise ValueError('Complete baseline must remain within the frozen power scale.')
    u = contract.normalize_prices(prices.to_numpy())
    edges = np.r_[seconds, end]/86400.
    B = input_builder.price_intervals((edges, b))
    U = input_builder.price_intervals((edges, u))
    observed = aggregate.notna().to_numpy()
    observation_times = (expected.tz_convert('UTC')-origin.tz_convert('UTC')).total_seconds().to_numpy()/86400.
    observations = BM1ObservationSeries(observation_times, aggregate.to_numpy()/contract.power_scale,
        'day', observed, 'point', 'right')
    record = CurveTrainingRecord(contract.population_id, 'TRAIN', B, (0., float(edges[-1])),
        (U, parameters.state_response, parameters.price_response, parameters.C, parameters.k, parameters.Xi),
        parameters.sigma_x, sigma_y, initial_mass_builder, observations, edges[1:-1])
    metadata = dict(contract=contract.to_record(), origin=origin.isoformat(),
        timestamps=[t.isoformat() for t in expected], initial_observation_excluded=False,
        midnight_resets=0, training_rows=len(expected), scored_observations=int(observed.sum()),
        missing_aggregate_rows=int(aggregate.isna().sum()),
        terminal_posterior_time=expected[-1].isoformat(),
        parameter_time_unit='day', baseline_availability='caller-declared offline TRAIN input plus explicit pre-observation inputs; no causal baseline claim')
    return record, metadata
