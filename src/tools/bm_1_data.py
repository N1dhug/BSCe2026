"""Historical BM1 data loading and signal preparation; no fitting or optimization."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools.import_data.import_input import input_data
from src.tools.import_data.import_prices import spot_price_data
from src.tools.input_generator.input_builder import input_builder


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
    n_days = int(n_days)
    if n_days < 1:
        raise ValueError('n_days must be positive.')
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
    baseline_weeks = int(baseline_weeks)
    if baseline_weeks < 3:
        raise ValueError('baseline_weeks must be at least 3 for trimmed averaging.')
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
    earliest = SEASON_START + pd.Timedelta(weeks=int(baseline_weeks))
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
    if len(candidates) < int(n_days):
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
