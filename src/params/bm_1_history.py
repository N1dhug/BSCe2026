"""Frozen historical H contracts; declaration checks never establish metadata.

Chronology alone cannot make previously inspected observations a blind holdout.
The caller must record metadata evidence and an exposure audit separately.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.validation import estimation as checks


def _timestamp(value, name):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f'{name} needs a valid timezone-aware timestamp.')
    return stamp


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Declare {name}.')
    return value


@dataclass(frozen=True)
class BM1ChronologicalSplit:
    """Half-open TRAIN / MODEL_SELECTION / FINAL_HOLDOUT, on UTC chronology."""
    train_start: object
    selection_start: object
    holdout_start: object
    holdout_end: object
    exposure_intervals: tuple = ()
    holdout_audit_reference: str | None = None

    def __post_init__(self):
        stamps = tuple(_timestamp(getattr(self, n), n) for n in
                       ('train_start', 'selection_start', 'holdout_start', 'holdout_end'))
        if not all(a < b for a, b in zip(stamps[:-1], stamps[1:])):
            raise ValueError('TRAIN, MODEL_SELECTION and FINAL_HOLDOUT must be chronological and nonempty.')
        for name, stamp in zip(('train_start', 'selection_start', 'holdout_start', 'holdout_end'), stamps):
            object.__setattr__(self, name, stamp)
        intervals = []
        for start, end, reason in self.exposure_intervals:
            start, end = _timestamp(start, 'exposure start'), _timestamp(end, 'exposure end')
            if start >= end:
                raise ValueError('Exposure intervals must be nonempty and half-open.')
            intervals.append((start, end, _nonempty(reason, 'exposure reason')))
        object.__setattr__(self, 'exposure_intervals', tuple(intervals))
        if self.holdout_audit_reference is not None:
            _nonempty(self.holdout_audit_reference, 'holdout exposure audit reference')

    def role_mask(self, times, role):
        windows = {'TRAIN': (self.train_start, self.selection_start),
                   'MODEL_SELECTION': (self.selection_start, self.holdout_start),
                   'FINAL_HOLDOUT': (self.holdout_start, self.holdout_end)}
        if role not in windows:
            raise ValueError('Unknown chronological role.')
        index = pd.DatetimeIndex(times)
        if index.tz is None or index.hasnans:
            raise ValueError('Role assignment requires timezone-aware timestamps.')
        left, right = windows[role]
        return (index >= left) & (index < right)

    def require_blind_holdout(self):
        if self.holdout_audit_reference is None:
            raise ValueError('Blind holdout requires a recorded exposure audit, not just a date split.')
        if any(a < self.holdout_end and b > self.holdout_start for a, b, _ in self.exposure_intervals):
            raise ValueError('Previously exposed data overlap FINAL_HOLDOUT.')

    def to_record(self):
        return dict(schema='bm1-chronological-split-v1',
            **{n: getattr(self, n).isoformat() for n in
               ('train_start', 'selection_start', 'holdout_start', 'holdout_end')},
            exposure_intervals=[[a.isoformat(), b.isoformat(), reason] for a, b, reason in self.exposure_intervals],
            holdout_audit_reference=self.holdout_audit_reference)


@dataclass(frozen=True)
class BM1HistoricalCurveContract:
    """One population and fixed scales for all training/validation days.

    Spot prices may be declared as a proxy, but do not establish consumer price
    response. Interval data are rejected until a distinct likelihood is supplied.
    """
    population_id: str
    measurement_columns: tuple
    split: BM1ChronologicalSplit
    power_unit: str
    power_scale: float
    price_low: float
    price_high: float
    price_unit: str
    price_role: str
    observation_kind: str
    metadata_reference: str
    baseline_reference: str
    scale_fitted_before: object = None

    def __post_init__(self):
        for name in ('population_id', 'price_unit', 'metadata_reference', 'baseline_reference'):
            _nonempty(getattr(self, name), name)
        if isinstance(self.measurement_columns, str):
            raise TypeError('measurement_columns must be a sequence of column names.')
        columns = tuple(self.measurement_columns)
        if not columns or len(set(columns)) != len(columns) or any(not isinstance(c, str) or not c.strip() for c in columns):
            raise ValueError('Declare one fixed, nonempty set of unique measurement columns.')
        object.__setattr__(self, 'measurement_columns', columns)
        if not isinstance(self.split, BM1ChronologicalSplit):
            raise TypeError('An explicit chronological split is required.')
        if self.power_unit not in ('W', 'kW', 'MW'):
            raise ValueError('Declare power units; interval energy is not a point-power observation.')
        if self.observation_kind != 'point':
            raise NotImplementedError('Historical curve fitting requires established point observations; interval likelihood is separate.')
        if self.price_role not in ('consumer_price', 'spot_proxy'):
            raise ValueError('Declare consumer_price or spot_proxy; do not infer a tariff from spot prices.')
        power = checks.finite_scalar(self.power_scale, 'power_scale', positive=True)
        low = checks.finite_scalar(self.price_low, 'price_low')
        high = checks.finite_scalar(self.price_high, 'price_high')
        if low >= high:
            raise ValueError('Frozen price bounds must strictly increase.')
        for name, value in (('power_scale', power), ('price_low', low), ('price_high', high)):
            object.__setattr__(self, name, value)
        if self.scale_fitted_before is not None:
            stamp = _timestamp(self.scale_fitted_before, 'scale_fitted_before')
            if stamp > self.split.selection_start:
                raise ValueError('Scale estimation may use TRAIN only, never MODEL_SELECTION or FINAL_HOLDOUT.')
            object.__setattr__(self, 'scale_fitted_before', stamp)

    def normalize_prices(self, values):
        prices = checks.finite_vector(values, 'prices')
        normalized = (prices-self.price_low)/(self.price_high-self.price_low)
        if np.any(normalized < 0.) or np.any(normalized > 1.):
            raise ValueError('Prices exceed frozen bounds; no clipping or holdout-driven rescaling.')
        return normalized

    def to_record(self):
        return dict(schema='bm1-historical-curve-contract-v1', population_id=self.population_id,
            measurement_columns=list(self.measurement_columns), split=self.split.to_record(),
            power_unit=self.power_unit, power_scale=self.power_scale,
            price_low=self.price_low, price_high=self.price_high, price_unit=self.price_unit,
            price_role=self.price_role, observation_kind=self.observation_kind,
            metadata_reference=self.metadata_reference, baseline_reference=self.baseline_reference,
            scale_fitted_before=None if self.scale_fitted_before is None else self.scale_fitted_before.isoformat(),
            aggregation='fixed-population mean; any missing meter makes aggregate missing',
            interpretation='Declared metadata, not independently established scientific/causal acceptance.')
