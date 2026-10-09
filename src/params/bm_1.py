"""Owned BM observations: explicit point samples or complete interval averages.

Units, missing masks and observation availability belong to these contracts.
Likelihood updates and numerical remapping live in method.bm_1_observation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from src.validation import estimation as checks

@dataclass(frozen=True)
class BM1ObservationSeries:
    """Owned point observations; times use the explicitly declared time unit.

    price_side selects the price's left or right limit at an input jump;
    baseline is evaluated at the observation time. Filters honor this contract;
    the container samples no signals. Times must share the model's unit, with
    no implicit conversion. Interval data require a different observation model.
    Missing measurements require an explicit False mask. No truth or control
    target is stored, and no unobserved value is inferred from a future sample.
    """
    times: ArrayLike
    values: ArrayLike
    time_unit: str
    observed: ArrayLike | None = None
    observation_kind: str = 'point'
    price_side: str = 'right'

    def __post_init__(self):
        if self.observation_kind != 'point':
            raise NotImplementedError('Only point observations are prepared; interval data need an explicit model.')
        if self.time_unit not in ('second', 'hour', 'day'):
            raise ValueError('time_unit must be second, hour or day; no implicit conversion is performed.')
        if self.price_side not in ('left', 'right'):
            raise ValueError('price_side must be left or right.')
        times, values, observed = checks.point_observations(self.times, self.values, self.observed)
        for array in (times, values, observed):
            array.setflags(write=False)
        object.__setattr__(self, 'times', times)
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'observed', observed)

    def prefix(self, cutoff: float) -> 'BM1ObservationSeries':
        """Copy observations at or before cutoff; never include future values."""
        cutoff = checks.finite_scalar(cutoff, 'cutoff')
        keep = self.times <= cutoff
        if not np.any(keep):
            raise ValueError('The requested prefix contains no observation times.')
        return BM1ObservationSeries(
            self.times[keep], self.values[keep], self.time_unit, self.observed[keep],
            self.observation_kind, self.price_side,
        )

@dataclass(frozen=True)
class BM1IntervalObservationSeries:
    """Y_i = (end_i-start_i)**-1 integral D(t,X_t) dt + epsilon_i.

    All times use ``time_unit``. Values and sigma_y use normalized average
    power. An energy adapter must explicitly divide by elapsed hours and the
    frozen power scale before constructing this object. Availability is at
    the interval END; a partial interval never enters a prefix or posterior.
    """

    starts: object
    ends: object
    values: object
    time_unit: str
    observed: object = None
    observation_kind: str = 'interval_average'

    def __post_init__(self):
        if self.time_unit not in ('second', 'hour', 'day'):
            raise ValueError('Declare second, hour or day; no implicit conversion.')
        if self.observation_kind != 'interval_average':
            raise ValueError('This series represents normalized interval averages.')
        ends, values, observed = checks.point_observations(self.ends, self.values, self.observed)
        starts = checks.increasing_grid(self.starts, 'interval starts', minimum_size=1)
        if (starts.shape != ends.shape or np.any(starts >= ends)
                or np.any(starts[1:] < ends[:-1])):
            raise ValueError('Intervals must have positive lengths and cannot overlap.')
        for name, value in (('starts', starts), ('ends', ends), ('values', values), ('observed', observed)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def times(self):
        """Observation availability times, for shared fitting/checkpoint code."""
        return self.ends

    def prefix(self, cutoff):
        cutoff = checks.finite_scalar(cutoff, 'cutoff')
        keep = self.ends <= cutoff
        if not np.any(keep):
            raise ValueError('The requested prefix contains no complete intervals.')
        return type(self)(self.starts[keep], self.ends[keep], self.values[keep],
                          self.time_unit, self.observed[keep])
