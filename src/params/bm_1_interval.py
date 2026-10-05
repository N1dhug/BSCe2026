"""Explicit interval averages in normalized power; no implicit unit conversion."""

from dataclasses import dataclass

import numpy as np

from src.validation import estimation as checks


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
