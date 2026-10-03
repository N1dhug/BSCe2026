"""H1's immutable continuous monotone response family; no fitting at import."""

from dataclasses import dataclass

import numpy as np

from src.validation import estimation as checks


@dataclass(frozen=True)
class BM1PiecewiseResponse:
    """Shared f(x) or g(u), selected by argument value rather than date.

    Endpoints are fixed at 1 and -1; ordered inner values give continuity and
    non-increasing response without clipping or projection. The same object
    can be passed to every day's BM1Parameters. Curve fitting, population,
    normalization and blind split metadata belong to the eventual H study.
    """

    knots: object
    values: object

    def __post_init__(self):
        knots = checks.increasing_grid(self.knots, 'response knots').copy()
        values = checks.finite_vector(self.values, 'response values').copy()
        if knots[0] != 0. or knots[-1] != 1. or values.shape != knots.shape:
            raise ValueError('Matching response knots must span exactly [0,1].')
        if values[0] != 1. or values[-1] != -1. or np.any(np.diff(values) > 0.):
            raise ValueError('Response values must decrease from exactly 1 to -1.')
        knots.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, 'knots', knots)
        object.__setattr__(self, 'values', values)

    @classmethod
    def from_inner_values(cls, inner_values, *, pieces):
        """The declared two-/four-piece families, with all pieces coupled."""
        pieces = checks.positive_integer(pieces, 'pieces', minimum=2)
        if pieces not in (2, 4):
            raise ValueError('H1 declares two or four pieces.')
        inner = checks.finite_vector(inner_values, 'inner values')
        if inner.size != pieces - 1:
            raise ValueError('One shared value is needed at each inner knot.')
        return cls(np.linspace(0., 1., pieces + 1), np.r_[1., inner, -1.])

    @classmethod
    def linear(cls, *, pieces=4):
        pieces = checks.positive_integer(pieces, 'pieces', minimum=2)
        if pieces not in (2, 4):
            raise ValueError('H1 declares two or four pieces.')
        knots = np.linspace(0., 1., pieces + 1)
        return cls(knots, 1. - 2. * knots)

    def __call__(self, argument):
        if np.iscomplexobj(argument):
            raise ValueError('Response argument must be real.')
        argument = np.asarray(argument, dtype=float)
        if (np.any(~np.isfinite(argument)) or np.any(argument < 0.)
                or np.any(argument > 1.)):
            raise ValueError('Response argument must lie in [0,1]; no extrapolation/clipping.')
        return np.interp(argument.reshape(-1), self.knots, self.values).reshape(argument.shape)

    def to_record(self):
        """Curve geometry only; this does not claim a trained/validated model."""
        return dict(schema='bm1-piecewise-response-v1', knots=self.knots.tolist(),
                    values=self.values.tolist(), interpolation='linear',
                    monotonicity='non-increasing', argument_domain=[0., 1.])

    @classmethod
    def from_record(cls, record):
        if (record.get('schema') != 'bm1-piecewise-response-v1'
                or record.get('interpolation') != 'linear'
                or record.get('monotonicity') != 'non-increasing'
                or record.get('argument_domain') != [0., 1.]):
            raise ValueError('Unsupported response description.')
        return cls(record['knots'], record['values'])
