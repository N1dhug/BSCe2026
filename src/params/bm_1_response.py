"""Immutable monotone piecewise-linear BM responses; no fitting at import."""

from dataclasses import dataclass

import numpy as np

from src.validation import estimation as checks


@dataclass(frozen=True)
class BM1PiecewiseResponse:
    """Continuous non-increasing response from (0,1) to (1,-1)."""

    knots: object
    values: object

    def __post_init__(self):
        knots = checks.increasing_grid(self.knots, 'response knots').copy()
        values = checks.finite_vector(self.values, 'response values').copy()
        if len(knots) < 3 or knots[0] != 0. or knots[-1] != 1. or values.shape != knots.shape:
            raise ValueError('Response knots must span [0,1] with matching values and at least two pieces.')
        if values[0] != 1. or values[-1] != -1. or np.any(np.diff(values) > 0.):
            raise ValueError('Response values must decrease from exactly 1 to -1.')
        knots.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, 'knots', knots)
        object.__setattr__(self, 'values', values)

    @classmethod
    def from_inner_values(cls, inner_values, *, pieces):
        """Construct a response from ordered values at equally spaced inner knots."""
        pieces = checks.positive_integer(pieces, 'pieces', minimum=2)
        inner = checks.finite_vector(inner_values, 'inner values')
        if inner.size != pieces - 1:
            raise ValueError('One shared value is needed at each inner knot.')
        return cls(np.linspace(0., 1., pieces + 1), np.r_[1., inner, -1.])

    @classmethod
    def from_drop_fractions(cls, fractions):
        """Use one fraction of the remaining drop per inner knot; each is in [0,1]."""
        fractions = checks.finite_vector(fractions, 'drop fractions')
        if fractions.size < 1 or np.any((fractions < 0) | (fractions > 1)):
            raise ValueError('Use at least one drop fraction in [0,1].')
        inner = -1. + 2. * np.cumprod(1. - fractions)
        return cls.from_inner_values(inner, pieces=fractions.size+1)

    @classmethod
    def linear(cls, *, pieces=2):
        pieces = checks.positive_integer(pieces, 'pieces', minimum=2)
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
