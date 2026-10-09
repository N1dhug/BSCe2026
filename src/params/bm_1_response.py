"""G's immutable two-piece price response; no fitting at import."""

from dataclasses import dataclass

import numpy as np

from src.validation import estimation as checks


@dataclass(frozen=True)
class BM1PiecewiseResponse:
    """Two straight pieces through (0,1), (0.5,a) and (1,-1).

    The default a=0 gives the linear state response. Fitted a describes price
    response. Geometry is continuous and monotone without clipping.
    """

    knots: object
    values: object

    def __post_init__(self):
        knots = checks.increasing_grid(self.knots, 'response knots').copy()
        values = checks.finite_vector(self.values, 'response values').copy()
        if not np.array_equal(knots,[0.,.5,1.]) or values.shape != knots.shape:
            raise ValueError('G response knots must be exactly [0,0.5,1].')
        if values[0] != 1. or values[-1] != -1. or np.any(np.diff(values) > 0.):
            raise ValueError('Response values must decrease from exactly 1 to -1.')
        knots.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, 'knots', knots)
        object.__setattr__(self, 'values', values)

    @classmethod
    def from_inner_values(cls, inner_values, *, pieces):
        """The two-piece price response with one fitted midpoint."""
        pieces = checks.positive_integer(pieces, 'pieces', minimum=2)
        if pieces != 2:
            raise ValueError('G declares two price-response pieces.')
        inner = checks.finite_vector(inner_values, 'inner values')
        if inner.size != pieces - 1:
            raise ValueError('One shared value is needed at each inner knot.')
        return cls(np.linspace(0., 1., pieces + 1), np.r_[1., inner, -1.])

    @classmethod
    def linear(cls, *, pieces=2):
        pieces = checks.positive_integer(pieces, 'pieces', minimum=2)
        if pieces != 2:
            raise ValueError('G declares two response pieces.')
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
