"""Monotone linear, PCHIP, Bernstein and beta-CDF BM response geometry."""

from dataclasses import dataclass
from functools import cached_property
from bisect import bisect_right
from math import isfinite

import numpy as np
from scipy.interpolate import BPoly, PPoly
from scipy.special import betainc, betaln, xlogy

from src.validation import estimation as checks
from src.tools.input_generator.input_builder import input_builder


RESPONSE_FAMILIES = ('piecewise-linear', 'pchip', 'bernstein', 'beta-cdf')
BETA_SHAPE_MAX = 10.


def _argument(argument):
    if np.iscomplexobj(argument):
        raise ValueError('Response argument must be real.')
    argument = np.asarray(argument, dtype=float)
    if np.any(~np.isfinite(argument)) or np.any((argument < 0.) | (argument > 1.)):
        raise ValueError('Response argument must lie in [0,1]; no extrapolation/clipping.')
    return argument


def _checked_scalar(evaluate):
    def checked(x):
        if not isfinite(x) or not 0 <= x <= 1:
            raise ValueError('RK4 stage left the response domain; no extrapolation.')
        if x == 0: return 1.
        if x == 1: return -1.
        return evaluate(x)
    return checked


@dataclass(frozen=True)
class BM1PiecewiseResponse:
    """Continuous non-increasing response from (0,1) to (1,-1)."""

    knots: object
    values: object
    interpolation: str = 'linear'

    def __post_init__(self):
        knots = checks.increasing_grid(self.knots, 'response knots').copy()
        values = checks.finite_vector(self.values, 'response values').copy()
        if len(knots) < 3 or knots[0] != 0. or knots[-1] != 1. or values.shape != knots.shape:
            raise ValueError('Response knots must span [0,1] with matching values and at least two pieces.')
        if values[0] != 1. or values[-1] != -1. or np.any(np.diff(values) > 0.):
            raise ValueError('Response values must decrease from exactly 1 to -1.')
        if self.interpolation not in ('linear', 'pchip', 'bernstein'):
            raise ValueError('Unknown response interpolation.')
        if self.interpolation == 'bernstein' and not np.allclose(knots, np.linspace(0, 1, len(knots)), rtol=0, atol=1e-14):
            raise ValueError('Bernstein coefficients require an equally spaced index grid.')
        knots.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, 'knots', knots)
        object.__setattr__(self, 'values', values)

    @classmethod
    def from_inner_values(cls, inner_values, *, pieces, interpolation='linear'):
        """Construct a response from ordered values at equally spaced inner knots."""
        pieces = checks.positive_integer(pieces, 'pieces', minimum=2)
        inner = checks.finite_vector(inner_values, 'inner values')
        if inner.size != pieces - 1:
            raise ValueError('One shared value is needed at each inner knot.')
        return cls(np.linspace(0., 1., pieces + 1), np.r_[1., inner, -1.], interpolation)

    @classmethod
    def from_drop_fractions(cls, fractions, *, interpolation='linear'):
        """Use one fraction of the remaining drop per inner knot; each is in [0,1]."""
        fractions = checks.finite_vector(fractions, 'drop fractions')
        if fractions.size < 1 or np.any((fractions < 0) | (fractions > 1)):
            raise ValueError('Use at least one drop fraction in [0,1].')
        inner = -1. + 2. * np.cumprod(1. - fractions)
        return cls.from_inner_values(inner, pieces=fractions.size+1, interpolation=interpolation)

    @classmethod
    def linear(cls, *, pieces=2):
        pieces = checks.positive_integer(pieces, 'pieces', minimum=2)
        knots = np.linspace(0., 1., pieces + 1)
        return cls(knots, 1. - 2. * knots)

    def __call__(self, argument):
        argument = _argument(argument)
        if self.interpolation == 'linear':
            return np.interp(argument.reshape(-1), self.knots, self.values).reshape(argument.shape)
        values = self.polynomial(argument)
        return np.where(argument == 0, 1., np.where(argument == 1, -1., values))

    @property
    def is_linear(self):
        return np.array_equal(self.values, 1-2*self.knots)

    @cached_property
    def polynomial(self):
        if self.interpolation == 'pchip':
            # Reuse the existing monotone input-function constructor.
            curve = input_builder.md(self.knots, np.r_[0., -np.diff(self.values)])
            curve.extrapolate = False
            return curve
        if self.interpolation == 'bernstein':
            return BPoly(self.values[:, None], [0., 1.], extrapolate=False)
        raise ValueError('Linear responses do not need a polynomial object.')

    @cached_property
    def max_abs_slope(self):
        if self.interpolation == 'bernstein':
            return float((len(self.values)-1)*np.max(-np.diff(self.values)))
        if self.interpolation == 'pchip':
            derivative = self.polynomial.derivative()
            stationary = derivative.derivative().roots(extrapolate=False)
            stationary = stationary[np.isfinite(stationary) & (stationary >= 0) & (stationary <= 1)]
            return float(np.max(np.abs(derivative(np.r_[self.knots, stationary]))))
        return float(np.max(np.abs(np.diff(self.values)/np.diff(self.knots))))

    def scalar_evaluator(self):
        if self.is_linear:
            return _checked_scalar(lambda x: 1-2*x)
        if self.interpolation == 'linear':
            knots, values = tuple(self.knots), tuple(self.values)
            slopes = tuple(np.diff(self.values)/np.diff(self.knots))
            def evaluate(x):
                j = min(bisect_right(knots, x)-1, len(slopes)-1)
                return values[j]+slopes[j]*(x-knots[j])
        else:
            polynomial = self.polynomial
            if self.interpolation == 'bernstein':
                polynomial = PPoly.from_bernstein_basis(polynomial)
            knots = tuple(polynomial.x)
            coefficients = tuple(tuple(column) for column in polynomial.c.T)
            def evaluate(x):
                j = min(bisect_right(knots, x)-1, len(coefficients)-1)
                z, value = x-knots[j], 0.
                for coefficient in coefficients[j]:
                    value = value*z+coefficient
                return value
        return _checked_scalar(evaluate)

    def to_record(self):
        """Curve geometry only; this does not claim a trained/validated model."""
        schema = 'bm1-piecewise-response-v1' if self.interpolation == 'linear' else 'bm1-response-v2'
        return dict(schema=schema, knots=self.knots.tolist(),
                    values=self.values.tolist(), interpolation=self.interpolation,
                    monotonicity='non-increasing', argument_domain=[0., 1.])

    @classmethod
    def from_record(cls, record):
        if (record.get('schema') not in ('bm1-piecewise-response-v1', 'bm1-response-v2')
                or record.get('interpolation') not in ('linear', 'pchip', 'bernstein')
                or record.get('monotonicity') != 'non-increasing'
                or record.get('argument_domain') != [0., 1.]):
            raise ValueError('Unsupported response description.')
        return cls(record['knots'], record['values'], record['interpolation'])


@dataclass(frozen=True)
class BM1BetaResponse:
    """1-2*BetaCDF; bounded positive shapes keep endpoint derivatives finite."""
    a: float
    b: float

    def __post_init__(self):
        if not all(np.isfinite(v) and 1 <= v <= BETA_SHAPE_MAX for v in (self.a, self.b)):
            raise ValueError(f'Beta response shapes must lie in [1,{BETA_SHAPE_MAX}].')

    @property
    def is_linear(self):
        return self.a == self.b == 1.

    def __call__(self, argument):
        return 1.-2.*betainc(self.a, self.b, _argument(argument))

    @cached_property
    def max_abs_slope(self):
        if self.is_linear: return 2.
        mode = (self.a-1)/(self.a+self.b-2)
        return float(2*np.exp(xlogy(self.a-1, mode)+xlogy(self.b-1, 1-mode)-betaln(self.a,self.b)))

    def scalar_evaluator(self):
        return _checked_scalar(lambda x: 1.-2.*betainc(self.a, self.b, x))

    def to_record(self):
        return dict(schema='bm1-response-v2', interpolation='beta-cdf', a=float(self.a), b=float(self.b),
                    shape_bounds=[1., BETA_SHAPE_MAX], monotonicity='non-increasing', argument_domain=[0.,1.])


def response_from_record(record):
    if record.get('interpolation') != 'beta-cdf':
        return BM1PiecewiseResponse.from_record(record)
    if (record.get('schema') != 'bm1-response-v2' or record.get('argument_domain') != [0.,1.]
            or record.get('monotonicity') != 'non-increasing'):
        raise ValueError('Unsupported response description.')
    return BM1BetaResponse(record['a'], record['b'])

