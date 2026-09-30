"""Synthetic BM1 estimation cases and truth generation for F4-B.

This module is an experiment/validation layer.  It reuses the existing BM1
Euler--Maruyama solver and input builders; it does not implement a second SDE
solver or expose synthetic truth through the estimator input contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.model.bm_1 import bm_1
from src.params.bm_1 import BM1ObservationSeries
from src.tools.input_generator.input_builder import input_builder
from src.validation.bm_1 import cases


CORE_PRICE_EDGES = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=float)
CORE_PRICE_VALUES = np.array([0.65, 0.35, 0.65], dtype=float)
CORE_OBSERVATION_TIMES = np.arange(1, 25, dtype=float) / 24.0
INITIAL_SEED_ROOT = 4101
BROWNIAN_SEED_ROOT = 4102
MEASUREMENT_SEED_ROOT = 4103


def _readonly_vector(values: ArrayLike, name: str, *, minimum_size: int = 1) -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise ValueError(f'{name} must be real-valued.')
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < minimum_size or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be a finite one-dimensional vector.')
    array = array.copy()
    array.setflags(write=False)
    return array


def _finite_scalar(value, name, *, positive=False, nonnegative=False):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind in ('b', 'c'):
        raise TypeError(f'{name} must be a real scalar.')
    value = float(array)
    if not np.isfinite(value):
        raise ValueError(f'{name} must be finite.')
    if positive and value <= 0.0:
        raise ValueError(f'{name} must be strictly positive.')
    if nonnegative and value < 0.0:
        raise ValueError(f'{name} must be non-negative.')
    return value


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be a non-negative integer.')
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f'{name} must be a non-negative integer.') from exc
    if value < 0:
        raise ValueError(f'{name} must be non-negative.')
    return int(value)


def _seed(root, sequence_id):
    """Derive one stable integer seed from a stream root and sequence id."""
    return int(np.random.SeedSequence([root, sequence_id]).generate_state(1, dtype=np.uint32)[0])


@dataclass(frozen=True)
class BM1SyntheticCase:
    """Declared synthetic BM1 experiment; no control target belongs to the API."""

    horizon: tuple[float, float] = (0.0, 1.0)
    baseline: float = 0.5
    price_edges: ArrayLike = field(default_factory=lambda: CORE_PRICE_EDGES.copy())
    price_values: ArrayLike = field(default_factory=lambda: CORE_PRICE_VALUES.copy())
    observation_times: ArrayLike = field(default_factory=lambda: CORE_OBSERVATION_TIMES.copy())
    C: float = 1.0
    k: float = 2.0
    Xi: float = 1.0
    sigma_x: float = 0.5
    sigma_y: float = 0.03
    initial_alpha: float = 20.0
    initial_beta: float = 20.0
    truth_max_step: float = 1.0 / 1152.0
    truth_fine_max_step: float = 1.0 / 2304.0
    initial_seed_root: int = INITIAL_SEED_ROOT
    brownian_seed_root: int = BROWNIAN_SEED_ROOT
    measurement_seed_root: int = MEASUREMENT_SEED_ROOT

    def __post_init__(self):
        horizon = _readonly_vector(self.horizon, 'horizon', minimum_size=2)
        if horizon.size != 2 or horizon[1] <= horizon[0]:
            raise ValueError('horizon must contain exactly two increasing values.')
        baseline = _finite_scalar(self.baseline, 'baseline')
        if not 0.0 <= baseline <= 1.0:
            raise ValueError('baseline must lie in [0, 1].')
        price_edges = _readonly_vector(self.price_edges, 'price_edges', minimum_size=2)
        prices = _readonly_vector(self.price_values, 'price_values')
        if price_edges.size != prices.size + 1:
            raise ValueError('price_edges must contain one more value than price_values.')
        if price_edges[0] != horizon[0] or price_edges[-1] != horizon[1] or np.any(np.diff(price_edges) <= 0.0):
            raise ValueError('price_edges must strictly increase across the complete horizon.')
        if np.any(prices < 0.0) or np.any(prices > 1.0):
            raise ValueError('price_values must lie in [0, 1].')
        observation_times = _readonly_vector(self.observation_times, 'observation_times')
        if (np.any(np.diff(observation_times) <= 0.0)
                or observation_times[0] <= horizon[0]
                or observation_times[-1] > horizon[1]):
            raise ValueError('observation_times must strictly increase in (start, end].')
        C = _finite_scalar(self.C, 'C', positive=True)
        k = _finite_scalar(self.k, 'k', positive=True)
        Xi = _finite_scalar(self.Xi, 'Xi')
        if not 0.0 <= Xi <= 1.0:
            raise ValueError('Xi must lie in [0, 1].')
        sigma_x = _finite_scalar(self.sigma_x, 'sigma_x', nonnegative=True)
        sigma_y = _finite_scalar(self.sigma_y, 'sigma_y', positive=True)
        initial_alpha = _finite_scalar(self.initial_alpha, 'initial_alpha', positive=True)
        initial_beta = _finite_scalar(self.initial_beta, 'initial_beta', positive=True)
        truth_max_step = _finite_scalar(self.truth_max_step, 'truth_max_step', positive=True)
        truth_fine_max_step = _finite_scalar(self.truth_fine_max_step, 'truth_fine_max_step', positive=True)
        if truth_fine_max_step >= truth_max_step:
            raise ValueError('truth_fine_max_step must be smaller than truth_max_step.')
        roots = tuple(_nonnegative_integer(value, name) for value, name in (
            (self.initial_seed_root, 'initial_seed_root'),
            (self.brownian_seed_root, 'brownian_seed_root'),
            (self.measurement_seed_root, 'measurement_seed_root'),
        ))
        if len(set(roots)) != 3:
            raise ValueError('Synthetic RNG stream roots must be distinct.')
        object.__setattr__(self, 'horizon', (float(horizon[0]), float(horizon[1])))
        object.__setattr__(self, 'baseline', baseline)
        object.__setattr__(self, 'price_edges', price_edges)
        object.__setattr__(self, 'price_values', prices)
        object.__setattr__(self, 'observation_times', observation_times)
        object.__setattr__(self, 'C', C)
        object.__setattr__(self, 'k', k)
        object.__setattr__(self, 'Xi', Xi)
        object.__setattr__(self, 'sigma_x', sigma_x)
        object.__setattr__(self, 'sigma_y', sigma_y)
        object.__setattr__(self, 'initial_alpha', initial_alpha)
        object.__setattr__(self, 'initial_beta', initial_beta)
        object.__setattr__(self, 'truth_max_step', truth_max_step)
        object.__setattr__(self, 'truth_fine_max_step', truth_fine_max_step)
        object.__setattr__(self, 'initial_seed_root', roots[0])
        object.__setattr__(self, 'brownian_seed_root', roots[1])
        object.__setattr__(self, 'measurement_seed_root', roots[2])

    @property
    def output_times(self):
        values = np.concatenate(([self.horizon[0]], np.asarray(self.observation_times)))
        values.setflags(write=False)
        return values

    @property
    def breakpoints(self):
        values = np.asarray(self.price_edges[1:-1], dtype=float).copy()
        values.setflags(write=False)
        return values

    def build_model(self):
        """Build BM1 and [U,f,g,C,k,Xi]; A is an inert legacy constructor placeholder."""
        baseline = input_builder.energy((
            np.asarray(self.horizon, dtype=float),
            np.full(2, self.baseline, dtype=float),
        ))
        price = input_builder.price_intervals((self.price_edges, self.price_values))
        parameters = [
            price,
            cases.linear_state_response,
            cases.linear_price_response,
            self.C,
            self.k,
            self.Xi,
        ]
        # bm_1 still stores an F3 target A, but EM state/demand generation never
        # reads it.  Passing B here avoids inventing an estimation target.
        return bm_1(baseline, baseline, self.horizon, self.output_times.size), parameters


@dataclass(frozen=True)
class BM1SyntheticTruth:
    """Synthetic ground truth retained for scoring, never estimator input."""

    times: ArrayLike
    state: ArrayLike
    price: ArrayLike
    baseline: ArrayLike
    demand: ArrayLike
    initial_state: float
    diagnostics: dict

    def __post_init__(self):
        arrays = {}
        for name in ('times', 'state', 'price', 'baseline', 'demand'):
            arrays[name] = _readonly_vector(getattr(self, name), name)
        size = arrays['times'].size
        if any(values.size != size for name, values in arrays.items() if name != 'times'):
            raise ValueError('Synthetic truth arrays must have matching lengths.')
        initial_state = _finite_scalar(self.initial_state, 'initial_state')
        if not 0.0 <= initial_state <= 1.0:
            raise ValueError('initial_state must lie in [0, 1].')
        diagnostics = {}
        for key, value in dict(self.diagnostics).items():
            if isinstance(value, np.ndarray):
                copied = value.copy()
                copied.setflags(write=False)
                diagnostics[key] = copied
            else:
                diagnostics[key] = value
        for name, values in arrays.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, 'initial_state', initial_state)
        object.__setattr__(self, 'diagnostics', diagnostics)

    @property
    def valid(self):
        return int(self.diagnostics.get('total_violations', 0)) == 0


@dataclass(frozen=True)
class BM1SyntheticSequence:
    """Public observations plus separately held synthetic scoring information."""

    observations: BM1ObservationSeries
    truth: BM1SyntheticTruth
    measurement_noise: ArrayLike
    sequence_id: int

    def __post_init__(self):
        if not isinstance(self.observations, BM1ObservationSeries):
            raise TypeError('observations must be a BM1ObservationSeries.')
        if not isinstance(self.truth, BM1SyntheticTruth):
            raise TypeError('truth must be a BM1SyntheticTruth.')
        noise = _readonly_vector(self.measurement_noise, 'measurement_noise')
        if noise.size != self.observations.times.size:
            raise ValueError('measurement_noise must match the observation count.')
        sequence_id = _nonnegative_integer(self.sequence_id, 'sequence_id')
        object.__setattr__(self, 'measurement_noise', noise)
        object.__setattr__(self, 'sequence_id', sequence_id)


@dataclass(frozen=True)
class BM1SyntheticRefinement:
    """Coupled base/fine truth pair using identical X0 and measurement noise."""

    coarse: BM1SyntheticSequence
    fine: BM1SyntheticSequence

    @property
    def maximum_abs_state_difference(self):
        return float(np.max(np.abs(self.coarse.truth.state - self.fine.truth.state)))

    @property
    def maximum_abs_demand_difference(self):
        return float(np.max(np.abs(self.coarse.truth.demand - self.fine.truth.demand)))


def build_f4_b_case(**overrides):
    """Return the locked F4-A core case, with explicit overrides for diagnostics/tests."""
    return BM1SyntheticCase(**overrides)


def _draw_initial_state(case, sequence_id):
    rng = np.random.default_rng(_seed(case.initial_seed_root, sequence_id))
    return float(rng.beta(case.initial_alpha, case.initial_beta))


def _draw_measurement_noise(case, sequence_id):
    rng = np.random.default_rng(_seed(case.measurement_seed_root, sequence_id))
    return rng.normal(0.0, case.sigma_y, size=case.observation_times.size)


def _truth_grid(case, max_step):
    """Build the declared uniform truth grid and exact output indices.

    F4-B's 1/1152 and 1/2304 day grids are mathematically nested.  Building
    them directly avoids output-segment floating-point roundoff from silently
    adding different substeps and destroying Brownian coupling.
    """
    start, end = case.horizon
    ratio = (end - start) / max_step
    n_steps = int(round(ratio))
    if n_steps < 1 or not np.isclose(ratio, n_steps, atol=1e-12, rtol=0.0):
        raise ValueError('truth max step must divide the synthetic horizon exactly.')
    integration_t = np.linspace(start, end, n_steps + 1, dtype=float)

    def indices_for(values, name):
        values = np.asarray(values, dtype=float)
        scaled = (values - start) / (end - start) * n_steps
        indices = np.rint(scaled).astype(int)
        if (np.any(indices < 0) or np.any(indices > n_steps)
                or not np.allclose(integration_t[indices], values, atol=1e-14, rtol=0.0)):
            raise ValueError(f'{name} must lie exactly on the declared uniform truth grid.')
        return indices

    output_indices = indices_for(case.output_times, 'output_times')
    indices_for(case.breakpoints, 'price breakpoints')
    return integration_t, output_indices


def _simulate(case, sequence_id, *, initial_state, measurement_noise, max_step,
              brownian_increments):
    model, parameters = case.build_model()
    integration_t, output_indices = _truth_grid(case, max_step)

    def drift(time, states):
        return model._drift(time, states, parameters)

    def diffusion(time, states):
        del time
        return model.diffusion_coefficient(states, case.sigma_x)

    paths, diagnostics = model.euler_maruyama.solve_paths(
        integration_t,
        [initial_state],
        drift,
        diffusion,
        n_paths=1,
        boundary='none',
        brownian_increments=brownian_increments,
        return_diagnostics=True,
    )
    state = paths[0, output_indices]
    times = case.output_times
    price, baseline, _, _, _, demand = model.eval(
        times,
        state,
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[4],
        parameters[5],
    )
    diagnostics = diagnostics.copy()
    diagnostics.update({
        'output_t': np.asarray(times, dtype=float).copy(),
        'output_indices': output_indices.copy(),
        'breakpoints': np.asarray(case.breakpoints, dtype=float).copy(),
        'requested_max_step': max_step,
    })
    observations = BM1ObservationSeries(
        times=case.observation_times,
        values=np.asarray(demand[1:], dtype=float) + measurement_noise,
        time_unit='day',
        observed=np.ones(case.observation_times.size, dtype=bool),
        observation_kind='point',
        price_side='right',
    )
    truth = BM1SyntheticTruth(
        times=times,
        state=state,
        price=price,
        baseline=baseline,
        demand=demand,
        initial_state=initial_state,
        diagnostics=diagnostics,
    )
    return BM1SyntheticSequence(observations, truth, measurement_noise, sequence_id)


def _coupled_truth_noise(case, sequence_id):
    """Draw the canonical fine Wiener path and its exactly coarsened counterpart."""
    coarse_t, _ = _truth_grid(case, case.truth_max_step)
    fine_t, _ = _truth_grid(case, case.truth_fine_max_step)
    ratio = case.truth_max_step / case.truth_fine_max_step
    factor = int(round(ratio))
    if factor < 2 or not np.isclose(ratio, factor, atol=1e-12, rtol=0.0):
        raise ValueError('Coupled refinement requires an integer truth_max_step / truth_fine_max_step ratio.')
    if fine_t.size - 1 != factor * (coarse_t.size - 1) or not np.allclose(
        fine_t[::factor], coarse_t, atol=1e-14, rtol=0.0
    ):
        raise ValueError('Declared base/fine truth grids are not nested; coupling would be ambiguous.')
    model, _ = case.build_model()
    fine_brownian = model.euler_maruyama.brownian_increments(
        fine_t,
        n_paths=1,
        seed=_seed(case.brownian_seed_root, sequence_id),
    )
    coarse_brownian = model.euler_maruyama.coarsen_brownian_increments(
        fine_brownian,
        factor=factor,
    )
    return coarse_brownian, fine_brownian


def generate_synthetic_sequence(case=None, *, sequence_id=0):
    """Generate the canonical coarse member of the reproducible truth pair.

    Sequence identity is independent of whether the caller also requests the
    fine refinement: synthetic and synthetic --refine therefore use exactly
    the same coarse truth for a given case and sequence_id.
    """
    case = build_f4_b_case() if case is None else case
    if not isinstance(case, BM1SyntheticCase):
        raise TypeError('case must be a BM1SyntheticCase.')
    sequence_id = _nonnegative_integer(sequence_id, 'sequence_id')
    initial_state = _draw_initial_state(case, sequence_id)
    measurement_noise = _draw_measurement_noise(case, sequence_id)
    coarse_brownian, _ = _coupled_truth_noise(case, sequence_id)
    return _simulate(
        case,
        sequence_id,
        initial_state=initial_state,
        measurement_noise=measurement_noise,
        max_step=case.truth_max_step,
        brownian_increments=coarse_brownian,
    )


def generate_coupled_refinement(case=None, *, sequence_id=0):
    """Generate base/fine EM truth using one fine Wiener path and exact coarsening."""
    case = build_f4_b_case() if case is None else case
    if not isinstance(case, BM1SyntheticCase):
        raise TypeError('case must be a BM1SyntheticCase.')
    sequence_id = _nonnegative_integer(sequence_id, 'sequence_id')
    initial_state = _draw_initial_state(case, sequence_id)
    measurement_noise = _draw_measurement_noise(case, sequence_id)
    coarse_brownian, fine_brownian = _coupled_truth_noise(case, sequence_id)
    coarse = _simulate(
        case,
        sequence_id,
        initial_state=initial_state,
        measurement_noise=measurement_noise,
        max_step=case.truth_max_step,
        brownian_increments=coarse_brownian,
    )
    fine = _simulate(
        case,
        sequence_id,
        initial_state=initial_state,
        measurement_noise=measurement_noise,
        max_step=case.truth_fine_max_step,
        brownian_increments=fine_brownian,
    )
    return BM1SyntheticRefinement(coarse=coarse, fine=fine)
