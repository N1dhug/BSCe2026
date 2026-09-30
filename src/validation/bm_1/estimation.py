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
from src.method.beta_mixture import beta_cell_average_basis
from src.params.bm_1 import (
    BM1ObservationSeries, BM1FVFilterResult, bm1_observation_update, run_fv_filter,
)
from src.tools.input_generator.input_builder import input_builder
from src.validation import estimation as checks
from src.validation.bm_1 import cases


CORE_PRICE_EDGES = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=float)
CORE_PRICE_VALUES = np.array([0.65, 0.35, 0.65], dtype=float)
CORE_OBSERVATION_TIMES = np.arange(1, 25, dtype=float) / 24.0
INITIAL_SEED_ROOT = 4101
BROWNIAN_SEED_ROOT = 4102
MEASUREMENT_SEED_ROOT = 4103
F4_C_BOOTSTRAP_SEED = 4201


def _readonly_vector(values: ArrayLike, name: str, *, minimum_size: int = 1) -> NDArray[np.float64]:
    array = checks.finite_vector(values, name, minimum_size=minimum_size)
    array.setflags(write=False)
    return array


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
        baseline = checks.finite_scalar(self.baseline, 'baseline')
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
        C = checks.finite_scalar(self.C, 'C', positive=True)
        k = checks.finite_scalar(self.k, 'k', positive=True)
        Xi = checks.finite_scalar(self.Xi, 'Xi')
        if not 0.0 <= Xi <= 1.0:
            raise ValueError('Xi must lie in [0, 1].')
        sigma_x = checks.finite_scalar(self.sigma_x, 'sigma_x')
        if sigma_x < 0.0:
            raise ValueError('sigma_x must be non-negative.')
        sigma_y = checks.finite_scalar(self.sigma_y, 'sigma_y', positive=True)
        initial_alpha = checks.finite_scalar(self.initial_alpha, 'initial_alpha', positive=True)
        initial_beta = checks.finite_scalar(self.initial_beta, 'initial_beta', positive=True)
        truth_max_step = checks.finite_scalar(self.truth_max_step, 'truth_max_step', positive=True)
        truth_fine_max_step = checks.finite_scalar(self.truth_fine_max_step, 'truth_fine_max_step', positive=True)
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
        initial_state = checks.finite_scalar(self.initial_state, 'initial_state')
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



@dataclass(frozen=True)
class BM1FVCReference:
    """One F4-C synthetic sequence with assimilating and open-propagation filters."""

    case: BM1SyntheticCase
    sequence: BM1SyntheticSequence
    filtered: BM1FVFilterResult
    open_propagation: BM1FVFilterResult


def beta_initial_mass(case, model, x_steps):
    """Discretize the declared Beta initial prior exactly on the FV cells."""
    if not isinstance(case, BM1SyntheticCase):
        raise TypeError('case must be a BM1SyntheticCase.')
    x_steps = checks.positive_integer(x_steps, 'x_steps', minimum=3)
    edges, _, dx = model.fokker_planc_reference.grid(x_steps, bounds=(0.0, 1.0))
    density = beta_cell_average_basis(
        edges, case.initial_alpha, case.initial_beta,
    )[0]
    return checks.probability_mass(density * dx, 'initial_mass')


def missing_copy(observations):
    """Return the same observation times as explicit missing values."""
    if not isinstance(observations, BM1ObservationSeries):
        raise TypeError('observations must be a BM1ObservationSeries.')
    return BM1ObservationSeries(
        observations.times,
        np.full(observations.times.size, np.nan),
        observations.time_unit,
        np.zeros(observations.times.size, dtype=bool),
        observations.observation_kind,
        observations.price_side,
    )


def _run_f4_c_pair(case, sequence, *, x_steps, max_step, likelihood_quadrature_order):
    """Run assimilating and open filters on one already fixed synthetic sequence."""
    model, parameters = case.build_model()
    initial_mass = beta_initial_mass(case, model, x_steps)
    common = dict(
        model=model,
        parameters=parameters,
        sigma_x=case.sigma_x,
        sigma_y=case.sigma_y,
        initial_mass=initial_mass,
        initial_time=case.horizon[0],
        breakpoints=case.breakpoints,
        max_step=max_step,
        likelihood_quadrature_order=likelihood_quadrature_order,
    )
    filtered = run_fv_filter(observations=sequence.observations, **common)
    open_propagation = run_fv_filter(observations=missing_copy(sequence.observations), **common)
    return BM1FVCReference(
        case=case,
        sequence=sequence,
        filtered=filtered,
        open_propagation=open_propagation,
    )


def run_f4_c_reference(sequence_id=0, *, x_steps=201, max_step=1.0 / 192.0,
                       likelihood_quadrature_order=1):
    """Run the F4-C known-parameter FV filter and its no-assimilation baseline."""
    case = build_f4_b_case()
    sequence = generate_synthetic_sequence(case, sequence_id=sequence_id)
    return _run_f4_c_pair(
        case,
        sequence,
        x_steps=x_steps,
        max_step=max_step,
        likelihood_quadrature_order=likelihood_quadrature_order,
    )


def _filter_metrics(reference):
    truth = reference.sequence.truth.state[1:]
    open_mean = reference.open_propagation.posterior_mean
    predictive = reference.filtered.predicted_mean
    posterior = reference.filtered.posterior_mean

    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values - truth))))

    integration_t = reference.filtered.integration_t
    return {
        'rmse_open': rmse(open_mean),
        'rmse_predictive': rmse(predictive),
        'rmse_posterior': rmse(posterior),
        'total_log_likelihood': reference.filtered.total_log_likelihood,
        'maximum_abs_mass_error': reference.filtered.maximum_abs_mass_error,
        'minimum_cell_mass': reference.filtered.minimum_cell_mass,
        'n_steps': int(integration_t.size - 1),
        'actual_max_step': float(np.max(np.diff(integration_t))),
    }


def _refinement_summary(rows):
    reference = rows[-1]
    assimilation = reference['rmse_open'] - reference['rmse_posterior']
    effects = [abs(row['rmse_posterior'] - reference['rmse_posterior']) for row in rows[:-1]]
    likelihood_effects = [
        abs(row['total_log_likelihood'] - reference['total_log_likelihood'])
        for row in rows[:-1]
    ]
    maximum_effect = max(effects, default=0.0)
    return {
        'reference_level': reference['level'],
        'reference_posterior_rmse': reference['rmse_posterior'],
        'reference_assimilation_improvement': assimilation,
        'maximum_posterior_rmse_effect': maximum_effect,
        'effect_to_assimilation_ratio': (
            np.nan if abs(assimilation) < 1e-14 else maximum_effect / abs(assimilation)
        ),
        'maximum_log_likelihood_effect': max(likelihood_effects, default=0.0),
    }


def run_f4_c_refinement(
    sequence_id=0,
    *,
    spatial_cells=(101, 201, 401, 801),
    time_steps=(1.0 / 96.0, 1.0 / 192.0, 1.0 / 384.0),
    quadrature_orders=(1, 3, 5),
):
    """Refine space, FP time and measurement quadrature one dimension at a time."""
    sequence_id = _nonnegative_integer(sequence_id, 'sequence_id')
    cells = tuple(checks.positive_integer(value, 'spatial cell count', minimum=3)
                  for value in spatial_cells)
    steps = tuple(checks.finite_scalar(value, 'time max_step', positive=True)
                  for value in time_steps)
    orders = tuple(checks.positive_integer(value, 'quadrature order', minimum=1)
                   for value in quadrature_orders)
    if not cells or not steps or not orders:
        raise ValueError('Refinement levels must be non-empty.')
    if tuple(sorted(set(cells))) != cells:
        raise ValueError('spatial_cells must be unique and increasing.')
    if tuple(sorted(set(steps), reverse=True)) != steps:
        raise ValueError('time_steps must be unique and strictly refined (decreasing).')
    if tuple(sorted(set(orders))) != orders:
        raise ValueError('quadrature_orders must be unique and increasing.')

    case = build_f4_b_case()
    sequence = generate_synthetic_sequence(case, sequence_id=sequence_id)

    def run(axis, level, x_steps, max_step, quadrature_order):
        reference = _run_f4_c_pair(
            case,
            sequence,
            x_steps=x_steps,
            max_step=max_step,
            likelihood_quadrature_order=quadrature_order,
        )
        return {
            'axis': axis,
            'level': level,
            'x_steps': x_steps,
            'requested_max_step': max_step,
            'quadrature_order': quadrature_order,
            **_filter_metrics(reference),
        }

    spatial = [
        run('space', n, n, 1.0 / 192.0, 1)
        for n in cells
    ]
    time = [
        run('time', h, 201, h, 1)
        for h in steps
    ]
    quadrature = [
        run('quadrature', q, 201, 1.0 / 192.0, q)
        for q in orders
    ]
    return {
        'sequence_id': sequence_id,
        'truth_state': sequence.truth.state[1:].copy(),
        'spatial': spatial,
        'time': time,
        'quadrature': quadrature,
        'summary': {
            'spatial': _refinement_summary(spatial),
            'time': _refinement_summary(time),
            'quadrature': _refinement_summary(quadrature),
        },
    }



def _piecewise_constant_quantile(edges, mass, probability):
    """Quantile of the FV piecewise-constant density without center snapping."""
    edges = checks.increasing_grid(edges, 'edges')
    mass = checks.probability_mass(mass, 'mass')
    probability = checks.finite_scalar(probability, 'probability')
    if not 0.0 <= probability <= 1.0:
        raise ValueError('probability must lie in [0, 1].')
    if mass.size != edges.size - 1:
        raise ValueError('mass must contain one value per cell.')
    if probability == 0.0:
        return float(edges[0])
    if probability == 1.0:
        return float(edges[-1])
    cumulative = np.cumsum(mass)
    cell = int(np.searchsorted(cumulative, probability, side='left'))
    before = 0.0 if cell == 0 else float(cumulative[cell - 1])
    cell_mass = float(mass[cell])
    if cell_mass <= 0.0:
        raise FloatingPointError('Quantile landed in a zero-mass cell; inspect probability mass.')
    fraction = (probability - before) / cell_mass
    return float(edges[cell] + fraction * (edges[cell + 1] - edges[cell]))


def _wilson_interval(successes, total, z=1.959963984540054):
    successes = checks.positive_integer(successes, 'successes', minimum=0)
    total = checks.positive_integer(total, 'total', minimum=1)
    if successes > total:
        raise ValueError('successes cannot exceed total.')
    z = checks.finite_scalar(z, 'z', positive=True)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _paired_bootstrap_interval(values_a, values_b, statistic, *, replicates, seed):
    """Percentile bootstrap resampling whole paired synthetic sequences."""
    a = checks.finite_vector(values_a, 'values_a')
    b = checks.finite_vector(values_b, 'values_b')
    if a.shape != b.shape:
        raise ValueError('Paired bootstrap vectors must have the same shape.')
    replicates = checks.positive_integer(replicates, 'bootstrap_replicates', minimum=100)
    seed = _nonnegative_integer(seed, 'bootstrap_seed')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, a.size, size=(replicates, a.size))
    samples = np.asarray(statistic(a[indices], b[indices]), dtype=float)
    if samples.shape != (replicates,) or not np.all(np.isfinite(samples)):
        raise FloatingPointError('Bootstrap statistic returned invalid values.')
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def _open_log_scores(model, parameters, case, observations, open_result, quadrature_order):
    scores = np.empty(observations.times.size, dtype=float)
    previous = case.horizon[0]
    for index, (time, value) in enumerate(zip(observations.times, observations.values)):
        update = bm1_observation_update(
            model,
            parameters,
            case.sigma_y,
            open_result.posterior_mass[index],
            previous,
            float(time),
            float(value),
            price_side=observations.price_side,
            likelihood_quadrature_order=quadrature_order,
        )
        scores[index] = update.log_evidence
        previous = float(time)
    return scores


def run_f4_c_population(
    n_sequences=200,
    *,
    sequence_start=1,
    x_steps=201,
    max_step=1.0 / 192.0,
    likelihood_quadrature_order=1,
    bootstrap_replicates=5000,
    bootstrap_seed=F4_C_BOOTSTRAP_SEED,
):
    """Run the preregistered F4-C population study on unseen synthetic sequences.

    Sequence zero is intentionally excluded from the default study because it was
    used for sanity and numerical refinement. Whole sequences are the independent
    resampling unit for paired uncertainty.
    """
    n_sequences = checks.positive_integer(n_sequences, 'n_sequences', minimum=2)
    sequence_start = _nonnegative_integer(sequence_start, 'sequence_start')
    x_steps = checks.positive_integer(x_steps, 'x_steps', minimum=3)
    max_step = checks.finite_scalar(max_step, 'max_step', positive=True)
    quadrature_order = checks.positive_integer(
        likelihood_quadrature_order, 'likelihood_quadrature_order', minimum=1,
    )
    bootstrap_replicates = checks.positive_integer(
        bootstrap_replicates, 'bootstrap_replicates', minimum=100,
    )
    bootstrap_seed = _nonnegative_integer(bootstrap_seed, 'bootstrap_seed')

    case = build_f4_b_case()
    model, parameters = case.build_model()
    initial_mass = beta_initial_mass(case, model, x_steps)
    empty = BM1ObservationSeries(
        case.observation_times,
        np.full(case.observation_times.size, np.nan),
        'day',
        np.zeros(case.observation_times.size, dtype=bool),
        'point',
        'right',
    )
    open_result = run_fv_filter(
        model,
        parameters,
        case.sigma_x,
        case.sigma_y,
        initial_mass,
        case.horizon[0],
        empty,
        breakpoints=case.breakpoints,
        max_step=max_step,
        likelihood_quadrature_order=quadrature_order,
    )
    open_mean = open_result.posterior_mean

    sequence_ids = np.arange(sequence_start, sequence_start + n_sequences, dtype=int)
    sequences = [generate_synthetic_sequence(case, sequence_id=int(i)) for i in sequence_ids]
    invalid = [sequence.sequence_id for sequence in sequences if not sequence.truth.valid]
    if invalid:
        raise RuntimeError(
            'F4-C population truth contains raw EM boundary violations for sequence ids '
            + ', '.join(str(value) for value in invalid)
            + '; do not drop or repair these sequences silently.'
        )

    n_times = case.observation_times.size
    mse_open = np.empty(n_sequences)
    mse_predictive = np.empty(n_sequences)
    mse_posterior = np.empty(n_sequences)
    filter_log_score = np.empty(n_sequences)
    open_log_score = np.empty(n_sequences)

    coverage_hours = (6, 12, 24)
    coverage_indices = []
    for hour in coverage_hours:
        matches = np.flatnonzero(np.isclose(case.observation_times, hour / 24.0, atol=1e-14, rtol=0.0))
        if matches.size != 1:
            raise RuntimeError(f'Expected exactly one {hour}h observation time.')
        coverage_indices.append(int(matches[0]))
    covered = {hour: np.zeros(n_sequences, dtype=bool) for hour in coverage_hours}
    widths = {hour: np.empty(n_sequences) for hour in coverage_hours}

    first_filter_score = np.empty(n_sequences)
    first_open_score = np.empty(n_sequences)

    for row, sequence in enumerate(sequences):
        filtered = run_fv_filter(
            model,
            parameters,
            case.sigma_x,
            case.sigma_y,
            initial_mass,
            case.horizon[0],
            sequence.observations,
            breakpoints=case.breakpoints,
            max_step=max_step,
            likelihood_quadrature_order=quadrature_order,
        )
        truth = sequence.truth.state[1:]
        if truth.size != n_times:
            raise RuntimeError('Synthetic truth and observation lengths no longer match.')
        mse_open[row] = np.mean(np.square(open_mean - truth))
        mse_predictive[row] = np.mean(np.square(filtered.predicted_mean - truth))
        mse_posterior[row] = np.mean(np.square(filtered.posterior_mean - truth))
        filter_scores = filtered.log_evidence
        baseline_scores = _open_log_scores(
            model, parameters, case, sequence.observations, open_result, quadrature_order,
        )
        filter_log_score[row] = float(np.mean(filter_scores))
        open_log_score[row] = float(np.mean(baseline_scores))
        first_filter_score[row] = filter_scores[0]
        first_open_score[row] = baseline_scores[0]

        for hour, index in zip(coverage_hours, coverage_indices):
            mass = filtered.posterior_mass[index]
            lower = _piecewise_constant_quantile(filtered.x_edges, mass, 0.05)
            upper = _piecewise_constant_quantile(filtered.x_edges, mass, 0.95)
            state = float(truth[index])
            covered[hour][row] = lower <= state <= upper
            widths[hour][row] = upper - lower

    def global_rmse(mse):
        return float(np.sqrt(np.mean(mse)))

    rmse_open = global_rmse(mse_open)
    rmse_predictive = global_rmse(mse_predictive)
    rmse_posterior = global_rmse(mse_posterior)

    def rmse_ratio_stat(numerator_mse, denominator_mse):
        return np.sqrt(np.mean(numerator_mse, axis=1) / np.mean(denominator_mse, axis=1))

    posterior_ratio_ci = _paired_bootstrap_interval(
        mse_posterior,
        mse_open,
        rmse_ratio_stat,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    predictive_ratio_ci = _paired_bootstrap_interval(
        mse_predictive,
        mse_open,
        rmse_ratio_stat,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 1,
    )

    log_score_difference = filter_log_score - open_log_score
    def mean_difference_stat(filter_values, open_values):
        return np.mean(filter_values - open_values, axis=1)

    log_score_ci = _paired_bootstrap_interval(
        filter_log_score,
        open_log_score,
        mean_difference_stat,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 2,
    )

    coverage = {}
    for hour in coverage_hours:
        successes = int(np.sum(covered[hour]))
        low, high = _wilson_interval(successes, n_sequences)
        coverage[hour] = {
            'successes': successes,
            'total': n_sequences,
            'coverage': successes / n_sequences,
            'wilson95': (low, high),
            'mean_width': float(np.mean(widths[hour])),
        }

    return {
        'sequence_ids': sequence_ids,
        'n_sequences': n_sequences,
        'settings': {
            'x_steps': x_steps,
            'max_step': max_step,
            'likelihood_quadrature_order': quadrature_order,
            'bootstrap_replicates': bootstrap_replicates,
            'bootstrap_seed': bootstrap_seed,
        },
        'rmse': {
            'open': rmse_open,
            'predictive': rmse_predictive,
            'posterior': rmse_posterior,
            'predictive_over_open': rmse_predictive / rmse_open,
            'posterior_over_open': rmse_posterior / rmse_open,
            'predictive_over_open_bootstrap95': predictive_ratio_ci,
            'posterior_over_open_bootstrap95': posterior_ratio_ci,
        },
        'predictive_log_score': {
            'filter_mean_per_observation': float(np.mean(filter_log_score)),
            'open_mean_per_observation': float(np.mean(open_log_score)),
            'paired_difference': float(np.mean(log_score_difference)),
            'paired_difference_bootstrap95': log_score_ci,
            'first_observation_max_abs_difference': float(
                np.max(np.abs(first_filter_score - first_open_score))
            ),
        },
        'coverage90_posterior': coverage,
        'diagnostics': {
            'invalid_truth_sequences': tuple(invalid),
            'open_maximum_abs_mass_error': open_result.maximum_abs_mass_error,
            'open_minimum_cell_mass': open_result.minimum_cell_mass,
        },
        'per_sequence': {
            'mse_open': mse_open,
            'mse_predictive': mse_predictive,
            'mse_posterior': mse_posterior,
            'filter_log_score': filter_log_score,
            'open_log_score': open_log_score,
        },
    }
