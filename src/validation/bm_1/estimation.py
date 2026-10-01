"""Synthetic BM1 estimation cases and truth generation for F4-B.

This module is an experiment/validation layer.  It reuses the existing BM1
Euler--Maruyama solver and input builders; it does not implement a second SDE
solver or expose synthetic truth through the estimator input contract.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
import operator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.model.bm_1 import bm_1
from src.method.beta_mixture import beta_cell_average_basis
from src.method.bm_1_estimation import (
    BM1FVFilterResult, bm1_observation_update, run_fv_filter,
)
from src.params.bm_1 import BM1ObservationSeries
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
F4_E_NUMERICAL_SEQUENCE_ID = 1211
F4_E_SCREEN_SEQUENCE_START = 1212
F4_E_SCREEN_SEQUENCE_COUNT = 50
F4_E_RHO_GRID = tuple(0.5 + 0.025 * j for j in range(41))
F4_E_TIME_STEPS = (1.0 / 192.0, 1.0 / 384.0, 1.0 / 768.0)
F4_E_TIME_X_STEPS = 201
F4_E_TIME_QUADRATURE_ORDER = 3
F4_E_SPATIAL_CELLS = (201, 401, 801)
F4_E_SPATIAL_MAX_STEP = 1.0 / 768.0
F4_E_SPATIAL_QUADRATURE_ORDER = 3
F4_E_QUADRATURE_ORDERS = (1, 3, 5)
F4_E_QUADRATURE_X_STEPS = 401
F4_E_QUADRATURE_MAX_STEP = 1.0 / 768.0
F4_E_LIKELIHOOD_WINDOW = 5.0
F4_E_CENTERED_LIKELIHOOD_TOL = 0.10
F4_E_PARAMETER_GRIDS = {
    'rho': F4_E_RHO_GRID,
    'Xi': tuple(0.025 * j for j in range(41)),
    'k': tuple(0.5 + 0.075 * j for j in range(41)),
    'sigma_x': tuple(0.5 * 2.0 ** ((j - 20) / 10.0) for j in range(41)),
    'sigma_y': tuple(0.03 * 2.0 ** ((j - 20) / 10.0) for j in range(41)),
}
F4_E_SCREEN_X_STEPS = 401
F4_E_SCREEN_MAX_STEP = 1.0 / 768.0
F4_E_SCREEN_QUADRATURE_ORDER = 3
F4_E_LR_THRESHOLD = 3.841
F4_E_PAIRS = (('rho', 'Xi'), ('k', 'Xi'), ('sigma_x', 'sigma_y'))
F4_E_PAIR_SEQUENCE_COUNT = 10
F4_E_PAIR_GRID_STRIDE = 2


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


def _simulate_batch(case, sequence_ids, *, initial_states, measurement_noises,
                    max_step, brownian_increments):
    """Use the existing EM solver once for independent, explicitly seeded paths.

    With boundary='none', path[:, 1:] contains exactly the raw proposals.
    Diagnostics are reconstructed per path, never pooled across sequences.
    """
    model, parameters = case.build_model()
    integration_t, output_indices = _truth_grid(case, max_step)
    paths = model.euler_maruyama.solve_paths(
        integration_t, initial_states,
        lambda time, states: model._drift(time, states, parameters),
        lambda time, states: model.diffusion_coefficient(states, case.sigma_x),
        n_paths=len(sequence_ids), boundary='none',
        brownian_increments=brownian_increments, return_diagnostics=False,
    )
    results = []
    times = case.output_times
    for sequence_id, path, initial_state, noise in zip(
        sequence_ids, paths, initial_states, measurement_noises,
    ):
        raw = path[1:]
        lower, upper = (raw < 0.0).astype(int), (raw > 1.0).astype(int)
        violations = int(lower.sum() + upper.sum())
        diagnostics = {
            'integration_t': integration_t.copy(), 'n_paths': 1,
            'n_steps': integration_t.size - 1, 'boundary': 'none',
            'minimum_raw_proposal': float(raw.min()),
            'maximum_raw_proposal': float(raw.max()),
            'lower_violations': int(lower.sum()), 'upper_violations': int(upper.sum()),
            'total_violations': violations, 'violating_paths': int(violations > 0),
            'lower_violations_by_step': lower, 'upper_violations_by_step': upper,
            'boundary_corrections': 0, 'output_t': times.copy(),
            'output_indices': output_indices.copy(), 'breakpoints': case.breakpoints,
            'requested_max_step': max_step,
        }
        state = path[output_indices]
        price, baseline, _, _, _, demand = model.eval(
            times, state, parameters[0], parameters[1], parameters[2],
            parameters[4], parameters[5],
        )
        observations = BM1ObservationSeries(
            case.observation_times, np.asarray(demand[1:]) + noise, 'day',
            np.ones(case.observation_times.size, dtype=bool), 'point', 'right',
        )
        truth = BM1SyntheticTruth(times, state, price, baseline, demand,
                                  float(initial_state), diagnostics)
        results.append(BM1SyntheticSequence(observations, truth, noise, sequence_id))
    return results


def _simulate(case, sequence_id, *, initial_state, measurement_noise, max_step,
              brownian_increments):
    return _simulate_batch(
        case, (sequence_id,), initial_states=[initial_state],
        measurement_noises=[measurement_noise], max_step=max_step,
        brownian_increments=brownian_increments,
    )[0]


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



def generate_synthetic_sequences(case=None, *, sequence_ids, refine=False):
    """Batch F4-B generation with the same per-ID RNG streams as single calls.

    refine=True returns coupled coarse/fine pairs; no sequence is dropped,
    repaired or re-seeded. Batch size and ordering do not define randomness.
    """
    case = build_f4_b_case() if case is None else case
    if not isinstance(case, BM1SyntheticCase):
        raise TypeError('case must be a BM1SyntheticCase.')
    ids = tuple(_nonnegative_integer(i, 'sequence_id') for i in sequence_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('sequence_ids must be non-empty and unique.')
    if not isinstance(refine, (bool, np.bool_)):
        raise TypeError('refine must be boolean.')
    initial = [_draw_initial_state(case, i) for i in ids]
    noise = [_draw_measurement_noise(case, i) for i in ids]
    brownian = [_coupled_truth_noise(case, i) for i in ids]
    coarse = _simulate_batch(
        case, ids, initial_states=initial, measurement_noises=noise,
        max_step=case.truth_max_step,
        brownian_increments=np.concatenate([pair[0] for pair in brownian]),
    )
    if not refine:
        return coarse
    fine = _simulate_batch(
        case, ids, initial_states=initial, measurement_noises=noise,
        max_step=case.truth_fine_max_step,
        brownian_increments=np.concatenate([pair[1] for pair in brownian]),
    )
    return [BM1SyntheticRefinement(c, f) for c, f in zip(coarse, fine)]


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

    cache = {}

    def run(axis, level, x_steps, max_step, quadrature_order):
        key = (x_steps, max_step, quadrature_order)
        if key not in cache:
            cache[key] = _run_f4_c_pair(
                case, sequence, x_steps=x_steps, max_step=max_step,
                likelihood_quadrature_order=quadrature_order,
            )
        reference = cache[key]
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
    sequences = generate_synthetic_sequences(case, sequence_ids=sequence_ids)
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



def _piecewise_constant_variance(edges, mass):
    """Variance under the FV piecewise-constant density, including within-cell spread."""
    edges = checks.increasing_grid(edges, 'edges')
    mass = checks.probability_mass(mass, 'mass')
    if mass.size != edges.size - 1:
        raise ValueError('mass must contain one value per cell.')
    left, right = edges[:-1], edges[1:]
    centers = 0.5 * (left + right)
    second = (left * left + left * right + right * right) / 3.0
    mean = float(mass @ centers)
    variance = float(mass @ second - mean * mean)
    if variance < -1e-14:
        raise FloatingPointError('Piecewise-constant variance became materially negative.')
    return max(variance, 0.0)


def _calibration_summary_for_sequences(case, model, parameters, sequences, *,
                                       x_steps, max_step, quadrature_order):
    """Coverage and posterior-scale diagnostics for one fixed sequence collection."""
    n_sequences = len(sequences)
    if n_sequences < 1:
        raise ValueError('At least one sequence is required for calibration diagnostics.')
    hours = (6, 12, 24)
    indices = []
    for hour in hours:
        matches = np.flatnonzero(
            np.isclose(case.observation_times, hour / 24.0, atol=1e-14, rtol=0.0)
        )
        if matches.size != 1:
            raise RuntimeError(f'Expected exactly one {hour}h observation time.')
        indices.append(int(matches[0]))

    initial_mass = beta_initial_mass(case, model, x_steps)
    covered = {hour: np.zeros(n_sequences, dtype=bool) for hour in hours}
    widths = {hour: np.empty(n_sequences) for hour in hours}
    squared_error = {hour: np.empty(n_sequences) for hour in hours}
    posterior_variance = {hour: np.empty(n_sequences) for hour in hours}
    maximum_mass_error = 0.0
    minimum_cell_mass = np.inf

    for row, sequence in enumerate(sequences):
        filtered = run_fv_filter(
            model, parameters, case.sigma_x, case.sigma_y, initial_mass,
            case.horizon[0], sequence.observations, breakpoints=case.breakpoints,
            max_step=max_step, likelihood_quadrature_order=quadrature_order,
        )
        maximum_mass_error = max(maximum_mass_error, filtered.maximum_abs_mass_error)
        minimum_cell_mass = min(minimum_cell_mass, filtered.minimum_cell_mass)
        truth = sequence.truth.state[1:]
        for hour, index in zip(hours, indices):
            mass = filtered.posterior_mass[index]
            lower = _piecewise_constant_quantile(filtered.x_edges, mass, 0.05)
            upper = _piecewise_constant_quantile(filtered.x_edges, mass, 0.95)
            state = float(truth[index])
            mean = float(mass @ filtered.x_grid)
            covered[hour][row] = lower <= state <= upper
            widths[hour][row] = upper - lower
            squared_error[hour][row] = (mean - state) ** 2
            posterior_variance[hour][row] = _piecewise_constant_variance(
                filtered.x_edges, mass
            )

    horizons = {}
    for hour in hours:
        successes = int(np.sum(covered[hour]))
        low, high = _wilson_interval(successes, n_sequences)
        rmse = float(np.sqrt(np.mean(squared_error[hour])))
        posterior_sd = float(np.sqrt(np.mean(posterior_variance[hour])))
        horizons[hour] = {
            'successes': successes,
            'total': n_sequences,
            'coverage': successes / n_sequences,
            'wilson95': (low, high),
            'mean_width': float(np.mean(widths[hour])),
            'posterior_mean_rmse': rmse,
            'root_mean_posterior_variance': posterior_sd,
            'rmse_over_posterior_sd': (
                np.nan if posterior_sd == 0.0 else rmse / posterior_sd
            ),
        }
    return {
        'horizons': horizons,
        'maximum_abs_mass_error': maximum_mass_error,
        'minimum_cell_mass': float(minimum_cell_mass),
    }

def run_f4_c_calibration_sensitivity(
    n_sequences=200,
    *,
    sequence_start=1,
    variants=None,
):
    """Post-hoc numerical sensitivity of the preregistered 6/12/24 h coverage result.

    This does not replace or retune the primary C4 result. It reuses the same
    fixed synthetic sequence ids and changes one numerical dimension at a time
    to diagnose whether coverage is materially driven by filter discretization.
    """
    n_sequences = checks.positive_integer(n_sequences, 'n_sequences', minimum=2)
    sequence_start = _nonnegative_integer(sequence_start, 'sequence_start')
    if variants is None:
        variants = (
            ('base', 201, 1.0 / 192.0, 1),
            ('space401', 401, 1.0 / 192.0, 1),
            ('time384', 201, 1.0 / 384.0, 1),
            ('quad3', 201, 1.0 / 192.0, 3),
        )
    parsed = []
    names = set()
    for item in variants:
        if len(item) != 4:
            raise ValueError('Each calibration variant must be (name, x_steps, max_step, quadrature_order).')
        name, x_steps, max_step, order = item
        if not isinstance(name, str) or not name:
            raise ValueError('Calibration variant names must be non-empty strings.')
        if name in names:
            raise ValueError('Calibration variant names must be unique.')
        names.add(name)
        parsed.append((
            name,
            checks.positive_integer(x_steps, 'x_steps', minimum=3),
            checks.finite_scalar(max_step, 'max_step', positive=True),
            checks.positive_integer(order, 'quadrature_order', minimum=1),
        ))

    case = build_f4_b_case()
    model, parameters = case.build_model()
    sequence_ids = np.arange(sequence_start, sequence_start + n_sequences, dtype=int)
    sequences = generate_synthetic_sequences(case, sequence_ids=sequence_ids)
    invalid = [sequence.sequence_id for sequence in sequences if not sequence.truth.valid]
    if invalid:
        raise RuntimeError(
            'Calibration sensitivity truth contains raw EM boundary violations for sequence ids '
            + ', '.join(str(value) for value in invalid)
        )

    results = {}
    for name, x_steps, max_step, order in parsed:
        summary = _calibration_summary_for_sequences(
            case, model, parameters, sequences,
            x_steps=x_steps, max_step=max_step, quadrature_order=order,
        )
        results[name] = {
            'x_steps': x_steps,
            'max_step': max_step,
            'quadrature_order': order,
            **summary,
        }
    return {
        'sequence_ids': sequence_ids,
        'n_sequences': n_sequences,
        'invalid_truth_sequences': tuple(invalid),
        'variants': results,
    }

def run_f4_c_truth_refinement(
    n_sequences=200,
    *,
    sequence_start=1,
    x_steps=201,
    max_step=1.0 / 192.0,
    likelihood_quadrature_order=1,
):
    """Post-hoc paired coarse/fine truth-EM sensitivity for C4 calibration.

    The F4-B coupled generator supplies the same initial draw, Brownian path and
    measurement noise to both truth resolutions. The filter numerics are held fixed.
    """
    n_sequences = checks.positive_integer(n_sequences, 'n_sequences', minimum=2)
    sequence_start = _nonnegative_integer(sequence_start, 'sequence_start')
    x_steps = checks.positive_integer(x_steps, 'x_steps', minimum=3)
    max_step = checks.finite_scalar(max_step, 'max_step', positive=True)
    order = checks.positive_integer(
        likelihood_quadrature_order, 'likelihood_quadrature_order', minimum=1,
    )
    case = build_f4_b_case()
    model, parameters = case.build_model()
    sequence_ids = np.arange(sequence_start, sequence_start + n_sequences, dtype=int)
    pairs = generate_synthetic_sequences(case, sequence_ids=sequence_ids, refine=True)
    invalid = [
        (pair.coarse.sequence_id, pair.coarse.truth.valid, pair.fine.truth.valid)
        for pair in pairs
        if not pair.coarse.truth.valid or not pair.fine.truth.valid
    ]
    if invalid:
        raise RuntimeError(
            'Truth-refinement diagnostic contains raw EM boundary violations; '
            f'details={invalid}'
        )
    coarse = [pair.coarse for pair in pairs]
    fine = [pair.fine for pair in pairs]
    coarse_summary = _calibration_summary_for_sequences(
        case, model, parameters, coarse,
        x_steps=x_steps, max_step=max_step, quadrature_order=order,
    )
    fine_summary = _calibration_summary_for_sequences(
        case, model, parameters, fine,
        x_steps=x_steps, max_step=max_step, quadrature_order=order,
    )

    state_differences = np.stack([
        np.abs(pair.coarse.truth.state[1:] - pair.fine.truth.state[1:])
        for pair in pairs
    ])
    demand_differences = np.stack([
        np.abs(pair.coarse.truth.demand[1:] - pair.fine.truth.demand[1:])
        for pair in pairs
    ])
    measurement_differences = np.stack([
        np.abs(pair.coarse.observations.values - pair.fine.observations.values)
        for pair in pairs
    ])
    return {
        'sequence_ids': sequence_ids,
        'n_sequences': n_sequences,
        'settings': {
            'x_steps': x_steps,
            'max_step': max_step,
            'likelihood_quadrature_order': order,
            'coarse_truth_max_step': case.truth_max_step,
            'fine_truth_max_step': case.truth_fine_max_step,
        },
        'invalid_truth_pairs': tuple(invalid),
        'coarse': coarse_summary,
        'fine': fine_summary,
        'truth_difference': {
            'maximum_abs_state': float(np.max(state_differences)),
            'rms_state': float(np.sqrt(np.mean(np.square(state_differences)))),
            'maximum_abs_demand': float(np.max(demand_differences)),
            'rms_demand': float(np.sqrt(np.mean(np.square(demand_differences)))),
            'maximum_abs_observation': float(np.max(measurement_differences)),
        },
    }



@dataclass(frozen=True)
class F4ELikelihoodCandidate:
    """One fixed BM1 parameter candidate for F4-E likelihood replay.

    F4-E keeps f and g fixed.  rho=1/C is used instead of C so the candidate
    coordinates match the preregistered identifiability study.
    """

    rho: float
    Xi: float
    k: float
    sigma_x: float
    sigma_y: float

    def __post_init__(self):
        rho = checks.finite_scalar(self.rho, 'rho', positive=True)
        Xi = checks.finite_scalar(self.Xi, 'Xi')
        k = checks.finite_scalar(self.k, 'k', positive=True)
        sigma_x = checks.finite_scalar(self.sigma_x, 'sigma_x')
        sigma_y = checks.finite_scalar(self.sigma_y, 'sigma_y', positive=True)
        if not 0.0 <= Xi <= 1.0:
            raise ValueError('Xi must lie in [0, 1].')
        if sigma_x < 0.0:
            raise ValueError('sigma_x must be non-negative.')
        object.__setattr__(self, 'rho', rho)
        object.__setattr__(self, 'Xi', Xi)
        object.__setattr__(self, 'k', k)
        object.__setattr__(self, 'sigma_x', sigma_x)
        object.__setattr__(self, 'sigma_y', sigma_y)

    @property
    def C(self):
        return 1.0 / self.rho

    def as_tuple(self):
        return (self.rho, self.Xi, self.k, self.sigma_x, self.sigma_y)


def f4_e_truth_candidate(case=None):
    """Return the fixed-parameter candidate that generated the declared case."""
    case = build_f4_b_case() if case is None else case
    if not isinstance(case, BM1SyntheticCase):
        raise TypeError('case must be a BM1SyntheticCase.')
    return F4ELikelihoodCandidate(
        rho=1.0 / case.C,
        Xi=case.Xi,
        k=case.k,
        sigma_x=case.sigma_x,
        sigma_y=case.sigma_y,
    )


def _f4_e_parameters(base_parameters, candidate):
    """Replace only F4-E's physical candidate coordinates; keep U, f and g fixed."""
    if not isinstance(candidate, F4ELikelihoodCandidate):
        raise TypeError('candidate must be an F4ELikelihoodCandidate.')
    parameters = list(base_parameters)
    if len(parameters) != 6:
        raise ValueError('BM1 parameter vector must be [U, f, g, C, k, Xi].')
    parameters[3] = candidate.C
    parameters[4] = candidate.k
    parameters[5] = candidate.Xi
    return parameters


def _evaluate_f4_e_prepared(
    case,
    observations,
    candidate,
    *,
    model,
    base_parameters,
    initial_mass,
    x_steps,
    max_step,
    likelihood_quadrature_order,
):
    """Evaluate one candidate with common immutable inputs prepared by the caller."""
    parameters = _f4_e_parameters(base_parameters, candidate)
    filtered = run_fv_filter(
        model,
        parameters,
        candidate.sigma_x,
        candidate.sigma_y,
        initial_mass,
        case.horizon[0],
        observations,
        breakpoints=case.breakpoints,
        max_step=max_step,
        likelihood_quadrature_order=likelihood_quadrature_order,
    )
    log_evidence = filtered.log_evidence[filtered.observed].copy()
    log_evidence.setflags(write=False)
    integration_t = filtered.integration_t
    return {
        'candidate': candidate,
        'total_log_likelihood': float(np.sum(log_evidence)),
        'log_evidence': log_evidence,
        'maximum_abs_mass_error': filtered.maximum_abs_mass_error,
        'minimum_cell_mass': filtered.minimum_cell_mass,
        'x_steps': x_steps,
        'max_step': max_step,
        'actual_max_step': float(np.max(np.diff(integration_t))),
        'n_steps': int(integration_t.size - 1),
        'likelihood_quadrature_order': likelihood_quadrature_order,
    }


def evaluate_f4_e_candidate(
    case,
    observations,
    candidate,
    *,
    x_steps=201,
    max_step=1.0 / 192.0,
    likelihood_quadrature_order=1,
):
    """Replay one fixed candidate from the common prior and return predictive evidence.

    No synthetic truth enters this API.  Every call rebuilds the initial law and
    starts run_fv_filter from case.horizon[0]; callers must not carry a posterior
    from another candidate into this evaluation.
    """
    return evaluate_f4_e_candidates(
        case,
        observations,
        (candidate,),
        x_steps=x_steps,
        max_step=max_step,
        likelihood_quadrature_order=likelihood_quadrature_order,
    )[0]


def evaluate_f4_e_candidates(
    case,
    observations,
    candidates,
    *,
    x_steps=201,
    max_step=1.0 / 192.0,
    likelihood_quadrature_order=1,
):
    """Evaluate candidates independently while reusing only candidate-free setup.

    Model/input objects and the declared initial probability are common. Every
    candidate still gets a fresh parameter vector and a complete run_fv_filter
    replay from the same initial mass. Caller order and duplicates are preserved.
    """
    if not isinstance(case, BM1SyntheticCase):
        raise TypeError('case must be a BM1SyntheticCase.')
    if not isinstance(observations, BM1ObservationSeries):
        raise TypeError('observations must be a BM1ObservationSeries.')
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError('candidates must be non-empty.')
    for candidate in candidates:
        if not isinstance(candidate, F4ELikelihoodCandidate):
            raise TypeError('Every candidate must be an F4ELikelihoodCandidate.')
    x_steps = checks.positive_integer(x_steps, 'x_steps', minimum=3)
    max_step = checks.finite_scalar(max_step, 'max_step', positive=True)
    order = checks.positive_integer(
        likelihood_quadrature_order, 'likelihood_quadrature_order', minimum=1,
    )
    model, base_parameters = case.build_model()
    initial_mass = beta_initial_mass(case, model, x_steps)
    return tuple(
        _evaluate_f4_e_prepared(
            case,
            observations,
            candidate,
            model=model,
            base_parameters=base_parameters,
            initial_mass=initial_mass,
            x_steps=x_steps,
            max_step=max_step,
            likelihood_quadrature_order=order,
        )
        for candidate in candidates
    )


def f4_e_centered_log_likelihood(evaluations):
    """Return candidate values and centered log-likelihood without changing order."""
    evaluations = tuple(evaluations)
    if not evaluations:
        raise ValueError('evaluations must be non-empty.')
    values = np.asarray(
        [item['total_log_likelihood'] for item in evaluations], dtype=float,
    )
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError('All F4-E total log-likelihoods must be finite.')
    centered = values - float(np.max(values))
    centered.setflags(write=False)
    return centered



def _f4_e_rho_curve(sequence_id, rho_values, x_steps, max_step, quadrature_order):
    """Evaluate one numerical configuration on the fixed E1 rho grid."""
    sequence_id = _nonnegative_integer(sequence_id, 'sequence_id')
    rho_values = tuple(checks.finite_scalar(value, 'rho', positive=True) for value in rho_values)
    if not rho_values or len(set(rho_values)) != len(rho_values):
        raise ValueError('rho_values must be non-empty and unique.')
    if tuple(sorted(rho_values)) != rho_values:
        raise ValueError('rho_values must be strictly increasing.')
    x_steps = checks.positive_integer(x_steps, 'x_steps', minimum=3)
    max_step = checks.finite_scalar(max_step, 'max_step', positive=True)
    quadrature_order = checks.positive_integer(
        quadrature_order, 'quadrature_order', minimum=1,
    )

    case = build_f4_b_case()
    sequence = generate_synthetic_sequence(case, sequence_id=sequence_id)
    if not sequence.truth.valid:
        raise RuntimeError(
            f'F4-E sequence {sequence_id} has raw EM boundary violations; '
            'do not repair or drop it silently.'
        )
    truth = f4_e_truth_candidate(case)
    candidates = tuple(
        F4ELikelihoodCandidate(
            rho=rho,
            Xi=truth.Xi,
            k=truth.k,
            sigma_x=truth.sigma_x,
            sigma_y=truth.sigma_y,
        )
        for rho in rho_values
    )
    evaluations = evaluate_f4_e_candidates(
        case,
        sequence.observations,
        candidates,
        x_steps=x_steps,
        max_step=max_step,
        likelihood_quadrature_order=quadrature_order,
    )
    total = np.asarray(
        [item['total_log_likelihood'] for item in evaluations], dtype=float,
    )
    centered = f4_e_centered_log_likelihood(evaluations)
    maximum = float(np.max(total))
    maximizers = tuple(
        int(index) for index in np.flatnonzero(
            np.isclose(total, maximum, atol=1e-12, rtol=0.0)
        )
    )
    return {
        'rho_values': rho_values,
        'total_log_likelihood': tuple(float(value) for value in total),
        'centered_log_likelihood': tuple(float(value) for value in centered),
        'maximizer_indices': maximizers,
        'maximizer_rho': (
            float(rho_values[maximizers[0]]) if len(maximizers) == 1 else None
        ),
        'x_steps': x_steps,
        'requested_max_step': max_step,
        'actual_max_step': max(item['actual_max_step'] for item in evaluations),
        'n_steps': max(item['n_steps'] for item in evaluations),
        'likelihood_quadrature_order': quadrature_order,
        'maximum_abs_mass_error': max(
            item['maximum_abs_mass_error'] for item in evaluations
        ),
        'minimum_cell_mass': min(item['minimum_cell_mass'] for item in evaluations),
    }


def _f4_e_rho_curve_task(task):
    sequence_id, rho_values, x_steps, max_step, quadrature_order = task
    key = (int(x_steps), float(max_step), int(quadrature_order))
    return key, _f4_e_rho_curve(
        sequence_id, rho_values, x_steps, max_step, quadrature_order,
    )


def _f4_e_likelihood_stability(
    middle,
    fine,
    *,
    window=F4_E_LIKELIHOOD_WINDOW,
    tolerance=F4_E_CENTERED_LIKELIHOOD_TOL,
):
    """Compare two centered likelihood curves on the preregistered local window."""
    window = checks.finite_scalar(window, 'likelihood window', positive=True)
    tolerance = checks.finite_scalar(
        tolerance, 'centered likelihood tolerance', positive=True,
    )
    if tuple(middle['rho_values']) != tuple(fine['rho_values']):
        raise ValueError('Likelihood curves must use the same rho grid.')
    rho = np.asarray(middle['rho_values'], dtype=float)
    middle_curve = np.asarray(middle['centered_log_likelihood'], dtype=float)
    fine_curve = np.asarray(fine['centered_log_likelihood'], dtype=float)
    if (
        rho.ndim != 1 or middle_curve.shape != rho.shape or fine_curve.shape != rho.shape
        or not np.all(np.isfinite(middle_curve)) or not np.all(np.isfinite(fine_curve))
    ):
        raise ValueError('Likelihood curves must be finite vectors matching rho_values.')
    mask = (middle_curve >= -window) | (fine_curve >= -window)
    if not np.any(mask):
        raise RuntimeError('Likelihood comparison window unexpectedly contains no grid points.')
    maximum_difference = float(np.max(np.abs(middle_curve[mask] - fine_curve[mask])))
    middle_max = tuple(middle['maximizer_indices'])
    fine_max = tuple(fine['maximizer_indices'])
    unique_same_maximizer = bool(
        len(middle_max) == 1 and len(fine_max) == 1 and middle_max == fine_max
    )
    return {
        'middle_maximizer_indices': middle_max,
        'fine_maximizer_indices': fine_max,
        'unique_same_maximizer': unique_same_maximizer,
        'comparison_indices': tuple(int(index) for index in np.flatnonzero(mask)),
        'comparison_rho_range': (
            float(rho[np.flatnonzero(mask)[0]]),
            float(rho[np.flatnonzero(mask)[-1]]),
        ),
        'maximum_centered_log_likelihood_change': maximum_difference,
        'window': window,
        'tolerance': tolerance,
        'passes': bool(unique_same_maximizer and maximum_difference <= tolerance),
    }


def run_f4_e_likelihood_refinement(
    *,
    sequence_id=F4_E_NUMERICAL_SEQUENCE_ID,
    rho_values=F4_E_RHO_GRID,
    time_steps=F4_E_TIME_STEPS,
    time_x_steps=F4_E_TIME_X_STEPS,
    time_quadrature_order=F4_E_TIME_QUADRATURE_ORDER,
    spatial_cells=F4_E_SPATIAL_CELLS,
    spatial_max_step=F4_E_SPATIAL_MAX_STEP,
    spatial_quadrature_order=F4_E_SPATIAL_QUADRATURE_ORDER,
    quadrature_orders=F4_E_QUADRATURE_ORDERS,
    quadrature_x_steps=F4_E_QUADRATURE_X_STEPS,
    quadrature_max_step=F4_E_QUADRATURE_MAX_STEP,
    workers=1,
    progress=None,
):
    """Run preregistered E1 rho-likelihood refinement with deduplicated configs."""
    sequence_id = _nonnegative_integer(sequence_id, 'sequence_id')
    rho_values = tuple(checks.finite_scalar(value, 'rho', positive=True) for value in rho_values)
    if not rho_values or len(set(rho_values)) != len(rho_values) or tuple(sorted(rho_values)) != rho_values:
        raise ValueError('rho_values must be unique and strictly increasing.')
    time_steps = tuple(
        checks.finite_scalar(value, 'time max_step', positive=True) for value in time_steps
    )
    spatial_cells = tuple(
        checks.positive_integer(value, 'spatial cell count', minimum=3)
        for value in spatial_cells
    )
    quadrature_orders = tuple(
        checks.positive_integer(value, 'quadrature order', minimum=1)
        for value in quadrature_orders
    )
    if len(time_steps) != 3 or tuple(sorted(set(time_steps), reverse=True)) != time_steps:
        raise ValueError('E1 time_steps must contain exactly three unique decreasing levels.')
    if len(spatial_cells) != 3 or tuple(sorted(set(spatial_cells))) != spatial_cells:
        raise ValueError('E1 spatial_cells must contain exactly three unique increasing levels.')
    if len(quadrature_orders) != 3 or tuple(sorted(set(quadrature_orders))) != quadrature_orders:
        raise ValueError('E1 quadrature_orders must contain exactly three unique increasing levels.')
    time_x_steps = checks.positive_integer(time_x_steps, 'time_x_steps', minimum=3)
    time_quadrature_order = checks.positive_integer(
        time_quadrature_order, 'time_quadrature_order', minimum=1,
    )
    spatial_max_step = checks.finite_scalar(
        spatial_max_step, 'spatial_max_step', positive=True,
    )
    spatial_quadrature_order = checks.positive_integer(
        spatial_quadrature_order, 'spatial_quadrature_order', minimum=1,
    )
    quadrature_x_steps = checks.positive_integer(
        quadrature_x_steps, 'quadrature_x_steps', minimum=3,
    )
    quadrature_max_step = checks.finite_scalar(
        quadrature_max_step, 'quadrature_max_step', positive=True,
    )
    workers = checks.positive_integer(workers, 'workers', minimum=1)
    if progress is not None and not callable(progress):
        raise TypeError('progress must be callable or None.')

    axis_configs = {
        'time': tuple(
            (time_x_steps, h, time_quadrature_order) for h in time_steps
        ),
        'spatial': tuple(
            (n, spatial_max_step, spatial_quadrature_order) for n in spatial_cells
        ),
        'quadrature': tuple(
            (quadrature_x_steps, quadrature_max_step, q) for q in quadrature_orders
        ),
    }
    unique_configs = tuple(dict.fromkeys(
        config for configs in axis_configs.values() for config in configs
    ))
    tasks = tuple(
        (sequence_id, rho_values, config[0], config[1], config[2])
        for config in unique_configs
    )
    curves = {}
    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            key, curve = _f4_e_rho_curve_task(task)
            curves[key] = curve
            if progress is not None:
                progress(completed, len(tasks), key, dict(curve))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = {
                executor.submit(_f4_e_rho_curve_task, task): (
                    int(task[2]), float(task[3]), int(task[4])
                )
                for task in tasks
            }
            completed = 0
            for future in as_completed(futures):
                key, curve = future.result()
                curves[key] = curve
                completed += 1
                if progress is not None:
                    progress(completed, len(tasks), key, dict(curve))

    rows = {
        axis: tuple(curves[config] for config in configs)
        for axis, configs in axis_configs.items()
    }
    summary = {
        axis: _f4_e_likelihood_stability(values[-2], values[-1])
        for axis, values in rows.items()
    }
    registered = bool(
        sequence_id == F4_E_NUMERICAL_SEQUENCE_ID
        and rho_values == F4_E_RHO_GRID
        and time_steps == F4_E_TIME_STEPS
        and time_x_steps == F4_E_TIME_X_STEPS
        and time_quadrature_order == F4_E_TIME_QUADRATURE_ORDER
        and spatial_cells == F4_E_SPATIAL_CELLS
        and spatial_max_step == F4_E_SPATIAL_MAX_STEP
        and spatial_quadrature_order == F4_E_SPATIAL_QUADRATURE_ORDER
        and quadrature_orders == F4_E_QUADRATURE_ORDERS
        and quadrature_x_steps == F4_E_QUADRATURE_X_STEPS
        and quadrature_max_step == F4_E_QUADRATURE_MAX_STEP
    )
    return {
        'stage': 'E1',
        'sequence_id': sequence_id,
        'rho_values': rho_values,
        'workers': workers,
        'time': rows['time'],
        'spatial': rows['spatial'],
        'quadrature': rows['quadrature'],
        'summary': summary,
        'unique_configurations': unique_configs,
        'matches_registered_design': registered,
        'passes_numerical_refinement': bool(
            registered and all(item['passes'] for item in summary.values())
        ),
        'criteria': {
            'centered_log_likelihood_window': F4_E_LIKELIHOOD_WINDOW,
            'maximum_centered_log_likelihood_change': F4_E_CENTERED_LIKELIHOOD_TOL,
            'requires_unique_same_grid_maximizer': True,
        },
    }

def f4_e_screen_numerics_from_refinement(report):
    """Require the registered E1 evidence and recheck its curves, not just flags.

    Lock N=401, h<=1/768, q=3 after all three E1 comparisons pass. This is
    rho-specific evidence for screening, not a five-parameter convergence claim.
    """
    if not isinstance(report, dict):
        raise TypeError('e1_refinement must be an E1 report dictionary.')
    if (report.get('stage') != 'E1'
            or report.get('sequence_id') != F4_E_NUMERICAL_SEQUENCE_ID
            or report.get('matches_registered_design') is not True
            or report.get('passes_numerical_refinement') is not True
            or tuple(report.get('rho_values', ())) != F4_E_RHO_GRID):
        raise ValueError('E2 requires a passing, registered E1 report on sequence 1211.')
    expected = {
        'time': tuple((201, h, 3) for h in F4_E_TIME_STEPS),
        'spatial': tuple((n, 1.0 / 768.0, 3) for n in F4_E_SPATIAL_CELLS),
        'quadrature': tuple((401, 1.0 / 768.0, q) for q in F4_E_QUADRATURE_ORDERS),
    }
    summaries = {}
    for axis, configurations in expected.items():
        rows = report.get(axis, ())
        if len(rows) != len(configurations):
            raise ValueError(f'E1 {axis} evidence is incomplete.')
        for row, config in zip(rows, configurations):
            actual = (row['x_steps'], row['requested_max_step'],
                      row['likelihood_quadrature_order'])
            if actual != config or tuple(row['rho_values']) != F4_E_RHO_GRID:
                raise ValueError(f'E1 {axis} settings do not match the registered design.')
            total = checks.finite_vector(row['total_log_likelihood'], 'E1 log likelihood')
            centered = checks.finite_vector(row['centered_log_likelihood'], 'E1 centered curve')
            if (total.shape != (41,) or centered.shape != total.shape
                    or not np.allclose(centered, total - np.max(total), atol=1e-12, rtol=0)):
                raise ValueError('E1 centered curve does not match its raw likelihood.')
            maxima = tuple(int(i) for i in np.flatnonzero(
                np.isclose(total, np.max(total), atol=1e-12, rtol=0)))
            if tuple(row['maximizer_indices']) != maxima:
                raise ValueError('E1 grid maximizers do not match its likelihood.')
            error = checks.finite_scalar(row['maximum_abs_mass_error'], 'E1 mass error')
            minimum = checks.finite_scalar(row['minimum_cell_mass'], 'E1 minimum mass')
            if error < 0 or error > 1e-10 or minimum < -1e-14:
                raise ValueError('E1 has invalid probability diagnostics.')
        summaries[axis] = _f4_e_likelihood_stability(rows[-2], rows[-1])
        if not summaries[axis]['passes']:
            raise ValueError(f'E1 {axis} likelihood stability gate failed on recheck.')
    return {
        'x_steps': F4_E_SCREEN_X_STEPS,
        'max_step': F4_E_SCREEN_MAX_STEP,
        'likelihood_quadrature_order': F4_E_SCREEN_QUADRATURE_ORDER,
        'e1_sequence_id': F4_E_NUMERICAL_SEQUENCE_ID,
        'e1_source_commit': report.get('source_commit'),
        'e1_summary': summaries,
        'refinement_scope': 'rho slice on sequence 1211; other parameters are screening only',
    }


def _f4_e_slice_summary(values, total_log_likelihood, truth_value):
    """Discrete maxima and LR-set span; ties and disconnected sets stay visible."""
    grid = checks.finite_vector(values, 'parameter grid', minimum_size=2)
    total = checks.finite_vector(total_log_likelihood, 'slice log likelihood', minimum_size=2)
    truth_value = checks.finite_scalar(truth_value, 'truth value')
    if total.shape != grid.shape or np.any(np.diff(grid) <= 0):
        raise ValueError('Likelihood must match a strictly increasing parameter grid.')
    truth_indices = np.flatnonzero(grid == truth_value)
    if truth_indices.size != 1:
        raise ValueError('The parameter grid must contain exactly one truth node.')
    maximum = float(np.max(total))
    centered = total - maximum
    indices = tuple(int(i) for i in np.flatnonzero(
        np.isclose(total, maximum, atol=1e-12, rtol=0)))
    lr_indices = np.flatnonzero(-2.0 * centered <= F4_E_LR_THRESHOLD)
    return {
        'parameter_values': tuple(float(x) for x in grid),
        'total_log_likelihood': tuple(float(x) for x in total),
        'centered_log_likelihood': tuple(float(x) for x in centered),
        'truth_value': truth_value,
        'truth_index': int(truth_indices[0]),
        'maximizer_indices': indices,
        'maximizer_values': tuple(float(grid[i]) for i in indices),
        'unique_mle': len(indices) == 1,
        'grid_mle': float(grid[indices[0]]) if len(indices) == 1 else None,
        'mle_on_boundary': 0 in indices or grid.size - 1 in indices,
        'mle_on_lower_boundary': 0 in indices,
        'mle_on_upper_boundary': grid.size - 1 in indices,
        'log_likelihood_loss_at_truth': float(-centered[truth_indices[0]]),
        'lr_indices': tuple(int(i) for i in lr_indices),
        'lr_values': tuple(float(grid[i]) for i in lr_indices),
        'lr_span': (float(grid[lr_indices[0]]), float(grid[lr_indices[-1]])),
        'lr_span_width': float(grid[lr_indices[-1]] - grid[lr_indices[0]]),
        'lr_connected_components': int(1 + np.sum(np.diff(lr_indices) > 1)),
        'lr_touches_boundary': bool(lr_indices[0] == 0 or lr_indices[-1] == grid.size - 1),
        'truth_in_lr_set': bool(truth_indices[0] in lr_indices),
    }


def _f4_e_screen_sequence_task(task):
    sequence_id, parameter_grids, x_steps, max_step, order = task
    case = build_f4_b_case()
    sequence = generate_synthetic_sequence(case, sequence_id=sequence_id)
    if not sequence.truth.valid:
        raise RuntimeError(f'F4-E sequence {sequence_id} has raw EM boundary violations; '
                           'do not repair or drop it silently.')
    truth = f4_e_truth_candidate(case)
    slices = {}
    for name, grid in parameter_grids.items():
        candidates = tuple(replace(truth, **{name: value}) for value in grid)
        evaluations = evaluate_f4_e_candidates(
            case, sequence.observations, candidates, x_steps=x_steps,
            max_step=max_step, likelihood_quadrature_order=order,
        )
        summary = _f4_e_slice_summary(
            grid, [item['total_log_likelihood'] for item in evaluations], getattr(truth, name),
        )
        summary.update({
            'log_evidence': tuple(tuple(float(x) for x in item['log_evidence'])
                                  for item in evaluations),
            'maximum_abs_mass_error': max(item['maximum_abs_mass_error'] for item in evaluations),
            'minimum_cell_mass': min(item['minimum_cell_mass'] for item in evaluations),
            'actual_max_step': max(item['actual_max_step'] for item in evaluations),
            'n_steps': max(item['n_steps'] for item in evaluations),
        })
        if summary['maximum_abs_mass_error'] > 1e-10 or summary['minimum_cell_mass'] < -1e-14:
            raise RuntimeError(f'E2 sequence {sequence_id}, parameter {name} failed probability checks.')
        slices[name] = summary
    return {
        'sequence_id': sequence_id,
        'observation_times': tuple(float(x) for x in sequence.observations.times),
        'observations': tuple(float(x) for x in sequence.observations.values),
        'truth_boundary_violations': int(sequence.truth.diagnostics['total_violations']),
        'slices': slices,
    }


def run_f4_e_parameter_screening(
    *, e1_refinement, n_sequences=F4_E_SCREEN_SEQUENCE_COUNT,
    sequence_start=F4_E_SCREEN_SEQUENCE_START, parameter_grids=None,
    x_steps=None, max_step=None, likelihood_quadrature_order=None, workers=1, progress=None,
):
    """E2 one-parameter slices, each with all four nuisance parameters at truth.

    Every candidate replays its own filter. Parallelism is over sequences only;
    result order is always sequence-ID order. Nonregistered small diagnostics are
    allowed in the validation API and labelled; the CLI exposes the locked design.
    """
    locked = f4_e_screen_numerics_from_refinement(e1_refinement)
    n_sequences = checks.positive_integer(n_sequences, 'n_sequences', minimum=1)
    sequence_start = _nonnegative_integer(sequence_start, 'sequence_start')
    workers = checks.positive_integer(workers, 'workers', minimum=1)
    x_steps = checks.positive_integer(locked['x_steps'] if x_steps is None else x_steps,
                                      'x_steps', minimum=3)
    max_step = checks.finite_scalar(locked['max_step'] if max_step is None else max_step,
                                    'max_step', positive=True)
    order = checks.positive_integer(locked['likelihood_quadrature_order']
                                    if likelihood_quadrature_order is None else likelihood_quadrature_order,
                                    'likelihood_quadrature_order', minimum=1)
    if progress is not None and not callable(progress):
        raise TypeError('progress must be callable or None.')
    grids = F4_E_PARAMETER_GRIDS if parameter_grids is None else dict(parameter_grids)
    if set(grids) != set(F4_E_PARAMETER_GRIDS):
        raise ValueError('E2 requires grids for rho, Xi, k, sigma_x and sigma_y.')
    truth = f4_e_truth_candidate()
    validated = {}
    for name in F4_E_PARAMETER_GRIDS:
        grid = checks.finite_vector(grids[name], f'{name} grid', minimum_size=2)
        if np.any(np.diff(grid) <= 0) or np.count_nonzero(grid == getattr(truth, name)) != 1:
            raise ValueError(f'{name} grid must strictly increase and contain the truth node.')
        for value in grid:
            replace(truth, **{name: float(value)})
        validated[name] = tuple(float(x) for x in grid)
    ids = tuple(range(sequence_start, sequence_start + n_sequences))
    tasks = tuple((i, validated, x_steps, max_step, order) for i in ids)
    results = {}
    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            row = _f4_e_screen_sequence_task(task)
            results[row['sequence_id']] = row
            if progress is not None:
                progress(completed, n_sequences, row['sequence_id'], row)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, n_sequences)) as executor:
            futures = [executor.submit(_f4_e_screen_sequence_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()  # failures abort; no selected subset is summarized
                results[row['sequence_id']] = row
                if progress is not None:
                    progress(completed, n_sequences, row['sequence_id'], row)
    rows = tuple(results[i] for i in ids)
    summary = {}
    for name, grid in validated.items():
        slices = tuple(row['slices'][name] for row in rows)
        mles = np.asarray([s['grid_mle'] for s in slices if s['unique_mle']], dtype=float)
        # Raw sums retain the absolute evidence as well as the centered shape;
        # summing centered slices differs only by a candidate-free constant.
        summed = np.sum([s['total_log_likelihood'] for s in slices], axis=0)
        summary[name] = {
            'unique_mle_count': int(mles.size),
            'nonunique_mle_count': int(n_sequences - mles.size),
            'unique_mle_median': float(np.median(mles)) if mles.size else None,
            'unique_mle_q05_q95': tuple(float(x) for x in np.quantile(mles, [.05, .95]))
                                    if mles.size else None,
            'unique_mle_range': (float(np.min(mles)), float(np.max(mles))) if mles.size else None,
            'boundary_mle_frequency': float(np.mean([s['mle_on_boundary'] for s in slices])),
            'lower_boundary_mle_frequency': float(np.mean([s['mle_on_lower_boundary'] for s in slices])),
            'upper_boundary_mle_frequency': float(np.mean([s['mle_on_upper_boundary'] for s in slices])),
            'median_log_likelihood_loss_at_truth': float(np.median(
                [s['log_likelihood_loss_at_truth'] for s in slices])),
            'median_lr_span_width': float(np.median([s['lr_span_width'] for s in slices])),
            'lr_boundary_frequency': float(np.mean([s['lr_touches_boundary'] for s in slices])),
            'disconnected_lr_set_count': sum(s['lr_connected_components'] > 1 for s in slices),
            'aggregate_slice': _f4_e_slice_summary(grid, summed, getattr(truth, name)),
        }
    registered = bool(
        n_sequences == F4_E_SCREEN_SEQUENCE_COUNT and sequence_start == F4_E_SCREEN_SEQUENCE_START
        and validated == F4_E_PARAMETER_GRIDS and x_steps == locked['x_steps']
        and max_step == locked['max_step'] and order == locked['likelihood_quadrature_order'])
    return {
        'stage': 'E2', 'sequence_ids': ids, 'n_sequences': n_sequences, 'workers': workers,
        'parameter_grids': validated,
        'settings': {'x_steps': x_steps, 'max_step': max_step, 'likelihood_quadrature_order': order,
                     'truth_max_step': build_f4_b_case().truth_max_step,
                     'rng_seed_roots': (INITIAL_SEED_ROOT, BROWNIAN_SEED_ROOT, MEASUREMENT_SEED_ROOT)},
        'e1_numerical_lock': locked,
        'matches_registered_design': registered,
        'sequence_results': rows, 'summary': summary,
        'maximum_abs_mass_error': max(s['maximum_abs_mass_error'] for row in rows
                                     for s in row['slices'].values()),
        'minimum_cell_mass': min(s['minimum_cell_mass'] for row in rows
                                 for s in row['slices'].values()),
        'interpretation': {
            'nuisance_parameters': 'held at truth; these are slices, not profiles or joint fits',
            'lr_threshold': F4_E_LR_THRESHOLD,
            'lr_set': 'discrete shape diagnostic; span may be disconnected or support-truncated',
            'lr_is_calibrated_confidence_interval': False,
            'Xi_truth_on_physical_boundary': True,
            'parameter_selection_for_F4_F': 'deferred until E3/E4 and parameter-specific refinement',
        },
    }


def f4_e_screening_inputs(report, *, require_registered=True):
    """Validate E2 evidence before reusing its observations in E3.

    All sequence rows and per-observation likelihood sums are checked, including
    the 40 records not selected for E3. No regeneration or truth-state input.
    """
    if not isinstance(report, dict) or report.get('stage') != 'E2':
        raise ValueError('E3 requires a complete E2 screening report.')
    ids = tuple(_nonnegative_integer(i, 'sequence_id') for i in report['sequence_ids'])
    n = checks.positive_integer(report['n_sequences'], 'n_sequences', minimum=1)
    if len(ids) != n or len(set(ids)) != n or tuple(sorted(ids)) != ids:
        raise ValueError('E2 sequence IDs must be unique, ordered and complete.')
    settings = report['settings']
    cells = checks.positive_integer(settings['x_steps'], 'x_steps', minimum=3)
    step = checks.finite_scalar(settings['max_step'], 'max_step', positive=True)
    order = checks.positive_integer(settings['likelihood_quadrature_order'], 'quadrature order', minimum=1)
    truth = f4_e_truth_candidate()
    if set(report['parameter_grids']) != set(F4_E_PARAMETER_GRIDS):
        raise ValueError('E2 requires all five parameter grids.')
    grids = {}
    for name in F4_E_PARAMETER_GRIDS:
        grid = checks.finite_vector(report['parameter_grids'][name], f'{name} grid', minimum_size=2)
        if np.any(np.diff(grid) <= 0) or np.count_nonzero(grid == getattr(truth, name)) != 1:
            raise ValueError('E2 grids must strictly increase and contain truth.')
        for value in grid:
            replace(truth, **{name: float(value)})
        grids[name] = tuple(float(x) for x in grid)
    declared_settings = (cells, step, order)
    registered = bool(
        report.get('matches_registered_design') is True
        and ids == tuple(range(F4_E_SCREEN_SEQUENCE_START,
                               F4_E_SCREEN_SEQUENCE_START + F4_E_SCREEN_SEQUENCE_COUNT))
        and grids == F4_E_PARAMETER_GRIDS
        and declared_settings == (F4_E_SCREEN_X_STEPS, F4_E_SCREEN_MAX_STEP,
                                   F4_E_SCREEN_QUADRATURE_ORDER)
        and settings.get('truth_max_step') == build_f4_b_case().truth_max_step
        and tuple(settings.get('rng_seed_roots', ())) ==
        (INITIAL_SEED_ROOT, BROWNIAN_SEED_ROOT, MEASUREMENT_SEED_ROOT))
    if report.get('matches_registered_design') is True and not registered:
        raise ValueError('E2 claims a registered design but its settings/IDs/grids differ.')
    if require_registered and not registered:
        raise ValueError('Official E3 requires the registered 50-sequence E2 report.')
    if registered:
        lock = report['e1_numerical_lock']
        if ((lock['x_steps'], lock['max_step'], lock['likelihood_quadrature_order']) != declared_settings
                or lock['e1_sequence_id'] != F4_E_NUMERICAL_SEQUENCE_ID):
            raise ValueError('E2 numerical settings differ from its E1 lock.')
        for axis in ('time', 'spatial', 'quadrature'):
            gate = lock['e1_summary'][axis]
            change = checks.finite_scalar(gate['maximum_centered_log_likelihood_change'], 'E1 change')
            if (gate['passes'] is not True or gate['unique_same_maximizer'] is not True
                    or gate['window'] != F4_E_LIKELIHOOD_WINDOW
                    or gate['tolerance'] != F4_E_CENTERED_LIKELIHOOD_TOL
                    or not 0 <= change <= F4_E_CENTERED_LIKELIHOOD_TOL):
                raise ValueError('E2 does not contain passing registered E1 gates.')
    rows = tuple(report['sequence_results'])
    if len(rows) != n or tuple(row['sequence_id'] for row in rows) != ids:
        raise ValueError('E2 rows must match every declared sequence ID in order.')
    for row in rows:
        if row['truth_boundary_violations'] != 0:
            raise ValueError('E2 has invalid synthetic truth; do not omit the sequence.')
        obs = BM1ObservationSeries(row['observation_times'], row['observations'], 'day')
        if not np.array_equal(obs.times, CORE_OBSERVATION_TIMES):
            raise ValueError('E2 observations do not follow the locked 24-hour point contract.')
        if set(row['slices']) != set(grids):
            raise ValueError('Every E2 sequence must have all five slices.')
        for name, grid in grids.items():
            item = row['slices'][name]
            if tuple(item['parameter_values']) != grid:
                raise ValueError('E2 per-sequence grid differs from the report grid.')
            total = checks.finite_vector(item['total_log_likelihood'], 'E2 likelihood')
            centered = checks.finite_vector(item['centered_log_likelihood'], 'E2 centered likelihood')
            evidence = np.asarray(item['log_evidence'], dtype=float)
            if (total.shape != (len(grid),) or centered.shape != total.shape
                    or evidence.shape != (len(grid), 24) or not np.all(np.isfinite(evidence))
                    or not np.allclose(total, np.sum(evidence, axis=1), atol=1e-10, rtol=0)
                    or not np.allclose(centered, total - np.max(total), atol=1e-10, rtol=0)):
                raise ValueError('E2 likelihood curves disagree with predictive log-evidence.')
            error = checks.finite_scalar(item['maximum_abs_mass_error'], 'E2 mass error')
            minimum = checks.finite_scalar(item['minimum_cell_mass'], 'E2 minimum mass')
            if not 0 <= error <= 1e-10 or minimum < -1e-14:
                raise ValueError('E2 probability diagnostics failed.')
    return {'sequence_ids': ids, 'rows': rows, 'parameter_grids': grids,
            'x_steps': cells, 'max_step': step, 'likelihood_quadrature_order': order,
            'matches_registered_design': registered}


def _f4_e_surface_summary(pair, first_values, second_values, total_log_likelihood):
    """2D grid maxima and conditional ridges; no new statistical acceptance gate."""
    a = checks.finite_vector(first_values, 'first parameter grid', minimum_size=2)
    b = checks.finite_vector(second_values, 'second parameter grid', minimum_size=2)
    total = np.asarray(total_log_likelihood, dtype=float)
    if (total.shape != (a.size, b.size) or not np.all(np.isfinite(total))
            or np.any(np.diff(a) <= 0) or np.any(np.diff(b) <= 0)):
        raise ValueError('A finite surface must match its two increasing parameter grids.')
    truth = f4_e_truth_candidate()
    ia = np.flatnonzero(a == getattr(truth, pair[0]))
    ib = np.flatnonzero(b == getattr(truth, pair[1]))
    if ia.size != 1 or ib.size != 1:
        raise ValueError('E3 subsampled grids must retain both truth nodes.')
    centered = total - np.max(total)
    maxima = tuple(tuple(int(x) for x in ij) for ij in np.argwhere(
        np.isclose(centered, 0., atol=1e-12, rtol=0)))
    boundary = tuple(name for axis, name in enumerate(pair) if any(
        ij[axis] in (0, total.shape[axis] - 1) for ij in maxima))

    def ridge(values, axis):
        # At each fixed value on the other axis, retain every conditional maximum.
        reference = np.max(values, axis=axis, keepdims=True)
        mask = np.isclose(values, reference, atol=1e-12, rtol=0)
        return tuple(tuple(int(j) for j in np.flatnonzero(row)) for row in mask)

    return {
        'pair': tuple(pair), 'first_values': tuple(float(x) for x in a),
        'second_values': tuple(float(x) for x in b),
        'total_log_likelihood': tuple(tuple(float(x) for x in row) for row in total),
        'centered_log_likelihood': tuple(tuple(float(x) for x in row) for row in centered),
        'maximizer_indices': maxima,
        'maximizer_values': tuple((float(a[i]), float(b[j])) for i, j in maxima),
        'unique_mle': len(maxima) == 1,
        'grid_mle': (float(a[maxima[0][0]]), float(b[maxima[0][1]])) if len(maxima) == 1 else None,
        'boundary_dimensions': boundary,
        'truth_indices': (int(ia[0]), int(ib[0])),
        'truth_values': (getattr(truth, pair[0]), getattr(truth, pair[1])),
        'log_likelihood_loss_at_truth': float(-centered[ia[0], ib[0]]),
        'profile_first_centered_log_likelihood': tuple(float(x) for x in np.max(centered, axis=1)),
        'profile_second_centered_log_likelihood': tuple(float(x) for x in np.max(centered, axis=0)),
        'ridge_second_indices_given_first': ridge(total, 1),
        'ridge_first_indices_given_second': ridge(total.T, 1),
    }


def _f4_e_pair_surface_task(task):
    row, pair, first_values, second_values, stride, cells, step, order = task
    case = build_f4_b_case()
    truth = f4_e_truth_candidate(case)
    observations = BM1ObservationSeries(row['observation_times'], row['observations'], 'day')
    candidates = tuple(replace(truth, **{pair[0]: a, pair[1]: b})
                       for a in first_values for b in second_values)
    evaluations = evaluate_f4_e_candidates(case, observations, candidates,
        x_steps=cells, max_step=step, likelihood_quadrature_order=order)
    shape = (len(first_values), len(second_values))
    total = np.asarray([e['total_log_likelihood'] for e in evaluations]).reshape(shape)
    surface = _f4_e_surface_summary(pair, first_values, second_values, total)
    ia, ib = surface['truth_indices']
    discrepancies = (float(np.max(np.abs(total[:, ib] -
                         np.asarray(row['slices'][pair[0]]['total_log_likelihood'])[::stride]))),
                     float(np.max(np.abs(total[ia, :] -
                         np.asarray(row['slices'][pair[1]]['total_log_likelihood'])[::stride]))))
    if max(discrepancies) > 1e-10:
        raise RuntimeError('E3 truth-coordinate cuts do not reproduce E2 slices; '
                           'check observations, history replay and numerical settings.')
    surface.update({
        'sequence_id': row['sequence_id'],
        'e2_truth_cut_maximum_abs_differences': discrepancies,
        'log_evidence': tuple(tuple(float(x) for x in e['log_evidence']) for e in evaluations),
        'candidate_order': 'first parameter outer, second parameter inner',
        'maximum_abs_mass_error': max(e['maximum_abs_mass_error'] for e in evaluations),
        'minimum_cell_mass': min(e['minimum_cell_mass'] for e in evaluations),
        'actual_max_step': max(e['actual_max_step'] for e in evaluations),
        'n_steps': max(e['n_steps'] for e in evaluations),
    })
    if surface['maximum_abs_mass_error'] > 1e-10 or surface['minimum_cell_mass'] < -1e-14:
        raise RuntimeError('E3 surface failed probability diagnostics.')
    return (row['sequence_id'], tuple(pair)), surface


def run_f4_e_pair_surfaces(
    *, e2_screening, n_sequences=F4_E_PAIR_SEQUENCE_COUNT,
    grid_stride=F4_E_PAIR_GRID_STRIDE, workers=1, progress=None,
):
    """Registered E3 surfaces on the first ten E2 records, without new draws.

    The validation API also supports labelled small E2 diagnostics. The CLI
    explicitly requires the official E2 artifact before invoking this runner.
    Independent (sequence,pair) jobs are parallel; candidate histories stay local.
    """
    inputs = f4_e_screening_inputs(e2_screening, require_registered=False)
    n_sequences = checks.positive_integer(n_sequences, 'n_sequences', minimum=1)
    stride = checks.positive_integer(grid_stride, 'grid_stride', minimum=1)
    workers = checks.positive_integer(workers, 'workers', minimum=1)
    if n_sequences > len(inputs['rows']):
        raise ValueError('E3 cannot select more sequences than the supplied E2 report.')
    if progress is not None and not callable(progress):
        raise TypeError('progress must be callable or None.')
    rows = inputs['rows'][:n_sequences]
    grids = {name: values[::stride] for name, values in inputs['parameter_grids'].items()}
    truth = f4_e_truth_candidate()
    for name, values in grids.items():
        if len(values) < 2 or getattr(truth, name) not in values:
            raise ValueError('Subsampled E3 grids must retain at least two nodes and truth.')
    tasks = tuple((row, pair, grids[pair[0]], grids[pair[1]], stride,
                   inputs['x_steps'], inputs['max_step'], inputs['likelihood_quadrature_order'])
                  for row in rows for pair in F4_E_PAIRS)
    surfaces = {}
    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            key, surface = _f4_e_pair_surface_task(task)
            surfaces[key] = surface
            if progress is not None:
                progress(completed, len(tasks), key, surface)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            futures = [executor.submit(_f4_e_pair_surface_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                key, surface = future.result()
                surfaces[key] = surface
                if progress is not None:
                    progress(completed, len(tasks), key, surface)
    pair_results = {}
    for pair in F4_E_PAIRS:
        values = tuple(surfaces[(row['sequence_id'], pair)] for row in rows)
        summed = np.sum([s['total_log_likelihood'] for s in values], axis=0)
        pair_results['__'.join(pair)] = {
            'sequence_surfaces': values,
            'aggregate_surface': _f4_e_surface_summary(pair, grids[pair[0]], grids[pair[1]], summed),
            'nonunique_mle_count': sum(not s['unique_mle'] for s in values),
            'boundary_mle_frequency': {name: float(np.mean(
                [name in s['boundary_dimensions'] for s in values])) for name in pair},
        }
    return {
        'stage': 'E3', 'sequence_ids': tuple(row['sequence_id'] for row in rows),
        'n_sequences': n_sequences, 'pairs': F4_E_PAIRS, 'grid_stride': stride, 'workers': workers,
        'settings': {name: inputs[name] for name in
                     ('x_steps', 'max_step', 'likelihood_quadrature_order')},
        'matches_registered_design': bool(inputs['matches_registered_design']
            and n_sequences == F4_E_PAIR_SEQUENCE_COUNT and stride == F4_E_PAIR_GRID_STRIDE),
        'e2_source_commit': e2_screening.get('source_commit'),
        'observations': tuple({'sequence_id': row['sequence_id'],
            'observation_times': tuple(row['observation_times']), 'values': tuple(row['observations'])}
            for row in rows),
        'pair_results': pair_results,
        'maximum_abs_mass_error': max(s['maximum_abs_mass_error'] for s in surfaces.values()),
        'minimum_cell_mass': min(s['minimum_cell_mass'] for s in surfaces.values()),
        'maximum_e2_truth_cut_difference': max(max(s['e2_truth_cut_maximum_abs_differences'])
                                               for s in surfaces.values()),
        'interpretation': {
            'purpose': 'diagnostic pair surfaces; three other parameters held at truth',
            'ridge_definition': 'all grid maximizers on each conditional row/column',
            'profiles': 'grid maxima within the displayed two-parameter support',
            'aggregate': 'same first ten E2 records only; no additional replication sample',
            'new_recovery_gate': False, 'parameter_selection_for_F4_F': 'deferred to E4',
        },
    }


def run_f4_preparation_checks() -> dict:
    """Independent F4-0 quadrature checks; not a fitted or recursive model.

    Moved intact from root/f4/verify_preparation.py. Continuous reference formulas
    deliberately do not call the model's demand function.
    """
    from scipy.integrate import quad
    from scipy.special import betaln
    from src.params.bm_1 import bayes_update, gaussian_log_likelihood, mass_to_log
    from src.method.beta_mixture import PreparedBetaProjector, beta_cell_average_basis
    from src.method.fokker_planc import bm_1_fokker_planc
    from src.diagnostics.bm_1 import beta_mixture_variance_floor
    def prior_pdf(x):
        if not 0. < x < 1.:
            return 0.
        return np.exp(19*np.log(x)+19*np.log1p(-x)-betaln(20, 20))

    def demand(x):
        # Independent formula for linear f/g, B=U=1/2, k=2, Xi=1.
        return .5+.5*np.tanh(2*(.5-x))

    def posterior_kernel(x):
        return prior_pdf(x)*np.exp(-.5*((.5-demand(x))/.01)**2)

    z_scaled, z_error = quad(posterior_kernel, 0., 1., epsabs=1e-13, epsrel=1e-12, points=[.5])
    v_numerator, v_error = quad(lambda x: (x-.5)**2*posterior_kernel(x), 0., 1.,
                                 epsabs=1e-14, epsrel=1e-11, points=[.5])
    variance = v_numerator/z_scaled
    example = bayes_update(mass_to_log([.6, .4]), gaussian_log_likelihood(.65, [.3, .7], .1))
    bank_example = bayes_update(mass_to_log([.6, .4]), np.log([.2, .8]))

    # Nonmonotone uncertainty: an admissible BM1 example with negative initial
    # variance derivative, obtained from Ito's formula. This is not a forecast.
    def wide_pdf(x):
        return 6*x*(1-x)
    covariance = quad(lambda x: (x-.5)*(demand(x)-.5)*wide_pdf(x), 0, 1,
                      epsabs=1e-13)[0]
    noise_contribution = quad(lambda x: (.1*x*(1-x))**2*wide_pdf(x), 0, 1,
                             epsabs=1e-13)[0]
    projections = []
    for cells in (201, 801):
        dx = 1./cells
        edges = np.linspace(0, 1, cells+1)
        centers = (edges[:-1]+edges[1:])/2
        # Integrate the continuous posterior in each cell, not just its PDF.
        target_mass = np.array([quad(posterior_kernel, a, b, epsabs=1e-15)[0]
                                for a,b in zip(edges[:-1], edges[1:])])/z_scaled
        target = target_mass/dx
        for degree in (30, 38):
            method = bm_1_fokker_planc()
            basis = method._normalise_basis_discretely(method.beta_basis(centers, degree), dx)
            result = PreparedBetaProjector(basis, dx).project(target)
            mass = result['density']*dx
            projections.append({'cells':cells, 'basis':f'legacy_{degree}',
                'mean':float(mass@centers), 'center_variance':float(mass@(centers-.5)**2),
                **result['diagnostics']})
        # Illustrative, fixed dictionary only; no final F4-D basis selection.
        shapes = np.array([20., 200., 1000., 2000.])
        basis = beta_cell_average_basis(edges, shapes, shapes)
        result = PreparedBetaProjector(basis, dx).project(target)
        mass = result['density']*dx
        projections.append({'cells':cells, 'basis':'illustration_symmetric_20_200_1000_2000',
            'mean':float(mass@centers), 'center_variance':float(mass@(centers-.5)**2),
            **result['diagnostics']})
    return {
        'scope':'F4-0 static/preparatory checks only; no recursive filter, parameter recovery or week forecast',
        'source_base':'5a4793cd57d70a46beae2a6c758b4cd4e0556536',
        'single_measurement':{
            'prior':'Beta(20,20)', 'B':.5, 'U':.5, 'Y':.5, 'k':2., 'Xi':1., 'sigma_y':.01,
            'continuous_posterior_mean':.5, 'continuous_posterior_variance':variance,
            'gaussian_predictive_density':float(z_scaled/(np.sqrt(2*np.pi)*.01)),
            'quad_absolute_error_estimates':[z_error, v_error],
            'continuous_variance_floor_30':float(beta_mixture_variance_floor(.5,30)),
            'continuous_variance_floor_38':float(beta_mixture_variance_floor(.5,38)),
        },
        'two_node_bayes_example':{'posterior':example.mass.tolist(), 'log_evidence':example.log_evidence},
        'two_hypothesis_algebra_example':{'posterior':bank_example.mass.tolist(),
                                         'log_evidence':bank_example.log_evidence},
        'nonmonotone_variance_example':{'prior':'Beta(2,2)', 'rho':1.,'B':.5,'U':.5,
            'k':2.,'Xi':1.,'sigma_x':.1,'initial_variance':.05,
            'two_covariance':2*covariance,'expected_diffusion_squared':noise_contribution,
            'initial_variance_derivative':2*covariance+noise_contribution},
        'static_projection':projections,
        'dictionary_warning':'Symmetric four-component illustration only; shifted, bimodal and boundary posteriors unvalidated.',
        'variance_warning':'Continuous beta variance floor is not an exact finite-grid theorem; cell-center variances are labeled separately.',
    }
