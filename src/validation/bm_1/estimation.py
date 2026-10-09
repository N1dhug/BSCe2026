"""Reusable synthetic/numerical reference calculations; retired campaigns use original sources."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from contextlib import nullcontext
from multiprocessing import get_context
import operator
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp
from src.model.bm_1 import bm_1
from src.method.beta_mixture import beta_cell_average_basis
from src.method.bm_1_estimation import BM1FVFilterResult, run_fv_filter, parameter_bank_candidate, parameter_bank_weights, run_mixture_filter
from src.params.bm_1 import BM1ObservationSeries
from src.method.bm_1_observation import mass_to_log
from src.tools.input_generator.input_builder import input_builder
from src.validation import estimation as checks
from src.validation.bm_1 import cases

CORE_PRICE_EDGES = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=float)

CORE_PRICE_VALUES = np.array([0.65, 0.35, 0.65], dtype=float)

CORE_OBSERVATION_TIMES = np.arange(1, 25, dtype=float) / 24.0

INITIAL_SEED_ROOT = 4101

BROWNIAN_SEED_ROOT = 4102

MEASUREMENT_SEED_ROOT = 4103

F4_E_RHO_GRID = tuple(0.5 + 0.025 * j for j in range(41))

F4_E_LIKELIHOOD_WINDOW = 5.0

F4_E_PARAMETER_GRIDS = {
    'rho': F4_E_RHO_GRID,
    'Xi': tuple(0.025 * j for j in range(41)),
    'k': tuple(0.5 + 0.075 * j for j in range(41)),
    'sigma_x': tuple(0.5 * 2.0 ** ((j - 20) / 10.0) for j in range(41)),
    'sigma_y': tuple(0.03 * 2.0 ** ((j - 20) / 10.0) for j in range(41)),
}

F4_F_REFINEMENT_IDS = (1263, 1264, 1265)

F4_F_LOG_ODDS_TOL = 0.10

F4_F_WEIGHT_TV_TOL = 0.01

F4_F_EVIDENCE_TOL = 0.10

F4_F_STATE_L1_TOL = 0.02

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
    if cell == mass.size:
        # Validated unit mass can accumulate to just below one. Use the
        # occupied support's upper edge in that roundoff-sized tail; do not
        # change the masses or place a quantile inside trailing empty cells.
        return float(edges[np.flatnonzero(mass > 0.0)[-1] + 1])
    before = 0.0 if cell == 0 else float(cumulative[cell - 1])
    cell_mass = float(mass[cell])
    if cell_mass <= 0.0:
        raise FloatingPointError('Quantile landed in a zero-mass cell; inspect probability mass.')
    fraction = (probability - before) / cell_mass
    return float(edges[cell] + fraction * (edges[cell + 1] - edges[cell]))

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
        keep_history=False,
    )
    log_evidence = filtered.log_evidence[filtered.observed].copy()
    log_evidence.setflags(write=False)
    return {
        'candidate': candidate,
        'total_log_likelihood': float(np.sum(log_evidence)),
        'log_evidence': log_evidence,
        'maximum_abs_mass_error': filtered.maximum_abs_mass_error,
        'minimum_cell_mass': filtered.minimum_cell_mass,
        'x_steps': x_steps,
        'max_step': max_step,
        'actual_max_step': filtered.actual_max_step,
        'n_steps': filtered.n_steps,
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
    if observations.time_unit != 'day':
        raise ValueError('Synthetic F4 likelihood observations must use time_unit=day.')
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

def f4_f_parameter_prior(values, name, coordinate):
    """Declared discrete prior, or exact bin masses of a continuous prior.

    Midpoint bins have support exactly [first node, last node]. Refinements of
    continuous priors preserve that measure, including half-width end bins.
    The original F0 uniform discrete prior remains a separate primary prior.
    """
    values = checks.increasing_grid(values, 'parameter values')
    if np.any(values <= 0.0) or coordinate not in ('rho', 'sigma_x'):
        raise ValueError('Prior nodes must be positive rho or sigma_x values.')
    edges = np.r_[values[0], .5 * (values[:-1] + values[1:]), values[-1]]
    if name == 'discrete_uniform':
        mass = np.ones(values.size)
    elif name == 'uniform_parameter':
        mass = np.diff(edges)
    elif name == 'uniform_capacity' and coordinate == 'rho':
        mass = -np.diff(1.0 / edges)
    elif name == 'uniform_log' and coordinate == 'sigma_x':
        mass = np.diff(np.log(edges))
    elif name == 'wrong_concentrated':
        # Deliberately wrong but positive everywhere; a zero-prior experiment
        # would permanently exclude hypotheses and is already covered by F0.
        center = .6 if coordinate == 'rho' else .25
        scaled = (np.log(values) - np.log(center)) / .20
        logs = -.5 * scaled**2
        return np.exp(logs - logsumexp(logs))
    else:
        raise ValueError('Unknown prior family for this coordinate.')
    return mass / np.sum(mass)

def _f4_f_numerics(cells=401, step=1/768, order=3, backend='fv'):
    return dict(x_steps=checks.positive_integer(cells, 'cells', minimum=3),
                max_step=checks.finite_scalar(step, 'max_step', positive=True),
                likelihood_quadrature_order=checks.positive_integer(order, 'quadrature order'),
                backend=backend)

def _f4_f_case(coordinate, truth):
    if coordinate not in ('rho', 'sigma_x'):
        raise ValueError('Unknown bank coordinate.')
    return build_f4_b_case(**({'C': 1.0 / truth} if coordinate == 'rho' else {'sigma_x': truth}))

_F4_F_PROJECTORS = {}

def _f4_f_projector(cells, model):
    """Process-local fixed geometry only; no observation or candidate result cache."""
    from src.method.beta_mixture import PreparedBetaProjector
    from src.validation.bm_1.mixture_estimation import _shape_aware_dictionary
    if cells not in _F4_F_PROJECTORS:
        edges, _, dx = model.fokker_planc_reference.grid(cells, bounds=(0.0, 1.0))
        basis = _shape_aware_dictionary(edges, .0125)['basis']
        _F4_F_PROJECTORS[cells] = PreparedBetaProjector(basis, dx)
    return _F4_F_PROJECTORS[cells]

def _f4_f_candidate_replay(task):
    """Fresh complete conditional replay; truth is never an estimator input."""
    from time import perf_counter
    from src.method.bm_1_estimation import _observation_demand
    group_index, node_index, coordinate, node, case, observations, settings = task
    started = perf_counter()
    model, base = case.build_model()
    parameters, noise = parameter_bank_candidate(base, case.sigma_x, coordinate, node)
    initial = beta_initial_mass(case, model, settings['x_steps'])
    options = dict(breakpoints=case.breakpoints, max_step=settings['max_step'],
                   likelihood_quadrature_order=settings['likelihood_quadrature_order'])
    if settings['backend'] == 'fv':
        filtered = run_fv_filter(model, parameters, noise, case.sigma_y, initial,
                                 case.horizon[0], observations, **options)
        mass_error = filtered.maximum_abs_mass_error
        minimum = filtered.minimum_cell_mass
        actual_step = max(d['actual_max_step'] for d in filtered.prediction_diagnostics)
        fallbacks = 0
    elif settings['backend'] == 'mixture':
        filtered = run_mixture_filter(model, parameters, noise, case.sigma_y, initial,
                                     case.horizon[0], observations,
                                     projector=_f4_f_projector(settings['x_steps'], model),
                                     cfl=.4, **options)
        diagnostics = filtered.prediction_diagnostics
        projections = (filtered.initial_projection_diagnostics,) + filtered.posterior_projection_diagnostics
        mass_error = max(max(d['maximum_abs_raw_prediction_mass_error'],
                             d['maximum_abs_projected_mass_error']) for d in diagnostics)
        mass_error = max(mass_error, max(d['mass_error'] for d in projections))
        minimum = min(min(d['minimum_raw_prediction'] for d in diagnostics),
                      float(np.min(filtered.posterior_mass)))
        actual_step = max(d['maximum_internal_step'] for d in diagnostics)
        fallbacks = sum(d['projection_fallback_count'] for d in diagnostics)
        fallbacks += sum(int(d['used_fallback']) for d in projections)
    else:
        raise ValueError('Unknown bank backend.')
    centers = filtered.x_grid
    # Mid-cell demand means are diagnostics; evidence uses the declared cell
    # quadrature in bm1_observation_update, never this mean approximation.
    demands = np.stack([_observation_demand(model, parameters, float(t), centers,
                                          observations.price_side, case.horizon[0])
                        for t in observations.times])
    return group_index, node_index, dict(
        conditional_log_evidence=filtered.log_evidence,
        predicted_mass=filtered.predicted_mass, posterior_mass=filtered.posterior_mass,
        x_edges=filtered.x_edges, x_grid=centers,
        predicted_demand_mean=np.sum(demands * filtered.predicted_mass, axis=1),
        maximum_abs_mass_error=float(mass_error), minimum_cell_mass=float(minimum),
        actual_max_step=float(actual_step), fallbacks=int(fallbacks),
        elapsed_seconds=perf_counter() - started)

def _f4_f_reduce_conditionals(group, conditionals, prior_name='discrete_uniform'):
    """Reduce independent replays with previous weights for predictive quantities."""
    values = np.asarray(group['values'])
    observed = group['observations'].observed
    scores = np.stack([c['conditional_log_evidence'] for c in conditionals], axis=1)
    initial = mass_to_log(f4_f_parameter_prior(values, prior_name, group['coordinate']))
    reduced = parameter_bank_weights(scores, initial, observed)
    weights, old_weights = np.exp(reduced['log_weights']), np.exp(reduced['prior_log_weights'])
    pred = np.stack([c['predicted_mass'] for c in conditionals], axis=1)
    post = np.stack([c['posterior_mass'] for c in conditionals], axis=1)
    predicted = np.einsum('kh,khx->kx', old_weights, pred)
    posterior = np.einsum('kh,khx->kx', weights, post)
    checks.probability_rows(predicted, 'bank predicted mass')
    checks.probability_rows(posterior, 'bank posterior mass')
    mean = weights @ values
    variance = np.sum(weights * (values[None, :] - mean[:, None])**2, axis=1)
    cdf = np.cumsum(weights, axis=1)
    intervals = [[float(values[min(np.searchsorted(row, p), values.size-1)])
                  for p in (.025, .975)] for row in cdf]
    conditional_predicted_mean = np.einsum('khx,x->kh', pred, conditionals[0]['x_grid'])
    conditional_posterior_mean = np.einsum('khx,x->kh', post, conditionals[0]['x_grid'])
    evidence_prefix = np.cumsum(np.where(observed, reduced['log_evidence'], 0.0))
    return dict(
        coordinate=group['coordinate'], values=values.tolist(), prior_name=prior_name,
        initial_log_weights=initial.tolist(), log_weights=reduced['log_weights'].tolist(),
        conditional_log_evidence=scores.tolist(), bank_log_evidence=reduced['log_evidence'].tolist(),
        cumulative_log_evidence=evidence_prefix.tolist(), parameter_mean=mean.tolist(),
        parameter_sd=np.sqrt(variance).tolist(), equal_tail_95=intervals,
        effective_hypotheses=np.exp(-np.sum(np.where(weights > 0, weights * reduced['log_weights'], 0), axis=1)).tolist(),
        endpoint_weight=(weights[:, 0] + weights[:, -1]).tolist(),
        map_values=[values[row == np.max(row)].tolist() for row in reduced['log_weights']],
        capacity_mean=(weights @ (1.0 / values)).tolist() if group['coordinate'] == 'rho' else None,
        conditional_predicted_mean=conditional_predicted_mean.tolist(),
        conditional_posterior_mean=conditional_posterior_mean.tolist(),
        predicted_mean=(predicted @ conditionals[0]['x_grid']).tolist(),
        posterior_mean=(posterior @ conditionals[0]['x_grid']).tolist(),
        predictive_demand_mean=np.sum(old_weights * np.stack(
            [c['predicted_demand_mean'] for c in conditionals], axis=1), axis=1).tolist(),
        maximum_abs_mass_error=max(c['maximum_abs_mass_error'] for c in conditionals),
        minimum_cell_mass=min(c['minimum_cell_mass'] for c in conditionals),
        actual_max_step=max(c['actual_max_step'] for c in conditionals),
        fallbacks=sum(c['fallbacks'] for c in conditionals),
        conditional_seconds=sum(c['elapsed_seconds'] for c in conditionals),
        # Kept in memory for paired comparisons, omitted from sensitivity JSON.
        _predicted_mass=predicted, _posterior_mass=posterior,
        _x_edges=conditionals[0]['x_edges'])

def _f4_f_group(coordinate, sequence_id, truth, values, settings, label):
    case = _f4_f_case(coordinate, truth)
    sequence = generate_synthetic_sequence(case, sequence_id=sequence_id)
    if not sequence.truth.valid:
        raise RuntimeError(f'F4-F truth failed at ID{sequence_id}; no repair or dropped sequence.')
    return dict(coordinate=coordinate, sequence_id=int(sequence_id), truth=float(truth),
                values=tuple(values), settings=settings, label=label,
                case=case, observations=sequence.observations,
                truth_state=sequence.truth.state[1:].tolist())

def _f4_f_public_row(group, reduced, *, keep_mass=False):
    row = {k: v for k, v in reduced.items() if not k.startswith('_')}
    row.update(sequence_id=group['sequence_id'], truth=group['truth'],
               settings=group['settings'], label=group['label'],
               observation_times=group['observations'].times.tolist(),
               observations=group['observations'].values.tolist(),
               truth_state=group['truth_state'])
    if keep_mass:
        row.update(x_edges=reduced['_x_edges'].tolist(),
                   marginal_predicted_mass=reduced['_predicted_mass'].tolist(),
                   marginal_posterior_mass=reduced['_posterior_mass'].tolist())
    return row

def _f4_f_execute_groups(groups, workers, progress=None, *, priors=False, keep_mass=False):
    """A persistent Windows-compatible spawn pool over complete hypotheses.

    Results are placed by group/node index, not completion order. Each replay
    carries its own conditional posterior through the observation chronology.
    """
    workers = checks.positive_integer(workers, 'workers')
    tasks, exact_replays, consumers = [], {}, {}
    for i, g in enumerate(groups):
        case_key = tuple((name, np.asarray(getattr(g['case'], name)).dtype.str,
                          np.asarray(getattr(g['case'], name)).shape,
                          np.asarray(getattr(g['case'], name)).tobytes())
                         for name in g['case'].__dataclass_fields__)
        obs = g['observations']
        observation_key = (obs.time_unit, obs.observation_kind, obs.price_side,
                           obs.times.tobytes(), obs.values.tobytes(), obs.observed.tobytes())
        for j, value in enumerate(g['values']):
            # No rounding or approximate match. Only the same fixed-candidate
            # replay may be shared across grids/priors; never another candidate's
            # conditional posterior. Neither parameter grid nor prior enters it.
            key = (g['coordinate'], float(value), case_key, observation_key,
                   tuple(sorted(g['settings'].items())))
            if key not in exact_replays:
                exact_replays[key] = (i, j)
                consumers[(i, j)] = []
                tasks.append((i, j, g['coordinate'], float(value), g['case'], obs, g['settings']))
            consumers[exact_replays[key]].append((i, j))
    pending = {i: [None] * len(g['values']) for i, g in enumerate(groups)}
    rows = [None] * len(groups)
    manager = (ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'))
               if workers > 1 else nullcontext(None))
    with manager as pool:
        if pool is None:
            results = map(_f4_f_candidate_replay, tasks)
        else:
            futures = {pool.submit(_f4_f_candidate_replay, t): (t[0], t[1]) for t in tasks}

            def completed_results():
                for future in as_completed(futures):
                    group_index, node_index = futures.pop(future)
                    try:
                        result = future.result()
                    except (RuntimeError, ValueError, FloatingPointError) as exc:
                        g = groups[group_index]
                        raise RuntimeError(f"Conditional replay failed: ID{g['sequence_id']}, "
                                           f"{g['coordinate']}={g['values'][node_index]}, {g['label']}: {exc}") from exc
                    yield result

            results = completed_results()
        for completed, (i, j, conditional) in enumerate(results, 1):
            for destination, node_index in consumers[(i, j)]:
                pending[destination][node_index] = conditional
                if all(c is not None for c in pending[destination]):
                    group = groups[destination]
                    reduced = _f4_f_reduce_conditionals(group, pending[destination])
                    row = _f4_f_public_row(group, reduced, keep_mass=keep_mass)
                    if priors:
                        names = ['uniform_parameter', 'wrong_concentrated']
                        names.append('uniform_capacity' if group['coordinate'] == 'rho' else 'uniform_log')
                        row['prior_sensitivity'] = {name: _f4_f_public_row(
                            group, _f4_f_reduce_conditionals(group, pending[destination], name)) for name in names}
                    rows[destination] = row
                    del pending[destination]
            if progress is not None:
                progress(dict(completed=completed, total=len(tasks), sequence_id=groups[i]['sequence_id'],
                              coordinate=groups[i]['coordinate'], label=groups[i]['label']))
    return rows

def _f4_f_weight_comparison(middle, fine):
    """Probability and log-odds errors at *every* observation prefix."""
    if middle['values'] != fine['values'] or middle['observations'] != fine['observations']:
        raise ValueError('Numerical comparisons require identical nodes and observations.')
    a = np.cumsum(np.asarray(middle['conditional_log_evidence']), axis=0)
    b = np.cumsum(np.asarray(fine['conditional_log_evidence']), axis=0)
    errors = b - a
    # max pairwise log-odds error = range of the per-node cumulative errors.
    all_odds = float(np.max(np.ptp(errors, axis=1)))
    # E1's existing five-log-unit likelihood window, now at every prefix and
    # taking the union of both resolutions. All nodes remain in Bayes updates,
    # the TV/evidence gates, saved curves, and the all-node diagnostic.
    relevant = ((a >= np.max(a, axis=1)[:, None] - F4_E_LIKELIHOOD_WINDOW)
                | (b >= np.max(b, axis=1)[:, None] - F4_E_LIKELIHOOD_WINDOW))
    odds = max(float(np.ptp(errors[k, keep])) for k, keep in enumerate(relevant))
    tv = float(np.max(.5 * np.sum(np.abs(np.exp(middle['log_weights']) - np.exp(fine['log_weights'])), axis=1)))
    evidence = float(np.max(np.abs(np.asarray(middle['cumulative_log_evidence'])
                                  - np.asarray(fine['cumulative_log_evidence']))))
    mean = float(np.max(np.abs(np.asarray(middle['parameter_mean']) - np.asarray(fine['parameter_mean']))))
    passes = (odds <= F4_F_LOG_ODDS_TOL and tv <= F4_F_WEIGHT_TV_TOL
              and evidence <= F4_F_EVIDENCE_TOL)
    return dict(maximum_log_odds_error=odds, maximum_weight_tv=tv,
                maximum_all_node_log_odds_error=all_odds,
                maximum_cumulative_evidence_error=evidence, maximum_parameter_mean_change=mean,
                passes=bool(passes))

def _f4_f_probability_gate(rows):
    return bool(rows) and all(
        np.isfinite(r['maximum_abs_mass_error']) and 0 <= r['maximum_abs_mass_error'] <= 1e-10
        and np.isfinite(r['minimum_cell_mass']) and r['minimum_cell_mass'] >= 0
        and r['fallbacks'] == 0 for r in rows)

def _f4_f_report_header(stage, workers):
    case = build_f4_b_case()
    return dict(stage=stage, registered_design=True, workers=workers,
                observation_contract=dict(time_unit='day', observation_kind='point', price_side='right'),
                inputs=dict(baseline=case.baseline, price_edges=case.price_edges.tolist(),
                            price_values=case.price_values.tolist(), state_response='1-2*x', price_response='1-2*u'),
                truth_settings=dict(initial_beta=[20., 20.], truth_max_step=case.truth_max_step,
                                    seed_roots=[INITIAL_SEED_ROOT, BROWNIAN_SEED_ROOT, MEASUREMENT_SEED_ROOT]),
                fixed_parameters=dict(k=2., Xi=1., sigma_y=.03),
                criteria=dict(maximum_log_odds_error=F4_F_LOG_ODDS_TOL,
                              log_likelihood_window=F4_E_LIKELIHOOD_WINDOW,
                              maximum_weight_tv=F4_F_WEIGHT_TV_TOL,
                              maximum_cumulative_evidence_error=F4_F_EVIDENCE_TOL,
                              maximum_state_l1=F4_F_STATE_L1_TOL,
                              maximum_abs_mass_error=1e-10, no_fallbacks=True))

def _f4_f_refinement_configs(coordinate):
    settings = [_f4_f_numerics(), _f4_f_numerics(step=1/384), _f4_f_numerics(step=1/1536),
                _f4_f_numerics(cells=201), _f4_f_numerics(cells=801),
                _f4_f_numerics(order=1), _f4_f_numerics(order=5)]
    if coordinate == 'sigma_x':
        # Development attempt A failed for low-noise hypotheses. Keep those
        # original controls and add a declared finer candidate, not a relaxed gate.
        settings += [_f4_f_numerics(3201, 1/1536), _f4_f_numerics(3201, 1/3072),
                     _f4_f_numerics(1601, 1/1536), _f4_f_numerics(6401, 1/1536),
                     _f4_f_numerics(3201, 1/1536, 5)]
    return settings
