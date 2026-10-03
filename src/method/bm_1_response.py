"""H development: one shared curve pair fitted by pooled predictive likelihood.

Only declared TRAIN records enter the objective. Every new candidate replays
every complete record from its declared prior; conditional states are never
reused across candidates. Records must represent independent initializations,
or an entire contiguous stream, rather than daily resets of the same stream.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from src.method.bm_1_estimation import run_fv_filter
from src.model.bm_1 import bm_1
from src.params.bm_1 import BM1ObservationSeries
from src.params.bm_1_response import BM1PiecewiseResponse
from src.solver.finite_volume import finite_volume_1d_cached
from src.validation import estimation as checks
from src.validation.model.bm import bm_validation


@dataclass(frozen=True)
class CurveTrainingRecord:
    """Normalized point observations and fixed inputs for one training stream.

    Input/initial-law callbacks are pure frozen study inputs. The caller must
    also freeze population, normalization and chronology before constructing
    these records. This object cannot infer those decisions from observations.
    """
    record_id: str
    role: str
    baseline: object
    horizon: object
    parameters: object
    sigma_x: float
    sigma_y: float
    initial_mass_builder: object
    observations: BM1ObservationSeries
    breakpoints: object = None

    def __post_init__(self):
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError('Declare a nonempty training record id.')
        if self.role != 'TRAIN':
            raise ValueError('Curve fitting accepts TRAIN only; keep validation/holdout separate.')
        if not callable(self.baseline) or not callable(self.initial_mass_builder):
            raise TypeError('Baseline and initial-law builder must be callable.')
        horizon = checks.increasing_grid(self.horizon, 'horizon')
        if horizon.size != 2:
            raise ValueError('horizon needs exactly start and end.')
        if not isinstance(self.observations, BM1ObservationSeries):
            raise TypeError('observations must be BM1ObservationSeries.')
        if (self.observations.times[0] <= horizon[0]
                or self.observations.times[-1] > horizon[-1]
                or not np.any(self.observations.observed)):
            raise ValueError('Training observations must lie in (start,end] and contain data.')
        parameters = tuple(bm_validation.parameters(self.parameters))
        sigma_x = checks.finite_scalar(self.sigma_x, 'sigma_x')
        sigma_y = checks.finite_scalar(self.sigma_y, 'sigma_y', positive=True)
        if sigma_x < 0.:
            raise ValueError('sigma_x must be nonnegative.')
        breaks = checks.finite_vector([] if self.breakpoints is None else self.breakpoints,
                                     'breakpoints', minimum_size=0)
        if np.any(breaks <= horizon[0]) or np.any(breaks >= horizon[-1]):
            raise ValueError('Breakpoints must lie strictly within the training stream.')
        breaks = np.unique(breaks); breaks.setflags(write=False)
        object.__setattr__(self, 'horizon', tuple(horizon))
        object.__setattr__(self, 'parameters', parameters)
        object.__setattr__(self, 'sigma_x', sigma_x)
        object.__setattr__(self, 'sigma_y', sigma_y)
        object.__setattr__(self, 'breakpoints', breaks)


def curve_pair_from_logits(logits, pieces):
    """Positive drops summing to 2; one reference logit removes redundancy."""
    pieces = checks.positive_integer(pieces, 'pieces', minimum=2)
    if pieces not in (2, 4):
        raise ValueError('The declared H families have two or four pieces.')
    logits = checks.finite_vector(logits, 'curve logits')
    if logits.size != 2 * (pieces - 1) or np.any(np.abs(logits) > 10.):
        raise ValueError('One independent drop-logit per inner knot and curve is required.')
    curves = []
    for row in logits.reshape(2, pieces - 1):
        drops = 2. * softmax(np.r_[row, 0.])
        inner = 1. - np.cumsum(drops[:-1])
        curves.append(BM1PiecewiseResponse.from_inner_values(inner, pieces=pieces))
    return tuple(curves)


def curve_pair_logits(f, g, *, pieces):
    """Lift a strictly monotone simpler fit to the finer family's start."""
    pieces = checks.positive_integer(pieces, 'pieces', minimum=2)
    if pieces not in (2,4):
        raise ValueError('The declared H families have two or four pieces.')
    rows = []
    for curve in (f, g):
        drops = -np.diff(curve(np.linspace(0., 1., pieces + 1)))
        if np.any(drops <= 0.):
            raise ValueError('Finite drop logits require strictly positive drops.')
        rows.extend(np.log(drops[:-1]) - np.log(drops[-1]))
    return np.asarray(rows)


def replay_curve_record(record, f, g, *, cells=401, max_step=1/768,
                        quadrature_order=3, keep_history=False):
    """One complete fresh replay, shared by fitting and full-history validation.

    B/U callbacks already need NumPy broadcasting for the BM1 forward output.
    Vectorized input sampling retains the identical solver grid and exact cache
    keys. It never shares an operator or posterior across curve candidates.
    """
    if not isinstance(record, CurveTrainingRecord):
        raise TypeError('A declared curve training/development record is required.')
    cells = checks.positive_integer(cells, 'cells', minimum=3)
    model = bm_1(record.baseline, record.baseline, record.horizon, 2)
    model.fokker_planc_reference = finite_volume_1d_cached(vectorized_inputs=True)
    edges, _, _ = model.fokker_planc_reference.grid(cells, bounds=(0., 1.))
    initial = checks.probability_mass(record.initial_mass_builder(edges.copy()), 'initial mass')
    if initial.size != cells:
        raise ValueError('Initial-law builder must supply one mass per requested cell.')
    parameters = [record.parameters[0], f, g, *record.parameters[3:]]
    return run_fv_filter(model, parameters, record.sigma_x, record.sigma_y,
        initial, record.horizon[0], record.observations, breakpoints=record.breakpoints,
        max_step=max_step, likelihood_quadrature_order=quadrature_order,
        keep_history=keep_history)


def pooled_curve_log_likelihood(records, f, g, *, cells=401, max_step=1/768,
                                quadrature_order=3):
    """Whole-record pooled score, with one identical f/g pair in every replay."""
    records = tuple(records)
    if (not records or any(not isinstance(r, CurveTrainingRecord) for r in records)
            or len({r.record_id for r in records}) != len(records)
            or len({r.observations.time_unit for r in records}) != 1):
        raise ValueError('Declare distinct TRAIN records with one common time unit.')
    outputs = []
    for record in records:
        result = replay_curve_record(record, f, g, cells=cells, max_step=max_step,
                                     quadrature_order=quadrature_order)
        if result.maximum_abs_mass_error > 1e-10 or result.minimum_cell_mass < 0.:
            raise RuntimeError('Curve candidate probability control failed; no candidate/record dropped.')
        outputs.append(dict(record_id=record.record_id, log_likelihood=result.total_log_likelihood,
                            log_evidence=result.log_evidence, observed=result.observed,
                            final_step=result.final_step,
                            maximum_abs_mass_error=result.maximum_abs_mass_error,
                            minimum_cell_mass=result.minimum_cell_mass))
    return dict(log_likelihood=sum(o['log_likelihood'] for o in outputs), records=tuple(outputs))


def fit_shared_curve_pair(records, *, pieces, starts, regularization,
                          max_evaluations, max_iterations=40, logit_bound=4.,
                          cells=401, max_step=1/768, quadrature_order=3, progress=None):
    """Bounded development optimizer, with one total distinct-candidate budget.

    Regularization is lambda times squared inner-knot departure from 1-2z.
    Logit bounds exclude exact flat pieces; report that approximation explicitly.
    Budget exhaustion preserves the best evaluated candidate and is not optimizer
    success. No exception from a model/probability failure is converted to a cost.
    """
    # Validate the family and all starts before doing any likelihood work.
    pieces = checks.positive_integer(pieces, 'pieces', minimum=2)
    curve_pair_from_logits(np.zeros(2*(pieces-1)), pieces)
    if np.iscomplexobj(starts):
        raise ValueError('starts must be real.')
    starts = np.asarray(starts, dtype=float)
    if (starts.ndim != 2 or not starts.shape[0] or starts.shape[1] != 2*(pieces-1)
            or np.any(~np.isfinite(starts))):
        raise ValueError('Declare a finite nonempty set of curve-logit starts.')
    regularization = checks.finite_scalar(regularization, 'regularization')
    bound = checks.finite_scalar(logit_bound, 'logit_bound', positive=True)
    if regularization < 0. or bound > 10. or np.any(np.abs(starts) > bound):
        raise ValueError('Nonnegative regularization and starts within a logit bound <=10 are required.')
    budget = checks.positive_integer(max_evaluations, 'max_evaluations')
    max_iterations = checks.positive_integer(max_iterations, 'max_iterations')
    records = tuple(records)
    trace, memo, results = [], {}, []
    best = None
    begun = perf_counter()

    class BudgetReached(Exception):
        pass

    def evaluate(logits):
        nonlocal best
        key = tuple(float(v) for v in logits)
        if key in memo:
            return memo[key]
        if len(trace) >= budget:
            raise BudgetReached()
        f, g = curve_pair_from_logits(logits, pieces)
        replay = pooled_curve_log_likelihood(records, f, g, cells=cells,
            max_step=max_step, quadrature_order=quadrature_order)
        linear = 1. - 2.*f.knots[1:-1]
        penalty = regularization * sum(np.sum((curve.values[1:-1]-linear)**2) for curve in (f,g))
        objective = -replay['log_likelihood'] + penalty
        if not np.isfinite(objective):
            raise RuntimeError('Pooled curve objective must be finite.')
        memo[key] = float(objective)
        trace.append(dict(logits=list(key), log_likelihood=replay['log_likelihood'],
                          penalty=float(penalty), objective=float(objective),
                          record_log_likelihoods={o['record_id']:o['log_likelihood'] for o in replay['records']},
                          maximum_abs_mass_error=max(o['maximum_abs_mass_error'] for o in replay['records'])))
        if best is None or objective < best['objective']:
            best = dict(logits=np.asarray(logits).copy(), f=f, g=g, objective=float(objective),
                        log_likelihood=replay['log_likelihood'], replay=replay)
        if progress is not None and (len(trace) == 1 or len(trace)%10 == 0):
            progress(len(trace), budget)
        return float(objective)

    exhausted = False
    attempted = 0
    for start in starts:
        attempted += 1
        try:
            result = minimize(evaluate, start, method='L-BFGS-B', bounds=[(-bound,bound)]*start.size,
                options=dict(maxiter=max_iterations, maxfun=budget, ftol=1e-9, gtol=1e-5))
            results.append(dict(success=bool(result.success), message=str(result.message),
                                objective=float(result.fun), logits=result.x.tolist(), nfev=int(result.nfev)))
        except BudgetReached:
            exhausted = True
            break
    if best is None:
        raise RuntimeError('No curve candidate was evaluated.')
    return dict(**best, trace=trace, optimizer_results=results,
                budget_exhausted=exhausted, distinct_evaluations=len(trace),
                starts_attempted=attempted, starts_completed=len(results),
                max_evaluations=budget, starts=starts.copy(), pieces=pieces,
                regularization=regularization, logit_bound=bound,
                max_iterations=max_iterations, elapsed_seconds=perf_counter()-begun,
                scope='Conditional pooled TRAIN fit; no holdout or scientific model acceptance.')
