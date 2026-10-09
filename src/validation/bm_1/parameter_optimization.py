"""Small conditional physical-parameter pilot; no replacement for F4 banks.

The retired pilot driver is preserved in archive/bm/originals/source-7249bf9.zip.
Only synthetic DEVELOPMENT records are used; the design is written before fitting.
Every method gets its own cache and a strict count of full TRAIN evaluations,
including finite differences; two starts share a fixed, reserved budget.
"""

from dataclasses import asdict, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter
import platform

import numpy as np
import scipy
from scipy.optimize import minimize, least_squares

from src.method.bm_1_estimation import fv_filter_step, _observation_demand
from src.solver.finite_volume import finite_volume_1d_cached
from src.tools.bm_1_reports import _atomic_report
from src.validation.bm_1 import estimation as existing


SOURCE_BASE = '06942cf5cd29d457f595e611d5564b5c94bb9fdc'
TRAIN_IDS = (1326, 1327)
CHECK_IDS = (1211, 1212)  # Previously used DEVELOPMENT data, never blind holdout.
NUMERICS = dict(cells=401, max_step=1/768, quadrature_order=3)
GROUPS = (('rho',), ('k',), ('rho', 'Xi'), ('sigma_x', 'sigma_y'))


class BudgetStop(Exception):
    pass


class TimeStop(Exception):
    pass


def physical_candidate(z, names, fixed):
    """Dimensionless box [0,1]; noise coordinates interpolate logarithmically."""
    z = np.asarray(z, dtype=float)
    if z.shape != (len(names),) or not np.all(np.isfinite(z)) or np.any((z < 0) | (z > 1)):
        raise ValueError('Candidate must lie in the declared unit box.')
    changes = {}
    for name, value in zip(names, z):
        axis = existing.F4_E_PARAMETER_GRIDS[name]
        lo, hi = axis[0], axis[-1]
        changes[name] = (float(np.exp(np.log(lo)+value*np.log(hi/lo)))
                         if name.startswith('sigma_') else float(lo+value*(hi-lo)))
    return replace(fixed, **changes)


def replay(case, observations, candidate, *, cells=401, max_step=1/768,
           quadrature_order=3):
    """Fresh prior, existing step kernel, prediction BEFORE each Y update.

Store O(times) scalars, not density history. Mean quadrature uses the same
piecewise-constant cell law and input side as the likelihood quadrature.
The current measurement never enters its own predicted mean.
"""
    model, base = case.build_model()
    model.fokker_planc_reference = finite_volume_1d_cached(vectorized_inputs=True)
    parameters = existing._f4_e_parameters(base, candidate)
    mass = existing.beta_initial_mass(case, model, cells)
    time = case.horizon[0]
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    rows, error, minimum = [], 0., 1.
    for next_time, y, observed in zip(observations.times, observations.values, observations.observed):
        step = fv_filter_step(model, parameters, candidate.sigma_x, candidate.sigma_y,
            mass, time, float(next_time), float(y) if observed else np.nan,
            observed=bool(observed), price_side=observations.price_side,
            breakpoints=case.breakpoints, max_step=max_step,
            likelihood_quadrature_order=quadrature_order)
        error = max(error, step.diagnostics['maximum_abs_mass_error'])
        minimum = min(minimum, step.diagnostics['minimum_cell_mass'])
        if error > 1e-10 or minimum < 0:
            raise RuntimeError('Probability control failed; no repair or penalty substitution.')
        points = step.x_grid[:, None] + .5*np.diff(step.x_edges)[:, None]*nodes
        demand = _observation_demand(model, parameters, float(next_time), points.ravel(),
                                    observations.price_side, time).reshape(points.shape)
        mean = float(step.predicted_mass @ (demand @ (.5*weights)))
        if observed:
            rows.append(dict(time=float(next_time), y=float(y), mean=mean,
                             residual=float(y-mean), log_score=float(step.log_evidence)))
        mass, time = step.posterior_mass, step.time
    if not rows:
        raise ValueError('At least one observation required.')
    return dict(rows=rows, maximum_abs_mass_error=float(error), minimum_cell_mass=float(minimum))


def pooled(case, records, candidate, **numerics):
    outputs = [dict(record_id=rid, **replay(case, obs, candidate, **numerics))
               for rid, obs in records]
    rows = [row for output in outputs for row in output['rows']]
    residuals = np.array([row['residual'] for row in rows])
    return dict(candidate=asdict(candidate), records=outputs,
                negative_log_likelihood=-sum(row['log_score'] for row in rows),
                residuals=residuals.tolist(),
                # Fixed weights, never parameter-dependent whitening.
                least_squares_cost=float(.5*np.sum((residuals/case.sigma_y)**2)),
                observation_rmse=float(np.sqrt(np.mean(residuals**2))))


class Evaluator:
    """Per-method exact-key cache, strict per-start and total replay limits."""

    def __init__(self, case, records, names, budget, *, deadline, checkpoint=None):
        self.case, self.records, self.names = case, records, names
        self.fixed = existing.f4_e_truth_candidate(case)
        self.budget, self.limit, self.deadline = budget, budget, deadline
        self.trace, self.memo = [], {}
        self.checkpoint = checkpoint

    def __call__(self, z):
        key = tuple(float(x) for x in z)
        if key in self.memo:
            return self.memo[key]
        if perf_counter() >= self.deadline:
            raise TimeStop()
        if len(self.trace) >= min(self.budget, self.limit):
            raise BudgetStop()
        candidate = physical_candidate(z, self.names, self.fixed)
        begun = perf_counter()
        result = pooled(self.case, self.records, candidate, **NUMERICS)
        row = dict(z=list(key), seconds=perf_counter()-begun, **result)
        self.trace.append(row)
        self.memo[key] = row
        if self.checkpoint is not None:
            self.checkpoint(self.trace)
        return row


def optimize(evaluate, method):
    """Both starts are attempted, even if the first consumes its allocation."""
    results = []
    allocations = (evaluate.budget//2, evaluate.budget-evaluate.budget//2)
    for value, allocation in zip((.25, .75), allocations):
        evaluate.limit = min(evaluate.budget, len(evaluate.trace)+allocation)
        start = np.full(len(evaluate.names), value)
        before = len(evaluate.trace)
        try:
            if method == 'mle':
                result = minimize(lambda z: evaluate(z)['negative_log_likelihood'], start,
                    method='L-BFGS-B', jac='2-point', bounds=[(0., 1.)]*len(start),
                    options=dict(maxiter=40, maxfun=evaluate.budget, ftol=1e-7,
                                 gtol=1e-5, finite_diff_rel_step=1e-5))
            elif method == 'ls':
                result = least_squares(lambda z: np.asarray(evaluate(z)['residuals'])/evaluate.case.sigma_y,
                    start, bounds=(0., 1.), method='trf', jac='2-point', diff_step=1e-5,
                    ftol=1e-7, xtol=1e-7, gtol=1e-5, max_nfev=evaluate.budget)
            else:
                raise ValueError('Unknown optimizer.')
            results.append(dict(start=value, success=bool(result.success),
                                message=str(result.message), scipy_nfev=int(result.nfev)))
        except BudgetStop:
            results.append(dict(start=value, success=False, message='reserved_candidate_budget'))
        results[-1]['new_candidates'] = len(evaluate.trace)-before
    return results


def selected(evaluate, score):
    return min(evaluate.trace, key=lambda row: row[score])


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
            for path in sorted((root/'src').rglob('*.py'))}


def run(output, *, max_seconds=600., progress=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('Positive time budget required.')
    begun = perf_counter()
    deadline = begun+max_seconds
    hashes = source_hashes()
    design = dict(scope='Old synthetic DEVELOPMENT only; conditional known nuisance parameters.',
        scientific_acceptance=False, source_base=SOURCE_BASE, source_sha256=hashes,
        train_ids=list(TRAIN_IDS), development_check_ids=list(CHECK_IDS),
        fixed_curves='f(x)=1-2x; g(u)=1-2u', groups=[list(g) for g in GROUPS],
        numerics=NUMERICS, single_budget=41, pair_budget=81, starts=[.25,.75],
        start_budget='floor(budget/2), remainder; unused first allocation not reassigned',
        methods=['grid MLE/LS','L-BFGS-B MLE','TRF LS (response parameters only)'],
        optimizer_settings=dict(maxiter_mle=40, ftol=1e-7, gtol=1e-5,
                                xtol_ls=1e-7, relative_difference_step=1e-5),
        ls_weight='1 / fixed sigma_y squared; no noise-parameter LS fit',
        evaluation='One-step predicted Y before current update, not open forecast.',
        max_seconds=max_seconds, python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, timings='Serial common scorer including means; single Linux run.')
    _atomic_report(output/'design.json', design)
    case = existing.build_f4_b_case()
    observations = {}
    for rid in TRAIN_IDS+CHECK_IDS:
        sequence = existing.generate_synthetic_sequence(case, sequence_id=rid)
        if not sequence.truth.valid:
            raise RuntimeError('Existing synthetic truth invalid; do not substitute a seed.')
        observations[rid] = sequence.observations
    train = [(rid, observations[rid]) for rid in TRAIN_IDS]
    check = [(rid, observations[rid]) for rid in CHECK_IDS]
    _atomic_report(output/'observations.json', {str(rid):dict(times=obs.times.tolist(),
        values=obs.values.tolist(), observed=obs.observed.tolist()) for rid,obs in observations.items()})
    completed = []
    summaries = []
    # 3 methods for each response group and 2 for the noise pair.
    total_methods = 11
    try:
        for names in GROUPS:
            budget = 41 if len(names)==1 else 81
            label = '_'.join(names)
            methods = ('grid','mle') if 'sigma_x' in names else ('grid','mle','ls')
            for method in methods:
                path = output/(label+'_'+method+'.json')
                start = perf_counter()
                evaluate = Evaluator(case, train, names, budget, deadline=deadline,
                    checkpoint=lambda trace: _atomic_report(path, dict(completed=False,trace=trace),
                                                           replace_existing=True))
                if method == 'grid':
                    axes = [np.linspace(0.,1.,41 if len(names)==1 else 9)]*len(names)
                    for z in product(*axes):
                        evaluate(z)
                    statuses = []
                    scores = ['negative_log_likelihood']
                    if 'sigma_x' not in names:
                        scores.append('least_squares_cost')
                else:
                    statuses = optimize(evaluate, method)
                    scores = ['negative_log_likelihood' if method=='mle' else 'least_squares_cost']
                seconds = perf_counter()-start
                selections = []
                for score in scores:
                    best = selected(evaluate,score)
                    candidate = existing.F4ELikelihoodCandidate(**best['candidate'])
                    if perf_counter() >= deadline:
                        raise TimeStop()
                    evaluation = pooled(case,check,candidate,**NUMERICS)
                    # Fixed selected candidates, not a claim about the optimizer's whole box.
                    refinement = {}
                    for axis,settings in [('time',dict(NUMERICS,max_step=1/1536)),
                                          ('space',dict(NUMERICS,cells=801)),
                                          ('quadrature',dict(NUMERICS,quadrature_order=5))]:
                        if perf_counter() >= deadline:
                            raise TimeStop()
                        fine = pooled(case,train,candidate,**settings)
                        refinement[axis] = dict(
                            negative_log_likelihood=fine['negative_log_likelihood'],
                            least_squares_cost=fine['least_squares_cost'],
                            delta_nll=fine['negative_log_likelihood']-best['negative_log_likelihood'],
                            delta_ls=fine['least_squares_cost']-best['least_squares_cost'],
                            max_mean_change=max(abs(a['mean']-b['mean'])
                                for a_record,b_record in zip(fine['records'],best['records'])
                                for a,b in zip(a_record['rows'],b_record['rows'])))
                    selections.append(dict(score=score,selected=best,development_check=evaluation,
                                           selected_refinement=refinement))
                    summaries.append(dict(group=label,method=method,score=score,
                        candidates=len(evaluate.trace),fit_seconds=seconds,
                        candidate=best['candidate'],train_nll=best['negative_log_likelihood'],
                        train_ls=best['least_squares_cost'],
                        check_rmse=evaluation['observation_rmse'],
                        check_nll=evaluation['negative_log_likelihood'],statuses=statuses))
                _atomic_report(path,dict(completed=True,method=method,group=list(names),
                    fit_seconds=seconds,budget=budget,statuses=statuses,trace=evaluate.trace,
                    selections=selections),replace_existing=True)
                completed.append(label+'_'+method)
                _atomic_report(output/'summary.json',dict(completed=False,scientific_acceptance=False,
                    methods_completed=completed,rows=summaries),replace_existing=True)
                if progress is not None:
                    progress(label, method, len(evaluate.trace), begun, len(completed), total_methods)
        if source_hashes()!=hashes:
            raise RuntimeError('Source changed during pilot.')
        report = dict(completed=True,scientific_acceptance=False,methods_completed=completed,
                      elapsed_seconds=perf_counter()-begun,rows=summaries)
    except TimeStop:
        report = dict(completed=False,scientific_acceptance=False,stop_reason='time_budget',
                      methods_completed=completed,elapsed_seconds=perf_counter()-begun,rows=summaries)
    _atomic_report(output/'summary.json',report,replace_existing=True)
    return report

