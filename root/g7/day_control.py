"""G-only full-day composition over unchanged ODE, M254, FV and EM kernels.

Not an executable experiment by itself. The stdlib launcher must admit and
register the complete graph before importing this module into its worker.
"""
import json
import hashlib
from pathlib import Path
from time import perf_counter
import numpy as np

from src.control.bm_1 import BM1ControlNumerics, multistart_powell
from src.control.objective import monte_carlo_tracking_objective
from src.method.bm_1_distribution_panel import baseline, target, uniform_grid_indices
from src.method.bm_1_g7 import OwnedBankPriceEvaluator
from src.method.bm_1_g7_jobs import atomic_json, digest, NodeJobStop
from src.method.bm_1_g8 import G8BankOrigin
from src.method.euler_maruyama import euler_maruyama
from audited_feedback import FeedbackBankControl
from audited_node_control import validate_audit
from feedback_phase import CONFIGS, run_feedback_phase
from day_review import forward_checks, compare_laws, review_em, paired_improvement
from day_contract import validate_design

PATHS = ('root/g7/day_control.py', 'root/g7/day_review.py', 'root/g7/day_contract.py')
POLICIES = ('neutral', 'deterministic', 'stochastic')


def check_sources(root, manifest):
    import hashlib
    root = Path(root).resolve()
    for name, expected in manifest.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source changed: '+name)
    for name in PATHS:
        if name not in manifest:
            raise ValueError('Register all loaded full-day helpers.')
    if Path(__file__).resolve() != root/PATHS[0]:
        raise ValueError('Loaded full-day driver differs from registered source root.')
    import day_review, day_contract
    if (Path(day_review.__file__).resolve() != root/PATHS[1]
            or Path(day_contract.__file__).resolve() != root/PATHS[2]):
        raise ValueError('Loaded review/contract differs from registered source root.')


def make_control(origin, name, store, root, manifest, *, horizon=(12.,36.),
                 edges=(12.,18.,24.,30.,36.)):
    backend, cells, step = CONFIGS['m254_fine' if name == 'quadrature' else name]
    evaluator = OwnedBankPriceEvaluator(origin, baseline=baseline, target=target,
        horizon=horizon, price_edges=list(edges), backend=backend,
        numerics=BM1ControlNumerics(x_steps=cells, solver_max_step=step,
                                  objective_max_step=1/192 if name == 'quadrature' else 1/96))
    return FeedbackBankControl(evaluator, store, source_root=root, source_sha256=manifest)


def receipts(control, prices):
    """Read complete authenticated packets; no copying or charging old jobs."""
    result = []
    with control.store.lock():
        for j, rho in enumerate(control.evaluator.origin.rho_values):
            job = dict(contract=control.contract_id, index=j, rho=float(rho),
                       prices=np.asarray(prices, float).tolist())
            path = control.store.root/'nodes'/(digest(job)+'.json')
            payload = control.store.receipt(path, job)
            audit = validate_audit(payload, index=j, backend=control.evaluator.backend)
            if (control.evaluator.backend == 'mixture'
                    and audit['record']['maximum_projection_cell_mass_l1'] > .05):
                raise ValueError('Registered forward projection-fit gate failed.')
            item = json.loads(path.read_bytes())
            if item['job'] != job or item['payload_sha256'] != digest(payload):
                raise ValueError('Receipt changed during packet read.')
            result.append(item)
    return result


def search(objective, plan, *, output, name, deadline):
    """Two bounded native Powell starts; save every completed scalar candidate.

    Call-cap failure is retained as nonconvergence, not mathematical failure
    suppression. Numerical/store/deadline errors propagate immediately.
    """
    history = []; starts = []
    for index, initial in enumerate(plan['optimizer']['starts']):
        calls = 0
        def evaluate(prices):
            nonlocal calls
            if perf_counter() >= deadline:
                raise NodeJobStop('Common 300s optimizer hard deadline.')
            if calls >= 24 or len(history) >= 48:
                raise NodeJobStop('Registered complete-objective call cap.')
            calls += 1
            value = float(objective(prices))
            if not np.isfinite(value):
                raise ValueError('Nonfinite completed price objective.')
            history.append(dict(start=index, prices=np.asarray(prices).tolist(), J=value))
            atomic_json(output/(name+'_search_history.json'), dict(history=history, starts=starts))
            return value
        try:
            value = multistart_powell(evaluate, [initial], lower=0, upper=1,
                                     options=dict(maxfev=24, maxiter=100))
            status = dict(start=index, converged=True, message=str(value.best.message))
        except RuntimeError as exc:
            # Native Powell raises exactly this after maxfev/nonconvergence.
            # Never mask NodeJobStop, kernel failures or a corrupt receipt.
            if isinstance(exc, NodeJobStop) or not str(exc).startswith('Every bounded Powell start failed. Messages: '):
                raise
            status = dict(start=index, converged=False, message=str(exc))
        starts.append(dict(status, completed_calls=calls))
        atomic_json(output/(name+'_search_history.json'), dict(history=history, starts=starts))
    if not history:
        raise ValueError('No complete search candidate exists.')
    best = min(range(len(history)), key=lambda k: (history[k]['J'], k))
    selected_converged = starts[history[best]['start']]['converged']
    return dict(prices=history[best]['prices'], J=history[best]['J'],
                selected_completed_candidate=best, history=history, starts=starts,
                selected_start_converged=selected_converged,
                any_start_converged=any(s['converged'] for s in starts),
                global_optimum_claimed=False)


def deterministic_job(control, prices, store, *, seconds, deadline):
    """MAP rho and its conditional mean state, never the posterior mean ODE."""
    origin = control.evaluator.origin
    node = origin.map_origin()
    state = float(node.posterior_mass@(.5*(node.x_edges[:-1]+node.x_edges[1:])))
    problem = node.control_problem(baseline=baseline, target=target, horizon=(12.,36.),
        price_edges=[12.,18.,24.,30.,36.], deterministic_initial_state=state)
    evaluator = problem.build_deterministic_evaluator(BM1ControlNumerics(
        x_steps=257, objective_max_step=1/96, solver_max_step=1/96))
    job = dict(kind='g7-day-deterministic-v1', contract=control.contract_id,
               index=0, selected_node=int(node.selected_node), initial_state=state,
               prices=np.asarray(prices, float).tolist())
    def compute(job_deadline):
        original = evaluator.model._drift
        def guarded(*args, **kwargs):
            if perf_counter() >= min(deadline, job_deadline):
                raise NodeJobStop('Deterministic objective reservation exhausted.')
            return original(*args, **kwargs)
        evaluator.model._drift = guarded
        try:
            result = evaluator.evaluate(prices)
            if perf_counter() >= min(deadline, job_deadline):
                raise NodeJobStop('Deterministic objective returned after its deadline.')
            return dict(J=float(result.total), prices=result.prices, state=result.state,
                        demand=result.demand, initial_state=state, selected_node=node.selected_node)
        finally:
            del evaluator.model._drift
    with store.lock():
        return store.node(job, seconds, compute)['J']


def em_node(control, prices, index, *, deadline):
    """Raw coupled EM from the continuous fine M254 conditional mixture."""
    s = control.evaluator.supports[index]; origin = control.evaluator.origin
    def rng(role):
        return np.random.default_rng(np.random.SeedSequence(20261008, spawn_key=(role,index)))
    generator = rng(1)
    components = generator.choice(255, 256, p=origin.conditional_coefficients[index])
    initial = generator.beta(components+1., 255.-components)
    fine_time = 12.+np.arange(24*384+1)/384
    dw = rng(2).standard_normal((256, len(fine_time)-1))*np.sqrt(np.diff(fine_time))
    # Binary hashes avoid turning millions of increments into Python/JSON scalars.
    fingerprint = hashlib.sha256(b'g7-em-inputs-float64-little-endian-v1')
    fingerprint.update(digest(dict(entropy=20261008, roles=[1,2], node=index,
                                   shapes=[initial.shape,dw.shape,fine_time.shape])).encode())
    for array in (initial, dw, fine_time):
        fingerprint.update(np.asarray(array, dtype='<f8').tobytes(order='C'))
    inputs = fingerprint.hexdigest()
    original = s.model._drift
    def guarded(*args, **kwargs):
        if perf_counter() >= deadline:
            raise NodeJobStop('Coupled EM node reservation exhausted.')
        return original(*args, **kwargs)
    s.model._drift = guarded
    levels = []; violations = 0; truth = None
    try:
        parameters = s.parameters_for_prices(prices)
        for level, (times, increments) in enumerate((
                (fine_time[::2], euler_maruyama.coarsen_brownian_increments(dw, 2)),
                (fine_time, dw))):
            indices = uniform_grid_indices(times, s.midpoints)
            value = s.model.euler_maruyama_solver(initial, parameters, s.sigma_x,
                n_paths=256, boundary='none', t_eval=times, breakpoints=[18.,24.,30.],
                max_step=np.inf, brownian_increments=increments, return_diagnostics=True)
            violations += int(value[-1]['total_violations'])
            if violations:
                raise ValueError('Raw EM boundary violation; no reflection or clipping.')
            demand = np.atleast_2d(value[7])
            levels.append(monte_carlo_tracking_objective(s.widths, demand[:,indices],
                                                         s.target).path_objectives)
            if level == 1:
                truth = float(demand[0,-1])
        if perf_counter() >= deadline:
            raise NodeJobStop('EM result arrived after its reservation.')
    finally:
        del s.model._drift
    return dict(index=index, input_sha256=inputs, coarse_costs=levels[0],
                fine_costs=levels[1], truth_fine_path0_demand=truth,
                raw_boundary_violations=violations,
                initial_states=initial, fine_time=fine_time,
                regeneration=dict(entropy=20261008, initial_role=1, brownian_role=2,
                                  node=index, numpy_version=np.__version__))


def _run_day(plan, origin_records, store, *, source_root, manifest, reservations, output,
             stage_deadline):
    """Search -> frozen policies -> common crosschecks -> EM -> actual feedback."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    validate_design(plan); check_sources(source_root, manifest)
    if set(origin_records) != set(CONFIGS):
        raise ValueError('Four separate owned origin records required.')
    origins = {k:G8BankOrigin.from_record(v) for k,v in origin_records.items()}
    for name, (backend, cells, _) in CONFIGS.items():
        a = origins[name]
        if a.time != 12 or a.backend != backend or len(a.rho_values) != 5 or a.conditional_mass.shape[1] != cells:
            raise ValueError('Wrong own origin cutoff/backend/grid.')
        if not compare_laws(a, origins['m254_fine'], tolerances=plan['tolerances'])['passed']:
            raise ValueError('Owned starting posterior grids are inconsistent.')
    controls = {name:make_control(origin, name, store, source_root, manifest)
                for name,origin in origins.items()}
    controls['quadrature'] = make_control(origins['m254_fine'], 'quadrature', store, source_root, manifest)
    started = perf_counter(); searches = {}; stage_deadline[0] = started+300.
    def det(prices):
        check_sources(source_root, manifest)
        return deterministic_job(controls['m254_coarse'], prices, store,
            seconds=reservations['search_deterministic'], deadline=started+300.)
    def mix(prices):
        value = controls['m254_coarse'].evaluate(prices, seconds_per_node=reservations['search_m254']/5)
        receipts(controls['m254_coarse'], prices)
        return value['total']
    for name, objective in [('deterministic', det), ('stochastic', mix)]:
        searches[name] = search(objective, plan, output=output, name=name, deadline=started+300.)
    policies = dict(neutral=[.5]*4, **{k:v['prices'] for k,v in searches.items()})
    atomic_json(output/'day_policies.json', dict(policies=policies, searches=searches))
    checks = {}; evaluations = {}; em_records = {}
    validation_started = perf_counter()
    stage_deadline[0] = validation_started+300.
    def validation_guard():
        if perf_counter()-validation_started >= 300:
            raise NodeJobStop('Common crossreview/EM/feedback 300s deadline.')
    for policy, prices in policies.items():
        values = {}
        for name, control in controls.items():
            validation_guard()
            rate = reservations['quadrature' if name == 'quadrature' else 'cross_'+name]
            values[name] = control.evaluate(prices, seconds_per_node=rate/5)
            receipts(control, prices)
        evaluations[policy] = values
        checks[policy] = forward_checks(values, controls, plan['tolerances'])
        atomic_json(output/(policy+'_forward_review.json'), checks[policy])
        if not all(c['passed'] for c in checks[policy].values()):
            raise ValueError('Full-day forward crosscheck failed before EM/feedback.')
    em_control = controls['m254_fine']
    for policy, prices in policies.items():
        records = []
        for j in range(5):
            validation_guard(); check_sources(source_root, manifest)
            job = dict(kind='g7-day-coupled-em-v1', contract=em_control.contract_id,
                index=j, prices=prices, entropy=20261008, roles=[1,2], paths=256,
                coarse_step=1/192, fine_step=1/384, boundary='none')
            with store.lock():
                value = store.node(job, reservations['em_coupled']/5,
                    lambda d,j=j:em_node(em_control, prices, j,
                        deadline=min(d, validation_started+300.)))
            records.append(value)
        em_records[policy] = records
    em_review = {k:review_em(v, origins['m254_fine'].weights,
                            evaluations[k]['fv_fine']['total']) for k,v in em_records.items()}
    paired = {k:paired_improvement(em_records['neutral'], em_records[k],
                                  origins['m254_fine'].weights) for k in ('deterministic','stochastic')}
    atomic_json(output/'day_em_review.json', dict(costs=em_review, paired_improvements=paired))
    if not all(c['passed'] for c in em_review.values()):
        raise ValueError('Full-day raw EM crosscheck failed before feedback.')
    truth = em_records['stochastic'][2]['truth_fine_path0_demand']
    noise = float(np.random.default_rng(np.random.SeedSequence(20261008,
        spawn_key=(3,))).normal(0., origins['m254_fine'].sigma_y))
    applied = policies['stochastic']
    packet = dict(schema='g7-audited-feedback-input-v2', applied_prices=applied,
        observation=dict(kind='point', time=36, value=truth+noise,
                         truth=truth, noise=noise, source='prespecified stochastic fine EM node2/path0'),
        price_candidates=[[.5,.5],[.3,.7],[.7,.3]], projection_l1_limit=.05,
        forwards={name:dict(origin=origins[name].to_record(), prices=applied,
            contract_id=controls[name].contract_id, receipts=receipts(controls[name], applied))
            for name in CONFIGS})
    atomic_json(output/'generated_feedback_inputs.json', packet)
    validation_guard()
    feedback = run_feedback_phase(packet, store, source_root=source_root,
        source_sha256=manifest, reservations=reservations, output=output)
    post = {}; scores = {}; next_controls = {}; next_values = {}
    for name in CONFIGS:
        job_id = feedback['posterior_jobs'][name]
        receipt = json.loads((store.root/'nodes'/(job_id+'.json')).read_bytes())
        payload = store.receipt(store.root/'nodes'/(job_id+'.json'), receipt['job'])
        from audited_feedback import validate_update
        validate_update(payload, receipt['job'])
        post[name] = G8BankOrigin.from_record(payload['posterior_origin'])
        scores[name] = [payload['log_evidence'], *payload['conditional_log_evidence']]
        next_controls[name] = make_control(post[name], name, store, source_root, manifest,
                                         horizon=(36.,38.), edges=(36.,37.,38.))
        next_prices = feedback['candidates'][feedback['selected']]
        # Exact durable reuse only; no additional native forward is needed.
        saved = receipts(next_controls[name], next_prices)
        next_values[name] = next_controls[name].evaluator.combine_nodes(
                            next_prices, [r['payload'] for r in saved])
    feedback_checks = {}
    for label, a, b in [('m254_refinement','m254_coarse','m254_fine'),
                         ('fv_refinement','fv_coarse','fv_fine'),
                         ('m254_vs_fv','m254_fine','fv_fine')]:
        feedback_checks[label] = compare_laws(post[a], post[b], tolerances=plan['tolerances'],
                                              scores_a=scores[a], scores_b=scores[b])
    continuation_checks = forward_checks(next_values, next_controls, plan['tolerances'])
    validation_guard(); check_sources(source_root, manifest)
    numerical = all(c['passed'] for c in feedback_checks.values()) and all(
                    c['passed'] for c in continuation_checks.values())
    costs = {p:{n:{key:float(v[key]) for key in (
        'total','bias','variance','within_parameter_variance','between_parameter_variance')}
        for n,v in values.items()} for p,values in evaluations.items()}
    report = dict(schema='g7-full-day-result-v1', full_day_experiment_completed=True,
        horizon=[12,36], next_horizon=[36,38], policies=policies,
        searches=searches, costs=costs, forward_checks=checks, em=em_review,
        paired_improvements=paired, feedback=feedback,
        feedback_checks=feedback_checks, continuation_checks=continuation_checks,
        numerical_validation_passed=bool(numerical), global_optimum_claimed=False,
        scientific_accept=False, historical_validation=False,
        note='Numerical chain only; parent budget/source review and scientific interpretation are separate.')
    atomic_json(output/'day_result.json', report)
    return report


def run_day(plan, origin_records, store, *, source_root, manifest, reservations, output):
    """One process owns the store; every job also respects its current stage."""
    original = store.node
    had_override = 'node' in store.__dict__; override = store.__dict__.get('node')
    stage_deadline = [np.inf]
    def node(job, reservation, compute, *, retry_ids=()):
        if retry_ids:
            raise ValueError('No automatic or implicit retry in this registered day.')
        deadline = stage_deadline[0]
        if perf_counter()+reservation > deadline:
            raise NodeJobStop('Whole job reservation does not fit the remaining stage cap.')
        def guarded(job_deadline):
            result = compute(min(job_deadline, deadline))
            if perf_counter() >= deadline:
                raise NodeJobStop('Result completed after the common stage deadline.')
            return result
        return original(job, reservation, guarded)
    store.node = node
    try:
        return _run_day(plan, origin_records, store, source_root=source_root,
            manifest=manifest, reservations=reservations, output=output,
            stage_deadline=stage_deadline)
    except Exception as exc:
        atomic_json(Path(output)/'day_stop.json', dict(status='stopped',
            error_type=type(exc).__name__, message=str(exc), automatic_retry=False,
            full_day_experiment_completed=False, scientific_accept=False))
        raise
    finally:
        if had_override: store.node = override
        else: del store.node
