"""F4-G joint-bank numerical studies, separate from the G0 method check.

G1a holds the 25-node joint parameter measure fixed while refining FV time,
state cells and observation quadrature. Parameter-grid refinement is G1b.
"""

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import replace
from multiprocessing import get_context
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from src.method.bm_1_estimation import parameter_bank_weights, run_fv_filter
from src.params.bm_1 import conservative_mass_transfer
from src.validation import estimation as checks
from src.validation.bm_1 import estimation as core


G1A_SEQUENCE_IDS = ((1350, 1351, 1352), (1353, 1354, 1355))
G1A_NUMERICS = {
    'rho__Xi': dict(time_steps=(1/768, 1/1536, 1/3072),
                   spatial_cells=(201, 401, 801), quadrature_orders=(1, 3, 5)),
    'sigma_x__sigma_y': dict(time_steps=(1/768, 1/1536, 1/3072),
                            spatial_cells=(1601, 3201, 6401), quadrature_orders=(1, 3, 5)),
}


def _array(values, shape, name):
    raw = np.asarray(values)
    if raw.shape != shape or not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f'{name} must be a real array of shape {shape}.')
    result = raw.astype(float, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite.')
    return result


def _equal_array(values, expected, name, tolerance=1e-10):
    expected = np.asarray(expected)
    value = _array(values, expected.shape, name)
    if not np.allclose(value, expected, atol=tolerance, rtol=0):
        raise ValueError(f'{name} disagrees with the registered design/raw scores.')
    return value


def _mass(values, rows, cells, name):
    mass = _array(values, (rows, cells), name)
    if np.any(mass < 0) or np.max(np.abs(mass.sum(axis=1)-1)) > 1e-10:
        raise ValueError(f'{name} must contain normalized nonnegative cell masses.')
    return mass


def validated_g0(report, *, require_source=False):
    """Recheck G0 raw algebra/design/probability gates before the dependent CLI.

    Full conditional prefix histories are not in G0 JSON. Their recorded gates
    remain execution evidence; stored marginal/final masses are checked here.
    """
    expected_header = dict(stage='G0', registered_design=True, grid_stride=10,
                           identity_tolerance=1e-10, conditional_runs=100, passes=True)
    if (any(report.get(k) != v for k, v in expected_header.items())
            or report.get('registered_design') is not True or report.get('passes') is not True):
        raise ValueError('A complete passing registered G0 report is required.')
    if require_source and (not isinstance(report.get('source_commit'), str)
                           or not report['source_commit'] or report.get('working_tree_dirty') is not False):
        raise ValueError('G0 requires a recorded source commit and clean working tree.')
    rows = report.get('rows', [])
    if len(rows) != 2:
        raise ValueError('G0 requires both complete pair rows.')
    seen = set()
    for row in rows:
        pair = tuple(row['coordinates'])
        if pair not in core.F4_G_PAIRS or pair in seen:
            raise ValueError('G0 pair rows must be unique and registered.')
        seen.add(pair)
        index = core.F4_G_PAIRS.index(pair)
        cells = (401, 3201)[index]
        case = core.build_f4_b_case(Xi=(.75, 1.)[index])
        truth = core.f4_e_truth_candidate(case)
        expected = dict(sequence_id=core.F4_G_METHOD_IDS[index],
                        settings=dict(x_steps=cells, max_step=1/1536, likelihood_quadrature_order=3),
                        truth=[getattr(truth, n) for n in pair],
                        fixed_parameters={n:getattr(truth,n) for n in ('rho','Xi','k','sigma_x','sigma_y') if n not in pair},
                        truth_max_step=1/1152,
                        inputs=dict(baseline=.5, price_edges=[0.,1/3,2/3,1.], price_values=[.65,.35,.65],
                                    state_response='1-2*x', price_response='1-2*u'),
                        observation_contract=dict(time_unit='day', observation_kind='point', price_side='right'))
        if any(row.get(k) != v for k,v in expected.items()) or row.get('observed') != [True]*24:
            raise ValueError('G0 case, inputs, observations or numerical settings were changed.')
        _equal_array(row['initial_beta'],[20.,20.],'G0 initial Beta',0.)
        _equal_array(row['seed_roots'],[4101,4102,4103],'G0 seed roots',0.)
        sequence = core.generate_synthetic_sequence(case, sequence_id=row['sequence_id'])
        _equal_array(row['observation_times'], sequence.observations.times, 'G0 times', 1e-12)
        _equal_array(row['observations'], sequence.observations.values, 'G0 seed observations', 1e-12)
        h = np.array([(a,b) for a in core.F4_E_PARAMETER_GRIDS[pair[0]][::10]
                      for b in core.F4_E_PARAMETER_GRIDS[pair[1]][::10]])
        _equal_array(row['hypotheses'], h, 'G0 hypotheses', 1e-12)
        prior = np.full(25, -np.log(25))
        _equal_array(row['initial_log_weights'], prior, 'G0 joint prior')
        online = _array(row['online_conditional_log_evidence'], (24,25), 'G0 online scores')
        offline = _array(row['offline_conditional_log_evidence'], (24,25), 'G0 offline scores')
        joint = prior + np.cumsum(offline, axis=0)
        evidence = logsumexp(joint, axis=1)
        logs = _equal_array(row['online_log_weights'], joint-evidence[:,None], 'G0 log weights')
        _equal_array(row['offline_cumulative_bank_log_evidence'], evidence, 'G0 offline evidence')
        bank = _array(row['online_bank_log_evidence'], (24,), 'G0 bank evidence')
        _equal_array(np.cumsum(bank), evidence, 'G0 cumulative bank evidence')
        prefix = row['prefix_errors']
        names = {'log_weights','conditional_log_evidence','cumulative_log_evidence',
                 'conditional_state_mass','marginal_state_mass'}
        if set(prefix) != names or set(row['maximum_errors']) != names:
            raise ValueError('G0 requires every identity gate.')
        raw_errors = dict(log_weights=np.max(np.abs(logs-(joint-evidence[:,None])),axis=1),
                          conditional_log_evidence=np.max(np.abs(online-offline),axis=1),
                          cumulative_log_evidence=np.abs(np.cumsum(bank)-evidence))
        for name in names:
            values = _array(prefix[name], (24,), 'G0 prefix '+name)
            if np.any(values < 0) or np.max(values) > 1e-10 or row['maximum_errors'][name] != float(values.max()):
                raise ValueError('G0 identity gates failed or maxima disagree.')
            if name in raw_errors:
                _equal_array(values, raw_errors[name], 'G0 prefix '+name)
        model = case.build_model()[0]
        _equal_array(row['initial_mass'], core.beta_initial_mass(case,model,cells), 'G0 Beta prior')
        _equal_array(row['x_edges'], np.linspace(0,1,cells+1), 'G0 state edges', 1e-12)
        for phase in ('predicted','posterior'):
            _mass(row[f'marginal_{phase}_mass'],24,cells,'G0 marginal '+phase)
        final = _mass(row['final_conditional_posterior_mass'],25,cells,'G0 final conditionals')
        _equal_array(row['marginal_posterior_mass'][-1], np.exp(logs[-1])@final, 'G0 final marginal')
        summary = core._f4_g_joint_summary(h, logs, pair)
        for name,value in summary.items():
            if name == 'marginal_parameters':
                if set(row[name]) != set(pair):
                    raise ValueError('G0 marginal coordinates were changed.')
                for coordinate, marginal in value.items():
                    for field,expected_value in marginal.items():
                        _equal_array(row[name][coordinate][field],expected_value,'G0 marginal '+field)
            elif name == 'map_hypotheses':
                if row[name] != value:
                    raise ValueError('G0 MAP ties disagree with raw weights.')
            else:
                _equal_array(row[name],value,'G0 '+name)
        _probability_gate([row], raise_error=True)
        step = checks.finite_scalar(row['actual_max_step'],'G0 actual step',positive=True)
        if step > 1/1536+1e-14 or row.get('passes') is not True:
            raise ValueError('G0 actual step or row acceptance failed.')
    return True


def _probability_gate(rows, *, raise_error=False):
    valid = bool(rows) and all(
        np.isfinite(r['maximum_abs_mass_error']) and 0 <= r['maximum_abs_mass_error'] <= 1e-10
        and np.isfinite(r['minimum_cell_mass']) and r['minimum_cell_mass'] >= 0 for r in rows)
    if raise_error and not valid:
        raise ValueError('Joint-bank probability gates failed.')
    return bool(valid)


def _configurations(levels):
    if set(levels) != {'time_steps','spatial_cells','quadrature_orders'}:
        raise ValueError('G1a requires time, spatial and quadrature triples.')
    time = tuple(checks.finite_scalar(v,'time step',positive=True) for v in levels['time_steps'])
    cells = tuple(checks.positive_integer(v,'state cells',minimum=3) for v in levels['spatial_cells'])
    orders = tuple(checks.positive_integer(v,'quadrature order') for v in levels['quadrature_orders'])
    if (any(len(v)!=3 for v in (time,cells,orders)) or not time[0]>time[1]>time[2]
            or not cells[0]<cells[1]<cells[2] or not orders[0]<orders[1]<orders[2]):
        raise ValueError('G1a levels must strictly refine in all three axes.')
    def settings(c=cells[1],t=time[1],q=orders[1]):
        return dict(x_steps=c,max_step=t,likelihood_quadrature_order=q)
    return [('middle',settings()),('time_coarse',settings(t=time[0])),('time_fine',settings(t=time[2])),
            ('spatial_coarse',settings(c=cells[0])),('spatial_fine',settings(c=cells[2])),
            ('quadrature_coarse',settings(q=orders[0])),('quadrature_fine',settings(q=orders[2]))]


def _candidate_replay(task):
    i,j,group = task
    started = perf_counter()
    case,obs,settings = group['case'],group['observations'],group['settings']
    model,base = case.build_model()
    backend = group.get('backend','reference')
    if backend in ('lapack', 'cached'):
        from src.solver.finite_volume import finite_volume_1d_lapack, finite_volume_1d_cached
        model.fokker_planc_reference = (finite_volume_1d_cached(vectorized_inputs=True) if backend == 'cached'
                                       else finite_volume_1d_lapack())
    elif backend!='reference':
        raise ValueError('Unknown conditional FV backend.')
    candidate = replace(core.f4_e_truth_candidate(case),
                        **dict(zip(group['pair'],group['hypotheses'][j])))
    try:
        result = run_fv_filter(model,core._f4_e_parameters(base,candidate),candidate.sigma_x,candidate.sigma_y,
                              core.beta_initial_mass(case,model,settings['x_steps']),case.horizon[0],obs,
                              breakpoints=case.breakpoints,max_step=settings['max_step'],
                              likelihood_quadrature_order=settings['likelihood_quadrature_order'])
    except (RuntimeError,ValueError,FloatingPointError) as exc:
        raise RuntimeError(f"{group.get('stage','G1a')} replay failed: ID{group['sequence_id']}, {group['pair']}="
                           f"{group['hypotheses'][j].tolist()}, {group['label']}: {exc}") from exc
    extra = {}
    if group.get('keep_demand', False):
        from src.method.bm_1_estimation import _observation_demand, _legendre_rule
        parameters = core._f4_e_parameters(base, candidate)
        nodes, weights = _legendre_rule(settings['likelihood_quadrature_order'])
        points = result.x_grid[:, None] + np.diff(result.x_edges)[:, None]*nodes[None, :]/2
        demand_cache = {}
        means = []
        for time in obs.times:
            price_time = time if obs.price_side == 'right' else np.nextafter(time, case.horizon[0])
            key = (float(model.B(time)), float(parameters[0](price_time)))
            if key not in demand_cache:
                demand_cache[key] = _observation_demand(model, parameters, float(time), points.ravel(),
                    obs.price_side, case.horizon[0]).reshape(points.shape) @ (weights/2)
            means.append(demand_cache[key])
        demands = np.asarray(means)
        extra = {phase+'_demand_mean': np.sum(demands*getattr(result, phase+'_mass'), axis=1)
                 for phase in ('predicted', 'posterior')}
    return i,j,dict(scores=result.log_evidence,predicted=result.predicted_mass,posterior=result.posterior_mass,
                    maximum_abs_mass_error=result.maximum_abs_mass_error,minimum_cell_mass=result.minimum_cell_mass,
                    actual_max_step=max(d['actual_max_step'] for d in result.prediction_diagnostics),
                    elapsed_seconds=perf_counter()-started, **extra)


def benchmark_joint_fv(*, record_days=7, repeats=2, progress=None):
    """Paired backend check on old G2a data, not new scientific evidence.

    Five hypotheses (four support corners and the center) per pair, all four
    FV settings. Timing is serial, alternating backend order, with identical
    observations; a speed ratio is measured, never a test pass criterion.
    """
    from src.validation.bm_1 import joint_grid as grid
    repeats = checks.positive_integer(repeats, 'repeats')
    rows = []
    configs = grid._grid_configurations(G1A_NUMERICS)
    for pair, sid in zip(core.F4_G_PAIRS, grid.G2A_IDS):
        case, sequence, _ = grid._g2a_truth_record(pair, sid, record_days)
        hypotheses = grid._parameter_grid(pair, 65, 65)[0]
        selected = [0, 64, 32*65+32, 64*65, 65*65-1]
        for label, settings in configs['__'.join(pair)]:
            for j in selected:
                group = dict(case=case, observations=sequence.observations, pair=pair,
                    hypotheses=hypotheses[j:j+1], sequence_id=sid, label=label, settings=settings)
                durations = {'lapack': [], 'cached': []}
                errors = {name: 0. for name in ('scores', 'predicted', 'posterior', 'actual_max_step')}
                diagnostics = []
                for repeat in range(repeats):
                    results = {}
                    for backend in (('lapack', 'cached') if repeat % 2 == 0 else ('cached', 'lapack')):
                        results[backend] = _candidate_replay((0, 0, dict(group, backend=backend)))[2]
                        durations[backend].append(results[backend]['elapsed_seconds'])
                    for name in errors:
                        errors[name] = max(errors[name], float(np.max(abs(
                            np.asarray(results['cached'][name])-np.asarray(results['lapack'][name])))))
                    diagnostics.extend(results.values())
                rows.append(dict(coordinates=list(pair), sequence_id=sid, label=label,
                    hypothesis=group['hypotheses'][0].tolist(), settings=settings, seconds=durations,
                    maximum_errors=errors,
                    maximum_abs_mass_error=max(r['maximum_abs_mass_error'] for r in diagnostics),
                    minimum_cell_mass=min(r['minimum_cell_mass'] for r in diagnostics),
                    passes=bool(max(errors.values())<=1e-10 and _probability_gate(diagnostics))))
                if progress is not None: progress(len(rows), 40)
    times = {b:sum(float(np.median(r['seconds'][b])) for r in rows) for b in ('lapack','cached')}
    return dict(stage='FV-backend-benchmark', record_days=record_days, repeats=repeats,
        sequence_ids=list(grid.G2A_IDS), rows=rows, median_seconds_sum=times,
        measured_serial_speedup=times['lapack']/times['cached'],
        comparison_tolerance=1e-10, passes=all(r['passes'] for r in rows),
        interpretation='Paired implementation equivalence and serial timing on old records; no new G2 acceptance or Windows runtime promise.')


def _reduce(group, conditionals):
    h = np.asarray(group['hypotheses'])
    scores = np.column_stack([c['scores'] for c in conditionals])
    prior = np.asarray(group.get('initial_log_weights',np.full(len(h),-np.log(len(h)))))
    reduced = parameter_bank_weights(scores,prior,group['observations'].observed)
    weights,old = np.exp(reduced['log_weights']),np.exp(reduced['prior_log_weights'])
    predicted = np.zeros_like(conditionals[0]['predicted'])
    posterior = np.zeros_like(conditionals[0]['posterior'])
    for j,c in enumerate(conditionals):
        predicted += old[:,j,None]*c['predicted']
        posterior += weights[:,j,None]*c['posterior']
    diagnostics = dict(maximum_abs_mass_error=max(c['maximum_abs_mass_error'] for c in conditionals),
                       minimum_cell_mass=min(c['minimum_cell_mass'] for c in conditionals),
                       conditional_actual_max_steps=[float(c['actual_max_step']) for c in conditionals],
                       conditional_seconds=sum(c['elapsed_seconds'] for c in conditionals))
    return _reduced_row(group,scores,prior,reduced,predicted,posterior,diagnostics)


def _reduced_row(group, scores, prior, reduced, predicted, posterior, diagnostics):
    """Describe a completed bank; conditional trajectories remain independent."""
    h = np.asarray(group['hypotheses'])
    case,obs,cells = group['case'],group['observations'],group['settings']['x_steps']
    truth = core.f4_e_truth_candidate(case)
    mass_error = max(diagnostics['maximum_abs_mass_error'],
                     float(np.max(np.abs(predicted.sum(axis=1)-1))),float(np.max(np.abs(posterior.sum(axis=1)-1))))
    def raw_array(value):
        return value if group.get('array_output', False) else value.tolist()
    return dict(coordinates=list(group['pair']),sequence_id=group['sequence_id'],label=group['label'],
                settings=group['settings'],hypotheses=h.tolist(),initial_log_weights=prior.tolist(),
                truth=[getattr(truth,n) for n in group['pair']],
                fixed_parameters={n:getattr(truth,n) for n in ('rho','Xi','k','sigma_x','sigma_y') if n not in group['pair']},
                observation_times=obs.times.tolist(),observations=obs.values.tolist(),observed=obs.observed.tolist(),
                initial_mass=core.beta_initial_mass(case,case.build_model()[0],cells).tolist(),
                conditional_log_evidence=raw_array(scores),log_weights=raw_array(reduced['log_weights']),
                bank_log_evidence=reduced['log_evidence'].tolist(),
                cumulative_log_evidence=np.cumsum(reduced['log_evidence']).tolist(),
                marginal_predicted_mass=raw_array(predicted),marginal_posterior_mass=raw_array(posterior),
                x_edges=np.linspace(0,1,cells+1).tolist(),maximum_abs_mass_error=mass_error,
                minimum_cell_mass=diagnostics['minimum_cell_mass'],
                conditional_actual_max_steps=diagnostics['conditional_actual_max_steps'],
                conditional_seconds=diagnostics['conditional_seconds'],
                **core._f4_g_joint_summary(h,reduced['log_weights'],group['pair']))


def compare_banks(left, right):
    """Same joint measure and data; all-prefix odds, TV, evidence and state L1.

    Fine state masses are conservatively mapped to the left grid only for this
    comparison. No remapped state enters a replay or parameter update.
    """
    for name in ('coordinates','sequence_id','hypotheses','initial_log_weights',
                 'observation_times','observations','observed','truth','fixed_parameters'):
        if left[name] != right[name]:
            raise ValueError('G1a comparisons require identical hypotheses, prior and experiment data.')
    count,h = len(left['observation_times']),len(left['hypotheses'])
    a,b = [_array(r['conditional_log_evidence'],(count,h),'conditional scores') for r in (left,right)]
    a,b = np.cumsum(a,axis=0),np.cumsum(b,axis=0)
    errors = b-a
    relevant = (a>=a.max(axis=1)[:,None]-5.) | (b>=b.max(axis=1)[:,None]-5.)
    odds = np.array([np.ptp(err[keep]) for err,keep in zip(errors,relevant)])
    all_odds = np.ptp(errors,axis=1)
    logs = [_array(r['log_weights'],(count,h),'joint log weights') for r in (left,right)]
    for r,cumulative,log in zip((left,right),(a,b),logs):
        raw = np.asarray(r['initial_log_weights'])+cumulative
        expected = raw-logsumexp(raw,axis=1)[:,None]
        _equal_array(log,expected,'joint weights')
        _equal_array(r['cumulative_log_evidence'],logsumexp(raw,axis=1),'bank evidence')
    tv = .5*np.abs(np.exp(logs[0])-np.exp(logs[1])).sum(axis=1)
    evidence = np.abs(np.asarray(right['cumulative_log_evidence'])-np.asarray(left['cumulative_log_evidence']))
    state = {}
    for phase in ('predicted','posterior'):
        l = _mass(left[f'marginal_{phase}_mass'],count,len(left['x_edges'])-1,'left '+phase)
        r = _mass(right[f'marginal_{phase}_mass'],count,len(right['x_edges'])-1,'right '+phase)
        mapped = np.array([conservative_mass_transfer(right['x_edges'],m,left['x_edges']) for m in r])
        state[phase+'_l1'] = np.abs(mapped-l).sum(axis=1)
    prefix = dict(log_odds_error=odds,all_node_log_odds_error=all_odds,weight_tv=tv,
                  cumulative_evidence_error=evidence,**state)
    maxima = {'maximum_'+k:float(v.max()) for k,v in prefix.items()}
    passes = (maxima['maximum_log_odds_error']<=.10 and maxima['maximum_weight_tv']<=.01
              and maxima['maximum_cumulative_evidence_error']<=.10
              and max(maxima['maximum_predicted_l1'],maxima['maximum_posterior_l1'])<=.02)
    return dict(**maxima,prefix_errors={k:v.tolist() for k,v in prefix.items()},passes=bool(passes))


def _run_groups(groups, workers, progress=None, *, reducer=_reduce):
    """One spawn pool; every conditional completes before a group is reduced."""
    tasks = [(i,j,g) for i,g in enumerate(groups) for j in range(len(g['hypotheses']))]
    pending = {i:[None]*len(g['hypotheses']) for i,g in enumerate(groups)}
    rows = [None]*len(groups)
    manager = ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) if workers>1 else nullcontext(None)
    with manager as pool:
        results = map(_candidate_replay,tasks) if pool is None else (
            f.result() for f in as_completed([pool.submit(_candidate_replay,t) for t in tasks]))
        for completed,(i,j,c) in enumerate(results,1):
            if i not in pending or pending[i][j] is not None:
                raise RuntimeError('Joint study received a duplicate replay.')
            pending[i][j] = c
            if all(v is not None for v in pending[i]):
                rows[i] = reducer(groups[i],pending.pop(i))
            if progress is not None:
                progress(dict(completed=completed,total=len(tasks),pair=groups[i]['pair'],
                              sequence_id=groups[i]['sequence_id'],label=groups[i]['label']))
    if pending or any(row is None for row in rows):
        raise RuntimeError('Joint study did not complete every replay; no subset is reported.')
    return rows


def _run_groups_streaming(groups, workers, progress, accumulator_factory):
    """Bound in-flight results to 2*workers and consume in hypothesis order.

    A completed conditional is reduced immediately and then released. Ordered
    consumption makes summation independent of worker completion order; the
    bounded queue also prevents a slow early task from growing a result cache.
    No conditional result is shared across different experiments or settings.
    """
    tasks = ((i,j,g) for i,g in enumerate(groups) for j in range(len(g['hypotheses'])))
    total = sum(len(g['hypotheses']) for g in groups)
    manager = ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) if workers>1 else nullcontext(None)
    rows = [];current = None;accumulator = None;expected_j = 0
    with manager as pool:
        queue = deque()
        def enqueue():
            task = next(tasks,None)
            if task is not None:
                queue.append(pool.submit(_candidate_replay,task) if pool is not None else task)
        for _ in range(2*workers if pool is not None else 1): enqueue()
        completed = 0
        while queue:
            item = queue.popleft()
            i,j,conditional = item.result() if pool is not None else _candidate_replay(item)
            enqueue()
            if i!=current:
                if accumulator is not None: rows.append(accumulator.finish())
                if i!=len(rows): raise RuntimeError('Streaming bank order changed.')
                current=i;expected_j=0;accumulator=accumulator_factory(groups[i])
            if j!=expected_j: raise RuntimeError('Streaming hypothesis order changed.')
            accumulator.add(j,conditional);expected_j+=1;completed+=1
            del conditional,item
            if progress is not None:
                progress(dict(completed=completed,total=total,pair=groups[i]['pair'],
                              sequence_id=groups[i]['sequence_id'],label=groups[i]['label']))
        if accumulator is not None: rows.append(accumulator.finish())
    if completed!=total or len(rows)!=len(groups):
        raise RuntimeError('Streaming study did not complete every replay.')
    return rows


def run_joint_bank_refinement(*, workers=8, progress=None, sequence_ids=G1A_SEQUENCE_IDS,
                              grid_stride=10, numerics=None):
    """G1a: 42 fixed-measure banks / 1050 complete independent FV replays.

    API overrides are diagnostics and cannot pass as a registered report.
    One persistent spawn pool covers both pairs and every configuration.
    """
    workers = checks.positive_integer(workers,'workers')
    if len(sequence_ids)!=2:
        raise ValueError('G1a requires one ID collection per pair.')
    ids = tuple(tuple(core._nonnegative_integer(v,'sequence ID') for v in group) for group in sequence_ids)
    flat = [v for group in ids for v in group]
    if any(not group for group in ids) or len(set(flat))!=len(flat):
        raise ValueError('G1a IDs must be nonempty and distinct across both pairs.')
    stride = checks.positive_integer(grid_stride,'grid stride')
    if 40%stride:
        raise ValueError('Grid stride must divide 40 and retain both endpoints.')
    levels = G1A_NUMERICS if numerics is None else numerics
    if set(levels)!=set(G1A_NUMERICS):
        raise ValueError('G1a requires numerics for both registered pairs.')
    configs = {key:_configurations(value) for key,value in levels.items()}
    # Owned normalized scalars: mutating a report/callback cannot alter the lock.
    levels = {key:dict(time_steps=tuple(value[j][1]['max_step'] for j in (1,0,2)),
                       spatial_cells=tuple(value[j][1]['x_steps'] for j in (3,0,4)),
                       quadrature_orders=tuple(value[j][1]['likelihood_quadrature_order'] for j in (5,0,6)))
              for key,value in configs.items()}
    groups = []
    for pair,sids,Xi in zip(core.F4_G_PAIRS,ids,(.75,1.)):
        h = np.array([(a,b) for a in core.F4_E_PARAMETER_GRIDS[pair[0]][::stride]
                      for b in core.F4_E_PARAMETER_GRIDS[pair[1]][::stride]])
        case = core.build_f4_b_case(Xi=Xi)
        for sid in sids:
            sequence = core.generate_synthetic_sequence(case,sequence_id=sid)
            if not sequence.truth.valid:
                raise RuntimeError(f'G1a truth failed at ID{sid}; no repair or dropped record.')
            for label,settings in configs['__'.join(pair)]:
                groups.append(dict(pair=pair,sequence_id=sid,hypotheses=h,case=case,
                                   observations=sequence.observations,label=label,settings=settings))
    rows = _run_groups(groups,workers,progress)
    comparisons = []
    for pair,sids in zip(core.F4_G_PAIRS,ids):
        for sid in sids:
            selected = {r['label']:r for r in rows if tuple(r['coordinates'])==pair and r['sequence_id']==sid}
            for axis in ('time','spatial','quadrature'):
                for left,right,accept in ((axis+'_coarse','middle',False),('middle',axis+'_fine',True)):
                    result = compare_banks(selected[left],selected[right])
                    if axis=='time':
                        a,b = [np.array(selected[label]['conditional_actual_max_steps']) for label in (left,right)]
                        refined = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                                       and np.all((b>0)&(b<a*(1-1e-12))))
                        result.update(actual_step_refined=refined)
                        result['passes'] = bool(result['passes'] and refined)
                    comparisons.append(dict(coordinates=list(pair),sequence_id=sid,axis=axis,
                                            left=left,right=right,selected=accept,**result))
    case = core.build_f4_b_case()
    registered = ids==G1A_SEQUENCE_IDS and stride==10 and numerics is None
    return dict(stage='G1a',design_version=1,registered_design=registered,workers=workers,
                sequence_ids=[list(group) for group in ids],grid_stride=stride,conditional_runs=sum(len(g['hypotheses']) for g in groups),
                numerical_levels=levels,observation_contract=dict(time_unit='day',observation_kind='point',price_side='right'),
                inputs=dict(baseline=case.baseline,price_edges=case.price_edges.tolist(),price_values=case.price_values.tolist(),
                            state_response='1-2*x',price_response='1-2*u'),
                truth_settings=dict(initial_beta=[20.,20.],truth_max_step=case.truth_max_step,seed_roots=[4101,4102,4103]),
                criteria=dict(log_likelihood_window=5.,maximum_log_odds_error=.10,maximum_weight_tv=.01,
                              maximum_cumulative_evidence_error=.10,maximum_state_l1=.02,maximum_abs_mass_error=1e-10,
                              nonnegative_mass=True,actual_time_refinement=True,actual_step_relative_margin=1e-12),
                rows=rows,comparisons=comparisons,
                candidate_fv_numerics={key:dict(value[0][1]) for key,value in configs.items()},
                passes=bool(_probability_gate(rows) and all(c['passes'] for c in comparisons if c['selected'])),
                interpretation='FV numerics on a fixed 25-node joint measure; parameter-grid convergence is G1b, not certified here.')
