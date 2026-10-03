"""G2a v2 rectangular development pilot with resumable ordered reductions.

The design follows the rejected v1 record and its posthoc grid diagnosis.
It reuses those records, so it is not a held-out confirmation. An execution
block changes checkpoint frequency only, never the scientific design.
"""
from collections import deque

import numpy as np
from scipy.special import logsumexp

from src.validation import estimation as checks
from src.validation.bm_1 import estimation as core, joint_estimation as fv
from src.validation.bm_1 import joint_grid as grid, joint_diagnostics as hats


STAGE = 'G2a-rectangular'
DAYS = 7
ANCHOR = 129
GRIDS = {'rho__Xi': ((9, 65), (17, 129)),
         'sigma_x__sigma_y': ((33, 65), (65, 129))}
CELLS = {'rho__Xi': (1601, 3201), 'sigma_x__sigma_y': (6401, 12801)}
LABELS = ('middle', 'time_fine', 'spatial_fine', 'quadrature_fine')


def configurations(pair):
    middle, fine = CELLS['__'.join(pair)]
    def setting(cells=middle, step=1/1536, order=3):
        return dict(x_steps=cells, max_step=step, likelihood_quadrature_order=order)
    return [('middle', setting()), ('time_fine', setting(step=1/3072)),
            ('spatial_fine', setting(cells=fine)), ('quadrature_fine', setting(order=5))]


def units():
    return [(pair, sid, label) for pair, sid in zip(core.F4_G_PAIRS, grid.G2A_IDS)
            for label in LABELS]


def unit_name(unit):
    pair, sid, label = unit
    return f"{'_'.join(pair)}_{sid}_{label}"


def design_header():
    base = grid._g2a_header(DAYS, grid.G2A_IDS, (33, 65), fv.G1A_NUMERICS,
                           registered=True, backend='cached')
    for key in ('parameter_grid_counts', 'candidate_parameter_nodes_per_axis',
                'control_parameter_nodes_per_axis', 'numerical_levels'):
        del base[key]
    return dict(base, stage=STAGE, design_version=2,
        parameter_grids={k:[list(n) for n in v] for k, v in GRIDS.items()},
        physical_node_anchor=ANCHOR,
        fv_settings={'__'.join(pair):dict(configurations(pair)) for pair in core.F4_G_PAIRS},
        unit_names=[unit_name(u) for u in units()],
        expected_conditional_runs=sum(np.prod(GRIDS['__'.join(p)][1]).item() for p, _, _ in units()),
        design_origin='Fixed before v2 replay; data-informed by rejected v1 and its posthoc diagnosis',
        interpretation='Same-record numerical development pilot; passing v2 does not release the existing square G2b/G2c design.')


def require_previous(report, *, require_source=True):
    grid.validated_g2a(report, require_source=True)
    if report['record_days'] != DAYS or not report['completed'] or report['passes']:
        raise ValueError('Rectangular v2 requires the complete rejected seven-day G2a v1 report.')


def parameter_grid(pair, counts):
    """Physical nodes anchored once; first coordinate varies slowest."""
    counts = tuple(checks.positive_integer(n, 'axis nodes', minimum=2) for n in counts)
    if len(counts) != 2 or any(n > ANCHOR or (ANCHOR-1) % (n-1) for n in counts):
        raise ValueError('Rectangular nodes must be nested in the physical anchor.')
    full = grid._parameter_grid(pair, ANCHOR, ANCHOR)[0].reshape(ANCHOR, ANCHOR, 2)
    indices = [np.arange(0, ANCHOR, (ANCHOR-1)//(n-1)) for n in counts]
    hypotheses = full[indices[0], :][:, indices[1]].reshape(-1, 2)
    nodes = [np.linspace(0, 1, n) for n in counts]
    edges = [np.r_[0., (u[:-1]+u[1:])/2, 1.] for u in nodes]
    prior = np.log(np.outer(*[np.diff(e) for e in edges]).ravel())
    return hypotheses, nodes, edges, prior


def make_group(unit, truth=None):
    if unit not in units():
        raise ValueError('Unregistered rectangular pilot unit.')
    pair, sid, label = unit
    case, sequence, record = truth if truth is not None else grid._g2a_truth_record(pair, sid, DAYS)
    if not record['passes']:
        raise ValueError('Rectangular pilot truth gate failed; no repaired or dropped record.')
    counts = GRIDS['__'.join(pair)]
    return dict(stage=STAGE, pair=pair, sequence_id=sid, label=label, case=case,
        observations=sequence.observations, hypotheses=parameter_grid(pair, counts[-1])[0],
        counts=counts, settings=dict(configurations(pair))[label], prior_bridge=False,
        backend='cached', array_output=True)


class RectangularReduction(grid._StreamingGridReduction):
    """Reuse the ordered per-hypothesis Bayes sums, with independent axis sizes."""
    def __init__(self, group):
        self.group = group
        self.seen = 0
        self.banks = []
        count, cells = len(group['observations'].times), group['settings']['x_steps']
        finest = tuple(group['counts'][-1])
        for shape in group['counts']:
            shape = tuple(shape)
            if any((a-1) % (b-1) for a, b in zip(finest, shape)):
                raise ValueError('Candidate axes must be nested in the reference grid.')
            indices = np.arange(np.prod(finest)).reshape(finest)[
                ::(finest[0]-1)//(shape[0]-1), ::(finest[1]-1)//(shape[1]-1)].ravel()
            h, nodes, edges, prior = parameter_grid(group['pair'], shape)
            n = len(h)
            self.banks.append(dict(shape=shape, indices={int(j):i for i,j in enumerate(indices)},
                h=h, unit=nodes, edges=edges, prior=prior, scores=np.empty((count,n)),
                scale=np.full((2,count), -np.inf), denominator=np.zeros((2,count)),
                numerator=np.zeros((2,count,cells)), steps=np.empty(n),
                seconds=0., mass_error=0., minimum=np.inf, seen=0))

    def finish(self):
        if self.seen != len(self.group['hypotheses']):
            raise RuntimeError('Incomplete rectangular pilot bank.')
        rows = []
        for bank in self.banks:
            if bank['seen'] != len(bank['h']):
                raise RuntimeError('Incomplete nested rectangular grid.')
            reduced = fv.parameter_bank_weights(bank['scores'], bank['prior'], self.group['observations'].observed)
            evidence = np.cumsum(reduced['log_evidence'])
            fv._equal_array(bank['scale']+np.log(bank['denominator']),
                np.stack([np.r_[0., evidence[:-1]], evidence]), 'rectangular normalizers')
            marginal = bank['numerator']/bank['denominator'][:, :, None]
            diagnostics = dict(maximum_abs_mass_error=bank['mass_error'], minimum_cell_mass=bank['minimum'],
                conditional_actual_max_steps=bank['steps'].tolist(), conditional_seconds=bank['seconds'])
            row = fv._reduced_row(dict(self.group, hypotheses=bank['h'],
                label=f'{bank["shape"][0]}x{bank["shape"][1]}/{self.group["label"]}/quadrature'),
                bank['scores'], bank['prior'], reduced, *marginal, diagnostics)
            row.update(parameter_grid_shape=list(bank['shape']), prior_kind='quadrature',
                parameter_unit_nodes=[u.tolist() for u in bank['unit']],
                parameter_dual_edges=[e.tolist() for e in bank['edges']], fv_label=self.group['label'])
            rows.append(row)
        return rows

    def checkpoint(self):
        """Store initialized data only; no uninitialized scores or infinities."""
        if not self.seen:
            raise ValueError('An empty reducer has no checkpoint.')
        banks = []
        for bank in self.banks:
            local = [i for j, i in bank['indices'].items() if j < self.seen]
            banks.append(dict(scores=bank['scores'][:, local], steps=bank['steps'][local],
                **{k:bank[k] for k in ('scale', 'denominator', 'numerator', 'seconds', 'mass_error', 'minimum')}))
        return dict(completed_hypotheses=self.seen, banks=banks)

    def restore(self, payload):
        """Check every partial likelihood normalizer and state-mass sum."""
        seen = checks.positive_integer(payload['completed_hypotheses'], 'completed hypotheses')
        if seen > len(self.group['hypotheses']) or len(payload['banks']) != len(self.banks):
            raise ValueError('Checkpoint is outside the registered group.')
        count, cells = len(self.group['observations'].times), self.group['settings']['x_steps']
        for bank, saved in zip(self.banks, payload['banks']):
            local = [i for j, i in bank['indices'].items() if j < seen]
            scores = fv._array(saved['scores'], (count, len(local)), 'checkpoint scores')
            steps = fv._array(saved['steps'], (len(local),), 'checkpoint steps')
            if np.any((steps <= 0) | (steps > self.group['settings']['max_step']+1e-14)):
                raise ValueError('Checkpoint actual steps are invalid.')
            cumulative = np.cumsum(scores, axis=0)
            raw = bank['prior'][local] + np.stack([np.vstack([np.zeros(len(local)), cumulative[:-1]]), cumulative])
            scale = raw.max(axis=2)
            denominator = np.exp(raw-scale[:, :, None]).sum(axis=2)
            fv._equal_array(saved['scale'], scale, 'checkpoint scale')
            fv._equal_array(np.asarray(saved['denominator'])/denominator, np.ones((2,count)), 'checkpoint denominator')
            numerator = fv._array(saved['numerator'], (2,count,cells), 'checkpoint state sum')
            for phase in numerator / denominator[:, :, None]:
                fv._mass(phase, count, cells, 'checkpoint normalized state sum')
            for k in ('seconds', 'mass_error', 'minimum'):
                value = checks.finite_scalar(saved[k], 'checkpoint '+k)
                if value < 0:
                    raise ValueError('Checkpoint diagnostics must be nonnegative.')
                bank[k] = value
            if bank['mass_error'] > 1e-10:
                raise ValueError('Checkpoint conditional probability gate failed.')
            bank['scores'][:, local] = scores
            bank['steps'][local] = steps
            for k in ('scale', 'denominator', 'numerator'):
                bank[k] = np.asarray(saved[k]).copy()
            bank['seen'] = len(local)
        reference = self.banks[-1]
        for bank in self.banks[:-1]:
            for j, local in bank['indices'].items():
                if j >= seen: break
                fv._equal_array(bank['scores'][:,local], reference['scores'][:,j], 'checkpoint nested scores', 0.)
                fv._equal_array(bank['steps'][local], reference['steps'][j], 'checkpoint nested steps', 0.)
        self.seen = seen
        return self


def advance(reducer, stop, pool=None, progress=None, *, workers=1):
    """One bounded ordered block; a caller may retain its spawn pool across blocks."""
    if not reducer.seen < stop <= len(reducer.group['hypotheses']):
        raise ValueError('Invalid checkpoint block endpoint.')
    tasks = iter((0, j, reducer.group) for j in range(reducer.seen, stop))
    queue = deque()
    def enqueue():
        task = next(tasks, None)
        if task is not None:
            queue.append(pool.submit(fv._candidate_replay, task) if pool is not None else task)
    for _ in range(2*workers if pool is not None else 1):
        enqueue()
    while queue:
        item = queue.popleft()
        _, j, result = item.result() if pool is not None else fv._candidate_replay(item)
        enqueue()
        reducer.add(j, result)
        del result, item
        if progress is not None:
            progress(reducer.seen, len(reducer.group['hypotheses']))


def validate_rows(rows, group):
    """Rebuild final raw algebra and experiment metadata without FV reruns."""
    shapes = [tuple(n) for n in group['counts']]
    if len(rows) != len(shapes) or [tuple(r['parameter_grid_shape']) for r in rows] != shapes:
        raise ValueError('Missing, duplicate or reordered rectangular banks.')
    case, obs = group['case'], group['observations']
    count, cells = len(obs.times), group['settings']['x_steps']
    truth = core.f4_e_truth_candidate(case)
    for row, shape in zip(rows, shapes):
        h, nodes, edges, prior = parameter_grid(group['pair'], shape)
        metadata = dict(coordinates=list(group['pair']), sequence_id=group['sequence_id'],
            label=f'{shape[0]}x{shape[1]}/{group["label"]}/quadrature', fv_label=group['label'],
            settings=group['settings'], prior_kind='quadrature', observed=[True]*count,
            truth=[getattr(truth, p) for p in group['pair']],
            fixed_parameters={p:getattr(truth,p) for p in ('rho','Xi','k','sigma_x','sigma_y') if p not in group['pair']})
        for k, value in metadata.items(): grid._check_recorded(row.get(k), value, 'rectangular '+k)
        for k, value in dict(hypotheses=h, initial_log_weights=prior, observation_times=obs.times,
            observations=obs.values, initial_mass=core.beta_initial_mass(case,case.build_model()[0],cells),
            x_edges=np.linspace(0,1,cells+1)).items(): fv._equal_array(row.get(k), value, 'rectangular '+k)
        for k, value in (('parameter_unit_nodes', nodes), ('parameter_dual_edges', edges)):
            if len(row[k]) != 2: raise ValueError('Rectangular axes are incomplete.')
            for got, want in zip(row[k], value): fv._equal_array(got, want, 'rectangular '+k)
        scores = fv._array(row['conditional_log_evidence'], (count,len(h)), 'rectangular scores')
        raw = prior + np.cumsum(scores, axis=0)
        evidence = logsumexp(raw, axis=1); logs = raw-evidence[:,None]
        for k, value in dict(log_weights=logs, cumulative_log_evidence=evidence,
            bank_log_evidence=np.diff(np.r_[0.,evidence])).items(): fv._equal_array(row[k], value, 'rectangular '+k)
        for k, value in core._f4_g_joint_summary(h, logs, group['pair']).items():
            grid._check_recorded(row.get(k), value, 'rectangular summary '+k)
        for phase in ('predicted', 'posterior'):
            fv._mass(row['marginal_'+phase+'_mass'], count, cells, 'rectangular '+phase)
        steps = fv._array(row['conditional_actual_max_steps'], (len(h),), 'rectangular actual steps')
        if np.any((steps <= 0) | (steps > group['settings']['max_step']+1e-14)):
            raise ValueError('Rectangular actual steps are invalid.')
        if checks.finite_scalar(row['conditional_seconds'], 'conditional seconds') < 0:
            raise ValueError('Conditional seconds must be nonnegative.')
    # Both reductions came from the same independently continued hypotheses.
    fine = rows[-1]; shape = shapes[-1]
    for row, coarse in zip(rows[:-1], shapes[:-1]):
        index = np.arange(np.prod(shape)).reshape(shape)[
            ::(shape[0]-1)//(coarse[0]-1), ::(shape[1]-1)//(coarse[1]-1)].ravel()
        for field in ('conditional_log_evidence', 'conditional_actual_max_steps'):
            values = np.asarray(fine[field])
            want = values[:, index] if values.ndim == 2 else values[index]
            fv._equal_array(row[field], want, 'nested rectangular '+field, 0.)
    return True


def compare_parameter_rows(left, right):
    for key in ('coordinates', 'sequence_id', 'observation_times', 'observations', 'settings', 'x_edges'):
        grid._check_recorded(left[key], right[key], 'parameter comparison '+key)
    laws = [hats.tensor_hat_law(r['conditional_log_evidence'], r['parameter_grid_shape']) for r in (left,right)]
    comparison = hats.compare_rectangular_hats(*laws)
    prefix = comparison['prefix_errors']
    for phase in ('predicted', 'posterior'):
        mass = [fv._mass(r['marginal_'+phase+'_mass'], len(laws[0]['mass']), len(r['x_edges'])-1,
                        'parameter '+phase) for r in (left,right)]
        prefix[phase+'_l1'] = abs(mass[1]-mass[0]).sum(axis=1).tolist()
    maximum = {'maximum_'+k:float(np.max(v)) for k,v in prefix.items()}
    return dict(**maximum, prefix_errors=prefix, passes=bool(comparison['parameter_only_thresholds_pass']
        and max(maximum['maximum_predicted_l1'], maximum['maximum_posterior_l1']) <= .02))


def summarize(rows):
    """Only all eight complete raw banks can grant numerical v2 acceptance."""
    comparisons, checkpoints = [], []
    for pair, sid in zip(core.F4_G_PAIRS, grid.G2A_IDS):
        shapes = GRIDS['__'.join(pair)]
        lookup = {(tuple(r['parameter_grid_shape']),r['fv_label']):r for r in rows if tuple(r['coordinates'])==pair}
        for shape in shapes:
            if (shape, 'middle') not in lookup: continue
            middle = lookup[shape, 'middle']
            for axis in ('time', 'spatial', 'quadrature'):
                if (shape, axis+'_fine') not in lookup: continue
                fine = lookup[shape, axis+'_fine']
                comparison = fv.compare_banks(middle, fine)
                if axis == 'time':
                    refined = bool(np.all(np.asarray(fine['conditional_actual_max_steps']) <
                        np.asarray(middle['conditional_actual_max_steps'])*(1-1e-12)))
                    comparison.update(actual_step_refined=refined, passes=bool(comparison['passes'] and refined))
                comparisons.append(dict(coordinates=list(pair), sequence_id=sid, kind='FV_'+axis,
                    parameter_grid_shape=list(shape), left=middle['label'], right=fine['label'], **comparison))
        if all((shape, 'middle') in lookup for shape in shapes):
            a, b = [lookup[shape, 'middle'] for shape in shapes]
            comparisons.append(dict(coordinates=list(pair), sequence_id=sid, kind='parameter_hat',
                left=a['label'], right=b['label'], **compare_parameter_rows(a,b)))
            law = hats.tensor_hat_law(a['conditional_log_evidence'], a['parameter_grid_shape'])
            for day in grid.G2A_DAYS:
                if day > DAYS: continue
                k = 24*day-1
                checkpoints.append(dict(coordinates=list(pair), sequence_id=sid, days=day,
                    observation_count=k+1, hat_unit_mean=law['mean'][k].tolist(),
                    hat_unit_sd=law['sd'][k].tolist(), hat_unit_covariance=float(law['covariance'][k]),
                    cumulative_log_evidence=float(law['evidence'][k]),
                    mean_predictive_y_log_score=float(law['evidence'][k]/(k+1))))
    complete = len(rows)==16 and len(comparisons)==14
    return dict(stage=STAGE, design_version=2, completed=complete, bank_reductions=len(rows),
        comparisons=comparisons, checkpoints=checkpoints,
        passes=bool(complete and fv._probability_gate(rows) and all(c['passes'] for c in comparisons)),
        failed_comparisons=[dict(coordinates=c['coordinates'], kind=c['kind'], left=c['left'], right=c['right'])
                            for c in comparisons if not c['passes']],
        releases_existing_g2b_g2c=False)
