"""G1b0 pilot and G1b1 paired grid refinement before a larger population.

Conditional trajectories are shared only across quadratures of the same
experiment. Each hypothesis retains its own state; every FV control is fresh.
"""

import re
from copy import deepcopy

import numpy as np
from scipy.special import logsumexp

from src.validation import estimation as checks
from src.validation.bm_1 import estimation as core
from src.validation.bm_1 import joint_estimation as fv
from src.params.bm_1 import conservative_mass_transfer


G1B0_IDS = (1356, 1357)
G1B0_COUNTS = (3, 5, 9)
G1B1_COUNTS = (5, 9, 17)
G1B2_COUNTS = (5, 9, 17, 33)
G1B3_COUNTS = (5, 9, 17, 33, 65)
G1B0_CRITERIA = dict(maximum_parameter_tv=.02, maximum_unit_mean_error=.01,
                     maximum_unit_sd_error=.02, maximum_cumulative_evidence_error=.10,
                     maximum_state_l1=.02)


def validated_g1a(report, *, require_source=False):
    """Recompute the complete registered G1a dependency, including all gates."""
    header = dict(stage='G1a', design_version=1, registered_design=True,
                  sequence_ids=[list(g) for g in fv.G1A_SEQUENCE_IDS], grid_stride=10,
                  conditional_runs=1050, passes=True,
                  observation_contract=dict(time_unit='day', observation_kind='point', price_side='right'),
                  inputs=dict(baseline=.5,price_edges=[0.,1/3,2/3,1.],price_values=[.65,.35,.65],
                              state_response='1-2*x',price_response='1-2*u'),
                  truth_settings=dict(initial_beta=[20.,20.],truth_max_step=1/1152,seed_roots=[4101,4102,4103]),
                  criteria=dict(log_likelihood_window=5.,maximum_log_odds_error=.10,maximum_weight_tv=.01,
                                maximum_cumulative_evidence_error=.10,maximum_state_l1=.02,
                                maximum_abs_mass_error=1e-10,nonnegative_mass=True,
                                actual_time_refinement=True,actual_step_relative_margin=1e-12))
    if any(report.get(k)!=v for k,v in header.items()) or report.get('passes') is not True or report.get('registered_design') is not True:
        raise ValueError('A complete passing registered G1a report is required.')
    if require_source and (report.get('working_tree_dirty') is not False
                           or not re.fullmatch('[0-9a-f]{40}',str(report.get('source_commit')))):
        raise ValueError('G1a requires a clean recorded source commit.')
    dependency = report.get('dependencies',{}).get('g0',{})
    if (not re.fullmatch('[0-9a-f]{64}',str(dependency.get('sha256')))
            or not re.fullmatch('[0-9a-f]{40}',str(dependency.get('source_commit')))):
        raise ValueError('G1a requires its recorded G0 dependency.')
    levels = {k:{n:list(v) for n,v in values.items()} for k,values in fv.G1A_NUMERICS.items()}
    actual_levels = report.get('numerical_levels',{})
    if (set(actual_levels)!=set(levels)
            or any(fv._configurations(actual_levels[k])!=fv._configurations(levels[k]) for k in levels)):
        raise ValueError('G1a numerical levels were changed.')
    candidates = {k:fv._configurations(v)[0][1] for k,v in levels.items()}
    if report.get('candidate_fv_numerics')!=candidates:
        raise ValueError('G1a candidate numerics were changed.')
    expected, sequences = {}, {}
    for pair,ids,Xi in zip(core.F4_G_PAIRS,fv.G1A_SEQUENCE_IDS,(.75,1.)):
        h = np.array([(a,b) for a in core.F4_E_PARAMETER_GRIDS[pair[0]][::10]
                      for b in core.F4_E_PARAMETER_GRIDS[pair[1]][::10]])
        case = core.build_f4_b_case(Xi=Xi)
        for sid in ids:
            sequence = core.generate_synthetic_sequence(case,sequence_id=sid)
            sequences[sid] = (case,sequence)
            for label,settings in fv._configurations(levels['__'.join(pair)]):
                expected[(pair,sid,label)] = (h,settings)
    rows = report.get('rows',[])
    if len(rows)!=42:
        raise ValueError('G1a requires all 42 banks.')
    seen = {}
    for r in rows:
        key = (tuple(r['coordinates']),r['sequence_id'],r['label'])
        if key not in expected or key in seen:
            raise ValueError('G1a banks are missing, duplicate or unregistered.')
        seen[key] = r
        h,settings = expected[key];case,seq = sequences[r['sequence_id']]
        truth = core.f4_e_truth_candidate(case)
        if (r['settings']!=settings or r['observed']!=[True]*24
                or r['truth']!=[getattr(truth,n) for n in key[0]]
                or r['fixed_parameters']!={n:getattr(truth,n) for n in ('rho','Xi','k','sigma_x','sigma_y') if n not in key[0]}):
            raise ValueError('G1a experiment/settings were changed.')
        for name,value in [('hypotheses',h),('initial_log_weights',np.full(25,-np.log(25))),
                           ('observation_times',seq.observations.times),('observations',seq.observations.values),
                           ('x_edges',np.linspace(0,1,settings['x_steps']+1)),
                           ('initial_mass',core.beta_initial_mass(case,case.build_model()[0],settings['x_steps']))]:
            fv._equal_array(r[name],value,'G1a '+name)
        scores = fv._array(r['conditional_log_evidence'],(24,25),'G1a scores')
        raw = np.full(25,-np.log(25))+np.cumsum(scores,axis=0)
        evidence = logsumexp(raw,axis=1);logs = raw-evidence[:,None]
        fv._equal_array(r['log_weights'],logs,'G1a weights')
        fv._equal_array(r['cumulative_log_evidence'],evidence,'G1a evidence')
        bank = fv._array(r['bank_log_evidence'],(24,),'G1a bank increments')
        fv._equal_array(np.cumsum(bank),evidence,'G1a bank evidence')
        summary = core._f4_g_joint_summary(h,logs,key[0])
        for name,value in summary.items():
            if name=='map_hypotheses':
                if r[name]!=value: raise ValueError('G1a MAP ties disagree.')
            elif name=='marginal_parameters':
                if set(r[name])!=set(value): raise ValueError('G1a marginal coordinates disagree.')
                for coordinate,marginal in value.items():
                    for field,want in marginal.items():
                        fv._equal_array(r[name][coordinate][field],want,'G1a marginal '+field)
            else: fv._equal_array(r[name],value,'G1a '+name)
        for phase in ('predicted','posterior'):
            fv._mass(r['marginal_'+phase+'_mass'],24,settings['x_steps'],'G1a '+phase)
        step = fv._array(r['conditional_actual_max_steps'],(25,),'G1a actual steps')
        if np.any((step<=0)|(step>settings['max_step']+1e-14)):
            raise ValueError('G1a actual steps are invalid.')
    fv._probability_gate(rows,raise_error=True)
    comparisons = report.get('comparisons',[])
    if len(comparisons)!=36: raise ValueError('G1a requires all 36 comparisons.')
    seen_comparisons = set()
    for c in comparisons:
        pair,sid,axis,selected = tuple(c['coordinates']),c['sequence_id'],c['axis'],c['selected']
        if axis not in ('time','spatial','quadrature') or not isinstance(selected,bool):
            raise ValueError('G1a comparison axis/selection is invalid.')
        key = (pair,sid,axis,selected)
        if key in seen_comparisons: raise ValueError('G1a duplicate comparison.')
        seen_comparisons.add(key)
        left,right = ('middle',axis+'_fine') if selected else (axis+'_coarse','middle')
        if c['left']!=left or c['right']!=right: raise ValueError('G1a comparison selection changed.')
        if (pair,sid,left) not in seen or (pair,sid,right) not in seen:
            raise ValueError('G1a comparison references an unregistered bank.')
        a,b = seen[(pair,sid,left)],seen[(pair,sid,right)]
        calculated = fv.compare_banks(a,b)
        if axis=='time':
            refined = bool(np.all(np.array(b['conditional_actual_max_steps'])<
                                  np.array(a['conditional_actual_max_steps'])*(1-1e-12)))
            if c.get('actual_step_refined') is not refined: raise ValueError('G1a time gate disagrees.')
            calculated['passes'] = calculated['passes'] and refined
        for name,value in calculated.items():
            if name=='prefix_errors':
                if set(c[name])!=set(value): raise ValueError('G1a prefix gates changed.')
                for field,want in value.items(): fv._equal_array(c[name][field],want,'G1a prefix '+field)
            elif name=='passes':
                if c.get(name) is not bool(value): raise ValueError('G1a gate disagrees.')
            else: fv._equal_array(c[name],value,'G1a '+name)
        if selected and not calculated['passes']: raise ValueError('G1a active gate failed.')
    return True


def _parameter_grid(pair, count, finest):
    """Nested physical nodes and prior dual cells in unit prior coordinates."""
    unit = np.linspace(0,1,finest)[::(finest-1)//(count-1)]
    axes = []
    for name in pair:
        old = np.asarray(core.F4_E_PARAMETER_GRIDS[name])[::10]
        full = (np.geomspace(old[0],old[-1],finest) if name.startswith('sigma')
                else np.linspace(old[0],old[-1],finest))
        # Preserve G0's physical nodes exactly wherever the nested grid contains them.
        if (finest-1)%4==0: full[::(finest-1)//4] = old
        axes.append(full[::(finest-1)//(count-1)])
    edges = np.r_[0.,(unit[:-1]+unit[1:])/2,1.]
    prior = np.outer(np.diff(edges),np.diff(edges)).ravel()
    return np.array([(a,b) for a in axes[0] for b in axes[1]]),unit,edges,np.log(prior)


def _reduce_grids(group, conditionals):
    rows = []
    finest = group['counts'][-1]
    for n in group['counts']:
        step = (finest-1)//(n-1)
        indices = np.arange(finest**2).reshape(finest,finest)[::step,::step].ravel()
        h,unit,edges,prior = _parameter_grid(group['pair'],n,finest)
        bridge = group.get('prior_bridge',True) and n==group['counts'][1] and group['label']=='middle'
        for kind in ('quadrature','discrete_uniform') if bridge else ('quadrature',):
            g = dict(group,hypotheses=h,initial_log_weights=prior if kind=='quadrature' else np.full(n*n,-np.log(n*n)),
                     label=f'{n}/{group["label"]}/{kind}')
            row = fv._reduce(g,[conditionals[j] for j in indices])
            row.update(parameter_nodes_per_axis=n,prior_kind=kind,parameter_unit_nodes=unit.tolist(),
                       parameter_dual_edges=edges.tolist(),fv_label=group['label'])
            rows.append(row)
    return rows


class _StreamingGridReduction:
    """Stable online sums of independent conditional state laws.

    For prefix k, prediction uses pi_h exp(L_{k-1,h}), posterior pi_h
    exp(L_{k,h}). Each sum is rescaled by its running largest log weight.
    Division by the corresponding weight sum is deferred until all hypotheses
    have arrived. Only the marginal numerators and raw scores are retained.
    """
    def __init__(self, group):
        if group.get('prior_bridge',True) or not np.all(group['observations'].observed):
            raise ValueError('Streaming grids require quadrature-only fully observed records.')
        self.group = group
        self.seen = 0
        self.banks = []
        finest = group['counts'][-1]
        count = len(group['observations'].times)
        cells = group['settings']['x_steps']
        for n in group['counts']:
            step = (finest-1)//(n-1)
            indices = np.arange(finest*finest).reshape(finest,finest)[::step,::step].ravel()
            h,unit,edges,prior = _parameter_grid(group['pair'],n,finest)
            bank = dict(n=n,indices={int(j):local for local,j in enumerate(indices)},
                        h=h,unit=unit,edges=edges,prior=prior,scores=np.empty((count,n*n)),
                        scale=np.full((2,count),-np.inf),denominator=np.zeros((2,count)),
                        numerator=np.zeros((2,count,cells)),steps=np.empty(n*n),
                        seconds=0.,mass_error=0.,minimum=np.inf,seen=0)
            self.banks.append(bank)

    def add(self, j, conditional):
        if j!=self.seen: raise RuntimeError('Streaming reduction requires every hypothesis in order.')
        self.seen+=1
        count = len(self.group['observations'].times)
        cells = self.group['settings']['x_steps']
        for name in ('maximum_abs_mass_error','minimum_cell_mass','actual_max_step','elapsed_seconds'):
            value = checks.finite_scalar(conditional[name],name)
            if name!='minimum_cell_mass' and value<0:
                raise ValueError('Streaming replay diagnostics must be nonnegative.')
        scores = fv._array(conditional['scores'],(count,),'streaming scores')
        states = np.stack([fv._mass(conditional[p],count,cells,'streaming '+p) for p in ('predicted','posterior')])
        cumulative = np.cumsum(scores)
        log_likelihood = np.stack([np.r_[0.,cumulative[:-1]],cumulative])
        for bank in self.banks:
            local = bank['indices'].get(j)
            if local is None: continue
            log_weight = bank['prior'][local]+log_likelihood
            scale = np.maximum(bank['scale'],log_weight)
            old_factor = np.exp(bank['scale']-scale)
            factor = np.exp(log_weight-scale)
            bank['numerator'] *= old_factor[:,:,None]
            bank['numerator'] += factor[:,:,None]*states
            bank['denominator'] *= old_factor
            bank['denominator'] += factor
            bank['scale'] = scale
            bank['scores'][:,local] = scores
            bank['steps'][local] = conditional['actual_max_step']
            bank['seconds'] += conditional['elapsed_seconds']
            bank['mass_error'] = max(bank['mass_error'],conditional['maximum_abs_mass_error'])
            bank['minimum'] = min(bank['minimum'],conditional['minimum_cell_mass'])
            bank['seen']+=1

    def finish(self):
        if self.seen!=len(self.group['hypotheses']):
            raise RuntimeError('Incomplete streaming parameter grid.')
        rows = []
        for bank in self.banks:
            n = bank['n']
            if bank['seen']!=n*n: raise RuntimeError('Incomplete nested streaming grid.')
            reduced = fv.parameter_bank_weights(bank['scores'],bank['prior'],self.group['observations'].observed)
            # Independent normalizers provide an algebraic guard on old/new weights.
            evidence = np.cumsum(reduced['log_evidence'])
            normalizers = bank['scale']+np.log(bank['denominator'])
            fv._equal_array(normalizers,np.stack([np.r_[0.,evidence[:-1]],evidence]),'streaming normalizers')
            marginal = bank['numerator']/bank['denominator'][:,:,None]
            diagnostics = dict(maximum_abs_mass_error=bank['mass_error'],minimum_cell_mass=bank['minimum'],
                               conditional_actual_max_steps=bank['steps'].tolist(),conditional_seconds=bank['seconds'])
            group = dict(self.group,hypotheses=bank['h'],label=f'{n}/{self.group["label"]}/quadrature')
            row = fv._reduced_row(group,bank['scores'],bank['prior'],reduced,marginal[0],marginal[1],diagnostics)
            row.update(parameter_nodes_per_axis=n,prior_kind='quadrature',parameter_unit_nodes=bank['unit'].tolist(),
                       parameter_dual_edges=bank['edges'].tolist(),fv_label=self.group['label'])
            rows.append(row)
        return rows


def compare_parameter_banks(left, right, *, allow_prior_change=False):
    """TV of joint dual-cell laws on their common refinement, at all prefixes.

    Reconstruct constant density in each parameter dual cell solely to compare
    quadratures. This never changes Bayes weights or any conditional state.
    """
    for name in ('coordinates','sequence_id','observation_times','observations','observed',
                 'truth','fixed_parameters','settings'):
        if left[name]!=right[name]: raise ValueError('Parameter comparison requires the same experiment and FV numerics.')
    if not allow_prior_change and (left['prior_kind']!='quadrature' or right['prior_kind']!='quadrature'):
        raise ValueError('Grid convergence requires the same continuous prior measure.')
    count = len(left['observation_times']);laws = [];means = [];sds = []
    edges = np.unique(np.r_[left['parameter_dual_edges'],right['parameter_dual_edges']])
    for row in (left,right):
        n = checks.positive_integer(row['parameter_nodes_per_axis'],'parameter nodes',minimum=2)
        pair = tuple(row['coordinates'])
        if pair not in core.F4_G_PAIRS: raise ValueError('Unknown parameter pair.')
        fv._equal_array(row['hypotheses'],_parameter_grid(pair,n,n)[0],'physical parameter grid',1e-12)
        unit = np.linspace(0,1,n);dual = np.r_[0.,(unit[:-1]+unit[1:])/2,1.]
        fv._equal_array(row['parameter_unit_nodes'],unit,'unit parameter nodes',0.)
        fv._equal_array(row['parameter_dual_edges'],dual,'parameter dual edges',0.)
        scores = fv._array(row['conditional_log_evidence'],(count,n*n),'grid scores')
        prior = fv._array(row['initial_log_weights'],(n*n,),'grid prior')
        expected_prior = (np.log(np.outer(np.diff(dual),np.diff(dual)).ravel())
                          if row['prior_kind']=='quadrature' else np.full(n*n,-np.log(n*n)))
        if row['prior_kind'] not in ('quadrature','discrete_uniform'): raise ValueError('Unknown parameter prior.')
        fv._equal_array(prior,expected_prior,'grid quadrature prior')
        raw = prior+np.cumsum(scores,axis=0);evidence = logsumexp(raw,axis=1)
        log = raw-evidence[:,None];fv._equal_array(row['log_weights'],log,'grid weights')
        fv._equal_array(row['cumulative_log_evidence'],evidence,'grid evidence')
        weights = np.exp(log)
        coordinates = np.array([(a,b) for a in unit for b in unit])
        mean = weights@coordinates
        means.append(mean);sds.append(np.sqrt(np.sum(weights[:,:,None]*(coordinates[None,:,:]-mean[:,None,:])**2,axis=1)))
        transfer = np.column_stack([conservative_mass_transfer(dual,v,edges) for v in np.eye(n)])
        laws.append(np.einsum('ai,kij,bj->kab',transfer,weights.reshape(count,n,n),transfer,optimize=True))
    delta = np.abs(laws[1]-laws[0])
    prefix = dict(parameter_tv=.5*delta.sum(axis=(1,2)),
                  unit_mean_error=np.max(abs(means[1]-means[0]),axis=1),
                  unit_sd_error=np.max(abs(sds[1]-sds[0]),axis=1),
                  cumulative_evidence_error=abs(np.array(right['cumulative_log_evidence'])-left['cumulative_log_evidence']))
    for phase in ('predicted','posterior'):
        mass = [fv._mass(r['marginal_'+phase+'_mass'],count,len(r['x_edges'])-1,'grid '+phase) for r in (left,right)]
        if left['x_edges']!=right['x_edges']: raise ValueError('Parameter comparison requires the same state grid.')
        prefix[phase+'_l1'] = np.abs(mass[1]-mass[0]).sum(axis=1)
    maxima = {'maximum_'+k:float(v.max()) for k,v in prefix.items()}
    passes = (all(maxima[k]<=v for k,v in G1B0_CRITERIA.items() if k!='maximum_state_l1')
              and max(maxima['maximum_predicted_l1'],maxima['maximum_posterior_l1'])<=G1B0_CRITERIA['maximum_state_l1'])
    return dict(**maxima,prefix_errors={k:v.tolist() for k,v in prefix.items()},passes=bool(passes))


def run_joint_parameter_grid(*, workers=8, progress=None, sequence_ids=G1B0_IDS,
                             counts=G1B0_COUNTS, numerics=None):
    """G1b0 pilot: 648 fresh replays, 26 reductions, 14 active comparisons."""
    return _run_joint_grid(workers=workers,progress=progress,sequence_ids=sequence_ids,
                           counts=counts,numerics=numerics,stage='G1b0',design_version=1,
                           registered_counts=G1B0_COUNTS,prior_bridge=True)


def run_joint_parameter_grid_refinement(*, workers=8, progress=None, sequence_ids=G1B0_IDS,
                                        counts=G1B1_COUNTS, numerics=None):
    """G1b1 v2: 2312 fresh replays, 24 reductions, 14 active comparisons.

    The same records, support, continuous prior and gates are retained. No
    parameter prior bridge is repeated; the original G1b0 artifact retains it.
    The CLI additionally checks reproducibility against its raw G1b0 dependency.
    """
    return _run_joint_grid(workers=workers,progress=progress,sequence_ids=sequence_ids,
                           counts=counts,numerics=numerics,stage='G1b1',design_version=2,
                           registered_counts=G1B1_COUNTS,prior_bridge=False)


def run_joint_parameter_grid_fine(*, workers=8, progress=None, sequence_ids=G1B0_IDS,
                                  counts=G1B2_COUNTS, numerics=None):
    """G1b2 v3: 17x17 vs 33x33; 5/9 banks reproduce G1b0 without a G1b1 rerun."""
    result = _run_joint_grid(workers=workers,progress=progress,sequence_ids=sequence_ids,
                            counts=counts,numerics=numerics,stage='G1b2',design_version=3,
                            registered_counts=G1B2_COUNTS,prior_bridge=False,streaming=True)
    result['conditional_backend'] = 'lapack-tridiagonal; unchanged FV scheme'
    result['reduction'] = 'ordered streaming log-scaled sums; at most 2*workers in flight'
    return result


def run_joint_parameter_grid_dense(*, workers=8, progress=None, sequence_ids=G1B0_IDS,
                                   counts=G1B3_COUNTS, numerics=None):
    """G1b3 v4: 33x33 vs 65x65, retaining all coarser comparisons."""
    result = _run_joint_grid(workers=workers,progress=progress,sequence_ids=sequence_ids,
                            counts=counts,numerics=numerics,stage='G1b3',design_version=4,
                            registered_counts=G1B3_COUNTS,prior_bridge=False,streaming=True)
    result['conditional_backend'] = 'lapack-tridiagonal; unchanged FV scheme'
    result['reduction'] = 'ordered streaming log-scaled sums; at most 2*workers in flight'
    return result


def _grid_configurations(levels):
    if set(levels)!=set(fv.G1A_NUMERICS): raise ValueError('Joint grids require FV controls for both pairs.')
    return {k:[v for v in fv._configurations(values)
               if v[0] in ('middle','time_fine','spatial_fine','quadrature_fine')]
            for k,values in levels.items()}


def _run_joint_grid(*, workers, progress, sequence_ids, counts, numerics, stage,
                    design_version, registered_counts, prior_bridge, streaming=False):
    workers = checks.positive_integer(workers,'workers')
    ids = tuple(core._nonnegative_integer(v,'sequence ID') for v in sequence_ids)
    sizes = tuple(checks.positive_integer(v,'parameter nodes',minimum=2) for v in counts)
    if (len(ids)!=2 or len(set(ids))!=2 or len(sizes)!=len(registered_counts) or any(a>=b for a,b in zip(sizes,sizes[1:]))
            or any((sizes[-1]-1)%(n-1) for n in sizes)):
        raise ValueError('Joint grids require two distinct IDs and the declared nested increasing parameter grids.')
    levels = fv.G1A_NUMERICS if numerics is None else numerics
    configurations = _grid_configurations(levels)
    groups = []
    for pair,sid,Xi in zip(core.F4_G_PAIRS,ids,(.75,1.)):
        case = core.build_f4_b_case(Xi=Xi);seq = core.generate_synthetic_sequence(case,sequence_id=sid)
        if not seq.truth.valid: raise RuntimeError(f'{stage} truth failed at ID{sid}; no dropped record.')
        h = _parameter_grid(pair,sizes[-1],sizes[-1])[0]
        for label,settings in configurations['__'.join(pair)]:
            groups.append(dict(stage=stage,pair=pair,sequence_id=sid,case=case,observations=seq.observations,
                               hypotheses=h,counts=sizes,label=label,settings=settings,prior_bridge=prior_bridge,
                               backend='lapack' if streaming else 'reference'))
    if streaming:
        banks = fv._run_groups_streaming(groups,workers,progress,_StreamingGridReduction)
    else:
        banks = fv._run_groups(groups,workers,progress,reducer=_reduce_grids)
    rows = [r for bank in banks for r in bank]
    comparisons = _grid_comparisons(rows,ids,sizes,prior_bridge=prior_bridge)
    return dict(stage=stage,design_version=design_version,
                registered_design=bool(ids==G1B0_IDS and sizes==registered_counts and numerics is None),
                sequence_ids=list(ids),parameter_grid_counts=list(sizes),workers=workers,
                conditional_runs=sum(len(g['hypotheses']) for g in groups),bank_reductions=len(rows),
                prior_measure=dict(rho__Xi='product uniform in rho and Xi',sigma_x__sigma_y='product uniform in log(sigma_x) and log(sigma_y)',
                                   quadrature='tensor trapezoid in normalized prior coordinates',
                                   comparison='piecewise constant dual-cell laws; comparison only'),
                candidate_fv_numerics={k:dict(v[0][1]) for k,v in configurations.items()},
                fv_controls={k:{label:dict(s) for label,s in values} for k,values in configurations.items()},
                parameter_criteria=dict(G1B0_CRITERIA),fv_criteria=dict(log_odds=.10,weight_tv=.01,evidence=.10,state_l1=.02,
                                                                     mass_error=1e-10,actual_step_relative_margin=1e-12),
                observation_contract=dict(time_unit='day',observation_kind='point',price_side='right'),
                inputs=dict(baseline=.5,price_edges=[0.,1/3,2/3,1.],price_values=[.65,.35,.65],state_response='1-2*x',price_response='1-2*u'),
                truth_settings=dict(initial_beta=[20.,20.],truth_max_step=1/1152,seed_roots=[4101,4102,4103]),
                candidate_parameter_nodes_per_axis=sizes[-2],rows=rows,comparisons=comparisons,
                passes=bool(fv._probability_gate(rows) and all(c['passes'] for c in comparisons if c['selected'])),
                interpretation=('New-record grid/prior pilot; a larger held-out population and G2 remain separate.'
                                if stage=='G1b0' else ('Paired refinement of rejected G1b0; same records, not held-out recovery.'
                                if stage=='G1b1' else 'Consolidated refinement after rejected grids; same records, not held-out recovery.')))


def _grid_comparisons(rows, ids, sizes, *, prior_bridge, include_first_fv=False):
    comparisons = []
    for pair,sid in zip(core.F4_G_PAIRS,ids):
        lookup = {r['label']:r for r in rows if tuple(r['coordinates'])==pair and r['sequence_id']==sid}
        # Certify both the candidate measure and the control containing new nodes.
        # These reductions already exist: no extra conditional replay is needed.
        for n in (sizes if include_first_fv else sizes[1:]):
            for axis in ('time','spatial','quadrature'):
                a,b = [lookup[f'{n}/{label}/quadrature'] for label in ('middle',axis+'_fine')]
                result = fv.compare_banks(a,b)
                if axis=='time':
                    refined = bool(np.all(np.array(b['conditional_actual_max_steps'])<
                                          np.array(a['conditional_actual_max_steps'])*(1-1e-12)))
                    result.update(actual_step_refined=refined)
                    result['passes'] = bool(result['passes'] and refined)
                comparisons.append(dict(coordinates=list(pair),sequence_id=sid,kind='FV_'+axis,
                                        parameter_nodes_per_axis=n,left=a['label'],right=b['label'],selected=True,**result))
        for n,m in zip(sizes,sizes[1:]):
            selected = m==sizes[-1]
            a,b = [lookup[f'{v}/middle/quadrature'] for v in (n,m)]
            comparisons.append(dict(coordinates=list(pair),sequence_id=sid,kind='parameter_grid',
                                    left=a['label'],right=b['label'],selected=selected,**compare_parameter_banks(a,b)))
        if prior_bridge:
            a,b = [lookup[f'{sizes[1]}/middle/{kind}'] for kind in ('discrete_uniform','quadrature')]
            comparisons.append(dict(coordinates=list(pair),sequence_id=sid,kind='prior_bridge',
                                    left=a['label'],right=b['label'],selected=False,
                                    **compare_parameter_banks(a,b,allow_prior_change=True)))
    return comparisons


def validated_g1b0(report, *, require_source=False):
    """Recompute the complete rejected G1b0 pilot and its passing FV controls."""
    return _validated_grid_report(report,require_source=require_source,stage='G1b0',
                                  version=1,counts=G1B0_COUNTS,prior_bridge=True)


def validated_g1b3(report, *, require_source=False):
    """Validate every dense bank, original gate and recorded reproduction.

    The G1c CLI also reads the original G1b0 file and recomputes reproduction;
    a digest or a previously recorded passing flag alone is insufficient.
    """
    return _validated_grid_report(report,require_source=require_source,stage='G1b3',
                                  version=4,counts=G1B3_COUNTS,prior_bridge=False)


def _validate_recorded_reproduction(value):
    fields = {'conditional_log_evidence','log_weights','bank_log_evidence','cumulative_log_evidence',
              'marginal_predicted_mass','marginal_posterior_mass','conditional_actual_max_steps'}
    if (value.get('tolerance')!=1e-10 or value.get('bank_comparisons')!=16
            or value.get('passes') is not True or set(value.get('maximum_errors',{}))!=fields
            or 'validation_error' in value):
        raise ValueError('Passing complete shared-replay reproduction is required.')
    errors = fv._array(list(value['maximum_errors'].values()),(7,),'reproduction errors')
    if np.any((errors<0)|(errors>1e-10)):
        raise ValueError('Shared-replay reproduction errors failed.')


def _validated_grid_report(report, *, require_source, stage, version, counts, prior_bridge):
    configurations = _grid_configurations(fv.G1A_NUMERICS)
    header = dict(stage=stage,design_version=version,registered_design=True,
                  sequence_ids=list(G1B0_IDS),parameter_grid_counts=list(counts),
                  candidate_parameter_nodes_per_axis=counts[-2],conditional_runs=8*counts[-1]**2,bank_reductions=8*len(counts)+(2 if prior_bridge else 0),
                  parameter_criteria=dict(G1B0_CRITERIA),
                  fv_criteria=dict(log_odds=.10,weight_tv=.01,evidence=.10,state_l1=.02,
                                   mass_error=1e-10,actual_step_relative_margin=1e-12),
                  prior_measure=dict(rho__Xi='product uniform in rho and Xi',
                                     sigma_x__sigma_y='product uniform in log(sigma_x) and log(sigma_y)',
                                     quadrature='tensor trapezoid in normalized prior coordinates',
                                     comparison='piecewise constant dual-cell laws; comparison only'),
                  candidate_fv_numerics={k:v[0][1] for k,v in configurations.items()},
                  fv_controls={k:{label:s for label,s in values} for k,values in configurations.items()},
                  observation_contract=dict(time_unit='day',observation_kind='point',price_side='right'),
                  inputs=dict(baseline=.5,price_edges=[0.,1/3,2/3,1.],price_values=[.65,.35,.65],
                              state_response='1-2*x',price_response='1-2*u'),
                  truth_settings=dict(initial_beta=[20.,20.],truth_max_step=1/1152,seed_roots=[4101,4102,4103]))
    if (any(report.get(k)!=v for k,v in header.items())
            or report.get('registered_design') is not True or not isinstance(report.get('passes'),bool)
            or (prior_bridge and report['passes'])):
        raise ValueError('A complete rejected registered Joint grid report is required.')
    if require_source and (report.get('working_tree_dirty') is not False
                           or not re.fullmatch('[0-9a-f]{40}',str(report.get('source_commit')))):
        raise ValueError('Joint grid requires a clean recorded source commit.')
    dependency = report.get('dependencies',{})
    chain = []
    if not prior_bridge:
        parent = dependency.get('g1b0',{})
        chain.append(parent)
        dependency = parent
        _validate_recorded_reproduction(report.get('replay_reproduction',{}))
        if (report.get('conditional_backend')!='lapack-tridiagonal; unchanged FV scheme'
                or report.get('reduction')!='ordered streaming log-scaled sums; at most 2*workers in flight'):
            raise ValueError('Dense grid backend/reduction changed.')
    dependency = dependency.get('g1a',{})
    chain.extend((dependency,dependency.get('g0',{})))
    for source in chain:
        if (not re.fullmatch('[0-9a-f]{64}',str(source.get('sha256')))
                or not re.fullmatch('[0-9a-f]{40}',str(source.get('source_commit')))):
            raise ValueError('Joint grid requires its recorded G1a/G0 dependency chain.')
    rows = report.get('rows',[])
    if len(rows)!=header['bank_reductions']: raise ValueError('Grid report requires every bank.')
    expected = {}
    for pair,sid,Xi in zip(core.F4_G_PAIRS,G1B0_IDS,(.75,1.)):
        case = core.build_f4_b_case(Xi=Xi)
        seq = core.generate_synthetic_sequence(case,sequence_id=sid)
        for n in counts:
            for label,settings in configurations['__'.join(pair)]:
                expected[(pair,sid,f'{n}/{label}/quadrature')] = (case,seq,n,label,'quadrature',settings)
        if prior_bridge: expected[(pair,sid,'5/middle/discrete_uniform')] = (case,seq,5,'middle','discrete_uniform',configurations['__'.join(pair)][0][1])
    seen = {}
    for row in rows:
        key = (tuple(row['coordinates']),row['sequence_id'],row['label'])
        if key not in expected or key in seen: raise ValueError('Joint grid banks are missing, duplicate or unregistered.')
        seen[key] = row
        case,seq,n,label,kind,settings = expected[key]
        truth = core.f4_e_truth_candidate(case)
        if (row['parameter_nodes_per_axis']!=n or row['prior_kind']!=kind or row['fv_label']!=label
                or row['settings']!=settings or row['observed']!=[True]*24
                or row['truth']!=[getattr(truth,name) for name in key[0]]
                or row['fixed_parameters']!={name:getattr(truth,name) for name in ('rho','Xi','k','sigma_x','sigma_y') if name not in key[0]}):
            raise ValueError('Joint grid experiment, grid or FV numerics changed.')
        h,unit,dual,prior = _parameter_grid(key[0],n,counts[-1])
        if kind=='discrete_uniform': prior = np.full(n*n,-np.log(n*n))
        for name,value,tolerance in [('hypotheses',h,1e-12),('parameter_unit_nodes',unit,0.),
                                     ('parameter_dual_edges',dual,0.),('initial_log_weights',prior,1e-12),
                                     ('observation_times',seq.observations.times,1e-12),
                                     ('observations',seq.observations.values,1e-12),
                                     ('x_edges',np.linspace(0,1,settings['x_steps']+1),0.),
                                     ('initial_mass',core.beta_initial_mass(case,case.build_model()[0],settings['x_steps']),1e-10)]:
            fv._equal_array(row[name],value,'Joint grid '+name,tolerance)
        scores = fv._array(row['conditional_log_evidence'],(24,n*n),'Joint grid scores')
        raw = prior+np.cumsum(scores,axis=0);evidence = logsumexp(raw,axis=1);logs = raw-evidence[:,None]
        fv._equal_array(row['log_weights'],logs,'Joint grid weights')
        fv._equal_array(row['cumulative_log_evidence'],evidence,'Joint grid evidence')
        bank = fv._array(row['bank_log_evidence'],(24,),'Joint grid increments')
        fv._equal_array(np.cumsum(bank),evidence,'Joint grid bank evidence')
        summary = core._f4_g_joint_summary(h,logs,key[0])
        for name,value in summary.items():
            if name=='map_hypotheses':
                if row[name]!=value: raise ValueError('Joint grid MAP ties disagree.')
            elif name=='marginal_parameters':
                if set(row[name])!=set(value): raise ValueError('Joint grid marginal coordinates disagree.')
                for coordinate,marginal in value.items():
                    if set(row[name][coordinate])!=set(marginal): raise ValueError('Joint grid marginal fields disagree.')
                    for field,want in marginal.items():
                        fv._equal_array(row[name][coordinate][field],want,'Joint grid marginal '+field)
            else: fv._equal_array(row[name],value,'Joint grid '+name)
        for phase in ('predicted','posterior'):
            fv._mass(row['marginal_'+phase+'_mass'],24,settings['x_steps'],'Joint grid '+phase)
        steps = fv._array(row['conditional_actual_max_steps'],(n*n,),'Joint grid actual steps')
        if np.any((steps<=0)|(steps>settings['max_step']+1e-14)): raise ValueError('Joint grid actual steps are invalid.')
    fv._probability_gate(rows,raise_error=True)
    for key,row in seen.items():
        n = row['parameter_nodes_per_axis'];fine = seen[(key[0],key[1],f"{counts[-1]}/{row['fv_label']}/quadrature")]
        indices = np.arange(counts[-1]**2).reshape(counts[-1],counts[-1])[::(counts[-1]-1)//(n-1),::(counts[-1]-1)//(n-1)].ravel()
        fv._equal_array(row['conditional_log_evidence'],np.array(fine['conditional_log_evidence'])[:,indices],'Joint grid nested scores',0.)
        fv._equal_array(row['conditional_actual_max_steps'],np.array(fine['conditional_actual_max_steps'])[indices],'Joint grid nested steps',0.)
    calculated = _grid_comparisons(rows,G1B0_IDS,counts,prior_bridge=prior_bridge)
    comparisons = report.get('comparisons',[])
    def comparison_key(c):
        return tuple(c['coordinates']),c['sequence_id'],c['kind'],c['left'],c['right']
    actual = {comparison_key(c):c for c in comparisons}
    if len(comparisons)!=len(calculated) or len(actual)!=len(calculated):
        raise ValueError('Grid report requires every unique comparison.')
    for want in calculated:
        key = comparison_key(want)
        if key not in actual: raise ValueError('Joint grid comparison is missing or unregistered.')
        got = actual[key]
        if set(got)!=set(want): raise ValueError('Joint grid comparison fields changed.')
        for name,value in want.items():
            if name=='prefix_errors':
                if set(got[name])!=set(value): raise ValueError('Joint grid prefix criteria changed.')
                for field,v in value.items(): fv._equal_array(got[name][field],v,'Joint grid prefix '+field)
            elif isinstance(value,bool):
                if got[name] is not value: raise ValueError('Joint grid selection or gate disagrees.')
            elif name.startswith('maximum_'): fv._equal_array(got[name],value,'Joint grid '+name)
            elif got[name]!=value: raise ValueError('Joint grid comparison design changed.')
        if want['selected'] and want['kind'].startswith('FV_') and not want['passes']:
            raise ValueError('Joint grid FV controls must pass before parameter-only refinement.')
        if prior_bridge and want['selected'] and want['kind']=='parameter_grid' and want['passes']:
            raise ValueError('Refinement requires both original parameter grids rejected.')
    expected_pass = all(c['passes'] for c in calculated if c['selected'])
    if report['passes'] is not bool(expected_pass):
        raise ValueError('Grid report overall gate disagrees with raw comparisons.')
    return True


def compare_grid_overlap(previous, current, *, counts=(5,9)):
    """Reproduction gate for all 16 shared quadrature banks, without new fits."""
    fields = ('conditional_log_evidence','log_weights','bank_log_evidence','cumulative_log_evidence',
              'marginal_predicted_mass','marginal_posterior_mass','conditional_actual_max_steps')
    equal_fields = ('coordinates','sequence_id','settings','hypotheses','initial_log_weights',
                    'observation_times','observations','observed','truth','fixed_parameters','x_edges',
                    'parameter_unit_nodes','parameter_dual_edges','prior_kind')
    def shared_rows(report):
        rows = [r for r in report['rows'] if r['prior_kind']=='quadrature' and r['parameter_nodes_per_axis'] in counts]
        keyed = {(tuple(r['coordinates']),r['sequence_id'],r['label']):r for r in rows}
        if len(rows)!=16 or len(keyed)!=16: raise ValueError('Grid replay check requires all 16 shared banks.')
        return keyed
    a,b = shared_rows(previous),shared_rows(current)
    if set(a)!=set(b): raise ValueError('Grid replay check requires identical shared bank keys.')
    maxima = {field:0. for field in fields}
    for key,left in a.items():
        right = b[key]
        for field in equal_fields:
            if field in ('hypotheses','initial_log_weights','observation_times','observations',
                         'x_edges','parameter_unit_nodes','parameter_dual_edges'):
                fv._equal_array(right[field],left[field],'Shared '+field,1e-12)
            elif right[field]!=left[field]: raise ValueError('Grid replay check experiment/prior changed.')
        for field in fields:
            l = np.asarray(left[field]);r = fv._array(right[field],l.shape,'Shared '+field)
            if not np.all(np.isfinite(l)): raise ValueError('Previous grid values must be finite.')
            maxima[field] = max(maxima[field],float(np.max(np.abs(r-l))))
    return dict(tolerance=1e-10,bank_comparisons=16,maximum_errors=maxima,
                passes=bool(all(value<=1e-10 for value in maxima.values())))


# G1c0 is a different continuous law, never an override of G1b dual-cell TV.
G1C0_TV_ORDERS = (8, 16, 32)
G1C0_QUADRATURE_TOLERANCE = 1e-8


def _hat_law(row):
    """Positive tensor-hat density in normalized prior coordinates.

    q_i = integral H_i. Nodal density exp(log(w_ij)-log(q_i*q_j))
    integrates to one. Its joint state extension uses each node's own
    conditional state; integration gives the existing Bayes mixture exactly.
    """
    n = checks.positive_integer(row['parameter_nodes_per_axis'],'hat nodes',minimum=2)
    unit = np.linspace(0,1,n)
    fv._equal_array(row['parameter_unit_nodes'],unit,'hat unit nodes',0.)
    if row['prior_kind']!='quadrature': raise ValueError('Hat law requires the continuous quadrature prior.')
    q = np.full(n,1/(n-1));q[[0,-1]] *= .5
    prior = np.log(np.outer(q,q).ravel())
    fv._equal_array(row['initial_log_weights'],prior,'hat prior',1e-12)
    count = len(row['observation_times'])
    logs = fv._array(row['log_weights'],(count,n*n),'hat weights')
    weights = np.exp(logs).reshape(count,n,n)
    density = np.exp(logs-prior).reshape(count,n,n)
    mass = np.einsum('kij,i,j->k',density,q,q,optimize=True)
    if not np.all(np.isfinite(density)) or np.any(density<0) or np.max(abs(mass-1))>1e-10:
        raise ValueError('Hat law must be finite, nonnegative and normalized.')
    # Exact first and second moments of each normalized one-dimensional hat.
    step = 1/(n-1)
    center = unit.copy();center[0] = step/3;center[-1] = 1-step/3
    second = unit**2+step**2/6
    second[0] = step**2/6;second[-1] = 1-2*step/3+step**2/6
    mean = np.column_stack([weights.sum(axis=2)@center,weights.sum(axis=1)@center])
    e2 = np.column_stack([weights.sum(axis=2)@second,weights.sum(axis=1)@second])
    variance = e2-mean**2
    if np.min(variance)<-1e-12: raise ValueError('Hat variance is negative.')
    cross = np.einsum('kij,i,j->k',weights,center,center,optimize=True)
    return dict(unit=unit,density=density,mass=mass,mean=mean,
                sd=np.sqrt(np.maximum(variance,0.)),covariance=cross-mean[:,0]*mean[:,1])


def _hat_interpolate(density, source, target):
    """Evaluate a tensor-linear field on the union of primal node grids."""
    j = np.clip(np.searchsorted(source,target,side='right')-1,0,len(source)-2)
    t = (target-source[j])/(source[j+1]-source[j])
    first = density[j,:]*(1-t[:,None])+density[j+1,:]*t[:,None]
    return first[:,j]*(1-t[None,:])+first[:,j+1]*t[None,:]


def _bilinear_abs_integral(corners, order):
    """Integral of |bilinear| on each unit cell; exact inner integration.

    Split the outer axis at both boundary zero crossings. On each resulting
    interval the exact inner integral is smooth (linear or rational). Gauss
    refinement is checked independently by the caller; it is not a proof of
    a rigorous error bound.
    """
    a0,a1,b0,b1 = corners  # y=0 and y=1 endpoint lines in x
    cuts = [np.zeros_like(a0),np.ones_like(a0)]
    for start,end in ((a0,a1),(b0,b1)):
        root = np.divide(-start,end-start,out=np.ones_like(start),where=end!=start)
        cuts.append(np.where((root>0)&(root<1),root,1.))
    cuts = np.sort(np.stack(cuts),axis=0)
    nodes,weights = np.polynomial.legendre.leggauss(order)
    nodes = (nodes+1)/2;weights = weights/2
    total = np.zeros_like(a0)
    for left,right in zip(cuts[:-1],cuts[1:]):
        x = left[...,None]+(right-left)[...,None]*nodes
        a = a0[...,None]+(a1-a0)[...,None]*x
        b = b0[...,None]+(b1-b0)[...,None]*x
        denominator = 2*abs(a-b)
        crossing = np.divide(a*a+b*b,denominator,out=np.zeros_like(a),where=denominator>0)
        inner = np.where(a*b>=0,(abs(a)+abs(b))/2,crossing)
        total += (right-left)*(inner@weights)
    return total


def compare_hat_laws(left, right):
    """G1c0 comparison, keeping the original dual-cell comparison intact."""
    if (left['coordinates']!=right['coordinates'] or left['sequence_id']!=right['sequence_id']
            or left['observation_times']!=right['observation_times'] or left['observations']!=right['observations']
            or left['x_edges']!=right['x_edges']):
        raise ValueError('Hat comparison requires the same experiment and state grid.')
    laws = [_hat_law(row) for row in (left,right)]
    union = np.unique(np.r_[laws[0]['unit'],laws[1]['unit']])
    areas = np.outer(np.diff(union),np.diff(union))
    tv = []
    for d0,d1 in zip(laws[0]['density'],laws[1]['density']):
        delta = (_hat_interpolate(d1,laws[1]['unit'],union)
                 -_hat_interpolate(d0,laws[0]['unit'],union))
        corners = (delta[:-1,:-1],delta[1:,:-1],delta[:-1,1:],delta[1:,1:])
        tv.append([.5*float(np.sum(areas*_bilinear_abs_integral(corners,order)))
                   for order in G1C0_TV_ORDERS])
    tv = np.array(tv)
    refinement = np.max(abs(np.diff(tv,axis=1)),axis=1)
    prefix = dict(parameter_tv=tv[:,-1],tv_quadrature_refinement=refinement,
                  unit_mean_error=np.max(abs(laws[1]['mean']-laws[0]['mean']),axis=1),
                  unit_sd_error=np.max(abs(laws[1]['sd']-laws[0]['sd']),axis=1),
                  cumulative_evidence_error=abs(np.array(right['cumulative_log_evidence'])-left['cumulative_log_evidence']))
    for phase in ('predicted','posterior'):
        mass = [fv._mass(r['marginal_'+phase+'_mass'],len(tv),len(r['x_edges'])-1,'hat '+phase) for r in (left,right)]
        prefix[phase+'_l1'] = abs(mass[1]-mass[0]).sum(axis=1)
    maxima = {'maximum_'+k:float(v.max()) for k,v in prefix.items()}
    tol = G1C0_QUADRATURE_TOLERANCE
    passes = (maxima['maximum_tv_quadrature_refinement']<=tol
              and maxima['maximum_parameter_tv']+tol<=G1B0_CRITERIA['maximum_parameter_tv']
              and all(maxima[k]<=v for k,v in G1B0_CRITERIA.items()
                      if k not in ('maximum_parameter_tv','maximum_state_l1'))
              and max(maxima['maximum_predicted_l1'],maxima['maximum_posterior_l1'])<=G1B0_CRITERIA['maximum_state_l1'])
    return dict(**maxima,prefix_errors={k:v.tolist() for k,v in prefix.items()},
                tv_by_quadrature_order=tv.tolist(),passes=bool(passes))


def run_joint_parameter_reconstruction(report, *, require_source=True):
    """G1c0 same-record reconstruction pilot; no new conditional replay.

    Raw input is fully revalidated. This is data-informed development, not a
    held-out study and not acceptance of the original G1b dual-cell law.
    """
    validated_g1b3(report,require_source=require_source)
    rows = [r for r in report['rows'] if r['fv_label']=='middle']
    summaries = []
    for row in rows:
        law = _hat_law(row)
        summaries.append(dict(coordinates=row['coordinates'],sequence_id=row['sequence_id'],label=row['label'],
            parameter_nodes_per_axis=row['parameter_nodes_per_axis'],parameter_unit_nodes=law['unit'].tolist(),
            nodal_density=law['density'].tolist(),total_mass=law['mass'].tolist(),unit_mean=law['mean'].tolist(),
            unit_sd=law['sd'].tolist(),unit_covariance=law['covariance'].tolist()))
    comparisons = []
    for pair,sid in zip(core.F4_G_PAIRS,G1B0_IDS):
        lookup = {r['parameter_nodes_per_axis']:r for r in rows if tuple(r['coordinates'])==pair and r['sequence_id']==sid}
        for a,b in zip(G1B3_COUNTS[:-1],G1B3_COUNTS[1:]):
            comparisons.append(dict(coordinates=list(pair),sequence_id=sid,left=lookup[a]['label'],right=lookup[b]['label'],
                selected=(a==G1B3_COUNTS[-2] and b==G1B3_COUNTS[-1]),**compare_hat_laws(lookup[a],lookup[b])))
    selected = [c for c in comparisons if c['selected']]
    return dict(stage='G1c0',design_version=1,registered_design=True,sequence_ids=list(G1B0_IDS),
        candidate_parameter_nodes_per_axis=G1B3_COUNTS[-2],control_parameter_nodes_per_axis=G1B3_COUNTS[-1],
        conditional_runs=0,source_conditional_runs=report['conditional_runs'],parameter_grid_counts=list(G1B3_COUNTS),
        representation='positive tensor-linear hats in unit prior coordinates; joint extension retains node-conditional states',
        prior_measure=dict(report['prior_measure'],comparison='continuous tensor-hat density in unit prior coordinates'),
        parameter_criteria=dict(G1B0_CRITERIA),
        tv_quadrature=dict(orders=list(G1C0_TV_ORDERS),maximum_successive_difference=G1C0_QUADRATURE_TOLERANCE,
                           acceptance_margin=G1C0_QUADRATURE_TOLERANCE,rigorous_error_bound=False),
        original_dual_cell_passes=report['passes'],original_comparisons=report['comparisons'],
        certifies_original_dual_cell_law=False,rows=summaries,comparisons=comparisons,
        passes=bool(len(selected)==2 and all(c['passes'] for c in selected)),
        interpretation='Data-informed same-record reconstruction pilot only; original G1b failures retained; no G2 or held-out recovery certification.')


def _check_recorded(got, want, name, *, tolerance=1e-10):
    """Check nested scientific evidence; execution metadata is handled separately."""
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got)!=set(want):
            raise ValueError(f'{name} fields disagree.')
        for key,value in want.items():
            _check_recorded(got[key],value,name+'/'+key,tolerance=tolerance)
    elif isinstance(want, (bool, str)) or want is None:
        if type(got) is not type(want) or got!=want:
            raise ValueError(f'{name} disagrees.')
    elif isinstance(want, list):
        try:
            numeric = np.issubdtype(np.asarray(want).dtype,np.number)
        except ValueError:
            numeric = False
        if numeric:
            fv._equal_array(got,want,name,tolerance)
        else:
            if not isinstance(got,list) or len(got)!=len(want):
                raise ValueError(f'{name} rows disagree.')
            for i,value in enumerate(want):
                _check_recorded(got[i],value,f'{name}/{i}',tolerance=tolerance)
    else:
        fv._equal_array(got,want,name,tolerance)


def validated_g1c0(report, dense, *, require_source=False):
    """Reconstruct the complete G1c0 result from its raw G1b3 dependency."""
    if require_source and (report.get('working_tree_dirty') is not False
            or not re.fullmatch('[0-9a-f]{40}',str(report.get('source_commit')))):
        raise ValueError('G1c0 requires a clean recorded source commit.')
    expected = run_joint_parameter_reconstruction(dense,require_source=require_source)
    for name,value in expected.items():
        _check_recorded(report.get(name),value,'G1c0 '+name)
    if report.get('passes') is not True:
        raise ValueError('G2a requires a passing G1c0 pilot.')
    return True


G2A_IDS = (1358, 1359)
G2A_COUNTS = (33, 65)
G2A_DAYS = (1, 3, 7)
G2A_TRUTH_CRITERIA = dict(maximum_rms_demand_over_sigma_y=.10,
                          maximum_abs_demand_over_sigma_y=.50,
                          no_raw_boundary_violations=True)


def g2a_case(pair, record_days):
    """Same BM1 case with a repeated daily input and one continuous horizon."""
    pair = tuple(pair)
    if pair not in core.F4_G_PAIRS:
        raise ValueError('G2a requires one of the two registered parameter pairs.')
    days = checks.positive_integer(record_days,'record days')
    if days not in G2A_DAYS:
        raise ValueError('G2a record days must be 1, 3 or 7.')
    return core.build_f4_b_case(horizon=(0.,float(days)),
        price_edges=np.arange(3*days+1,dtype=float)/3,
        price_values=np.tile(core.CORE_PRICE_VALUES,days),
        observation_times=np.arange(1,24*days+1,dtype=float)/24,
        Xi=.75 if pair[0]=='rho' else 1.)


def _g2a_truth_record(pair, sid, days):
    case = g2a_case(pair,days)
    coupled = core.generate_coupled_refinement(case,sequence_id=sid)
    coarse,fine = coupled.coarse,coupled.fine
    delta = fine.truth.demand[1:]-coarse.truth.demand[1:]
    rms = float(np.sqrt(np.mean(delta**2))/case.sigma_y)
    maximum = float(np.max(abs(delta))/case.sigma_y)
    valid = bool(coarse.truth.valid and fine.truth.valid)
    def member(sequence):
        d = sequence.truth.diagnostics
        return dict(state=sequence.truth.state.tolist(),demand=sequence.truth.demand.tolist(),
            observations=sequence.observations.values.tolist(),valid=bool(sequence.truth.valid),
            integration_steps=int(d['n_steps']),minimum_raw_proposal=float(d['minimum_raw_proposal']),
            maximum_raw_proposal=float(d['maximum_raw_proposal']),
            raw_boundary_violations=int(d['total_violations']),boundary_corrections=int(d['boundary_corrections']))
    record = dict(coordinates=list(pair),sequence_id=sid,record_days=days,
        observation_times=fine.observations.times.tolist(),initial_state=fine.truth.initial_state,
        measurement_noise=fine.measurement_noise.tolist(),price_edges=case.price_edges.tolist(),
        price_values=case.price_values.tolist(),baseline=case.baseline,initial_beta=[20.,20.],
        coarse_max_step=case.truth_max_step,fine_max_step=case.truth_fine_max_step,
        seed_roots=[4101,4102,4103],coarse=member(coarse),fine=member(fine),
        rms_demand_over_sigma_y=rms,maximum_abs_demand_over_sigma_y=maximum,
        passes=bool(valid and rms<=G2A_TRUTH_CRITERIA['maximum_rms_demand_over_sigma_y']
                    and maximum<=G2A_TRUTH_CRITERIA['maximum_abs_demand_over_sigma_y']))
    return case,fine,record


def _g2a_header(days, ids, counts, levels, *, registered):
    return dict(stage='G2a',design_version=1,registered_design=registered,
        sequence_ids=list(ids),record_days=days,checkpoint_days=[d for d in G2A_DAYS if d<=days],
        parameter_grid_counts=list(counts),candidate_parameter_nodes_per_axis=counts[0],
        control_parameter_nodes_per_axis=counts[1],numerical_levels=deepcopy(levels),
        truth_criteria=dict(G2A_TRUTH_CRITERIA),
        observation_contract=dict(time_unit='day',observation_kind='point',price_side='right',sampling_hours=1),
        representation='positive tensor-linear hats in unit prior coordinates; conditional states continue without daily resets',
        prior_measure=dict(rho__Xi='product uniform in rho and Xi',
            sigma_x__sigma_y='product uniform in log(sigma_x) and log(sigma_y)',
            quadrature='tensor trapezoid in normalized prior coordinates'),
        fv_criteria=dict(log_likelihood_window=5.,maximum_log_odds_error=.10,maximum_weight_tv=.01,
            maximum_cumulative_evidence_error=.10,maximum_state_l1=.02,maximum_abs_mass_error=1e-10,
            nonnegative_mass=True,actual_time_refinement=True,actual_step_relative_margin=1e-12),
        parameter_criteria=dict(G1B0_CRITERIA),
        tv_quadrature=dict(orders=list(G1C0_TV_ORDERS),maximum_successive_difference=G1C0_QUADRATURE_TOLERANCE,
            acceptance_margin=G1C0_QUADRATURE_TOLERANCE,rigorous_error_bound=False),
        conditional_backend='lapack-tridiagonal; unchanged FV scheme',
        reduction='ordered streaming log-scaled sums; at most 2*workers in flight',
        certifies_original_dual_cell_law=False,
        interpretation='New-record numerical pilot only at the recorded horizon; no population, sampling/mismatch robustness, recovery or forecast certification.')


def _g2a_comparisons(rows, ids, counts):
    original = _grid_comparisons(rows,ids,counts,prior_bridge=False,include_first_fv=True)
    fv_controls = [c for c in original if c['kind'].startswith('FV_')]
    dual = [dict(c,selected=False) for c in original if c['kind']=='parameter_grid']
    hat = []
    for pair,sid in zip(core.F4_G_PAIRS,ids):
        lookup = {r['parameter_nodes_per_axis']:r for r in rows
                  if tuple(r['coordinates'])==pair and r['sequence_id']==sid and r['fv_label']=='middle'}
        a,b = [lookup[n] for n in counts]
        hat.append(dict(coordinates=list(pair),sequence_id=sid,kind='parameter_hat',
                        left=a['label'],right=b['label'],selected=True,**compare_hat_laws(a,b)))
    return fv_controls+hat,dual


def _g2a_checkpoints(rows, records, days, candidate_count):
    summaries = []
    for r in rows:
        if r['fv_label']!='middle' or r['parameter_nodes_per_axis']!=candidate_count:
            continue
        law = _hat_law(r)
        truth = next(s for s in records if s['sequence_id']==r['sequence_id'])
        state = np.asarray(truth['fine']['state'])[1:]
        edges = np.asarray(r['x_edges']);centers=(edges[:-1]+edges[1:])/2
        means = {phase:np.asarray(r['marginal_'+phase+'_mass'])@centers for phase in ('predicted','posterior')}
        for d in G2A_DAYS:
            if d>days: continue
            k = 24*d;index=k-1
            summaries.append(dict(coordinates=r['coordinates'],sequence_id=r['sequence_id'],days=d,
                observation_count=k,cumulative_log_evidence=r['cumulative_log_evidence'][index],
                mean_predictive_y_log_score=r['cumulative_log_evidence'][index]/k,
                hat_unit_mean=law['mean'][index].tolist(),hat_unit_sd=law['sd'][index].tolist(),
                hat_unit_covariance=float(law['covariance'][index]),
                **{phase+'_state_rmse':float(np.sqrt(np.mean((values[:k]-state[:k])**2)))
                   for phase,values in means.items()}))
    return summaries


def run_joint_continued_pilot(*, record_days=7, workers=8, progress=None,
                             sequence_ids=G2A_IDS, counts=G2A_COUNTS, numerics=None):
    """G2a: nested 33/65 grids, every FV control, and no daily state resets.

    API overrides support small diagnostics/tests, never registered evidence.
    Seven-day rows supply day 1/3/7 prefixes without shorter filter reruns.
    """
    workers = checks.positive_integer(workers,'workers')
    days = checks.positive_integer(record_days,'record days')
    if days not in G2A_DAYS: raise ValueError('G2a record days must be 1, 3 or 7.')
    ids = tuple(core._nonnegative_integer(v,'sequence ID') for v in sequence_ids)
    sizes = tuple(checks.positive_integer(v,'parameter nodes',minimum=2) for v in counts)
    if len(ids)!=2 or len(set(ids))!=2 or len(sizes)!=2 or sizes[0]>=sizes[1] or (sizes[1]-1)%(sizes[0]-1):
        raise ValueError('G2a requires two distinct IDs and two nested increasing grids.')
    levels = fv.G1A_NUMERICS if numerics is None else numerics
    configs = _grid_configurations(levels)
    # Normalize and own metadata; caller mutations cannot alter the design.
    levels = {key:{name:list(values) for name,values in lv.items()} for key,lv in levels.items()}
    header = _g2a_header(days,ids,sizes,levels,
        registered=bool(ids==G2A_IDS and sizes==G2A_COUNTS and numerics is None))
    sequences = [_g2a_truth_record(pair,sid,days) for pair,sid in zip(core.F4_G_PAIRS,ids)]
    records = [v[2] for v in sequences]
    truth_pass = all(r['passes'] for r in records)
    result = dict(header,workers=workers,truth_records=records,truth_passes=bool(truth_pass),
        completed=False,conditional_runs=0,bank_reductions=0,rows=[],comparisons=[],
        original_dual_cell_comparisons=[],checkpoints=[],passes=False)
    if not truth_pass:
        return result
    groups = []
    for pair,sid,(case,sequence,_) in zip(core.F4_G_PAIRS,ids,sequences):
        h = _parameter_grid(pair,sizes[-1],sizes[-1])[0]
        for label,settings in configs['__'.join(pair)]:
            groups.append(dict(stage='G2a',pair=pair,sequence_id=sid,case=case,
                observations=sequence.observations,hypotheses=h,counts=sizes,label=label,
                settings=settings,prior_bridge=False,backend='lapack'))
    banks = fv._run_groups_streaming(groups,workers,progress,_StreamingGridReduction)
    rows = [r for bank in banks for r in bank]
    comparisons,dual = _g2a_comparisons(rows,ids,sizes)
    result.update(completed=True,conditional_runs=sum(len(g['hypotheses']) for g in groups),
        bank_reductions=len(rows),rows=rows,comparisons=comparisons,
        original_dual_cell_comparisons=dual,
        checkpoints=_g2a_checkpoints(rows,records,days,sizes[0]),
        passes=bool(fv._probability_gate(rows) and all(c['passes'] for c in comparisons)))
    return result


def validated_g2a(report, *, require_source=False):
    """Raw algebra, seed data and every comparison; no new conditional replay.

    Complete negative reports remain valid review evidence. Callers advancing
    to a dependent population must additionally require passes=True and 7 days.
    """
    if require_source and (report.get('working_tree_dirty') is not False
            or not re.fullmatch('[0-9a-f]{40}',str(report.get('source_commit')))):
        raise ValueError('G2a requires a clean recorded source commit.')
    if require_source:
        dependency = report.get('dependencies',{})
        for stage in ('g1c0','g1b3','g1b0','g1a','g0'):
            dependency = dependency.get(stage,{})
            if (not re.fullmatch('[0-9a-f]{64}',str(dependency.get('sha256')))
                    or not re.fullmatch('[0-9a-f]{40}',str(dependency.get('source_commit')))):
                raise ValueError('G2a requires its complete recorded G1c0 dependency chain.')
    if 'raw_validation' in report and report['raw_validation']!={'passes':True}:
        raise ValueError('G2a final raw validation failed; report remains diagnostic evidence.')
    days = checks.positive_integer(report.get('record_days'),'record days')
    if days not in G2A_DAYS: raise ValueError('G2a requires 1, 3 or 7 days.')
    levels = {key:{name:list(v) for name,v in lv.items()} for key,lv in fv.G1A_NUMERICS.items()}
    header = _g2a_header(days,G2A_IDS,G2A_COUNTS,levels,registered=True)
    for name,value in header.items(): _check_recorded(report.get(name),value,'G2a '+name)
    sequences = [_g2a_truth_record(pair,sid,days) for pair,sid in zip(core.F4_G_PAIRS,G2A_IDS)]
    records = [v[2] for v in sequences]
    _check_recorded(report.get('truth_records'),records,'G2a truth')
    truth_pass = bool(all(r['passes'] for r in records))
    if report.get('truth_passes') is not truth_pass: raise ValueError('G2a truth gate disagrees.')
    if not truth_pass:
        for name,want in dict(completed=False,conditional_runs=0,bank_reductions=0,
            rows=[],comparisons=[],original_dual_cell_comparisons=[],checkpoints=[],passes=False).items():
            _check_recorded(report.get(name),want,'G2a stopped '+name)
        return True
    configs = _grid_configurations(levels)
    expected = {}
    for pair,sid,(case,seq,_) in zip(core.F4_G_PAIRS,G2A_IDS,sequences):
        for label,settings in configs['__'.join(pair)]:
            for n in G2A_COUNTS: expected[(pair,sid,n,label)] = (case,seq,settings)
    rows = report.get('rows',[]);seen={}
    if len(rows)!=len(expected): raise ValueError('G2a requires all 16 complete banks.')
    for r in rows:
        key = (tuple(r['coordinates']),r['sequence_id'],r['parameter_nodes_per_axis'],r['fv_label'])
        if key not in expected or key in seen: raise ValueError('G2a banks are duplicate or unregistered.')
        seen[key]=r;case,seq,settings=expected[key];pair,sid,n,label=key
        h,unit,edges,prior = _parameter_grid(pair,n,G2A_COUNTS[-1])
        truth=core.f4_e_truth_candidate(case);k=len(seq.observations.times);cells=settings['x_steps']
        metadata = dict(coordinates=list(pair),sequence_id=sid,label=f'{n}/{label}/quadrature',
            settings=settings,truth=[getattr(truth,p) for p in pair],
            fixed_parameters={p:getattr(truth,p) for p in ('rho','Xi','k','sigma_x','sigma_y') if p not in pair},
            observed=[True]*k,prior_kind='quadrature',fv_label=label)
        for name,value in metadata.items(): _check_recorded(r.get(name),value,'G2a bank '+name)
        for name,value in dict(hypotheses=h,parameter_unit_nodes=unit,parameter_dual_edges=edges,
            initial_log_weights=prior,observation_times=seq.observations.times,observations=seq.observations.values,
            initial_mass=core.beta_initial_mass(case,case.build_model()[0],cells),
            x_edges=np.linspace(0,1,cells+1)).items(): fv._equal_array(r.get(name),value,'G2a bank '+name)
        scores=fv._array(r['conditional_log_evidence'],(k,n*n),'G2a scores')
        raw=prior+np.cumsum(scores,axis=0);evidence=logsumexp(raw,axis=1);logs=raw-evidence[:,None]
        fv._equal_array(r['log_weights'],logs,'G2a weights')
        fv._equal_array(r['cumulative_log_evidence'],evidence,'G2a evidence')
        fv._equal_array(r['bank_log_evidence'],np.diff(np.r_[0.,evidence]),'G2a evidence increments')
        summary=core._f4_g_joint_summary(h,logs,pair)
        for name,value in summary.items(): _check_recorded(r.get(name),value,'G2a summary '+name)
        for phase in ('predicted','posterior'):
            fv._mass(r['marginal_'+phase+'_mass'],k,cells,'G2a '+phase)
        steps=fv._array(r['conditional_actual_max_steps'],(n*n,),'G2a steps')
        if np.any((steps<=0)|(steps>settings['max_step']+1e-14)):
            raise ValueError('G2a actual steps are invalid.')
        if checks.finite_scalar(r['conditional_seconds'],'conditional seconds')<0:
            raise ValueError('G2a conditional seconds must be nonnegative.')
    comparisons,dual=_g2a_comparisons(rows,G2A_IDS,G2A_COUNTS)
    _check_recorded(report.get('comparisons'),comparisons,'G2a comparisons')
    _check_recorded(report.get('original_dual_cell_comparisons'),dual,'G2a original dual-cell diagnostics')
    _check_recorded(report.get('checkpoints'),_g2a_checkpoints(rows,records,days,G2A_COUNTS[0]),'G2a checkpoints')
    probability=fv._probability_gate(rows)
    for name,want in dict(completed=True,conditional_runs=8*G2A_COUNTS[-1]**2,
        bank_reductions=len(expected),passes=bool(probability and all(c['passes'] for c in comparisons))).items():
        _check_recorded(report.get(name),want,'G2a '+name)
    return True
