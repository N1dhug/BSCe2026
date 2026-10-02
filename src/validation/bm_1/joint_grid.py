"""G1b0 pilot and G1b1 paired grid refinement before a larger population.

Conditional trajectories are shared only across quadratures of the same
experiment. Each hypothesis retains its own state; every FV control is fresh.
"""

import re

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
        laws.append(np.einsum('ai,kij,bj->kab',transfer,weights.reshape(count,n,n),transfer))
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


def _grid_comparisons(rows, ids, sizes, *, prior_bridge):
    comparisons = []
    for pair,sid in zip(core.F4_G_PAIRS,ids):
        lookup = {r['label']:r for r in rows if tuple(r['coordinates'])==pair and r['sequence_id']==sid}
        # Certify both the candidate measure and the control containing new nodes.
        # These reductions already exist: no extra conditional replay is needed.
        for n in sizes[1:]:
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
    """Require the complete rejected G1b0 pilot, with passing FV controls.

    Recompute raw summaries and gates; a failure is preserved as a dependency,
    rather than changed into an acceptance or an automatically selected grid.
    Conditional state histories are not stored; marginal masses are checked.
    """
    configurations = _grid_configurations(fv.G1A_NUMERICS)
    header = dict(stage='G1b0',design_version=1,registered_design=True,
                  sequence_ids=list(G1B0_IDS),parameter_grid_counts=list(G1B0_COUNTS),
                  candidate_parameter_nodes_per_axis=5,conditional_runs=648,bank_reductions=26,
                  passes=False,parameter_criteria=dict(G1B0_CRITERIA),
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
            or report.get('registered_design') is not True or report.get('passes') is not False):
        raise ValueError('A complete rejected registered G1b0 report is required.')
    if require_source and (report.get('working_tree_dirty') is not False
                           or not re.fullmatch('[0-9a-f]{40}',str(report.get('source_commit')))):
        raise ValueError('G1b0 requires a clean recorded source commit.')
    dependency = report.get('dependencies',{}).get('g1a',{})
    for source in (dependency,dependency.get('g0',{})):
        if (not re.fullmatch('[0-9a-f]{64}',str(source.get('sha256')))
                or not re.fullmatch('[0-9a-f]{40}',str(source.get('source_commit')))):
            raise ValueError('G1b0 requires its recorded G1a/G0 dependency chain.')
    rows = report.get('rows',[])
    if len(rows)!=26: raise ValueError('G1b0 requires all 26 banks.')
    expected = {}
    for pair,sid,Xi in zip(core.F4_G_PAIRS,G1B0_IDS,(.75,1.)):
        case = core.build_f4_b_case(Xi=Xi)
        seq = core.generate_synthetic_sequence(case,sequence_id=sid)
        for n in G1B0_COUNTS:
            for label,settings in configurations['__'.join(pair)]:
                expected[(pair,sid,f'{n}/{label}/quadrature')] = (case,seq,n,label,'quadrature',settings)
        expected[(pair,sid,'5/middle/discrete_uniform')] = (case,seq,5,'middle','discrete_uniform',configurations['__'.join(pair)][0][1])
    seen = {}
    for row in rows:
        key = (tuple(row['coordinates']),row['sequence_id'],row['label'])
        if key not in expected or key in seen: raise ValueError('G1b0 banks are missing, duplicate or unregistered.')
        seen[key] = row
        case,seq,n,label,kind,settings = expected[key]
        truth = core.f4_e_truth_candidate(case)
        if (row['parameter_nodes_per_axis']!=n or row['prior_kind']!=kind or row['fv_label']!=label
                or row['settings']!=settings or row['observed']!=[True]*24
                or row['truth']!=[getattr(truth,name) for name in key[0]]
                or row['fixed_parameters']!={name:getattr(truth,name) for name in ('rho','Xi','k','sigma_x','sigma_y') if name not in key[0]}):
            raise ValueError('G1b0 experiment, grid or FV numerics changed.')
        h,unit,dual,prior = _parameter_grid(key[0],n,9)
        if kind=='discrete_uniform': prior = np.full(n*n,-np.log(n*n))
        for name,value,tolerance in [('hypotheses',h,1e-12),('parameter_unit_nodes',unit,0.),
                                     ('parameter_dual_edges',dual,0.),('initial_log_weights',prior,1e-12),
                                     ('observation_times',seq.observations.times,1e-12),
                                     ('observations',seq.observations.values,1e-12),
                                     ('x_edges',np.linspace(0,1,settings['x_steps']+1),0.),
                                     ('initial_mass',core.beta_initial_mass(case,case.build_model()[0],settings['x_steps']),1e-10)]:
            fv._equal_array(row[name],value,'G1b0 '+name,tolerance)
        scores = fv._array(row['conditional_log_evidence'],(24,n*n),'G1b0 scores')
        raw = prior+np.cumsum(scores,axis=0);evidence = logsumexp(raw,axis=1);logs = raw-evidence[:,None]
        fv._equal_array(row['log_weights'],logs,'G1b0 weights')
        fv._equal_array(row['cumulative_log_evidence'],evidence,'G1b0 evidence')
        bank = fv._array(row['bank_log_evidence'],(24,),'G1b0 increments')
        fv._equal_array(np.cumsum(bank),evidence,'G1b0 bank evidence')
        summary = core._f4_g_joint_summary(h,logs,key[0])
        for name,value in summary.items():
            if name=='map_hypotheses':
                if row[name]!=value: raise ValueError('G1b0 MAP ties disagree.')
            elif name=='marginal_parameters':
                if set(row[name])!=set(value): raise ValueError('G1b0 marginal coordinates disagree.')
                for coordinate,marginal in value.items():
                    if set(row[name][coordinate])!=set(marginal): raise ValueError('G1b0 marginal fields disagree.')
                    for field,want in marginal.items():
                        fv._equal_array(row[name][coordinate][field],want,'G1b0 marginal '+field)
            else: fv._equal_array(row[name],value,'G1b0 '+name)
        for phase in ('predicted','posterior'):
            fv._mass(row['marginal_'+phase+'_mass'],24,settings['x_steps'],'G1b0 '+phase)
        steps = fv._array(row['conditional_actual_max_steps'],(n*n,),'G1b0 actual steps')
        if np.any((steps<=0)|(steps>settings['max_step']+1e-14)): raise ValueError('G1b0 actual steps are invalid.')
    fv._probability_gate(rows,raise_error=True)
    for key,row in seen.items():
        n = row['parameter_nodes_per_axis'];fine = seen[(key[0],key[1],f"9/{row['fv_label']}/quadrature")]
        indices = np.arange(81).reshape(9,9)[::8//(n-1),::8//(n-1)].ravel()
        fv._equal_array(row['conditional_log_evidence'],np.array(fine['conditional_log_evidence'])[:,indices],'G1b0 nested scores',0.)
        fv._equal_array(row['conditional_actual_max_steps'],np.array(fine['conditional_actual_max_steps'])[indices],'G1b0 nested steps',0.)
    calculated = _grid_comparisons(rows,G1B0_IDS,G1B0_COUNTS,prior_bridge=True)
    comparisons = report.get('comparisons',[])
    def comparison_key(c):
        return tuple(c['coordinates']),c['sequence_id'],c['kind'],c['left'],c['right']
    actual = {comparison_key(c):c for c in comparisons}
    if len(comparisons)!=18 or len(actual)!=18: raise ValueError('G1b0 requires all 18 unique comparisons.')
    for want in calculated:
        key = comparison_key(want)
        if key not in actual: raise ValueError('G1b0 comparison is missing or unregistered.')
        got = actual[key]
        if set(got)!=set(want): raise ValueError('G1b0 comparison fields changed.')
        for name,value in want.items():
            if name=='prefix_errors':
                if set(got[name])!=set(value): raise ValueError('G1b0 prefix criteria changed.')
                for field,v in value.items(): fv._equal_array(got[name][field],v,'G1b0 prefix '+field)
            elif isinstance(value,bool):
                if got[name] is not value: raise ValueError('G1b0 selection or gate disagrees.')
            elif name.startswith('maximum_'): fv._equal_array(got[name],value,'G1b0 '+name)
            elif got[name]!=value: raise ValueError('G1b0 comparison design changed.')
        if want['selected'] and want['kind'].startswith('FV_') and not want['passes']:
            raise ValueError('G1b0 FV controls must pass before parameter-only refinement.')
        if want['selected'] and want['kind']=='parameter_grid' and want['passes']:
            raise ValueError('G1b1 follows the rejected parameter grids for both pairs.')
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
