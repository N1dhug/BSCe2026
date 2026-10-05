"""Predeclared, bounded interval curve/noise controls, without fitting."""

from dataclasses import replace
from time import perf_counter

import numpy as np

from src.model.bm_1 import bm_1
from src.params.bm_1 import conservative_mass_transfer
from src.params.bm_1_response import BM1PiecewiseResponse
from src.solver.finite_volume import finite_volume_1d_fitted
from src.validation.bm_1.interval_characteristic import (
    CharacteristicNumericalError, balanced_characteristic_prediction,
    characteristic_gaussian_update, record_from_development_report, replay_characteristic)


CASES = {
    'linear-period': dict(f=[0.],g=[0.],pieces=2,sigma_y=.08,span=2.,frequencies=32),
    'two-narrow': dict(f=[.3],g=[-.25],pieces=2,sigma_y=.04,span=2.,frequencies=64),
    'four-wide': dict(f=[.7,.2,-.55],g=[.45,-.1,-.8],pieces=4,sigma_y=.16,span=3.,frequencies=48),
}


def declared_control(report, reference, case):
    """Validate the saved synthetic design before any evolution."""
    record = record_from_development_report(report)
    if (reference.get('schema') != 'bm1-interval-characteristic-cn-control-v1'
            or reference.get('fixed_development_design') != report.get('declared_design')
            or reference.get('historical_observation_values_read') != 0
            or not reference.get('development_numerics_pass')):
        raise ValueError('Use the passed positive CN reference on the identical synthetic day.')
    baseline = reference['replays'][0]
    if (baseline.get('status') != 'completed'
            or (baseline['cells'],baseline['max_step'],baseline['frequencies'],baseline['span'])
               != (1281,1/12288,64,4.)
            or baseline.get('state_scheme') != 'positive_crank_nicolson'):
        raise ValueError('Saved baseline differs from the declared positive CN control.')
    if case == 'runtime':
        return record,dict(case=case,interval_index=0,cells=1281,max_step=1/12288,
            frequencies=64,span=4.,paired_repetitions=3,order=['uncached-first','cached-first','uncached-first'])
    if case not in CASES:
        raise ValueError('Unknown bounded curve control.')
    spec = CASES[case]
    f = BM1PiecewiseResponse.from_inner_values(spec['f'],pieces=spec['pieces'])
    g = BM1PiecewiseResponse.from_inner_values(spec['g'],pieces=spec['pieces'])
    parameters = (record.parameters[0],f,g,*record.parameters[3:])
    record = replace(record,record_id=case,parameters=parameters,sigma_y=spec['sigma_y'])
    settings = [(1281,1/12288,spec['frequencies'],spec['span'])]
    if case != 'linear-period':
        settings += [(1281,1/24576,spec['frequencies'],spec['span']),
                     (1921,1/24576,spec['frequencies'],spec['span']),
                     (1281,1/12288,2*spec['frequencies'],spec['span']),
                     (1281,1/12288,round(4*spec['frequencies']/spec['span']),4.)]
    return record,dict(case=case,f=f.to_record(),g=g.to_record(),sigma_y=spec['sigma_y'],
        settings=[dict(cells=n,max_step=h,frequencies=k,span=L) for n,h,k,L in settings],
        observations_reused=True,physical_parameters_reused=True,prior_reused=True,
        inputs_constant_within_hours=True,
        scope='Two joint curve/noise fixtures; not isolated causal effects, fit or simulated recovery.')


def _compare(name, a, b, tolerances):
    if a.get('status') != 'completed' or b.get('status') != 'completed':
        return dict(name=name,available=False,development_tolerances_pass=False)
    mass = conservative_mass_transfer(a['x_edges'],a['final_posterior_mass'],b['x_edges'])
    errors = dict(total_log_likelihood=abs(a['log_likelihood']-b['log_likelihood']),
        final_state_mean=abs(a['posterior_state_means'][-1]-b['posterior_state_means'][-1]),
        final_mass_l1=float(np.abs(mass-b['final_posterior_mass']).sum()))
    return dict(name=name,available=True,errors=errors,
        development_tolerances_pass=all(errors[k] <= tolerances[k] for k in errors))


def _runtime_pair(record, index):
    """Paired same-process one-hour measurement; excludes fitter/whole-day claims."""
    model = bm_1(record.baseline,record.baseline,record.horizon,2)
    solver = finite_volume_1d_fitted(vectorized_inputs=True)
    model.fokker_planc_reference = solver
    edges,x,_ = solver.grid(1281)
    prior = record.initial_mass_builder(edges)
    start,end = record.observations.starts[0],record.observations.ends[0]
    drift,diffusion,options = model._fv_evolution_fields(record.parameters,record.sigma_x,
        np.array([start,end]),[],1/12288)
    p = record.parameters
    reward = lambda t,x:model.eval(t,x,p[0],p[1],p[2],p[4],p[5])[-1]
    center,value = float(reward(start,x) @ prior),float(record.observations.values[0])
    order = (False,True) if index%2 == 0 else (True,False)
    predictions, times = {}, {}
    for reuse in order:
        begun = perf_counter()
        predictions[reuse] = balanced_characteristic_prediction(solver,[start,end],prior,
            drift,diffusion,reward,max_step=1/12288,frequencies=64,span=4.,
            state_scheme='positive_crank_nicolson',tilt=(value-center)/record.sigma_y**2,
            center=center,reuse_frozen_operators=reuse,**options)
        times[reuse] = perf_counter()-begun
    same = all(np.array_equal(predictions[False][key],predictions[True][key])
               for key in ('cf','log_amplitude','predicted_mass'))
    if not same:
        raise ValueError('Pure operator reuse changed a numerical array.')
    update = characteristic_gaussian_update(predictions[True],value,record.sigma_y)
    return dict(status='completed',pair=index,order=list(order),bitwise_identical_arrays=same,
        uncached_seconds=times[False],cached_seconds=times[True],log_evidence=update['log_evidence'],
        operator_reuse=predictions[True]['operator_reuse'])


def run_curve_control(report, reference, case, *, saved_rows=(), checkpoint=None, progress=None):
    """Five replays per flexible case, checkpointed independently; no retries."""
    record,design = declared_control(report,reference,case)
    outputs = list(saved_rows)
    total = 3 if case == 'runtime' else len(design['settings'])
    if len(outputs) > total:
        raise ValueError('Checkpoint exceeds the declared control.')
    weights = [1.]*total if case == 'runtime' else [r['cells']/r['max_step']*r['frequencies'] for r in design['settings']]
    for index in range(len(outputs),total):
        begun = perf_counter()
        if case == 'runtime':
            row = _runtime_pair(record,index)
        else:
            settings = design['settings'][index]
            try:
                row = replay_characteristic(record,**settings,state_scheme='positive_crank_nicolson')
            except CharacteristicNumericalError as exc:
                row = dict(status='numerical_guard_failed',**settings,reason=str(exc),
                    diagnostics=exc.diagnostics,scientific_acceptance=False)
            row['elapsed_seconds'] = perf_counter()-begun
        outputs.append(row)
        if checkpoint is not None:
            checkpoint(outputs)
        if progress is not None:
            progress(sum(weights[:len(outputs)]),sum(weights))
    comparisons = []
    if case == 'linear-period':
        comparisons = [_compare('saved_period4_frequency64',outputs[0],reference['replays'][0],report['development_tolerances'])]
    elif case != 'runtime':
        for name,i,j in (('fixed_state_time',0,1),('fixed_time_state',1,2),
                         ('combined_candidate_reference',0,2),('frequency_doubling',0,3),('period_change',0,4)):
            comparisons.append(_compare(name,outputs[i],outputs[j],report['development_tolerances']))
    runtime = None
    if case == 'runtime':
        old = float(np.median([row['uncached_seconds'] for row in outputs]))
        new = float(np.median([row['cached_seconds'] for row in outputs]))
        runtime = dict(uncached_median_seconds=old,cached_median_seconds=new,ratio=old/new,
            scope='One synthetic hour, same process and numerical settings; not historical-fit or whole-day speedup.')
    return dict(schema='bm1-interval-curve-control-v1',fixed_development_design=report['declared_design'],
        declared_control=design,replays=outputs,comparisons=comparisons,runtime=runtime,
        development_tolerances=report['development_tolerances'],
        development_numerics_pass=(all(row['bitwise_identical_arrays'] for row in outputs) if case == 'runtime'
                                   else all(c['development_tolerances_pass'] for c in comparisons)),
        historical_observation_values_read=0,numerical_acceptance=False,scientific_acceptance=False,
        interpretation='Bounded optional synthetic development control. No optimizer, repairs, '
            'relaxed guards, time adaptation, historical backend adoption or H2/H3 acceptance.')
