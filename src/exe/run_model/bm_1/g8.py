"""Registered G8 synthetic point-observation/filter/bank integration study.

No optimizer, historical observations, extra parameter grids or hidden reset.
Run from repository root; choose a fresh --output directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import scipy
from src.control.bm_1 import BM1ControlNumerics, sample_beta_initial_states
from src.method.bm_1_distribution_panel import distribution_summary, baseline, uniform_grid_indices
from src.method.bm_1_g8 import (
    G8BankOrigin, RHO_VALUES, SIGMA_Y, context, known_filter, bank_filter, weighted_control_evaluation,
)
from src.method.bm_1_estimation import bm1_observation_update
from src.method.euler_maruyama import euler_maruyama
from src.params.bm_1 import BM1ObservationSeries, conservative_mass_transfer

TOLERANCES = dict(state_mean=.005,demand_mean=.005,state_variance=.002,demand_variance=.002,
                  state_cdf=.03,demand_cdf=.03,state_quantile=.02,demand_quantile=.02,
                  posterior_mass_l1=.05,projection_l1=.05,log_evidence=.05,
                  bank_weight=.03,checkpoint=1e-10,truth_path=.001,mass=1e-9)


def plain(value):
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        if np.isnan(value):return None
        return 'Infinity' if value>0 else '-Infinity'
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def write(path,value):
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def generate_observations(directory,deadline):
    """A new seeded one-path fixture; no G6b MC path selected by fit quality."""
    model,_,parameters,_,_,_=context(129,deadline=deadline)
    fine_time=np.arange(4609)/192
    initial=sample_beta_initial_states(1,20261011)
    fine=euler_maruyama.brownian_increments(fine_time,1,20261012)
    coarse=euler_maruyama.coarsen_brownian_increments(fine,2)
    paths=[]
    diagnostics=[]
    for times,increments in [(fine_time,fine),(fine_time[::2],coarse)]:
        result=model.euler_maruyama_solver(initial,parameters,.2,n_paths=1,boundary='none',
            t_eval=times,breakpoints=[6.,12.,18.],max_step=np.inf,
            brownian_increments=increments,return_diagnostics=True)
        if result[-1]['total_violations']:
            raise RuntimeError('G8 raw EM truth path left [0,1]; no boundary repair.')
        positions=uniform_grid_indices(times,np.arange(25.))
        paths.append((np.atleast_2d(result[1])[0,positions],np.atleast_2d(result[7])[0,positions]))
        diagnostics.append({key:value for key,value in result[-1].items() if not isinstance(value,np.ndarray)})
    noise=np.random.default_rng(20261013).normal(0.,SIGMA_Y,24)
    observed=np.ones(24,dtype=bool);observed[[5,16]]=False
    values=paths[0][1][1:]+noise;values[~observed]=np.nan
    np.savez_compressed(directory/'fixture.npz',times=np.arange(1.,25.),values=values,observed=observed,
        truth_states=paths[0][0],truth_demand=paths[0][1],coarse_truth_states=paths[1][0],
        coarse_truth_demand=paths[1][1],observation_noise=noise,initial_state=initial,
        fine_time=fine_time,fine_increments=fine)
    error=max(float(np.max(abs(paths[0][j]-paths[1][j]))) for j in (0,1))
    write(directory/'fixture.json',dict(diagnostics=diagnostics,coupled_hourly_max_error=error,
                                      within_registered_truth_tolerance=error<=TOLERANCES['truth_path']))
    return BM1ObservationSeries(np.arange(1.,25.),values,'hour',observed),paths[0]


def summarize(model,parameters,times,edges,predicted,posterior,scores,mask,truth):
    grid=.5*(edges[:-1]+edges[1:])
    summary=dict(prediction=distribution_summary(model,parameters,times,grid,predicted,edges),
                 posterior=distribution_summary(model,parameters,times,grid,posterior,edges),
                 total_log_evidence=float(np.sum(scores[mask])),log_evidence=scores,
                 missing_mass_change=float(np.max(abs(predicted[~mask]-posterior[~mask]))) if np.any(~mask) else 0.,
                 maximum_mass_error=float(max(np.max(abs(predicted.sum(axis=1)-1.)),np.max(abs(posterior.sum(axis=1)-1.)))),
                 minimum_mass=float(min(predicted.min(),posterior.min())))
    for kind in ('prediction','posterior'):
        for quantity,key in [('state','state_mean'),('demand','demand_mean')]:
            actual=truth[0 if quantity=='state' else 1][1:]
            summary[kind][quantity+'_rmse_to_latent_truth']=float(np.sqrt(np.mean((summary[kind][key]-actual)**2)))
    return summary


def retain_known(directory,name,result,model,parameters,seconds,truth):
    arrays=dict(times=result.times,edges=result.x_edges,predicted_mass=result.predicted_mass,
                posterior_mass=result.posterior_mass,log_evidence=result.log_evidence,observed=result.observed)
    if hasattr(result,'posterior_coefficients'):
        arrays.update(predicted_coefficients=result.predicted_coefficients,posterior_coefficients=result.posterior_coefficients)
    np.savez_compressed(directory/(name+'.npz'),**arrays)
    summary=summarize(model,parameters,result.times,result.x_edges,result.predicted_mass,result.posterior_mass,
                      result.log_evidence,result.observed,truth)
    summary.update(seconds=seconds,prediction_diagnostics=result.prediction_diagnostics)
    if hasattr(result,'posterior_projection_diagnostics'):
        summary.update(initial_projection=result.initial_projection_diagnostics,
                       posterior_projection=result.posterior_projection_diagnostics)
    write(directory/(name+'.json'),summary)
    return summary


def retain_bank(directory,name,records,model,parameters,seconds,truth):
    times=np.array([r['time'] for r in records]);mask=np.array([r['observed'] for r in records])
    predicted=np.stack([r['predicted_mass'] for r in records]);posterior=np.stack([r['posterior_mass'] for r in records])
    scores=np.array([r['log_evidence'] for r in records]);logs=np.stack([r['log_weights'] for r in records])
    conditional=np.stack([r['conditional_mass'] for r in records]);edges=records[0]['origin'].x_edges
    arrays=dict(times=times,edges=edges,predicted_mass=predicted,posterior_mass=posterior,
                log_evidence=scores,observed=mask,log_weights=logs,conditional_mass=conditional,
                prior_log_weights=np.stack([r['prior_log_weights'] for r in records]),
                conditional_log_evidence=np.stack([r['conditional_log_evidence'] for r in records]))
    if records[-1]['conditional_coefficients'] is not None:
        arrays.update(conditional_coefficients=np.stack([r['conditional_coefficients'] for r in records]),
                      basis=records[-1]['origin'].basis)
    np.savez_compressed(directory/(name+'.npz'),**arrays)
    for record in records:
        if record['time'] in (12.,24.):write(directory/(name+'_origin_'+str(int(record['time']))+'.json'),record['origin'].to_record())
    summary=summarize(model,parameters,times,edges,predicted,posterior,scores,mask,truth)
    weights=np.exp(logs);nodes=records[-1]['origin'].rho_values
    summary.update(seconds=seconds,weights=weights,rho_mean=weights @ nodes,
        rho_variance=np.sum(weights*(nodes[None,:]-(weights @ nodes)[:,None])**2,axis=1),
        final_entropy=float(-np.sum(weights[-1]*logs[-1])),initial_entropy=float(np.log(len(nodes))),
        final_map_rho=float(nodes[np.argmax(logs[-1])]),
        maximum_conditional_mass_error=float(np.max(abs(conditional.sum(axis=2)-1.))),
        minimum_conditional_mass=float(conditional.min()),
        missing_weight_change=float(np.max(abs(logs[~mask]-arrays['prior_log_weights'][~mask]))),
        initial_projection=records[0]['initial_projection'],
        posterior_projection=[r['posterior_projection'] for r in records])
    write(directory/(name+'.json'),summary)
    return summary


def compare(directory,left,right,bank=False):
    a=json.loads((directory/(left+'.json')).read_text());b=json.loads((directory/(right+'.json')).read_text())
    checks={}
    for kind in ('prediction','posterior'):
        checks[kind]={}
        for key in ('state_mean','demand_mean','state_variance','demand_variance','state_cdf','demand_cdf','state_quantile','demand_quantile'):
            error=float(np.max(abs(np.asarray(a[kind][key])-np.asarray(b[kind][key]))))
            checks[kind][key]=dict(max_error=error,passed=error<=TOLERANCES[key])
    with np.load(directory/(left+'.npz')) as ar,np.load(directory/(right+'.npz')) as br:
        remapped=np.array([conservative_mass_transfer(ar['edges'],m,br['edges']) for m in ar['posterior_mass']])
        l1=float(np.max(np.sum(abs(remapped-br['posterior_mass']),axis=1)))
        observed=ar['observed'];score=float(np.max(abs(ar['log_evidence'][observed]-br['log_evidence'][observed])))
        checks['posterior_l1']=dict(max_error=l1,passed=l1<=TOLERANCES['posterior_mass_l1'])
        checks['log_evidence']=dict(max_error=score,passed=score<=TOLERANCES['log_evidence'])
        if bank:
            error=float(np.max(abs(np.exp(ar['log_weights'])-np.exp(br['log_weights']))))
            checks['weights']=dict(max_error=error,passed=error<=TOLERANCES['bank_weight'])
    gates=[v['passed'] for kind in ('prediction','posterior') for v in checks[kind].values()]
    gates += [checks['posterior_l1']['passed'],checks['log_evidence']['passed']]
    if bank:gates.append(checks['weights']['passed'])
    checks['all_pass']=bool(all(gates))
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='files/output/bm_1/g8_point_bank_NEW')
    parser.add_argument('--open-reference',help='Saved G6b fv_129_h48.npz; reused only for the same declared model.')
    args=parser.parse_args();directory=Path(args.output);directory.mkdir(parents=True,exist_ok=False)
    imports={name:Path(module.__file__) for name,module in list(sys.modules.items())
             if name.startswith('src.') and getattr(module,'__file__',None)}
    imports['runner']=Path(__file__)
    sources={str(p.resolve().relative_to(Path.cwd())):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(imports.values()))}
    registration=dict(study_id=directory.name,scope='one synthetic 24h point-demand record; rho-only bank; no fitting/price optimization',
        source_branch='g/wm-weekly-forecast-plan-20261005',source_parent='a0556d1831858b67e36412e711b0c2f7edde04a2',
        model=dict(C=4.,k=2.,Xi=.4,sigma_x=.2,f='1-2x',g='1-2u'),
        baseline='.5+.1*sin(2*pi*t/24)',prices=[.3,.7,.3,.7],price_edges=[0.,6.,12.,18.,24.],
        prior='Beta(20,20)',time_unit='hour',observations='instantaneous D(X,t)+independent Gaussian noise; right-limit prices',
        sigma_y=SIGMA_Y,missing_hours=[6,17],rho_values=RHO_VALUES,parameter_prior='uniform on five registered nodes',
        truth_seeds=dict(initial=20261011,brownian=20261012,measurement=20261013),truth_steps=[1/192,1/96],
        fv_levels=[dict(cells=129,step=1/48),dict(cells=257,step=1/96)],
        mixture_levels=[dict(cells=129,degree=38,step=1/48),dict(cells=129,degree=78,step=1/48)],
        mixture_degree_reason='M38 preserves G6b reference; M78 lowers continuous central variance floor from .00610 to .00309; fixed before observation generation',
        likelihood_quadrature_order=3,cfl=.4,pilot_cap_seconds=60,main_cap_seconds=600,
        checkpoint_cutoff=12.,checkpoint_backend_replays=['bank_fv257','bank_mix78'],
        handoff=dict(cutoff=12.,horizon=[12.,13.],price=[.3],target='.5+.1*sin(2*pi*t/24)+.03*sin(2*pi*t/24)',optimizer=False),
        tolerances=TOLERANCES,source_sha256=sources,open_reference=args.open_reference,
        software=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        policy='No adaptive extra grid/degree/node/noise search. Failed numerical gates remain negative or inconclusive.')
    write(directory/'registration.json',registration)  # Before any synthetic random generation.
    start=time.perf_counter();pilot_deadline=start+60
    observations,truth=generate_observations(directory,pilot_deadline)
    summaries={};failures={}
    known_specs=[('known_fv129',129,None,1/48),('known_mix38',129,38,1/48),
                 ('known_fv257',257,None,1/96),('known_mix78',129,78,1/48)]
    for index,(name,cells,degree,step) in enumerate(known_specs):
        if index==2:
            pilot_seconds=time.perf_counter()-start
            write(directory/'pilot.json',dict(seconds=pilot_seconds,passed=pilot_seconds<=60,retained=['known_fv129','known_mix38']))
            print(f'Pilot {pilot_seconds:.3f}s; remaining main budget 600s.',flush=True)
            main_start=time.perf_counter();deadline=main_start+600
        limit=pilot_deadline if index<2 else deadline;t=time.perf_counter()
        try:
            result,model,parameters,_=known_filter(cells,observations,step,degree=degree,deadline=limit)
            summaries[name]=retain_known(directory,name,result,model,parameters,time.perf_counter()-t,truth)
            print(f'{name}: {summaries[name]["seconds"]:.3f}s; logZ={summaries[name]["total_log_evidence"]:.5f}',flush=True)
        except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failures[name]=dict(type=type(exc).__name__,message=str(exc),seconds=time.perf_counter()-t)
            write(directory/(name+'_failure.json'),failures[name]);print(f'{name}: STOP {exc}',flush=True)
        if time.perf_counter()>=limit:raise RuntimeError('Registered G8 budget stop; retained partial outputs.')
    bank_specs=[('bank_fv129',129,None,1/48),('bank_fv257',257,None,1/96),
                ('bank_mix38',129,38,1/48),('bank_mix78',129,78,1/48)]
    for name,cells,degree,step in bank_specs:
        t=time.perf_counter()
        def checkpoint(record):
            # A completed observation is an atomic checkpoint boundary.
            if record['time'] in (12.,24.):write(directory/(name+'_origin_'+str(int(record['time']))+'.json'),record['origin'].to_record())
        try:
            records,model,parameters,_=bank_filter(cells,observations,step,degree=degree,deadline=deadline,on_step=checkpoint)
            summaries[name]=retain_bank(directory,name,records,model,parameters,time.perf_counter()-t,truth)
            print(f'{name}: {summaries[name]["seconds"]:.3f}s; final weights={np.round(summaries[name]["weights"][-1],4)}',flush=True)
        except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failures[name]=dict(type=type(exc).__name__,message=str(exc),seconds=time.perf_counter()-t)
            write(directory/(name+'_failure.json'),failures[name]);print(f'{name}: STOP {exc}',flush=True)
        if time.perf_counter()>=deadline:raise RuntimeError('Registered G8 budget stop.')
    resumes={}
    tail=BM1ObservationSeries(observations.times[12:],observations.values[12:],'hour',observations.observed[12:])
    for name,cells,degree,step in [('bank_fv257',257,None,1/96),('bank_mix78',129,78,1/48)]:
        if name not in summaries:continue
        t=time.perf_counter();origin=G8BankOrigin.from_record(json.loads((directory/(name+'_origin_12.json')).read_text()))
        try:
            records,_,_,_=bank_filter(cells,tail,step,degree=degree,origin=origin,deadline=deadline)
            with np.load(directory/(name+'.npz')) as raw:
                mass_error=float(np.max(abs(np.stack([r['conditional_mass'] for r in records])-raw['conditional_mass'][12:])))
                weight_error=float(np.max(abs(np.stack([r['log_weights'] for r in records])-raw['log_weights'][12:])))
                coefficient_error=0. if degree is None else float(np.max(abs(np.stack([r['conditional_coefficients'] for r in records])-raw['conditional_coefficients'][12:])))
                evidence_error=float(np.nanmax(abs(np.array([r['log_evidence'] for r in records])-raw['log_evidence'][12:])))
            resumes[name]=dict(seconds=time.perf_counter()-t,mass_error=mass_error,log_weight_error=weight_error,
                               coefficient_error=coefficient_error,evidence_error=evidence_error,
                               passed=max(mass_error,weight_error,coefficient_error,evidence_error)<=TOLERANCES['checkpoint'])
            write(directory/(name+'_resumed_origin_24.json'),records[-1]['origin'].to_record())
        except (RuntimeError,ValueError,FloatingPointError) as exc:
            resumes[name]=dict(seconds=time.perf_counter()-t,passed=False,message=str(exc))
    comparisons={}
    for label,left,right,bank in [('known_fv_refinement','known_fv129','known_fv257',False),
             ('known_mix38_fv','known_mix38','known_fv257',False),('known_mix78_fv','known_mix78','known_fv257',False),
             ('bank_fv_refinement','bank_fv129','bank_fv257',True),('bank_mix38_fv','bank_mix38','bank_fv257',True),
             ('bank_mix78_fv','bank_mix78','bank_fv257',True)]:
        if left in summaries and right in summaries:comparisons[label]=compare(directory,left,right,bank)
    handoff={}
    if 'bank_fv257' in summaries:
        origin=G8BankOrigin.from_record(json.loads((directory/'bank_fv257_origin_12.json').read_text()))
        t=time.perf_counter();numerics=BM1ControlNumerics(x_steps=257,objective_max_step=1/96,solver_max_step=1/96)
        goal=lambda time:baseline(time)+.03*np.sin(2*np.pi*np.asarray(time)/24)
        handoff=weighted_control_evaluation(origin,baseline=baseline,target=goal,horizon=(12.,13.),
                                            price_edges=[12.,13.],prices=[.3],numerics=numerics)
        node=origin.map_origin();problem=node.control_problem(baseline=baseline,target=goal,horizon=(12.,13.),price_edges=[12.,13.])
        handoff.update(map_cost=problem.build_evaluator(numerics,fast=False).evaluate([.3]).total,
                       seconds=time.perf_counter()-t,map_rho=float(origin.rho_values[node.selected_node]),
                       cutoff=12.,future_observations_used=False,optimized=False)
    open_reference={}
    if args.open_reference:
        original=Path(args.open_reference)
        registration_path=original.parent/'registration.json'
        old=json.loads(registration_path.read_text())
        if (old['model']!=registration['model'] or old['prices']!=registration['prices']
                or old['price_edges']!=registration['price_edges'] or old['prior'].get('law')!='Beta(20,20)'
                or old['baseline']!=registration['baseline']):
            raise ValueError('Saved open reference has a different synthetic contract.')
        shutil.copy2(original,directory/'open_reference.npz');shutil.copy2(registration_path,directory/'open_reference_registration.json')
        with np.load(original) as raw:
            indices=uniform_grid_indices(raw['time'],observations.times) if np.allclose(np.diff(raw['time']),np.diff(raw['time'])[0]) else np.searchsorted(raw['time'],observations.times)
            if not np.allclose(raw['time'][indices],observations.times,rtol=0,atol=1e-12):raise ValueError('Saved reference lacks hourly outputs.')
            mass=raw['cell_mass'][indices];edges=raw['edges']
        model,_,parameters,_,_,_=context(len(edges)-1)
        scores=np.array([bm1_observation_update(model,parameters,SIGMA_Y,row,0.,float(t),float(y),likelihood_quadrature_order=3).log_evidence
                         if observed else np.nan for row,t,y,observed in zip(mass,observations.times,observations.values,observations.observed)])
        open_reference=summarize(model,parameters,observations.times,edges,mass,mass,scores,observations.observed,truth)
        open_reference.update(reused_without_FV_replay=True,source_sha256=hashlib.sha256(original.read_bytes()).hexdigest())
        write(directory/'open_reference.json',open_reference)
    write(directory/'summary.json',dict(summaries=summaries,failures=failures,comparisons=comparisons,
        checkpoint_resume=resumes,handoff=handoff,open_reference=open_reference,
        pilot_seconds=pilot_seconds,main_seconds=time.perf_counter()-main_start,invocation_seconds=time.perf_counter()-start,
        scope='synthetic development integration; method gates do not imply parameter recovery or historical acceptance'))
    print('G8 complete: raw filters, node histories, origin records and checks retained.',flush=True)


if __name__=='__main__':main()
