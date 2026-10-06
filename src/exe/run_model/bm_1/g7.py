"""Registered G7 two-hour handoff/control prototype and priced day admission.

600 s cumulative, including <=60 s pilot, <=100 objective calls/300 s search,
own mixture/FV validation, paired raw EM and actual point-observation update.
This short prototype does not replace the planned full-day G7 acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np
from src.control.bm_1 import BM1ControlNumerics,multistart_powell
from src.control.objective import monte_carlo_tracking_objective
from src.method.bm_1_g7 import OwnedBankPriceEvaluator
from src.method.bm_1_g8 import G8BankOrigin
from src.method.bm_1_distribution_panel import baseline,target,distribution_summary,uniform_grid_indices
from src.method.euler_maruyama import euler_maruyama
from src.params.bm_1 import conservative_mass_transfer
from src.exe.run_model.bm_1.g8 import TOLERANCES
from src.exe.run_model.bm_1.g8_bernstein import SolverAudit,write,sha


def compare_forward(left,right,reference_evaluator):
    a=left['predicted_origin'];b=right['predicted_origin']
    transferred=np.array([conservative_mass_transfer(a.x_edges,m,b.x_edges) for m in a.conditional_mass])
    conditional=float(np.max(np.sum(abs(transferred-b.conditional_mass),axis=1)))
    ma=a.weights@transferred;mb=b.weights@b.conditional_mass
    marginal=float(np.sum(abs(ma-mb)))
    s=reference_evaluator.supports[0];p=s.parameters_for_prices(right['prices'])
    x=.5*(b.x_edges[:-1]+b.x_edges[1:])
    summaries=[distribution_summary(s.model,p,np.array([b.time]),x,m[None,:],b.x_edges) for m in (ma,mb)]
    errors={key:float(np.max(abs(np.asarray(summaries[0][key])-np.asarray(summaries[1][key]))))
            for key in ('state_mean','demand_mean','state_variance','demand_variance','state_cdf',
                        'demand_cdf','state_quantile','demand_quantile')}
    j=abs(left['total']-right['total']);jlimit=max(.001,.02*abs(right['total']))
    passed=(conditional<=TOLERANCES['posterior_mass_l1'] and marginal<=TOLERANCES['posterior_mass_l1'] and j<=jlimit
            and all(value<=TOLERANCES[key] for key,value in errors.items()))
    return dict(conditional_l1=conditional,marginal_l1=marginal,
                terminal_errors=errors,objective_error=j,objective_gate=jlimit,passed=passed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coarse-origin',type=Path,required=True)
    p.add_argument('--fine-origin',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();deadline=start+570.
    source={};audit=[];manifest=json.loads(args.manifest.read_text())
    for name,module in list(sys.modules.items()):
        if name.startswith('src.') and getattr(module,'__file__',None):
            path=Path(module.__file__).resolve();rel=str(path.relative_to(Path.cwd()));data=path.read_bytes()
            source[rel]=sha(path);blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if rel!='src/method/bm_1_g7.py' and manifest.get(rel)!=blob:
                raise ValueError('Imported source differs from main: '+rel)
            audit.append(dict(path=rel,git_blob=blob,sha256=source[rel]))
    for rel in [str(Path(__file__).resolve().relative_to(Path.cwd())),'tests/test_bm1_g7.py']:
        source[rel]=sha(Path(rel))
    inputs={str(x.resolve()):sha(x) for x in (args.coarse_origin,args.fine_origin,args.manifest)}
    registration=dict(source_parent='1bf048940483a4707035e84f707ce4ef407e9fe0',source_sha256=source,input_sha256=inputs,
        budget_seconds=600,compute_seconds=570,export_reserve_seconds=30,pilot_seconds=60,
        prior_panel_debit_seconds=983.266544185,test_charge_seconds=2.,
        scope='2h G7 prototype12->14; full24h job admission only; no full-day optimizer after refusal',
        target='A=.5 fixed ex ante; baseline same frozen analytic G8 function',
        price_edges=[12.,13.,14.],bounds=[0.,1.],terminal_weight=0.,smoothing_weight=0.,
        nodes='Same five owned rho/state laws; cutoff logweights frozen during price forward',
        search='One neutral start per deterministic-MAP and full-bank FV policy; Powell maxfev48 each, max96 calls/300s. Exact incumbent preserved even if optimizer does not converge.',
        cross='neutral/deterministic/stochastic policies; owned M254257/M254513, independent FV513, EM256 per node, coupled h1/192 and1/384, common random numbers',
        seeds=dict(initial=20261021,brownian=20261022,measurement_noise=20261023),
        tolerances=TOLERANCES,objective_gate='max(.001, .02*abs(referenceJ))',
        mc_gate='abs(Jmc-Jfv)<=max(.001,4*sampling_SE); coupled timeerror<=.001',
        admission='1.25 pilot-based complete job estimate incl<=100 search calls and three chosen-policy checks fits remaining570s; otherwise no search',
        tests='One reset/own-state/missing-update contract check passed previously, conservatively charged2s; not repeated',
        excluded='No historical Y, baseline/physical refit, parameter grid, future measurement inside objective or global optimum claim')
    write(out/'registration.json',registration);write(out/'source_audit.json',audit)
    report=dict(registration=registration,prototype_accept=False,full_day_G7_accept=False,historical_accept=False)
    def timeout(*_):raise TimeoutError('G7 registered numerical allowance exhausted.')
    signal.signal(signal.SIGALRM,timeout)
    def arm(at):signal.setitimer(signal.ITIMER_REAL,max(.001,at-time.perf_counter()))
    def checkpoint(stage):
        report['elapsed_seconds']=time.perf_counter()-start;write(out/'progress.json',report)
        print(stage,'elapsed',round(report['elapsed_seconds'],2),'s remaining budget',round(deadline-time.perf_counter(),2),'s',flush=True)
    arm(min(deadline,start+60.))
    try:
        origins=[G8BankOrigin.from_record(json.loads(p.read_text())) for p in (args.coarse_origin,args.fine_origin)]
        if any(o.time!=12. for o in origins):raise ValueError('Expected saved hour12 own origins.')
        numerics=[BM1ControlNumerics(x_steps=n,objective_max_step=1/96,solver_max_step=h)
                  for n,h in [(257,1/96),(513,1/192)]]
        audits=[]
        def configure(projector):
            audit=SolverAudit(projector.projection_solver,projector.dx);audits.append(audit)
            projector.projection_solver=audit
        common=dict(baseline=baseline,target=target,horizon=(12.,14.),price_edges=[12.,13.,14.])
        mix=[OwnedBankPriceEvaluator(o,numerics=n,configure_projector=configure,**common) for o,n in zip(origins,numerics)]
        fv=[OwnedBankPriceEvaluator(o,numerics=n,backend='fv',**common) for o,n in zip(origins,numerics)]
        pilots=[];neutral=[]
        for evaluator in (*mix,*fv):
            t=time.perf_counter();neutral.append(evaluator.evaluate([.5,.5]));pilots.append(time.perf_counter()-t)
            # Export before starting another solve: timeout may interrupt it.
            value=neutral[-1];name='pilot_'+str(len(neutral))
            np.savez_compressed(out/(name+'.npz'),**{k:value[k] for k in
                ['times','midpoints','widths','target','prices','weights','expected_demand','demand_variance','conditional_mass']},
                conditional_coefficients=np.empty(0) if value['conditional_coefficients'] is None else value['conditional_coefficients'])
            write(out/(name+'_origin.json'),value['predicted_origin'].to_record())
            report['completed_pilot_forwards']=dict(seconds=pilots.copy(),completed=len(neutral))
            checkpoint('pilot '+evaluator.backend)
        report['pilot']=dict(seconds=time.perf_counter()-start,forward_seconds=pilots,
            own_refinement=compare_forward(neutral[0],neutral[1],mix[1]),
            mixture_vs_fv=compare_forward(neutral[1],neutral[3],fv[1]))
        report['pilot']['passed']=all(report['pilot'][k]['passed'] for k in ('own_refinement','mixture_vs_fv'))
        write(out/'pilot.json',report['pilot'])
        if not report['pilot']['passed']:raise RuntimeError('Own price-forward numerical gate failed; stop before optimizer.')
        remaining=deadline-time.perf_counter()
        day_search=1.25*100*12*pilots[2]
        day_checks=1.25*3*12*sum(pilots[:2])+30.
        report['day_job']=dict(hours=24,search_estimate_seconds=day_search,cross_estimate_seconds=day_checks,
            total_estimate_seconds=day_search+day_checks,remaining_seconds=remaining,
            admitted=day_search+day_checks<=remaining,performed=False,
            reason='Measured two-hour rates price the day; implementation/prototype does not silently replace full-day acceptance.')
        prototype_estimate=1.25*(96*pilots[2]+2*sum(pilots[:2])+3*pilots[3])+30.
        report['prototype_admission']=dict(estimated_seconds=prototype_estimate,remaining_seconds=remaining,
                                         admitted=prototype_estimate<=remaining)
        write(out/'admission.json',dict(day=report['day_job'],prototype=report['prototype_admission']))
        if not report['prototype_admission']['admitted']:raise TimeoutError('Complete short-control job cannot fit remaining allowance.')
        arm(min(deadline,time.perf_counter()+300.))
        # Deterministic MAP approximation uses that node's own conditional mean.
        node=origins[0].map_origin();x=.5*(node.x_edges[:-1]+node.x_edges[1:])
        problem=node.control_problem(**common,deterministic_initial_state=float(node.posterior_mass@x))
        deterministic=problem.build_deterministic_evaluator(numerics[0])
        policies={'neutral':np.array([.5,.5])};searches={};calls=0
        for name,evaluator in [('deterministic',deterministic),('stochastic',fv[0])]:
            history=[]
            def objective(prices):
                nonlocal calls
                if calls>=96:raise TimeoutError('Shared objective-call cap exhausted.')
                calls+=1;value=float(evaluator(prices));history.append(dict(prices=np.asarray(prices).copy(),value=value))
                write(out/(name+'_search.json'),dict(history=history,calls=calls))
                if len(history)%10==0:checkpoint(name+' search '+str(len(history)))
                return value
            t=time.perf_counter();error=None;success=False
            try:
                result=multistart_powell(objective,[[.5,.5]],options=dict(maxfev=48,maxiter=6,xtol=1e-3,ftol=1e-5))
                success=True
            except RuntimeError as exc:error=str(exc)
            if not history:raise RuntimeError('No complete candidate to preserve.')
            best=min(history,key=lambda item:item['value']);policies[name]=best['prices']
            searches[name]=dict(success=success,error=error,seconds=time.perf_counter()-t,
                                 objective_calls=len(history),best_value=best['value'],prices=best['prices'])
        report['searches']=searches;report['objective_calls']=calls;write(out/'policies.json',policies)
        arm(deadline);evaluations={};checks={}
        for name,prices in policies.items():
            values=neutral if name=='neutral' else [e.evaluate(prices) for e in (*mix,*fv)]
            evaluations[name]=values
            checks[name]=dict(own_refinement=compare_forward(values[0],values[1],mix[1]),
                mixture_vs_fv=compare_forward(values[1],values[3],fv[1]))
            report.setdefault('policy_costs',{})[name]=dict(mixture257=values[0]['total'],mixture513=values[1]['total'],
                fv257=values[2]['total'],fv513=values[3]['total'],bias=values[1]['bias'],variance=values[1]['variance'],
                within=values[1]['within_parameter_variance'],between=values[1]['between_parameter_variance'])
            for level,value in zip(['mix257','mix513','fv257','fv513'],values):
                np.savez_compressed(out/(name+'_'+level+'.npz'),**{k:value[k] for k in
                    ['times','midpoints','widths','target','prices','weights','expected_demand','demand_variance','conditional_mass']},
                    conditional_coefficients=np.empty(0) if value['conditional_coefficients'] is None else value['conditional_coefficients'])
                write(out/(name+'_'+level+'_origin.json'),value['predicted_origin'].to_record())
            checkpoint(name+' own validation')
        report['checks']=checks
        # Stratified posterior MC: same conditional Beta draw and dW for every policy.
        paths={name:[[],[]] for name in policies};final_true=None;mc_diags=[]
        fine_time=12.+np.arange(769)/384
        coarse_time=fine_time[::2]
        indices=[uniform_grid_indices(t,mix[1].supports[0].midpoints) for t in (coarse_time,fine_time)]
        initial=[];increments=[]
        for j,s in enumerate(mix[1].supports):
            rng=np.random.default_rng(20261021+j);component=rng.choice(255,256,p=origins[1].conditional_coefficients[j])
            states=rng.beta(component+1.,255.-component)
            dw=euler_maruyama.brownian_increments(fine_time,256,20261022+j)
            initial.append(states);increments.append(dw)
            for name,prices in policies.items():
                parameters=s.parameters_for_prices(prices)
                for level,(t,dw_level) in enumerate([(coarse_time,euler_maruyama.coarsen_brownian_increments(dw,2)),(fine_time,dw)]):
                    result=s.model.euler_maruyama_solver(states,parameters,s.sigma_x,n_paths=256,boundary='none',
                        t_eval=t,breakpoints=[13.],max_step=np.inf,brownian_increments=dw_level,return_diagnostics=True)
                    if result[-1]['total_violations']:raise RuntimeError('Raw EM boundary violation; no correction.')
                    demand=np.atleast_2d(result[7]);cost=monte_carlo_tracking_objective(s.widths,demand[:,indices[level]],s.target)
                    paths[name][level].append(cost.path_objectives)
                    mc_diags.append(dict(node=j,policy=name,level=level,total_violations=0))
                    if j==2 and name=='stochastic' and level==1:final_true=float(demand[0,-1])
            checkpoint('paired EM node '+str(j))
        np.savez_compressed(out/'mc_common_inputs.npz',initial_states=np.array(initial),fine_increments=np.array(increments),fine_time=fine_time)
        mc={};weights=origins[1].weights
        for name,levels in paths.items():
            arrays=np.array(levels);mean=arrays.mean(-1)@weights
            se=np.sqrt(np.sum(weights**2*np.var(arrays[1],axis=1,ddof=1)/256))
            diff=abs(float(mean[1])-evaluations[name][3]['total'])
            mc[name]=dict(coarse_J=float(mean[0]),fine_J=float(mean[1]),sampling_SE=float(se),fv513_error=diff,
                          coupled_time_error=abs(float(mean[1]-mean[0])),passed=bool(diff<=max(.001,4*se) and abs(mean[1]-mean[0])<=.001))
            np.savez_compressed(out/(name+'_mc_costs.npz'),path_costs=arrays)
        report['mc']=mc;report['mc_diagnostics']=mc_diags
        noise=float(np.random.default_rng(20261023).normal(0.,origins[1].sigma_y));y=final_true+noise
        updates=[]
        for evaluator,value in zip(mix,evaluations['stochastic'][:2]):updates.append(evaluator.assimilate(value,y))
        for name,update in zip(['coarse','fine'],updates):write(out/(name+'_updated_origin.json'),update['origin'].to_record())
        a,b=[update['origin'] for update in updates]
        transferred=np.array([conservative_mass_transfer(a.x_edges,m,b.x_edges) for m in a.conditional_mass])
        feedback_errors=dict(conditional_l1=float(np.max(np.sum(abs(transferred-b.conditional_mass),axis=1))),
            marginal_l1=float(np.sum(abs(a.weights@transferred-b.weights@b.conditional_mass))),
            weights=float(np.max(abs(a.weights-b.weights))),
            log_evidence=abs(updates[0]['log_evidence']-updates[1]['log_evidence']),
            conditional_log_evidence=float(np.max(abs(updates[0]['conditional_log_evidence']-updates[1]['conditional_log_evidence']))))
        feedback_pass=(max(feedback_errors['conditional_l1'],feedback_errors['marginal_l1'])<=TOLERANCES['posterior_mass_l1']
            and feedback_errors['weights']<=TOLERANCES['bank_weight']
            and max(feedback_errors['log_evidence'],feedback_errors['conditional_log_evidence'])<=TOLERANCES['log_evidence'])
        report['feedback']=dict(time=14.,value=y,truth_demand=final_true,noise=noise,
            source='First prespecified fine EM path of true rho=.25, under chosen stochastic price; new synthetic measurement.',
            fine_weights=updates[1]['origin'].weights,log_evidence=updates[1]['log_evidence'],
            coarse_weights=updates[0]['origin'].weights,coarse_log_evidence=updates[0]['log_evidence'],
            conditional_log_evidence=[update['conditional_log_evidence'] for update in updates],
            errors=feedback_errors,passed=feedback_pass,
            maximum_weight_change=float(np.max(abs(updates[1]['origin'].weights-origins[1].weights))))
        report['projection_audits']=[a.record() for a in audits]
        report['prototype_accept']=bool(all(v['passed'] for c in checks.values() for v in c.values())
            and all(m['passed'] for m in mc.values())
            and feedback_pass
            and all(a.record()['maximum_projection_cell_mass_l1']<=TOLERANCES['projection_l1'] for a in audits))
        report['scope']='Short own-posterior control/update prototype; optimizer convergence reported separately; no full-day/global/historical acceptance.'
    except (TimeoutError,RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        report['failure']=dict(type=type(exc).__name__,message=str(exc));checkpoint('STOP')
    finally:
        signal.setitimer(signal.ITIMER_REAL,0.)
        if 'audits' in locals():report['projection_audits']=[a.record() for a in audits]
        report['wall_seconds']=time.perf_counter()-start
        report['source_unchanged']=all(sha(Path(path))==digest for path,digest in source.items())
        report['inputs_unchanged']=all(sha(Path(path))==digest for path,digest in inputs.items())
        report['prototype_accept']=bool(report['prototype_accept'] and report['source_unchanged'] and report['inputs_unchanged'] and report['wall_seconds']<=600)
        write(out/'summary.json',report)
    print(json.dumps({k:report.get(k) for k in ('wall_seconds','failure','pilot','day_job','searches','prototype_accept')},indent=2),flush=True)


if __name__=='__main__':main()
