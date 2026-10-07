"""Registered G7 seed repair; frozen solvers and completed FV/M254 stay unchanged.

One registered invocation only. Original M254 forwards are read-only inputs.
Each completed auxiliary job is atomically stored before another job starts.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

from restore_auxiliary import canonical, read_owned, read_completed_fv, sha, verified_json


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False).encode()+b'\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


@contextmanager
def job_deadline(deadline, stop_type):
    """POSIX worker timer interrupts LS; parent timeout applies on every OS."""
    if time.perf_counter() >= deadline:
        raise stop_type('Auxiliary job reservation is already exhausted.')
    supported = hasattr(signal, 'setitimer')
    if supported:
        def stop(*unused):
            raise stop_type('Auxiliary job reservation expired.')
        previous = signal.signal(signal.SIGALRM, stop)
        signal.setitimer(signal.ITIMER_REAL, max(1e-6, deadline-time.perf_counter()))
    try:
        yield
        if time.perf_counter() >= deadline:
            raise stop_type('Auxiliary job exceeded its reservation.')
    finally:
        if supported:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            signal.signal(signal.SIGALRM, previous)


def validate_streams(em, registered):
    """Reject seed collisions and changed role/node identities before any job."""
    if len(em['initial_seeds'])!=5 or len(em['brownian_seeds'])!=5:
        raise ValueError('Five initial and five Brownian streams required.')
    values=em['initial_seeds']+em['brownian_seeds']+[em['noise_seed']]
    if len(set(values))!=11 or not all(type(s) is int and s>=0 for s in values):
        raise ValueError('Eleven distinct nonnegative integer streams required.')
    for name in ('initial_seeds','brownian_seeds','noise_seed'):
        if em[name]!=registered[name]:
            raise ValueError('Role/node RNG identities differ from registration.')


def worker(out, old):
    # PYTHONPATH/cwd point to the authenticated old numerical checkout.
    import numpy as np
    from src.control.bm_1 import BM1ControlNumerics
    from src.control.objective import monte_carlo_tracking_objective
    from src.method.bm_1_distribution_panel import baseline, target, uniform_grid_indices
    from src.method.bm_1_g7 import OwnedBankPriceEvaluator
    from src.method.bm_1_g7_jobs import NodeJobStore, NodeJobStop, CheckpointedBankControl, atomic_json, digest
    from src.method.bm_1_g8 import G8BankOrigin
    from src.method.euler_maruyama import euler_maruyama
    from src.params.bm_1 import conservative_mass_transfer
    from src.exe.run_model.bm_1.g7 import compare_forward
    from src.exe.run_model.bm_1.g8_bernstein import SolverAudit
    from src.exe.run_model.bm_1.g8 import TOLERANCES

    reg = json.loads((out/'registration.json').read_text())
    plan = verified_json(out/'design.json', reg['design_sha256'])
    spec = plan['proposed_measurement']
    validate_streams(spec['em'],reg['streams'])
    audit = verified_json(out/'previous_raw_audit.json', reg['previous_audit_sha256'])
    for name, expected in reg['launcher_sha256'].items():
        if sha(out/name) != expected:
            raise ValueError('Registered launcher source changed.')
    store = NodeJobStore(out/'campaign', prior_debit=reg['prior_debit'],
                        ceiling=1800., max_new_seconds=reg['worker_seconds'])
    report = dict(schema='g7-neutral-auxiliary-worker-v1', scientific_accept=False,
                  full_day_budget_admitted=False, actual_optimizer_started=False,
                  auxiliary_gates_passed=False, new_m254_forwards=0, stages={})
    started = time.perf_counter(); evaluators = {}; evaluations = {}; updates = {}; audits = {}
    caps = {j['id']:j['reservation_each_seconds'] for j in spec['jobs']}
    continuation = reg.get('continuation')
    if continuation:
        caps.update(reg['job_caps'])
    reg_hash = sha(out/'registration.json')

    def job(kind, index, seconds, compute, **identity):
        key = dict(index=index, kind=kind, registration_sha256=reg_hash, **identity)
        def guarded(deadline):
            with job_deadline(deadline, NodeJobStop):
                return compute()
        before = time.perf_counter()
        with store.lock():
            payload = store.node(key, seconds, guarded)
        report['stages'].setdefault(kind, []).append(dict(index=index, wall_seconds=time.perf_counter()-before,
                                                          job_id=digest(key)))
        atomic_json(out/'progress.json', report)
        print('Completed',kind,index,'elapsed',round(time.perf_counter()-started,2),
              'campaign remaining',round(1800-store.charged(),2),flush=True)
        return payload

    def bank_plain(value):
        return {k:v.to_record() if k=='predicted_origin' else v for k,v in value.items() if k!='_owner'}

    try:
        owned_box = {}
        def first_restore():
            owned_box.update(read_owned(old, audit, spec['reuse']))
            if continuation:
                owned_box['fv_reused'] = read_completed_fv(Path(continuation['root']), continuation)
            return dict(source_files=41, old_completed_nodes=10, old_ledger_sha256=owned_box['ledger_sha256'])
        # Charge verification together with the first evaluator setup.
        for index,(backend,label,cells) in enumerate([('mixture','fine',513),('mixture','coarse',257),
                                                    ('fv','fine',513),('fv','coarse',257)]):
            name = ('m254_' if backend=='mixture' else 'fv_')+label
            def setup(backend=backend,label=label,cells=cells,name=name,index=index):
                verification = first_restore() if index==0 else {}
                original = owned_box['registration']
                origin = G8BankOrigin.from_verified_bytes((old/'inputs'/(label+'.json')).read_bytes(),
                                                          original['input_files'][label]['sha256'])
                def configure(projector):
                    audit_solver = SolverAudit(projector.projection_solver, projector.dx)
                    projector.projection_solver = audit_solver; audits[name] = audit_solver
                evaluator = OwnedBankPriceEvaluator(origin,baseline=baseline,target=target,horizon=tuple(spec['horizon']),
                    price_edges=spec['price_edges'], numerics=BM1ControlNumerics(x_steps=cells,
                    objective_max_step=spec['objective_max_step'],solver_max_step=spec['solver_steps'][str(cells)]),
                    backend=backend,configure_projector=configure if backend=='mixture' else None)
                probe = object.__new__(CheckpointedBankControl)
                probe.evaluator=evaluator; probe.root=old/'frozen_source'; probe.sources=owned_box['sources']
                probe.check_sources(); contract=probe.describe()
                if backend=='mixture':
                    key=spec['reuse'][label+'_contract']
                    if digest(contract)!=key:
                        raise ValueError('Rehydrated evaluator differs from the entire owned numerical contract.')
                    result=evaluator.combine_nodes(spec['prices'],owned_box['nodes'][label])
                    evaluations[name]=result
                evaluators[name]=evaluator
                return dict(backend=backend,label=label,contract=contract,contract_id=digest(contract),**verification)
            receipt=job('restore_setup',index,caps['restore_setup'],setup,backend=backend,label=label)
            if name not in evaluators:
                raise NodeJobStop('Setup is invocation-specific; cached metadata cannot restore live objects.')
            atomic_json(out/(name+'_setup.json'),receipt)

        for label in ('fine','coarse'):
            name='fv_'+label;e=evaluators[name];nodes=[]
            contract_id=digest(evaluators_contract(e,owned_box['sources'],CheckpointedBankControl,old))
            for j in range(5):
                if continuation and contract_id != digest(owned_box['fv_reused']['contracts'][label]):
                    raise ValueError('Live FV contract differs from imported full numerical contract.')
                nodes.append(job('fv_'+label+'_nodes',j,caps['fv_'+label+'_nodes'],
                     (lambda j=j,label=label:owned_box['fv_reused']['nodes'][label][j]) if continuation else
                     (lambda j=j,e=e:e.evaluate_node(spec['prices'],j)),
                     contract=contract_id,prices=spec['prices'],receipt_reuse=bool(continuation)))
            value=e.combine_nodes(spec['prices'],nodes);evaluations[name]=value
            atomic_json(out/(name+'_bank.json'),bank_plain(value))

        fine_time=12.+np.arange(769)/384.;coarse_time=fine_time[::2]
        indices=[uniform_grid_indices(t,evaluators['m254_fine'].supports[0].midpoints) for t in (coarse_time,fine_time)]
        mc_nodes=[]
        for j,support in enumerate(evaluators['m254_fine'].supports):
            def em(j=j,support=support):
                rng=np.random.default_rng(spec['em']['initial_seeds'][j])
                component=rng.choice(255,256,p=evaluators['m254_fine'].origin.conditional_coefficients[j])
                initial=rng.beta(component+1.,255.-component)
                dw=euler_maruyama.brownian_increments(fine_time,256,spec['em']['brownian_seeds'][j])
                levels=[];parameters=support.parameters_for_prices(spec['prices'])
                for level,(t,increments) in enumerate([(coarse_time,euler_maruyama.coarsen_brownian_increments(dw,2)),(fine_time,dw)]):
                    result=support.model.euler_maruyama_solver(initial,parameters,support.sigma_x,n_paths=256,
                        boundary='none',t_eval=t,breakpoints=[13.],max_step=np.inf,
                        brownian_increments=increments,return_diagnostics=True)
                    if result[-1]['total_violations']:
                        raise RuntimeError('Raw EM boundary violation; no clipping/reflection.')
                    demand=np.atleast_2d(result[7]);state=np.atleast_2d(result[1])
                    cost=monte_carlo_tracking_objective(support.widths,demand[:,indices[level]],support.target)
                    if not all(np.all(np.isfinite(a)) for a in (state,demand,cost.path_objectives)):
                        raise RuntimeError('Nonfinite EM science arrays cannot be exported.')
                    diagnostics=dict(result[-1])
                    if diagnostics.get('requested_max_step') == np.inf:
                        diagnostics['requested_max_step']='Infinity'
                    levels.append(dict(state=state,demand=demand,path_costs=cost.path_objectives,diagnostics=diagnostics))
                return dict(node=j,component=component,initial=initial,fine_time=fine_time,
                            fine_increments=dw,levels=levels)
            mc_nodes.append(job('coupled_em_nodes',j,caps['coupled_em_nodes'],em,
                                origin_contract=spec['reuse']['fine_contract'],prices=spec['prices']))
        truth=float(mc_nodes[2]['levels'][1]['demand'][0][-1])
        noise=float(np.random.default_rng(spec['em']['noise_seed']).normal(0.,evaluators['m254_fine'].origin.sigma_y))
        y=truth+noise
        atomic_json(out/'measurement.json',dict(time=14.,value=y,truth=truth,noise=noise,
                    policy=spec['prices'],truth_node=2,truth_rho=.25,truth_path=0))
        for name in ('m254_fine','m254_coarse','fv_fine','fv_coarse'):
            e=evaluators[name];value=evaluations[name]
            def update(e=e,value=value):
                u=e.assimilate(value,y)
                # Missing measurement must preserve every owned array and weight.
                missing=e.assimilate(value,0.,observed=False)['origin']
                if (not np.array_equal(missing.conditional_mass,value['predicted_origin'].conditional_mass)
                        or not np.array_equal(missing.log_weights,value['predicted_origin'].log_weights)):
                    raise RuntimeError('Missing observation changed the predicted law.')
                return dict(origin=u['origin'].to_record(),conditional_log_evidence=u['conditional_log_evidence'],
                            log_evidence=u['log_evidence'],projections=u['projections'],missing_update_preserved=True)
            updates[name]=job('feedback_'+name,-1,caps['feedback_'+name],update,value=y,prices=spec['prices'])
            atomic_json(out/(name+'_update.json'),updates[name])

        def reduce():
            pairs={}
            for left,right in [('m254_coarse','m254_fine'),('fv_coarse','fv_fine'),
                               ('m254_fine','fv_fine'),('m254_coarse','fv_coarse')]:
                pairs[left+'__'+right]=compare_forward(evaluations[left],evaluations[right],evaluators[right])
            costs=np.array([[n['levels'][i]['path_costs'] for n in mc_nodes] for i in (0,1)])
            w=evaluators['m254_fine'].origin.weights;means=costs.mean(-1)@w
            se=float(np.sqrt(np.sum(w*w*np.var(costs[1],axis=1,ddof=1)/256)))
            errors={name:abs(float(means[1])-evaluations[name]['total']) for name in ('m254_fine','fv_fine')}
            em_pass=all(v<=max(.001,4*se) for v in errors.values()) and abs(float(means[1]-means[0]))<=.001
            feedback={}
            for left,right in [('m254_coarse','m254_fine'),('fv_coarse','fv_fine'),
                               ('m254_fine','fv_fine'),('m254_coarse','fv_coarse')]:
                a,b=[G8BankOrigin.from_record(updates[n]['origin']) for n in (left,right)]
                transferred=np.array([conservative_mass_transfer(a.x_edges,m,b.x_edges) for m in a.conditional_mass])
                v=dict(conditional_l1=float(np.max(np.sum(abs(transferred-b.conditional_mass),axis=1))),
                    marginal_l1=float(np.sum(abs(a.weights@transferred-b.weights@b.conditional_mass))),
                    weights=float(np.max(abs(a.weights-b.weights))),
                    log_evidence=abs(updates[left]['log_evidence']-updates[right]['log_evidence']),
                    conditional_log_evidence=float(np.max(abs(np.array(updates[left]['conditional_log_evidence'])-
                                                              np.array(updates[right]['conditional_log_evidence'])))))
                v['passed']=bool(max(v['conditional_l1'],v['marginal_l1'])<=TOLERANCES['posterior_mass_l1']
                    and v['weights']<=TOLERANCES['bank_weight']
                    and max(v['log_evidence'],v['conditional_log_evidence'])<=TOLERANCES['log_evidence'])
                feedback[left+'__'+right]=v
            projection={name:a.record() for name,a in audits.items()}
            passed=all(v['passed'] for v in pairs.values()) and em_pass and all(v['passed'] for v in feedback.values())
            passed=passed and all(v['maximum_projection_cell_mass_l1']<=TOLERANCES['projection_l1'] for v in projection.values())
            return dict(forward_pairs=pairs,em=dict(coarse_J=float(means[0]),fine_J=float(means[1]),
                stratified_SE=se,fp_errors=errors,coupled_time_error=abs(float(means[1]-means[0])),passed=bool(em_pass)),
                feedback_pairs=feedback,projection_audits=projection,auxiliary_gates_passed=bool(passed),
                costs={name:v['total'] for name,v in evaluations.items()},
                new_fv_forwards=0 if continuation else 10,reused_fv_forwards=10 if continuation else 0,
                new_em_level_forwards=10,previous_unsaved_em_level_forwards=2 if continuation else 0,
                new_point_updates=4,
                future_price_search_permitted=False,independent_quadrature_accept=False)
        reduced=job('reduce_audit',-1,caps['reduce_audit'],reduce)
        report.update(reduced);report['stop_reason']='Registered neutral auxiliary measurement completed.'
    except Exception as exc:
        report['stop_reason']=repr(exc);report['error_type']=type(exc).__name__
    finally:
        report['worker_body_seconds']=time.perf_counter()-started
        report['ledger_increment']=store.charged()-reg['prior_debit']
        report['partial_projection_audits']={name:a.record() for name,a in audits.items()}
        report['old_ledger_unchanged']=sha(old/'campaign/ledger.json')==audit['raw_data_archive']['files']['campaign/ledger.json']['sha256']
        atomic_json(out/'worker_summary.json',report)
        print('STOP',report.get('stop_reason'),'seconds',round(report['worker_body_seconds'],2),flush=True)


def evaluators_contract(e, sources, cls, old):
    probe=object.__new__(cls);probe.evaluator=e;probe.sources=sources;probe.root=old/'frozen_source'
    probe.check_sources()
    return probe.describe()


def parent(out,old):
    reg=json.loads((out/'registration.json').read_text())
    if (out/'invocation.json').exists() or (out/'campaign/ledger.json').exists():
        raise ValueError('One invocation only; explicit new registration required for resume/retry.')
    if sha(out/'design.json')!=reg['design_sha256']:
        raise ValueError('Design bytes changed.')
    plan=json.loads((out/'design.json').read_text())
    continuation = reg.get('continuation')
    budget=verified_json(out/'previous_final_budget.json',reg['previous_final_budget_sha256'])
    completed=verified_json(out/'previous_completed_invocation.json',reg['previous_completed_invocation_sha256'])
    review=verified_json(out/'previous_auxiliary_review.json',reg['previous_auxiliary_review_sha256'])
    expected_caps={'restore_setup':8,'fv_fine_nodes':.5,'fv_coarse_nodes':.5,'coupled_em_nodes':4,
                   'feedback_m254_fine':4,'feedback_m254_coarse':3,
                   'feedback_fv_fine':.5,'feedback_fv_coarse':.5,'reduce_audit':3}
    seeds=plan['proposed_measurement']['em']
    validate_streams(seeds,reg['streams'])
    previous=verified_json(Path(continuation['root'])/'invocation.json',continuation['invocation_sha256'])
    valid=(reg['schema']=='g7-auxiliary-seed-repair-registration-v1' and
           reg['prior_debit']==budget['campaign_debit']==completed['campaign_debit']+20 and
           review['execution_commit']==reg['previous_execution_commit'] and
           previous['scope_charge']==continuation['preserved_scope_charge'] and
           reg['campaign_ceiling']==1800 and reg['scope_ceiling']==120 and
           reg['worker_seconds']==70 and reg['stop_export_reserve']==30 and reg['review_charge']==20 and
           reg['scope_ceiling']==reg['worker_seconds']+reg['stop_export_reserve']+reg['review_charge'] and
           reg['job_caps']==expected_caps and reg['prior_debit']+reg['scope_ceiling']<=1800 and
           seeds['initial_seeds']==reg['streams']['initial_seeds'] and
           seeds['brownian_seeds']==reg['streams']['brownian_seeds'] and
           seeds['noise_seed']==reg['streams']['noise_seed'] and reg['one_invocation_only'])
    if not valid:
        raise ValueError('Registered preserved budget/scope differs from the reviewed design/addendum.')
    for name,expected in reg['launcher_sha256'].items():
        if sha(out/name)!=expected:raise ValueError('Launcher bytes changed.')
    env=dict(os.environ,PYTHONPATH=str(old/'frozen_source'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.perf_counter();hard_stop=False
    with (out/'worker.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-B',str(out/Path(__file__).name),
                   '--output',str(out),'--previous',str(old),'--worker'],cwd=old/'frozen_source',env=env,
                   stdout=log,stderr=subprocess.STDOUT)
        try:process.wait(timeout=reg['worker_seconds'])
        except subprocess.TimeoutExpired:
            hard_stop=True;process.kill();process.wait()
    wall=time.perf_counter()-start
    ledger=out/'campaign/ledger.json'
    increment=sum(e['charged_seconds'] for e in json.loads(ledger.read_text())['entries']) if ledger.exists() else 0.
    result=dict(schema='g7-neutral-auxiliary-invocation-v1',worker_wall_seconds=wall,
        ledger_increment=increment,native_charge=max(wall,increment),hard_stop=hard_stop,
        returncode=process.returncode,review_charge=reg['review_charge'],scientific_accept=False,
        full_day_budget_admitted=False,actual_optimizer_started=False)
    write(out/'invocation.json',result)
    before=time.perf_counter()
    # Source is repo-backed. Raw archive contains exact registration, receipts,
    # common random inputs, logs and partial results; no input/source duplication.
    archive=out.parent/(out.name+'_data.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='writer.lock' and not p.name.endswith('.py'):
                z.write(p,str(p.relative_to(out)))
    result['export_seconds']=time.perf_counter()-before
    result['scope_charge']=result['native_charge']+result['export_seconds']+reg['review_charge']
    result['campaign_debit']=reg['prior_debit']+result['scope_charge']
    result['remaining_seconds']=1800-result['campaign_debit']
    result['within_scope_ceiling']=result['scope_charge']<=reg['scope_ceiling']
    write(out/'invocation.json',result)
    # Final invocation is exported separately: archive's original invocation
    # precedes measured ZIP time. Keep both bytes rather than rewriting a ZIP.
    write(out/'archive_index.json',dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
          final_invocation_path=str(out/'invocation.json'),final_invocation_sha256=sha(out/'invocation.json')))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args.output.resolve(),args.previous.resolve())
    else:parent(args.output.resolve(),args.previous.resolve())
