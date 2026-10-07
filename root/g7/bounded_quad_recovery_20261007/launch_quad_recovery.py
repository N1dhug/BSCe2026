"""One preregistered recovery: only missing quad nodes3/4, old journals read-only."""
import argparse,hashlib,json,os,signal,subprocess,sys,time,zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from receipt_recovery import checked_json,sha
from restore_auxiliary import read_owned,canonical,verified_receipt
def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False).encode() + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    if os.name == 'posix':
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


@contextmanager
def job_deadline(deadline, stop_type):
    if time.perf_counter() >= deadline:
        raise stop_type('Reservation already exhausted.')
    supported = hasattr(signal, 'setitimer')
    if supported:
        def stop(*unused):
            raise stop_type('Registered per-job reservation expired.')
        previous = signal.signal(signal.SIGALRM, stop)
        signal.setitimer(signal.ITIMER_REAL, max(1e-6, deadline-time.perf_counter()))
    try:
        yield
        if time.perf_counter() >= deadline:
            raise stop_type('Registered per-job reservation exceeded.')
    finally:
        if supported:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            signal.signal(signal.SIGALRM, previous)



def validate(out):
    reg=json.loads((out/'registration.json').read_bytes())
    for name,h in reg['files_sha256'].items():
        if sha(out/name)!=h:raise ValueError('Registered bytes changed: '+name)
    if (reg['schema']!='g7-bounded-quad-recovery-registration-v1' or reg['worker_seconds']!=70
        or reg['execution_ceiling']!=100 or reg['campaign_ceiling']!=1910
        or reg['review_charge']!=10 or reg['export_reserve']!=20 or reg['preparation_charge']!=10
        or reg['missing_node_indices']!=[3,4] or not reg['user_approved_additional_max_seconds']==110
        or reg['prior_debit']!=1809.1415078599962 or reg['prior_debit']+100>1910):
        raise ValueError('Recovery budget or finite node scope changed.')
    return reg

def worker(out,old,seed,partial):
    reg=validate(out);native=old/'frozen_source'
    manifest=checked_json(out/'native_manifest.json',reg['files_sha256']['native_manifest.json'])
    for name,h in manifest.items():
        if sha(native/name)!=h:raise ValueError('Native source changed: '+name)
    import numpy as np
    from src.control.bm_1 import BM1ControlNumerics
    from src.method.bm_1_distribution_panel import baseline,target
    from src.method.bm_1_g7 import OwnedBankPriceEvaluator
    from src.method.bm_1_g7_jobs import NodeJobStore,NodeJobStop,CheckpointedBankControl,atomic_json,digest
    from src.method.bm_1_g8 import G8BankOrigin
    from src.exe.run_model.bm_1.g7 import compare_forward
    from src.exe.run_model.bm_1.g8 import TOLERANCES
    from src.exe.run_model.bm_1.g8_bernstein import SolverAudit
    from receipt_recovery import VerifiedReceiptIndex,guarded_store_class,audit_store
    Guard=guarded_store_class(NodeJobStore,native_source_path=native/'src/method/bm_1_g7_jobs.py',native_source_sha256=manifest['src/method/bm_1_g7_jobs.py'])
    store=Guard(out/'campaign',prior_debit=reg['prior_debit'],ceiling=1910.,max_new_seconds=70.)
    report=dict(schema='g7-bounded-quad-recovery-worker-v1',completed=False,new_forwards=0,reused_new_forwards=13,scientific_accept=False,full_day_budget_admitted=False,stages=[])
    start=time.perf_counter();box={};reg_hash=sha(out/'registration.json')
    def job(kind,index,cap,compute,**identity):
        key=dict(kind=kind,index=index,registration_sha256=reg_hash,**identity)
        def timed(deadline):
            with job_deadline(deadline,NodeJobStop):return compute()
        before=time.perf_counter()
        with store.lock():payload=store.node(key,cap,timed)
        report['stages'].append(dict(kind=kind,index=index,job_id=digest(key),wall_seconds=time.perf_counter()-before))
        atomic_json(out/'progress.json',report)
        print('Completed',kind,index,'elapsed',round(time.perf_counter()-start,3),flush=True)
        return payload
    def typed_origin(record):
        if record['backend']=='mixture':
            # Parent contract/bank has an external whole-file hash. Preserve its
            # embedded basis bytes via the native portability loader.
            raw=canonical(record)
            return G8BankOrigin.from_verified_bytes(raw,hashlib.sha256(raw).hexdigest())
        return G8BankOrigin.from_record(record)
    try:
        def verify():
            plan=checked_json(out/'design.json',reg['files_sha256']['design.json'])
            audit=checked_json(out/'previous_audit.json',reg['files_sha256']['previous_audit.json'])
            box['owned']=read_owned(old,audit,plan['reuse'])
            pm=checked_json(out/'partial_manifest.json',reg['files_sha256']['partial_manifest.json'])
            checked_json(partial/'registration.json',pm['source_registration_sha256'])
            ledger=checked_json(partial/'campaign/ledger.json',pm['source_ledger_sha256'])
            entries={r['job_id']:r for r in ledger['entries']}
            payloads={};jobs={}
            for r in pm['entries']:
                p=partial/'campaign/nodes'/(r['job_id']+'.json')
                if entries[r['job_id']]['status']!='complete' or digest(r['job'])!=r['job_id']:raise ValueError('Incomplete/changed imported receipt.')
                receipt=checked_json(p,r['receipt_sha256'])
                if receipt['job']!=r['job'] or receipt['payload_sha256']!=r['payload_sha256'] or digest(receipt['payload'])!=r['payload_sha256']:raise ValueError('Imported payload changed.')
                payloads[r['job_id']]=receipt['payload'];jobs[r['job_id']]=r['job']
            def rows(kind,policy=None):
                return sorted((payloads[k] for k,j in jobs.items() if j['kind']==kind and (policy is None or j.get('policy')==policy)),key=lambda x:x.get('index',-1))
            box['quad_nodes']=rows('quadrature_selected')
            if [x['index'] for x in box['quad_nodes']]!=[0,1,2]:raise ValueError('Exact partial quad indices0/1/2 required.')
            box['candidate_nodes']={p:rows('coarse_candidates',p) for p in ('candidate1','candidate2')}
            selection=rows('select_coarse')[0]
            if selection['candidate_index']!=0 or selection['coarse_costs'][0]!=min(selection['coarse_costs']):raise ValueError('Previously selected neutral candidate changed.')
            box['selection']=selection
            quad_job=[j for j in jobs.values() if j['kind']=='restore_setup' and j['evaluator']=='quad'][0]
            box['quad_setup']=payloads[digest(quad_job)]
            if digest(box['quad_setup']['contract'])!=box['quad_setup']['contract_id']:raise ValueError('Quad contract changed.')
            if any(j['contract']!=box['quad_setup']['contract_id'] or j['prices']!=[.5,.5] for j in jobs.values() if j['kind']=='quadrature_selected'):raise ValueError('Quad source node contract differs.')
            idx=VerifiedReceiptIndex(seed,out/'adoption_manifest.json',manifest_sha256=reg['files_sha256']['adoption_manifest.json'],eager_payload_validation=False)
            fvrows=sorted((r for r in idx.entries.values() if r['job']['kind']=='fv_fine_nodes'),key=lambda r:r['job']['index'])
            if [r['job']['index'] for r in fvrows]!=list(range(5)):raise ValueError('Five neutral FV nodes required.')
            box['fv_nodes']=[idx.payload(r['job']) for r in fvrows]
            box['fv_setup']=checked_json(seed/'fv_fine_setup.json',reg['external_files_sha256']['fv_setup'])
            if digest(box['fv_setup']['contract'])!=box['fv_setup']['contract_id'] or any(r['job']['contract']!=box['fv_setup']['contract_id'] or r['job']['prices']!=[.5,.5] for r in fvrows):raise ValueError('FV contract differs.')
            return dict(partial_receipts=19,reused_new_forwards=13,owned_nodes=10,fv_nodes=5,selection=selection,source_ledger_read_only=True)
        job('verify_read_only_sources_receipts_inputs',-1,8,verify)
        def setup():
            origin=G8BankOrigin.from_verified_bytes((old/'inputs/fine.json').read_bytes(),reg['external_files_sha256']['fine_input'])
            def configure(projector):
                box['audit']=SolverAudit(projector.projection_solver,projector.dx);projector.projection_solver=box['audit']
            e=OwnedBankPriceEvaluator(origin,baseline=baseline,target=target,horizon=(12.,14.),price_edges=[12.,13.,14.],numerics=BM1ControlNumerics(x_steps=513,objective_max_step=1/192,solver_max_step=1/192),backend='mixture',configure_projector=configure)
            adapter=object.__new__(CheckpointedBankControl);adapter.evaluator=e;adapter.root=native;adapter.sources=manifest;adapter.check_sources();contract=adapter.describe();cid=store.bind(contract)
            if cid!=box['quad_setup']['contract_id']:raise ValueError('Rehydrated quad differs from frozen partial contract.')
            box['evaluator']=e;box['contract_id']=cid
            # Reuse native combine_nodes with lightweight support metadata only.
            # No additional numerical evaluator/projector or forward is created.
            def reduce_old(contract,nodes,prices):
                origin=typed_origin(contract['origin']);s0=e.supports[0]
                a=object.__new__(OwnedBankPriceEvaluator);a.origin=origin;a.backend=contract['backend']
                a.supports=tuple(SimpleNamespace(validate_prices=s0.validate_prices,solver_t_eval=np.asarray(contract['times']),x=.5*(origin.x_edges[:-1]+origin.x_edges[1:]),dx=float(origin.x_edges[1]-origin.x_edges[0]),widths=np.asarray(contract['widths']),midpoints=np.asarray(contract['midpoints']),target=np.asarray(target(np.asarray(contract['midpoints']))),model=s.model) for s in e.supports)
                return a.combine_nodes(prices,nodes)
            owned=box['owned'];box['coarse']=[reduce_old(owned['contracts']['coarse'],owned['nodes']['coarse'],[.5,.5])]
            for k,p in enumerate(([.3,.7],[.7,.3]),1):box['coarse'].append(reduce_old(owned['contracts']['coarse'],box['candidate_nodes']['candidate'+str(k)],p))
            box['fine']=reduce_old(owned['contracts']['fine'],owned['nodes']['fine'],[.5,.5])
            box['fv']=reduce_old(box['fv_setup']['contract'],box['fv_nodes'],[.5,.5])
            if any(abs(b['total']-v)>1e-12 for b,v in zip(box['coarse'],box['selection']['coarse_costs'])):raise ValueError('Coarse selection cost no longer agrees.')
            return dict(contract_id=cid,contract=contract,only_quad_runtime_rehydrated=True,old_banks_natively_reduced=True)
        s=job('rehydrate_quad_only',-1,4,setup);atomic_json(out/'quad_setup.json',s)
        nodes=box['quad_nodes'];e=box['evaluator']
        for j in (3,4):
            nodes.append(job('missing_quad_nodes',j,18,lambda j=j:e.evaluate_node([.5,.5],j),contract=box['contract_id'],prices=[.5,.5]))
            report['new_forwards']+=1
        def finish():
            quad=e.combine_nodes([.5,.5],nodes)
            def summary(b):
                origin=b['predicted_origin'];x=.5*(origin.x_edges[:-1]+origin.x_edges[1:])
                return dict(J=float(b['total']),bias=float(b['bias']),variance=float(b['variance']),within_parameter_variance=float(b['within_parameter_variance']),between_parameter_variance=float(b['between_parameter_variance']),node_costs=b['node_costs'].tolist(),weights=b['weights'].tolist(),expected_energy_gap=float(b['widths']@(b['expected_demand']-baseline(b['midpoints']))),endpoint_state_mean=float((origin.weights@origin.conditional_mass)@x))
            pairs={'coarse__fine':compare_forward(box['coarse'][0],box['fine'],e),'fine__quadrature':compare_forward(box['fine'],quad,e),'fine__fv':compare_forward(box['fine'],box['fv'],e)}
            projection=box['audit'].record()
            result=dict(schema='g7-completed-bounded-price-review-v1',selected_index=0,selected_prices=[.5,.5],candidate_prices=[[.5,.5],[.3,.7],[.7,.3]],coarse=[summary(b) for b in box['coarse']],selected_fine=summary(box['fine']),selected_quadrature=summary(quad),selected_fv=summary(box['fv']),forward_pairs=pairs,forward_gates_passed=all(p['passed'] for p in pairs.values()),quad_bank_complete=True,quad_reused_node_indices=[0,1,2],quad_new_node_indices=[3,4],new_projection_audit=projection,new_projection_gate_passed=projection['maximum_projection_cell_mass_l1']<=TOLERANCES['projection_l1'],quad_projection_audit_complete=False,missing_projection_audit_indices=[0,1,2],numerical_validation_passed=False,scientific_accept=False,improvement_screen_passed=False,full_day_budget_admitted=False,continuous_global_optimality_claim=False,pure_quadrature_error_certificate=False,limitation='Interrupted original worker lost SolverAudit fit-error maxima for quad nodes0/1/2. Saved native trajectory diagnostics preserve mass/coefficient checks but not raw-target projection L1. No replay permitted; full projection gate remains unproven.',reference_evaluator_use='Only terminal model/price response used by compare_forward; quad support has same pinned fine/FV model and513 grid. No forward in comparisons.')
            atomic_json(out/'complete_quad_bank.json',{k:(v.to_record() if k=='predicted_origin' else v) for k,v in quad.items() if k not in ('_owner','conditional_mass','conditional_coefficients','diagnostics')})
            atomic_json(out/'result.json',result)
            return result
        final=job('full_bank_reduce_and_comparisons',-1,2,finish);report.update(final);report['completed']=True
    except Exception as exc:
        import traceback
        report.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        report['worker_elapsed_seconds']=time.perf_counter()-start;report['campaign_debit_in_ledger']=store.charged()
        try:
            with store.lock():audit_store(store,full=True)
            report['journal_integrity_passed']=True
        except Exception as exc:report.update(journal_integrity_passed=False,journal_error=repr(exc))
        atomic_json(out/'worker_summary.json',report)
        print(json.dumps({k:report.get(k) for k in ('completed','new_forwards','worker_elapsed_seconds','error','forward_gates_passed','new_projection_gate_passed','numerical_validation_passed')},indent=2),flush=True)

def parent(out,old,seed,partial):
    start=time.perf_counter();reg=validate(out)
    if any((out/n).exists() for n in ('invocation_started.json','invocation.json','campaign/ledger.json')):raise ValueError('Recovery scope already started; no retry.')
    for label in ('coarse','fine'):
        if sha(old/'inputs'/(label+'.json'))!=reg['external_files_sha256'][label+'_input']:raise ValueError('Original input missing/changed.')
    write(out/'invocation_started.json',dict(registration_sha256=sha(out/'registration.json'),prior_debit=reg['prior_debit'],worker_loss_reserve=70,export_loss_reserve=20,review_charge=10))
    env=dict(os.environ,PYTHONPATH=str(old/'frozen_source'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1');hard_stop=False
    with (out/'worker.log').open('w') as log:
        p=subprocess.Popen([sys.executable,'-B',str(out/Path(__file__).name),'--output',str(out),'--previous',str(old),'--seed',str(seed),'--partial',str(partial),'--worker'],cwd=old/'frozen_source',env=env,stdout=log,stderr=subprocess.STDOUT)
        while p.poll() is None:
            elapsed=time.perf_counter()-start;write(out/'parent_progress.json',dict(worker_wall_seconds=elapsed,worker_pid=p.pid))
            if elapsed>=70:hard_stop=True;p.kill();p.wait();break
            time.sleep(.5)
    wall=time.perf_counter()-start;ledger=out/'campaign/ledger.json';increment=sum(e['charged_seconds'] for e in json.loads(ledger.read_bytes())['entries']) if ledger.exists() else 0.
    result=dict(schema='g7-bounded-quad-recovery-invocation-v1',worker_wall_seconds=wall,ledger_increment=increment,native_charge=max(wall,increment),hard_stop=hard_stop,returncode=p.returncode,review_charge=10,preparation_charge=10,scientific_accept=False,full_day_budget_admitted=False)
    write(out/'invocation.json',result);before=time.perf_counter();archive=out.parent/(out.name+'_data.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file() and f.name!='writer.lock':z.write(f,str(f.relative_to(out)))
    result.update(export_seconds=time.perf_counter()-before)
    result['scope_charge']=result['native_charge']+result['export_seconds']+10
    result['additional_charge']=10+result['scope_charge'];result['campaign_debit']=reg['prior_debit']+result['scope_charge'];result['remaining_seconds']=1910-result['campaign_debit'];result['within_scope_ceiling']=result['scope_charge']<=100 and result['additional_charge']<=110 and result['export_seconds']<=20
    write(out/'invocation.json',result);write(out/'archive_index.json',dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,final_invocation_sha256=sha(out/'invocation.json')))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('output','previous','seed','partial'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--worker',action='store_true');a=p.parse_args()
    (worker if a.worker else parent)(a.output.resolve(),a.previous.resolve(),a.seed.resolve(),a.partial.resolve())
