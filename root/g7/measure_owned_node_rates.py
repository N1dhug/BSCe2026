"""One bounded G7 own-node rate measurement; no optimizer, FV or EM replay.

Parent subprocess timeout preserves fsynced receipts after a hard stop.
Original solver partition, owned input bytes and supervisor M254 remain fixed.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        stream.write((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def worker(args):
    import numpy as np
    from src.control.bm_1 import BM1ControlNumerics
    from src.method.bm_1_g7 import OwnedBankPriceEvaluator
    from src.method.bm_1_g7_jobs import CheckpointedBankControl,NodeJobStore,NodeJobStop,atomic_json
    from src.method.bm_1_g8 import G8BankOrigin
    from src.method.bm_1_distribution_panel import baseline,target
    from src.exe.run_model.bm_1.g8_bernstein import SolverAudit
    from src.exe.run_model.bm_1.g7 import compare_forward

    out=Path(args.output)
    reg=json.loads((out/'registration.json').read_text())
    hashes=json.loads((out/'source_manifest.json').read_text())
    store=NodeJobStore(out/'campaign',prior_debit=reg['prior_debit'],
                       ceiling=reg['campaign_ceiling'],max_new_seconds=reg['worker_seconds'])
    report=dict(schema='g7-owned-node-runtime-worker-v1',banks={},
                scientific_accept=False,full_day_accept=False,actual_optimizer_started=False)
    evaluators={}
    started=time.perf_counter()

    class MeasuredEvaluator(OwnedBankPriceEvaluator):
        def evaluate_node(self,prices,index):
            before=time.perf_counter()
            result=super().evaluate_node(prices,index)
            result['measurement']=dict(forward_seconds=time.perf_counter()-before,
                                       projection_audit=self.audit.record())
            print('Completed native node',self.origin.conditional_mass.shape[1],index,
                  'seconds',round(result['measurement']['forward_seconds'],3),flush=True)
            return result

    try:
        for cells,label in [(513,'fine'),(257,'coarse')]:
            local={}
            def setup(deadline):
                origin=G8BankOrigin.from_record(json.loads((out/'inputs'/(label+'.json')).read_text()))
                if origin.time!=12. or len(origin.rho_values)!=5:
                    raise ValueError('Expected own hour12 five-node bank.')
                audits=[]
                def configure(projector):
                    audit=SolverAudit(projector.projection_solver,projector.dx)
                    audits.append(audit)
                    projector.projection_solver=audit
                evaluator=MeasuredEvaluator(origin,baseline=baseline,target=target,
                    horizon=tuple(reg['horizon']),price_edges=reg['price_edges'],
                    numerics=BM1ControlNumerics(x_steps=cells,
                        objective_max_step=reg['objective_max_step'],
                        solver_max_step=reg['solver_steps'][str(cells)]),
                    configure_projector=configure)
                evaluator.audit=audits[0]
                local.update(evaluator=evaluator)
                if time.perf_counter()>=deadline:
                    raise NodeJobStop('Setup exceeded its reservation.')
                return dict(kind='setup',cells=cells)
            with store.lock():
                store.node(dict(index=-1,kind='setup',cells=cells,registration_sha256=sha(out/'registration.json')),
                           reg['setup_reservation'],setup)
            # This invocation never replays setup or interrupted node attempts.
            if not local:
                raise NodeJobStop('Existing setup requires a separately registered resume launcher.')
            evaluator=local['evaluator'];evaluators[label]=evaluator
            local['control']=CheckpointedBankControl(evaluator,store,source_root=Path.cwd(),source_sha256=hashes)
            before=time.perf_counter()
            result=local['control'].evaluate(reg['prices'],
                       seconds_per_node=reg['per_node_reservation'][str(cells)])
            wall=time.perf_counter()-before
            atomic_json(out/(label+'_bank.json'),
                        {k:v.to_record() if k=='predicted_origin' else v
                         for k,v in result.items() if k!='_owner'})
            audit=evaluator.audit.record()
            valid=audit['maximum_projection_cell_mass_l1']<=reg['gates']['projection_l1']
            report['banks'][label]=dict(cells=cells,complete=True,bank_wall_seconds=wall,
                total=result['total'],bias=result['bias'],variance=result['variance'],
                within=result['within_parameter_variance'],between=result['between_parameter_variance'],
                projection_audit=audit,projection_gate_passed=valid,
                contract_id=local['control'].contract_id)
            write(out/'progress.json',report)
            print('Bank',label,'complete',round(wall,3),'s; campaign remaining',
                  round(reg['campaign_ceiling']-store.charged(),3),flush=True)
            if not valid:
                raise RuntimeError('Existing projection L1 gate failed; stop before another solve.')
        coarse=json.loads((out/'coarse_bank.json').read_text())
        fine=json.loads((out/'fine_bank.json').read_text())
        for record in (coarse,fine):
            record['predicted_origin']=G8BankOrigin.from_record(record['predicted_origin'])
        report['own_refinement']=compare_forward(coarse,fine,evaluators['fine'])
        report['objective_difference']=abs(coarse['total']-fine['total'])
        report['objective_tolerance']=max(.001,.02*abs(fine['total']))
        report['objective_gate_passed']=report['objective_difference']<=report['objective_tolerance']
        report['short_pair_passed']=bool(report['own_refinement']['passed'] and report['objective_gate_passed'])
        report['stop_reason']='Registered two-bank measurement completed; no optimizer permitted.'
    except Exception as exc:
        report['stop_reason']=repr(exc)
        report['short_pair_passed']=False
        report['error_type']=type(exc).__name__
    finally:
        report['elapsed_worker_body']=time.perf_counter()-started
        report['ledger_increment']=store.charged()-reg['prior_debit']
        report['partial_projection_audits']={k:e.audit.record() for k,e in evaluators.items()}
        write(out/'worker_summary.json',report)
        print('Stopped:',report.get('stop_reason'),'elapsed',round(report['elapsed_worker_body'],3),flush=True)
    return 0


def parent(args):
    out=Path(args.output)
    reg=json.loads((out/'registration.json').read_text())
    if (out/'invocation.json').exists() or (out/'campaign/ledger.json').exists():
        raise ValueError('One registered invocation only; no automatic retry or budget reset.')
    started=time.perf_counter()
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(out),'--worker']
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
             MKL_NUM_THREADS='1',PYTHONPATH=str(Path.cwd()))
    with (out/'worker.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        hard_stop=False
        try:
            process.wait(timeout=reg['worker_seconds'])
        except subprocess.TimeoutExpired:
            hard_stop=True
            process.kill()
            process.wait()
    worker_wall=time.perf_counter()-started
    ledger_path=out/'campaign/ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    increment=sum(e['charged_seconds'] for e in ledger['entries']) if ledger else 0.
    result=dict(schema='g7-owned-node-runtime-invocation-v1',worker_wall_seconds=worker_wall,
        worker_returncode=process.returncode,hard_stop=hard_stop,
        ledger_increment=increment,
        native_charge=max(worker_wall,increment),review_charge=reg['review_charge'],
        scientific_accept=False,actual_optimizer_started=False,
        note='No retries; running reservations retained after process kill.')
    write(out/'invocation.json',result)
    export_started=time.perf_counter()
    # Archive exact original input/source/registration and all partial/completed node bytes.
    raw={str(p.relative_to(out)):base64.b64encode(p.read_bytes()).decode()
         for p in out.rglob('*') if p.is_file() and p.name!='writer.lock'}
    encoded=json.dumps(dict(schema='g7-raw-byte-archive-v1',files=raw),
                       sort_keys=True,separators=(',',':')).encode()
    compressed=zlib.compress(encoded,6)
    payload=base64.b64encode(compressed).decode()
    destination=Path('root/g7/node_runtime_raw_20261007')
    destination.mkdir(exist_ok=True)
    chunks=[]
    for index,start in enumerate(range(0,len(payload),30000)):
        path=destination/('part%03d.txt'%index)
        path.write_text(payload[start:start+30000],encoding='ascii')
        chunks.append(dict(path=str(path),sha256=sha(path)))
    write(destination/'index.json',dict(schema='g7-zlib-base64-byte-archive-v1',
        encoding='concatenate parts, base64 decode, zlib decompress, JSON decode; each file is base64 original bytes',
        archive_sha256=hashlib.sha256(encoded).hexdigest(),compressed_sha256=hashlib.sha256(compressed).hexdigest(),
        raw_bytes=len(encoded),compressed_bytes=len(compressed),chunks=chunks,
        files={p:dict(sha256=hashlib.sha256(base64.b64decode(value)).hexdigest(),
                       bytes=len(base64.b64decode(value))) for p,value in raw.items()}))
    export_seconds=time.perf_counter()-export_started
    result.update(export_seconds=export_seconds,
        scope_charge=result['native_charge']+export_seconds+reg['review_charge'])
    result['campaign_debit']=reg['prior_debit']+result['scope_charge']
    result['remaining_seconds']=reg['campaign_ceiling']-result['campaign_debit']
    result['within_scope_ceiling']=result['scope_charge']<=reg['scope_ceiling']
    write(out/'invocation.json',result)
    write(Path('root/g7/node_runtime_invocation_20261007.json'),result)
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    sys.exit(worker(args) if args.worker else parent(args))
