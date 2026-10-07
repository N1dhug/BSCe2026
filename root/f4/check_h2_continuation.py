"""Minimal identity/prior/failure/optimizer controls; no model propagation."""
from time import perf_counter,time
BEGUN=perf_counter()
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    directory=a.run.resolve()
    if (directory/'offline_control.json').exists():
        raise FileExistsError('Control evidence already exists; do not repeat it automatically.')
    ledger_path=directory/'preparation_ledger.json';ledger=json.loads(ledger_path.read_text())
    entry=dict(label='native-development|continuation-scoped-offline-control',status='reserved',charged_seconds=10.,started_utc_seconds=time())
    ledger['entries'].append(entry);ledger['charged_seconds']=sum(e['charged_seconds'] for e in ledger['entries'])
    ledger_path.write_text(json.dumps(ledger)+'\n')
    try:
        from src.exe.run_model import bm_1_h2_continuation as x
        c=x.c;manifest,origin,old=x.verify_origin(directory)
        index=c.load(directory/'predecessor_job_index.json');passed=[]
        def rejects(fn):
            try:fn()
            except ValueError:return
            raise AssertionError('Expected identity/prior/failure rejection.')
        key='be1ef45a774a1c7999e31ec7950fb4361cfd82454ac661cc4864cca21301453b'
        saved=c._read_array_checkpoint((old/'jobs'/f'{key}.npz').read_bytes())
        meta={k:v for k,v in saved['token'].items() if k!='identity'};prior=saved['prior_log_state']
        def forbidden():raise AssertionError('A control attempted model propagation.')
        with tempfile.TemporaryDirectory() as temporary:
            store=x.AdoptingJobStore(Path(temporary),manifest['identity'],old,origin['predecessor_identity'],index)
            result=store.run(meta,prior,forbidden)
            assert c.fingerprint(result)==c.fingerprint(saved['result']) and len(store.adopted)==1 and not store.events
            assert next(iter(store.completed.values()))['token']['identity']==origin['predecessor_identity']
            passed.append('old-result-and-token-preserved')
            altered=prior[::-1].copy();assert not c.np.array_equal(altered,prior)
            rejects(lambda:store.run(meta,altered,forbidden));passed.append('different-exact-prior-rejected')
            poisoned=dict(index);poisoned[key]='0'*64
            bad=x.AdoptingJobStore(Path(temporary),manifest['identity'],old,origin['predecessor_identity'],poisoned)
            rejects(lambda:bad.run(meta,prior,forbidden));passed.append('changed-raw-hash-rejected')
            newkey=c.digest(dict(identity=manifest['identity'],**meta))
            c.write(Path(temporary)/'jobs'/f'{newkey}.failed.json',dict(reason='control-only-terminal-marker'))
            rejects(lambda:store.run(meta,prior,forbidden));passed.append('new-failure-blocks-old-adoption')
        pending=c.load(old/'extension_block12_raw_audit.json')['optimizer']['4']['pending_candidate']
        failure=c.load(old/'jobs/6910062eb2676f4470c7b543ec9e30c88528773ba149c4e4bb5503476534819d.failed.json')
        with tempfile.TemporaryDirectory() as temporary:
            store=x.AdoptingJobStore(Path(temporary),manifest['identity'],old,origin['predecessor_identity'],index)
            meta={k:v for k,v in failure['token'].items() if k!='identity'}
            oldprior=c._read_array_checkpoint((old/'jobs'/f'{key}.npz').read_bytes())['result']['log_state']
            # Surrogate payload tests persistence only, never a numerical update.
            value=store.run(meta,oldprior,lambda:dict(log_state=oldprior.copy(),control_surrogate=True))
            assert value['control_surrogate'] and len(store.events)==1
            newkey=c.digest(dict(identity=manifest['identity'],**meta))
            raw=c._read_array_checkpoint((Path(temporary)/'jobs'/f'{newkey}.npz').read_bytes())
            assert raw['token']['identity']==manifest['identity'] and (old/'jobs/6910062eb2676f4470c7b543ec9e30c88528773ba149c4e4bb5503476534819d.failed.json').exists()
            passed.append('old-failure-preserved-new-job-namespace')
        data=c.receipt()
        engine=object.__new__(x.Engine);engine.data=data
        for grid in ('base','fine'):
            spec=x.Engine.spec(engine,grid)
            assert spec['settings']['centers'][:5]==list(x.ORIGINAL_CENTERS)
            assert spec['settings']['centers']==list(x.EXTENDED_CENTERS)
        passed.append('unchanged-five-center-prefix')
        class EndOfTrace(Exception):pass
        class CachedEngine:
            def __init__(self,d):
                self.directory=Path(d);self.predecessor=old;self.origin=origin;self.manifest=manifest
            def check(self,**kwargs):raise EndOfTrace()
            def replay(self,*args,**kwargs):raise AssertionError('Optimizer control attempted model replay.')
        with tempfile.TemporaryDirectory() as temporary:
            try:x.optimizer_family(CachedEngine(temporary),4)
            except EndOfTrace:passed.append('exact-27-call-optimizer-replay')
            else:raise AssertionError('Expected the preserved optimizer to request another call.')
        from root.f4.run_h2_continuation import authorized_decision,validate_campaign
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/'decision.json';c.write(target,dict(approved=False))
            rejects(lambda:authorized_decision(target,directory));passed.append('unapproved-deadline-rejected-before-reservation')
        assert perf_counter()-BEGUN<10
        report=dict(schema='f4-h2-continuation-offline-control-v1',status='passed',checks=passed,
            native_propagations=0,model_replays=0,old_failure_preserved=True,
            control_payloads_surrogates_in_temporary_directories=True,production_jobs_created=0,
            source_sha256=c.sha(Path(__file__)),elapsed_seconds=perf_counter()-BEGUN)
        c.write(directory/'offline_control.json',report);print(json.dumps(report),flush=True)
    finally:
        entry.update(status='closed',charged_seconds=perf_counter()-BEGUN)
        total=0.
        for e in ledger['entries']:total+=e['charged_seconds']
        ledger['charged_seconds']=total
        ledger_path.write_text(json.dumps(ledger)+'\n')

if __name__=='__main__':main()
