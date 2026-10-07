"""One focused stdlib recovery check; real receipt verification, no forwards."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

from receipt_recovery import (RecoveryBlocked, VerifiedReceiptIndex, audit_store,
                              canonical, digest, guarded_store_class, sha)


def native_store(path):
    # Execute the exact reviewed native writer/lock code with only its scalar
    # isfinite dependency substituted. No solver/numpy/scipy imports occur.
    tree = ast.parse(path.read_text())
    names = {'plain', 'encode', 'digest', 'atomic_json', 'NodeJobStop', 'NodeJobStore'}
    imports = {'contextlib', 'hashlib', 'json', 'os', 'pathlib', 'time'}
    kept = [n for n in tree.body if
            (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
            or (isinstance(n, ast.Import) and all(a.name in imports for a in n.names))
            or (isinstance(n, ast.ImportFrom) and n.module in imports)]
    class FiniteResult(list):
        def all(self): return all(self)
    def isfinite(x):
        return FiniteResult(math.isfinite(v) for v in x) if isinstance(x, list) else math.isfinite(x)
    namespace = {'np': SimpleNamespace(isfinite=isfinite)}
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def check(args):
    start = time.perf_counter()
    reg = json.loads(args.registration.read_text())
    for name, expected in reg['source_sha256'].items():
        if sha(args.registration.parent / name) != expected:
            raise RecoveryBlocked('Registered check/interface source changed.')
    if sha(args.native_source) != reg['native_store_sha256']:
        raise RecoveryBlocked('Native writer source changed.')
    index = VerifiedReceiptIndex(args.raw_root, args.manifest,
                                  manifest_sha256=reg['manifest_sha256'])
    cases = []
    def rejects(name, action):
        try: action()
        except (ValueError, RuntimeError): cases.append(name); return
        raise AssertionError('Expected fail-closed rejection: ' + name)
    # Altered policy/identity never resolves to a cached original job.
    original = next(e['job'] for e in index.entries.values() if e['job']['kind'] == 'fv_fine_nodes')
    rejects('changed_policy_rejected', lambda: index.payload(dict(original, prices=[.6, .5])))
    rejects('wrong_manifest_hash_rejected', lambda: VerifiedReceiptIndex(
        args.raw_root, args.manifest, manifest_sha256='0'*64))
    adopted_fv = sum(e['job']['kind'] in ('fv_fine_nodes', 'fv_coarse_nodes') for e in index.entries.values())
    assert adopted_fv == 10 and len(index.entries) == 24
    ns = native_store(args.native_source)
    base = ns['NodeJobStore']; atomic = ns['atomic_json']
    guarded = guarded_store_class(base, native_source_path=args.native_source,
                                   native_source_sha256=reg['native_store_sha256'])
    computes = [0]
    with tempfile.TemporaryDirectory(prefix='g7-recovery-check-') as directory:
        root = Path(directory)
        def new(name):
            return guarded(root/name, prior_debit=reg['prior_debit'], ceiling=1800, max_new_seconds=10)
        def compute(deadline):
            computes[0] += 1
            return {'stub': 'no model', 'value': 7}
        def call(store, job, callback=compute, **kw):
            with store.lock(): return store.node(job, 1, callback, **kw)
        job = {'index': 0, 'kind': 'stub', 'contract': 'fixed'}
        s = new('normal'); first = call(s, job)
        old_charge = s.charged(); second = call(s, job)
        assert first == second and computes[0] == 1 and s.charged() == old_charge
        cases.append('completed_reuse_no_compute_no_recharge')
        assert len(audit_store(s, full=True)['entries']) == 1
        # Crash after fsynced receipt, before journal completion.
        l = s.read(); l['entries'][0].update(status='running', charged_seconds=1.)
        atomic(s.ledger_path, l)
        assert call(s, job) == first and computes[0] == 1
        assert s.read()['entries'][0]['status'] == 'recovered-complete'
        assert s.charged() == reg['prior_debit'] + 1
        cases.append('receipt_before_completion_crash_preserves_full_reserve')
        # A receipt with no reservation blocks another job BEFORE compute.
        s = new('orphan'); call(s, job); l = s.read(); l['entries'] = []; atomic(s.ledger_path, l)
        count = computes[0]
        rejects('orphan_blocks_unrelated_job_before_compute', lambda: call(s, dict(job,index=1)))
        assert computes[0] == count and s.blocked_path.exists()
        rejects('blocked_scope_cannot_auto_resume', lambda: new('orphan'))
        # Complete journal without its file cannot be silently replayed.
        s = new('missing'); call(s, job); (s.root/'nodes'/(digest(job)+'.json')).unlink()
        count = computes[0]
        rejects('missing_complete_receipt_blocks_before_compute', lambda: call(s, job))
        assert computes[0] == count
        # The inherited receipt hash check remains active on cache access.
        s = new('tamper'); call(s, job); p=s.root/'nodes'/(digest(job)+'.json')
        r=json.loads(p.read_text());r['payload']['value']=8;atomic(p,r)
        rejects('tampered_payload_rejected', lambda: call(s, job))
        # Interrupt without receipt: explicit retry retains the earlier reserve.
        s = new('retry'); key=digest(job);l=s.read()
        l['entries'].append(dict(job_id=key,node=0,attempt=1,status='running',reserved_seconds=1.,charged_seconds=1.))
        atomic(s.ledger_path,l);call(s,job,retry_ids=(key,))
        assert len(s.read()['entries'])==2 and s.read()['entries'][0]['charged_seconds']==1.
        cases.append('explicit_retry_preserves_interrupted_attempt')
        # Inject a completion writer that drops entries: readback detects it.
        s = new('write_loss'); original_finish=base._finish
        def lost_write(self, ledger, entry, seconds, status, **extra):
            original_finish(self,ledger,entry,seconds,status,**extra)
            changed=self.read();changed['entries']=[];atomic(self.ledger_path,changed)
        base._finish=lost_write
        try:
            rejects('completion_write_loss_detected',lambda:call(s,job))
        finally: base._finish=original_finish
        assert s.blocked_path.exists() and (s.root/'nodes'/(key+'.json')).exists()
        # Inject a dropped reservation before compute; no expensive work occurs.
        s = new('reservation_loss'); original_atomic=ns['atomic_json']
        def lost_reservation(path,value):
            if Path(path)==s.ledger_path and value['entries'] and value['entries'][-1]['status']=='running':
                value=json.loads(canonical(value));value['entries']=[]
            original_atomic(path,value)
        ns['atomic_json']=lost_reservation;count=computes[0]
        try: rejects('reservation_write_loss_blocks_compute',lambda:call(s,job))
        finally: ns['atomic_json']=original_atomic
        assert computes[0]==count
        # A changed disk journal must not be overwritten by a stale snapshot.
        s = new('stale');snapshot=s.read();changed=s.read()
        changed['seed']['max_new_seconds']=9.;atomic(s.ledger_path,changed)
        rejects('stale_finish_refuses_overwrite',lambda:s._finish(snapshot,{},0.,'complete'))
        assert s.read()==changed
    assert not any(name in sys.modules for name in ('numpy','scipy','src.model.bm_1'))
    assert sha(args.raw_root/'campaign/ledger.json')==index.manifest['source_files']['campaign/ledger.json']
    elapsed=time.perf_counter()-start
    assert elapsed <= reg['verification_reserve_seconds']
    return dict(schema='g7-receipt-recovery-check-v1',passed=True,cases=cases,
                verified_actual_receipts=24,reusable_actual_fv_receipts=adopted_fv,
                source_original_ledger_unchanged=True,source_journal_integrity_passed=False,
                explicit_read_only_adoption_verified=True,original_store_resume_permitted=False,
                actual_model_forwards=0,numerical_imports=0,stub_computes=computes[0],
                check_seconds=elapsed,scientific_accept=False,full_day_budget_admitted=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('registration','manifest','raw-root','native-source','report'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=check(a)
    a.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
