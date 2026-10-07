"""One timed read-only check of optimized authentication; no model imports."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time


def main(old,seed):
    started=time.perf_counter();root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root))
    import receipt_recovery as recovery
    from restore_auxiliary import read_owned
    from launch_bounded_price import validate,write,select_candidate
    reg,plan,caps=validate(root)
    def stop(*unused):raise TimeoutError('Read-only preparation cap exhausted.')
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,5.8)
    checks=[]
    try:
        audit=recovery.checked_json(root/'previous_audit.json',reg['previous_audit_sha256'])
        owned=read_owned(old,audit,plan['reuse'])
        checks.extend(['41_source_hashes','original_ledger_and_inputs','10_M254_file_payload_job_hashes'])
        index=recovery.VerifiedReceiptIndex(seed,root/'adoption_manifest.json',
            manifest_sha256=reg['adoption_manifest_sha256'],eager_payload_validation=False)
        checks.extend(['24_complete_receipt_file_hashes','4_original_metadata_hashes'])
        rows=sorted((e for e in index.entries.values() if e['job']['kind']=='fv_fine_nodes'),
                    key=lambda e:e['job']['index'])
        if [e['job']['index'] for e in rows]!=list(range(5)):raise AssertionError('FV nodes incomplete')
        for entry in rows:index.payload(entry['job']);checks.append('FV_full_job_and_payload_'+str(entry['job']['index']))
        original_sha=recovery.sha
        def corrupted(path):
            return '0'*64 if '/campaign/nodes/' in str(path) else original_sha(path)
        recovery.sha=corrupted
        try:
            recovery.VerifiedReceiptIndex(seed,root/'adoption_manifest.json',
                manifest_sha256=reg['adoption_manifest_sha256'],eager_payload_validation=False)
        except recovery.RecoveryBlocked:checks.append('changed_receipt_bytes_rejected')
        else:raise AssertionError('Changed bytes accepted')
        finally:recovery.sha=original_sha
        if select_candidate([1,1,1])!=0:raise AssertionError('Tie order changed')
        checks.append('tie_rule_retained')
        if sum(j['count']*j['reservation_each_seconds'] for j in plan['proposal']['jobs'])!=263:
            raise AssertionError('Addendum job sum differs')
        checks.append('263_job_264_worker_292_scope_admitted')
        if any(n=='numpy' or n=='scipy' or n.startswith('src.') for n in sys.modules):
            raise AssertionError('Numerical imports occurred')
        checks.append('zero_numerical_imports')
        result=dict(schema='g7-addendum-reuse-check-v1',passed=True,checks=checks,
            seconds=time.perf_counter()-started,new_forwards=0,numerical_imports=0,
            numerical_or_scientific_accept=False)
    except Exception as exc:
        result=dict(schema='g7-addendum-reuse-check-v1',passed=False,error=repr(exc),
                    checks=checks,seconds=time.perf_counter()-started,new_forwards=0)
    finally:signal.setitimer(signal.ITIMER_REAL,0.)
    write(root/'addendum_reuse_check_20261007.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--seed',type=Path,required=True);args=parser.parse_args()
    main(args.previous.resolve(),args.seed.resolve())
