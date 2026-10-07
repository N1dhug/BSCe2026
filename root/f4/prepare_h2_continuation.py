"""Prepare a separate continuation from audited raw jobs; zero model jobs."""
from time import perf_counter,time
BEGUN = perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predecessor',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();directory=a.run.resolve();old=a.predecessor.resolve()
    if directory.exists():raise FileExistsError('Preserve prepared/failed directory; no automatic repeat.')
    prior_seconds=425.8919726690074
    def ordered(values):
        total=0.
        for v in values:total+=v
        return total
    previous=json.loads(a.campaign.read_text())
    assert previous['charged_seconds']==prior_seconds+ordered(e['charged_seconds'] for e in previous['entries'])
    assert ordered(e['charged_seconds'] for e in previous['entries'] if e['label'].startswith('native-development|'))+20<=1080
    directory.mkdir(parents=True)
    entry=dict(label='native-development|continuation-read-only-preparation',status='reserved',charged_seconds=20.,started_utc_seconds=time())
    ledger=dict(schema='f4-h2-continuation-preparation-ledger-v1',predecessor_campaign_path=str(a.campaign.resolve().relative_to(ROOT)),
        predecessor_campaign_sha256=hashlib.sha256(a.campaign.read_bytes()).hexdigest(),entries=[entry],
        charged_seconds=20.,model_execution_approved=False)
    ledger_path=directory/'preparation_ledger.json'
    ledger_path.write_text(json.dumps(ledger)+'\n')
    timer=threading.Timer(max(.01,19.5-(perf_counter()-BEGUN)),lambda:os._exit(124));timer.daemon=True;timer.start()
    try:
        from src.exe.run_model.bm_1_h2_continuation import c,SOURCES,EXTENDED_CENTERS
        with c._RunLock(old),c._RunLock(directory):
            old_manifest=c.load(old/'manifest.json')
            c.require(c.source_hashes()==old_manifest['source_sha256'],'Original numerical sources changed.')
            data=c.receipt()
            c.require(c.environment()==old_manifest['environment'],'Original environment changed.')
            c.require(data['packet']['identity']==old_manifest['packet_identity']
                and data['observation_sha256']==old_manifest['observation_sha256'],'Original frozen data changed.')
            audit=c.load(old/'extension_block12_raw_audit.json')
            c.require(audit['status']=='passed' and audit['raw_jobs']==3655 and len(audit['failed_jobs'])==1,
                      'Require the reviewed terminal predecessor.')
            evidence=old.parent/'f4_h2_bank_fine_control_20261007'
            fine=c.load(evidence/'report.json');review=c.load(evidence/'raw_review.json')
            c.require(fine['status']==review['status']=='passed' and review['own_fine_prior_hours']==24
                and fine['numerical_sources_unchanged'] and review['native_propagations']==0,
                'Require the own-prior fine control and raw review.')
            index={}
            for path in sorted((old/'jobs').glob('*.npz')):
                raw=c._read_array_checkpoint(path.read_bytes());token=raw['token'];key=c.digest(token)
                c.require(path.name==key+'.npz' and token['identity']==old_manifest['identity']
                    and raw['checksum']==c.fingerprint({k:v for k,v in raw.items() if k!='checksum'}),
                    'Original raw job checksum/token failed.')
                c.state_check(raw['prior_log_state']);c.state_check(raw['result']['log_state'])
                index[key]=c.sha(path)
            c.require(len(index)==3655,'Incomplete predecessor raw-job chain.')
            c.write(directory/'predecessor_job_index.json',index)
            names=('manifest.json','packet.json','inputs.npz','observations.npz','EXTENSION_FAILED.json',
                'extension_execution.json','same_day_execution.json','budgeted_execution.json',
                'extension_block12_raw_audit.json','extension_optimizer_2_calls.json','extension_optimizer_2_result.json',
                'extension_optimizer_4_calls.json','extension_candidate_registry.json','extension_report_012.json')
            metadata={name:c.sha(old/name) for name in names}
            metadata.update({str(p.relative_to(old)):c.sha(p) for p in sorted((old/'jobs').glob('*.failed.json'))})
            origin=dict(schema='f4-h2-continuation-origin-v1',predecessor_path=str(old.relative_to(ROOT)),
                predecessor_identity=old_manifest['identity'],metadata_sha256=metadata,
                own_fine_evidence_sha256={name:c.sha(evidence/name) for name in ('report.json','raw_review.json','registration.json')},
                predecessor_failure_preserved=True,old_optimizer_calls={'2':20,'4':27},raw_jobs=3655,
                exact_old_prefix_unchanged=True,native_propagations=0)
            c.write(directory/'origin.json',origin)
            manifest=dict(old_manifest,schema='f4-h2-contour-continuation-v1',
                predecessor_identity=old_manifest['identity'],origin_sha256=c.sha(directory/'origin.json'),
                job_index_sha256=c.sha(directory/'predecessor_job_index.json'),centers=list(EXTENDED_CENTERS),
                continuation_source_sha256={path:c.sha(ROOT/path) for path in SOURCES},
                family_new_pair_caps={'2':24,'4':48},global_pair_cap=73,
                old_job_identities_preserved=True,model_execution_approved=False,
                preparation_source_parent='de70a1d6dde63ecdc6737eaa013913b6a57fac9d')
            manifest.pop('identity');manifest['identity']=c.digest(manifest)
            c.write(directory/'manifest.json',manifest)
            for name,value in (('packet.json',data['packet']),('inputs.npz',data['inputs']),('observations.npz',data['observations'])):
                c.write(directory/name,value)
            c.require(c.source_hashes()==old_manifest['source_sha256'],'Source changed during preparation.')
            report=dict(schema='f4-h2-continuation-prepared-v1',status='prepared',raw_jobs=len(index),
                source_files=len(old_manifest['source_sha256']),manifest_identity=manifest['identity'],
                native_propagations=0,model_execution_approved=False,old_failure_preserved=True,
                elapsed_seconds=perf_counter()-BEGUN)
            c.write(directory/'preparation_report.json',report);print(json.dumps(report),flush=True)
    except BaseException as exc:
        (directory/'PREPARATION_FAILED.json').write_text(json.dumps(dict(status='failed',reason=str(exc),automatic_retry=False))+'\n')
        raise
    finally:
        entry.update(status='closed',charged_seconds=perf_counter()-BEGUN)
        ledger['charged_seconds']=ordered(e['charged_seconds'] for e in ledger['entries'])
        temporary=ledger_path.with_suffix('.partial');temporary.write_text(json.dumps(ledger)+'\n');os.replace(temporary,ledger_path)
        timer.cancel()

if __name__=='__main__':main()
