"""Bounded separate continuation; requires an explicit new deadline approval."""
from time import perf_counter,time
BEGUN=perf_counter()
import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
import re
import sys
import threading
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
from src.exe.run_model.bm_1_h2_continuation import c,Engine,prefinal,verify_origin

def authorized_decision(path,directory):
    d=c.load(path)
    c.require(d['approved'] is True and isinstance(d.get('user_approval'),str) and bool(d['user_approval'].strip()),
              'New numerical deadline has not been approved.')
    c.require(d['identity']==c.digest({k:v for k,v in d.items() if k!='identity'}),'Decision identity changed.')
    c.require(d['ceiling_seconds']==64800 and d['native_development_ceiling_seconds']==1080
        and d['invocation_cap_seconds']==600 and d['stop_export_reserve_seconds']==60,
        'Existing ceilings or stop reserve changed.')
    c.require(d['deadline_utc_seconds']>time()+75,'Approved continuation deadline expired.')
    for grid in ('base','fine'):
        c.require(set(d['protected_rates'][grid])=={'observed','prediction','moments','cdf'}
            and all(math.isfinite(v) and v>0 for v in d['protected_rates'][grid].values()),
            'Invalid registered protected rates.')
    c.require(d['offline_control_passed'] is True,'Continuation control must pass before admission.')
    c.require(c.sha(directory/'offline_control.json')==d['offline_control_sha256'],
              'Registered offline control changed.')
    c.require(d['manifest_identity']==c.load(directory/'manifest.json')['identity'],'Decision belongs to another run.')
    return d

def validate_campaign(ledger,decision):
    old_path=ROOT/decision['predecessor_campaign_path'];old=c.load(old_path)
    c.require(c.sha(old_path)==decision['predecessor_campaign_sha256'],'Old campaign changed; do not refund/reconcile.')
    prep_path=ROOT/decision['preparation_ledger_path'];prep=c.load(prep_path)
    c.require(c.sha(prep_path)==decision['preparation_ledger_sha256'],'Preparation charges changed.')
    prefix=old['entries']+prep['entries']
    c.require(ledger['schema']=='f4-h2-contour-continuation-campaign-v3'
        and ledger['decision_identity']==decision['identity'] and ledger['entries'][:len(prefix)]==prefix,
        'Lost predecessor/preparation charges.')
    c.require(ledger['ceiling_seconds']==64800 and ledger['deadline_utc_seconds']==decision['deadline_utc_seconds'],
              'Budget/deadline differs from approval.')
    c.require(all(e['status'] in ('closed','reserved') and math.isfinite(e['charged_seconds']) and e['charged_seconds']>=0
        for e in ledger['entries']),'Invalid campaign charges.')
    c.require(ledger['charged_seconds']==c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
        and ledger['charged_seconds']<=64800,'Campaign total changed or exhausted.')
    c.require(c.ordered_sum(e['charged_seconds'] for e in ledger['entries'] if e['label'].startswith('native-development|'))<=1080,
              'Native development allowance exceeded.')
    c.require(ledger['validation_reserve_seconds']>=old['validation_reserve_seconds'],'Stored validation reserve reduced.')
    return ledger

@contextmanager
def reservation(path,decision,label):
    with c._RunLock(path.with_suffix('.lock')):
        if path.exists():ledger=validate_campaign(c.load(path),decision)
        else:
            old=c.load(ROOT/decision['predecessor_campaign_path']);prep=c.load(ROOT/decision['preparation_ledger_path'])
            ledger=dict(old,schema='f4-h2-contour-continuation-campaign-v3',
                entries=old['entries']+prep['entries'],deadline_utc_seconds=decision['deadline_utc_seconds'],decision_identity=decision['identity'])
            ledger['charged_seconds']=c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
            validate_campaign(ledger,decision);c.write(path,ledger)
        before=ledger['charged_seconds'];startup=perf_counter()-BEGUN
        grant=min(600.,64800-before,decision['deadline_utc_seconds']-time()+startup)
        c.require(grant>75,'Insufficient budget/time for a bounded block.')
        entry=dict(label='compact-production|contour-continuation-'+label,status='reserved',charged_seconds=grant,started_utc_seconds=time())
        ledger['entries'].append(entry);ledger['charged_seconds']=c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
        c.write(path,ledger,replace=True)
        watchdog=threading.Timer(max(.01,grant-(perf_counter()-BEGUN)-1.),lambda:os._exit(124))
        watchdog.daemon=True;watchdog.start()
        try:
            yield dict(ledger=ledger,before=before,begun=BEGUN,grant=grant,protected_rates=decision['protected_rates'],
                deadline=min(time()+grant-(perf_counter()-BEGUN)-60.,decision['deadline_utc_seconds']-60.))
        finally:
            entry.update(status='closed',charged_seconds=perf_counter()-BEGUN)
            ledger['charged_seconds']=c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
            c.write(path,ledger,replace=True);watchdog.cancel()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--decision',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--stage',choices=('prefinal','export-final','final'),default='prefinal')
    p.add_argument('--workbook',type=Path)
    a=p.parse_args();directory=a.run.resolve()
    d=authorized_decision(a.decision,directory)  # Reject before reservation/model setup.
    c.require(re.fullmatch('[0-9a-f]{40}',a.source_commit) is not None,'Immutable controller commit required.')
    manifest,origin,old=verify_origin(directory)
    c.require(not any((directory/n).exists() for n in ('STOP','FAILED.json','CONTINUATION_FAILED.json','PREPARATION_FAILED.json')),
              'Preserved stop/failure; no automatic retry.')
    engine=None;number=len(list(directory.glob('continuation_report_*.json')))+1
    with reservation(a.campaign,d,a.stage) as allocation,c._RunLock(old),c._RunLock(directory):
        try:
            execution=dict(schema='f4-h2-continuation-execution-v1',source_commit=a.source_commit,
                manifest_identity=manifest['identity'],decision_identity=d['identity'],automatic_retry=False)
            path=directory/'continuation_execution.json'
            if path.exists():c.require(c.load(path)==execution,'Controller/decision changed mid-run.')
            else:c.write(path,execution)
            engine=Engine(directory,c.receipt(),manifest,allocation)
            if a.stage=='prefinal':result=prefinal(engine)
            elif a.stage=='export-final':
                c.require(a.workbook is not None,'Original workbook is required after forecast freeze.')
                c.load_final_observations(engine,a.workbook);result=dict(status='final_data_exported')
            else:
                c.require((directory/'final_observations.npz').exists(),'Final data must be exported after forecast freeze.')
                c.final_evaluation(engine,None);result=dict(status='final_completed')
            verify_origin(directory)
        except (Exception,KeyboardInterrupt) as exc:
            paused=isinstance(exc,(TimeoutError,KeyboardInterrupt))
            result=dict(status='paused' if paused else 'failed',reason=str(exc),traceback=traceback.format_exc(),automatic_retry=False)
            if not paused:c.write(directory/'CONTINUATION_FAILED.json',result)
        finally:
            if engine is not None:engine.close()
        result.update(native_jobs=0 if engine is None else len(engine.store.events),
            adopted_jobs=0 if engine is None else len(engine.store.adopted),scientific_acceptance=False,blind_h3_acceptance=False)
        c.write(directory/f'continuation_report_{number:03d}.json',result)
    print(c.json.dumps(result),flush=True)
    if result['status']=='failed':raise SystemExit(2)

if __name__=='__main__':main()
