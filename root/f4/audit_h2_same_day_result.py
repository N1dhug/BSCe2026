"""Reuse the original raw optimizer auditor for an authorized day's report."""
from time import perf_counter
BEGUN=perf_counter()-.25
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'


def audit(directory, report_name):
    from root.f4 import run_h2_same_day as s
    from root.f4.audit_h2_budgeted_result import audit as original_audit
    c=s.c;directory=Path(directory)
    outer=c.load(directory/'same_day_execution.json')
    decision=c.load(ROOT/'root/f4/h2_same_day_budget_decision_20261007.json')
    c.require(outer['identity']==c.digest({k:v for k,v in outer.items() if k!='identity'})
              and outer['decision_identity']==decision['identity'],'Authorized identity differs.')
    c.require(outer['source_sha256']=={p:c.sha(ROOT/p) for p in s.SOURCES},'Authorized source differs.')
    c.require(c.environment()==c.load(directory/'manifest.json')['environment'],'Raw review environment differs.')
    result=original_audit(directory,report_name=report_name)
    result.update(authorized_execution_identity=outer['identity'],decision_identity=decision['identity'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--report',required=True)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from root.f4 import run_h2_same_day as s
    c=s.c;decision=c.load(ROOT/'root/f4/h2_same_day_budget_decision_20261007.json')
    c.require(not args.output.exists(),'Preserve every audit receipt.')
    with c._RunLock(args.campaign.with_suffix('.lock')),c._RunLock(args.directory):
        ledger=s.validate_campaign(c.load(args.campaign),decision)
        native=c.ordered_sum(e['charged_seconds'] for e in ledger['entries'] if e['label'].startswith('native-development|'))
        grant=min(40.,900.-native,ledger['ceiling_seconds']-ledger['charged_seconds'])
        c.require(grant>=15.,'Insufficient raw-audit budget; never reset development charges.')
        entry=dict(label='native-development|same-day-raw-result-audit',started_utc_seconds=s.time(),status='reserved',charged_seconds=grant)
        ledger['entries'].append(entry)
        ledger['charged_seconds']=c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
        c.write(args.campaign,ledger,replace=True)
        timer=s.threading.Timer(max(.01,grant-(perf_counter()-BEGUN)-1.),lambda:os._exit(124))
        timer.daemon=True;timer.start()
        try:
            result=audit(args.directory,args.report)
            result['elapsed_seconds']=perf_counter()-BEGUN
            result['adapter_sha256']=c.sha(Path(__file__))
            c.write(args.output,result)
        finally:
            entry.update(status='closed',charged_seconds=perf_counter()-BEGUN)
            ledger['charged_seconds']=c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
            c.write(args.campaign,ledger,replace=True);timer.cancel()
    print(json.dumps(result),flush=True)
