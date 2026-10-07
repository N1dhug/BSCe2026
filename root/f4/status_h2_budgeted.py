"""Read-only weighted current-candidate ETA; preserve the frozen running source."""
import argparse
import json
from pathlib import Path
import sys
from time import time,perf_counter
import zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def status(directory,campaign):
    from src.exe.run_model.progress import runtime_text
    directory=Path(directory)
    ledger=json.loads(Path(campaign).read_text());entry=ledger['entries'][-1]
    outer=json.loads((directory/'budgeted_execution.json').read_text());rates=outer['rates']['base']
    report=dict(ledger_status=entry['status'],invocation_elapsed_seconds=max(0.,time()-entry.get('started_utc_seconds',time())))
    registry=json.loads((directory/'candidate_registry.json').read_text())
    calls={};pending=[]
    for family in ('2','4'):
        path=directory/f'optimizer_{family}_calls.json'
        rows=json.loads(path.read_text())['calls'] if path.exists() else []
        calls[family]=len(rows);completed={r['key'] for r in rows}
        pending.extend((family,key) for key in registry['families'][family] if key not in completed)
    report['completed_objective_calls']=calls
    if pending:
        family,key=pending[-1];done=0.;initial=0.;fresh=[];hours=0
        for path in (directory/'jobs').glob('*.npz'):
            with zipfile.ZipFile(path) as z:r=json.loads(z.read('metadata.json'))['report']
            token=r['token']
            if token['candidate']!=key or token['grid']!='base' or token['branch']!='prefinal' or token['hour']>=72 or token['kind'] not in ('prediction','observed'):continue
            weight=rates[token['kind']];done+=weight;hours+=1
            if path.stat().st_mtime<=entry.get('started_utc_seconds',0.):initial+=weight
            else:fresh.append(path.stat().st_mtime-r['elapsed_seconds'])
        total=24*rates['prediction']+48*rates['observed']
        elapsed=max(0.,time()-min(fresh)) if fresh else 0.
        report.update(active_family=int(family),curve=key,completed_hours=hours,total_hours=72,
            current_candidate_weighted_eta=runtime_text(perf_counter()-elapsed,done,total,initial_completed=initial),
            eta_scope='Current candidate only; weights distinguish prediction and observed work. Not a convergence prediction.')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--campaign',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(status(args.directory,args.campaign)),flush=True)
