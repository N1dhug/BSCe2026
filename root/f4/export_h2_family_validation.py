"""Export a completed family's original gates from cached jobs only."""
from time import perf_counter,time
BEGUN=perf_counter()-.25
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from root.f4 import run_h2_cap_extension as x
c=x.c;s=x.s


def review(directory,pieces):
    fit=c.load(directory/f'extension_optimizer_{pieces}_result.json')
    c.require(fit['converged'] is True,'Only completed optimizer families.')
    candidate=fit['candidate'];key=c.curve_key(candidate)
    manifest=c.load(directory/'manifest.json');obs=c._read_array_checkpoint((directory/'observations.npz').read_bytes())
    jobs={}
    for path in sorted((directory/'jobs').glob('*.npz')):
        raw=c._read_array_checkpoint(path.read_bytes());t=raw['token']
        if t['candidate']!=key or t['branch']!='prefinal' or t['hour']>=96:continue
        c.require(path.name==c.digest(t)+'.npz' and t['identity']==manifest['identity']
                  and raw['checksum']==c.fingerprint({k:v for k,v in raw.items() if k!='checksum'}),'Raw job identity/checksum failed.')
        j=t['grid'],t['hour'],t['kind'];c.require(j not in jobs,'Duplicate raw job.');jobs[j]=raw
    streams={}
    for grid in ('base','fine'):
        rows=[];last=None
        for hour in range(96):
            seen=hour>=24 and bool(obs['observed'][hour-24]);kind='observed' if seen else 'prediction'
            raw=jobs[grid,hour,kind];r=raw['result']
            c.state_check(raw['prior_log_state']);c.state_check(r['log_state'])
            if last is not None:c.same_state(raw['prior_log_state'],last)
            if seen:c.require(raw['token']['value']==float(obs['values'][hour-24]),'Different observation.')
            moment=None
            if hour>=72:
                m=jobs[grid,hour,'moments'];c.same_state(m['prior_log_state'],raw['prior_log_state']);moment=m['result']
            rows.append(dict(hour=hour,log_state=r['log_state'],observed=seen,
                             log_evidence=float(r['log_evidence']) if seen else 0.,moments=moment))
            last=r['log_state']
        streams[grid]=rows
    comparison=c.compare_streams(streams['base'],streams['fine'],{'train':(24,72),'selection':(72,96)})
    f,g=c.curve(candidate)
    return dict(schema='f4-h2-family-validation-raw-export-v1',status='passed',pieces=pieces,
        numerical_manifest_identity=manifest['identity'],optimizer=fit,comparison=comparison,
        metrics={grid:c.selection_metrics(rows,obs) for grid,rows in streams.items()},
        curves={'f':f.to_record(),'g':g.to_record()},raw_jobs=len(jobs),new_model_jobs=0,
        final_observations_decoded=0,scientific_acceptance=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--pieces',type=int,choices=(2,4),required=True)
    a=p.parse_args();campaign=c.OUT/'f4_h2_same_day_campaign_20261007.json'
    budget=c.load(ROOT/'root/f4/h2_same_day_budget_decision_20261007.json')
    with c._RunLock(campaign.with_suffix('.lock')),c._RunLock(a.directory):
        ledger=s.validate_campaign(c.load(campaign),budget)
        native=c.ordered_sum(e['charged_seconds'] for e in ledger['entries'] if e['label'].startswith('native-development|'))
        grant=min(15.,900.-native);c.require(grant>=6.,'Insufficient raw export allowance.')
        entry=dict(label='native-development|family-validation-raw-export',started_utc_seconds=time(),status='reserved',charged_seconds=grant)
        ledger['entries'].append(entry);ledger['charged_seconds']=c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries']);c.write(campaign,ledger,replace=True)
        timer=s.threading.Timer(max(.01,grant-(perf_counter()-BEGUN)-1.),lambda:os._exit(124));timer.daemon=True;timer.start()
        try:
            result=review(a.directory,a.pieces);result.update(elapsed_seconds=perf_counter()-BEGUN,source_sha256=c.sha(Path(__file__)))
            c.write(a.directory/f'family_{a.pieces}_validation.json',result);print(json.dumps(result))
        finally:
            entry.update(status='closed',charged_seconds=perf_counter()-BEGUN)
            ledger['charged_seconds']=c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries']);c.write(campaign,ledger,replace=True);timer.cancel()
