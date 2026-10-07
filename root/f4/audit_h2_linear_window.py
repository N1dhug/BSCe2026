"""Read-only complete-window audit and updated job pricing; no model replay."""
from time import perf_counter
BEGUN=perf_counter()-.25
import argparse
import json
import os
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'


def audit(directory,charged):
    from src.exe.run_model import bm_1_h2_compact as c
    from root.f4.measure_h2_compact_rates import work_graph
    import numpy as np
    directory=Path(directory); m=c.load(directory/'manifest.json')
    c.require(m['identity']==c.digest({k:v for k,v in m.items() if k!='identity'}),'Manifest digest changed.')
    c.require(m['source_sha256']==c.source_hashes(),'Raw audit needs the original executed source.')
    with zipfile.ZipFile(directory/'source.zip') as z:
        c.require(set(z.namelist())==set(m['source_sha256']),'Source archive members differ.')
        for name,h in m['source_sha256'].items():
            c.require(c.hashlib.sha256(z.read(name)).hexdigest()==h,'Source archive bytes differ.')
    data=c.receipt(); jobs={}; source_job_hashes={}
    rates={g:dict(v) for g,v in c.load(directory/'nonlinear_probe.json')['rates'].items()}
    for path in sorted((directory/'jobs').glob('*.npz')):
        r=c._read_array_checkpoint(path.read_bytes()); t=r['token']
        c.require(t['identity']==m['identity'] and path.name==c.digest(t)+'.npz'
            and r['checksum']==c.fingerprint({k:v for k,v in r.items() if k!='checksum'}),'Raw job identity/checksum differs.')
        if t['candidate']!='linear': continue
        c.require(t['branch']=='prefinal','Unexpected branch in linear reference.')
        key=t['grid'],t['hour'],t['kind']; c.require(key not in jobs,'Duplicate native job.')
        jobs[key]=r; source_job_hashes[path.name]=c.sha(path)
        rates[t['grid']][t['kind']]=max(rates[t['grid']][t['kind']],r['elapsed_seconds'])
    expected={(g,h,'observed') for g in ('base','fine') for h in range(27,96)}
    expected|={(g,h,'moments') for g in ('base','fine') for h in range(72,96)}
    c.require(set(jobs)==expected,'Incomplete or unexpected mandatory linear jobs.')
    streams={}; maxima=dict(mass=0.,conditional=0.,overlap=0.)
    for grid in ('base','fine'):
        logs=data['jobs'][(grid,0,'prediction')]['prior_log_state']; rows=[]
        for hour in range(96):
            kind='prediction' if hour<24 else 'observed'; key=grid,hour,kind
            if key in jobs:
                saved=jobs[key]; r=saved['result']; prior=saved['prior_log_state']
                c.require(saved['token']['value']==float(data['observations']['values'][hour-24]),'Different observation.')
            else:
                raw=data['jobs'][key]; prior=raw['prior_log_state']; r=dict(raw)
                r['log_state']=raw['posterior_log_state'] if kind=='observed' else raw['log_state']
            c.same_state(prior,logs); c.state_check(r['log_state'])
            maxima['mass']=max(maxima['mass'],r['maximum_abs_mass_error'])
            if hour>=24:
                maxima['conditional']=max(maxima['conditional'],r['maximum_relative_conditional_guard'])
                maxima['overlap']=max(maxima['overlap'],r['maximum_overlap_log_difference'])
            moments=None
            if hour>=72:
                moment=jobs[(grid,hour,'moments')]; c.same_state(moment['prior_log_state'],logs)
                moments=moment['result']; c.state_check(moments['log_state'])
                w=np.exp(moments['log_state']); mean=float(w@moments['conditional_mean'])
                variance=float(w@(moments['conditional_variance']+(moments['conditional_mean']-mean)**2))
                c.require(abs(mean-moments['interval_mean'])<=1e-13 and abs(variance-moments['interval_variance'])<=1e-13,'Predictive moment algebra differs.')
            logs=r['log_state']; rows.append(dict(hour=hour,log_state=logs,observed=hour>=24,
                log_evidence=r['log_evidence'] if hour>=24 else 0.,moments=moments))
        streams[grid]=rows
    c.require(maxima['mass']<=1e-10 and maxima['conditional']<=1e-8 and maxima['overlap']<=1e-7,'Original numerical guard failed.')
    compared=c.compare_streams(streams['base'],streams['fine'],{'train':(24,72),'selection':(72,96)})
    c.require(compared==c.load(directory/'linear_full_window.json')['comparison'],'Whole-window summary differs from raw states.')
    full=work_graph(rates,48,24,charged)
    # Remove only the mandatory linear work already completed and verified.
    search_credit=1.5*(48*rates['base']['observed']+24*rates['base']['prediction'])
    validation_credit=1.5*(24*rates['base']['observed']+72*rates['fine']['observed']
        +24*rates['fine']['prediction']+24*(rates['base']['moments']+rates['fine']['moments']))
    remaining_validation=full['protected_validation_seconds']-validation_credit
    remaining49_total=full['projected_total_seconds']-search_credit-validation_credit
    per_hour_train=sum(jobs[('base',h,'observed')]['elapsed_seconds'] for h in range(27,72))+3*data['rates']['base']['observed']
    return dict(schema='f4-h2-whole-linear-raw-audit-v1',status='passed',source_commit=m['source_commit'],
        identity=m['identity'],reused_PC_prefix_hours=27,new_jobs=len(jobs),total_stream_hours=96,
        source_job_sha256=source_job_hashes,comparison=compared,guards=maxima,
        rates=rates,charged_at_pricing=charged,remaining49_global_max_total_seconds=remaining49_total,
        remaining49_global_max_within10h=remaining49_total<=36000.,
        protected_remaining_validation_seconds=remaining_validation,
        maximum_time_fundable_for_search=36000.-charged-remaining_validation-60.,
        summed_linear_train_job_seconds=per_hour_train,
        native_propagations=0,optimizer_started=bool(list(directory.glob('optimizer_*_calls.json'))),
        scientific_acceptance=False,blind_h3_acceptance=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from src.exe.run_model import bm_1_h2_compact as c
    with c.reservation(c.OUT/'f4_h2_remaining_budget.json',90.,'whole-linear-raw-audit',native=True,begun=BEGUN) as allocation:
        result=audit(args.directory,allocation['before'])
        c.write(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_job_sha256'}),flush=True)
