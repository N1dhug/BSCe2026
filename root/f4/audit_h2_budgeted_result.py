"""Audit saved candidate likelihoods and reconstruct COBYQA without model replay."""
from time import perf_counter
BEGUN=perf_counter()-.25
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'


def audit(directory):
    import numpy as np
    from scipy.optimize import minimize
    from src.exe.run_model import bm_1_h2_compact as c
    from root.f4.run_h2_budgeted_search import EXTRA
    directory=Path(directory); manifest=c.load(directory/'manifest.json')
    outer=c.load(directory/'budgeted_execution.json')
    c.require(manifest['identity']==c.digest({k:v for k,v in manifest.items() if k!='identity'})
        and manifest['environment']==c.environment(),'Numerical manifest/environment differs.')
    c.require(manifest['source_sha256']==c.source_hashes(),'Numerical source differs from the run.')
    c.require(outer['source_sha256']=={p:c.sha(ROOT/p) for p in EXTRA},'Outer controller source differs.')
    c.require(outer['identity']==c.digest({k:v for k,v in outer.items() if k!='identity'})
        and outer['numerical_manifest_identity']==manifest['identity'],'Execution identity differs.')
    for name,sources in (('source.zip',manifest['source_sha256']),('budgeted_source.zip',outer['source_sha256'])):
        with zipfile.ZipFile(directory/name) as z:
            c.require(set(z.namelist())==set(sources),'Source archive members differ.')
            for path,h in sources.items():c.require(c.hashlib.sha256(z.read(path)).hexdigest()==h,'Archived source differs.')
    data=c.receipt();jobs={};counts={};failures=[];job_hashes={}
    maxima=dict(mass=0.,conditional=0.,overlap=0.)
    for name,value in (('observations.npz',data['observations']),('inputs.npz',data['inputs'])):
        c.require(c.fingerprint(c._read_array_checkpoint((directory/name).read_bytes()))==c.fingerprint(value),'Frozen inputs differ.')
    c.require(c.load(directory/'packet.json')==data['packet'],'Frozen data packet differs.')
    for path in sorted((directory/'jobs').glob('*.npz')):
        r=c._read_array_checkpoint(path.read_bytes());t=r['token']
        c.require(path.name==c.digest(t)+'.npz' and t['identity']==manifest['identity']
            and r['checksum']==c.fingerprint({k:v for k,v in r.items() if k!='checksum'}),'Job checksum/identity failed.')
        key=t['candidate'],t['grid'],t['branch'],t['hour'],t['kind']
        c.require(key not in jobs,'Duplicate job identity.')
        c.require(t['branch'] in ('prefinal','open') and t['grid'] in ('base','fine')
            and np.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>0.,'Invalid job branch/grid/timing.')
        if t['kind']=='observed':
            c.require(t['branch']=='prefinal' and 24<=t['hour']<96
                and bool(data['observations']['observed'][t['hour']-24])
                and t['value']==float(data['observations']['values'][t['hour']-24]),'Wrong observed-hour value/mask.')
            c.same_state(r['result']['log_state'],r['result']['log_posterior_mass'])
        else:c.require(t['value'] is None,'Prediction/moment/CDF job has an observation.')
        for field,metric in (('maximum_abs_mass_error','mass'),('maximum_relative_conditional_guard','conditional'),('maximum_overlap_log_difference','overlap')):
            if field in r['result']:maxima[metric]=max(maxima[metric],r['result'][field])
        job_hashes[path.name]=c.sha(path)
        jobs[key]=r;counts[t['candidate']]=counts.get(t['candidate'],0)+1
        c.state_check(r['prior_log_state']);c.state_check(r['result']['log_state'])
    for path in sorted((directory/'jobs').glob('*.failed.json')):
        r=c.load(path);c.require(r['token']['identity']==manifest['identity']
            and path.name==c.digest(r['token'])+'.failed.json','Failure identity differs.')
        failures.append(r)
    candidates={'linear':c.LINEAR,c.curve_key(c.PROBE):c.PROBE};objectives={};optimizer_reports={}
    class EndOfTrace(Exception):pass
    for pieces in (2,4):
        path=directory/f'optimizer_{pieces}_calls.json'
        if not path.exists():continue
        trace=c.load(path);calls=trace['calls'];cursor=0;pending=[]
        c.require(trace['identity']==manifest['identity'],'Optimizer identity changed.')
        for row in calls:
            candidate=dict(pieces=pieces,logits=row['logits']);key=c.curve_key(candidate)
            c.require(key==row['key'] and row['penalty']==c.penalty(candidate)
                and row['objective']==-row['score']+row['penalty'],'Objective/penalty algebra changed.')
            candidates[key]=candidate;objectives[key]=row['score']
        def cached(x):
            nonlocal cursor
            candidate=dict(pieces=pieces,logits=np.asarray(x).tolist())
            if cursor==len(calls):
                pending.append(candidate);raise EndOfTrace()
            row=calls[cursor]
            c.require(candidate['logits']==row['logits'],'Pinned COBYQA requested a different cached point.')
            cursor+=1
            return row['objective']
        replayed=None
        try:
            replayed=minimize(cached,np.zeros(2*(pieces-1)),method='COBYQA',bounds=[(-4.,4.)]*(2*(pieces-1)),
                options=dict(maxfev={2:17,4:33}[pieces],initial_tr_radius=.5,final_tr_radius=.02,scale=False))
        except EndOfTrace:pass
        c.require(cursor==len(calls),'Unconsumed optimizer calls.')
        result_path=directory/f'optimizer_{pieces}_result.json'
        if result_path.exists():
            saved=c.load(result_path)
            c.require(replayed is not None and saved['status']==int(replayed.status)
                and saved['objective']==float(replayed.fun) and saved['candidate']['logits']==replayed.x.tolist()
                and saved['converged']==(bool(replayed.success) and int(replayed.status)==0),'Optimizer termination reconstruction differs.')
            optimizer_reports[str(pieces)]=dict(calls=len(calls),status=saved['status'],converged=saved['converged'])
        else:
            c.require(len(pending)==1,'Incomplete optimizer has no reproducible pending point.')
            candidates[c.curve_key(pending[0])]=pending[0]
            optimizer_reports[str(pieces)]=dict(calls=len(calls),converged=False,pending_candidate=pending[0])
    from src.exe.run_model.bm_1_history import _beta_mass
    verified_scores={};partial={}
    # Every fitted objective must own its full 24-prior + 48-TRAIN stream.
    for key,candidate in candidates.items():
        if key=='linear':logs=data['jobs'][('base',0,'prediction')]['prior_log_state']
        else:
            n=data['packet']['numerics']['base_cells'];p=data['packet']['prior']
            with np.errstate(divide='ignore'):
                logs=np.log(_beta_mass(np.linspace(0.,1.,n+1),alpha=p['alpha'],beta=p['beta']))
        scores=[];completed=0
        for hour in range(72):
            kind='prediction' if hour<24 else 'observed';j=key,'base','prefinal',hour,kind
            if j in jobs:
                raw=jobs[j];prior=raw['prior_log_state'];result=raw['result']
                if hour>=24:c.require(raw['token']['value']==float(data['observations']['values'][hour-24]),'Wrong TRAIN observation.')
            elif key=='linear' and hour<27:
                raw=data['jobs'][('base',hour,kind)];prior=raw['prior_log_state'];result=dict(raw)
                result['log_state']=raw['posterior_log_state'] if kind=='observed' else raw['log_state']
            else:break
            c.same_state(prior,logs);logs=result['log_state'];completed+=1
            maxima['mass']=max(maxima['mass'],result['maximum_abs_mass_error'])
            if hour>=24:
                scores.append(result['log_evidence'])
                maxima['conditional']=max(maxima['conditional'],result['maximum_relative_conditional_guard'])
                maxima['overlap']=max(maxima['overlap'],result['maximum_overlap_log_difference'])
        score=c.ordered_sum(scores)
        if key in objectives:
            c.require(completed==72 and score==objectives[key],'Objective lacks its exact complete chronological raw TRAIN likelihood.')
            verified_scores[key]=score
        else:partial[key]=completed
    c.require(maxima['mass']<=1e-10 and maxima['conditional']<=1e-8 and maxima['overlap']<=1e-7,'Stored numerical guard failed.')
    registry=c.load(directory/'candidate_registry.json') if (directory/'candidate_registry.json').exists() else {'families':{'2':[],'4':[]}}
    c.require(registry.get('identity',manifest['identity'])==manifest['identity'],'Candidate registry identity differs.')
    c.require(set(counts).issubset(candidates),'Saved jobs belong to an untraced candidate.')
    c.require(all(len(keys)==len(set(keys)) and set(keys).issubset(candidates) for keys in registry['families'].values()),'Candidate registry duplicates/untraced pairs.')
    total=1+sum(len(v) for v in registry['families'].values())
    c.require(len(registry['families']['2'])<=16 and len(registry['families']['4'])<=32 and total<=49,'Candidate ceiling exceeded.')
    report=c.load(directory/'budgeted_report.json')
    from root.f4.audit_h2_saved_stages import retained_stages
    stages=retained_stages(directory,data,jobs,candidates)
    if report['status']=='prefinal_completed':
        c.require(all(stages[k] for k in ('selection_verified','open_forecast_verified','handoff_verified'))
            and set(optimizer_reports)=={'2','4'} and all(r['converged'] for r in optimizer_reports.values()),
            'A completed report lacks the complete fit/validation/forecast/handoff evidence.')
    return dict(schema='f4-h2-budgeted-raw-result-audit-v1',status='passed',run_status=report['status'],
        numerical_manifest_identity=manifest['identity'],execution_identity=outer['identity'],
        numerical_source_files=len(manifest['source_sha256']),raw_jobs=len(jobs),
        completed_objectives=len(verified_scores),reserved_distinct_pairs=total,
        optimizer=optimizer_reports,verified_train_scores=verified_scores,
        partial_stream_hours=partial,guards=maxima,failed_jobs=failures,stages=stages,
        source_job_sha256=job_hashes,
        audit_source_sha256={p:c.sha(ROOT/p) for p in ('root/f4/audit_h2_budgeted_result.py','root/f4/audit_h2_saved_stages.py')},
        native_propagations=0,scientific_acceptance=False,blind_h3_acceptance=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from src.exe.run_model import bm_1_h2_compact as c
    with c.reservation(c.OUT/'f4_h2_remaining_budget.json',90.,'budgeted-result-raw-audit',native=True,begun=BEGUN):
        try:result=audit(args.directory)
        except Exception:
            c.write(args.output,dict(status='failed',traceback=traceback.format_exc(),native_propagations=0))
            raise
        c.write(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_job_sha256'}),flush=True)
