"""One fixed G8 posterior-dictionary repair, reusing the saved development case.

No new data, parameter search, adaptive basis, FV restart, or price optimizer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import scipy

from src.exe.run_model.bm_1.g8 import write, TOLERANCES, retain_known, retain_bank, compare
from src.method.bm_1_g8 import context, known_filter, bank_filter, G8BankOrigin
from src.method.bm_1_g8_basis import BASIS_KIND, dictionary_shapes
from src.params.bm_1 import BM1ObservationSeries, conservative_mass_transfer


def projection_max(values):
    if isinstance(values,dict):
        return max([float(v) if k in ('cell_mass_l1_error','raw_to_projected_cell_mass_l1')
                    else projection_max(v) for k,v in values.items()],default=0.)
    if isinstance(values,(tuple,list)):
        return max((projection_max(v) for v in values),default=0.)
    return 0.


def node_comparison(directory,left,right):
    with np.load(directory/(left+'.npz')) as a,np.load(directory/(right+'.npz')) as b:
        reference=b['conditional_mass']
        transferred=np.array([[conservative_mass_transfer(a['edges'],row,b['edges']) for row in hour]
                              for hour in a['conditional_mass']])
        l1=np.sum(abs(transferred-reference),axis=2)
        x=.5*(b['edges'][:-1]+b['edges'][1:])
        mean=np.max(abs(transferred@x-reference@x))
        score=np.max(abs(a['conditional_log_evidence'][a['observed']]-b['conditional_log_evidence'][b['observed']]))
    return dict(maximum_conditional_mass_l1=float(l1.max()),maximum_conditional_state_mean_error=float(mean),
                maximum_conditional_log_evidence_error=float(score),
                passed=bool(l1.max()<=TOLERANCES['posterior_mass_l1'] and mean<=TOLERANCES['state_mean']
                            and score<=TOLERANCES['log_evidence']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True,help='Retained g8_run_v2 directory.')
    parser.add_argument('--output',required=True,help='A fresh output directory.')
    args=parser.parse_args();reference=Path(args.reference);out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    alpha,beta=dictionary_shapes()
    imported={Path(module.__file__) for name,module in list(sys.modules.items())
              if name.startswith('src.') and getattr(module,'__file__',None)}
    imported.add(Path(__file__))
    hashes={str(path.resolve().relative_to(Path.cwd())):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(imported)}
    expected=['registration.json','fixture.npz','summary.json']
    expected += [name+suffix for name in ['known_fv129','known_fv257','bank_fv129','bank_fv257'] for suffix in ['.npz','.json']]
    reference_hashes={name:hashlib.sha256((reference/name).read_bytes()).hexdigest() for name in expected}
    registration=dict(study='G8 fixed localized Beta dictionary',source_parent='e77aaafdf2613fa2e846f458386145ce121ae33b',
        basis_kind=BASIS_KIND,alpha=alpha,beta=beta,components=42,cell_average_basis=True,
        selection='One design chosen after exposed G8 development evidence: kappa=256 is the smallest power of two whose central variance floor is below half the smallest saved FV node variance .00216612; 41 equispaced means plus original Beta(20,20). No independent validation claim.',
        comparisons='Same 24h fixture, physical model, five rho nodes, likelihood GL3, cfl .4 and G8 tolerances.',
        levels=[dict(cells=129,step=1/48),dict(cells=257,step=1/96)],
        budget_seconds=600,pilot_seconds=60,checkpoint=12.,tolerances=TOLERANCES,
        source_sha256=hashes,reference_sha256=reference_hashes,
        software=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        stop_policy='One candidate only. No degree/concentration/mean search after failure. FV remains accepted reference; failed mixture remains unaccepted.')
    write(out/'registration.json',registration)
    start=time.perf_counter();deadline=start+600
    for name in expected:shutil.copy2(reference/name,out/('reference_'+name if name in ('registration.json','summary.json') else name))
    old=json.loads((reference/'registration.json').read_text())
    if (old['model']!=dict(C=4.,k=2.,Xi=.4,sigma_x=.2,f='1-2x',g='1-2u')
        or old['sigma_y']!=.03 or old['missing_hours']!=[6,17] or old['time_unit']!='hour'):
        raise ValueError('Reference does not match the G8 point-demand contract.')
    with np.load(out/'fixture.npz') as f:
        obs=BM1ObservationSeries(f['times'],f['values'],'hour',f['observed'])
        truth=(f['truth_states'],f['truth_demand'])
    summaries={};failures={};comparisons={};node_checks={};resumes={}
    for name,cells,step in [('known_local129',129,1/48),('known_local257',257,1/96)]:
        t=time.perf_counter()
        try:
            result,model,parameters,projector=known_filter(cells,obs,step,basis_kind=BASIS_KIND,
                deadline=min(deadline,start+60) if cells==129 else deadline)
            summaries[name]=retain_known(out,name,result,model,parameters,time.perf_counter()-t,truth)
            np.savez_compressed(out/(name+'_basis.npz'),basis=projector.basis,alpha=alpha,beta=beta)
            print(name,round(summaries[name]['seconds'],3),'s',flush=True)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failures[name]=dict(type=type(exc).__name__,message=str(exc),seconds=time.perf_counter()-t)
            write(out/(name+'_failure.json'),failures[name])
            print(name,'STOP',str(exc),flush=True)
            break
        if cells==129:
            pilot=dict(seconds=time.perf_counter()-start,projection_l1=projection_max(summaries[name]['posterior_projection']),
                       comparison=compare(out,name,'known_fv257'))
            pilot['passed']=bool(pilot['seconds']<=60 and pilot['projection_l1']<=TOLERANCES['projection_l1']
                                 and pilot['comparison']['all_pass'])
            write(out/'pilot.json',pilot)
            if not pilot['passed']:
                failures['pilot']='Registered numerical gate failed; bank and refinement stopped.'
                break
    if 'known_local257' in summaries:
        for name,cells,step in [('bank_local129',129,1/48),('bank_local257',257,1/96)]:
            t=time.perf_counter()
            def checkpoint(record):
                if record['time'] in (12.,24.):
                    write(out/(name+'_origin_'+str(int(record['time']))+'.json'),record['origin'].to_record())
            try:
                records,model,parameters,projector=bank_filter(cells,obs,step,basis_kind=BASIS_KIND,
                    deadline=deadline,on_step=checkpoint)
                summaries[name]=retain_bank(out,name,records,model,parameters,time.perf_counter()-t,truth)
                print(name,round(summaries[name]['seconds'],3),'s',flush=True)
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                failures[name]=dict(type=type(exc).__name__,message=str(exc),seconds=time.perf_counter()-t)
                write(out/(name+'_failure.json'),failures[name])
                break
    for name in summaries:
        reference_name='bank_fv257' if name.startswith('bank') else 'known_fv257'
        comparisons[name+'_vs_fv257']=compare(out,name,reference_name,bank=name.startswith('bank'))
        if name.startswith('bank'):node_checks[name]=node_comparison(out,name,reference_name)
    for kind in ['known','bank']:
        a,b=kind+'_local129',kind+'_local257'
        if a in summaries and b in summaries:
            comparisons[kind+'_local_refinement']=compare(out,a,b,bank=kind=='bank')
            if kind=='bank':node_checks['local_refinement']=node_comparison(out,a,b)
    if 'bank_local257' in summaries:
        name='bank_local257';t=time.perf_counter()
        origin=G8BankOrigin.from_record(json.loads((out/(name+'_origin_12.json')).read_text()))
        tail=BM1ObservationSeries(obs.times[12:],obs.values[12:],'hour',obs.observed[12:])
        try:
            records,_,_,_=bank_filter(257,tail,1/96,basis_kind=BASIS_KIND,origin=origin,deadline=deadline)
            with np.load(out/(name+'.npz')) as raw:
                errors={field:float(np.max(abs(np.stack([r[field] for r in records])-raw[field][12:])))
                        for field in ['conditional_mass','conditional_coefficients','log_weights']}
                errors['log_evidence']=float(np.nanmax(abs(np.array([r['log_evidence'] for r in records])-raw['log_evidence'][12:])))
            resumes=dict(seconds=time.perf_counter()-t,errors=errors,passed=max(errors.values())<=TOLERANCES['checkpoint'])
            write(out/(name+'_resumed_origin_24.json'),records[-1]['origin'].to_record())
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            resumes=dict(passed=False,message=str(exc),seconds=time.perf_counter()-t)
    projection_gates={name:dict(maximum_l1=projection_max(d['posterior_projection']),
                               passed=projection_max(d['posterior_projection'])<=TOLERANCES['projection_l1'])
                      for name,d in summaries.items()}
    passed=(len(summaries)==4 and not failures and all(v['all_pass'] for v in comparisons.values())
            and all(v['passed'] for v in node_checks.values())
            and all(v['passed'] for v in projection_gates.values()) and resumes.get('passed',False))
    result=dict(registration=registration,summaries=summaries,failures=failures,comparisons=comparisons,
                conditional_checks=node_checks,projection_gates=projection_gates,checkpoint_resume=resumes,
                invocation_seconds=time.perf_counter()-start,numerical_development_accept=bool(passed),
                historical_accept=False,parameter_recovery_accept=False,price_optimization_performed=False)
    write(out/'summary.json',result)
    print('G8 basis study complete; development numerical accept =',passed,flush=True)


if __name__=='__main__':main()

