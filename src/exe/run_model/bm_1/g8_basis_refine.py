"""One recorded follow-up for the conditional-node resolution failure in G8.

Uses the SAME dictionary and fixture; new work is exactly FV513 and mixture513
five-node banks. Retains the initial failed coarse check without reclassification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

from src.exe.run_model.bm_1.g8 import write, retain_bank, compare, TOLERANCES
from src.exe.run_model.bm_1.g8_basis import node_comparison, projection_max
from src.method.bm_1_g8 import bank_filter
from src.method.bm_1_g8_basis import BASIS_KIND
from src.params.bm_1 import BM1ObservationSeries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();source=Path(args.reference);out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    earlier=json.loads((source/'summary.json').read_text())
    # Never turn a changed source replay into continuation evidence.
    for path,digest in earlier['registration']['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Numerical source changed after the initial dictionary study: '+path)
    if (not all(v['all_pass'] for v in earlier['comparisons'].values())
            or not all(v['passed'] for v in earlier['projection_gates'].values())
            or not earlier['checkpoint_resume']['passed']):
        raise ValueError('Follow-up is only authorized for the isolated conditional-grid failure.')
    original_failure=node_comparison(source,'bank_fv129','bank_fv257')
    if original_failure['passed']:raise ValueError('Expected saved coarse FV conditional failure was not found.')
    imported={Path(module.__file__) for name,module in list(sys.modules.items())
              if name.startswith('src.') and getattr(module,'__file__',None)}
    imported.add(Path(__file__))
    hashes={str(p.resolve().relative_to(Path.cwd())):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(imported)}
    retained=['fixture.npz','bank_local257.npz','bank_local257.json','bank_fv257.npz','bank_fv257.json']
    registration=dict(study='G8 conditional bank resolution follow-up',parent_study_sha256=hashlib.sha256((source/'summary.json').read_bytes()).hexdigest(),
        reason='All original marginal/projection gates pass, but conditional-node 129-to-257 fails for both mixture and saved FV; test exactly one finer grid.',
        original_fv_conditional_failure=original_failure,prior_numerical_accept=earlier['numerical_development_accept'],
        new_solves=['bank_fv513','bank_local513'],cells=513,step=1/192,basis_kind=BASIS_KIND,
        total_followup_cap_seconds=300,prior_invocation_seconds=earlier['invocation_seconds'],
        combined_numerical_cap_seconds=600,tolerances=TOLERANCES,
        source_sha256=hashes,reference_sha256={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in retained},
        stop_policy='No further grid, dictionary or physical-parameter changes if this paired refinement fails.',
        conditional_gates='Every rho node mass L1<=.05, state mean<=.005, pre-update log-evidence<=.05; plus unchanged marginal gates.')
    write(out/'registration.json',registration)
    start=time.perf_counter();deadline=start+min(300.,600.-earlier['invocation_seconds'])
    for name in retained:shutil.copy2(source/name,out/name)
    with np.load(out/'fixture.npz') as f:
        obs=BM1ObservationSeries(f['times'],f['values'],'hour',f['observed'])
        truth=f['truth_states'],f['truth_demand']
    summaries={};failures={}
    for name,basis in [('bank_fv513',None),('bank_local513',BASIS_KIND)]:
        t=time.perf_counter()
        def checkpoint(record):
            if record['time'] in (12.,24.):
                write(out/(name+'_origin_'+str(int(record['time']))+'.json'),record['origin'].to_record())
        try:
            records,model,parameters,_=bank_filter(513,obs,1/192,basis_kind=basis,deadline=deadline,on_step=checkpoint)
            summaries[name]=retain_bank(out,name,records,model,parameters,time.perf_counter()-t,truth)
            print(name,round(summaries[name]['seconds'],3),'s',flush=True)
        except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failures[name]=dict(message=str(exc),seconds=time.perf_counter()-t)
            write(out/(name+'_failure.json'),failures[name]);break
    comparisons={};conditional={}
    for label,left,right in [('fv_refinement','bank_fv257','bank_fv513'),
                             ('local_refinement','bank_local257','bank_local513'),
                             ('local257_vs_fv513','bank_local257','bank_fv513'),
                             ('local513_vs_fv513','bank_local513','bank_fv513')]:
        if (out/(left+'.npz')).exists() and (out/(right+'.npz')).exists():
            comparisons[label]=compare(out,left,right,bank=True)
            conditional[label]=node_comparison(out,left,right)
    projection=projection_max(summaries.get('bank_local513',{}).get('posterior_projection',[]))
    elapsed=time.perf_counter()-start
    passed=(len(summaries)==2 and not failures and all(v['all_pass'] for v in comparisons.values())
            and all(v['passed'] for v in conditional.values()) and projection<=TOLERANCES['projection_l1']
            and elapsed<=300 and elapsed+earlier['invocation_seconds']<=600)
    write(out/'summary.json',dict(registration=registration,summaries=summaries,failures=failures,
        comparisons=comparisons,conditional_checks=conditional,maximum_bayes_projection_l1=projection,
        invocation_seconds=elapsed,combined_numerical_seconds=elapsed+earlier['invocation_seconds'],
        numerical_development_accept_257_and_513=bool(passed),coarse_129_conditional_accept=False,
        initial_study_accept_unchanged=earlier['numerical_development_accept'],
        historical_accept=False,parameter_recovery_accept=False,price_optimization_performed=False))
    print('Paired G8 node refinement complete; fine-grid development accept =',passed,flush=True)


if __name__=='__main__':main()

