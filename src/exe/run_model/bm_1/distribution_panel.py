"""Registered, bounded G6b synthetic panel; raw outputs precede comparison.

Run from repository root: python -m src.exe.run_model.bm_1.distribution_panel
Use --output NEW_DIRECTORY to retain another invocation separately.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import time

import numpy as np
from src.control.bm_1 import sample_beta_initial_states
from src.method.euler_maruyama import euler_maruyama
from src.method.bm_1_distribution_panel import solve_distribution, solve_em, linear_noise_reference

TOLERANCES = {
    'state_mean': .005, 'demand_mean': .005,
    'state_variance': .002, 'demand_variance': .002,
    'state_cdf': .03, 'demand_cdf': .03,
    'state_quantile': .02, 'demand_quantile': .02,
    'objective_relative': .02, 'initial_projection_l1': .05,
    'mass_error': 1e-9, 'negative_mass': 1e-12,
}
CASES = [
    ('fv_65_h24', 'fv', 65, None, 1/24),
    ('fv_129_h24', 'fv', 129, None, 1/24),
    ('fv_129_h48', 'fv', 129, None, 1/48),
    ('mix_20_h24', 'mixture', 129, 20, 1/24),
    ('mix_20_h48', 'mixture', 129, 20, 1/48),
    ('mix_38_h24', 'mixture', 129, 38, 1/24),
    ('mix_38_h48', 'mixture', 129, 38, 1/48),
]


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return plain(value.item())
    # The explicitly supplied EM grid uses max_step=inf (no added knots).
    # Preserve this legitimate configuration in strict JSON, not a NaN token.
    if isinstance(value, float) and not np.isfinite(value):
        return 'Infinity' if value>0 else '-Infinity' if value<0 else 'NaN'
    if isinstance(value, dict): return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [plain(item) for item in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')


def retain(directory, name, result, elapsed):
    arrays = result.pop('arrays', {})
    diagnostics = result.get('diagnostics', {})
    # Keep detailed diagnostics once in the compressed raw arrays, not huge JSON.
    for key in list(diagnostics):
        if isinstance(diagnostics[key], np.ndarray):
            arrays['diagnostic_'+key] = diagnostics.pop(key)
    result['seconds'] = elapsed
    np.savez_compressed(directory/(name+'.npz'), **arrays)
    write_json(directory/(name+'.json'), result)
    return result


def compare(candidate, reference, sampling=False):
    report = {}
    for key in ('state_mean','demand_mean','state_variance','demand_variance',
                'state_cdf','demand_cdf','state_quantile','demand_quantile'):
        difference = np.abs(np.asarray(candidate['hourly'][key])-np.asarray(reference['hourly'][key]))
        tolerance = TOLERANCES[key]
        standard_error = np.asarray(candidate['hourly'].get(key+'_se', 0.)) if sampling else 0.
        allowed = tolerance+3*standard_error
        report[key] = {'max_abs_error':float(difference.max()),
                       'within_tolerance_plus_3se':bool(np.all(difference<=allowed))}
    error = abs(candidate['objective']-reference['objective'])
    allowed = TOLERANCES['objective_relative']*abs(reference['objective'])
    if sampling: allowed += 3*candidate['objective_se']
    report['objective'] = {'abs_error':error, 'relative_error':error/abs(reference['objective']),
                           'within_tolerance_plus_3se':bool(error<=allowed)}
    report['all_pass'] = all(item['within_tolerance_plus_3se'] for item in report.values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='files/output/bm_1/g6b_distribution_panel_20261006')
    parser.add_argument('--reuse-interface-pilot', help='First attempt directory after midpoint-index-only repair.')
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), Path('src/method/bm_1_distribution_panel.py')]
    # Hash imported implementation and validation modules, not an unrelated test suite.
    import sys
    source_files += [Path(module.__file__) for name,module in list(sys.modules.items())
                     if name.startswith('src.') and getattr(module,'__file__',None)]
    sources = {str(path.resolve().relative_to(Path.cwd())):hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(set(source_files))}
    manifest = {'study_id':directory.name, 'scope':'one synthetic 24-hour development case; no model fit',
        'time_unit':'hours', 'model':{'C':4.,'k':2.,'Xi':.4,'sigma_x':.2,'f':'1-2x','g':'1-2u'},
        'baseline':'.5+.1*sin(2*pi*t/24)', 'target':'.5',
        'prices':[.3,.7,.3,.7], 'price_edges':[0,6,12,18,24],
        'prior':{'law':'Beta(20,20)','mean':.5,'variance':1/164},
        'seeds':{'initial':20261006,'brownian':20261007},
        'paths':{'pilot':256,'main':2048}, 'mcse_multiplier':3,
        'mcse_scope':'pointwise uncertainty screen, not a simultaneous confidence claim; quantile SE not estimated',
        'fv_mixture_cases':CASES, 'em_actual_steps':[1/48,1/96],
        'em_objective_steps':[1/24,1/48], 'mixture_cfl':.4,
        'pilot_seconds_cap':60, 'main_seconds_cap':600,
        'tolerances':TOLERANCES, 'source_sha256':sources,
        'protocol':'levels and tolerances frozen before random generation; pilot retained, no extra search',
        'reused_pilot':args.reuse_interface_pilot,
        'interface_revisions':[
            'v1 stopped at floating-point midpoint lookup; mixture raw output complete',
            'v2 stopped serializing EM max_step=inf; EM npz complete, scalar diagnostics not retained',
            'v3 uses nearest verified uniform knot and explicit Infinity JSON string; no scientific levels changed'],
        'prior_interface_probe':{'registered':False,'raw_complete':False,'runtime_measured':False,
            'method':'mixture degree20 grid129 h1/24',
            'objective':.2415663970348848,'initial_l1':.2916670208378914,
            'use':'interface observation only; excluded from comparison and runtime accounting'},
        'runtime_environment':{'python':sys.version,'numpy':np.__version__,
            'blas_threads':os.environ.get('OPENBLAS_NUM_THREADS'),'workers':1}}
    write_json(directory/'registration.json', manifest)  # Before random generation or evaluation.
    start=time.perf_counter()
    initial=sample_beta_initial_states(2048,seed=20261006)
    fine_time=np.arange(2305)/96
    fine=euler_maruyama.brownian_increments(fine_time,n_paths=2048,seed=20261007)
    coarse=euler_maruyama.coarsen_brownian_increments(fine,2)
    np.savez_compressed(directory/'common_random_inputs.npz',initial_states=initial,
                        fine_time=fine_time,fine_increments=fine)
    results={}
    pilot_start=time.perf_counter();pilot_deadline=pilot_start+60
    if args.reuse_interface_pilot:
        previous=Path(args.reuse_interface_pilot)
        registration=json.loads((previous/'registration.json').read_text())
        for key in ('model','prior','seeds','fv_mixture_cases','tolerances'):
            if registration[key]!=plain(manifest[key]): raise ValueError('Reused pilot contract mismatch: '+key)
        for suffix in ('.json','.npz'):
            shutil.copy2(previous/('mix_20_h24'+suffix),directory/('mix_20_h24'+suffix))
        results['mix_20_h24']=json.loads((directory/'mix_20_h24.json').read_text())
    else:
        t=time.perf_counter()
        results['mix_20_h24']=retain(directory,'mix_20_h24',
            solve_distribution('mixture',129,20,1/24,deadline=pilot_deadline),time.perf_counter()-t)
    t=time.perf_counter()
    pilot_em=retain(directory,'em_pilot_256_h48',
        solve_em(1/24,initial[:256],coarse[:256],deadline=pilot_deadline),time.perf_counter()-t)
    pilot_seconds=time.perf_counter()-pilot_start
    write_json(directory/'pilot.json',{'seconds':pilot_seconds,'passed_budget':pilot_seconds<=60,
               'retained_main_result':'mix_20_h24','remaining_upper_budget_seconds':600})
    if pilot_seconds>60: raise RuntimeError('Pilot budget exhausted; do not start main.')
    print(f'Pilot {pilot_seconds:.3f}s; fixed levels locked; remaining cap 600s.',flush=True)
    main_start=time.perf_counter();deadline=main_start+600
    failures={}
    for name,method,cells,degree,step in CASES:
        if name in results: continue
        t=time.perf_counter()
        try:
            result=solve_distribution(method,cells,degree,step,deadline=deadline)
            results[name]=retain(directory,name,result,time.perf_counter()-t)
            print(f'{name}: {results[name]["seconds"]:.3f}s; J={result["objective"]:.8f}',flush=True)
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
            failures[name]={'exception':type(exc).__name__,'message':str(exc),'seconds':time.perf_counter()-t}
            write_json(directory/(name+'_failure.json'),failures[name])
            print(f'{name}: STOP {exc}',flush=True)
        if time.perf_counter()>=deadline: break
    for name,step,increments in [('em_2048_h48',1/24,coarse),('em_2048_h96',1/48,fine)]:
        if time.perf_counter()>=deadline: break
        t=time.perf_counter()
        try:
            results[name]=retain(directory,name,solve_em(step,initial,increments,deadline=deadline),time.perf_counter()-t)
            print(f'{name}: {results[name]["seconds"]:.3f}s; J={results[name]["objective"]:.8f}',flush=True)
        except (RuntimeError,ValueError) as exc:
            failures[name]={'exception':type(exc).__name__,'message':str(exc),'seconds':time.perf_counter()-t}
            write_json(directory/(name+'_failure.json'),failures[name])
    t=time.perf_counter()
    reference=linear_noise_reference()
    write_json(directory/'linear_noise_reference.json',reference)
    reference_seconds=time.perf_counter()-t
    comparisons={}
    fv=results.get('fv_129_h48')
    if fv:
        comparisons={name:compare(result,fv,sampling=name.startswith('em_'))
                     for name,result in results.items() if name!='fv_129_h48'}
    coupled={}
    if 'em_2048_h48' in results and 'em_2048_h96' in results:
        for key in ('states_hourly','demand_hourly','path_objectives'):
            with np.load(directory/'em_2048_h48.npz') as a,np.load(directory/'em_2048_h96.npz') as b:
                differences=a[key]-b[key]
            mean=differences.mean(axis=0)
            coupled[key]={'max_abs_mean_difference':float(np.max(abs(mean))),
                          'max_paired_standard_error':float(np.max(np.std(differences,axis=0,ddof=1)/np.sqrt(2048)))}
    # Separate grid, time and degree comparisons; CFL may make mixture h levels almost identical.
    refinement={}
    for name,left,right in [('fv_space','fv_65_h24','fv_129_h24'),('fv_time','fv_129_h24','fv_129_h48'),
                            ('mix20_time','mix_20_h24','mix_20_h48'),('mix38_time','mix_38_h24','mix_38_h48'),
                            ('mix_degree','mix_20_h48','mix_38_h48')]:
        if left in results and right in results: refinement[name]=compare(results[left],results[right])
    linear_comparison={key:float(np.max(abs(np.asarray(reference[key])-np.asarray(fv['hourly'][key]))))
                       for key in ('state_mean','state_variance','demand_mean','demand_variance')} if fv else {}
    write_json(directory/'summary.json',{'results':{name:{'objective':r['objective'],'seconds':r['seconds'],
        'objective_se':r.get('objective_se'),'initial_projection':r.get('initial_projection'),
        'diagnostics':r['diagnostics']} for name,r in results.items()},
        'failures':failures,'comparisons_to_finest_fv':comparisons,'refinement':refinement,
        'coupled_em':coupled,'linear_noise_max_abs_difference':linear_comparison,
        'pilot_seconds':pilot_seconds,'main_seconds':time.perf_counter()-main_start,
        'linear_reference_seconds':reference_seconds,'invocation_seconds':time.perf_counter()-start,
        'budget_accounting':'includes random-input retention, pilot, solvers and raw output; excludes unmeasured interface probe',
        'status':'development evidence only; method acceptance determined separately from probability conservation'})
    print('Panel complete; raw outputs and frozen registration retained.',flush=True)


if __name__=='__main__': main()
