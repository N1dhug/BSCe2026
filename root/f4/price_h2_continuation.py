"""Price cached timings and remaining stages; no model propagation or final Y."""
from time import perf_counter,time
BEGUN=perf_counter()
import argparse
import json
import os
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    directory=a.run.resolve();ledger_path=directory/'preparation_ledger.json';ledger=json.loads(ledger_path.read_text())
    if (directory/'runtime_plan.json').exists():raise FileExistsError('Preserve pricing evidence; no automatic repeat.')
    entry=dict(label='native-development|continuation-cached-runtime-pricing',status='reserved',charged_seconds=15.,started_utc_seconds=time())
    ledger['entries'].append(entry);ledger['charged_seconds']=sum(e['charged_seconds'] for e in ledger['entries'])
    ledger_path.write_text(json.dumps(ledger)+'\n')
    try:
        from src.exe.run_model.bm_1_h2_continuation import c,verify_origin
        from root.f4.measure_h2_compact_rates import work_graph
        manifest,origin,old=verify_origin(directory)
        campaign=c.load(ROOT/ledger['predecessor_campaign_path'])
        rates=c.load(ROOT/'root/f4/h2_cap_extension_decision_20261007.json')['protected_rates']
        original_rates={g:dict(v) for g,v in rates.items()}
        grouped={}
        for path in sorted((old/'jobs').glob('*.npz')):
            with zipfile.ZipFile(path) as z:raw=json.loads(z.read('metadata.json'))['report']
            t=raw['token'];grouped.setdefault((t['grid'],t['kind']),[]).append(raw['elapsed_seconds'])
        means={g:{k:c.ordered_sum(v)/len(v) for (grid,k),v in grouped.items() if grid==g} for g in ('base','fine')}
        maxima={g:{k:max(v) for (grid,k),v in grouped.items() if grid==g} for g in ('base','fine')}
        base=c.load(old.parent/'f4_h2_bank_extension_diagnostic_20261007/report.json')
        fine=c.load(old.parent/'f4_h2_bank_fine_control_20261007/report.json')
        # This timestamp was printed by the fine control before observed work;
        # preserve its exact captured stdout as timing evidence in the plan.
        prior_elapsed=22.3761129409977
        observed_fine_upper=fine['elapsed_seconds']-prior_elapsed
        for grid in ('base','fine'):
            for kind in rates[grid]:
                rates[grid][kind]=max(rates[grid][kind],maxima[grid][kind])
        rates['base']['observed']=max(rates['base']['observed'],base['elapsed_seconds'],maxima['base']['observed'])
        rates['fine']['observed']=max(rates['fine']['observed'],observed_fine_upper,maxima['fine']['observed'])
        graph=work_graph(rates,48,24,campaign['charged_seconds'])
        # The completed linear and two-piece families have full own validation.
        release=0.
        for _ in range(2):
            release+=1.5*(24*rates['base']['observed']+72*rates['fine']['observed']+
                24*rates['fine']['prediction']+24*rates['base']['moments']+24*rates['fine']['moments'])
        validation=graph['protected_validation_seconds']-release
        maximum_calls=49-27
        search_jobs=dict(observed=maximum_calls*48,prediction=(maximum_calls-1)*24)
        measured_search=search_jobs['observed']*means['base']['observed']+search_jobs['prediction']*means['base']['prediction']
        priced_search=1.5*(search_jobs['observed']*rates['base']['observed']+search_jobs['prediction']*rates['base']['prediction'])
        report=dict(schema='f4-h2-continuation-runtime-plan-v1',native_propagations=0,final_observations_decoded=0,
            complete_four_piece_calls=27,pending_candidate=28,maximum_remaining_calls=maximum_calls,
            old_job_counts={f'{g}/{k}':len(v) for (g,k),v in grouped.items()},
            saved_mean_seconds=means,saved_max_seconds=maxima,original_protected_rates=original_rates,
            proposed_protected_rates=rates,captured_fine_prior_stdout=dict(status='own_fine_prior_completed',hours=24,elapsed=prior_elapsed),
            fine_observed_phase_upper_seconds=observed_fine_upper,
            protected_remaining_validation_seconds=validation,validation_safety_factor=1.5,
            maximum_remaining_search_seconds_at_old_means=measured_search,
            maximum_remaining_search_seconds_with_safety_at_old_means=1.5*measured_search,
            maximum_remaining_search_seconds_at_protected_rates=priced_search,
            full49_scenario_admitted=False,convergence_guaranteed=False,
            timing_scope='Cached nonlinear jobs and one nine-contour base/fine control; no universal upper bound.',
            validation_work_graph=graph,completed_family_validation_release_seconds=release,
            elapsed_seconds=perf_counter()-BEGUN)
        c.write(directory/'runtime_plan.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('validation_work_graph','old_job_counts')}),flush=True)
    finally:
        entry.update(status='closed',charged_seconds=perf_counter()-BEGUN)
        total=0.
        for e in ledger['entries']:total+=e['charged_seconds']
        ledger['charged_seconds']=total;ledger_path.write_text(json.dumps(ledger)+'\n')

if __name__=='__main__':main()
