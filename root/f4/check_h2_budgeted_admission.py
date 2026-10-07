"""One algebra/source control for the explicit bounded-search admission."""
from time import perf_counter,time
BEGUN=perf_counter()-.25
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'


def control():
    from src.exe.run_model import bm_1_h2_compact as c
    from root.f4.measure_h2_compact_rates import work_graph
    run=c.OUT/'f4_h2_staged_night_v2_20261007'
    manifest=c.load(run/'manifest.json')
    assert manifest['source_sha256']==c.source_hashes(), 'Numerical source must remain completely unchanged.'
    audit=c.load(c.OUT/'f4_h2_linear_window_audit_20261007.json')
    data=c.receipt();rates=audit['rates']
    with tempfile.TemporaryDirectory() as temporary:
        allocation=dict(before=audit['charged_at_pricing'],begun=perf_counter(),deadline=time()+60.)
        engine=c.Engine(Path(temporary),data,manifest,allocation)
        engine.rates=rates
        graph=work_graph(rates,48,24,audit['charged_at_pricing'])
        engine.validation_remaining=graph['protected_validation_seconds']
        # Same four accounting groups as validated_family, without rereading
        # or replaying the already audited 96-hour streams.
        for grid in ('base','fine'):
            c.release_validation(engine,'linearselection',grid,'observed',24)
            c.release_validation(engine,'linearselection',grid,'prediction',0)
            c.release_validation(engine,'linearselection',grid,'moments',24)
        c.release_validation(engine,'lineartrain','fine','observed',48)
        c.release_validation(engine,'lineartrain','fine','prediction',24)
        remaining=engine.validation_remaining
        assert abs(remaining-audit['protected_remaining_validation_seconds'])<=1e-7
        c.release_validation(engine,'lineartrain','fine','observed',48)
        assert engine.validation_remaining==remaining, 'Duplicate physical curve must not release twice.'
        funded=36000.-allocation['before']-remaining-60.
        assert funded>0. and abs(funded-audit['maximum_time_fundable_for_search'])<=1e-7
        assert abs(allocation['before']+remaining+funded+60.-36000.)<=1e-7
        assert audit['remaining49_global_max_total_seconds']>36000.
        engine.check(search=True)
        allocation['before']=36000.-remaining-60.
        try:engine.check(search=True)
        except TimeoutError:pass
        else:raise AssertionError('Search consumed protected validation.')
    for p in ('root/f4/run_h2_budgeted_search.py','root/f4/export_h2_final_observations.py'):
        ast.parse((ROOT/p).read_bytes())
    return dict(status='passed',numerical_source_files_unchanged=len(manifest['source_sha256']),
        native_propagations=0,old49_full_graph_rejected=True,remaining_validation_seconds=remaining,
        initial_search_seconds_available=funded,duplicate_credit_rejected=True,
        search_reserve_stop_passed=True,source_sha256={p:c.sha(ROOT/p) for p in
            ('root/f4/run_h2_budgeted_search.py','root/H2_COMPACT_BUDGETED_SEARCH.md','root/f4/check_h2_budgeted_admission.py')},
        scientific_acceptance=False,blind_h3_acceptance=False)


if __name__=='__main__':
    from src.exe.run_model import bm_1_h2_compact as c
    with c.reservation(c.OUT/'f4_h2_remaining_budget.json',90.,'budgeted-admission-control',native=True,begun=BEGUN):
        result=control();c.write(c.OUT/'f4_h2_budgeted_admission_control_20261007.json',result)
    print(json.dumps(result),flush=True)
