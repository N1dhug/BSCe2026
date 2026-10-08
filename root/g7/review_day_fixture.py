"""Read-only completion review of a saved G day fixture; no evolution."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from audited_feedback import validate_update
from day_review import paired_improvement
from integrated_budget import price_plan, digest as standard_digest
from receipt_recovery import guarded_store_class, audit_store
from src.method.bm_1_g7_jobs import NodeJobStore, digest
from src.method.bm_1_g8 import G8BankOrigin


def main(run, output, source):
    start=perf_counter();run=Path(run);output=Path(output);source=Path(source)
    registration=json.loads((output/'registration.json').read_bytes())
    manifest=json.loads((run/'source_manifest.json').read_bytes())
    assert all(hashlib.sha256((source/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    result=json.loads((run/'day_result.json').read_bytes())
    assert result['numerical_validation_passed'] and not result['scientific_accept']
    assert set(result['costs'])=={'neutral','deterministic','stochastic'}
    assert all(set(v)=={'m254_coarse','m254_fine','fv_coarse','fv_fine','quadrature'} for v in result['costs'].values())
    assert all(c['passed'] for c in result['feedback_checks'].values())
    assert all(c['passed'] for c in result['continuation_checks'].values())
    assert result['feedback']['selected']==1
    packet=json.loads((run/'generated_feedback_inputs.json').read_bytes())
    Guard=guarded_store_class(NodeJobStore,native_source_path=source/'src/method/bm_1_g7_jobs.py',native_source_sha256=manifest['src/method/bm_1_g7_jobs.py'])
    store=Guard(run/'fixture_campaign',prior_debit=0,ceiling=300,max_new_seconds=300)
    before={p.name:p.read_bytes() for p in (store.root/'nodes').glob('*.json')}
    ledger_before=store.ledger_path.read_bytes();updates=0;em=[];reuse=0
    def forbidden(deadline):
        raise AssertionError('Completed receipt triggered compute')
    with store.lock():
        audit_store(store,full=True)
        for name,raw in before.items():
            row=json.loads(raw);payload=store.node(row['job'],1,forbidden)
            assert digest(payload)==row['payload_sha256'];reuse+=1
            if row['job'].get('kind')=='g7-owned-point-update-v2':
                posterior=validate_update(payload,row['job']);prior=G8BankOrigin.from_record(payload['predicted_origin'])
                assert not np.array_equal(posterior.log_weights,prior.log_weights)
                if posterior.backend=='mixture':assert not np.array_equal(posterior.conditional_coefficients,prior.conditional_coefficients)
                updates+=1
            if row['job'].get('kind')=='g7-day-coupled-em-v1' and row['job']['prices']==result['policies']['neutral']:em.append(payload)
        assert updates==4 and len(em)==5
        assert store.ledger_path.read_bytes()==ledger_before
        assert all((store.root/'nodes'/name).read_bytes()==raw for name,raw in before.items())
    em.sort(key=lambda v:v['index'])
    assert paired_improvement(em,em,np.full(5,.2))['mean']==0
    changed=copy.deepcopy(em);changed[0]['input_sha256']='0'*64
    try:paired_improvement(em,changed,np.full(5,.2))
    except ValueError:pass
    else:raise AssertionError('Unpaired input accepted')
    plan=json.loads((source/'root/g7/integration_design.json').read_bytes())
    budget=price_plan(plan,{},prior_debit=registration['prior_debit'],ceiling=registration['campaign_ceiling'],source_fingerprint=standard_digest(manifest),origin_fingerprint='fixture-only',versions={'numpy':np.__version__,'scipy':__import__('scipy').__version__},reserve_seconds=2)
    assert not budget['budget_admitted'] and budget['estimated_seconds'] is None and len(budget['unknown'])==16
    assert all(hashlib.sha256((source/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    report=dict(schema='g7-day-fixture-completion-review-v1',passed=True,fixture_graph_completed=True,actual_full_day_experiment_completed=False,scientific_accept=False,numerical_scientific_validation=False,completed_receipts_reused_without_compute=reuse,posterior_updates_validated=updates,raw_bytes_and_ledger_unchanged=True,unpaired_mc_rejected=True,identical_mc_pair_has_zero_difference=True,unknown_scientific_rates=budget['unknown'],full_day_budget_admitted=False,body_wall_seconds=perf_counter()-start,source_hashes_unchanged=True,note='Review of completed stub graph after original auxiliary reuse assertion used a different exact contract. No scientific kernel executed; no failed parent result overwritten.')
    (output/'review_result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main(*sys.argv[1:])
