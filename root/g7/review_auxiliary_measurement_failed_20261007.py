"""Read-only review of stored auxiliary receipts; never import a model solver."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from restore_auxiliary import canonical, sha, verified_json, verified_receipt


def review(first, resumed, original):
    start = time.perf_counter()
    reg = json.loads((resumed/'registration.json').read_text())
    report = json.loads((resumed/'worker_summary.json').read_text())
    invocation = json.loads((resumed/'invocation.json').read_text())
    previous = verified_json(first/'invocation.json', reg['continuation']['invocation_sha256'])
    origin = json.loads((original/'inputs/fine.json').read_text())
    w = np.exp(origin['log_weights']); w /= w.sum()
    counts = {}; receipts = {}
    for root in (first, resumed):
        ledger = json.loads((root/'campaign/ledger.json').read_text())
        assert len({e['job_id'] for e in ledger['entries']}) == len(ledger['entries'])
        n = 0
        for e in ledger['entries']:
            p = root/'campaign/nodes'/(e['job_id']+'.json')
            if e['status'] == 'failed':
                assert root == first and e['job_id'] == reg['continuation']['failed_em_job_id'] and not p.exists()
                continue
            assert e['status'] == 'complete' and e['attempt'] == 1
            r = json.loads(p.read_text())
            assert hashlib.sha256(canonical(r['job'])).hexdigest() == e['job_id']
            v = verified_receipt(p, sha(p), r['job']); n += 1
            if root == resumed:
                assert r['job']['registration_sha256'] == sha(resumed/'registration.json')
                receipts[(r['job']['kind'], r['job']['index'])] = v
        counts[root.name] = n
    setup = receipts[('restore_setup', 0)]['contract']
    mid = np.asarray(setup['midpoints']); widths = np.asarray(setup['widths'])
    target = np.asarray(setup['per_node'][0]['target'])
    costs = []; maximum_path_cost_error = 0.; violations = 0
    for j in range(5):
        p = receipts[('coupled_em_nodes', j)]
        times = np.asarray(p['fine_time']); jcost = []
        for level, item in enumerate(p['levels']):
            t = times[::2] if level == 0 else times
            indices = np.searchsorted(t, mid)
            assert np.allclose(t[indices], mid, rtol=0., atol=1e-12)
            state = np.asarray(item['state']); demand = np.asarray(item['demand'])
            assert state.shape == demand.shape == (256, len(t))
            assert np.isfinite(state).all() and np.isfinite(demand).all()
            assert state.min() >= 0. and state.max() <= 1.
            assert item['diagnostics']['requested_max_step'] == 'Infinity'
            violations += item['diagnostics']['total_violations']
            q = ((demand[:, indices]-target)**2)@widths
            err = float(np.max(abs(q-np.asarray(item['path_costs']))))
            assert err < 1e-12
            maximum_path_cost_error = max(maximum_path_cost_error, err); jcost.append(q)
        costs.append(jcost)
    assert violations == 0
    costs = np.asarray(costs)
    means = np.mean(costs, axis=2).T@w
    se_diagonal = float(np.sqrt(np.sum(w*w*np.var(costs[:, 1], axis=1, ddof=1)/256)))
    se_covariance_envelope = float(w@(np.std(costs[:, 1], axis=1, ddof=1)/16))
    assert np.max(abs(means-[report['em']['coarse_J'],report['em']['fine_J']])) < 1e-12
    assert abs(se_diagonal-report['em']['stratified_SE']) < 1e-12
    for name in ('m254_fine','m254_coarse','fv_fine','fv_coarse'):
        p = receipts[('feedback_'+name, -1)]
        logz = np.asarray(p['conditional_log_evidence'])
        a = np.asarray(origin['log_weights'])+logz; z = a.max()+np.log(np.exp(a-a.max()).sum())
        assert abs(z-p['log_evidence']) < 1e-12
        assert np.max(abs(a-z-np.asarray(p['origin']['log_weights']))) < 1e-12
        assert p['missing_update_preserved']
        mass = np.asarray(p['origin']['conditional_mass'])
        assert mass.min() >= 0 and np.max(abs(mass.sum(axis=1)-1.)) < 1e-9
    assert invocation['scope_charge'] == invocation['native_charge']+invocation['export_seconds']+20
    assert invocation['campaign_debit'] == previous['campaign_debit']+invocation['scope_charge']
    assert previous['scope_charge']+invocation['scope_charge'] <= 200
    assert reg['prior_debit'] == previous['campaign_debit']
    spec = json.loads((resumed/'design.json').read_text())['proposed_measurement']['em']
    initial = [spec['initial_seed']+j for j in range(5)]
    brownian = [spec['brownian_seed']+j for j in range(5)]
    overlaps = sorted(set(initial)&set(brownian))
    assert overlaps == [20261022,20261023,20261024,20261025]
    files = {str(p.relative_to(resumed)):dict(sha256=sha(p),bytes=p.stat().st_size)
             for p in sorted(resumed.rglob('*')) if p.is_file() and p.name != 'writer.lock'}
    return dict(schema='g7-auxiliary-raw-review-v1',raw_integrity_passed=True,
        completed_receipts=counts,maximum_recomputed_path_cost_error=maximum_path_cost_error,
        em_J_coarse=float(means[0]),em_J_fine=float(means[1]),raw_boundary_violations=violations,
        diagonal_SE=se_diagonal,covariance_envelope_plugin_SE=se_covariance_envelope,
        overlapping_initial_brownian_seeds=overlaps,noise_seed=spec['noise_seed'],
        noise_seed_overlaps_selected_truth_initial=spec['noise_seed']==initial[2],
        independent_mc_standard_error_verified=False,independent_synthetic_noise_verified=False,
        covariance_envelope_is_certified_interval=False,
        interpretation='Registered numeric gates pass; MC SE and generative independent-noise claims withheld. Fixed shared endpoint supports numerical backend comparison only.',
        conditional_numerical_comparison_pass=report['auxiliary_gates_passed'],
        new_model_forwards=0,scientific_accept=False,full_day_budget_admitted=False,
        execution_commit=(resumed/'execution_source.txt').read_text().strip(),
        first_scope_charge=previous['scope_charge'],continuation_scope_charge=invocation['scope_charge'],
        combined_scope_charge=previous['scope_charge']+invocation['scope_charge'],
        campaign_debit=invocation['campaign_debit'],remaining_seconds=invocation['remaining_seconds'],
        files=files,review_seconds=time.perf_counter()-start)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('first','resumed','original','report'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    result=review(args.first,args.resumed,args.original)
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
