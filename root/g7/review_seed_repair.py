"""Read-only seed repair review: reproduce stored random inputs, never propagate a model."""
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
    counts = {}; receipts = {}; journal_anomalies=[]; reconciled=[]
    for root in (first, resumed):
        ledger = json.loads((root/'campaign/ledger.json').read_text())
        assert len({e['job_id'] for e in ledger['entries']}) == len(ledger['entries'])
        n=0;entry_by_id={e['job_id']:e for e in ledger['entries']}
        for e in ledger['entries']:
            if e['status']=='failed':
                assert root==first and e['job_id']==reg['continuation']['failed_em_job_id'] and not (root/'campaign/nodes'/(e['job_id']+'.json')).exists()
        for p in sorted((root/'campaign/nodes').glob('*.json')):
            r=json.loads(p.read_text());key=hashlib.sha256(canonical(r['job'])).hexdigest()
            assert p.name==key+'.json'
            v=verified_receipt(p,sha(p),r['job']);e=entry_by_id.get(key)
            assert e is None or e['attempt']==1
            if root==first:
                assert e is not None and e['status']=='complete'
            else:
                assert r['job']['registration_sha256']==sha(resumed/'registration.json')
                kind,index=r['job']['kind'],r['job']['index']
                stage=next(x for x in report['stages'][kind] if x['index']==index)
                assert stage['job_id']==key
                assert e is None or e['status'] in ('running','complete')
                cap=reg['job_caps'][kind]
                if e is None or e['status']!='complete':
                    assert kind in ('fv_fine_nodes','fv_coarse_nodes')
                    journal_anomalies.append(dict(job_id=key,kind=kind,index=index,status='missing' if e is None else e['status']))
                charge=max(stage['wall_seconds'],cap if e is None or e['status']=='running' else e['charged_seconds'])
                reconciled.append(dict(job_id=key,kind=kind,index=index,receipt_sha256=sha(p),payload_sha256=r['payload_sha256'],status='verified-complete-receipt',original_entry=e,conservative_charged_seconds=charge))
                receipts[(kind,index)]=v
            n+=1
        assert n==14 if root==first else n==24
        counts[root.name]=n
    assert sorted((a['kind'],a['index'],a['status']) for a in journal_anomalies)==[('fv_coarse_nodes',0,'missing'),('fv_coarse_nodes',1,'missing'),('fv_fine_nodes',4,'running')]
    reconciled_increment=sum(e['conservative_charged_seconds'] for e in reconciled)
    assert reconciled_increment<=invocation['native_charge']
    setup = receipts[('restore_setup', 0)]['contract']
    mid = np.asarray(setup['midpoints']); widths = np.asarray(setup['widths'])
    target = np.asarray(setup['per_node'][0]['target'])
    costs = []; maximum_path_cost_error = 0.; violations = 0
    for j in range(5):
        p = receipts[('coupled_em_nodes', j)]
        times = np.asarray(p['fine_time']); jcost = []
        for level, item in enumerate(p['levels']):
            t = times[::2] if level == 0 else times
            indices = np.rint((mid-t[0])/(t[1]-t[0])).astype(int)
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
        setup_index = ('m254_fine','m254_coarse','fv_fine','fv_coarse').index(name)
        own_prior = receipts[('restore_setup', setup_index)]['contract']['origin']['log_weights']
        logz = np.asarray(p['conditional_log_evidence'])
        a = np.asarray(own_prior)+logz; z = a.max()+np.log(np.exp(a-a.max()).sum())
        assert abs(z-p['log_evidence']) < 1e-12
        assert np.max(abs(a-z-np.asarray(p['origin']['log_weights']))) < 1e-12
        assert p['missing_update_preserved']
        mass = np.asarray(p['origin']['conditional_mass'])
        assert mass.min() >= 0 and np.max(abs(mass.sum(axis=1)-1.)) < 1e-9
    assert invocation['scope_charge'] == invocation['native_charge']+invocation['export_seconds']+20
    budget=verified_json(resumed/'previous_final_budget.json',reg['previous_final_budget_sha256'])
    assert invocation['campaign_debit'] == budget['campaign_debit']+invocation['scope_charge']
    assert invocation['scope_charge'] <= reg['scope_ceiling']==120
    assert reg['prior_debit'] == budget['campaign_debit']
    spec = json.loads((resumed/'design.json').read_text())['proposed_measurement']['em']
    initial=spec['initial_seeds'];brownian=spec['brownian_seeds']
    overlaps=sorted(set(initial)&set(brownian))
    assert len(set(initial+brownian+[spec['noise_seed']]))==11 and not overlaps
    expected_initial=[int(np.random.SeedSequence(20261007,spawn_key=(0,j)).generate_state(1,dtype=np.uint64)[0]) for j in range(5)]
    expected_brownian=[int(np.random.SeedSequence(20261007,spawn_key=(1,j)).generate_state(1,dtype=np.uint64)[0]) for j in range(5)]
    expected_noise=int(np.random.SeedSequence(20261007,spawn_key=(2,0)).generate_state(1,dtype=np.uint64)[0])
    assert (initial,brownian,spec['noise_seed'])==(expected_initial,expected_brownian,expected_noise)
    initial_coefficients=np.asarray(origin['conditional_coefficients'])
    for j in range(5):
        p=receipts[('coupled_em_nodes',j)]
        rng=np.random.default_rng(initial[j])
        component=rng.choice(255,256,p=initial_coefficients[j])
        x=rng.beta(component+1.,255.-component)
        assert np.array_equal(component,p['component']) and np.array_equal(x,p['initial'])
        t=np.asarray(p['fine_time'])
        dw=np.random.default_rng(brownian[j]).standard_normal((len(t)-1,256)).T*np.sqrt(np.diff(t))[None,:]
        assert np.array_equal(dw,np.asarray(p['fine_increments']))
        assert np.array_equal(x,np.asarray(p['levels'][0]['state'])[:,0])
        assert np.array_equal(x,np.asarray(p['levels'][1]['state'])[:,0])
    measurement=json.loads((resumed/'measurement.json').read_text())
    truth=float(receipts[('coupled_em_nodes',2)]['levels'][1]['demand'][0][-1])
    noise=float(np.random.default_rng(spec['noise_seed']).normal(0.,origin['sigma_y']))
    assert measurement['truth']==truth and measurement['noise']==noise and measurement['value']==truth+noise
    for name in ('m254_fine','m254_coarse','fv_fine','fv_coarse'):
        receipt_id=next(e['job_id'] for e in json.loads((resumed/'campaign/ledger.json').read_text())['entries'] if e['node']==-1 and json.loads((resumed/'campaign/nodes'/(e['job_id']+'.json')).read_text())['job']['kind']=='feedback_'+name)
        assert json.loads((resumed/'campaign/nodes'/(receipt_id+'.json')).read_text())['job']['value']==measurement['value']
    files = {str(p.relative_to(resumed)):dict(sha256=sha(p),bytes=p.stat().st_size)
             for p in sorted(resumed.rglob('*')) if p.is_file() and p.name != 'writer.lock'}
    return dict(schema='g7-seed-repair-raw-review-v1',raw_integrity_passed=True,
        completed_receipts=counts,maximum_recomputed_path_cost_error=maximum_path_cost_error,
        journal_integrity_passed=False,journal_anomalies=journal_anomalies,
        receipt_reconciliation_verified=True,reconciled_increment=reconciled_increment,
        parent_wall_covers_reconciled_charges=True,original_ledger_unchanged=True,
        ledger_reconciliation=reconciled,
        em_J_coarse=float(means[0]),em_J_fine=float(means[1]),raw_boundary_violations=violations,
        diagonal_SE=se_diagonal,covariance_envelope_plugin_SE=se_covariance_envelope,
        overlapping_initial_brownian_seeds=overlaps,noise_seed=spec['noise_seed'],
        noise_seed_overlaps_selected_truth_initial=False,
        independent_mc_standard_error_verified=True,independent_synthetic_noise_verified=True,
        rng_interpretation='Verified distinct role/node PCG64 streams and exact input replay under standard pseudorandom independence assumption; not a proof of true-random independence.',
        raw_random_input_replay_verified=True,
        covariance_envelope_is_certified_interval=False,
        interpretation='Seed-role collision repaired and raw random inputs reproduced. Registered short synthetic numerical comparison only; not full-day, historical, optimized-policy or quadrature acceptance.',
        conditional_numerical_comparison_pass=report['auxiliary_gates_passed'],
        new_model_forwards=0,scientific_accept=False,full_day_budget_admitted=False,
        execution_commit=(resumed/'execution_source.txt').read_text().strip(),
        first_scope_charge=previous['scope_charge'],continuation_scope_charge=invocation['scope_charge'],
        previous_auxiliary_task_charge=budget['combined_task_charge'],new_seed_repair_scope_charge=invocation['scope_charge'],
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
