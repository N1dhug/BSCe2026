"""One stdlib check of G-only price/quad admission; no model propagation."""
import argparse
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time


def check(root, frozen, adapter_root):
    start=time.perf_counter();n=0
    def require(value):
        nonlocal n
        if not value: raise ValueError('Design admission check failed at '+str(n+1))
        n+=1
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    d=json.loads((root/'bounded_price_quadrature_design_20261007.json').read_text())
    prior=json.loads((root/'receipt_recovery_prior_budget_20261007.json').read_text())
    budget=json.loads((root/'price_design_budget_20261007.json').read_text())
    c=d['campaign'];p=d['proposal'];x=d['problem']
    require(prior['campaign_debit']==c['prior_debit_seconds']==budget['prior_debit'])
    require(c['debit_after_design_seconds']==prior['campaign_debit']+10==budget['campaign_debit'])
    require(c['remaining_after_design_seconds']==1800-budget['campaign_debit']==budget['remaining_seconds'])
    require(p['scope_ceiling_seconds']==p['worker_hard_timeout_seconds']+p['stop_export_reserve_seconds']+p['raw_review_reserve_seconds']==320)
    require(sum(j['count']*j['reservation_each_seconds'] for j in p['jobs'])==p['total_job_reservations_seconds']==260)
    require(p['total_job_reservations_seconds']+p['worker_unallocated_headroom_seconds']==270)
    require(p['scope_ceiling_seconds']<=budget['remaining_seconds'])
    require(p['remaining_after_maximum_scope_seconds']==budget['remaining_seconds']-320>0)
    known=set()
    for j in p['jobs']:
        require(j['id'] not in known and set(j['depends_on'])<=known)
        require(j['count']>0 and j['reservation_each_seconds']>0)
        known.add(j['id'])
    require(sum(j['new_forwards'] for j in p['jobs'])==p['maximum_new_forwards']==25)
    require(x['horizon']==[12,14] and x['price_edges']==[12,13,14] and x['nodes']==5)
    require(x['candidate_deltas']==[0,-.2,.2] and x['new_non_neutral_candidates']==2)
    require(x['candidate_prices']==[[.5,.5],[.3,.7],[.7,.3]])
    require(all(0<=u<=1 for row in x['candidate_prices'] for u in row))
    require(all(sum(row)==1 for row in x['candidate_prices']))
    require(x['equal_mean_price_is_not_energy_equality'] and not x['stochastic_energy_equality_added'])
    require(x['fine_solver_step_hours']==x['quadrature_solver_step_hours']==1/192)
    require(x['objective_step_hours']==1/96 and x['quadrature_step_hours']==1/192)
    # New midpoint states are required: no old-point averaging closes the gate.
    old={Fraction(12)+Fraction(2*i+1,192) for i in range(192)}
    new={Fraction(12)+Fraction(2*i+1,384) for i in range(384)}
    require(len(old)==192 and len(new)==384 and old.isdisjoint(new))
    require(x['fine_cells']==513 and x['coarse_cells']==257 and x['own_grid_origins'])
    require(d['selection']['only_coarse_data_used'] and not d['selection']['fine_or_fv_reselection'])
    require(not p['reservations_are_measured_bounds'] and p['unknown_policy_and_quadrature_runtime'])
    for flag in ('launch_ready','pre_run_registration_complete','numerical_run_performed',
                 'actual_optimizer_started','scientific_accept','full_day_budget_admitted'):
        require(d[flag] is False)
    manifest_path=root/'price_design_source_manifest_20261007.json'
    require(sha(manifest_path)==d['provenance']['native_source_manifest_sha256'])
    sources=json.loads(manifest_path.read_text());require(len(sources)==41)
    for path,value in sources.items():require(sha(frozen/path)==value)
    require(sha(adapter_root/'receipt_recovery.py')==d['provenance']['receipt_recovery_source_sha256'])
    require(sha(adapter_root/'restore_auxiliary.py')==d['provenance']['restore_auxiliary_source_sha256'])
    require(sha(adapter_root/'receipt_adoption_manifest_20261007.json')==d['provenance']['receipt_adoption_manifest_sha256'])
    tree=ast.parse((frozen/'src/exe/run_model/bm_1/g8.py').read_text())
    call=next(node.value for node in tree.body if isinstance(node,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='TOLERANCES' for t in node.targets))
    actual={kw.arg:ast.literal_eval(kw.value) for kw in call.keywords}
    gates=d['acceptance']['tolerances']
    for key in ('state_mean','demand_mean','state_variance','demand_variance','state_cdf',
                'demand_cdf','state_quantile','demand_quantile','mass','checkpoint'):
        require(gates[key]==actual[key])
    require(gates['conditional_mass_l1']==gates['marginal_mass_l1']==actual['posterior_mass_l1'])
    require(d['acceptance']['projection_cell_mass_l1']==actual['projection_l1'])
    require(not d['acceptance']['parameter_space_global_optimality_claim']
            and not d['acceptance']['policy_scientific_accept'])
    merge=json.loads((root/'pr112_113_merge_verification_20261007.json').read_text())
    require(merge['merged_changed_blobs_match_reviewed_head'] and merge['native_shared_tests_workflows_unchanged'])
    require(not any(name in sys.modules for name in ('numpy','scipy','src.model.bm_1')))
    return dict(schema='g7-price-design-check-v1',passed=True,checks=n,
                elapsed_seconds=time.perf_counter()-start,native_imports=0,new_forwards=0,
                old_midpoint_count=192,new_midpoint_count=384,new_midpoints_are_distinct=True,
                budget_design_admitted=True,launch_ready=False,scientific_accept=False,
                full_day_budget_admitted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--frozen-source',type=Path,required=True)
    p.add_argument('--adapter-root',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();r=check(a.root,a.frozen_source,a.adapter_root)
    a.report.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
