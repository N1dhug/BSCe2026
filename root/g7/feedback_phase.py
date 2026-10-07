"""Actual G feedback composition; full-day search/EM driver is a separate stage."""
import json
import numpy as np
from src.control.bm_1 import BM1ControlNumerics
from src.method.bm_1_distribution_panel import baseline, target
from src.method.bm_1_g7 import OwnedBankPriceEvaluator
from src.method.bm_1_g8 import G8BankOrigin
from src.method.bm_1_g7_jobs import atomic_json, digest
from audited_feedback import FeedbackBankControl

CONFIGS = dict(m254_coarse=('mixture',257,1/96),
               m254_fine=('mixture',513,1/192),
               fv_coarse=('fv',257,1/96),
               fv_fine=('fv',513,1/192))


def run_feedback_phase(packet, store, *, source_root, source_sha256,
                       reservations, output):
    """Use four completed audited forwards, one y, then three fresh candidates."""
    if (packet.get('schema') != 'g7-audited-feedback-input-v2'
            or set(packet['forwards']) != set(CONFIGS)
            or packet['price_candidates'] != [[.5,.5],[.3,.7],[.7,.3]]):
        raise ValueError('Registered four-backend feedback packet required.')
    observation = packet['observation']
    if (observation.get('kind') != 'point' or observation.get('time') != 36
            or not np.isfinite(observation['value'])):
        raise ValueError('One finite synthetic endpoint point-observation at36 required.')
    controls = {}; results = {}; prepared = {}
    # All four input contracts are checked before any new likelihood/forward.
    for name, (backend, cells, step) in CONFIGS.items():
        saved = packet['forwards'][name]
        origin = G8BankOrigin.from_record(saved['origin'])
        if (origin.time != 12 or len(origin.rho_values) != 5
                or origin.conditional_mass.shape[1] != cells
                or saved['prices'] != packet['applied_prices']
                or len(saved['prices']) != 4):
            raise ValueError('Forward origin/grid/applied policy differs.')
        e = OwnedBankPriceEvaluator(origin, baseline=baseline, target=target,
            horizon=(12.,36.), price_edges=[12.,18.,24.,30.,36.],
            numerics=BM1ControlNumerics(x_steps=cells, objective_max_step=1/96,
                                      solver_max_step=step), backend=backend)
        control = FeedbackBankControl(e, store, source_root=source_root,
                                      source_sha256=source_sha256)
        if control.contract_id != saved['contract_id']:
            raise ValueError('Adopted 24h-forward contract/source/audit policy differs.')
        receipts = saved['receipts']
        if len(receipts) != 5:
            raise ValueError('Five completed audited receipts required.')
        for j, item in enumerate(receipts):
            expected = dict(contract=control.contract_id, index=j,
                            rho=float(origin.rho_values[j]), prices=saved['prices'])
            if (item.get('job') != expected or item.get('schema') != 'g7-owned-node-v1'
                    or digest(item.get('payload')) != item.get('payload_sha256')):
                raise ValueError('Adopted full-day node identity changed.')
            from audited_node_control import validate_audit
            saved_audit = validate_audit(item['payload'], index=j, backend=backend)
            if (backend == 'mixture' and saved_audit['record']['maximum_projection_cell_mass_l1']
                    > packet['projection_l1_limit']):
                raise ValueError('Adopted forward fails registered projection-fit gate.')
        result = e.combine_nodes(saved['prices'], [r['payload'] for r in receipts])
        controls[name] = control; prepared[name] = (result, receipts)
    next_controls = {}
    for name, (backend, cells, step) in CONFIGS.items():
        result, receipts = prepared[name]
        update = controls[name].assimilate(result, observation['value'],
            seconds=reservations['feedback_update'], forward_receipts=receipts)
        projection = update['receipt']['posterior_projections']
        if any(p['cell_mass_l1_error'] > packet['projection_l1_limit'] for p in projection):
            raise ValueError('Updated posterior fails unchanged projection-fit gate.')
        results[name] = update
        next_controls[name] = controls[name].continuation(update, horizon=(36.,38.),
            price_edges=[36.,37.,38.],
            numerics=BM1ControlNumerics(x_steps=cells, objective_max_step=1/96,
                                      solver_max_step=step))
        atomic_json(output / ('posterior_'+name+'.json'),
                    dict(job_id=update['job_id'], origin=update['origin'].to_record()))
    candidates = packet['price_candidates']
    costs = []
    for prices in candidates:
        value = next_controls['m254_coarse'].evaluate(prices,
            seconds_per_node=reservations['feedback_policy_search']/5.)
        costs.append(float(value['total']))
    selected = min(range(len(costs)), key=lambda k: (costs[k], k))
    atomic_json(output / 'feedback_selection.json',
        dict(prices=candidates, costs=costs, selected=selected, tie_rule='first candidate',
             posterior_job=results['m254_coarse']['job_id']))
    forwards = {}
    for name, control in next_controls.items():
        # Coarse chosen result is already durable and is reused by exact identity.
        value = control.evaluate(candidates[selected],
            seconds_per_node=reservations['feedback_'+name]/5.)
        forwards[name] = dict(J=float(value['total']),
            origin=value['predicted_origin'].to_record(),
            weights=value['weights'].tolist(), prices=value['prices'].tolist())
    report = dict(schema='g7-feedback-phase-result-v2', feedback_phase_completed=True,
        initial_time=12, observation_time=36, next_time=38,
        applied_prices=packet['applied_prices'], observation=observation,
        posterior_jobs={k:v['job_id'] for k,v in results.items()},
        candidates=candidates, costs=costs, selected=selected, forwards=forwards,
        numerical_validation_passed=False, scientific_accept=False,
        full_day_experiment_completed=False,
        note='Update audit/continuation complete; common cross-backend/EM review is still required.')
    atomic_json(output / 'feedback_result.json', report)
    return report
