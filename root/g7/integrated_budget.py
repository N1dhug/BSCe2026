"""Stdlib G7 v2 whole-graph budget: unknown/incompatible rates stay unknown."""
import hashlib
import json
import math

COUNTS = dict(setup=1, search_deterministic=48, search_m254=48,
    cross_m254_coarse=3, cross_m254_fine=3, cross_fv_coarse=3,
    cross_fv_fine=3, quadrature=3, em_coupled=3, feedback_update=4,
    feedback_policy_search=3, feedback_m254_coarse=1, feedback_m254_fine=1,
    feedback_fv_coarse=1, feedback_fv_fine=1, audit_export=1)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def finite(value, name, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError('Invalid finite ' + name)
    return float(value)


def validate_plan(plan):
    if (plan.get('schema') != 'g7-integrated-control-proposal-v2'
            or {r['work']: r['count'] for r in plan['jobs']} != COUNTS
            or len(plan['jobs']) != len(COUNTS)):
        raise ValueError('Different v2 job graph; explicitly revise/register it.')
    p = plan['problem']; o = plan['optimizer']; f = plan['feedback']
    if (p['initial_time_hours'] != 12 or p['horizon_hours'] != 24
            or p['price_edges'] != [12,18,24,30,36] or p['price_bounds'] != [0,1]
            or p['nodes'] != 5 or p['terminal_weight'] or p['smoothing_weight']
            or p['energy_equality_added']
            or o['max_calls_total'] != 96 or o['max_calls_per_policy'] != 48
            or o['max_calls_per_start'] != 24
            or o['starts'] != [[.5]*4,[.3,.7,.3,.7]]
            or o['primary_stochastic_backend'] != 'mixture M254257'
            or o['fine_fv_reselection']
            or f['observation_time_hours'] != 36 or f['next_edges'] != [36,37,38]
            or f['price_candidates'] != [[.5,.5],[.3,.7],[.7,.3]]):
        raise ValueError('Frozen horizon/price/search/feedback contract changed.')
    for row in plan['jobs']:
        expected_horizon = (2 if row['work'].startswith('feedback_') and row['work'] != 'feedback_update'
                            else 24 if row['work'] not in ('setup','audit_export','feedback_update') else None)
        if row['horizon_hours'] != expected_horizon:
            raise ValueError('Whole-job rate has a different horizon.')


def price_plan(plan, rates, *, prior_debit, ceiling, source_fingerprint,
               origin_fingerprint, versions, reserve_seconds):
    validate_plan(plan)
    prior = finite(prior_debit, 'prior debit'); cap = finite(ceiling, 'ceiling', True)
    reserve = finite(reserve_seconds, 'parent stop/export/review reserve', True)
    safety = finite(plan['budget']['planning_safety_factor'], 'safety factor', True)
    if prior > cap or safety < 1 or not source_fingerprint or not origin_fingerprint or not versions:
        raise ValueError('Preserved budget and explicit source/origin/runtime identities required.')
    context = digest(dict(problem=plan['problem'], policies=plan['policies'],
        optimizer=plan['optimizer'], numerics=plan['numerics'], feedback=plan['feedback'],
        tolerances=plan['tolerances'], source_fingerprint=source_fingerprint,
        origin_fingerprint=origin_fingerprint, versions=versions))
    rows = []; unknown = []
    if set(rates) - set(COUNTS):
        raise ValueError('Unknown work type in rates.')
    for job in plan['jobs']:
        work = job['work']; rate = rates.get(work)
        valid = (isinstance(rate, dict) and rate.get('complete') is True
            and rate.get('raw_reviewed') is True and rate.get('context_sha256') == context
            and rate.get('horizon_hours') == job['horizon_hours']
            and rate.get('rate_unit') == job['rate_unit']
            and isinstance(rate.get('receipt_sha256'), str) and len(rate['receipt_sha256']) == 64
            and all(c in '0123456789abcdef' for c in rate['receipt_sha256']))
        if not valid:
            unknown.append(work); rows.append(dict(job, seconds=None)); continue
        measured = finite(rate.get('seconds'), work + ' measured rate', True)
        rows.append(dict(job, seconds=measured, reservation_seconds=safety * measured,
                         estimated_seconds=job['count'] * safety * measured,
                         rate_receipt_sha256=rate['receipt_sha256']))
    estimate = None if unknown else reserve + sum(r['estimated_seconds'] for r in rows)
    optimizer = None if unknown else sum(r['estimated_seconds'] for r in rows
        if r['work'] in ('search_deterministic','search_m254'))
    # Original plan retains a 300s common optimizer cap and 300s validation cap.
    validation = None if unknown else sum(r['estimated_seconds'] for r in rows
        if r['work'] not in ('setup','audit_export','search_deterministic','search_m254'))
    admitted = (not unknown and estimate <= cap-prior and optimizer <= 300 and validation <= 300)
    return dict(schema='g7-integrated-admission-v2', context_sha256=context,
        plan_sha256=digest(plan), rows=rows, unknown=unknown,
        prior_debit=prior, ceiling=cap, remaining_seconds=cap-prior,
        reserve_seconds=reserve, estimated_seconds=estimate,
        optimizer_estimated_seconds=optimizer, validation_estimated_seconds=validation,
        budget_admitted=admitted, scientific_accept=False,
        note='Measured representative rates plus reserve are not a worst-case bound.')
