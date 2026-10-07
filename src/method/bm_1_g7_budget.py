"""Complete G7 control job pricing. Missing measured rates never become zero."""
import numpy as np

# Conservative full jobs; cache reuse requires a separate raw identity audit.
WORK_COUNTS = {
    'setup':1, 'search_deterministic':48, 'search_fv':48,
    'cross_m254_coarse':3, 'cross_m254_fine':3,
    'cross_fv_coarse':3, 'cross_fv_fine':3,
    'quadrature':3, 'em_coupled':3,
    'feedback_m254_coarse':1, 'feedback_m254_fine':1,
    'feedback_fv_coarse':1, 'feedback_fv_fine':1,
    'audit_export':1,
}


def full_job_plan(*,horizon_hours=24.,nodes=5,safety_factor=1.25):
    if horizon_hours!=24. or nodes!=5 or not np.isfinite(safety_factor) or safety_factor<1.:
        raise ValueError('Registered full-day plan requires 24h, five nodes and reserve factor >=1.')
    return dict(schema='g7-complete-control-job-plan-v1',horizon_hours=24.,nodes=5,
        policies=['neutral','deterministic','stochastic'],safety_factor=float(safety_factor),
        jobs=[dict(work=k,count=v,rate_unit='whole job',nodes=1 if k=='search_deterministic' else nodes)
              for k,v in WORK_COUNTS.items()],
        monte_carlo=dict(paths_per_node=256,levels=['coupled coarse','fine']),
        target='original fixed A=.5; stochastic full within/between variance',
        energy_neutral_extension=False,
        scientific_accept=False,actual_optimizer_started=False)


def price_job_plan(plan,rates,*,campaign_debit,ceiling=1800.):
    """A budget predicate over externally raw-reviewed full-job receipts.

    There is no implicit 2h->24h conversion. Safety factor is a planning
    reserve, not a numerical or worst-case runtime certificate.
    """
    expected=full_job_plan(safety_factor=plan['safety_factor'])
    if plan!=expected: raise ValueError('Full job graph differs; re-register explicitly before changing it.')
    if not np.isfinite([campaign_debit,ceiling]).all() or not 0<=campaign_debit<=ceiling:
        raise ValueError('Preserved finite campaign ledger required.')
    unknown=[];rows=[]
    for job in plan['jobs']:
        work=job['work'];rate=rates.get(work)
        if rate is None:
            unknown.append(work);rows.append(dict(job,seconds=None));continue
        seconds=rate.get('seconds')
        if (seconds is None or not rate.get('complete',False) or not rate.get('raw_reviewed',False)
                or rate.get('horizon_hours')!=plan['horizon_hours'] or rate.get('nodes')!=job['nodes']):
            unknown.append(work);rows.append(dict(job,seconds=None));continue
        if not np.isfinite(seconds) or seconds<=0 or not rate.get('receipt_sha256') or not rate.get('source_fingerprint'):
            raise ValueError('Positive measured seconds and explicit raw/source provenance required.')
        rows.append(dict(job,seconds=float(seconds),estimated_seconds=job['count']*float(seconds)*plan['safety_factor']))
    total=None if unknown else sum(row['estimated_seconds'] for row in rows)
    return dict(schema='g7-complete-job-admission-v1',rows=rows,unknown=unknown,
                estimated_seconds=total,campaign_debit=float(campaign_debit),
                remaining_seconds=float(ceiling-campaign_debit),
                budget_admitted=not unknown and total<=ceiling-campaign_debit,
                scientific_accept=False,actual_optimizer_started=False,
                note='Measured representative rates plus explicit reserve are not a worst-case timing bound.')
