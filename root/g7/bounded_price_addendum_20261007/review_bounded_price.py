"""Read-only raw-cost/journal review; no numerical imports or propagation."""
import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from launch_bounded_price import validate, write
from receipt_recovery import audit_store, digest, sha


def review(out):
    started=time.perf_counter()
    reg,plan,caps=validate(out)
    ledger=json.loads((out/'campaign/ledger.json').read_bytes())
    if ledger['seed'] != dict(schema='g7-node-ledger-v1',prior_debit=reg['prior_debit'],
                              ceiling=1800.,max_new_seconds=reg['worker_seconds']):
        raise ValueError('Registered budget seed changed.')
    view=SimpleNamespace(root=out/'campaign',seed=ledger['seed'],read=lambda:ledger)
    audit_store(view,full=True)
    rows={e['job_id']:e for e in ledger['entries']}
    setup={name:json.loads((out/(name+'_setup.json')).read_bytes())
           for name in ('coarse','fine','quad','fv') if (out/(name+'_setup.json')).exists()}
    group={};count=0
    for path in sorted((out/'campaign/nodes').glob('*.json')):
        receipt=json.loads(path.read_bytes());job=receipt['job'];payload=receipt['payload']
        if digest(job)!=path.stem or digest(payload)!=receipt['payload_sha256']:
            raise ValueError('Receipt hash mismatch.')
        if job['registration_sha256']!=sha(out/'registration.json'):
            raise ValueError('Different registered invocation.')
        if rows[path.stem]['status'] not in ('complete','recovered-complete'):
            raise ValueError('Receipt is not journal-complete.')
        if job['kind'] not in ('coarse_candidates','fine_selected','quadrature_selected','fv_selected'):
            continue
        name=job['evaluator'];contract=setup[name]['contract'];count+=1
        if digest(contract)!=job['contract'] or job['prices'] not in plan['problem']['candidate_prices']:
            raise ValueError('Forward contract or candidate changed.')
        widths=contract['widths'];mean=payload['expected_demand'];second=payload['second_moment']
        target=contract['per_node'][0]['target']
        cost=sum(h*(s-2*a*m+a*a) for h,m,s,a in zip(widths,mean,second,target))
        if len(widths)!=len(mean) or len(mean)!=len(second) or len(target)!=len(mean):
            raise ValueError('Incomplete cost partition.')
        if not math.isclose(cost,payload['node_total'],rel_tol=0,abs_tol=1e-10):
            raise ValueError('Raw node moments do not reproduce stored J.')
        key=(job['policy'],name)
        group.setdefault(key,{})[job['index']]={'mean':mean,'second':second,'cost':cost,
                                                'weights':contract['origin']['log_weights']}
    worker=json.loads((out/'worker_summary.json').read_bytes()) if (out/'worker_summary.json').exists() else {}
    reduced={}
    for (policy,name),nodes in group.items():
        if set(nodes)!=set(range(5)):
            reduced[policy+'_'+name]={'complete':False,'nodes':sorted(nodes)};continue
        contract=setup[name]['contract'];logs=nodes[0]['weights'];shift=max(logs)
        unnormalized=[math.exp(x-shift) for x in logs];normalizer=sum(unnormalized)
        weights=[x/normalizer for x in unnormalized]
        widths=contract['widths'];target=contract['per_node'][0]['target']
        means=[sum(weights[j]*nodes[j]['mean'][l] for j in range(5)) for l in range(len(widths))]
        seconds=[sum(weights[j]*nodes[j]['second'][l] for j in range(5)) for l in range(len(widths))]
        bias=sum(h*(m-a)**2 for h,m,a in zip(widths,means,target))
        variance=sum(h*(s-m*m) for h,s,m in zip(widths,seconds,means))
        total=sum(weights[j]*nodes[j]['cost'] for j in range(5))
        within=sum(h*sum(weights[j]*(nodes[j]['second'][l]-nodes[j]['mean'][l]**2)
                       for j in range(5)) for l,h in enumerate(widths))
        between=sum(h*sum(weights[j]*(nodes[j]['mean'][l]-means[l])**2
                        for j in range(5)) for l,h in enumerate(widths))
        if abs(total-bias-variance)>1e-10 or abs(variance-within-between)>1e-10:
            raise ValueError('Bank objective/total variance identity failed.')
        summary=worker.get('coarse',[None]*3)[int(policy[-1])] if policy.startswith('candidate') else worker.get('selected_'+{'fine':'fine','quad':'quadrature','fv':'fv'}[name])
        if summary is not None:
            for label,value in [('J',total),('bias',bias),('variance',variance),
                                ('within_parameter_variance',within),('between_parameter_variance',between)]:
                if abs(summary[label]-value)>1e-10:raise ValueError('Report differs from raw costs.')
        reduced[policy+'_'+name]=dict(complete=True,J=total,bias=bias,variance=variance,
            within_parameter_variance=within,between_parameter_variance=between)
    result=dict(schema='g7-bounded-price-raw-review-v1',passed=True,
        journal_integrity_passed=True,receipts=len(list((out/'campaign/nodes').glob('*.json'))),
        new_forward_receipts=count,banks=reduced,worker_complete=worker.get('completed',False),
        numerical_validation_passed=worker.get('numerical_validation_passed',False),
        seconds=time.perf_counter()-started,new_forwards=0,numerical_imports=0,
        scientific_accept=False,full_day_budget_admitted=False,
        limitations='Hashes and costs reproduced; no rerun of transport or physical/historical validation.')
    write(out/'raw_review.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    review(parser.parse_args().output.resolve())
