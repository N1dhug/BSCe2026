"""One registered G-only price/partition invocation; old data stay read-only.

Numerical imports occur only in the worker under the pinned native checkout.
No retry, continuous search, EM, feedback, F or WM work is authorized here.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

from receipt_recovery import checked_json, sha
from restore_auxiliary import read_owned


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False).encode() + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    if os.name == 'posix':
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def select_candidate(costs):
    if len(costs) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                            or not __import__('math').isfinite(x) for x in costs):
        raise ValueError('Three complete finite coarse costs required.')
    return min(range(3), key=lambda k: (costs[k], k))


def validate(out):
    reg = json.loads((out / 'registration.json').read_bytes())
    plan = checked_json(out / 'design.json', reg['design_sha256'])
    budget = checked_json(out / 'previous_budget.json', reg['previous_budget_sha256'])
    caps = {j['id']: j['reservation_each_seconds'] for j in plan['proposal']['jobs']}
    expected = dict(verify_reuse=5, restore_setup=8, coarse_candidates=6,
                    select_coarse=1, fine_selected=12, quadrature_selected=18,
                    fv_selected=2, reduce_review=2)
    if (reg['schema'] != 'g7-bounded-price-registration-v1'
            or reg['prior_debit'] != budget['campaign_debit']
            or reg['campaign_ceiling'] != 1800 or reg['scope_ceiling'] != 320
            or reg['worker_seconds'] != 270 or reg['stop_export_reserve'] != 30
            or reg['review_charge'] != 20 or caps != expected
            or reg['prior_debit'] + reg['scope_ceiling'] > 1800
            or not reg['one_invocation_only'] or reg['automatic_retry']
            or plan['problem']['candidate_prices'] != [[.5, .5], [.3, .7], [.7, .3]]):
        raise ValueError('Budget, candidate list or invocation contract changed.')
    for name, expected_hash in reg['launcher_sha256'].items():
        if sha(out / name) != expected_hash:
            raise ValueError('Launcher bytes changed: ' + name)
    return reg, plan, caps


@contextmanager
def job_deadline(deadline, stop_type):
    if time.perf_counter() >= deadline:
        raise stop_type('Reservation already exhausted.')
    supported = hasattr(signal, 'setitimer')
    if supported:
        def stop(*unused):
            raise stop_type('Registered per-job reservation expired.')
        previous = signal.signal(signal.SIGALRM, stop)
        signal.setitimer(signal.ITIMER_REAL, max(1e-6, deadline-time.perf_counter()))
    try:
        yield
        if time.perf_counter() >= deadline:
            raise stop_type('Registered per-job reservation exceeded.')
    finally:
        if supported:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            signal.signal(signal.SIGALRM, previous)


def worker(out, old, seed):
    reg, plan, caps = validate(out)
    native = old / 'frozen_source'
    manifest = checked_json(out / 'native_manifest.json', reg['native_manifest_sha256'])
    for path, expected in manifest.items():
        if sha(native / path) != expected:
            raise ValueError('Native source changed: ' + path)
    # All imports below resolve to the authenticated original native checkout.
    import numpy as np
    from src.control.bm_1 import BM1ControlNumerics
    from src.method.bm_1_distribution_panel import baseline, target
    from src.method.bm_1_g7 import OwnedBankPriceEvaluator
    from src.method.bm_1_g7_jobs import (NodeJobStore, NodeJobStop,
        CheckpointedBankControl, atomic_json, digest)
    from src.method.bm_1_g8 import G8BankOrigin
    from src.exe.run_model.bm_1.g7 import compare_forward
    from src.exe.run_model.bm_1.g8 import TOLERANCES
    from src.exe.run_model.bm_1.g8_bernstein import SolverAudit
    from receipt_recovery import VerifiedReceiptIndex, guarded_store_class, audit_store

    Guard = guarded_store_class(NodeJobStore,
        native_source_path=native / 'src/method/bm_1_g7_jobs.py',
        native_source_sha256=manifest['src/method/bm_1_g7_jobs.py'])
    store = Guard(out / 'campaign', prior_debit=reg['prior_debit'],
                  ceiling=1800., max_new_seconds=270.)
    report = dict(schema='g7-bounded-price-worker-v1', stages={},
        actual_optimizer_started=False, numerical_validation_passed=False,
        scientific_accept=False, full_day_budget_admitted=False,
        historical_accept=False, stochastic_energy_equality_added=False,
        new_forwards=0, reused_m254_nodes=10, reused_fv_nodes=5)
    started = time.perf_counter()
    reg_hash = sha(out / 'registration.json')
    evaluators, contracts, audits, values, box = {}, {}, {}, {}, {}

    def job(kind, index, compute, **identity):
        key = dict(index=index, kind=kind, registration_sha256=reg_hash, **identity)
        def timed(deadline):
            with job_deadline(deadline, NodeJobStop):
                return compute()
        before = time.perf_counter()
        with store.lock():
            payload = store.node(key, caps[kind], timed)
        report['stages'].setdefault(kind, []).append(dict(index=index,
            job_id=digest(key), wall_seconds=time.perf_counter()-before))
        atomic_json(out / 'progress.json', report)
        print('Completed', kind, index, 'elapsed', round(time.perf_counter()-started, 2), flush=True)
        return payload

    def summary(value, evaluator):
        s = evaluator.supports[0]
        origin = value['predicted_origin']
        x = .5 * (origin.x_edges[:-1] + origin.x_edges[1:])
        return dict(J=float(value['total']), bias=float(value['bias']),
            variance=float(value['variance']),
            within_parameter_variance=float(value['within_parameter_variance']),
            between_parameter_variance=float(value['between_parameter_variance']),
            node_costs=value['node_costs'].tolist(), weights=value['weights'].tolist(),
            expected_energy_gap=float(s.widths @ (value['expected_demand']-baseline(s.midpoints))),
            endpoint_state_mean=float((origin.weights @ origin.conditional_mass) @ x))

    try:
        def restore():
            audit = checked_json(out / 'previous_audit.json', reg['previous_audit_sha256'])
            box['owned'] = read_owned(old, audit, plan['reuse'])
            index = VerifiedReceiptIndex(seed, out / 'adoption_manifest.json',
                manifest_sha256=reg['adoption_manifest_sha256'])
            rows = sorted((e for e in index.entries.values()
                           if e['job']['kind'] == 'fv_fine_nodes'), key=lambda e:e['job']['index'])
            if [e['job']['index'] for e in rows] != list(range(5)):
                raise ValueError('Five exact neutral FV nodes required.')
            box['fv_nodes'] = [index.payload(e['job']) for e in rows]
            setup = checked_json(seed / 'fv_fine_setup.json', reg['fv_setup_sha256'])
            if any(e['job']['contract'] != setup['contract_id'] or e['job']['prices'] != [.5,.5]
                   for e in rows) or digest(setup['contract']) != setup['contract_id']:
                raise ValueError('FV adoption contract differs.')
            box['fv_contract'] = setup['contract_id']
            return dict(native_sources=41, m254_nodes=10, fv_nodes=5,
                        all_adoption_receipts_verified=len(index.entries))
        job('verify_reuse', -1, restore)

        configs = [('coarse','mixture','coarse',257,1/96,1/96),
                   ('fine','mixture','fine',513,1/96,1/192),
                   ('quad','mixture','fine',513,1/192,1/192),
                   ('fv','fv','fine',513,1/96,1/192)]
        for i, (name, backend, label, cells, objective_step, solver_step) in enumerate(configs):
            def setup(name=name, backend=backend, label=label, cells=cells,
                      objective_step=objective_step, solver_step=solver_step):
                original = box['owned']['registration']
                origin = G8BankOrigin.from_verified_bytes((old/'inputs'/(label+'.json')).read_bytes(),
                    original['input_files'][label]['sha256'])
                def configure(projector):
                    audit = SolverAudit(projector.projection_solver, projector.dx)
                    projector.projection_solver = audit
                    audits[name] = audit
                e = OwnedBankPriceEvaluator(origin, baseline=baseline, target=target,
                    horizon=(12.,14.), price_edges=[12.,13.,14.],
                    numerics=BM1ControlNumerics(x_steps=cells,
                        objective_max_step=objective_step, solver_max_step=solver_step),
                    backend=backend, configure_projector=configure if backend=='mixture' else None)
                # job() holds the non-reentrant native OS lock. Describe/check
                # the unchanged contract without re-entering __init__'s lock.
                adapter = object.__new__(CheckpointedBankControl)
                adapter.evaluator=e; adapter.root=native; adapter.sources=manifest
                adapter.check_sources()
                contract=adapter.describe()
                contract_id=store.bind(contract)
                evaluators[name], contracts[name] = e, contract_id
                if name in ('coarse','fine'):
                    if contract_id != plan['reuse'][label+'_contract']:
                        raise ValueError('Complete neutral M254 contract changed.')
                    values['neutral_'+name] = e.combine_nodes([.5,.5], box['owned']['nodes'][label])
                if name == 'fv':
                    if contract_id != box['fv_contract']:
                        raise ValueError('Complete neutral FV contract changed.')
                    values['neutral_fv'] = e.combine_nodes([.5,.5], box['fv_nodes'])
                return dict(name=name, contract_id=contract_id,
                            contract=contract, runtime_rehydrated=True)
            payload = job('restore_setup', i, setup, evaluator=name)
            atomic_json(out / (name+'_setup.json'), payload)

        def forward(name, prices, kind, policy):
            e = evaluators[name]
            nodes=[]
            for j in range(5):
                nodes.append(job(kind,j,lambda j=j:e.evaluate_node(prices,j),
                    evaluator=name, contract=contracts[name], prices=prices, policy=policy))
                report['new_forwards'] += 1
            result = e.combine_nodes(prices, nodes)
            values[policy+'_'+name] = result
            return result

        report['actual_optimizer_started'] = True
        prices = plan['problem']['candidate_prices']
        coarse = [values['neutral_coarse']]
        for k in (1,2):
            coarse.append(forward('coarse', prices[k], 'coarse_candidates', 'candidate'+str(k)))
        selected = job('select_coarse',-1,lambda:dict(
            candidate_index=select_candidate([float(v['total']) for v in coarse]),
            coarse_costs=[float(v['total']) for v in coarse]))
        k = selected['candidate_index']
        report['selection'] = selected
        atomic_json(out / 'selection.json', selected)
        chosen = prices[k]
        fine = values['neutral_fine'] if k==0 else forward('fine',chosen,'fine_selected','selected')
        quad = forward('quad',chosen,'quadrature_selected','selected')
        fv = values['neutral_fv'] if k==0 else forward('fv',chosen,'fv_selected','selected')

        def reduce():
            pairs = {
                'coarse__fine':compare_forward(coarse[k],fine,evaluators['fine']),
                'fine__quadrature':compare_forward(fine,quad,evaluators['quad']),
                'fine__fv':compare_forward(fine,fv,evaluators['fv'])}
            projection = {name:a.record() for name,a in audits.items()}
            passed = all(p['passed'] for p in pairs.values()) and all(
                p['maximum_projection_cell_mass_l1']<=TOLERANCES['projection_l1']
                for p in projection.values())
            e0=abs(coarse[0]['total']-values['neutral_fine']['total'])
            es=abs(coarse[k]['total']-fine['total'])
            gain=values['neutral_fine']['total']-fine['total']
            result=dict(candidate_prices=prices, selected_index=k, selected_prices=chosen,
                coarse=[summary(v,evaluators['coarse']) for v in coarse],
                neutral_fine=summary(values['neutral_fine'],evaluators['fine']),
                selected_fine=summary(fine,evaluators['fine']),
                selected_quadrature=summary(quad,evaluators['quad']),
                selected_fv=summary(fv,evaluators['fv']),
                forward_pairs=pairs, projection_audits=projection,
                original_partition_fine_gain=float(gain),
                observed_grid_screening_margin=float(e0+es),
                numerical_validation_passed=bool(passed),
                improvement_screen_passed=bool(k!=0 and passed and gain>e0+es),
                scientific_accept=False, full_day_budget_admitted=False,
                continuous_global_optimality_claim=False,
                pure_quadrature_error_certificate=False)
            result['endpoint_state_difference_to_neutral']=float(
                result['selected_fine']['endpoint_state_mean']-result['neutral_fine']['endpoint_state_mean'])
            return result
        final=job('reduce_review',-1,reduce)
        report.update(final)
        report['completed']=True
    except Exception as exc:
        report.update(completed=False,error=repr(exc))
    finally:
        report['worker_elapsed_seconds']=time.perf_counter()-started
        report['campaign_debit_in_ledger']=store.charged()
        try:
            with store.lock():
                audit_store(store,full=True)
            report['journal_integrity_passed']=True
        except Exception as exc:
            report.update(journal_integrity_passed=False,journal_error=repr(exc))
        atomic_json(out / 'worker_summary.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='stages'},default=str),flush=True)


def parent(out, old, seed):
    reg, plan, caps=validate(out)
    if any((out/name).exists() for name in ('invocation_started.json','invocation.json','campaign/ledger.json')):
        raise ValueError('Scope already started; no automatic retry or resume.')
    write(out/'invocation_started.json',dict(registration_sha256=sha(out/'registration.json'),
        prior_debit=reg['prior_debit'],hard_loss_worker_reserve=reg['worker_seconds']))
    env=dict(os.environ,PYTHONPATH=str(old/'frozen_source'),OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    start=time.perf_counter();hard_stop=False
    with (out/'worker.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-B',str(out/Path(__file__).name),
            '--output',str(out),'--previous',str(old),'--seed',str(seed),'--worker'],
            cwd=old/'frozen_source',env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=reg['worker_seconds'])
        except subprocess.TimeoutExpired:
            hard_stop=True;process.kill();process.wait()
    wall=time.perf_counter()-start
    ledger=out/'campaign/ledger.json'
    increment=sum(e['charged_seconds'] for e in json.loads(ledger.read_bytes())['entries']) if ledger.exists() else 0.
    result=dict(schema='g7-bounded-price-invocation-v1',worker_wall_seconds=wall,
        ledger_increment=increment,native_charge=max(wall,increment),hard_stop=hard_stop,
        returncode=process.returncode,review_charge=reg['review_charge'],
        scientific_accept=False,full_day_budget_admitted=False)
    write(out/'invocation.json',result)
    before=time.perf_counter()
    archive=out.parent/(out.name+'_data.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='writer.lock':
                z.write(p,str(p.relative_to(out)))
    result['export_seconds']=time.perf_counter()-before
    result['scope_charge']=result['native_charge']+result['export_seconds']+reg['review_charge']
    result['campaign_debit']=reg['prior_debit']+result['scope_charge']
    result['remaining_seconds']=1800-result['campaign_debit']
    result['within_scope_ceiling']=result['scope_charge']<=reg['scope_ceiling']
    write(out/'invocation.json',result)
    write(out/'archive_index.json',dict(path=str(archive),sha256=sha(archive),
        bytes=archive.stat().st_size,final_invocation_sha256=sha(out/'invocation.json')))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--seed',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    call=worker if args.worker else parent
    call(args.output.resolve(),args.previous.resolve(),args.seed.resolve())
