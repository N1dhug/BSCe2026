"""Stdlib-only G day admission; default never executes a numerical worker.

One durable invocation, no automatic retry/refund. Actual execution requires
separate immutable authorized registration and reviewed whole-day rates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from integrated_budget import price_plan, digest, finite
from launch_integrated_feedback import pinned, write
from day_contract import validate_design

HELPERS = ('day_control', 'day_review', 'day_contract', 'launch_day_control', 'audited_node_control',
           'audited_feedback', 'feedback_phase', 'integrated_budget',
           'launch_integrated_feedback', 'receipt_recovery')


def preflight(root):
    root = Path(root).resolve()
    reg_raw = (root/'registration.json').read_bytes()
    reg = json.loads(reg_raw)
    if (reg.get('schema') != 'g7-full-day-registration-v1'
            or reg.get('stage') != 'full-day-and-feedback'
            or reg.get('execution_authorized') is not True
            or reg.get('automatic_retry') is not False):
        raise ValueError('Separate authorized full-day registration required.')
    plan = json.loads(pinned(root, 'design.json', reg['design_sha256']))
    validate_design(plan)
    rates = json.loads(pinned(root, 'rates.json', reg['rates_sha256']))
    previous = json.loads(pinned(root, 'previous_budget.json', reg['previous_budget_sha256']))
    if (previous['campaign_debit'] != reg['prior_debit']
            or previous['campaign_ceiling'] != reg['campaign_ceiling']):
        raise ValueError('Preserved campaign budget differs.')
    packet = json.loads(pinned(root, 'origins.json', reg['origins_sha256']))
    if (packet.get('schema') != 'g7-full-day-owned-origins-v1'
            or set(packet['origins']) != {'m254_coarse','m254_fine','fv_coarse','fv_fine'}):
        raise ValueError('Four frozen own posterior origins required.')
    manifest = json.loads(pinned(root, 'source_manifest.json', reg['source_manifest_sha256']))
    native = Path(reg['source_root']).resolve()
    required = {'root/g7/'+name+'.py' for name in HELPERS}
    required.add('root/g7/integration_native_manifest.json')
    if not required.issubset(manifest):
        raise ValueError('All native/driver/review/launcher helpers must be pinned.')
    for name, expected in manifest.items():
        pinned(native, name, expected)
    original = json.loads(pinned(native, 'root/g7/integration_native_manifest.json',
                                manifest['root/g7/integration_native_manifest.json']))
    if any(manifest.get(k) != v for k,v in original.items()):
        raise ValueError('Original frozen numerical kernels must be preserved.')
    for module, path in ((sys.modules[__name__], 'root/g7/launch_day_control.py'),
                         (sys.modules['integrated_budget'], 'root/g7/integrated_budget.py'),
                         (sys.modules['launch_integrated_feedback'], 'root/g7/launch_integrated_feedback.py'),
                         (sys.modules['day_contract'], 'root/g7/day_contract.py')):
        if Path(module.__file__).resolve() != native/path:
            raise ValueError('Loaded stdlib helper differs from source root.')
    if digest(manifest) != reg['source_fingerprint'] or digest(packet['origins']) != reg['origin_fingerprint']:
        raise ValueError('Measured context must bind actual frozen sources/own origins.')
    admission = price_plan(plan, rates, prior_debit=reg['prior_debit'],
        ceiling=reg['campaign_ceiling'], source_fingerprint=reg['source_fingerprint'],
        origin_fingerprint=reg['origin_fingerprint'], versions=reg['versions'],
        reserve_seconds=reg['stop_export_review_reserve'])
    if not admission['budget_admitted']:
        raise ValueError('Whole-day graph not admitted; missing rates cannot become zero.')
    scope = finite(reg['scope_cap'], 'full scope hardcap', True)
    worker = finite(reg['worker_seconds'], 'worker hardcap', True)
    reserve = finite(reg['stop_export_review_reserve'], 'stop/export/review reserve', True)
    if (worker+reserve > scope or scope > reg['campaign_ceiling']-reg['prior_debit']
            or admission['estimated_seconds']-reserve > worker):
        raise ValueError('Complete worker and reserve must fit the preserved campaign.')
    reservations = {r['work']:r['reservation_seconds'] for r in admission['rows']}
    return reg, plan, packet, manifest, admission, reservations, hashlib.sha256(reg_raw).hexdigest()


def worker(root):
    reg, plan, packet, manifest, admission, reservations, reg_hash = preflight(root)
    root = Path(root).resolve()
    claim = json.loads((root/'invocation.json').read_bytes())
    if (claim.get('registration_sha256') != reg_hash or claim.get('status') != 'running'
            or claim.get('token') != os.environ.get('G7_DAY_INVOCATION_TOKEN')):
        raise ValueError('A worker must belong to this durable parent invocation.')
    import numpy as np
    import scipy
    if dict(numpy=np.__version__, scipy=scipy.__version__) != reg['versions']:
        raise ValueError('Numerical runtime differs from registered rates.')
    from src.method.bm_1_g7_jobs import NodeJobStore
    from receipt_recovery import guarded_store_class
    from day_control import run_day
    validate_design(plan)
    native = Path(reg['source_root']).resolve()
    for name in manifest:
        if name.endswith('.py'):
            module_name = name[:-3].replace('/', '.') if name.startswith('src/') else Path(name).stem
            module = sys.modules.get(module_name)
            if module is not None and Path(module.__file__).resolve() != native/name:
                raise ValueError('Loaded module borrowed another source root: '+name)
    Guard = guarded_store_class(NodeJobStore,
        native_source_path=native/'src/method/bm_1_g7_jobs.py',
        native_source_sha256=manifest['src/method/bm_1_g7_jobs.py'])
    store = Guard(root/'campaign', prior_debit=reg['prior_debit'],
        ceiling=reg['campaign_ceiling'], max_new_seconds=reg['worker_seconds'])
    result = run_day(plan, packet['origins'], store, source_root=native,
                     manifest=manifest, reservations=reservations, output=root)
    # Recheck all external inputs/source bytes before even numerical completion.
    preflight(root)
    return result


def launch(root):
    root = Path(root).resolve()
    reg, _, _, _, admission, _, reg_hash = preflight(root)
    claim_path = root/'invocation.json'
    token = os.urandom(32).hex()
    claim = dict(schema='g7-full-day-invocation-v1', status='running',
        registration_sha256=reg_hash, token=token,
        conservative_charge=reg['scope_cap'], campaign_debit=reg['prior_debit']+reg['scope_cap'])
    # Exclusive create before numerical imports. A process loss retains scope.
    with claim_path.open('x', encoding='utf-8') as stream:
        json.dump(claim, stream, allow_nan=False); stream.flush(); os.fsync(stream.fileno())
    if os.name == 'posix':
        fd = os.open(root, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    env = os.environ.copy()
    env.update(PYTHONPATH=reg['source_root'], OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', G7_DAY_INVOCATION_TOKEN=token)
    started = perf_counter(); outcome = 'failed'; numerical = False
    try:
        with (root/'worker.log').open('xb') as log:
            try:
                proc = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                    str(root), '--worker'], env=env, stdout=log, stderr=subprocess.STDOUT,
                    cwd=reg['source_root'], timeout=reg['worker_seconds'])
                outcome = 'complete' if proc.returncode == 0 else 'failed'
            except subprocess.TimeoutExpired:
                outcome = 'hard-stop'
        if outcome == 'complete':
            report = json.loads((root/'day_result.json').read_bytes())
            numerical = report['numerical_validation_passed'] is True
        preflight(root)
    except Exception:
        outcome = 'failed'
        raise
    finally:
        wall = perf_counter()-started; charge = max(reg['scope_cap'], wall)
        budget = dict(schema='g7-full-day-final-budget-v1', status=outcome,
            parent_wall_seconds=wall, conservative_charge=charge,
            campaign_debit=reg['prior_debit']+charge, campaign_ceiling=reg['campaign_ceiling'],
            exceeded_cap=wall > reg['scope_cap'], numerical_validation_passed=bool(
                numerical and outcome == 'complete' and wall <= reg['scope_cap']),
            scientific_accept=False, automatic_retry=False)
        write(root/'final_budget.json', budget)
        write(claim_path, dict(claim, status=outcome, conservative_charge=charge,
                              campaign_debit=reg['prior_debit']+charge))
    return dict(status=outcome, budget=budget, admission=admission,
                scientific_accept=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker: result = worker(args.packet)
    elif args.execute: result = launch(args.packet)
    else:
        _, _, _, _, admission, _, _ = preflight(args.packet)
        result = dict(whole_graph_driver_implemented=True,
            budget_preflight_passed=True, native_input_validation_pending=True,
            budget=admission, new_native_forwards=0, scientific_accept=False)
    print(json.dumps(result, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
