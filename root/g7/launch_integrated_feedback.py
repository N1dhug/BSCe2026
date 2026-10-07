"""Stdlib parent: reject unadmitted G7 feedback before importing numerics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from integrated_budget import price_plan


def pinned(root, name, expected):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Pinned path leaves registered root.')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Registered file bytes changed: ' + name)
    return raw


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False).encode()+b'\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name == 'posix':
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


def preflight(root):
    root = Path(root).resolve()
    reg = json.loads((root/'registration.json').read_bytes())
    if (reg.get('schema') != 'g7-feedback-phase-registration-v2'
            or reg.get('stage') != 'feedback-only'
            or reg.get('automatic_retry') is not False
            or reg.get('execution_authorized') is not True):
        raise ValueError('Separate authorized feedback-phase registration required.')
    plan = json.loads(pinned(root, 'design.json', reg['design_sha256']))
    rates = json.loads(pinned(root, 'rates.json', reg['rates_sha256']))
    previous = json.loads(pinned(root, 'previous_budget.json', reg['previous_budget_sha256']))
    if previous['campaign_debit'] != reg['prior_debit'] or previous['campaign_ceiling'] != reg['campaign_ceiling']:
        raise ValueError('Preserved prior budget changed.')
    native_root = Path(reg['source_root']).resolve()
    manifest = json.loads(pinned(root, 'source_manifest.json', reg['source_manifest_sha256']))
    required = {'root/g7/audited_node_control.py','root/g7/audited_feedback.py',
        'root/g7/feedback_phase.py','root/g7/integrated_budget.py',
        'root/g7/launch_integrated_feedback.py','root/g7/receipt_recovery.py'}
    if not required.issubset(manifest):
        raise ValueError('All loaded G helper sources must be registered.')
    if Path(__file__).resolve() != native_root/'root/g7/launch_integrated_feedback.py':
        raise ValueError('Run the registered source-root launcher.')
    for name, expected in manifest.items():
        pinned(native_root, name, expected)
    admission = price_plan(plan, rates, prior_debit=reg['prior_debit'],
        ceiling=reg['campaign_ceiling'], source_fingerprint=reg['source_fingerprint'],
        origin_fingerprint=reg['origin_fingerprint'], versions=reg['versions'],
        reserve_seconds=reg['stop_export_review_reserve'])
    from integrated_budget import digest, finite
    if digest(manifest) != reg['source_fingerprint']:
        raise ValueError('Rate/source fingerprint differs from actual pinned sources.')
    packet = json.loads(pinned(root, 'feedback_inputs.json', reg['feedback_inputs_sha256']))
    if digest({k:v['origin'] for k,v in packet['forwards'].items()}) != reg['origin_fingerprint']:
        raise ValueError('Rate/origin fingerprint differs from actual owned origins.')
    if packet.get('projection_l1_limit') != plan['tolerances']['projection_l1']:
        raise ValueError('Feedback projection gate differs from registered plan.')
    scope = finite(reg['scope_cap'], 'phase hardcap', True)
    worker = finite(reg['worker_seconds'], 'worker cap', True)
    reserve = finite(reg['stop_export_review_reserve'], 'reserve', True)
    if worker+reserve > scope or scope > reg['campaign_ceiling']-reg['prior_debit']:
        raise ValueError('Worker/reserve do not fit preserved campaign budget.')
    if not admission['budget_admitted']:
        raise ValueError('Whole G7 graph is not admitted; unknown rates are not zero.')
    reservations = {r['work']:r['reservation_seconds'] for r in admission['rows']}
    phase_estimate = (4*reservations['feedback_update']
        +3*reservations['feedback_policy_search']
        +sum(reservations['feedback_'+name] for name in
             ('m254_coarse','m254_fine','fv_coarse','fv_fine')))
    if phase_estimate > worker:
        raise ValueError('Whole feedback phase does not fit worker reservation.')
    return reg, manifest, packet, admission, reservations


def worker(root):
    reg, manifest, packet, admission, reservations = preflight(root)
    import numpy as np
    import scipy
    if dict(numpy=np.__version__,scipy=scipy.__version__) != reg['versions']:
        raise ValueError('Native runtime versions differ from priced registration.')
    from src.method.bm_1_g7_jobs import NodeJobStore
    from receipt_recovery import guarded_store_class
    native = Path(reg['source_root'])
    Guard = guarded_store_class(NodeJobStore,
        native_source_path=native/'src/method/bm_1_g7_jobs.py',
        native_source_sha256=manifest['src/method/bm_1_g7_jobs.py'])
    store = Guard(root/'campaign', prior_debit=reg['prior_debit'],
        ceiling=reg['campaign_ceiling'], max_new_seconds=reg['worker_seconds'])
    from feedback_phase import run_feedback_phase
    return run_feedback_phase(packet, store, source_root=native,
        source_sha256=manifest, reservations=reservations, output=root)


def launch(root):
    root = Path(root).resolve()
    reg, _, _, admission, _ = preflight(root)
    claim = root/'invocation.json'
    # Persistent one-invocation claim. After loss, the full scope stays charged.
    with claim.open('x', encoding='utf-8') as stream:
        json.dump(dict(schema='g7-feedback-invocation-v2', status='running',
            registration_sha256=hashlib.sha256((root/'registration.json').read_bytes()).hexdigest(),
            conservative_charge=reg['scope_cap'],
            campaign_debit=reg['prior_debit']+reg['scope_cap']), stream)
        stream.flush(); os.fsync(stream.fileno())
    if os.name == 'posix':
        fd = os.open(root, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    env = os.environ.copy()
    env.update(PYTHONPATH=reg['source_root'], OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = perf_counter(); outcome = 'failed'
    with (root/'worker.log').open('xb') as log:
        try:
            result = subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),
                str(root),'--worker'],env=env,stdout=log,stderr=subprocess.STDOUT,
                timeout=reg['worker_seconds'], cwd=reg['source_root'])
            outcome = 'complete' if result.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            outcome = 'hard-stop'
        finally:
            wall = perf_counter()-started
            # Keep the full approved phase reserve on every outcome; no refund.
            write(root/'final_budget.json',dict(status=outcome,parent_wall_seconds=wall,
                conservative_charge=max(reg['scope_cap'],wall),
                campaign_debit=reg['prior_debit']+max(reg['scope_cap'],wall),
                campaign_ceiling=reg['campaign_ceiling'],
                exceeded_cap=wall>reg['scope_cap'], scientific_accept=False))
            write(claim,dict(status=outcome,conservative_charge=max(reg['scope_cap'],wall)))
    return dict(status=outcome, budget_admitted=admission['budget_admitted'],
                scientific_accept=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('packet',type=Path)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--execute',action='store_true')
    group.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker: result=worker(args.packet)
    elif args.execute: result=launch(args.packet)
    else:
        reg,_,_,admission,_=preflight(args.packet)
        result=dict(phase_budget_preflight_passed=True,native_input_validation_pending=True,
                    whole_graph_driver_implemented=False,
                    budget=admission,new_native_forwards=0)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
