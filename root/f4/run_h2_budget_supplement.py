"""Separately approved budget supplement; reuse the immutable H2 execution."""
from time import perf_counter, time
BEGUN = perf_counter()
import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
import re
import sys
import threading
import traceback
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from root.f4 import run_h2_continuation as original
c = original.c
SOURCES = ('root/f4/run_h2_budget_supplement.py', 'root/f4/checkpoint_h2_budget_supplement.py')


def base_campaign(decision):
    path = ROOT / decision['predecessor_campaign_path']
    c.require(c.sha(path) == decision['predecessor_campaign_sha256'], 'Preserved v3 campaign changed.')
    old_decision = c.load(ROOT / 'root/f4/h2_continuation_deadline_approval_20261008.json')
    c.require(old_decision['identity'] == decision['original_decision_identity'], 'Original approval changed.')
    return original.validate_campaign(c.load(path), old_decision), old_decision


def preparation(decision):
    path = ROOT / decision['preparation_ledger_path']
    c.require(c.sha(path) == decision['preparation_ledger_sha256'], 'Supplement preparation charge changed.')
    return c.load(path)['entries']


def validate_campaign(ledger, decision):
    old, _ = base_campaign(decision)
    prefix = old['entries'] + preparation(decision)
    c.require(ledger['schema'] == 'f4-h2-budget-supplement-campaign-v4'
        and ledger['decision_identity'] == decision['identity']
        and ledger['entries'][:len(prefix)] == prefix, 'Lost old charges or different supplement.')
    c.require(ledger['ceiling_seconds'] == decision['proposed_ceiling_seconds'] == 72000
        and ledger['deadline_utc_seconds'] == old['deadline_utc_seconds']
        and ledger['validation_reserve_seconds'] >= old['validation_reserve_seconds'], 'Budget/deadline/reserve changed.')
    c.require(all(e['status'] in ('closed', 'reserved') and math.isfinite(e['charged_seconds'])
        and e['charged_seconds'] >= 0 for e in ledger['entries']), 'Invalid charge.')
    c.require(ledger['charged_seconds'] == c.PRIOR_SECONDS + c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
        and ledger['charged_seconds'] <= 72000, 'Incorrect or exhausted supplement total.')
    c.require(c.ordered_sum(e['charged_seconds'] for e in ledger['entries']
        if e['label'].startswith('native-development|')) <= 1080, 'Native allowance exceeded.')
    return ledger


def authorized(decision, directory):
    c.require(decision['approved'] is True and isinstance(decision.get('user_approval'), str)
        and bool(decision['user_approval'].strip()), 'Additional budget has not been approved.')
    c.require(decision['identity'] == c.digest({k:v for k,v in decision.items() if k != 'identity'}), 'Decision changed.')
    old, prior_decision = base_campaign(decision)
    c.require(decision['original_ceiling_seconds'] == 64800 and decision['proposed_ceiling_seconds'] == 72000
        and decision['maximum_additional_seconds'] == 7200 and decision['native_development_ceiling_seconds'] == 1080
        and decision['maximum_candidate_calls'] == {'2':25, '4':49} and decision['global_pair_cap'] == 73
        and decision['invocation_cap_seconds'] == 600 and decision['stop_export_reserve_seconds'] == 60
        and decision['deadline_utc_seconds'] == old['deadline_utc_seconds'] > time() + 75, 'Registered limits changed/expired.')
    c.require(decision['manifest_identity'] == c.load(directory/'manifest.json')['identity'], 'Wrong run.')
    c.require(decision['source_sha256'] == {p:c.sha(ROOT/p) for p in SOURCES}, 'Supplement source changed.')
    control = ROOT / decision['offline_control_path']
    c.require(c.sha(control) == decision['offline_control_sha256'] and c.load(control)['passed'], 'No registered supplement control.')
    preparation(decision)
    return prior_decision


@contextmanager
def reservation(path, decision):
    with c._RunLock(path.with_suffix('.lock')):
        old, prior_decision = base_campaign(decision)
        if path.exists():
            ledger = validate_campaign(c.load(path), decision)
        else:
            ledger = dict(old, schema='f4-h2-budget-supplement-campaign-v4', ceiling_seconds=72000.,
                entries=old['entries']+preparation(decision), decision_identity=decision['identity'])
            ledger['charged_seconds'] = c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
            validate_campaign(ledger, decision)
            c.write(path, ledger)
        before = ledger['charged_seconds']
        grant = min(600., 72000.-before, decision['deadline_utc_seconds']-time()+perf_counter()-BEGUN)
        c.require(grant > 75, 'Insufficient remaining budget/deadline.')
        entry = dict(label='compact-production|separate-budget-supplement', status='reserved', charged_seconds=grant,
            started_utc_seconds=time())
        ledger['entries'].append(entry)
        ledger['charged_seconds'] = c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
        c.write(path, ledger, replace=True)
        watchdog = threading.Timer(max(.01, grant-(perf_counter()-BEGUN)-1), lambda:os._exit(124))
        watchdog.daemon = True; watchdog.start()
        try:
            yield dict(ledger=ledger, before=before, begun=BEGUN, grant=grant,
                protected_rates=prior_decision['protected_rates'],
                deadline=min(time()+grant-(perf_counter()-BEGUN)-60, decision['deadline_utc_seconds']-60))
        finally:
            entry.update(status='closed', charged_seconds=perf_counter()-BEGUN)
            ledger['charged_seconds'] = c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries'])
            c.write(path, ledger, replace=True); watchdog.cancel()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--decision', type=Path, required=True); p.add_argument('--source-commit', required=True)
    p.add_argument('--stage', choices=('prefinal', 'export-final', 'final'), default='prefinal')
    p.add_argument('--workbook', type=Path)
    a = p.parse_args(); directory = a.run.resolve(); decision = c.load(a.decision)
    authorized(decision, directory)  # Reject before any reservation or propagation.
    c.require(re.fullmatch('[0-9a-f]{40}', a.source_commit) is not None, 'Immutable controller commit required.')
    manifest, _, old_run = original.verify_origin(directory)
    c.require(not any((directory/n).exists() for n in ('STOP', 'FAILED.json', 'CONTINUATION_FAILED.json',
        'SUPPLEMENT_FAILED.json', 'PREPARATION_FAILED.json')), 'Preserved terminal stop; no retry.')
    old_execution = c.load(directory/'continuation_execution.json')
    c.require(old_execution['manifest_identity'] == manifest['identity']
        and old_execution['decision_identity'] == decision['original_decision_identity'], 'Original execution changed.')
    engine = None; number = len(list(directory.glob('supplement_report_*.json')))+1
    with (c._RunLock(ROOT/decision['predecessor_campaign_path'].replace('.json', '.lock')),
            reservation(a.campaign, decision) as allocation, c._RunLock(old_run), c._RunLock(directory)):
        execution = dict(schema='f4-h2-separate-budget-supplement-execution-v1', source_commit=a.source_commit,
            source_sha256=decision['source_sha256'], decision_identity=decision['identity'],
            original_execution_sha256=c.sha(directory/'continuation_execution.json'), manifest_identity=manifest['identity'])
        path = directory/'supplement_execution.json'
        if path.exists(): c.require(c.load(path) == execution, 'Supplement execution changed.')
        else: c.write(path, execution)
        try:
            engine = original.Engine(directory, c.receipt(), manifest, allocation)
            if a.stage == 'prefinal': result = original.prefinal(engine)
            elif a.stage == 'export-final':
                c.require(a.workbook is not None, 'Original workbook required after freeze.')
                c.load_final_observations(engine, a.workbook); result = dict(status='final_data_exported')
            else:
                c.require((directory/'final_observations.npz').exists(), 'No frozen final data.')
                c.final_evaluation(engine, None); result = dict(status='final_completed')
            original.verify_origin(directory)
        except (Exception, KeyboardInterrupt) as exc:
            paused = isinstance(exc, (TimeoutError, KeyboardInterrupt))
            result = dict(status='paused' if paused else 'failed', reason=str(exc), traceback=traceback.format_exc())
            if not paused: c.write(directory/'SUPPLEMENT_FAILED.json', result)
        finally:
            if engine is not None: engine.close()
        result.update(native_jobs=0 if engine is None else len(engine.store.events),
            scientific_acceptance=False, automatic_retry=False)
        c.write(directory/f'supplement_report_{number:03d}.json', result)
    print(c.json.dumps(result), flush=True)
    if result['status'] == 'failed': raise SystemExit(2)


if __name__ == '__main__': main()
