"""Explicit F-only recovery after filesystem loss; preserve terminal evidence.

Native numerical kernels, COBYQA and all original gates are delegated unchanged.
Every invocation reserves its full cap before imports or work. No automatic loop.
"""
from time import perf_counter, time
BEGUN = perf_counter()
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
SOURCE = 'root/f4/run_h2_filesystem_recovery_v2.py'
PRIOR = 425.8919726690074


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def total(values):
    result = 0.
    for value in values:
        result += value
    return result


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+'.partial')
    with partial.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, separators=(',', ':')); handle.write('\n')
        handle.flush(); os.fsync(handle.fileno())
    os.replace(partial, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Lock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+b')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.handle.close(); raise
        return self

    def __exit__(self, *args):
        self.handle.close()


def validate(ledger, decision):
    old_path = ROOT/decision['predecessor_campaign_path']; old = load(old_path)
    require(sha(old_path) == decision['predecessor_campaign_sha256'], 'Terminal ledger changed.')
    require(ledger['entries'][:len(old['entries'])] == old['entries'], 'Lost historical charge.')
    require(ledger['schema'] == 'f4-h2-filesystem-recovery-campaign-v6'
        and ledger['decision_identity'] == decision['identity']
        and ledger['ceiling_seconds'] == decision['ceiling_seconds'] == 72000
        and ledger['native_ceiling_seconds'] == decision['native_ceiling_seconds'] == 1140
        and ledger['deadline_utc_seconds'] == decision['deadline_utc_seconds']
        and ledger['validation_reserve_seconds'] >= old['validation_reserve_seconds'],
        'Recovery budget or reserve changed.')
    require(all(e['status'] in ('closed', 'reserved') and math.isfinite(e['charged_seconds'])
        and e['charged_seconds'] >= 0 for e in ledger['entries']), 'Invalid charge.')
    require(ledger['charged_seconds'] == PRIOR+total(e['charged_seconds'] for e in ledger['entries'])
        and ledger['charged_seconds'] <= 72000, 'Total mismatch or exhausted budget.')
    require(total(e['charged_seconds'] for e in ledger['entries']
        if e['label'].startswith('native-development|')) <= 1140, 'Native ceiling exceeded.')
    return ledger


@contextmanager
def reservation(path, decision, label, cap, *, native):
    with Lock(path.with_suffix('.lock')):
        ledger = validate(load(path), decision); before = ledger['charged_seconds']
        require(ledger['entries'][-1]['status'] == 'closed', 'Preserved interrupted reservation; no retry.')
        require(before+cap <= 72000 and time()+cap < decision['deadline_utc_seconds'], 'Budget/deadline expired.')
        if native:
            require(total(e['charged_seconds'] for e in ledger['entries']
                if e['label'].startswith('native-development|'))+cap <= 1140, 'Native cap cannot be admitted.')
        entry = dict(label=('native-development|' if native else 'compact-production|')+label,
            status='reserved', charged_seconds=cap, started_utc_seconds=time())
        ledger['entries'].append(entry); ledger['charged_seconds'] = PRIOR+total(e['charged_seconds'] for e in ledger['entries'])
        write(path, ledger)
        timer = threading.Timer(max(.01, cap-(perf_counter()-BEGUN)-.5), lambda: os._exit(124))
        timer.daemon = True; timer.start()
        try:
            yield dict(ledger=ledger, before=before, begun=BEGUN, grant=cap,
                protected_rates=decision['protected_rates'],
                deadline=min(time()+cap-(perf_counter()-BEGUN)-90, decision['deadline_utc_seconds']-60))
        finally:
            entry.update(status='closed', charged_seconds=perf_counter()-BEGUN)
            ledger['charged_seconds'] = PRIOR+total(e['charged_seconds'] for e in ledger['entries'])
            write(path, ledger); timer.cancel()


def numerical():
    from src.exe.run_model import bm_1_h2_continuation as original
    return original, original.c


def verify(directory, decision):
    original, c = numerical()
    manifest, origin, oldest = original.verify_origin(directory)
    terminal = ROOT/decision['terminal_run_path']
    require(manifest['recovery_source_sha256'] == sha(ROOT/SOURCE)
        and manifest['recovery_decision_identity'] == decision['identity'], 'Recovery source or decision changed.')
    require(sha(ROOT/decision['previous_recovery_failure_path'])
        == decision['previous_recovery_failure_sha256'], 'Earlier recovery failure changed.')
    receipt = load(directory/'terminal_origin.json')
    require(sha(directory/'terminal_origin.json') == manifest['terminal_origin_sha256']
        and receipt['identity'] == load(terminal/'manifest.json')['identity'], 'Terminal origin changed.')
    for name, expected in receipt['metadata_sha256'].items():
        require(sha(terminal/name) == expected, 'Terminal evidence changed: '+name)
    require(sha(directory/'terminal_job_index.json') == manifest['terminal_job_index_sha256'], 'Raw index changed.')
    return original, c, manifest, terminal, oldest


def make_store(original, c, directory, manifest, terminal, oldest):
    class Store(original.AdoptingJobStore):
        def run(self, meta, prior, compute):
            token = dict(identity=self.identity, **meta); key = c.digest(token)
            if (self.directory/'jobs'/f'{key}.npz').exists() or (self.directory/'jobs'/f'{key}.failed.json').exists():
                return c.JobStore.run(self, meta, prior, compute)
            old_token = dict(identity=terminal_identity, **meta); old_key = c.digest(old_token)
            if old_key not in terminal_index:
                return super().run(meta, prior, compute)
            path = terminal/'jobs'/f'{old_key}.npz'; raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == terminal_index[old_key], 'Terminal raw bytes changed.')
            saved = c._read_array_checkpoint(raw)
            require(saved['token'] == old_token and saved['checksum'] == c.fingerprint(
                {k:v for k,v in saved.items() if k != 'checksum'}), 'Terminal token/checksum changed.')
            c.state_check(prior); c.state_check(saved['result']['log_state'])
            require(c.fingerprint(saved['prior_log_state']) == c.fingerprint(prior), 'Terminal prior bytes/dtype changed.')
            c.same_state(saved['prior_log_state'], prior)
            receipt = dict(schema='f4-h2-terminal-success-adoption-v1', requested_token=token,
                original_token=saved['token'], original_checksum=saved['checksum'], original_key=old_key,
                predecessor_raw_sha256=terminal_index[old_key], exact_prior_fingerprint=c.fingerprint(prior),
                original_result_preserved=True, old_checkpoint_relabelled=False, native_propagations=0)
            target = self.directory/'adopted'/f'{key}.json'
            if target.exists():
                require(c.load(target) == receipt, 'Terminal adoption changed.')
            else:
                c.write(target, receipt)
            self.adopted[key] = receipt; self.completed[old_key] = saved
            return saved['result']
    terminal_identity = load(terminal/'manifest.json')['identity']
    terminal_index = load(directory/'terminal_job_index.json')
    origin = load(directory/'origin.json')
    return Store(directory, manifest['identity'], oldest, origin['predecessor_identity'],
        load(directory/'predecessor_job_index.json'))


def prepare_and_check(directory, decision):
    require(not directory.exists(), 'Preserve prepared/failed recovery; no repeat.')
    original, c = numerical(); terminal = ROOT/decision['terminal_run_path']
    old_manifest, origin, oldest = original.verify_origin(terminal)
    require(c.environment() == old_manifest['environment'], 'Pinned environment changed.')
    closure = load(ROOT/decision['restored_import_closure_path'])
    require(closure['static_complete'] and sha(ROOT/decision['restored_import_closure_path'])
        == decision['restored_import_closure_sha256'], 'Import closure changed.')
    for name, expected in closure['source_sha256'].items():
        require(sha(ROOT/name) == expected, 'Ancillary import bytes changed: '+name)
    from src.exe.run_model.bm_1_history import _beta_mass
    from src.control.bm_1_handoff import BM1ForecastOrigin
    require(callable(_beta_mass) and callable(BM1ForecastOrigin), 'Missing F runtime entrypoints.')
    data = c.receipt()
    require(data['packet']['identity'] == old_manifest['packet_identity']
        and data['observation_sha256'] == old_manifest['observation_sha256'], 'Data changed.')
    index = {p.stem:sha(p) for p in sorted((terminal/'jobs').glob('*.npz'))}
    require(len(index) == 1477, 'Incomplete terminal success chain.')
    for key, expected in load(terminal/'predecessor_job_index.json').items():
        require(sha(oldest/'jobs'/f'{key}.npz') == expected, 'Original predecessor bytes changed.')
    directory.mkdir()
    for name in ('origin.json', 'predecessor_job_index.json', 'packet.json', 'inputs.npz', 'observations.npz'):
        (directory/name).write_bytes((terminal/name).read_bytes())
    names = ['manifest.json', 'origin.json', 'SUPPLEMENT_FAILED.json', 'supplement_report_005.json',
        'continuation_optimizer_4_calls.json', 'continuation_candidate_registry.json',
        'predecessor_job_index.json', 'packet.json', 'inputs.npz', 'observations.npz']
    names += [str(p.relative_to(terminal)) for p in sorted((terminal/'jobs').glob('*.failed.json'))]
    c.write(directory/'terminal_origin.json', dict(identity=old_manifest['identity'],
        metadata_sha256={n:sha(terminal/n) for n in names}, terminal_failure_preserved=True))
    c.write(directory/'terminal_job_index.json', index)
    manifest = dict(old_manifest, schema='f4-h2-filesystem-recovery-v1',
        terminal_origin_sha256=sha(directory/'terminal_origin.json'),
        terminal_job_index_sha256=sha(directory/'terminal_job_index.json'),
        recovery_source_sha256=sha(ROOT/SOURCE), recovery_decision_identity=decision['identity'])
    manifest.pop('identity'); manifest['identity'] = c.digest(manifest); c.write(directory/'manifest.json', manifest)
    for name in ('continuation_optimizer_4_calls.json', 'continuation_candidate_registry.json'):
        adopted = c.load(terminal/name)
        adopted.update(identity=manifest['identity'], adopted_terminal_identity=old_manifest['identity'],
            adopted_terminal_bytes_sha256=sha(terminal/name), adopted_values_are_preserved=True)
        c.write(directory/name, adopted)
    original, c, manifest, terminal, oldest = verify(directory, decision)
    store = make_store(original, c, directory, manifest, terminal, oldest)
    sample_path = next(iter(sorted((terminal/'jobs').glob('*.npz'))))
    sample = c._read_array_checkpoint(sample_path.read_bytes())
    meta = {k:v for k,v in sample['token'].items() if k != 'identity'}
    def forbidden():
        raise AssertionError('A control must not propagate a model.')
    result = store.run(meta, sample['prior_log_state'], forbidden)
    require(c.fingerprint(result) == c.fingerprint(sample['result']), 'Adoption result changed.')
    wrong = sample['prior_log_state'].astype('float32')
    try:
        store.run(meta, wrong, forbidden)
    except ValueError:
        pass
    else:
        raise AssertionError('Different prior dtype was accepted.')
    class TraceStop(Exception):
        pass
    class Stub:
        predecessor = oldest
        origin = c.load(directory/'origin.json')
        def check(self, **kwargs):
            pass
        def replay(self, candidate, *args, **kwargs):
            require(c.curve_key(candidate) == decision['expected_next_candidate'], '48-call trace replay diverged.')
            raise TraceStop()
    stub = Stub(); stub.directory = directory; stub.manifest = manifest
    try:
        original.optimizer_family(stub, 4)
    except TraceStop:
        pass
    else:
        raise AssertionError('Expected the one remaining candidate after 48 cached calls.')
    require(len(c.load(oldest/'extension_optimizer_4_calls.json')['calls'])
        +len(c.load(directory/'continuation_optimizer_4_calls.json')['calls']) == 48, 'Trace length changed.')
    c.write(directory/'offline_control.json', dict(passed=True, raw_terminal_files_hash_verified=1477,
        raw_original_files_hash_verified=3655, exact_adoption_and_prior_dtype_rejection=True,
        optimizer_trace_calls_replayed=48, next_candidate=decision['expected_next_candidate'],
        numerical_propagations=0, source_files_unchanged=73, restored_import_closure_passed=True,
        restored_import_closure_sha256=decision['restored_import_closure_sha256'], scientific_acceptance=False))
    print(json.dumps(dict(status='prepared_and_control_passed', manifest_identity=manifest['identity'],
        numerical_propagations=0)), flush=True)


def run(directory, decision, allocation, source_commit, stage, workbook, control_sha256):
    original, c, manifest, terminal, oldest = verify(directory, decision)
    require(load(directory/'offline_control.json')['passed']
        and sha(directory/'offline_control.json') == control_sha256, 'Registered recovery control missing/changed.')
    require(not any((directory/n).exists() for n in ('RECOVERY_FAILED.json', 'FAILED.json',
        'CONTINUATION_FAILED.json', 'STOP')), 'Terminal recovery or preserved STOP; no retry.')
    class Engine(original.Engine):
        def __init__(self):
            super().__init__(directory, c.receipt(), manifest, allocation)
            self.store = make_store(original, c, directory, manifest, terminal, oldest)
    execution = dict(source_commit=source_commit, decision_identity=decision['identity'],
        manifest_identity=manifest['identity'], source_sha256=decision['source_sha256'],
        offline_control_sha256=control_sha256, automatic_retry=False)
    path = directory/'recovery_execution.json'
    if path.exists():
        require(load(path) == execution, 'Execution changed.')
    else:
        c.write(path, execution)
    engine = None
    try:
        engine = Engine()
        if stage == 'prefinal':
            result = original.prefinal(engine)
        elif stage == 'export-final':
            require(workbook is not None, 'Original workbook required.')
            c.load_final_observations(engine, workbook); result = dict(status='final_data_exported')
        else:
            require((directory/'final_observations.npz').exists(), 'Final data not frozen.')
            c.final_evaluation(engine, None); result = dict(status='final_completed')
        verify(directory, decision)
    except (Exception, KeyboardInterrupt) as exc:
        paused = isinstance(exc, (TimeoutError, KeyboardInterrupt))
        result = dict(status='paused' if paused else 'failed', reason=str(exc), traceback=traceback.format_exc())
        if not paused:
            c.write(directory/'RECOVERY_FAILED.json', result)
    finally:
        if engine is not None:
            engine.close()
    result.update(native_jobs=0 if engine is None else len(engine.store.events),
        scientific_acceptance=False, automatic_retry=False)
    number = len(list(directory.glob('recovery_report_*.json')))+1
    c.write(directory/f'recovery_report_{number:03d}.json', result)
    print(json.dumps(result), flush=True)
    return 2 if result['status'] == 'failed' else 0


def checkpoint(directory, decision, campaign, destination):
    original, c, manifest, terminal, oldest = verify(directory, decision)
    require(not destination.exists(), 'Never overwrite a checkpoint.')
    files = {}; jobs = {}
    for p in sorted(directory.rglob('*')):
        if not p.is_file() or p.name.endswith(('.lock', '.partial')):
            continue
        name = str(p.relative_to(directory)); raw = p.read_bytes()
        if name.startswith('jobs/') and p.suffix == '.npz':
            saved = c._read_array_checkpoint(raw)
            require(saved['token']['identity'] == manifest['identity'] and c.digest(saved['token']) == p.stem
                and saved['checksum'] == c.fingerprint({k:v for k,v in saved.items() if k != 'checksum'}),
                'Recovery raw token/checksum changed.')
            c.state_check(saved['prior_log_state']); c.state_check(saved['result']['log_state'])
            jobs[p.name] = hashlib.sha256(raw).hexdigest()
        files[name] = raw
    for p in sorted((directory/'adopted').glob('*.json')):
        receipt = c.load(p); token = receipt['original_token']
        predecessor = terminal if token['identity'] == load(terminal/'manifest.json')['identity'] else oldest
        key = receipt['original_key']; raw_path = predecessor/'jobs'/f'{key}.npz'
        saved = c._read_array_checkpoint(raw_path.read_bytes())
        require(c.sha(raw_path) == receipt['predecessor_raw_sha256']
            and saved['checksum'] == receipt['original_checksum']
            and saved['token'] == token and c.digest(token) == key
            and c.digest(receipt['requested_token']) == p.stem
            and receipt['requested_token']['identity'] == manifest['identity']
            and saved['checksum'] == c.fingerprint({k:v for k,v in saved.items() if k != 'checksum'})
            and receipt['exact_prior_fingerprint'] == c.fingerprint(saved['prior_log_state'])
            and receipt['old_checkpoint_relabelled'] is False, 'Adoption audit failed.')
    files['campaign_at_checkpoint.json'] = campaign.read_bytes()
    files['approved_decision.json'] = json.dumps(decision, separators=(',', ':')).encode()
    for name in (*manifest['source_sha256'], *manifest['continuation_source_sha256'], SOURCE):
        files['source/'+name] = (ROOT/name).read_bytes()
    index = dict(schema='f4-h2-durable-filesystem-recovery-v1', new_jobs=jobs,
        metadata_sha256={n:hashlib.sha256(raw).hexdigest() for n,raw in files.items()},
        terminal_archive_dependency=decision['terminal_checkpoint_archive'],
        full_predecessor_archive_chain_required=True, campaign_is_snapshot=True, scientific_acceptance=False)
    destination.parent.mkdir(parents=True, exist_ok=True); partial = destination.with_suffix('.zip.partial')
    with zipfile.ZipFile(partial, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name, raw in files.items():
            z.writestr(name, raw)
        z.writestr('durable_recovery.json', json.dumps(index, separators=(',', ':')))
    with zipfile.ZipFile(partial) as z:
        require(z.testzip() is None and len(z.namelist()) == len(set(z.namelist())), 'Archive corrupted.')
    with partial.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    fd = os.open(str(destination.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps(dict(path=str(destination), sha256=sha(destination),
        bytes=destination.stat().st_size, new_jobs=len(jobs), numerical_propagations=0)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decision', type=Path, required=True)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--stage', choices=('prepare-check', 'prefinal', 'export-final', 'final', 'checkpoint'), required=True)
    p.add_argument('--destination', type=Path); p.add_argument('--workbook', type=Path)
    p.add_argument('--control-sha256')
    a = p.parse_args(); decision = load(a.decision)
    require(decision['approved'] is True and decision['user_approval'].strip(), 'Recovery not authorized.')
    require(len(a.source_commit) == 40 and all(x in '0123456789abcdef' for x in a.source_commit), 'Pin immutable source commit.')
    require(sha(ROOT/SOURCE) == decision['source_sha256'][SOURCE], 'Unregistered controller.')
    canonical = json.dumps({k:v for k,v in decision.items() if k != 'identity'}, sort_keys=True,
        separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    require(hashlib.sha256(canonical.encode()).hexdigest() == decision['identity'], 'Decision changed.')
    require(decision['maximum_candidate_calls'] == {'2':25, '4':49} and decision['global_pair_cap'] == 73
        and decision['checkpoint_cap_seconds'] == 8 and decision['preparation_cap_seconds'] == 15
        and decision['invocation_cap_seconds'] == 600 and decision['stop_export_reserve_seconds'] == 60,
        'Registered limits changed.')
    directory = ROOT/decision['new_run_path']; native = a.stage in ('prepare-check', 'checkpoint')
    cap = (15 if a.stage == 'prepare-check' else 8) if native else 600
    if not native:
        ledger = validate(load(a.campaign), decision)
        require(total(e['charged_seconds'] for e in ledger['entries']
            if e['label'].startswith('native-development|'))+8 <= 1140, 'Protect next durable export before production.')
    with ExitStack() as locks:
        locks.enter_context(Lock(ROOT/decision['terminal_run_path']/'run.lock'))
        locks.enter_context(Lock(ROOT/decision['predecessor_campaign_path'].replace('.json', '.lock')/'run.lock'))
        if directory.exists():
            locks.enter_context(Lock(directory/'run.lock'))
        with reservation(a.campaign, decision, 'filesystem-recovery-'+a.stage, cap, native=native) as allocation:
            if a.stage == 'prepare-check':
                prepare_and_check(directory, decision); return 0
            if a.stage == 'checkpoint':
                require(a.destination is not None, 'Provide a unique durable destination.')
                checkpoint(directory, decision, a.campaign, a.destination); return 0
            return run(directory, decision, allocation, a.source_commit, a.stage, a.workbook, a.control_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
