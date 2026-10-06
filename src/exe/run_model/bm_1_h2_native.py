"""Bounded native H2 development only; never starts historical fitting.

One candidate at a time, eight persistent frequency workers, BLAS=1. Saved
precise references are read, not rerun. All invocations share the H ledger
and a cumulative 900-second native-development cap, including JIT/startup.
"""

from time import perf_counter, time
# Conservative interpreter-start allowance, then include imports, seed reads,
# JIT, workers, checkpoint writes and shutdown in the same invocation charge.
INVOCATION_BEGUN = perf_counter()-.25

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import threading
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'files/output/bm_1/runs/estimation'
SEED = OUT/'f4_h2_native_budget_seed_20261006.json'
PREFIX = 'native-development|'
NEW_SOURCE = ('src/method/bm_1_interval_native.py',
    'src/validation/bm_1/interval_native.py', 'src/exe/run_model/bm_1_h2_native.py',
    'root/f4/run_h2_native.ps1', 'root/f4/audit_h2_precise_pc.py')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ledger_valid(ledger):
    from src.exe.run_model.bm_1_h2_precise import PRIOR_COMPUTE_SECONDS
    _require(ledger['schema'] == 'f4-h2-campaign-budget-v1'
        and ledger['ceiling_seconds'] == 36000. and ledger['target_seconds'] == 21600.
        and math.isfinite(ledger['validation_reserve_seconds'])
        and ledger['validation_reserve_seconds'] >= 7200.
        and all(e['status'] in ('closed', 'reserved')
            and math.isfinite(e['charged_seconds']) and e['charged_seconds'] >= 0.
            for e in ledger['entries'])
        and ledger['charged_seconds'] == PRIOR_COMPUTE_SECONDS
            + sum(e['charged_seconds'] for e in ledger['entries']),
        'Invalid cumulative H campaign ledger.')


@contextmanager
def native_reservation(path, seconds, label, *, begun=None):
    begun = perf_counter() if begun is None else begun
    from src.exe.run_model.bm_1_h2_precise import _load, _write
    from src.tools.bm_1_reports import _RunLock
    with _RunLock(path.with_suffix('.lock')):
        ledger = _load(path) if path.exists() else _load(SEED)
        _ledger_valid(ledger)
        # A newer seed may extend the known PC ledger, never replace divergent
        # history or reset the cumulative time after moving between machines.
        if SEED.exists():
            seed = _load(SEED); _ledger_valid(seed)
            a, b = ledger['entries'], seed['entries']
            _require(a[:min(len(a), len(b))] == b[:min(len(a), len(b))],
                'PC/chat ledger histories diverged; reconcile before computing.')
            if len(b) > len(a):
                reserve = max(ledger['validation_reserve_seconds'], seed['validation_reserve_seconds'])
                ledger = seed; ledger['validation_reserve_seconds'] = reserve
        used = sum(e['charged_seconds'] for e in ledger['entries'] if e['label'].startswith(PREFIX))
        grant = min(seconds, 900.-used,
            36000.-ledger['validation_reserve_seconds']-ledger['charged_seconds'])
        _require(grant > 75., 'Native 900-second allowance cannot preserve the 60-second stop reserve.')
        entry = dict(label=PREFIX+label, started_utc_seconds=time(), status='reserved', charged_seconds=grant)
        ledger['entries'].append(entry); ledger['charged_seconds'] += grant
        _write(path, ledger)
        startup = perf_counter()-begun
        watchdog = threading.Timer(max(.01, grant-1.-startup), lambda: os._exit(124))
        watchdog.daemon = True; watchdog.start()
        try:
            yield dict(grant=grant, begun=begun, deadline=time()+max(0., grant-60.-startup),
                native_seconds_before=used, ledger=ledger)
        finally:
            from src.exe.run_model.bm_1_h2_precise import PRIOR_COMPUTE_SECONDS
            entry.update(status='closed', charged_seconds=perf_counter()-begun)
            ledger['charged_seconds'] = PRIOR_COMPUTE_SECONDS+sum(e['charged_seconds'] for e in ledger['entries'])
            _write(path, ledger); watchdog.cancel()


def _source():
    from src.exe.run_model.bm_1_h2_precise import _source as old_source, _sha
    return dict(old_source(), **{p: _sha(ROOT/p) for p in NEW_SOURCE})


def _environment():
    from src.exe.run_model.bm_1_h2_precise import _environment as old_environment
    import numba, llvmlite
    _require(numba.__version__ == '0.62.1' and llvmlite.__version__ == '0.45.1',
        'Use numba==0.62.1 and llvmlite==0.45.1 in the existing Python environment.')
    return dict(old_environment(), numba=numba.__version__, llvmlite=llvmlite.__version__,
        machine=platform.machine(), processor=platform.processor(), fastmath=False,
        workers=8, blas_threads=1, numba_threads=1)


def _references():
    """Read-only provenance: original hashes or exact checkout LF/CRLF only."""
    import numpy as np
    from src.exe.run_model.bm_1_h2_precise import _load, _sha
    from src.exe.run_model.bm_1_interval_terminal_tilt import TERMINAL_REFERENCE_DEPENDENCIES
    from src.method.bm_1_interval_precise import PRECISE_SETTINGS, freeze_hourly_record
    from src.method.bm_1_response import curve_pair_logits
    from src.validation.bm_1.interval_terminal_tilt import declared_terminal_control
    names = ('h2_hourly_interval_development_20261005.json',
        'h2_interval_positive_cn_control_20261005.json',
        'f4_h2_terminal_tilt_day_20261005.json.gz',
        'f4_h2_precise_pc_pilot_20261006/windows_review_20261006.zip')
    inputs = {str(OUT/n): _sha(OUT/n) for n in names}
    day_raw = gzip.decompress((OUT/names[2]).read_bytes())
    _require(hashlib.sha256(day_raw).hexdigest() ==
        'b35a376d28659e221c2ee65ca9b8ddee98e5d9c0ca2a57ab3642d5cb0b45676c', 'Archived day changed.')
    day = json.loads(day_raw)
    record, design = declared_terminal_control(_load(OUT/names[0]), _load(OUT/names[1]), 'day')
    _require(day['design'] == design and day['status'] == 'completed' and day['stage'] == 'day',
        'Archived day design/status changed.')
    canonical = lambda p: p.read_bytes().replace(b'\r\n', b'\n')
    for p in TERMINAL_REFERENCE_DEPENDENCIES:
        _require(hashlib.sha256(canonical(ROOT/p)).hexdigest() == day['source']['sha256'][p],
            'Archived day numerical dependency changed: '+p)
    for n in names[:2]:
        expected = [v for p, v in day['inputs_sha256'].items() if p.replace('\\', '/').endswith('/'+n)]
        _require(len(expected) == 1 and hashlib.sha256(canonical(OUT/n)).hexdigest() == expected[0],
            'Archived day input changed: '+n)
    prefix = canonical(ROOT/'src/validation/bm_1/interval_terminal_tilt.py').split(b'\ndef terminal_reference_control():')[0]
    _require(hashlib.sha256(prefix).hexdigest() ==
        '351b08fa915b125b9df8b08bab031d8c1c452aa231a94d67831989bf125e8292', 'Terminal inversion kernel changed.')
    loader = importlib.util.spec_from_file_location('h2_pc_reference_audit', ROOT/'root/f4/audit_h2_precise_pc.py')
    module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
    audit = module.audit(OUT/names[3])
    with zipfile.ZipFile(OUT/names[3]) as z:
        files = {i.filename.replace('\\', '/'): z.read(i) for i in z.infolist() if not i.is_dir()}
    run = 'h2_precise_development_pc_v1/'
    manifest = json.loads(files[run+'manifest.json'])
    candidate = dict(id='two_narrow', pieces=2,
        logits=curve_pair_logits(record.parameters[1], record.parameters[2], pieces=2).tolist())
    fine = [r for r in day['rows'] if r['setting_index'] == 1]
    _require(len(fine) == 24 and [r['interval_index'] for r in fine] == list(range(24)), 'Incomplete day.')
    cases = [dict(id='two_narrow', spec=freeze_hourly_record(record, PRECISE_SETTINGS),
        candidate=candidate, references=fine, units=24, full_logs_each_hour=True)]
    for name in ('linear', 'p4_c0_plus'):
        candidate = next(c for c in manifest['candidates'] if c['id'] == name)
        saved = json.loads(gzip.decompress(files[run+'candidates/'+name+'.json.gz']))['payload']
        cases.append(dict(id=name, spec=manifest['spec'], candidate=candidate,
            references=saved['rows'], final_reference_logs=saved['log_mass'], units=8, full_logs_each_hour=False))
    _require(audit['runtime_pilot_raw_verified'] is True, 'PC reference audit failed.')
    return cases, inputs


def _logs(values):
    import numpy as np
    return np.array([-np.inf if x is None else x for x in values])


def _comparison(row, logs, reference, reference_logs, cumulative, expected_cumulative):
    import numpy as np
    errors = dict(cumulative_score=abs(cumulative-expected_cumulative),
        mean=abs(row['state_mean']-reference.get('state_mean', reference.get('posterior_state_mean'))),
        hourly_score=0. if row['log_evidence'] is None else abs(row['log_evidence']-reference['log_evidence']))
    if reference_logs is not None:
        original = _logs(reference_logs)
        _require(np.array_equal(np.isneginf(logs), np.isneginf(original)), 'True-zero support differs from reference.')
        active = np.isfinite(original)
        errors['finite_log_mass'] = float(np.max(np.abs(logs[active]-original[active])))
    _require(all(math.isfinite(v) and v <= 1e-7 for v in errors.values()),
        'Native/reference equivalence failed: '+str(errors))
    return errors


def _worker_init(deadline):
    import numba
    numba.set_num_threads(1)
    timer = threading.Timer(max(0., deadline-time()), lambda: os._exit(124))
    timer.daemon = True; timer.start()


def _run(args, reservation):
    import numpy as np
    from scipy.special import logsumexp
    from src.exe.run_model.bm_1_h2_precise import _load, _sha, _write
    from src.method.bm_1_interval_precise import digest, replay_units, validate_checkpoint
    from src.method.bm_1_interval_native import freeze_native_operator, native_terminal_update, native_prediction, advance_frequency_block
    from src.validation.bm_1.interval_native import algebra_control
    report_path = args.run_dir/'report.json'
    if report_path.exists():
        previous = _load(report_path)
        _require(previous['status'] not in ('numerical_guard_failed', 'failed', 'completed'),
            'Preserve the completed/failed native run; no automatic repetition.')
    _require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT).strip(),
        'Commit tracked source before freezing native execution.')
    cases, inputs = _references()
    source, environment = _source(), _environment()
    manifest = dict(schema='f4-h2-native-development-v1', source_sha256=source,
        environment=environment, inputs_sha256=inputs, cases=cases, campaign=str(args.campaign),
        development=True, historical_observation_values_read=0, scientific_acceptance=False)
    manifest['identity'] = digest(manifest)
    path = args.run_dir/'manifest.json'
    if path.exists():
        _require(args.resume, 'Native checkpoint exists; declare --resume with identical source/environment.')
        _require(_load(path) == manifest, 'Frozen native manifest/environment/inputs changed.')
    else:
        _require(not args.resume, 'No frozen native run exists to resume.')
        _write(path, manifest)
    def unchanged():
        _require(_source() == source and all(_sha(p) == v for p, v in inputs.items()),
            'Frozen native source/reference changed during execution.')
    control_path = args.run_dir/'algebra.json'
    control_identity = digest(dict(source={p:source[p] for p in NEW_SOURCE[:2]}, environment=environment))
    if control_path.exists():
        control = _load(control_path)
        _require(control['identity'] == control_identity and control['passed'] is True
            and control['checksum'] == digest(control['results']), 'Saved independent control changed.')
    elif args.reuse_controls:
        control = _load(args.reuse_controls)
        _require(control['identity'] == control_identity and control['passed'] is True
            and control['checksum'] == digest(control['results']), 'Independent control provenance differs.')
        _write(control_path, control)
    else:
        _require(time() < reservation['deadline'], 'Budget expired before native controls.')
        try:
            results = algebra_control()
        except Exception as exc:
            _write(report_path, dict(status='numerical_guard_failed', reason=str(exc),
                manifest_identity=manifest['identity'], equivalence_pass=False))
            raise
        control = dict(identity=control_identity, passed=True, results=results, checksum=digest(results))
        _write(control_path, control)
    def stopped():
        return time() >= reservation['deadline'] or (args.run_dir/'STOP').exists()
    timing_path = args.run_dir/'hour_wall_times.json'
    timings = _load(timing_path) if timing_path.exists() else dict(manifest_identity=manifest['identity'], rows=[])
    _require(timings['manifest_identity'] == manifest['identity'], 'Timing manifest changed.')
    pool = mp.get_context('spawn').Pool(8, initializer=_worker_init, initargs=(reservation['deadline'],))
    def mapper(jobs):
        left = reservation['deadline']-time()
        if left <= 0. or stopped():
            raise mp.TimeoutError('Native computation deadline or STOP request.')
        pending = pool.map_async(advance_frequency_block, jobs)
        while not stopped():
            try:
                return pending.get(timeout=min(.5, max(.001, reservation['deadline']-time())))
            except mp.TimeoutError:
                pass
        raise mp.TimeoutError('Native computation deadline or STOP request.')
    stats, status, error = [], 'completed', None
    try:
        for case in cases:
            spec, candidate = case['spec'], case['candidate']
            checkpoint_path = args.run_dir/(case['id']+'.json.gz')
            if checkpoint_path.exists():
                saved = _load(checkpoint_path)
                _require(saved['native_manifest_identity'] == manifest['identity'], 'Native checkpoint source changed.')
                payload = validate_checkpoint(saved, spec, candidate)
                _require(payload['status'] != 'numerical_guard_failed', 'Failed native candidate cannot resume.')
            else:
                payload = dict(rows=[], log_mass=spec['initial_log_mass'], log_likelihood=0., status='paused')
            rows = payload['rows']; logs = _logs(payload['log_mass']); units = replay_units(spec)
            def reference_logs(index):
                return (case['references'][index]['log_posterior_mass'] if case['full_logs_each_hour'] else
                    case['final_reference_logs'] if index+1 == case['units'] else None)
            def verify_rows():
                for i, row in enumerate(rows):
                    state = _logs(row['log_posterior_mass'])
                    _require(len(state) == spec['settings']['cells'] and not np.isnan(state).any()
                        and not np.isposinf(state).any() and abs(float(logsumexp(state))) <= 1e-10,
                        'Invalid intermediate native log state.')
                    _require(all(row[k] <= bound for k, bound in (
                        ('maximum_relative_conditional_guard', 1e-8), ('maximum_overlap_log_difference', 1e-7),
                        ('maximum_abs_mass_error', 1e-10))), 'Saved native guards failed.')
                    expected = sum(r['log_evidence'] for r in case['references'][:i+1] if r['log_evidence'] is not None)
                    actual = sum(r['log_evidence'] for r in rows[:i+1] if r['log_evidence'] is not None)
                    _require(row['reference_errors'] == _comparison(row, state, case['references'][i],
                        reference_logs(i), actual, expected), 'Saved native equivalence diagnostics changed.')
                if rows:
                    _require(rows[-1]['log_posterior_mass'] == payload['log_mass'], 'Latest native state differs.')
            verify_rows()
            previous_timings = [r for r in timings['rows'] if r['case'] == case['id']]
            # A crash after the atomic state write may lose that hour's timing
            # row. It does not lose state; incomplete timing cannot admit a fit.
            _require([r['unit'] for r in previous_timings] == sorted(set(r['unit'] for r in previous_timings))
                and all(0 <= r['unit'] < len(rows) for r in previous_timings)
                and all(math.isfinite(r['wall_seconds']) and r['wall_seconds'] > 0. for r in previous_timings),
                'Invalid saved native timing chronology.')
            def publish(case_status):
                unchanged()
                payload.update(rows=rows, log_mass=[None if np.isneginf(v) else float(v) for v in logs],
                    log_likelihood=float(sum(r['log_evidence'] for r in rows if r['observed'])), status=case_status)
                saved = dict(identity=digest(dict(spec=spec, candidate=candidate)), payload=payload,
                    checksum=digest(payload), native_manifest_identity=manifest['identity'])
                validate_checkpoint(saved, spec, candidate); _write(checkpoint_path, saved)
            for index in range(len(rows), case['units']):
                if stopped():
                    raise mp.TimeoutError('Native computation deadline or STOP request.')
                begun = perf_counter(); unit = units[index]
                operator = freeze_native_operator(spec, candidate, unit)
                if unit['observed']:
                    result = native_terminal_update(operator, logs, unit['value'], spec['physical']['sigma_y'],
                        spec['settings'], mapper=mapper)
                    next_logs, score = result['log_posterior_mass'], result['log_evidence']
                else:
                    operator['reward_half'].fill(0.)
                    result = native_prediction(operator, logs, frequencies=1, span=2., mapper=mapper)
                    next_logs, score = result['log_amplitude'], None
                    result.update(contours=[], maximum_relative_conditional_guard=0., maximum_overlap_log_difference=0.)
                row = dict(unit=index, **unit, log_evidence=score,
                    state_mean=float(np.exp(next_logs) @ operator['x']),
                    **{k:result[k] for k in ('contours', 'maximum_relative_conditional_guard',
                        'maximum_overlap_log_difference', 'maximum_abs_mass_error')})
                expected = sum(r['log_evidence'] for r in case['references'][:index+1] if r['log_evidence'] is not None)
                cumulative = sum(r['log_evidence'] for r in rows if r['log_evidence'] is not None)+(score or 0.)
                row['reference_errors'] = _comparison(row, next_logs, case['references'][index],
                    reference_logs(index), cumulative, expected)
                logs = next_logs
                row['log_posterior_mass'] = [None if np.isneginf(v) else float(v) for v in logs]
                # Checkpoint writing is included separately in the measured hour.
                row['elapsed_seconds'] = perf_counter()-begun
                rows.append(row); publish('completed' if len(rows) == case['units'] else 'paused')
                wall = perf_counter()-begun
                checkpoint_wall = wall-row['elapsed_seconds']
                timings['rows'].append(dict(case=case['id'], unit=index, wall_seconds=wall,
                    includes_operator_ipc_checkpoint=True))
                _write(timing_path, timings)
                print(f"{case['id']}: {index+1}/{case['units']} {row['elapsed_seconds']:.3f}s + checkpoint {checkpoint_wall:.3f}s", flush=True)
            stats.append(dict(case=case['id'], units=len(rows), observed=sum(r['observed'] for r in rows),
                hour_seconds=[r['elapsed_seconds'] for r in rows],
                hour_wall_seconds=[r['wall_seconds'] for r in timings['rows'] if r['case'] == case['id']],
                timing_complete=sum(r['case'] == case['id'] for r in timings['rows']) == len(rows),
                maximum_reference_errors={k:max(r['reference_errors'].get(k, 0.) for r in rows)
                    for k in ('cumulative_score', 'mean', 'hourly_score', 'finite_log_mass')}))
    except mp.TimeoutError as exc:
        status, error = 'paused', str(exc)
    except Exception as exc:
        status, error = 'numerical_guard_failed', str(exc)
        _write(args.run_dir/'failure.json', dict(reason=str(exc), traceback=traceback.format_exc()))
    finally:
        pool.terminate(); pool.join()
    unchanged()
    report = dict(schema='f4-h2-native-development-report-v1', manifest_identity=manifest['identity'],
        status=status, reason=error, cases=stats, workers=8, blas_threads=1,
        elapsed_seconds=perf_counter()-reservation['begun'], native_seconds_before=reservation['native_seconds_before'],
        grant_seconds=reservation['grant'], stop_reserve_seconds=60.,
        equivalence_pass=status == 'completed', historical_observation_values_read=0,
        numerical_fit_readiness=False, scientific_acceptance=False,
        next_gate='Actual PC throughput, completed historical F contract and full priced task graph; no fit authorized.')
    _write(report_path, report)
    print(json.dumps({k:v for k,v in report.items() if k != 'cases'}, ensure_ascii=False), flush=True)
    return 0 if status == 'completed' else 3 if status == 'paused' else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, default=OUT/'f4_h2_remaining_budget.json')
    parser.add_argument('--seconds', type=float, default=900.)
    parser.add_argument('--reuse-controls', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    _require(math.isfinite(args.seconds) and 75. < args.seconds <= 900., 'Declare 75 < seconds <= 900.')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_DEFAULT_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    args.run_dir, args.campaign = args.run_dir.resolve(), args.campaign.resolve()
    os.chdir(ROOT)
    from src.tools.bm_1_reports import _RunLock
    with native_reservation(args.campaign, args.seconds, str(args.run_dir), begun=INVOCATION_BEGUN) as reservation, _RunLock(args.run_dir):
        try:
            return _run(args, reservation)
        except Exception as exc:
            from src.exe.run_model.bm_1_h2_precise import _write
            # Preserve an earlier report/checkpoint if preflight rejected a
            # changed source or completed run. Failure gets its own evidence.
            _write(args.run_dir/'preflight_failure.json', dict(status='failed', reason=str(exc),
                traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    raise SystemExit(main())
