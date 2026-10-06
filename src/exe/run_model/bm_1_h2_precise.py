"""F4-H2 bounded precise candidate batch. Parallelize candidates, never days.

The default development pilot reads saved synthetic inputs, not workbooks.
Historical preparation requires the existing explicit F contract. Acceptance,
model selection and final holdout are deliberately not inferred from this batch.
"""

import argparse
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
from time import perf_counter, sleep, time


ROOT = Path(__file__).resolve().parents[3]
PRIOR_COMPUTE_SECONDS = 425.8919726690074
DEFAULT_CAMPAIGN = Path('files/output/bm_1/runs/estimation/f4_h2_remaining_budget.json')
LEDGER_SEED = ROOT/'files/output/bm_1/runs/estimation/f4_h2_budget_seed_20261006.json'
SOURCE_FILES = (
    'src/exe/run_model/bm_1_h2_precise.py', 'src/method/bm_1_interval_precise.py',
    'src/validation/bm_1/interval_terminal_tilt.py', 'src/exe/run_model/bm_1/curves.py',
    'src/exe/run_model/bm_1_history.py', 'src/exe/run_model/bm_1_interval_terminal_tilt.py',
    'src/tools/bm_1_data.py', 'src/tools/bm_1_history_io.py', 'src/tools/bm_1_reports.py',
    'src/tools/checkpoint_io.py', 'src/params/bm_1_history.py', 'src/params/bm_1_interval.py',
    'src/control/bm_1.py', 'src/control/objective.py', 'src/exe/run_model/progress.py')


def _load(path):
    from src.tools.bm_1_reports import _read_likelihood_report
    return _read_likelihood_report(Path(path))[0]


def _sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _source():
    from src.exe.run_model.bm_1_interval_terminal_tilt import TERMINAL_REFERENCE_DEPENDENCIES
    files = set(SOURCE_FILES) | set(TERMINAL_REFERENCE_DEPENDENCIES)
    return {p: _sha(ROOT/p) for p in sorted(files)}


def _environment():
    import numpy, pandas, scipy
    return dict(python=platform.python_version(), numpy=numpy.__version__,
                scipy=scipy.__version__, pandas=pandas.__version__, platform=platform.platform())


def _write(path, value):
    from src.tools.bm_1_reports import _atomic_report
    _atomic_report(Path(path), value, replace_existing=True)


@contextmanager
def campaign_reservation(path, requested_seconds, label):
    """Charge a whole invocation BEFORE admission; refund only on orderly exit.

    A parent crash keeps the reservation charged. Worker watchdogs bound their
    lifetime even without a parent. The common ledger is locked across runs.
    """
    from src.tools.bm_1_reports import _RunLock
    path = Path(path).resolve()
    with _RunLock(path.with_suffix('.lock')):
        ledger = (_load(path) if path.exists() else _load(LEDGER_SEED) if LEDGER_SEED.exists() else dict(schema='f4-h2-campaign-budget-v1',
            ceiling_seconds=36000., target_seconds=21600., validation_reserve_seconds=7200.,
            charged_seconds=PRIOR_COMPUTE_SECONDS, entries=[]))
        if (ledger['schema'] != 'f4-h2-campaign-budget-v1' or ledger['ceiling_seconds'] != 36000.
                or ledger['target_seconds'] != 21600. or ledger['validation_reserve_seconds'] != 7200.
                or not math.isfinite(ledger['charged_seconds'])
                or ledger['charged_seconds'] < PRIOR_COMPUTE_SECONDS
                or ledger['charged_seconds'] != PRIOR_COMPUTE_SECONDS+sum(e['charged_seconds'] for e in ledger['entries'])):
            raise ValueError('Campaign ledger changed or failed its cumulative accounting check.')
        available = ledger['ceiling_seconds']-ledger['validation_reserve_seconds']-ledger['charged_seconds']
        grant = min(float(requested_seconds), available)
        if grant < 15.:
            raise ValueError('Remaining H budget cannot admit another invocation while preserving validation reserve.')
        begun = perf_counter(); before = ledger['charged_seconds']
        entry = dict(label=label, started_utc_seconds=time(), status='reserved', charged_seconds=grant)
        ledger['entries'].append(entry); ledger['charged_seconds'] += grant
        _write(path, ledger)
        watchdog = threading.Timer(max(0., grant-1.), lambda: os._exit(124))
        watchdog.daemon = True; watchdog.start()
        try:
            yield dict(grant=grant, before=before, begun=begun, ledger=ledger)
        finally:
            # Abrupt process death cannot execute this refund; reservation stays.
            elapsed = perf_counter()-begun
            entry.update(status='closed', charged_seconds=elapsed)
            ledger['charged_seconds'] = PRIOR_COMPUTE_SECONDS+sum(e['charged_seconds'] for e in ledger['entries'])
            _write(path, ledger)
            watchdog.cancel()


def configuration_template():
    from src.exe.run_model.bm_1_history import configuration_template as old_template
    from src.method.bm_1_interval_precise import PRECISE_SETTINGS
    config = old_template(hourly_assumptions=True)
    config['numerics'].update(cells=1921, max_step=1/24576)
    config['fit']['families'][0]['max_evaluations'] = 4
    config['fit']['families'][1]['max_evaluations'] = 12
    config['precise_search'] = dict(settings=PRECISE_SETTINGS, logit_step=.5,
        candidate_ceiling=17, pilot_candidates=8, pilot_train_intervals=8,
        convergence_claim=False, validation_reserve_seconds=7200.)
    config['runner'] = 'src.exe.run_model.bm_1_h2_precise; do not run legacy history-fit'
    return config


def _freeze(args, *, development):
    import numpy as np
    from src.method.bm_1_interval_precise import PRECISE_SETTINGS, candidate_panel, digest, freeze_hourly_record
    from src.tools.bm_1_reports import _atomic_report
    if (args.run_dir/'manifest.json').exists():
        raise FileExistsError('Preserve the prepared run; use --resume for its pilot.')
    source = _source()
    git = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, check=True, capture_output=True, text=True)
    if git.stdout.strip():
        raise ValueError('Use clean committed source; do not modify a checkout used by a running study.')
    inputs, provenance = {}, {}
    step, penalty = .5, 0.
    if development:
        from src.validation.bm_1.interval_characteristic import record_from_development_report
        path = ROOT/'files/output/bm_1/runs/estimation/h2_hourly_interval_development_20261005.json'
        record = replace(record_from_development_report(_load(path)), sigma_y=.04)
        inputs[str(path)] = _sha(path)
        provenance = dict(scope='saved synthetic day; sigma_y=.04 narrow-noise condition',
                          historical_observation_values_read=0)
    else:
        from src.exe.run_model.bm_1_history import prepare_training
        raw = args.config.read_bytes(); config = json.loads(raw)
        declared = config.get('precise_search')
        if (not declared or declared['settings'] != PRECISE_SETTINGS or declared['candidate_ceiling'] != 17
                or declared['pilot_candidates'] != 8 or declared['pilot_train_intervals'] != 8
                or declared['convergence_claim'] is not False or declared['validation_reserve_seconds'] != 7200.):
            raise ValueError('Use the declared 17-candidate precise batch and unchanged guards.')
        record, provenance, _, fit, paths = prepare_training(config, root=args.input_root)
        inputs = {str(p): _sha(p) for p in paths.values()}
        inputs[str(args.config.resolve())] = hashlib.sha256(raw).hexdigest()
        step, penalty = declared['logit_step'], fit['regularization']
    spec = freeze_hourly_record(record, PRECISE_SETTINGS)
    if np.sum(spec['observed'][:8]) == 0 or len(spec['starts']) < 8:
        raise ValueError('Pilot needs at least eight complete TRAIN intervals with observed evidence.')
    manifest = dict(schema='f4-h2-precise-batch-v1', source_sha256=source, environment=_environment(),
        commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip(),
        inputs_sha256=inputs, provenance=provenance, development=development,
        spec=spec, candidates=candidate_panel(step), regularization=penalty,
        numerical_fit_readiness=False, scientific_acceptance=False, campaign=str(args.campaign.resolve()))
    manifest['identity'] = digest(manifest)
    if source != _source() or any(_sha(p) != sha for p, sha in inputs.items()):
        raise ValueError('Source or input changed during preparation.')
    _atomic_report(args.run_dir/'manifest.json', manifest)
    print(f'Prepared {len(spec["starts"])} TRAIN intervals, 17 shared-curve candidates; no likelihood run.', flush=True)
    return manifest


def _validate_manifest(path, campaign):
    from src.method.bm_1_interval_precise import digest
    manifest = _load(path)
    content = {k: v for k, v in manifest.items() if k != 'identity'}
    if (manifest['schema'] != 'f4-h2-precise-batch-v1' or manifest['identity'] != digest(content)
            or manifest['source_sha256'] != _source() or manifest['environment'] != _environment()
            or manifest['campaign'] != str(Path(campaign).resolve())):
        raise ValueError('Frozen manifest, numerical source, environment or campaign changed.')
    if any(_sha(p) != sha for p, sha in manifest['inputs_sha256'].items()):
        raise ValueError('Frozen original inputs changed.')
    return manifest


def _worker(args):
    from src.method.bm_1_interval_precise import replay_precise_candidate
    # Timed death retains the last atomic hour if an interval stalls or the parent disappears.
    timer = threading.Timer(max(0., args.deadline-time()), lambda: os._exit(124))
    timer.daemon = True; timer.start()
    manifest = _validate_manifest(args.run_dir/'manifest.json', args.campaign)
    candidate = next(c for c in manifest['candidates'] if c['id'] == args.worker)
    path = args.run_dir/'candidates'/f'{candidate["id"]}.json.gz'
    saved = _load(path) if path.exists() else None
    result = replay_precise_candidate(manifest['spec'], candidate, saved=saved,
        limit_units=args.limit_units,
        should_pause=lambda: time() >= args.deadline-2. or (args.run_dir/'STOP').exists(),
        checkpoint=partial(_write, path))
    if manifest['source_sha256'] != _source():
        raise ValueError('Source changed during candidate replay.')
    timer.cancel()
    return 2 if result['payload']['status'] == 'numerical_guard_failed' else 0


def _snapshot(manifest, directory):
    from src.method.bm_1_interval_precise import validate_checkpoint
    rows = {}
    for candidate in manifest['candidates']:
        path = directory/'candidates'/f'{candidate["id"]}.json.gz'
        if path.exists():
            rows[candidate['id']] = validate_checkpoint(_load(path), manifest['spec'], candidate)
    return rows


def _projection(manifest, rows, elapsed, workers, charged_before):
    """Conservative planning envelope, not a historical runtime guarantee."""
    from src.method.bm_1_interval_precise import replay_units
    pilot_ids = [c['id'] for c in manifest['candidates'][:8]]
    all_units = replay_units(manifest['spec'])
    pilot_units = next(i+1 for i, r in enumerate(all_units) if r['observation_index'] == 7)
    complete = all(name in rows and len(rows[name]['rows']) >= pilot_units
                   and rows[name]['status'] != 'numerical_guard_failed' for name in pilot_ids)
    if not complete:
        return dict(available=False, budget_admissible=False, reason='Incomplete eight-candidate/eight-interval pilot.')
    observed = [r for name in pilot_ids for r in rows[name]['rows'][:pilot_units] if r['observed']]
    worst_hour = max(r['elapsed_seconds'] for r in observed)
    # Measured wall startup/checkpoint/contention is included conservatively by
    # charging the whole pilot elapsed again per full batch wave.
    future = 1.5*math.ceil(17/workers)*(len(all_units)*worst_hour+elapsed)
    total = charged_before+elapsed+future+7200.
    return dict(available=True, pilot_complete=True, worst_observed_interval_seconds=worst_hour,
        total_train_units=len(all_units), pilot_units=pilot_units, scheduling_waves=math.ceil(17/workers),
        planning_safety_factor=1.5, projected_search_seconds=future,
        projected_H_seconds_including_reserved_validation=total,
        target_6h_met=total <= 21600., budget_admissible=total <= 36000.,
        historical_rate_measured=not manifest['development'],
        interpretation='Planning only; unmeasured candidates/tails and later hours may cost more. Hard ledger remains binding.')


def _run_batch(args, reservation, manifest):
    from src.exe.run_model.progress import runtime_text
    from src.method.bm_1_interval_precise import replay_units
    rows = _snapshot(manifest, args.run_dir)
    if any(r['status'] == 'numerical_guard_failed' for r in rows.values()):
        raise ValueError('Preserve failed candidate evidence; do not retry the batch automatically.')
    units = replay_units(manifest['spec'])
    pilot_units = next(i+1 for i, u in enumerate(units) if u['observation_index'] == 7)
    pilot = args.command in ('development-pilot', 'pilot')
    if not pilot:
        report = _load(args.run_dir/'pilot_report.json')
        if (manifest['development'] or report['status'] != 'completed' or report['manifest_identity'] != manifest['identity']
                or report['workers'] != args.workers or not report['projection']['budget_admissible']
                or not report['projection']['historical_rate_measured']):
            raise ValueError('Search requires a matching completed historical parallel pilot with an admissible budget.')
        remaining_projection = report['projection']['projected_search_seconds']+7200.+reservation['before']
        if remaining_projection > 36000.:
            raise ValueError('Measured plan no longer fits the remaining H campaign budget.')
    candidates = manifest['candidates'][:8] if pilot else manifest['candidates']
    target_units = pilot_units if pilot else len(units)
    # Known cost proxy: missing/prediction units use two transform columns;
    # observed units use K+1 per contour. Future contour counts remain unknown.
    weights = [(u['end']-u['start'])*(manifest['spec']['settings']['frequencies']+1
                if u['observed'] else 2) for u in units[:target_units]]
    total_work = len(candidates)*sum(weights)
    pending = [c for c in candidates if len(rows.get(c['id'], {}).get('rows', [])) < target_units]
    previous_report = args.run_dir/'pilot_report.json'
    if pilot and not pending and previous_report.exists():
        previous = _load(previous_report)
        if previous['manifest_identity'] != manifest['identity'] or previous['workers'] != args.workers:
            raise ValueError('Preserve measured worker profile; use a separate run for another worker count.')
        print(f'Completed pilot reused without new likelihood work: {previous_report}', flush=True)
        return 0
    profile = args.run_dir/'runtime_profile.json'
    if profile.exists() and _load(profile)['workers'] != args.workers:
        raise ValueError('Worker count is frozen for a resumable measured pilot/search.')
    _write(profile, dict(workers=args.workers, blas_threads=1))
    initial = sum(min(len(rows.get(c['id'], {}).get('rows', [])), target_units) for c in candidates)
    initial_work = sum(sum(weights[:min(len(rows.get(c['id'], {}).get('rows', [])), target_units)])
                       for c in candidates)
    active, status = {}, 'completed'
    deadline = reservation['begun']+reservation['grant']-10.
    epoch_deadline = time()+max(0., deadline-perf_counter())
    next_print = 0.
    def stop_children():
        for proc, handle in active.values():
            if proc.poll() is None: proc.terminate()
        for proc, handle in active.values():
            try: proc.wait(timeout=2.)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=2.)
            handle.close()
        active.clear()
    try:
        while pending or active:
            if perf_counter() >= deadline or (args.run_dir/'STOP').exists():
                status = 'budget_paused' if perf_counter() >= deadline else 'stop_paused'; break
            while pending and len(active) < args.workers:
                candidate = pending.pop(0)
                log_path = args.run_dir/'logs'/f'{candidate["id"]}.log'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open('ab')
                command = [sys.executable, '-B', '-m', 'src.exe.run_model.bm_1_h2_precise',
                    '--worker', candidate['id'], '--run-dir', str(args.run_dir),
                    '--campaign', str(args.campaign), '--deadline', str(epoch_deadline),
                    '--limit-units', str(target_units)]
                active[candidate['id']] = (subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                                           stderr=subprocess.STDOUT), handle)
            for name, (proc, handle) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    handle.close(); del active[name]
                    if code != 0:
                        status = 'candidate_failed'; break
            if status != 'completed': break
            if perf_counter() >= next_print:
                rows = _snapshot(manifest, args.run_dir)
                done = sum(min(len(rows.get(c['id'], {}).get('rows', [])), target_units) for c in candidates)
                done_work = min(total_work, sum(sum(weights[:min(len(rows.get(c['id'], {}).get('rows', [])),
                    target_units)]) for c in candidates))
                print(f'F4-H2 {done}/{len(candidates)*target_units} candidate-hours; '+runtime_text(
                    reservation['begun'], done_work, total_work, initial_completed=initial_work), flush=True)
                next_print = perf_counter()+30.
            sleep(.2)
    except KeyboardInterrupt:
        status = 'interrupt_paused'
    finally:
        stop_children()
    _validate_manifest(args.run_dir/'manifest.json', args.campaign)
    rows = _snapshot(manifest, args.run_dir)
    if status == 'completed' and any(len(rows.get(c['id'], {}).get('rows', [])) != target_units for c in candidates):
        status = 'incomplete'
    if any(r['status'] == 'numerical_guard_failed' for r in rows.values()): status = 'numerical_guard_failed'
    elapsed = perf_counter()-reservation['begun']
    past_pilot_seconds = sum(e['charged_seconds'] for e in reservation['ledger']['entries'][:-1]
        if e['label'] in (f'pilot|{args.run_dir}', f'development-pilot|{args.run_dir}'))
    pilot_wall = past_pilot_seconds+elapsed if pilot else _load(previous_report)['pilot_total_wall_seconds']
    projection = _projection(manifest, rows, pilot_wall, args.workers, reservation['before'])
    report = dict(schema='f4-h2-precise-batch-report-v1', manifest_identity=manifest['identity'],
        status=status, workers=args.workers, blas_threads=1, development=manifest['development'],
        elapsed_seconds=elapsed, pilot_total_wall_seconds=pilot_wall,
        completed_units={name: len(r['rows']) for name, r in rows.items()},
        reused_units=initial, projection=projection, numerical_fit_readiness=False,
        scientific_acceptance=False, optimizer_converged=False, trained_model_exported=False,
        campaign=str(args.campaign), best_evaluated_by_family={})
    if not pilot and status == 'completed':
        from src.method.bm_1_response import curve_pair_from_logits
        for pieces in (2, 4):
            scores = []
            for candidate in manifest['candidates']:
                if candidate['id'] != 'linear' and candidate['pieces'] != pieces: continue
                f, g = curve_pair_from_logits(candidate['logits'], candidate['pieces'])
                penalty = manifest['regularization']*sum(float(((curve.values-(1.-2.*curve.knots))**2).sum()) for curve in (f, g))
                scores.append(dict(candidate=candidate['id'], log_likelihood=rows[candidate['id']]['log_likelihood'],
                    penalty=penalty, penalized_score=rows[candidate['id']]['log_likelihood']-penalty,
                    f=f.to_record(), g=g.to_record()))
            report['best_evaluated_by_family'][str(pieces)] = max(scores, key=lambda r: r['penalized_score'])
        report['interpretation'] = 'TRAIN coordinate probe only; no model-family selection or convergence claim.'
    _write(args.run_dir/('pilot_report.json' if pilot else 'search_report.json'), report)
    print(f'F4-H2 {status}; report={args.run_dir/("pilot_report.json" if pilot else "search_report.json")}', flush=True)
    return 2 if 'failed' in status else 0


def main(argv=None):
    for name in ('OPENBLAS_NUM_THREADS', 'OPENBLAS_DEFAULT_NUM_THREADS', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=('config', 'prepare', 'development-pilot', 'pilot', 'search'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--input-root', type=Path, default=ROOT)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--limit-units', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.run_dir is None or args.deadline is None or args.limit_units is None:
            parser.error('Incomplete internal worker invocation.')
        return _worker(args)
    if args.command is None or not 1 <= args.workers <= 8:
        parser.error('Choose a command and 1..8 workers; BLAS is fixed at 1.')
    if args.command == 'config':
        from src.tools.bm_1_reports import _atomic_report
        if args.output is None: parser.error('config requires --output.')
        _atomic_report(args.output, configuration_template())
        print(f'Incomplete F4-H2 contract: {args.output}; no scientific metadata guessed.', flush=True)
        return 0
    if args.run_dir is None or (args.command == 'prepare' and args.config is None):
        parser.error('Supply --run-dir and, for prepare, --config.')
    requested = args.seconds if args.seconds is not None else (14400. if args.command == 'search' else 900.)
    maximum = 21600. if args.command == 'search' else 900.
    if not math.isfinite(requested) or not 15. <= requested <= maximum:
        parser.error(f'Invocation seconds must be finite, between 15 and {maximum:g}.')
    args.run_dir, args.campaign = args.run_dir.resolve(), args.campaign.resolve()
    from src.tools.bm_1_reports import _RunLock
    with campaign_reservation(args.campaign, requested, f'{args.command}|{args.run_dir}') as reservation, _RunLock(args.run_dir):
        path = args.run_dir/'manifest.json'
        if args.command == 'prepare':
            _freeze(args, development=False); return 0
        if args.command == 'development-pilot' and not args.resume:
            manifest = _freeze(args, development=True)
        else:
            if not args.resume: parser.error('Prepared pilots/search require --resume.')
            manifest = _validate_manifest(path, args.campaign)
        return _run_batch(args, reservation, manifest)


if __name__ == '__main__':
    raise SystemExit(main())
