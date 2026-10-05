"""Resumable H1 numerical screening, independent of optional G studies.

python -B -m src.exe.run_model.bm_1.curves numerics --run-dir PATH --blas-threads 1
Use --limit-units 1 for a pilot. Resume the same clean source with --resume.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def _configure_blas(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--blas-threads', type=int, default=1)
    args, _ = parser.parse_known_args(argv)
    if args.blas_threads < 1:
        parser.error('--blas-threads must be positive')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'OPENBLAS_DEFAULT_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.blas_threads)


if __name__ == '__main__':
    _configure_blas(sys.argv[1:])

import numpy as np

from src.tools.bm_1_reports import _RunLock, _atomic_report, _read_likelihood_report
from src.exe.run_model.progress import runtime_text
from src.validation.bm_1.response_cases import (
    H1_IDS, H1_NUMERICS, H1_V2_NUMERICS, h1_design, h1_candidate_panel, h1_training_records, response_price_support,
)
from src.validation.bm_1.response_numerics import (
    run_numerical_unit, validate_filter_arrays, compare_numerical_unit,
    truth_diagnostic, negative_control_error,
)

ROOT = Path(__file__).resolve().parents[4]


def source_snapshot():
    def git(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return dict(commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain')),
        sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT/'src').rglob('*.py'))})


def _observation_digest(record):
    h = hashlib.sha256()
    for values in (record.observations.times, record.observations.values, record.observations.observed):
        h.update(np.asarray(values).tobytes())
    return h.hexdigest()



class _NumericalJobs:
    """Bounded process queue; drain admitted units before a time-budget pause.

    Workers return raw arrays only. The parent validates and atomically publishes
    in registered order. At most workers units are in flight, and a hard crash
    can lose those unpublished units, never an already-published unit.
    """
    def __init__(self, specs, workers, deadline):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        self.specs = iter(specs)
        self.deadline = deadline
        self.pending = []
        self.executor = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'))
        for _ in range(workers):
            self._submit()

    def _submit(self):
        from src.validation.bm_1.response_jobs import run_h1_job
        if perf_counter() >= self.deadline:
            return
        spec = next(self.specs, None)
        if spec is not None:
            self.pending.append((spec['unit'], self.executor.submit(run_h1_job, spec)))

    def take(self, unit):
        if not self.pending:
            raise StopIteration
        expected, future = self.pending.pop(0)
        if unit != expected:
            raise ValueError('Parallel H1 job order differs from registered unit order.')
        value = future.result()
        self._submit()
        return value

    def close(self):
        for _, future in self.pending:
            future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)


def run_numerics(args):
    from contextlib import ExitStack
    from src.validation.estimation import positive_integer
    workers = positive_integer(getattr(args, 'workers', 1), 'workers')
    begun = perf_counter()
    if args.limit_units is not None and args.limit_units < 1:
        raise ValueError('limit-units must be positive.')
    if not np.isfinite(args.max_hours) or args.max_hours <= 0.:
        raise ValueError('max-hours must be finite and positive.')
    source = source_snapshot()
    if source['dirty']:
        raise ValueError('H1 requires clean pinned source before running or resuming.')
    directory = args.run_dir
    exists = directory.exists()
    if exists != args.resume:
        raise ValueError('Use a new run directory, or --resume for an existing run.')
    version = int(getattr(args, 'design', 'v1')[1:])
    design = h1_design(version)
    numerics = H1_NUMERICS if version == 1 else H1_V2_NUMERICS
    backend = 'upwind' if version == 1 else 'fitted'
    manifest = dict(schema='bm1-h1-numerics-run-v1', source=source, design=design)
    completed, fresh, rows = 0, 0, []
    total = sum(map(len, H1_IDS.values()))*len(h1_candidate_panel())
    with _RunLock(directory), ExitStack() as stack:
        manifest_path = directory/'manifest.json'
        if args.resume:
            saved, _ = _read_likelihood_report(manifest_path)
            if saved != manifest:
                raise ValueError('H1 source/design changed; resume the original commit or use a new run.')
        else:
            _atomic_report(manifest_path, manifest)
        old_hashes = {}
        summary_path = directory/'summary.json'
        if summary_path.exists():
            old_summary, _ = _read_likelihood_report(summary_path)
            old_hashes = {row['unit']: row['sha256'] for row in old_summary['units']}
        jobs = None
        if workers > 1:
            # Read raw-file existence only; no score/candidate selection affects scheduling.
            specs = []
            for case_name, ids in H1_IDS.items():
                for index, sid in enumerate(ids):
                    for candidate_name, _, _ in h1_candidate_panel():
                        unit = f'{case_name}_{sid}_{candidate_name}'
                        if not (directory/'units'/f'{unit}.npz').exists():
                            if unit in old_hashes:
                                raise ValueError('A previously checkpointed H1 unit is missing.')
                            specs.append(dict(unit=unit, case=case_name, record_index=index,
                                              candidate=candidate_name, numerics=numerics, backend=backend))
            if args.limit_units is not None:
                specs = specs[:args.limit_units]
            jobs = _NumericalJobs(specs, workers, begun+args.max_hours*3600)
            stack.callback(jobs.close)
        stop = None
        for case_name in H1_IDS:
            records, truths = h1_training_records(case_name)
            for record, pair in zip(records, truths):
                linear_rows = None
                truth = truth_diagnostic(pair)
                for candidate_name, f, g in h1_candidate_panel():
                    unit = f'{case_name}_{record.record_id}_{candidate_name}'
                    path = directory/'units'/f'{unit}.npz'
                    identity = dict(unit=unit, observation_sha256=_observation_digest(record),
                                    f=f.to_record(), g=g.to_record())
                    if path.exists():
                        payload, digest = _read_likelihood_report(path)
                        if unit in old_hashes and digest != old_hashes[unit]:
                            raise ValueError('Saved H1 raw unit changed after its summary checkpoint.')
                        if payload['identity'] != identity:
                            raise ValueError('Saved H1 unit does not match the declared data and curves.')
                        values = payload['rows']
                    else:
                        if unit in old_hashes:
                            raise ValueError('A previously checkpointed H1 unit is missing.')
                        if ((args.limit_units is not None and fresh >= args.limit_units)
                                or (jobs is None and perf_counter()-begun >= args.max_hours*3600)):
                            stop = 'unit_budget' if args.limit_units is not None and fresh >= args.limit_units else 'time_budget'
                            break
                        if jobs is None:
                            values = run_numerical_unit(record, f, g, numerics=numerics, backend=backend)
                        else:
                            try:
                                values = jobs.take(unit)
                            except StopIteration:
                                stop = 'time_budget'
                                break
                        if source_snapshot() != source:
                            raise ValueError('Source changed during H1; no new unit published.')
                        _atomic_report(path, dict(identity=identity, rows=values))
                        _, digest = _read_likelihood_report(path)
                        fresh += 1
                    if set(values) != set(numerics):
                        raise ValueError('Saved H1 unit is incomplete.')
                    for level, settings in numerics.items():
                        validate_filter_arrays(values[level], record, cells=settings['cells'])
                    if candidate_name == 'linear':
                        linear_rows = values
                    controls = compare_numerical_unit(values, linear_rows)
                    negative_error = negative_control_error(values, record) if case_name == 'no_response' else None
                    rows.append(dict(unit=unit, sha256=digest, controls=controls, truth=truth,
                        negative_control_error=negative_error,
                        passes=bool(truth['passes'] and all(c['passes'] for c in controls.values())
                                    and (negative_error is None or negative_error <= 1e-10))))
                    completed += 1
                    if not args.quiet:
                        # Unit grids/costs are identical. Resume work is excluded from the timed numerator.
                        print(f'H1 numerics: {completed}/{total}; new={fresh}; '
                              + runtime_text(begun, fresh, total-(completed-fresh)), flush=True)
                if stop:
                    break
            if stop:
                break
        summary = dict(schema='bm1-h1-numerics-summary-v1', source=source, design=design,
            completed=completed == total, completed_units=completed, expected_units=total,
            new_units=fresh, units=rows, stop_reason=stop,
            numerical_panel_passes=completed == total and all(row['passes'] for row in rows),
            scientific_acceptance=False, workers=workers, elapsed_seconds=perf_counter()-begun,
            price_support=response_price_support([.125, .375, .625, .875]),
            interpretation='Development fixed-candidate panel only; no optimizer-wide, recovery, historical or forecast acceptance.')
        if source_snapshot() != source:
            raise ValueError('Source changed during H1; no summary published.')
        _atomic_report(summary_path, summary, replace_existing=summary_path.exists())
    print(f"H1: {completed}/{total}; panel passes={summary['numerical_panel_passes']}; "
          f"scientific acceptance=False; {summary_path}", flush=True)
    return summary



def read_numerical_dependency(directory):
    """Rebuild the complete v2 numerical decision from raw units, without FV.

    The source may be an older pinned commit. Data/design/curves and raw file
    hashes must agree; a modified summary passes flag cannot release fitting.
    """
    manifest, manifest_hash = _read_likelihood_report(directory/'manifest.json')
    summary, summary_hash = _read_likelihood_report(directory/'summary.json')
    design = h1_design(2)
    if (manifest.get('design') != design or summary.get('design') != design
            or manifest.get('source') != summary.get('source')
            or manifest.get('schema') != 'bm1-h1-numerics-run-v1'
            or summary.get('schema') != 'bm1-h1-numerics-summary-v1'):
        raise ValueError('H1 fitting requires the complete matching v2 numerical design/source.')
    if manifest['source'].get('dirty') is not False:
        raise ValueError('The numerical dependency must have clean pinned source.')
    saved_rows = summary.get('units', [])
    expected_count = sum(map(len, H1_IDS.values()))*len(h1_candidate_panel())
    if len(saved_rows) != expected_count or len({r['unit'] for r in saved_rows}) != expected_count:
        raise ValueError('H1 numerical dependency lacks all distinct expected units.')
    saved = {r['unit']: r for r in saved_rows}
    expected_names, hashes, maxima = [], {}, dict(prefix_log_evidence=0., centered_prefix_log_evidence=0., posterior_l1=0.)
    for case_name in H1_IDS:
        records, truths = h1_training_records(case_name)
        for record, pair in zip(records, truths):
            linear_rows = None
            if not truth_diagnostic(pair)['passes']:
                raise ValueError('H1 coupled truth prerequisite failed.')
            for name, f, g in h1_candidate_panel():
                unit = f'{case_name}_{record.record_id}_{name}'
                expected_names.append(unit)
                if unit not in saved:
                    raise ValueError('H1 numerical dependency contains unexpected or missing units.')
                payload, digest = _read_likelihood_report(directory/'units'/f'{unit}.npz')
                identity = dict(unit=unit, observation_sha256=_observation_digest(record), f=f.to_record(), g=g.to_record())
                if digest != saved[unit]['sha256'] or payload['identity'] != identity:
                    raise ValueError('H1 numerical raw data/curves/hash do not match their dependency.')
                values = payload['rows']
                if set(values) != set(H1_V2_NUMERICS):
                    raise ValueError('H1 numerical raw unit is incomplete.')
                for level, settings in H1_V2_NUMERICS.items():
                    validate_filter_arrays(values[level], record, cells=settings['cells'])
                if name == 'linear':
                    linear_rows = values
                controls = compare_numerical_unit(values, linear_rows)
                if not all(r['passes'] for r in controls.values()):
                    raise ValueError('H1 numerical dependency fails recomputed prefix gates.')
                if case_name == 'no_response' and negative_control_error(values, record) > 1e-10:
                    raise ValueError('H1 Xi=0 prerequisite failed.')
                hashes[unit] = digest
                for row in controls.values():
                    for key in maxima:
                        maxima[key] = max(maxima[key], row[key])
    if set(saved) != set(expected_names):
        raise ValueError('H1 numerical dependency contains extra units.')
    return dict(manifest_sha256=manifest_hash, summary_sha256=summary_hash,
                source=manifest['source'], raw_sha256=hashes, recomputed_maxima=maxima,
                numerical_panel_passes=True, fv_replays=0)


def _fit_cases(args, case_names, source, design, begun, settings):
    """Case-exclusive files; caller owns the run lock and validates dependency."""
    from src.method.bm_1_response import fit_shared_curve_pair, curve_pair_logits, curve_pair_from_logits, pooled_curve_log_likelihood
    from src.params.bm_1_response import BM1PiecewiseResponse
    from src.validation.bm_1.response import _score_record
    reports, new_total, paused = [], 0, False
    directory = args.run_dir
    for name in case_names:
        if perf_counter()-begun >= args.max_hours*3600:
            paused = True
            break
        records, _ = h1_training_records(name)
        linear = BM1PiecewiseResponse.linear()
        baseline_path = directory/f'{name}_linear.json.gz'
        if baseline_path.exists():
            baseline, _ = _read_likelihood_report(baseline_path)
        else:
            baseline = _score_record(pooled_curve_log_likelihood(records, linear, linear,
                backend='fitted', **settings))
            if source_snapshot() != source:
                raise ValueError('Fit source changed before baseline publication.')
            _atomic_report(baseline_path, baseline)
        previous = None
        for pieces in (2, 4):
            starts = design['two_starts'] if pieces == 2 else [
                curve_pair_logits(previous['f'], previous['g'], pieces=4).tolist(), [0.]*6]
            result_path = directory/f'{name}_{pieces}_result.json.gz'
            trace_path = directory/f'{name}_{pieces}_trace.json.gz'
            if result_path.exists():
                saved, _ = _read_likelihood_report(result_path)
                # A saved finished family must have a matching durable trace.
                trace, _ = _read_likelihood_report(trace_path)
                if saved['trace'] != trace or saved.get('paused') is not False:
                    raise ValueError('Completed fit and candidate checkpoint disagree.')
                if (saved.get('case') != name or saved.get('pieces') != pieces
                        or saved.get('starts') != starts or not trace
                        or saved.get('distinct_evaluations') != len(trace)):
                    raise ValueError('Completed fit family/start/budget metadata changed.')
                best = min(trace, key=lambda row: row['objective'])
                f, g = curve_pair_from_logits(best['logits'], pieces)
                if (saved['logits'] != best['logits'] or saved['objective'] != best['objective']
                        or saved['f'] != f.to_record() or saved['g'] != g.to_record()):
                    raise ValueError('Completed curve geometry differs from its best evaluated candidate.')
                # Validate completed checkpoints before downstream family reuse.
                from src.validation.bm_1.response_fit_checks import validate_fit_result
                f, g = validate_fit_result(saved, trace, case=name, pieces=pieces,
                    record_ids=[record.record_id for record in records])
                previous = dict(f=f, g=g)
                reports.append(saved)
                continue
            if (args.limit_candidates is not None and new_total >= args.limit_candidates
                    or perf_counter()-begun >= args.max_hours*3600):
                paused = True
                break
            trace = _read_likelihood_report(trace_path)[0] if trace_path.exists() else []
            resumed = len(trace)
            clock = perf_counter()
            def checkpoint(rows):
                if source_snapshot() != source:
                    raise ValueError('Fit source changed during candidate; no checkpoint published.')
                _atomic_report(trace_path, rows, replace_existing=trace_path.exists())
            def progress(done, total):
                if not args.quiet:
                    print(f'H1 fit {name}/{pieces}: {done}/{total}; '
                          +runtime_text(clock, done, total, initial_completed=resumed), flush=True)
            result = fit_shared_curve_pair(records, pieces=pieces, starts=starts,
                regularization=.1, max_evaluations=250, max_iterations=40, logit_bound=4.,
                backend='fitted', resume_trace=trace, checkpoint=checkpoint,
                max_new_evaluations=None if args.limit_candidates is None else args.limit_candidates-new_total,
                should_pause=lambda: perf_counter()-begun >= args.max_hours*3600,
                progress=progress, **settings)
            new_total += result['new_evaluations']
            previous = result
            payload = {key: value for key, value in result.items()
                       if key not in ('f', 'g', 'replay', 'logits', 'starts')}
            payload.update(case=name, pieces=pieces, f=result['f'].to_record(), g=result['g'].to_record(),
                logits=result['logits'].tolist(), starts=result['starts'].tolist(),
                replay=_score_record(result['replay']), linear_log_likelihood=baseline['log_likelihood'],
                numerical_acceptance=False, scientific_acceptance=False,
                interpretation='Selected candidate still needs its own refinements; TRAIN gain is not model selection.')
            reports.append(payload)
            if result['paused']:
                paused = True
                break
            if source_snapshot() != source:
                raise ValueError('Fit source changed before result publication.')
            _atomic_report(result_path, payload)
        if paused:
            break
    return reports, new_total, paused


def run_fit(args):
    """Conditional H1b development fits with resumable candidate-score traces."""
    import scipy
    import platform
    from src.validation.estimation import positive_integer
    workers = positive_integer(getattr(args, 'workers', 1), 'workers')
    if workers > 1 and args.limit_candidates is not None:
        raise ValueError('Use workers=1 for a global limit-candidates pilot; parallel cases retain the registered family budgets.')
    begun = perf_counter()
    if not np.isfinite(args.max_hours) or args.max_hours <= 0.:
        raise ValueError('max-hours must be finite and positive.')
    if args.limit_candidates is not None and args.limit_candidates < 1:
        raise ValueError('limit-candidates must be positive.')
    source = source_snapshot()
    if source['dirty']:
        raise ValueError('H1 fitting requires clean pinned source.')
    dependency = read_numerical_dependency(args.numerics_run)
    settings = H1_V2_NUMERICS['base']
    design = dict(schema='bm1-h1-shared-fit-v1', cases=list(H1_IDS),
        numerical_settings=settings, backend='fitted', lambda_=0.1,
        candidate_budget_per_family=250, max_iterations=40, logit_bound=4.,
        two_starts=[[0., 0.], [.8, -.8]], four_starts='lift two-piece best, then linear',
        optimizer='L-BFGS-B', ftol=1e-9, gtol=1e-5,
        scope='Conditional shared DEVELOPMENT fits; no selection/holdout, joint recovery or H2/H3 acceptance.')
    manifest = dict(source=source, dependency=dependency, design=design,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         blas_threads=args.blas_threads))
    directory = args.run_dir
    if directory.exists() != args.resume:
        raise ValueError('Use a new fit directory, or --resume with its original source/environment.')
    reports, new_total, paused = [], 0, False
    with _RunLock(directory):
        path = directory/'manifest.json'
        if args.resume:
            saved, _ = _read_likelihood_report(path)
            if saved != manifest:
                raise ValueError('Fit source/data/dependency/numerics/optimizer environment changed.')
        else:
            _atomic_report(path, manifest)
        if workers == 1:
            reports, new_total, paused = _fit_cases(args, list(H1_IDS), source, design, begun, settings)
        else:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            # Each task owns exactly one registered case and its two dependent
            # families. Candidate budgets/starter order within a case are unchanged.
            with ProcessPoolExecutor(max_workers=min(workers, len(H1_IDS)),
                    mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(_fit_cases, args, [name], source, design, begun, settings)
                           for name in H1_IDS]
                for future in futures:
                    case_reports, count, stopped = future.result()
                    reports.extend(case_reports)
                    new_total += count
                    paused = paused or stopped
        summary = dict(schema='bm1-h1-fit-summary-v1', source=source,
            completed=len(reports) == 6 and not paused, paused=paused,
            new_candidates=new_total, fits=reports, workers=workers, scientific_acceptance=False,
            elapsed_seconds=perf_counter()-begun,
            interpretation='H1b conditional development only; complete execution does not imply convergence or recovery.')
        if source_snapshot() != source:
            raise ValueError('Fit source changed before summary publication.')
        path = directory/'summary.json'
        _atomic_report(path, summary, replace_existing=path.exists())
    print(f"H1 fit: {len(reports)}/6 families; paused={paused}; {directory/'summary.json'}", flush=True)
    return summary

def run_fit_checks(args):
    """Recompute selected-curve numerics; preserve failures and resume raw units."""
    from src.validation.bm_1.response_fit_checks import (
        validate_fit_result, selected_curve_unit, curve_recovery_description,
        validate_selected_replay)
    begun=perf_counter()
    if not np.isfinite(args.max_hours) or args.max_hours <= 0.:
        raise ValueError('max-hours must be finite and positive.')
    source=source_snapshot()
    if source['dirty']:
        raise ValueError('Selected-curve checks require clean pinned source.')
    dependency=read_numerical_dependency(args.numerics_run)
    fit_manifest,fit_manifest_hash=_read_likelihood_report(args.fit_run/'manifest.json')
    fit_summary,fit_summary_hash=_read_likelihood_report(args.fit_run/'summary.json')
    if (fit_manifest['dependency'] != dependency or fit_summary.get('completed') is not True
            or fit_summary['source'] != fit_manifest['source'] or fit_manifest['source'].get('dirty') is not False):
        raise ValueError('Require the complete fit and its original raw numerical dependency.')
    inputs=[]
    for case in H1_IDS:
        records,_=h1_training_records(case)
        for pieces in (2,4):
            result,rhash=_read_likelihood_report(args.fit_run/f'{case}_{pieces}_result.json.gz')
            trace,thash=_read_likelihood_report(args.fit_run/f'{case}_{pieces}_trace.json.gz')
            f,g=validate_fit_result(result,trace,case=case,pieces=pieces,record_ids=[r.record_id for r in records])
            inputs.append((case,pieces,records,result,f,g,rhash,thash))
    manifest=dict(schema='bm1-h1-selected-checks-v1',source=source,dependency=dependency,
        fit_manifest_sha256=fit_manifest_hash,fit_summary_sha256=fit_summary_hash,
        fits=[dict(case=c,pieces=p,result_sha256=rh,trace_sha256=th) for c,p,_,_,_,_,rh,th in inputs])
    directory=args.run_dir
    if directory.exists() != args.resume:
        raise ValueError('Use a new selected-check directory or resume the same source.')
    units=[];fresh=0;stop=None
    with _RunLock(directory):
        mpath=directory/'manifest.json'
        if args.resume:
            saved,_=_read_likelihood_report(mpath)
            if saved != manifest:
                raise ValueError('Selected-check source or inputs changed.')
        else:
            _atomic_report(mpath,manifest)
        old={}
        if (directory/'summary.json').exists():
            saved,_=_read_likelihood_report(directory/'summary.json')
            old={r['unit']:r['sha256'] for r in saved['units']}
        for case,pieces,records,result,f,g,_,_ in inputs:
            for record,fit_record in zip(records,result['replay']['records']):
                unit=f'{case}_{pieces}_{record.record_id}'
                path=directory/'units'/f'{unit}.npz'
                identity=dict(unit=unit,f=f.to_record(),g=g.to_record(),observation_sha256=_observation_digest(record))
                if path.exists():
                    payload,digest=_read_likelihood_report(path)
                    if payload['identity'] != identity or (unit in old and digest != old[unit]):
                        raise ValueError('Selected raw unit changed.')
                else:
                    if unit in old:
                        raise ValueError('Selected raw unit is missing.')
                    if perf_counter()-begun >= args.max_hours*3600:
                        stop='time_budget';break
                    linear,_=_read_likelihood_report(args.numerics_run/'units'/f'{case}_{record.record_id}_linear.npz')
                    payload=dict(identity=identity,**selected_curve_unit(record,f,g,linear['rows'],fit_record))
                    if source_snapshot() != source:
                        raise ValueError('Selected-check source changed before publication.')
                    _atomic_report(path,payload);_,digest=_read_likelihood_report(path);fresh+=1
                # Recompute every acceptance flag from raw laws, including on resume.
                linear,_=_read_likelihood_report(args.numerics_run/'units'/f'{case}_{record.record_id}_linear.npz')
                validate_selected_replay(payload['rows'], record, fit_record)
                controls=compare_numerical_unit(payload['rows'],linear['rows'])
                units.append(dict(unit=unit,sha256=digest,controls=controls,
                    numerical_passes=all(v['passes'] for v in controls.values()),
                    recovery=curve_recovery_description(case,f,g)))
                if not args.quiet:
                    print(f'H1 selected checks: {len(units)}/12; '+runtime_text(begun,fresh,12-(len(units)-fresh)),flush=True)
            if stop:break
        summary=dict(schema='bm1-h1-selected-summary-v1',source=source,units=units,
            completed=len(units)==12,stop_reason=stop,new_units=fresh,
            selected_numerical_passes=len(units)==12 and all(r['numerical_passes'] for r in units),
            scientific_acceptance=False,elapsed_seconds=perf_counter()-begun,
            interpretation='Selected development curves only. No H2/H3, joint recovery or forecast acceptance.')
        if source_snapshot() != source:
            raise ValueError('Selected-check source changed before summary publication.')
        path=directory/'summary.json';_atomic_report(path,summary,replace_existing=path.exists())
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('history-inspect', help='Inspect H2 workbook headers without reading observations or fitting.')
    inspect.add_argument('--data-dir', type=Path, default=Path('files/input/data'))
    inspect.add_argument('--output', type=Path, default=Path('files/output/bm_1/runs/estimation/h2_header_inspection.json'))
    inspect.add_argument('--blas-threads', type=int, default=1)
    config = sub.add_parser('history-config', help='Write an incomplete H2 pilot configuration; no observations read.')
    config.add_argument('--output', type=Path, required=True)
    config.add_argument('--blas-threads', type=int, default=1)
    history = sub.add_parser('history-fit', help='Prepare or run a bounded conditional TRAIN-only H2 pilot.')
    history.add_argument('--config', type=Path, required=True)
    history.add_argument('--input-root', type=Path, default=ROOT)
    history.add_argument('--run-dir', type=Path, required=True)
    history.add_argument('--prepare-only', action='store_true')
    history.add_argument('--resume', action='store_true')
    history.add_argument('--limit-candidates', type=int)
    history.add_argument('--max-seconds', type=float, default=300.)
    history.add_argument('--blas-threads', type=int, default=1)
    history.add_argument('--quiet', action='store_true')
    numerics = sub.add_parser('numerics', help='Run/review the fixed H1 numerical panel.')
    numerics.add_argument('--design', choices=('v1', 'v2'), default='v1')
    numerics.add_argument('--run-dir', type=Path, required=True)
    numerics.add_argument('--resume', action='store_true')
    numerics.add_argument('--limit-units', type=int)
    numerics.add_argument('--max-hours', type=float, default=8.)
    numerics.add_argument('--blas-threads', type=int, default=1)
    numerics.add_argument('--quiet', action='store_true')
    numerics.add_argument('--workers', type=int, default=1)
    fit = sub.add_parser('fit', help='Run resumable conditional H1b shared fits.')
    fit.add_argument('--numerics-run', type=Path, required=True)
    fit.add_argument('--run-dir', type=Path, required=True)
    fit.add_argument('--resume', action='store_true')
    fit.add_argument('--limit-candidates', type=int)
    fit.add_argument('--max-hours', type=float, default=8.)
    fit.add_argument('--blas-threads', type=int, default=1)
    fit.add_argument('--quiet', action='store_true')
    fit.add_argument('--workers', type=int, default=1)
    verify = sub.add_parser('validate-fit', help='Check exact selected curves on time/space/quadrature refinements.')
    verify.add_argument('--numerics-run', type=Path, required=True)
    verify.add_argument('--fit-run', type=Path, required=True)
    verify.add_argument('--run-dir', type=Path, required=True)
    verify.add_argument('--resume', action='store_true')
    verify.add_argument('--max-hours', type=float, default=8.)
    verify.add_argument('--blas-threads', type=int, default=1)
    verify.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'history-config':
        from src.exe.run_model.bm_1_history import configuration_template
        _atomic_report(args.output, configuration_template())
        print(f'H2 configuration template: {args.output}; resolve metadata before fitting.', flush=True)
        return 0
    if args.command == 'history-fit':
        from src.exe.run_model.bm_1_history import run_pilot
        result = run_pilot(args)
        print(f'H2 pilot: completed={result.get("completed", False)}, paused={result.get("paused", False)}; no scientific acceptance.', flush=True)
        return 0
    if args.command == 'history-inspect':
        from src.tools.bm_1_history_io import inspect_history_workbooks
        result = inspect_history_workbooks(args.data_dir/'input.xlsx', args.data_dir/'prices.xlsx')
        # This metadata-only report may be refreshed; it is not a study checkpoint.
        _atomic_report(args.output, result, replace_existing=True)
        print(f"H2 headers inspected; no observations read, no fit run. Report: {args.output}", flush=True)
        print(f"Measurement columns: {len(result['input']['measurement_columns'])}. Metadata/split decisions remain required.", flush=True)
        return 0
    if args.command == 'validate-fit':
        result = run_fit_checks(args)
        return 2 if result['completed'] and not result['selected_numerical_passes'] else 0
    if args.command == 'fit':
        run_fit(args)
        return 0
    result = run_numerics(args)
    return 2 if result['completed'] and not result['numerical_panel_passes'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
