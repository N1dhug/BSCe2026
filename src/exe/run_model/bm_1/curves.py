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


def run_numerics(args):
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
    with _RunLock(directory):
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
                                or perf_counter()-begun >= args.max_hours*3600):
                            stop = 'unit_budget' if args.limit_units is not None and fresh >= args.limit_units else 'time_budget'
                            break
                        values = run_numerical_unit(record, f, g, numerics=numerics, backend=backend)
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
            scientific_acceptance=False, elapsed_seconds=perf_counter()-begun,
            price_support=response_price_support([.125, .375, .625, .875]),
            interpretation='Development fixed-candidate panel only; no optimizer-wide, recovery, historical or forecast acceptance.')
        if source_snapshot() != source:
            raise ValueError('Source changed during H1; no summary published.')
        _atomic_report(summary_path, summary, replace_existing=summary_path.exists())
    print(f"H1: {completed}/{total}; panel passes={summary['numerical_panel_passes']}; "
          f"scientific acceptance=False; {summary_path}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    numerics = sub.add_parser('numerics', help='Run/review the fixed H1 numerical panel.')
    numerics.add_argument('--design', choices=('v1', 'v2'), default='v1')
    numerics.add_argument('--run-dir', type=Path, required=True)
    numerics.add_argument('--resume', action='store_true')
    numerics.add_argument('--limit-units', type=int)
    numerics.add_argument('--max-hours', type=float, default=8.)
    numerics.add_argument('--blas-threads', type=int, default=1)
    numerics.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    result = run_numerics(args)
    return 2 if result['completed'] and not result['numerical_panel_passes'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
