"""One cheap joint-CN prefix experiment, reusing all twelve precise new hours."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--spacing', type=float, choices=(1/512, 1/1024), default=1/512)
    parser.add_argument('--candidate', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ('OPENBLAS_NUM_THREADS', 'OPENBLAS_DEFAULT_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    from src.tools.bm_1_reports import _atomic_report, _RunLock
    stream_bytes = (args.stream_dir/'report.json').read_bytes()
    stream = json.loads(stream_bytes)
    paths = list(stream['inputs_sha256'])
    development, cn = [json.loads(Path(p).read_bytes()) for p in paths[:2]]
    if args.candidate:
        from src.validation.bm_1.interval_joint_cn import run_joint_cn
        _atomic_report(args.run_dir/(args.candidate+'.json'),
            run_joint_cn(development, cn, args.candidate, intervals=6, accumulator_spacing=args.spacing))
        return 0
    from src.exe.run_model.bm_1.curves import source_snapshot
    from src.exe.run_model.bm_1_interval_search_pilot import CANDIDATES, compare_search
    from src.exe.run_model.bm_1_interval_terminal_tilt import TERMINAL_REFERENCE_DEPENDENCIES
    with _RunLock(args.run_dir):
        if (args.run_dir/'manifest.json').exists():
            raise FileExistsError('Preserve this one bounded experiment; no automatic repeated panel.')
        begun, source = perf_counter(), source_snapshot()
        if source['dirty'] or any(source['sha256'][p] != stream['source']['sha256'][p]
            for p in TERMINAL_REFERENCE_DEPENDENCIES):
            raise ValueError('Freeze the new prototype and preserve the precise reference kernels.')
        if stream['completed_intervals'] != 6 or stream['status'] != 'prefix_gate_failed':
            raise ValueError('This follow-up requires the actual six-hour negative joint-Euler result.')
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in stream['inputs_sha256'].items()):
            raise ValueError('Frozen inputs changed.')
        manifest = dict(schema='f4-h2-positive-joint-cn-pilot-v1', source=source,
            prior_stream_sha256=hashlib.sha256(stream_bytes).hexdigest(),
            settings=dict(cells=321, max_step=1/1536, accumulator_spacing=args.spacing),
            intervals=6, candidates=list(CANDIDATES), cumulative_budget_seconds=900.,
            prior_charged_seconds=stream['combined_charged_seconds'],
            changes='Positive CN in an isolated subclass; declared accumulator spacing, production defaults unchanged.',
            production_defaults_changed=False, precise_reference_intervals_recomputed=0)
        previous_path = args.stream_dir.parent/'f4_h2_joint_cn_20261006'/'report.json'
        if args.spacing == 1/1024:
            previous = json.loads(previous_path.read_bytes())
            if previous['status'] != 'completed' or previous['local_prefix_search_feasible']:
                raise ValueError('Accumulator-only follow-up requires the completed negative CN prefix.')
            manifest['prior_charged_seconds'] = previous['total_charged_seconds']
            manifest['prior_cn_sha256'] = hashlib.sha256(previous_path.read_bytes()).hexdigest()
        _atomic_report(args.run_dir/'manifest.json', manifest)
        units, status = [], 'completed'
        for candidate in CANDIDATES:
            # Cap the small follow-up itself at 60 seconds, also retaining the old total charge.
            remaining = min(60., 870.-manifest['prior_charged_seconds'])-(perf_counter()-begun)
            if remaining <= 0.:
                status = 'budget_stopped'; break
            try:
                subprocess.run([sys.executable, '-B', '-m', 'src.exe.run_model.bm_1_interval_joint_cn',
                    '--stream-dir', str(args.stream_dir), '--run-dir', str(args.run_dir),
                    '--spacing', str(args.spacing), '--candidate', candidate], timeout=remaining, check=True)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                status = 'budget_stopped' if isinstance(exc, subprocess.TimeoutExpired) else 'unit_failed'
                break
            units.append(json.loads((args.run_dir/(candidate+'.json')).read_bytes()))
            print(f'joint CN: {candidate}, fresh six-hour replay {units[-1]["elapsed_seconds"]:.3f}s', flush=True)
            if source_snapshot() != source:
                raise ValueError('Frozen source changed.')
        comparisons = []
        if status == 'completed':
            cheap = [r for u in units for r in u['rows']]
            exact = [r for r in stream['rows'] if r['method'] == 'terminal1921']
            for end in range(6):
                cumulative = []
                for candidate in CANDIDATES:
                    for method, collection in (('joint321', cheap), ('terminal1921', exact)):
                        local = [r for r in collection if r['candidate'] == candidate and r['interval_index'] <= end]
                        cumulative.append(dict(local[-1], log_evidence=sum(r['log_evidence'] for r in local if r['observed']),
                            elapsed_seconds=sum(r['elapsed_seconds'] for r in local)))
                c = next(c for c in compare_search(cumulative, stream['comparison_tolerances']) if c['method'] == 'joint321')
                c['completed_intervals'] = end+1
                comparisons.append(c)
        elapsed = perf_counter()-begun
        report = dict(**manifest, status=status, units=units, comparisons=comparisons,
            total_charged_seconds=manifest['prior_charged_seconds']+elapsed,
            local_prefix_search_feasible=status == 'completed' and all(c['local_search_feasible'] for c in comparisons),
            numerical_fit_readiness=False, historical_observation_values_read=0,
            scientific_acceptance=False)
        _atomic_report(args.run_dir/'report.json', report)
        print(json.dumps(dict(status=status, seconds=report['total_charged_seconds'],
            prefix_search_feasible=report['local_prefix_search_feasible'], comparisons=comparisons)), flush=True)
        return 0 if report['local_prefix_search_feasible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
