"""One cheap whole-day continuation; no new precise replay or optimizer."""

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
    parser.add_argument('--prefix-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ('OPENBLAS_NUM_THREADS', 'OPENBLAS_DEFAULT_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    from src.tools.bm_1_reports import _atomic_report, _RunLock
    prefix_bytes = (args.prefix_dir/'report.json').read_bytes()
    prefix = json.loads(prefix_bytes)
    # Inputs are those validated by the shared-prefix stream report.
    stream_path = args.prefix_dir.parent/'f4_h2_search_stream_20261006'/'report.json'
    stream = json.loads(stream_path.read_bytes())
    inputs = list(stream['inputs_sha256'])
    development, cn = [json.loads(Path(p).read_bytes()) for p in inputs[:2]]
    saved = next(u for u in prefix['units'] if u['candidate'] == 'two_saved')
    if args.worker:
        from src.validation.bm_1.interval_joint_cn import run_joint_cn
        result = run_joint_cn(development, cn, 'two_saved', intervals=24,
            accumulator_spacing=1/1024, saved_prefix=saved['rows'])
        _atomic_report(args.run_dir/'day.json', result)
        return 0
    from src.exe.run_model.bm_1.curves import source_snapshot
    from src.exe.run_model.bm_1_interval_terminal_tilt import _frozen_terminal_reference
    from src.validation.bm_1.interval_terminal_tilt import declared_terminal_control
    from src.params.bm_1 import conservative_mass_transfer
    import numpy as np
    with _RunLock(args.run_dir):
        if (args.run_dir/'manifest.json').exists():
            raise FileExistsError('Preserve the one cheap day; no new run budget.')
        begun, source = perf_counter(), source_snapshot()
        if source['dirty'] or not prefix['local_prefix_search_feasible'] or prefix['settings']['accumulator_spacing'] != 1/1024:
            raise ValueError('Freeze the code and require the passed accumulator-refined prefixes.')
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in stream['inputs_sha256'].items()):
            raise ValueError('The frozen inputs changed.')
        _, design = declared_terminal_control(development, cn, 'day')
        reference, equivalence = _frozen_terminal_reference(Path(inputs[2]).read_bytes(),
            {p: stream['inputs_sha256'][p] for p in inputs[:2]}, source, design)
        manifest = dict(schema='f4-h2-positive-joint-cn-day-v1', source=source,
            prefix_sha256=hashlib.sha256(prefix_bytes).hexdigest(), reference_equivalence=equivalence,
            prior_charged_seconds=prefix['total_charged_seconds'], cumulative_budget_seconds=900.,
            candidate='two_saved', settings=prefix['settings'], intervals=24,
            reused_search_intervals=6, precise_reference_intervals_recomputed=0)
        _atomic_report(args.run_dir/'manifest.json', manifest)
        remaining = min(60., 870.-prefix['total_charged_seconds'])-(perf_counter()-begun)
        try:
            if remaining <= 0.: raise subprocess.TimeoutExpired('joint-cn-day', 0.)
            subprocess.run([sys.executable, '-B', '-m', 'src.exe.run_model.bm_1_interval_joint_cn_day',
                '--prefix-dir', str(args.prefix_dir), '--run-dir', str(args.run_dir), '--worker'],
                timeout=remaining, check=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            _atomic_report(args.run_dir/'report.json', dict(**manifest, status='budget_or_unit_failed',
                reason=str(exc), total_charged_seconds=prefix['total_charged_seconds']+perf_counter()-begun,
                scientific_acceptance=False, numerical_fit_readiness=False))
            return 2
        result = json.loads((args.run_dir/'day.json').read_bytes())
        tolerances = stream['comparison_tolerances']
        comparisons, cheap_score, reference_score = [], 0., 0.
        for index, row in enumerate(result['rows']):
            exact = reference['rows'][24+index]
            if row['observed'] != (exact['log_evidence'] is not None):
                raise ValueError('Missing masks differ.')
            if row['observed']:
                cheap_score += row['log_evidence']; reference_score += exact['log_evidence']
            mass = np.exp([-np.inf if v is None else v for v in exact['log_posterior_mass']])
            mapped = conservative_mass_transfer(row['x_edges'], row['posterior_mass'], reference['replays'][1]['x_edges'])
            errors = dict(total_log_likelihood=abs(cheap_score-reference_score),
                final_state_mean=abs(row['posterior_state_mean']-exact['posterior_state_mean']),
                final_mass_l1=float(np.abs(mapped-mass).sum()))
            comparisons.append(dict(completed_intervals=index+1, errors=errors,
                gates_pass=all(errors[k] <= tolerances[k] for k in errors)))
        if source_snapshot() != source: raise ValueError('Frozen source changed.')
        report = dict(**manifest, status='completed', rows=result['rows'], comparisons=comparisons,
            new_replay_seconds=result['elapsed_seconds'],
            total_charged_seconds=prefix['total_charged_seconds']+perf_counter()-begun,
            prefix_numerical_gates_pass=all(c['gates_pass'] for c in comparisons),
            final_numerical_gates_pass=comparisons[-1]['gates_pass'],
            historical_observation_values_read=0, historical_fit_run=False,
            numerical_fit_readiness=False, scientific_acceptance=False)
        _atomic_report(args.run_dir/'report.json', report)
        print(json.dumps(dict(seconds=report['total_charged_seconds'], new_replay_seconds=result['elapsed_seconds'],
            all_prefixes_pass=report['prefix_numerical_gates_pass'], final=comparisons[-1],
            maxima={k: max(c['errors'][k] for c in comparisons) for k in tolerances})), flush=True)
        return 0 if report['prefix_numerical_gates_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
