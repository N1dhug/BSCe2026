"""F4-H2 recursive follow-up; shares the first-hour pilot's 15-minute budget."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def stream_unit(development, cn, candidate, method, index, previous):
    import numpy as np
    from src.exe.run_model.bm_1_interval_search_pilot import _record, SEARCH_SETTINGS, REFERENCE_SETTINGS
    from src.model.bm_1 import bm_1
    from src.solver.finite_volume import finite_volume_1d_fitted
    from src.method.bm_1_interval import interval_filter_step
    from src.validation.bm_1.interval_terminal_tilt import terminal_tilt_update
    from src.validation.bm_1.interval_characteristic import balanced_characteristic_prediction
    record = _record(development, cn, candidate)
    settings = SEARCH_SETTINGS[method] if method == 'joint321' else REFERENCE_SETTINGS
    model = bm_1(record.baseline, record.baseline, record.horizon, 2)
    solver = finite_volume_1d_fitted(vectorized_inputs=True)
    model.fokker_planc_reference = solver
    edges, x, _ = solver.grid(settings['cells'])
    if previous['x_edges'] != edges.tolist():
        raise ValueError('The recursive prior grid changed.')
    prior = np.array(previous['posterior_mass'])
    logs = np.array([-np.inf if v is None else v for v in previous['log_posterior_mass']])
    if previous['candidate'] != candidate or previous['method'] != method:
        raise ValueError('Never reuse state across candidates or methods.')
    start, end, value, seen = (array[index] for array in (record.observations.starts,
        record.observations.ends, record.observations.values, record.observations.observed))
    begun = perf_counter()
    if method == 'joint321':
        result = interval_filter_step(model, record.parameters, record.sigma_x,
            record.sigma_y, prior, start, end, value, observed=bool(seen),
            breakpoints=record.breakpoints, **{k: v for k, v in settings.items() if k not in ('cells', 'backend')})
        posterior, score = result.posterior_mass, result.log_evidence if seen else None
        log_posterior = [None if v == 0. else float(np.log(v)) for v in posterior]
        diagnostics = {k: result.diagnostics[k] for k in ('maximum_abs_mass_error',
            'minimum_cell_mass', 'accumulator_deposition_variance_bound')}
    else:
        p = record.parameters
        drift, diffusion, options = model._fv_evolution_fields(p, record.sigma_x,
            np.array([start, end]), [], settings['max_step'])
        reward = lambda t, states: model.eval(t, states, p[0], p[1], p[2], p[4], p[5])[-1]
        if seen:
            result = terminal_tilt_update(solver, [start, end], prior, drift, diffusion, reward,
                value, record.sigma_y, log_prior=logs,
                **{k: v for k, v in settings.items() if k != 'cells'}, **options)
            posterior, score = result['posterior_mass'], result['log_evidence']
            logs = result['log_posterior_mass']
            diagnostics = {k: result[k] for k in ('maximum_abs_mass_error',
                'maximum_relative_conditional_guard', 'maximum_overlap_log_difference')}
        else:
            result = balanced_characteristic_prediction(solver, [start, end], prior,
                drift, diffusion, reward, max_step=settings['max_step'], frequencies=1,
                log_prior=logs, state_scheme='positive_crank_nicolson', **options)
            logs = result['log_amplitude']
            posterior, score = np.exp(logs), None
            diagnostics = dict(maximum_abs_mass_error=result['maximum_abs_mass_error'])
        log_posterior = [None if np.isneginf(v) else float(v) for v in logs]
    return dict(candidate=candidate, method=method, interval_index=index, observed=bool(seen),
        log_evidence=None if score is None else float(score), x_edges=edges.tolist(),
        posterior_mass=posterior.tolist(), log_posterior_mass=log_posterior,
        posterior_state_mean=float(posterior @ x), elapsed_seconds=perf_counter()-begun,
        diagnostics=diagnostics, reused=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--unit', help=argparse.SUPPRESS)
    parser.add_argument('--previous', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ('OPENBLAS_NUM_THREADS', 'OPENBLAS_DEFAULT_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    from src.exe.run_model.bm_1_interval_search_pilot import CANDIDATES, compare_search
    from src.tools.bm_1_reports import _atomic_report, _RunLock
    pilot_bytes = (args.pilot_dir/'report.json').read_bytes()
    pilot = json.loads(pilot_bytes)
    inputs = list(pilot['inputs_sha256'])
    development, cn = [json.loads(Path(p).read_bytes()) for p in inputs[:2]]
    if args.unit:
        candidate, method, index = args.unit.split('/')
        row = stream_unit(development, cn, candidate, method, int(index), json.loads(args.previous.read_bytes()))
        _atomic_report(args.run_dir/f'{candidate}_{method}_{index}.json', row)
        return 0
    from src.exe.run_model.bm_1.curves import source_snapshot
    from src.exe.run_model.bm_1_interval_terminal_tilt import _frozen_terminal_reference
    from src.validation.bm_1.interval_terminal_tilt import declared_terminal_control
    with _RunLock(args.run_dir):
        if (args.run_dir/'manifest.json').exists():
            raise FileExistsError('Preserve the bounded stream trial; no new invocation budget.')
        source = source_snapshot()
        if source['dirty'] or any(source['sha256'].get(p) != h for p, h in pilot['source']['sha256'].items()):
            raise ValueError('Commit new orchestration and preserve all first-hour sources.')
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in pilot['inputs_sha256'].items()):
            raise ValueError('Pilot inputs changed.')
        first_comparisons = compare_search(pilot['units'], pilot['comparison_tolerances'])
        if pilot['status'] != 'completed' or not next(c for c in first_comparisons if c['method'] == 'joint321')['local_search_feasible']:
            raise ValueError('The 321-cell first-hour prerequisite did not pass.')
        _, design = declared_terminal_control(development, cn, 'day')
        saved, equivalence = _frozen_terminal_reference(Path(inputs[2]).read_bytes(),
            {p: pilot['inputs_sha256'][p] for p in inputs[:2]}, source, design)
        manifest = dict(schema='f4-h2-search-stream-v1', source=source,
            pilot_json_sha256=hashlib.sha256(pilot_bytes).hexdigest(), inputs_sha256=pilot['inputs_sha256'],
            intervals=8, candidates=list(CANDIDATES), method='joint321',
            comparison_tolerances=pilot['comparison_tolerances'], reference_equivalence=equivalence,
            combined_budget_seconds=900., prior_trial_seconds=pilot['total_charged_seconds'],
            stop_at_first_failed_prefix=True)
        _atomic_report(args.run_dir/'manifest.json', manifest)
        begun, rows, comparisons, status = perf_counter(), [], [], 'completed'
        charged = pilot['total_charged_seconds']
        for index in range(8):
            for method in ('joint321', 'terminal1921'):
                for candidate in CANDIDATES:
                    path = args.run_dir/f'{candidate}_{method}_{index}.json'
                    if index == 0:
                        row = next(r for r in pilot['units'] if r['candidate'] == candidate and r['method'] == method)
                        row = dict(row, interval_index=0, observed=True, reused=True)
                        _atomic_report(path, row)
                    elif method == 'terminal1921' and candidate == 'two_saved':
                        import numpy as np
                        unit = saved['rows'][24+index]
                        row = dict(candidate=candidate, method=method, interval_index=index,
                            observed=unit['log_evidence'] is not None, log_evidence=unit['log_evidence'],
                            x_edges=saved['replays'][1]['x_edges'], log_posterior_mass=unit['log_posterior_mass'],
                            posterior_mass=np.exp([float('-inf') if v is None else v for v in unit['log_posterior_mass']]).tolist(),
                            posterior_state_mean=unit['posterior_state_mean'], elapsed_seconds=unit['elapsed_seconds'],
                            diagnostics={k: unit[k] for k in ('maximum_abs_mass_error',
                                'maximum_relative_conditional_guard', 'maximum_overlap_log_difference')}, reused=True)
                        _atomic_report(path, row)
                    else:
                        remaining = 870.-charged-(perf_counter()-begun)
                        if remaining <= 0.:
                            status = 'budget_stopped'; break
                        previous_path = args.run_dir/f'{candidate}_{method}_{index-1}.json'
                        command = [sys.executable, '-B', '-m', 'src.exe.run_model.bm_1_interval_search_stream',
                            '--pilot-dir', str(args.pilot_dir), '--run-dir', str(args.run_dir),
                            '--unit', f'{candidate}/{method}/{index}', '--previous', str(previous_path)]
                        try:
                            subprocess.run(command, timeout=remaining, check=True)
                        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                            status = 'budget_stopped' if isinstance(exc, subprocess.TimeoutExpired) else 'unit_failed'
                            break
                        row = json.loads(path.read_bytes())
                    rows.append(row)
                if status != 'completed': break
            if status != 'completed': break
            cumulative = []
            for method in ('joint321', 'terminal1921'):
                for candidate in CANDIDATES:
                    local = [r for r in rows if r['candidate'] == candidate and r['method'] == method]
                    cumulative.append(dict(local[-1], log_evidence=sum(r['log_evidence'] for r in local if r['observed']),
                        elapsed_seconds=sum(r['elapsed_seconds'] for r in local)))
            comparison = next(c for c in compare_search(cumulative, pilot['comparison_tolerances']) if c['method'] == 'joint321')
            comparison['completed_intervals'] = index+1
            comparisons.append(comparison)
            elapsed = charged+perf_counter()-begun
            _atomic_report(args.run_dir/'checkpoint.json', dict(manifest=manifest, rows=rows,
                comparisons=comparisons, combined_charged_seconds=elapsed), replace_existing=True)
            from src.exe.run_model.progress import runtime_text
            print(f'stream {index+1}/8: '+runtime_text(begun, index, 7)+
                f'; prefix passes={comparison["local_search_feasible"]}; combined budget used={elapsed:.1f}s', flush=True)
            if source_snapshot() != source:
                raise ValueError('Frozen source changed.')
            if not comparison['local_search_feasible']:
                status = 'prefix_gate_failed'; break
        report = dict(**manifest, status=status, rows=rows, comparisons=comparisons,
            completed_intervals=len(comparisons), combined_charged_seconds=charged+perf_counter()-begun,
            numerical_fit_readiness=False, historical_observation_values_read=0,
            historical_fit_run=False, scientific_acceptance=False)
        _atomic_report(args.run_dir/'report.json', report)
        print(json.dumps(dict(status=status, intervals=len(comparisons), seconds=report['combined_charged_seconds'],
            last_comparison=comparisons[-1] if comparisons else None)), flush=True)
        return 0 if status == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
