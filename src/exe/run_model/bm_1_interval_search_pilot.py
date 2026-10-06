"""Bounded F4-H2 search feasibility; unchanged positive joint and tilt kernels."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, time


CANDIDATES = {
    'linear': ([0.], [0.]),
    'two_saved': ([.3], [-.25]),
    'two_near': ([.32], [-.23]),
}
SEARCH_SETTINGS = {
    'joint161': dict(cells=161, max_step=1/768, accumulator_spacing=1/256,
                     max_accumulator_nodes=4096, backend='fitted'),
    'joint321': dict(cells=321, max_step=1/1536, accumulator_spacing=1/512,
                     max_accumulator_nodes=4096, backend='fitted'),
}
REFERENCE_SETTINGS = dict(cells=1921, max_step=1/24576, frequencies=64, span=2.)


def _record(development, cn_reference, name):
    from dataclasses import replace
    from src.validation.bm_1.interval_curve_control import declared_control
    from src.params.bm_1_response import BM1PiecewiseResponse
    record, _ = declared_control(development, cn_reference, 'two-narrow')
    f, g = [BM1PiecewiseResponse.from_inner_values(values, pieces=2)
            for values in CANDIDATES[name]]
    return replace(record, record_id=name,
                   parameters=(record.parameters[0], f, g, *record.parameters[3:]))


def _unit(development, cn_reference, candidate, method):
    """A new prior for every candidate; no state reuse across candidates."""
    from dataclasses import replace
    import numpy as np
    from src.method.bm_1_response import replay_curve_record
    from src.model.bm_1 import bm_1
    from src.solver.finite_volume import finite_volume_1d_fitted
    from src.validation.bm_1.interval_terminal_tilt import terminal_tilt_update
    record = _record(development, cn_reference, candidate)
    f, g = record.parameters[1:3]
    begun = perf_counter()
    if method in SEARCH_SETTINGS:
        record = replace(record, observations=record.observations.prefix(1/24))
        result = replay_curve_record(record, f, g, keep_history=True, **SEARCH_SETTINGS[method])
        edges, posterior, mean, score = (result.x_edges, result.posterior_mass[-1],
            result.posterior_mean[-1], result.total_log_likelihood)
        diagnostics = dict(maximum_abs_mass_error=result.maximum_abs_mass_error,
            minimum_cell_mass=result.minimum_cell_mass,
            deposition_variance_bound=result.prediction_diagnostics[-1]['accumulator_deposition_variance_bound'])
        logs = [None if mass == 0. else float(np.log(mass)) for mass in posterior]
    else:
        model = bm_1(record.baseline, record.baseline, record.horizon, 2)
        solver = finite_volume_1d_fitted(vectorized_inputs=True)
        model.fokker_planc_reference = solver
        edges, states, _ = solver.grid(REFERENCE_SETTINGS['cells'])
        prior = record.initial_mass_builder(edges)
        start, end = record.observations.starts[0], record.observations.ends[0]
        p = record.parameters
        drift, diffusion, options = model._fv_evolution_fields(p, record.sigma_x,
            np.array([start, end]), [], REFERENCE_SETTINGS['max_step'])
        reward = lambda t, x: model.eval(t, x, p[0], p[1], p[2], p[4], p[5])[-1]
        result = terminal_tilt_update(solver, [start, end], prior, drift, diffusion,
            reward, record.observations.values[0], record.sigma_y,
            **{k: v for k, v in REFERENCE_SETTINGS.items() if k != 'cells'}, **options)
        posterior, score = result['posterior_mass'], result['log_evidence']
        mean = posterior @ states
        diagnostics = {k: result[k] for k in ('maximum_abs_mass_error',
            'maximum_relative_conditional_guard', 'maximum_overlap_log_difference', 'contours')}
        logs = [None if np.isneginf(v) else float(v) for v in result['log_posterior_mass']]
    return dict(candidate=candidate, method=method, log_evidence=float(score),
        x_edges=edges.tolist(), posterior_mass=posterior.tolist(),
        log_posterior_mass=logs, posterior_state_mean=float(mean), diagnostics=diagnostics,
        elapsed_seconds=perf_counter()-begun, reused=False)


def compare_search(rows, tolerances):
    """Compare signed candidate differences, not only an arbitrary common offset."""
    import numpy as np
    from src.params.bm_1 import conservative_mass_transfer
    refs = {r['candidate']: r for r in rows if r['method'] == 'terminal1921'}
    output = []
    for method in SEARCH_SETTINGS:
        local = {r['candidate']: r for r in rows if r['method'] == method}
        if set(refs) != set(CANDIDATES) or set(local) != set(CANDIDATES):
            output.append(dict(method=method, local_search_feasible=False, incomplete=True))
            continue
        absolute, pairs = [], []
        for name in CANDIDATES:
            a, b = local[name], refs[name]
            mapped = conservative_mass_transfer(a['x_edges'], a['posterior_mass'], b['x_edges'])
            absolute.append(dict(candidate=name, signed_score_error=a['log_evidence']-b['log_evidence'],
                score_error=abs(a['log_evidence']-b['log_evidence']),
                posterior_mean_error=abs(a['posterior_state_mean']-b['posterior_state_mean']),
                posterior_mass_l1=float(np.abs(mapped-np.array(b['posterior_mass'])).sum())))
        names = list(CANDIDATES)
        for i, a in enumerate(names):
            for b in names[i+1:]:
                exact = refs[b]['log_evidence']-refs[a]['log_evidence']
                cheap = local[b]['log_evidence']-local[a]['log_evidence']
                pairs.append(dict(a=a, b=b, reference_difference=exact, search_difference=cheap,
                    difference_error=abs(cheap-exact), sign_agrees=bool(np.sign(exact) == np.sign(cheap)),
                    reference_gap_above_gate=bool(abs(exact) > tolerances['total_log_likelihood'])))
        posterior_pass = all(r['posterior_mean_error'] <= tolerances['final_state_mean']
            and r['posterior_mass_l1'] <= tolerances['final_mass_l1'] for r in absolute)
        centered_pass = all(r['difference_error'] <= tolerances['total_log_likelihood'] for r in pairs)
        ordering_pass = all(r['sign_agrees'] for r in pairs)
        output.append(dict(method=method, absolute_errors=absolute, pairwise_differences=pairs,
            absolute_score_gate_pass=all(r['score_error'] <= tolerances['total_log_likelihood'] for r in absolute),
            difference_gate_pass=centered_pass, posterior_gates_pass=posterior_pass,
            ordering_pass=ordering_pass, local_search_feasible=centered_pass and posterior_pass and ordering_pass,
            median_local_speed_ratio=float(np.median([refs[n]['elapsed_seconds']/local[n]['elapsed_seconds'] for n in names])),
            scope='One first hour and three fixed curve pairs; no stream/parameter-domain acceptance.'))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-report', type=Path, required=True)
    parser.add_argument('--cn-reference', type=Path, required=True)
    parser.add_argument('--terminal-reference', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--unit', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ('OPENBLAS_NUM_THREADS', 'OPENBLAS_DEFAULT_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    from src.tools.bm_1_reports import _atomic_report, _RunLock
    development, cn = [json.loads(p.read_bytes()) for p in (args.development_report, args.cn_reference)]
    if args.unit:
        candidate, method = args.unit.split('/')
        result = _unit(development, cn, candidate, method)
        _atomic_report(args.run_dir/(candidate+'_'+method+'.json'), result)
        return 0
    from src.exe.run_model.bm_1.curves import source_snapshot
    from src.exe.run_model.bm_1_interval_terminal_tilt import _frozen_terminal_reference
    from src.validation.bm_1.interval_terminal_tilt import declared_terminal_control
    import numpy as np
    import scipy
    with _RunLock(args.run_dir):
        result_path = args.run_dir/'report.json'
        if result_path.exists():
            raise FileExistsError('Preserve completed evidence; do not rerun the pilot.')
        begun, source = perf_counter(), source_snapshot()
        if source['dirty']:
            raise ValueError('Commit the pilot and plan before freezing its source.')
        input_paths = (args.development_report, args.cn_reference, args.terminal_reference)
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
        _, design = declared_terminal_control(development, cn, 'day')
        saved, equivalence = _frozen_terminal_reference(args.terminal_reference.read_bytes(),
            {str(p): hashes[str(p)] for p in input_paths[:2]}, source, design)
        manifest = dict(schema='f4-h2-search-pilot-v1', source=source, inputs_sha256=hashes,
            candidate_inner_values=CANDIDATES, search_settings=SEARCH_SETTINGS,
            reference_settings=REFERENCE_SETTINGS, interval_index=0, intervals=1,
            sigma_y=.04, prior='Beta(2,3)', comparison_tolerances=development['development_tolerances'],
            budget_seconds=900., compute_admission_seconds=870., reference_equivalence=equivalence,
            environment=dict(numpy=np.__version__, scipy=scipy.__version__, python=sys.version))
        manifest_path = args.run_dir/'manifest.json'
        ledger_path = args.run_dir/'budget.json'
        if manifest_path.exists():
            if not args.resume or json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
                raise ValueError('Resume requires identical source, inputs, design and environment.')
            ledger = json.loads(ledger_path.read_text())
            # Conservatively charge interruption/idle time for an unfinished invocation.
            if ledger['invocation_started_epoch'] is not None:
                ledger['charged_seconds'] += max(0., time()-ledger['invocation_started_epoch'])
        else:
            if args.resume:
                raise ValueError('No pilot exists to resume.')
            _atomic_report(manifest_path, manifest)
            ledger = dict(charged_seconds=0., invocation_started_epoch=None)
        used = ledger['charged_seconds']
        ledger['invocation_started_epoch'] = time()
        _atomic_report(ledger_path, ledger, replace_existing=ledger_path.exists())
        units, status = [], 'completed'
        first = saved['rows'][24]
        if first['setting_index'] != 1 or first['interval_index'] != 0:
            raise ValueError('Saved reference is not the first 1921-cell hour.')
        import math
        ref = dict(candidate='two_saved', method='terminal1921',
            log_evidence=first['log_evidence'], x_edges=saved['replays'][1]['x_edges'],
            log_posterior_mass=first['log_posterior_mass'],
            posterior_mass=[0. if v is None else math.exp(v) for v in first['log_posterior_mass']],
            posterior_state_mean=first['posterior_state_mean'], elapsed_seconds=first['elapsed_seconds'],
            diagnostics={k: first[k] for k in ('maximum_relative_conditional_guard',
                'maximum_overlap_log_difference', 'maximum_abs_mass_error')}, reused=True)
        _atomic_report(args.run_dir/'two_saved_terminal1921.json', ref,
            replace_existing=(args.run_dir/'two_saved_terminal1921.json').exists())
        tasks = [(n, m) for m in (*SEARCH_SETTINGS, 'terminal1921') for n in CANDIDATES]
        for candidate, method in tasks:
            unit_path = args.run_dir/(candidate+'_'+method+'.json')
            if not unit_path.exists():
                remaining = 870.-used-(perf_counter()-begun)
                if remaining <= 0.:
                    status = 'budget_stopped'; break
                command = [sys.executable, '-B', '-m', 'src.exe.run_model.bm_1_interval_search_pilot',
                    '--development-report', str(args.development_report), '--cn-reference', str(args.cn_reference),
                    '--terminal-reference', str(args.terminal_reference), '--run-dir', str(args.run_dir),
                    '--unit', candidate+'/'+method]
                try:
                    subprocess.run(command, timeout=remaining, check=True)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                    status = 'budget_stopped' if isinstance(exc, subprocess.TimeoutExpired) else 'unit_failed'
                    break
            units.append(json.loads(unit_path.read_text()))
            from src.exe.run_model.progress import runtime_text
            # Exact units dominate cost; avoid an equal-task ETA.
            completed = sum(1. if r['method'] == 'terminal1921' else .01 for r in units)
            print(f'{candidate}/{method}: '+runtime_text(begun, completed, 3.06), flush=True)
            if source_snapshot() != source or any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)] for p in input_paths):
                raise ValueError('Frozen source/input changed.')
        ledger = dict(charged_seconds=used+perf_counter()-begun, invocation_started_epoch=None)
        _atomic_report(ledger_path, ledger, replace_existing=True)
        comparisons = compare_search(units, development['development_tolerances'])
        report = dict(**manifest, status=status, units=units, comparisons=comparisons,
            total_charged_seconds=ledger['charged_seconds'],
            reused_reference_intervals=1, old_reference_intervals_recomputed=0,
            numerical_fit_readiness=False, historical_fit_run=False,
            historical_observation_values_read=0, scientific_acceptance=False,
            next_step='Assess the declared comparisons before admitting any fitting or stream check.')
        _atomic_report(result_path, report)
        print(json.dumps(dict(status=status, seconds=ledger['charged_seconds'], comparisons=comparisons)), flush=True)
        return 0 if status == 'completed' and any(r['local_search_feasible'] for r in comparisons) else 2


if __name__ == '__main__':
    raise SystemExit(main())
