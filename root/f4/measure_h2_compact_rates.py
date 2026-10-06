"""One bounded F historical linear-reference rate/precision pilot, never a fit.

Read TRAIN/selection only. Preserve independent base/fine prior trajectories,
full hourly log states, failed work and the cumulative native/H ledger. A
short linear prefix cannot admit full-window or nonlinear numerical accuracy.
"""
from time import perf_counter, time
BEGUN = perf_counter()-.25

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'files/output/bm_1/runs/estimation'
READY = ROOT/'root/f4/h2_compact_data_ready_20261007.json'
EXTRA_SOURCE = ('src/method/bm_1_interval_forecast.py',
    'src/tools/bm_1_h2_data_packet.py', 'src/tools/import_data/import_input.py',
    'root/f4/h2_compact_data_policy_20261006.json',
    'root/f4/h2_compact_method_decision_20261006.json',
    'root/f4/h2_compact_data_ready_20261007.json',
    'root/f4/measure_h2_compact_rates.py', 'root/f4/run_h2_compact_rates.ps1',
    'root/f4/check_h2_compact_rates.py')
CANDIDATE = dict(id='linear', pieces=2, logits=[0., 0.])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_observations(packet, path):
    """Existing timestamp-first decoder; final/prior Y are never selected."""
    import numpy as np
    from src.tools.bm_1_history_io import read_raw_power_window
    policy = packet['policy']; split = policy['split']
    require(sha(path) == packet['input_sha256']['input'], 'Demand workbook changed since data freeze.')
    frame = read_raw_power_window(path, columns=policy['population']['measurement_columns'],
        start=split['train_start'], end=split['holdout_start'], timezone=split['timezone'],
        sheet=policy['header_evidence']['input_sheet'])
    raw = frame.to_numpy(dtype=float)
    require(raw.shape == (72, 13) and not np.isinf(raw).any(), 'Wrong fixed MED population/window or infinite reading.')
    seen = np.isfinite(raw).all(axis=1)
    group = np.full(72, np.nan)
    group[seen] = raw[seen].mean(axis=1)  # Never average a partial population.
    # Declared kWh/hour -> kW and frozen normalized average power; no clipping.
    values = group/packet['frozen_scales']['power_kW']
    require(sha(path) == packet['input_sha256']['input'], 'Workbook changed during observation selection.')
    complete = np.flatnonzero(seen[:48])
    require(len(complete) >= 3, 'Need three complete TRAIN hours for the declared prefix pilot.')
    return dict(times=[t.isoformat() for t in frame.index], observed=seen,
        values=values, group_kW=group, train_complete=int(seen[:48].sum()),
        selection_complete=int(seen[48:].sum()), pilot_end_hour=24+int(complete[2])+1,
        selected_train_indices=complete[:3].tolist(), final_values_decoded=0,
        prior_values_decoded=0, imputed_values=0)


def work_graph(rates, train_complete, selection_complete, charged):
    """Price every approved job; missing final Y is conservatively unknown.

    Linear prefix rates are an extrapolation, not permission to fit. The
    factor 1.5 applies once to numerical work, never divides candidates by 8.
    Auxiliary caps are explicit future reservations, not measured durations.
    """
    require(0 <= train_complete <= 48 and 0 <= selection_complete <= 24,
        'Invalid complete-hour counts.')
    for grid in ('base', 'fine'):
        require(set(rates[grid]) == {'observed', 'prediction', 'moments', 'cdf'}
            and all(math.isfinite(v) and v > 0. for v in rates[grid].values()), 'Incomplete measured rate types.')
    jobs = []
    def add(name, role, base, fine, kind):
        cost = 0.
        for grid, units in (('base', base), ('fine', fine)):
            rate = max(rates[grid]['observed'], rates[grid]['prediction']) if kind == 'unknown_final' else rates[grid][kind]
            cost += units*rate
        jobs.append(dict(id=name, role=role, base_units=base, fine_units=fine,
            rate_type=kind, planned_seconds=1.5*cost))
    add('49_candidate_train_observed', 'search', 49*train_complete, 0, 'observed')
    add('49_candidate_train_missing', 'search', 49*(48-train_complete), 0, 'prediction')
    add('49_candidate_prior_lead', 'search', 49*24, 0, 'prediction')
    add('three_selection_observed', 'validation', 3*selection_complete, 0, 'observed')
    add('three_selection_missing', 'validation', 3*(24-selection_complete), 0, 'prediction')
    add('three_fine_train_selection_observed', 'validation', 0, 3*(train_complete+selection_complete), 'observed')
    add('three_fine_train_selection_missing', 'validation', 0, 3*(72-train_complete-selection_complete), 'prediction')
    add('three_fine_prior_lead', 'validation', 0, 3*24, 'prediction')
    add('final_prequential_winner_and_linear', 'validation', 2*168, 2*168, 'unknown_final')
    add('selection_final_predictive_moments', 'validation', 3*24+2*168, 3*24+2*168, 'moments')
    add('open_week_moments', 'validation', 2*168, 2*168, 'moments')
    add('open_week_seven_distributions', 'validation', 2*7, 2*7, 'cdf')
    auxiliary = [dict(id='search_imports_JIT_optimizer_wrappers', cap_seconds=300., role='search'),
        dict(id='validation_imports_diagnostics_export_handoff', cap_seconds=300., role='validation')]
    validation = sum(j['planned_seconds'] for j in jobs if j['role'] == 'validation')+300.+60.
    total = charged+sum(j['planned_seconds'] for j in jobs)+600.+60.
    return dict(schema='f4-h2-linear-prefix-runtime-extrapolation-v1', jobs=jobs,
        auxiliary=auxiliary, stop_reserve_seconds=60., safety_factor=1.5,
        charged_seconds=charged, projected_total_seconds=total, projected_total_hours=total/3600.,
        protected_validation_seconds=max(7200., validation),
        within_ten_hour_ceiling=total <= 36000., within_six_hour_target=total <= 21600.,
        rate_scope='linear historical prefix only; not a worst-case bound over nonlinear families or later states',
        completed_pilot_work_deducted=False, auxiliary_measured=False,
        runtime_admitted=False, historical_fit_ready=False, scientific_acceptance=False)


def frozen_packet():
    from src.method.bm_1_interval_precise import digest
    from src.tools.bm_1_h2_data_packet import validate_policy
    from src.tools.bm_1_reports import _read_array_checkpoint
    ready = json.loads(READY.read_text(encoding='utf-8'))
    packet_path, input_path = ROOT/ready['packet_path'], ROOT/ready['inputs_path']
    packet = json.loads(packet_path.read_text(encoding='utf-8'))
    require(digest({k:v for k,v in packet.items() if k != 'identity'}) == packet['identity'] == ready['packet_identity'], 'Frozen packet identity differs.')
    validate_policy(packet['policy'])
    require(packet['policy'] == json.loads((ROOT/'root/f4/h2_compact_data_policy_20261006.json').read_text(encoding='utf-8')), 'Source policy and frozen packet differ.')
    require(packet['conditional_data_ready'] is True and packet['target_measurement_values_decoded'] == 0
        and packet['final_realized_prices_decoded'] == 0, 'Wrong data-only packet.')
    require(sha(ROOT/ready['audit_path']) == ready['audit_sha256'], 'Raw data audit changed.')
    audit = json.loads((ROOT/ready['audit_path']).read_text(encoding='utf-8'))
    require(audit['passed'] is True and audit['packet_identity'] == packet['identity'], 'Frozen raw data audit did not pass.')
    require(sha(input_path) == packet['outputs_sha256']['inputs.npz']
        and sha(ROOT/ready['baseline_path']) == packet['outputs_sha256']['baseline.csv'], 'Frozen B/U outputs changed.')
    inputs = _read_array_checkpoint(input_path.read_bytes())
    return ready, packet, inputs


def compare_states(base, fine):
    import numpy as np
    from src.params.bm_1 import conservative_mass_transfer
    be, fe = np.linspace(0., 1., len(base)+1), np.linspace(0., 1., len(fine)+1)
    bp, fp = np.exp(base), np.exp(fine)
    mean = abs(float(bp@((be[:-1]+be[1:])/2.))-float(fp@((fe[:-1]+fe[1:])/2.)))
    l1 = float(np.sum(np.abs(bp-conservative_mass_transfer(fe, fp, be))))
    require(mean <= .002 and l1 <= .05, 'Historical prefix base/fine state precision failed.')
    return dict(state_mean_difference=mean, conservative_state_L1_difference=l1)


def _pilot(args, reservation, output, manifest, packet, inputs, observations):
    import numpy as np
    from scipy.special import logsumexp
    from src.exe.run_model.bm_1_history import _beta_mass
    from src.exe.run_model.bm_1_h2_native import _worker_init
    from src.method.bm_1_interval_native import freeze_native_operator, advance_frequency_block, native_terminal_update
    from src.method.bm_1_interval_forecast import (interval_cdf_plan, native_interval_moments,
        native_interval_distribution, compare_interval_moments, compare_interval_distributions,
        interval_forecast_checkpoint)
    from src.tools.bm_1_reports import _atomic_report
    n = packet['numerics']; physical = packet['physical']; sigma = physical['sigma_y']
    plan = interval_cdf_plan(sigma)
    states, specs, scores = {}, {}, dict(base=0., fine=0.)
    for grid in ('base', 'fine'):
        settings = dict(cells=n[grid+'_cells'], max_step=n[grid+'_max_step'],
            frequencies=n['frequencies'], span=n['span'], centers=n['centers'],
            relative_guard=n['guards']['conditional'], overlap_log_tolerance=n['guards']['overlap'])
        mass = _beta_mass(np.linspace(0., 1., settings['cells']+1), **{k:packet['prior'][k] for k in ('alpha','beta')})
        with np.errstate(divide='ignore'):
            states[grid] = np.log(mass)
        require(abs(float(logsumexp(states[grid]))) <= 1e-10, 'Invalid declared Beta prior mass.')
        specs[grid] = dict(settings=settings, physical=physical, input_edges=inputs['input_edges'].tolist(),
            baseline=inputs['baseline'].tolist(), price=inputs['price'].tolist(), horizon=[0.,11.],
            time_unit='day', midnight_resets=0)
    timings, comparisons = [], []
    first, sample = set(), dict(base={}, fine={})
    def check_deadline():
        if time() >= reservation['deadline'] or (output/'STOP').exists():
            raise TimeoutError('Historical rate pilot stopped before its next job.')
    pool = mp.get_context('spawn').Pool(8, initializer=_worker_init, initargs=(reservation['deadline'],))
    def mapper(jobs):
        check_deadline()
        pending = pool.map_async(advance_frequency_block, jobs)
        while True:
            check_deadline()
            try:
                return pending.get(timeout=min(.5, max(.001, reservation['deadline']-time())))
            except mp.TimeoutError:
                pass
    def checkpoint(grid, kind, hour, begun, build_seconds, payload):
        # Preserve a completed scalar job even if it just crossed the compute
        # deadline. The next job is blocked; the 60 s reserve pays shutdown/I/O.
        cold = (grid, kind) not in first and kind != 'cdf'
        first.add((grid, kind))
        path = output/'jobs'/f'{len(timings):04d}.npz'
        _atomic_report(path, dict(manifest_identity=manifest['identity'], grid=grid,
            rate_type=kind, hour=hour, **payload))
        # Reused operator construction is priced again for separate moment/CDF
        # jobs: conservative cost for their future independently saved passes.
        elapsed = perf_counter()-begun+build_seconds
        row = dict(grid=grid, rate_type=kind, hour=hour, elapsed_seconds=elapsed,
            cold=cold, checkpoint=str(path.relative_to(output)), checkpoint_sha256=sha(path))
        timings.append(row)
        _atomic_report(output/'timings.json', timings, replace_existing=True)
        print(json.dumps(dict(grid=grid, kind=kind, hour=hour, seconds=elapsed, cold=cold)), flush=True)
    try:
        for hour in range(observations['pilot_end_hour']):
            check_deadline()
            obsindex = hour-24
            seen = hour >= 24 and bool(observations['observed'][obsindex])
            moment_results, cdf_results = {}, {}
            for grid in ('base','fine'):
                check_deadline()
                unit = dict(start=float(inputs['input_edges'][hour]), end=float(inputs['input_edges'][hour+1]))
                started = perf_counter()
                operator = freeze_native_operator(specs[grid], CANDIDATE, unit)
                build_seconds = perf_counter()-started
                prior = states[grid]
                if seen:
                    begun = perf_counter()
                    moment = native_interval_moments(operator, prior, sigma)
                    moment_results[grid] = moment
                    checkpoint(grid, 'moments', hour, begun, build_seconds,
                        dict(prior_log_state=prior, **moment))
                    if obsindex in observations['selected_train_indices'][1:]:
                        begun = perf_counter()
                        full = native_interval_distribution(operator, prior, plan, mapper=mapper)
                        active = np.isfinite(moment['log_state'])
                        require(np.array_equal(active,np.isfinite(full['log_state']))
                            and np.max(np.abs(moment['log_state'][active]-full['log_state'][active])) <= 1e-7,
                            'Moment/CF full log-state disagreement.')
                        cdf_results[grid] = full
                        checkpoint(grid, 'cdf', hour, begun, build_seconds,
                            dict(prior_log_state=prior, forecast=interval_forecast_checkpoint(
                                dict(hour=hour, interval_mean=moment['interval_mean'], interval_variance=moment['interval_variance']),
                                full['log_state'], full, manifest['identity'])))
                    begun = perf_counter()
                    result = native_terminal_update(operator, prior, float(observations['values'][obsindex]),
                        sigma, specs[grid]['settings'], mapper=mapper)
                    states[grid] = result['log_posterior_mass']; scores[grid] += result['log_evidence']
                    checkpoint(grid, 'observed', hour, begun, build_seconds,
                        dict(prior_log_state=prior, posterior_log_state=states[grid],
                            cumulative_log_score=scores[grid], **result))
                else:
                    begun = perf_counter()
                    result = advance_frequency_block((operator, prior, np.array([0], dtype=int), 2., 0., 0.))
                    states[grid] = result['log_amplitude']
                    checkpoint(grid, 'prediction', hour, begun, build_seconds,
                        dict(prior_log_state=prior, log_state=states[grid],
                            maximum_abs_mass_error=result['maximum_abs_mass_error']))
                require(abs(float(logsumexp(states[grid]))) <= 1e-10, 'Pilot log-state is not normalized.')
            comparison = dict(hour=hour, **compare_states(states['base'],states['fine']))
            comparison['pilot_total_log_score_difference'] = abs(scores['base']-scores['fine'])
            require(comparison['pilot_total_log_score_difference'] <= .05, 'Pilot total log-score precision failed.')
            if moment_results:
                comparison['moments'] = compare_interval_moments(moment_results['base'],moment_results['fine'])
                comparison['pre_update_prediction_states'] = compare_states(moment_results['base']['log_state'], moment_results['fine']['log_state'])
            if cdf_results:
                comparison['distributions'] = compare_interval_distributions(cdf_results['base']['distribution'],cdf_results['fine']['distribution'])
            comparisons.append(comparison)
            _atomic_report(output/'comparisons.json', comparisons, replace_existing=True)
        for grid in ('base','fine'):
            for kind in ('observed','prediction','moments','cdf'):
                warm = [r['elapsed_seconds'] for r in timings if r['grid'] == grid and r['rate_type'] == kind and not r['cold']]
                require(len(warm) >= 2, 'Need at least two warm measured jobs per grid/type.')
                sample[grid][kind] = max(warm)
        return dict(status='completed', prefix_precision_passed=True, rates=sample,
            cold_job_seconds=sum(r['elapsed_seconds'] for r in timings if r['cold']),
            observed_prefix_hours=3, chronological_prefix_hours=observations['pilot_end_hour'],
            cdf_probe_hours=[24+i for i in observations['selected_train_indices'][1:]],
            comparisons=comparisons, window_scores=scores, family=CANDIDATE,
            full_window_precision_passed=False, nonlinear_rates_measured=False,
            historical_fit_ready=False, numerical_likelihood_ready=False, runtime_admitted=False,
            blind_h3_acceptance=False, scientific_acceptance=False)
    finally:
        # Terminate on pause/failure as well: no detached workers or background work.
        pool.terminate(); pool.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-workbook', type=Path, default=ROOT/'files/input/data/input.xlsx')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, default=OUT/'f4_h2_remaining_budget.json')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--max-seconds', type=float, default=300.)
    parser.add_argument('--launcher-seconds', type=float, default=0., help='Charge measured immutable export/startup from the PC launcher too.')
    parser.add_argument('--check', action='store_true', help='Only focused data-boundary/workgraph control, no model propagation.')
    args = parser.parse_args()
    args.output = args.output.resolve(); args.campaign = args.campaign.resolve()
    require(not args.output.exists() and not args.output.with_name(args.output.name+'_review.zip').exists(), 'Preserve every attempt; no overwrite or automatic rerun.')
    require(re.fullmatch('[0-9a-f]{40}', args.source_commit) is not None, 'Declare exact exported source commit.')
    require(75. < args.max_seconds <= 300., 'Maximum pilot grant 300 s including 60 s stop reserve.')
    require(math.isfinite(args.launcher_seconds) and 0. <= args.launcher_seconds <= 60., 'Invalid launcher startup charge.')
    begun = BEGUN-args.launcher_seconds
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    from src.exe.run_model.bm_1_h2_native import native_reservation, _source, _environment
    from src.method.bm_1_interval_precise import digest
    from src.tools.bm_1_reports import _RunLock, _atomic_report
    def hashes():
        return dict(_source(), **{p:sha(ROOT/p) for p in EXTRA_SOURCE})
    result, error = None, None
    archive = args.output.with_name(args.output.name+'_review.zip')
    with native_reservation(args.campaign, args.max_seconds, 'compact-rate-control' if args.check else 'compact-historical-rate', begun=begun) as reservation, _RunLock(args.output):
        import openpyxl
        source = hashes()
        manifest = dict(schema='f4-h2-compact-historical-rate-v1', source_commit=args.source_commit,
            source_sha256=source, environment=dict(_environment(), openpyxl=openpyxl.__version__, workers=1 if args.check else 8),
            launcher_seconds=args.launcher_seconds,
            grant_seconds=reservation['grant'], stop_reserve_seconds=60., native_seconds_before=reservation['native_seconds_before'],
            candidate=CANDIDATE, mode='check' if args.check else 'linear_prefix_rates',
            no_optimizer=True, no_final_Y=True, numerical_acceptance_scope='linear 24-hour prior plus first three complete TRAIN hours only')
        try:
            ready, packet, inputs = frozen_packet()
            manifest['packet_identity'] = packet['identity']
            manifest['packet_input_sha256'] = {ready[k]:sha(ROOT/ready[k]) for k in ('packet_path','inputs_path','baseline_path','audit_path')}
            if args.check:
                from root.f4.check_h2_compact_rates import control
                manifest['identity'] = digest(manifest)
                _atomic_report(args.output/'manifest.json', manifest)
                result = control(packet)
            else:
                observations = load_observations(packet, args.input_workbook)
                observation_path = args.output/'observations.npz'
                _atomic_report(observation_path, observations)
                manifest['observation_sha256'] = sha(observation_path)
                manifest['input_workbook_sha256'] = sha(args.input_workbook)
                manifest['identity'] = digest(manifest)
                _atomic_report(args.output/'manifest.json', manifest)
                _atomic_report(args.output/'packet.json', packet)
                result = _pilot(args,reservation,args.output,manifest,packet,inputs,observations)
            require(time() < reservation['deadline'], 'Pilot exhausted its compute allowance; do not repeat automatically.')
            require(hashes() == source, 'Frozen source changed during rate measurement.')
            if not args.check:
                require(sha(args.input_workbook) == manifest['input_workbook_sha256'], 'Workbook changed during pilot.')
                for p,h in manifest['packet_input_sha256'].items():
                    require(sha(ROOT/p) == h, 'Frozen packet input changed during pilot.')
        except (Exception, KeyboardInterrupt) as exc:
            error = traceback.format_exc()
            result = dict(status='paused' if isinstance(exc, (TimeoutError,mp.TimeoutError,KeyboardInterrupt)) else 'failed',
                reason=str(exc), traceback=error, diagnostics=getattr(exc,'diagnostics',{}), historical_fit_ready=False,
                numerical_likelihood_ready=False, runtime_admitted=False, scientific_acceptance=False,
                automatic_retry=False)
            if not (args.output/'manifest.json').exists():
                manifest['identity'] = digest(manifest)
                _atomic_report(args.output/'manifest.json', manifest)
        for gate in ('historical_fit_ready','numerical_likelihood_ready','runtime_admitted',
                     'blind_h3_acceptance','scientific_acceptance'):
            result[gate] = False
        _atomic_report(args.output/'report.json', dict(result,
            elapsed_seconds_before_ledger_close=perf_counter()-begun))
        with zipfile.ZipFile(args.output/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in source.items():
                raw=(ROOT/p).read_bytes()
                require(hashlib.sha256(raw).hexdigest()==h, 'Source changed before export: '+p)
                z.writestr(p,raw)
        # Compress all source/arrays while the invocation is still charged.
        # Only the small authoritative closed ledger/estimate are appended later.
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(args.output.rglob('*')):
                if p.is_file() and p.name != 'run.lock' and not p.name.endswith('.partial'):
                    z.write(p,str(p.relative_to(args.output)))
    ledger_raw = args.campaign.read_bytes(); ledger=json.loads(ledger_raw)
    (args.output/'ledger.json').write_bytes(ledger_raw)
    if result['status']=='completed' and not args.check:
        estimate = work_graph(result['rates'],observations['train_complete'],observations['selection_complete'],ledger['charged_seconds'])
        estimate['cold_job_seconds'] = result['cold_job_seconds']
        estimate['cold_jobs_within_search_aux_cap'] = result['cold_job_seconds'] <= 300.
        _atomic_report(args.output/'work_graph.json',estimate)
    with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_STORED) as z:
        z.write(args.output/'ledger.json','ledger.json')
        if (args.output/'work_graph.json').exists():
            z.write(args.output/'work_graph.json','work_graph.json')
    print(json.dumps(dict(status=result['status'], review_zip=str(archive),
        H_charged_seconds=ledger['charged_seconds'], invocation_charged_seconds=ledger['entries'][-1]['charged_seconds'])),flush=True)
    if error:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
