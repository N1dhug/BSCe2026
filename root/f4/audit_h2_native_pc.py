"""Read-only native PC evidence audit; no likelihood replay or JIT compilation.

The CLI charges the bounded review against the already imported PC ledger.
audit() only reads evidence. CRLF comparisons do not alter execution
manifests, checkpoints, original hashes or future resume checks.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
from time import perf_counter
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'files/output/bm_1/runs/estimation'
REVIEW = OUT/'f4_h2_native_pc_pilot_20261006'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read_archive(path):
    files = {}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            p = PurePosixPath(info.filename.replace('\\', '/'))
            require(not p.is_absolute() and '..' not in p.parts and ':' not in str(p), 'Unsafe ZIP path.')
            if info.is_dir():
                continue
            require(str(p) not in files, 'Duplicate ZIP member.')
            files[str(p)] = z.read(info)  # Verifies original ZIP CRCs.
    return files


def checkout_hashes(path):
    raw = path.read_bytes()
    choices = {'byte_identical': hashlib.sha256(raw).hexdigest()}
    if path.suffix not in ('.zip', '.gz'):
        lf = raw.replace(b'\r\n', b'\n')
        choices.update(exact_checkout_lf=hashlib.sha256(lf).hexdigest(),
            exact_checkout_crlf=hashlib.sha256(lf.replace(b'\n', b'\r\n')).hexdigest())
    return choices


def ordered_sum(values):
    """Exact declared CPython 3.11 float summation, not Python 3.12 sum()."""
    total = 0.
    for value in values:
        total += value
    return total


def validate_pc_checkpoint(saved, spec, candidate):
    """Read-only identity/chronology/state checks with the PC's exact sum.

    Python 3.12 changed float sum(). The original execution validator and PC
    checkpoint are untouched: we reconstruct its 3.11 scalar bit-for-bit,
    without substituting evidence, changing checksums or accepting a tolerance.
    """
    import numpy as np
    from scipy.special import logsumexp
    from src.method.bm_1_interval_precise import digest, replay_units
    require(saved['identity'] == digest(dict(spec=spec,candidate=candidate))
        and saved['checksum'] == digest(saved['payload']), 'Checkpoint identity/checksum differs.')
    payload, units = saved['payload'], replay_units(spec)
    rows = payload['rows']; require(len(rows) <= len(units), 'Checkpoint exceeds declared stream.')
    for i,row in enumerate(rows):
        require(row['unit'] == i and all(row[k] == units[i][k] for k in units[i])
            and row['observed'] == (row['log_evidence'] is not None)
            and math.isfinite(row['state_mean']) and row['maximum_abs_mass_error'] <= 1e-10,
            'Checkpoint chronology/observation/probability differs.')
    score = ordered_sum(r['log_evidence'] for r in rows if r['observed'])
    require(math.isfinite(score) and score == payload['log_likelihood'], 'PC ordered cumulative evidence differs.')
    logs = np.array([-np.inf if v is None else v for v in payload['log_mass']])
    require(logs.shape == (spec['settings']['cells'],) and not np.isnan(logs).any()
        and not np.isposinf(logs).any() and abs(float(logsumexp(logs))) <= 1e-10,
        'Invalid final PC log-state.')
    require(bool(rows) or payload['log_mass'] == spec['initial_log_mass'], 'Declared initial law changed.')
    return payload


def audit(archive):
    import numpy as np
    from scipy.special import logsumexp
    from src.exe.run_model.bm_1_h2_native import (
        NEW_SOURCE, PREFIX, _references, _source, _logs, _comparison, _ledger_valid)
    from src.exe.run_model.bm_1_h2_precise import _load
    from src.method.bm_1_interval_precise import digest
    from src.solver.finite_volume import finite_volume_1d_fitted
    files = read_archive(archive); run = 'f4_h2_native_pc_20261006/'
    expected = {run+n for n in ('algebra.json', 'hour_wall_times.json', 'linear.json.gz',
        'manifest.json', 'p4_c0_plus.json.gz', 'report.json', 'run.lock', 'two_narrow.json.gz')}
    require(set(files) == expected | {'f4_h2_remaining_budget.json'}, 'Unexpected/missing PC evidence members.')
    read = lambda name: json.loads(files[name])
    m, report, timings = (read(run+n) for n in ('manifest.json', 'report.json', 'hour_wall_times.json'))
    require(m['identity'] == digest({k:v for k,v in m.items() if k != 'identity'}), 'Manifest identity changed.')
    require(m['schema'] == 'f4-h2-native-development-v1' and m['development'] is True
        and m['historical_observation_values_read'] == 0 and m['scientific_acceptance'] is False,
        'This audit accepts only the declared synthetic run.')
    require(report['schema'] == 'f4-h2-native-development-report-v1'
        and report['manifest_identity'] == m['identity'] and report['status'] == 'completed'
        and report['equivalence_pass'] is True and report['historical_observation_values_read'] == 0
        and report['scientific_acceptance'] is False and report['numerical_fit_readiness'] is False,
        'PC report is incomplete or claims unsupported acceptance.')
    env = m['environment']
    require(all(env[k] == v for k,v in dict(workers=8, blas_threads=1, numba_threads=1,
        fastmath=False, numba='0.62.1', llvmlite='0.45.1').items())
        and env['platform'].startswith('Windows-') and env['python'].startswith('3.11.') and report['workers'] == 8
        and report['blas_threads'] == 1, 'Declared worker/native environment differs.')
    require(set(m['source_sha256']) == set(_source()), 'Source dependency set differs.')
    bundle = REVIEW/'execution_sources.zip'
    frozen_source = None
    if bundle.exists():
        declared_bundle = _load(REVIEW/'execution_sources_manifest.json')
        require(declared_bundle['pc_manifest_identity'] == m['identity']
            and declared_bundle['source_sha256'] == m['source_sha256']
            and declared_bundle['bundle_sha256'] == hashlib.sha256(bundle.read_bytes()).hexdigest(),
            'Archived executed-source bundle changed.')
        frozen_source = read_archive(bundle)
        require(set(frozen_source) == set(m['source_sha256']), 'Executed-source bundle dependency set differs.')
    source_rows = []
    for name, declared in m['source_sha256'].items():
        choices = ({'archived_execution_bytes': hashlib.sha256(frozen_source[name]).hexdigest()}
            if frozen_source is not None else checkout_hashes(ROOT/name))
        kind = next((k for k,v in choices.items() if v == declared), None)
        require(kind is not None, 'Unexplained source difference: '+name)
        source_rows.append(dict(path=name, declared_sha256=declared, comparison=kind))
    expected_cases, local_inputs = _references()
    by_name = {Path(p).name: Path(p) for p in local_inputs}
    require(len(m['inputs_sha256']) == len(by_name), 'Reference input count differs.')
    for name, declared in m['inputs_sha256'].items():
        filename = name.replace('\\', '/').rsplit('/', 1)[-1]
        require(filename in by_name and declared in checkout_hashes(by_name[filename]).values(),
            'Reference input differs beyond checkout conversion: '+name)
    require([c['id'] for c in m['cases']] == [c['id'] for c in expected_cases], 'Case selection/order differs.')
    initial_prior_errors = {}
    for actual, reference in zip(m['cases'], expected_cases):
        a, b = json.loads(json.dumps(actual)), json.loads(json.dumps(reference))
        x, y = _logs(a['spec'].pop('initial_log_mass')), _logs(b['spec'].pop('initial_log_mass'))
        require(a == b, 'Mathematical case/curve/input/reference changed: '+actual['id'])
        require(np.array_equal(np.isneginf(x), np.isneginf(y)), 'Initial true-zero support changed.')
        active = np.isfinite(x); difference = float(np.max(np.abs(x[active]-y[active])))
        # Separate read-only comparison for CDF/log generation across Windows/
        # Linux libraries, stricter than the 1e-7 trajectory gate. No resume
        # identity is normalized or replaced; the original PC spec is used.
        require(difference <= 1e-10 and abs(float(logsumexp(x))) <= 1e-10,
            'Generated initial law differs beyond numerical rounding.')
        initial_prior_errors[actual['id']] = difference
    control = read(run+'algebra.json')
    control_identity = digest(dict(source={p:m['source_sha256'][p] for p in NEW_SOURCE[:2]}, environment=env))
    require(control['identity'] == control_identity and control['passed'] is True
        and control['checksum'] == digest(control['results']), 'Independent control identity/checksum failed.')
    witness = control['results']
    require(witness['passed'] is True and witness['true_zero_preserved'] is True
        and 0. <= witness['dense_transform_error'] <= 1e-12
        and 0. <= witness['finite_tail_error'] <= 1e-10
        and 0. <= witness['one_jump_log_joint_error'] <= 2e-4
        and len(witness['gaussian_controls']) == 2
        and all(0. <= v <= 2e-8 for r in witness['gaussian_controls'] for v in r.values()),
        'Independent algebra/control bound failed.')
    require(timings['manifest_identity'] == m['identity'], 'Timing identity differs.')
    require([(r['case'],r['unit']) for r in timings['rows']] ==
        [(c['id'],i) for c in m['cases'] for i in range(c['units'])], 'Timing chronology incomplete.')
    summaries, all_rows, all_walls = [], [], []
    for case in m['cases']:
        saved = json.loads(gzip.decompress(files[run+case['id']+'.json.gz']))
        require(saved['native_manifest_identity'] == m['identity'], 'Checkpoint native source identity changed.')
        payload = validate_pc_checkpoint(saved, case['spec'], case['candidate'])
        rows = payload['rows']; spec = case['spec']
        require(payload['status'] == 'completed' and len(rows) == case['units'], 'Incomplete declared prefix.')
        _, states, _ = finite_volume_1d_fitted.grid(spec['settings']['cells'])
        for i,row in enumerate(rows):
            logs = _logs(row['log_posterior_mass'])
            require(logs.shape == states.shape and not np.isnan(logs).any() and not np.isposinf(logs).any()
                and abs(float(logsumexp(logs))) <= 1e-10, 'Invalid intermediate log-state.')
            require(abs(float(np.exp(logs) @ states)-row['state_mean']) <= 1e-12, 'Saved mean differs from full log-state.')
            for k,bound in (('maximum_relative_conditional_guard',1e-8),
                ('maximum_overlap_log_difference',1e-7),('maximum_abs_mass_error',1e-10)):
                require(math.isfinite(row[k]) and 0. <= row[k] <= bound, 'Saved native guard failed: '+k)
            contours = row['contours']
            if row['observed']:
                require(0 < len(contours) <= len(spec['settings']['centers'])
                    and [c['center'] for c in contours] == spec['settings']['centers'][:len(contours)]
                    and contours[-1]['unresolved_cells'] == 0, 'Contour-prefix resolution failed.')
            else:
                require(not contours and row['log_evidence'] is None, 'Missing observation was assimilated.')
            original = case['references'][i]
            ref_logs = (original['log_posterior_mass'] if case['full_logs_each_hour'] else
                case['final_reference_logs'] if i+1 == case['units'] else None)
            cumulative = ordered_sum(r['log_evidence'] for r in rows[:i+1] if r['log_evidence'] is not None)
            reference_score = ordered_sum(r['log_evidence'] for r in case['references'][:i+1] if r['log_evidence'] is not None)
            require(row['reference_errors'] == _comparison(row,logs,original,ref_logs,cumulative,reference_score),
                'Stored equivalence error differs from raw reference comparison.')
        require(rows[-1]['log_posterior_mass'] == payload['log_mass'], 'Final native log-state changed.')
        wall_rows = [r for r in timings['rows'] if r['case'] == case['id']]
        walls = [r['wall_seconds'] for r in wall_rows]
        require(all(r['includes_operator_ipc_checkpoint'] is True for r in wall_rows)
            and all(math.isfinite(w) and w >= r['elapsed_seconds'] > 0. for w,r in zip(walls,rows)),
            'Invalid end-to-end hour timing.')
        summary = dict(case=case['id'], units=len(rows), observed=sum(r['observed'] for r in rows),
            hour_seconds=[r['elapsed_seconds'] for r in rows], hour_wall_seconds=walls, timing_complete=True,
            maximum_reference_errors={k:max(r['reference_errors'].get(k,0.) for r in rows)
                for k in ('cumulative_score','mean','hourly_score','finite_log_mass')})
        summaries.append(summary); all_rows.extend(rows); all_walls.extend(walls)
    require(report['cases'] == summaries and sum(all_walls) <= report['elapsed_seconds'], 'Summary timing/errors differ from raw evidence.')
    ledger = read('f4_h2_remaining_budget.json'); _ledger_valid(ledger)
    seed = _load(REVIEW/'seed_before_pc.json'); _ledger_valid(seed)
    require(ledger['entries'][:-1] == seed['entries'] and ledger['entries'][-1]['status'] == 'closed',
        'PC ledger does not exactly extend the frozen pre-PC ledger.')
    before = sum(e['charged_seconds'] for e in seed['entries'] if e['label'].startswith(PREFIX))
    entry = ledger['entries'][-1]
    label = PREFIX+str(PureWindowsPath(m['campaign']).parent/'f4_h2_native_pc_20261006')
    require(entry['label'] == label and report['native_seconds_before'] == before
        and report['grant_seconds'] == min(900.-before,36000.-seed['validation_reserve_seconds']-seed['charged_seconds'])
        and report['stop_reserve_seconds'] == 60.
        and report['elapsed_seconds'] <= entry['charged_seconds'] < report['grant_seconds'],
        'PC time reservation/charge does not match the report.')
    native_used = sum(e['charged_seconds'] for e in ledger['entries'] if e['label'].startswith(PREFIX))
    require(native_used <= 900., 'Native cumulative allowance exhausted/exceeded.')
    maximum = max(w for w,r in zip(all_walls,all_rows) if r['observed'])
    steady = max(w for c in summaries for i,w in enumerate(c['hour_wall_seconds'])
        if not (c['case'] == 'two_narrow' and i == 0) and m['cases'][[x['id'] for x in m['cases']].index(c['case'])]['references'][i]['log_evidence'] is not None)
    coarse_seconds = 1.5*9072*steady
    illustrative = ledger['charged_seconds']+coarse_seconds+1.5*1344*(2*2561/1921)*steady+3600.+60.
    return dict(schema='f4-h2-native-pc-raw-audit-v1', raw_verified=True,
        archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(), manifest_identity=m['identity'],
        source_files_verified=len(source_rows), source_equivalence=source_rows, environment=env,
        independent_control_verified=True, initial_prior_maximum_log_differences=initial_prior_errors,
        checkpoint_score_arithmetic='Declared CPython 3.11 ordered binary64, exact comparison; no tolerance or checksum replacement.',
        units=len(all_rows), observed_units=sum(r['observed'] for r in all_rows), cases=summaries,
        invocation_seconds=report['elapsed_seconds'], pc_charged_seconds=entry['charged_seconds'],
        campaign_charged_seconds_at_pc_exit=ledger['charged_seconds'], native_seconds_at_pc_exit=native_used,
        maximum_mass_error=max(r['maximum_abs_mass_error'] for r in all_rows),
        maximum_overlap_log_difference=max(r['maximum_overlap_log_difference'] for r in all_rows),
        maximum_relative_conditional_guard=max(r['maximum_relative_conditional_guard'] for r in all_rows),
        maximum_observed_hour_wall_seconds=maximum, maximum_after_initial_worker_startup_seconds=steady,
        planning=dict(historical_admission=False, full_49_plan_admissible=False,
            measured_historical_rates=False, measured_fine_rates=False, base_units=9072, fine_units=1344,
            prior_lead_units_assumed=0, safety_factor=1.5, coarse_only_49_illustrative_hours=coarse_seconds/3600.,
            full_49_illustrative_hours=illustrative/3600., fine_multiplier_is_operation_count_only=2*2561/1921,
            additional_jobs_placeholder_seconds=3600.,
            explanation='Even the base-only 49-pair planning estimate exceeds 10 hours at the warm synthetic maximum. '
                'The full illustration omits actual lead/startup and assumes an unmeasured fine multiplier/additional jobs. '
                'It is not historical throughput or a rigorous runtime bound.'),
        numerical_replays=0, historical_observation_values_read=0, scientific_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=REVIEW/'raw_audit.json')
    args = parser.parse_args(); begun = perf_counter()-.25
    import os
    for name in ('OPENBLAS_NUM_THREADS','OPENBLAS_DEFAULT_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name] = '1'
    from src.exe.run_model.bm_1_h2_native import native_reservation
    from src.exe.run_model.bm_1_h2_precise import _write
    require(not args.output.exists(),'Preserve the existing completed audit.')
    with native_reservation(OUT/'f4_h2_remaining_budget.json',90.,'native-pc-raw-review',begun=begun):
        result = audit(args.archive)
        _write(args.output,result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('source_equivalence','cases')},ensure_ascii=False))


if __name__ == '__main__':
    main()
