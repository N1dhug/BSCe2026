"""Read an existing F4-H2 PC archive; never replay a likelihood or alter a ledger.

CRLF reconstruction is a read-only provenance check. Original manifest/source
hashes and execution/resume checks are not normalized or weakened.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(archive):
    import numpy as np
    from scipy.special import logsumexp
    from src.exe.run_model.bm_1_h2_precise import (
        PRIOR_COMPUTE_SECONDS, _projection, _source)
    from src.method.bm_1_interval_precise import (
        PRECISE_SETTINGS, candidate_panel, digest, replay_units, validate_checkpoint)
    from src.solver.finite_volume import finite_volume_1d_fitted

    raw_archive = Path(archive).read_bytes()
    files = {}
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path = PurePosixPath(info.filename.replace('\\', '/'))
            require(not path.is_absolute() and '..' not in path.parts, 'Invalid archive path.')
            if info.is_dir():
                continue
            require(str(path) not in files, 'Duplicate archive member.')
            files[str(path)] = z.read(info)  # Also checks each ZIP CRC.
    run = 'h2_precise_development_pc_v1/'
    read = lambda name: json.loads(files[name])
    manifest = read(run+'manifest.json')
    report = read(run+'pilot_report.json')
    ledger = read('f4_h2_remaining_budget.json')
    config = read('h2_precise_contract_pc_v1.json')
    require(manifest['identity'] == digest({k: v for k, v in manifest.items() if k != 'identity'}),
            'Manifest identity differs.')
    require(manifest['schema'] == 'f4-h2-precise-batch-v1' and manifest['development'] is True,
            'This audit is for the declared synthetic pilot.')
    require(report['manifest_identity'] == manifest['identity'] and report['status'] == 'completed',
            'Pilot summary does not match the manifest.')
    require((report['workers'], report['blas_threads']) == (8, 1), 'Worker profile differs.')
    require(read(run+'runtime_profile.json') == dict(workers=8, blas_threads=1),
            'Saved runtime profile differs.')
    require(report['campaign'] == manifest['campaign'], 'Campaign identity differs.')
    require(manifest['candidates'] == candidate_panel() and manifest['regularization'] == 0.,
            'Synthetic candidate design differs.')
    spec = manifest['spec']
    require(spec['settings'] == PRECISE_SETTINGS and spec['midnight_resets'] == 0,
            'Numerical settings or state policy differs.')
    require(len(replay_units(spec)) == 24, 'Pilot projection must cover the synthetic 24-hour stream.')

    source_rows = []
    require(set(manifest['source_sha256']) == set(_source()), 'Source dependency set differs.')
    for name, expected in manifest['source_sha256'].items():
        raw = (ROOT/name).read_bytes()
        local_hash = hashlib.sha256(raw).hexdigest()
        crlf_hash = hashlib.sha256(raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()
        kind = ('byte_identical' if local_hash == expected else
                'exact_lf_to_crlf_checkout_conversion' if crlf_hash == expected else None)
        require(kind is not None, 'Unexplained source difference: '+name)
        source_rows.append(dict(path=name, declared_sha256=expected, local_sha256=local_hash,
                                comparison=kind))
    fixture_name = 'files/output/bm_1/runs/estimation/h2_hourly_interval_development_20261005.json'
    require(len(manifest['inputs_sha256']) == 1, 'Synthetic input set differs.')
    raw_fixture = (ROOT/fixture_name).read_bytes()
    input_hash = next(iter(manifest['inputs_sha256'].values()))
    require(input_hash in (hashlib.sha256(raw_fixture).hexdigest(),
            hashlib.sha256(raw_fixture.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()),
            'Saved synthetic fixture bytes differ beyond LF/CRLF.')

    rows_by_candidate, candidate_results, all_rows = {}, [], []
    _, states, _ = finite_volume_1d_fitted.grid(spec['settings']['cells'])
    selected = manifest['candidates'][:8]
    names = {run+'candidates/'+c['id']+'.json.gz' for c in selected}
    require({p for p in files if p.startswith(run+'candidates/')} == names,
            'Candidate checkpoint set differs.')
    for candidate in selected:
        saved = json.loads(gzip.decompress(files[run+'candidates/'+candidate['id']+'.json.gz']))
        payload = validate_checkpoint(saved, spec, candidate)
        rows = payload['rows']
        require(len(rows) == 8 and payload['status'] == 'prefix_complete', 'Pilot prefix is incomplete.')
        require(sum(r['observed'] for r in rows) == 7 and not rows[-1]['observed'],
                'Missing-hour mask differs.')
        require(rows[-1]['log_evidence'] is None and rows[-1]['contours'] == [],
                'Missing observation added a likelihood.')
        for row in rows:
            require(math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds'] >= 0.,
                    'Invalid saved time.')
            for key, tolerance in (('maximum_relative_conditional_guard', 1e-8),
                                   ('maximum_overlap_log_difference', 1e-7),
                                   ('maximum_abs_mass_error', 1e-10)):
                value = row[key]
                require(math.isfinite(value) and 0. <= value <= tolerance, 'Saved numerical guard failed.')
            if row['observed']:
                require(bool(row['contours']) and row['contours'][-1]['unresolved_cells'] == 0,
                        'Terminal contour bank left unresolved cells.')
                require([c['center'] for c in row['contours']] == spec['settings']['centers'][:len(row['contours'])],
                        'Contour order differs.')
        logs = np.array([-np.inf if v is None else v for v in payload['log_mass']])
        mean = float(np.exp(logs) @ states)
        require(abs(mean-rows[-1]['state_mean']) <= 1e-12, 'Saved final state mean differs.')
        require(not files[run+'logs/'+candidate['id']+'.log'], 'Worker log contains output needing review.')
        rows_by_candidate[candidate['id']] = payload
        all_rows.extend(rows)
        candidate_results.append(dict(candidate=candidate['id'], rows=len(rows),
            observed_rows=sum(r['observed'] for r in rows), log_likelihood=payload['log_likelihood'],
            replay_seconds=sum(r['elapsed_seconds'] for r in rows),
            minimum_finite_log_mass=float(logs[np.isfinite(logs)].min()),
            null_log_cells=sum(v is None for v in payload['log_mass']),
            log_normalization_error=abs(float(logsumexp(logs))),
            final_state_mean_error=abs(mean-rows[-1]['state_mean'])))

    require(ledger['schema'] == 'f4-h2-campaign-budget-v1' and
            (ledger['ceiling_seconds'], ledger['target_seconds'], ledger['validation_reserve_seconds'])
            == (36000., 21600., 7200.), 'Campaign budget differs.')
    require(all(e['status'] == 'closed' for e in ledger['entries']), 'Campaign has an open reservation.')
    require(ledger['charged_seconds'] == PRIOR_COMPUTE_SECONDS+sum(e['charged_seconds'] for e in ledger['entries']),
            'Cumulative accounting differs.')
    before = ledger['charged_seconds']-ledger['entries'][-1]['charged_seconds']
    require(_projection(manifest, rows_by_candidate, report['elapsed_seconds'], 8, before) == report['projection'],
            'Runtime projection differs from raw rows.')

    missing = []
    def find_missing(value, path=''):
        if isinstance(value, dict):
            for key, child in value.items():
                find_missing(child, path+'.'+key if path else key)
        elif value is None or (value == [] and path.endswith('measurement_columns')):
            missing.append(path)
    find_missing(config)
    observed = [r for r in all_rows if r['observed']]
    return dict(schema='f4-h2-pc-pilot-raw-audit-v1',
        archive_sha256=hashlib.sha256(raw_archive).hexdigest(), archive_bytes=len(raw_archive),
        archive_members_sha256={p: hashlib.sha256(raw).hexdigest() for p, raw in sorted(files.items())},
        declared_source_commit=manifest['commit'], manifest_identity=manifest['identity'],
        execution_environment=manifest['environment'], source_comparisons=source_rows,
        source_and_fixture_crlf_provenance_verified=True, original_resume_hash_policy_unchanged=True,
        runtime_pilot_raw_verified=True, checkpoint_count=len(candidate_results),
        completed_units=len(all_rows), observed_updates=len(observed), missing_predictions=len(all_rows)-len(observed),
        candidates=candidate_results,
        contour_count_distribution=dict(Counter(str(len(r['contours'])) for r in observed)),
        maximum_relative_conditional_guard=max(r['maximum_relative_conditional_guard'] for r in all_rows),
        maximum_overlap_log_difference=max(r['maximum_overlap_log_difference'] for r in all_rows),
        maximum_abs_mass_error=max(r['maximum_abs_mass_error'] for r in all_rows),
        minimum_retained_finite_log_mass=min(c['minimum_finite_log_mass'] for c in candidate_results),
        campaign_charged_seconds=ledger['charged_seconds'], pilot_wall_seconds=report['elapsed_seconds'],
        runtime_projection_raw_verified=True, projection_scope_replay_units=24,
        historical_contract_unresolved_fields=missing,
        historical_rate_measured=False, numerical_fit_readiness=False, scientific_acceptance=False,
        optimizer_converged=False, trained_model_exported=False,
        interpretation='Eight synthetic prefix checkpoints and saved guards verified. No solver replay, error-refinement certificate, historical fit, family selection or H2/H3 acceptance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.archive.resolve() != args.output.resolve(), 'Never overwrite the original archive.')
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'Raw pilot audit saved: {args.output}; no likelihood replay or campaign charge.')
