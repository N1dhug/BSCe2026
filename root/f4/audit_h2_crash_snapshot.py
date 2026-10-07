"""Inspect a restored crash snapshot without starting a model or changing a ledger.

The missing terminal report and open reservation are evidence of an interrupted
invocation. They are never replaced by a fabricated paused/successful report.
This audit checks stored bytes, states and objective arithmetic; it does not
admit numerical continuation in a different or incomplete runtime environment.
"""
from time import perf_counter
BEGUN = perf_counter()
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '1'


def audit(directory, campaign):
    import numpy as np
    from src.exe.run_model import bm_1_h2_compact as c

    directory, campaign = Path(directory), Path(campaign)
    restored = c.load(directory/'restoration.json')
    c.require(restored['model_started'] is False and (directory/'STOP').exists(),
              'Inspect a paused restoration, never an active run.')
    manifest = c.load(directory/'manifest.json')
    outer = c.load(directory/'budgeted_execution.json')
    for record in (manifest, outer):
        c.require(record['identity'] == c.digest({k:v for k,v in record.items() if k != 'identity'}),
                  'Manifest identity changed.')
    c.require(outer['numerical_manifest_identity'] == manifest['identity'], 'Mixed execution identity.')
    for name, sources in (('source.zip', manifest['source_sha256']),
                          ('budgeted_source.zip', outer['source_sha256'])):
        with zipfile.ZipFile(directory/name) as archive:
            c.require(set(archive.namelist()) == set(sources), 'Source archive membership changed.')
            for path, expected in sources.items():
                c.require(c.sha(ROOT/path) == expected and
                          hashlib.sha256(archive.read(path)).hexdigest() == expected,
                          'Frozen source changed: '+path)
    versions = {}
    for name in ('numpy', 'scipy', 'pandas', 'numba', 'llvmlite'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    c.require(versions['numpy'] == manifest['environment']['numpy'], 'Different array audit runtime.')
    data = c.receipt()
    for name, value in (('observations.npz', data['observations']), ('inputs.npz', data['inputs'])):
        c.require(c.fingerprint(c._read_array_checkpoint((directory/name).read_bytes())) == c.fingerprint(value),
                  'Frozen inputs differ.')
    c.require(c.load(directory/'packet.json') == data['packet'], 'Frozen packet differs.')
    jobs, hashes = {}, {}
    guards = dict(mass=0., conditional=0., overlap=0.)
    for path in sorted((directory/'jobs').glob('*.npz')):
        raw = c._read_array_checkpoint(path.read_bytes()); token = raw['token']
        c.require(path.name == c.digest(token)+'.npz' and token['identity'] == manifest['identity'],
                  'Job identity changed.')
        c.require(raw['checksum'] == c.fingerprint({k:v for k,v in raw.items() if k != 'checksum'}),
                  'Job payload changed.')
        c.require(token['branch'] == 'prefinal' and token['grid'] in ('base', 'fine'),
                  'Unexpected job outside this prefinal crash snapshot.')
        if token['kind'] == 'observed':
            hour = token['hour']-24
            c.require(0 <= hour < 72 and data['observations']['observed'][hour]
                      and token['value'] == float(data['observations']['values'][hour]),
                      'Observation identity changed.')
            c.same_state(raw['result']['log_state'], raw['result']['log_posterior_mass'])
        else: c.require(token['value'] is None, 'Unexpected observation on a prediction job.')
        key = (token['candidate'], token['grid'], token['branch'], token['hour'], token['kind'])
        c.require(key not in jobs and np.isfinite(raw['elapsed_seconds']) and raw['elapsed_seconds'] > 0.,
                  'Duplicate job or invalid recorded time.')
        c.state_check(raw['prior_log_state']); c.state_check(raw['result']['log_state'])
        for field, metric in (('maximum_abs_mass_error', 'mass'),
                              ('maximum_relative_conditional_guard', 'conditional'),
                              ('maximum_overlap_log_difference', 'overlap')):
            if field in raw['result']: guards[metric] = max(guards[metric], raw['result'][field])
        jobs[key] = raw; hashes[path.name] = c.sha(path)
    c.require(len(jobs) == restored['jobs'], 'Restoration job count changed.')
    c.require(guards['mass'] <= 1e-10 and guards['conditional'] <= 1e-8 and guards['overlap'] <= 1e-7,
              'Stored numerical guard failed.')
    registry = c.load(directory/'candidate_registry.json')
    c.require(registry['identity'] == manifest['identity'], 'Registry identity changed.')
    calls, objectives = {}, {}
    for family in ('2', '4'):
        path = directory/f'optimizer_{family}_calls.json'
        trace = c.load(path) if path.exists() else {'identity':manifest['identity'], 'calls':[]}
        c.require(trace['identity'] == manifest['identity'], 'Trace identity changed.')
        calls[family] = trace['calls']
        for row in trace['calls']:
            candidate = dict(pieces=int(family), logits=row['logits'])
            c.require(c.curve_key(candidate) == row['key'] and c.penalty(candidate) == row['penalty']
                      and row['objective'] == -row['score']+row['penalty'], 'Objective algebra changed.')
            objectives[row['key']] = row
        keys = registry['families'][family]
        c.require(len(keys) == len(set(keys)) <= {'2':16, '4':32}[family], 'Candidate cap changed.')
    streams, sums = {}, {}
    for key in set(objectives) | set().union(*map(set, registry['families'].values())):
        previous = None; score = 0.; completed = 0
        for hour in range(72):
            seen = hour >= 24 and bool(data['observations']['observed'][hour-24])
            kind = 'observed' if seen else 'prediction'
            raw = jobs.get((key, 'base', 'prefinal', hour, kind))
            if raw is not None: prior, result = raw['prior_log_state'], raw['result']
            elif key == 'linear' and hour < 27:
                raw = data['jobs'][('base', hour, kind)]; prior = raw['prior_log_state']; result = dict(raw)
                result['log_state'] = raw['posterior_log_state'] if seen else raw['log_state']
            else: break
            if previous is not None: c.same_state(prior, previous)
            previous = result['log_state']; completed += 1
            if seen: score += result['log_evidence']
        streams[key] = completed
        if key in objectives:
            c.require(completed == 72 and score == objectives[key]['score'], 'Incomplete or altered TRAIN score.')
            sums[key] = score
    saved_ledger = c.validate_ledger(c.load(directory/'campaign_at_checkpoint.json'))
    current_ledger = c.validate_ledger(c.load(campaign))
    c.require(current_ledger['entries'][:len(saved_ledger['entries'])] == saved_ledger['entries'],
              'Authoritative campaign differs from saved history.')
    open_entries = [e for e in current_ledger['entries'] if e['status'] == 'reserved']
    remaining = current_ledger['ceiling_seconds']-current_ledger['charged_seconds']
    reserve = c.load(directory/'budgeted_admission_001.json')['remaining_validation_seconds']
    best = min(objectives.values(), key=lambda r:r['objective'])
    return dict(schema='f4-h2-crash-snapshot-audit-v1', status='passed',
        scope='Stored bytes, states, TRAIN arithmetic and recovery budget only; no continuation admission.',
        numerical_manifest_identity=manifest['identity'], checkpoints=restored['checkpoints'],
        source_files_verified=len(manifest['source_sha256']), raw_jobs=len(jobs),
        completed_objective_calls={family:len(rows) for family,rows in calls.items()},
        pending_stream_hours={key:n for key,n in streams.items() if key not in objectives},
        completed_train_scores=sums, best_train_only=best, stored_guards=guards,
        terminal_report_present=(directory/'budgeted_report.json').exists(),
        optimizer_results_present={family:(directory/f'optimizer_{family}_result.json').exists() for family in calls},
        failed_job_files=[p.name for p in (directory/'jobs').glob('*.failed.json')],
        frozen_selection_present=(directory/'selection.json').exists(),
        frozen_forecast_present=(directory/'forecast_frozen.npz').exists(),
        final_observations_present=(directory/'final_observations.npz').exists(),
        campaign_sha256=c.sha(campaign), charged_seconds=current_ledger['charged_seconds'],
        open_reservations=open_entries, remaining_seconds=remaining,
        remaining_validation_seconds=reserve, additional_search_admitted=False,
        review_environment=versions, frozen_environment=manifest['environment'],
        missing_runtime_dependencies=[name for name,version in versions.items() if version is None],
        native_propagations=0, native_environment_admitted=False, automatic_refund=False,
        scientific_acceptance=False, blind_h3_acceptance=False, source_job_sha256=hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve previous recovery evidence.')
    result = audit(args.directory, args.campaign)
    result['read_only_review_seconds'] = perf_counter()-BEGUN
    result['review_charge_pending_reconciliation'] = True
    result['auditor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    from src.tools.bm_1_reports import _atomic_report
    _atomic_report(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'source_job_sha256'}), flush=True)
