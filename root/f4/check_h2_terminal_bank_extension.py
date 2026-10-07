"""One bounded diagnostic of candidate 28; never resume the terminal campaign.

Append midpoint contours to the original ordered prefix. Existing numerical
functions and guards are unchanged. Save each completed contour for analysis.
The 900 s development ceiling and the existing campaign deadline are enforced.
"""
from time import perf_counter, time
BEGUN = perf_counter()
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUN = ROOT/'files/output/bm_1/runs/estimation/f4_h2_cap_extension_20261007'
CAMPAIGN = RUN.parent/'f4_h2_same_day_campaign_20261007.json'
OUT = RUN.parent/'f4_h2_bank_extension_diagnostic_20261007'
CENTERS = (.5, .25, .75, 0., 1., .125, .375, .625, .875)
PRIOR_SECONDS = 425.8919726690074

def ordered(values):
    total = 0.
    for value in values:
        total += value
    return total

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('w') as handle:
        json.dump(data, handle, separators=(',', ':'), allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    if OUT.exists():
        raise FileExistsError('Preserve diagnostic; no automatic repeat.')
    ledger = read(CAMPAIGN)
    assert ledger['charged_seconds'] == PRIOR_SECONDS+ordered(e['charged_seconds'] for e in ledger['entries'])
    native = ordered(e['charged_seconds'] for e in ledger['entries'] if e['label'].startswith('native-development|'))
    grant = min(12.2, 900.-native, ledger['ceiling_seconds']-ledger['charged_seconds'], ledger['deadline_utc_seconds']-time())
    if grant < 8.:
        raise RuntimeError('Insufficient existing allowance; no numerical attempt.')
    OUT.mkdir()
    entry = dict(label='native-development|terminal-bank-one-hour-diagnostic', status='reserved',
                 started_utc_seconds=time(), charged_seconds=grant)
    ledger['entries'].append(entry)
    ledger['charged_seconds'] = PRIOR_SECONDS+ordered(e['charged_seconds'] for e in ledger['entries'])
    write(CAMPAIGN, ledger)
    soft_end = time()+grant-(perf_counter()-BEGUN)-1.5
    watchdog = threading.Timer(max(.01, grant-(perf_counter()-BEGUN)-.15), lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    pool = None
    rows = []
    report = dict(schema='f4-h2-terminal-bank-diagnostic-v1', status='started',
                  original_run_terminal=True, candidate_number=28, hour=24, grid='base',
                  centers=list(CENTERS), maximum_seconds=grant, old_run_modified=False,
                  model_fit_restarted=False, fine_validation=False, admitted=False)
    write(OUT/'registration.json', dict(report, source_sha256=sha(__file__),
        original_failure_sha256=sha(RUN/'EXTENSION_FAILED.json'),
        prior_file='be1ef45a774a1c7999e31ec7950fb4361cfd82454ac661cc4864cca21301453b.npz'))
    try:
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
            os.environ[name] = '1'
        import numpy as np
        import multiprocessing as mp
        from src.exe.run_model import bm_1_h2_compact as c
        from src.exe.run_model.bm_1_h2_native import _worker_init
        from src.method import bm_1_interval_native as n
        from src.validation.bm_1.interval_terminal_tilt import _terminal_candidate
        from root.f4.run_h2_same_day import validate_campaign
        validate_campaign(ledger, read(ROOT/'root/f4/h2_same_day_budget_decision_20261007.json'))
        manifest = read(RUN/'manifest.json')
        assert c.source_hashes() == manifest['source_sha256']
        prior_path = RUN/'jobs/be1ef45a774a1c7999e31ec7950fb4361cfd82454ac661cc4864cca21301453b.npz'
        prior = c._read_array_checkpoint(prior_path.read_bytes())
        assert prior['checksum'] == c.fingerprint({k:v for k,v in prior.items() if k!='checksum'})
        pending = read(RUN/'extension_block12_raw_audit.json')['optimizer']['4']['pending_candidate']
        failure = read(RUN/'jobs/6910062eb2676f4470c7b543ec9e30c88528773ba149c4e4bb5503476534819d.failed.json')
        assert prior['token']['candidate'] == c.curve_key(pending) == failure['token']['candidate']
        assert prior['token']['hour'] == 23 and prior['token']['kind'] == 'prediction'
        assert failure['token']['hour'] == 24 and failure['token']['grid'] == 'base'
        packet = read(RUN/'packet.json')
        inputs = c._read_array_checkpoint((RUN/'inputs.npz').read_bytes())
        data = dict(packet=packet, inputs=inputs)
        spec = c.Engine.spec(type('DataView', (), dict(data=data))(), 'base')
        assert tuple(spec['settings']['centers']) == CENTERS[:5]
        spec['settings']['centers'] = list(CENTERS)
        logs = prior['result']['log_state']
        c.state_check(logs)
        operator = n.freeze_native_operator(spec, pending, dict(start=float(inputs['input_edges'][24]), end=float(inputs['input_edges'][25])))
        pool = mp.get_context('spawn').Pool(8, initializer=_worker_init, initargs=(soft_end+1.,))
        def mapper(jobs):
            future = pool.map_async(n.advance_frequency_block, jobs)
            while True:
                if time() >= soft_end:
                    raise TimeoutError('Existing diagnostic allowance exhausted; preserve completed contours.')
                try:
                    return future.get(timeout=.1)
                except mp.TimeoutError:
                    pass
        original_prediction = n.native_prediction
        resolved = np.zeros(len(logs), dtype=bool)
        def recording_prediction(*args, **kwargs):
            prediction = original_prediction(*args, **kwargs)
            row = _terminal_candidate(prediction, failure['token']['value'], spec['physical']['sigma_y'], spec['settings']['relative_guard'])
            resolved[:] |= row['eligible']
            index = len(rows)
            raw_path = OUT/f'contour_{index:02d}.npz'
            with raw_path.open('wb') as handle:
                np.savez_compressed(handle, cf=prediction['cf'], log_amplitude=prediction['log_amplitude'],
                    predicted_mass=prediction['predicted_mass'], density=row['density'], relative=row['relative'],
                    eligible=row['eligible'], active=row['active'], log_joint=row['log_joint'], unresolved=row['active'] & ~resolved,
                    center=prediction['center'], tilt=prediction['tilt'], span=prediction['span'], frequencies=prediction['frequencies'])
            rows.append(dict(center=prediction['center'], eligible_cells=int(row['eligible'].sum()),
                unresolved_cells=int(np.sum(row['active'] & ~resolved)), alias_bound=row['alias_bound'],
                truncation_bound=row['truncation_bound'], roundoff_allowance=row['roundoff_allowance'],
                raw_sha256=sha(raw_path)))
            write(OUT/'contours.json', rows)
            return prediction
        n.native_prediction = recording_prediction
        try:
            result = n.native_terminal_update(operator, logs, failure['token']['value'], spec['physical']['sigma_y'], spec['settings'], mapper=mapper)
        finally:
            n.native_prediction = original_prediction
        with (OUT/'result.npz').open('wb') as handle:
            np.savez_compressed(handle, log_posterior_mass=result['log_posterior_mass'], prior_log_state=logs)
        report.update(status='base_hour_completed', log_evidence=result['log_evidence'], contours=result['contours'],
            maximum_relative_conditional_guard=result['maximum_relative_conditional_guard'],
            maximum_overlap_log_difference=result['maximum_overlap_log_difference'],
            maximum_abs_mass_error=result['maximum_abs_mass_error'], numerical_sources_unchanged=c.source_hashes()==manifest['source_sha256'])
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='paused' if isinstance(exc,(TimeoutError, KeyboardInterrupt)) else 'failed',
                      reason=str(exc), exception_type=type(exc).__name__)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        report.update(completed_contours=len(rows), elapsed_seconds=perf_counter()-BEGUN)
        write(OUT/'report.json', report)
        entry.update(status='closed', charged_seconds=perf_counter()-BEGUN)
        ledger['charged_seconds'] = PRIOR_SECONDS+ordered(e['charged_seconds'] for e in ledger['entries'])
        write(CAMPAIGN, ledger)
        watchdog.cancel()
    print(json.dumps(report))

if __name__ == '__main__':
    main()
