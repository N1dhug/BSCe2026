"""Approved one-hour fine control, with its own 24-hour prior; never resume fit."""
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
from root.f4.check_h2_terminal_bank_extension import read, write, sha, ordered, PRIOR_SECONDS, RUN, CAMPAIGN
BASE = RUN.parent/'f4_h2_bank_extension_diagnostic_20261007'
OUT = RUN.parent/'f4_h2_bank_fine_control_20261007'
DECISION = ROOT/'root/f4/h2_terminal_bank_fine_approval_20261007.json'

def main():
    import fcntl
    lock = (RUN.parent/'f4_h2_fine_control.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    decision = read(DECISION)
    assert decision['approved'] and decision['additional_native_seconds'] == 180
    assert sha(ROOT/'root/f4/h2_terminal_bank_fine_proposal_20261007.json') == decision['proposal_sha256']
    if OUT.exists():
        raise FileExistsError('Preserve completed or failed control; no automatic repeat.')
    ledger = read(CAMPAIGN)
    assert ledger['charged_seconds'] == PRIOR_SECONDS+ordered(e['charged_seconds'] for e in ledger['entries'])
    native = ordered(e['charged_seconds'] for e in ledger['entries'] if e['label'].startswith('native-development|'))
    grant = 180.
    assert native+grant <= 1080 and ledger['charged_seconds']+grant <= 64800
    assert time()+grant < ledger['deadline_utc_seconds'] == 1791410100
    OUT.mkdir()
    entry = dict(label='native-development|terminal-bank-approved-fine-control',status='reserved',started_utc_seconds=time(),charged_seconds=grant)
    ledger['entries'].append(entry)
    ledger['charged_seconds'] = PRIOR_SECONDS+ordered(e['charged_seconds'] for e in ledger['entries'])
    write(CAMPAIGN, ledger)
    soft_end = time()+140-(perf_counter()-BEGUN)
    watchdog = threading.Timer(max(.01,179-(perf_counter()-BEGUN)),lambda:os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    pool = None
    rows = []
    contours = []
    report = dict(schema='f4-h2-bank-fine-control-v1',status='started',candidate_number=28,
        grid='fine',observed_hours=[24],own_prior_hours=24,final_observations_decoded=0,
        production_resume_admitted=False,terminal_run_retry=False,guards_unchanged=True)
    write(OUT/'registration.json',dict(report,decision_sha256=sha(DECISION),source_sha256=sha(__file__),
        base_result_sha256=sha(BASE/'result.npz'),original_failure_sha256=sha(RUN/'EXTENSION_FAILED.json')))
    try:
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
            os.environ[name] = '1'
        import numpy as np
        import multiprocessing as mp
        from src.exe.run_model import bm_1_h2_compact as c
        from src.exe.run_model.bm_1_history import _beta_mass
        from src.exe.run_model.bm_1_h2_native import _worker_init
        from src.method import bm_1_interval_native as n
        from src.method.bm_1_interval_contour_extension import native_terminal_update_extended
        from src.validation.bm_1.interval_terminal_tilt import _terminal_candidate
        from root.f4.run_h2_same_day import validate_campaign
        validate_campaign(ledger,read(ROOT/'root/f4/h2_same_day_budget_decision_20261007.json'))
        manifest = read(RUN/'manifest.json')
        assert c.source_hashes() == manifest['source_sha256']
        candidate = read(RUN/'extension_block12_raw_audit.json')['optimizer']['4']['pending_candidate']
        assert c.curve_key(candidate) == decision['candidate_key']
        failure = read(RUN/'jobs/6910062eb2676f4470c7b543ec9e30c88528773ba149c4e4bb5503476534819d.failed.json')
        assert failure['token']['candidate'] == c.curve_key(candidate) and failure['token']['hour'] == 24
        packet = read(RUN/'packet.json')
        inputs = c._read_array_checkpoint((RUN/'inputs.npz').read_bytes())
        spec = c.Engine.spec(type('DataView',(),dict(data=dict(packet=packet,inputs=inputs)))(),'fine')
        assert spec['settings']['cells'] == 2561 and spec['settings']['max_step'] == 1/65536
        mass = _beta_mass(np.linspace(0.,1.,2562),alpha=packet['prior']['alpha'],beta=packet['prior']['beta'])
        with np.errstate(divide='ignore'):
            logs = np.log(mass)
        c.state_check(logs)
        for hour in range(24):
            if time() >= soft_end:
                raise TimeoutError('Fine control compute budget exhausted.')
            operator = n.freeze_native_operator(spec,candidate,dict(start=float(inputs['input_edges'][hour]),end=float(inputs['input_edges'][hour+1])))
            result = n.advance_frequency_block((operator,logs,np.array([0],dtype=int),2.,0.,0.))
            c.state_check(result['log_amplitude'])
            assert result['maximum_abs_mass_error'] <= 1e-10
            path = OUT/f'prior_{hour:02d}.npz'
            np.savez_compressed(path,prior_log_state=logs,log_state=result['log_amplitude'],predicted_mass=result['predicted_mass'])
            rows.append(dict(hour=hour,sha256=sha(path),maximum_abs_mass_error=result['maximum_abs_mass_error']))
            write(OUT/'prior_chain.json',rows)
            logs = result['log_amplitude']
        print(json.dumps(dict(status='own_fine_prior_completed',hours=len(rows),elapsed=perf_counter()-BEGUN)),flush=True)
        operator = n.freeze_native_operator(spec,candidate,dict(start=float(inputs['input_edges'][24]),end=float(inputs['input_edges'][25])))
        pool = mp.get_context('spawn').Pool(8,initializer=_worker_init,initargs=(soft_end+1.,))
        def mapper(jobs):
            future = pool.map_async(n.advance_frequency_block,jobs)
            while True:
                if time() >= soft_end:
                    raise TimeoutError('Fine control compute budget exhausted.')
                try:
                    return future.get(timeout=.1)
                except mp.TimeoutError:
                    pass
        original = n.native_prediction
        resolved = np.zeros(len(logs),dtype=bool)
        def recording(*args,**kwargs):
            prediction = original(*args,**kwargs)
            row = _terminal_candidate(prediction,failure['token']['value'],spec['physical']['sigma_y'],spec['settings']['relative_guard'])
            resolved[:] |= row['eligible']
            path = OUT/f'contour_{len(contours):02d}.npz'
            np.savez_compressed(path,cf=prediction['cf'],log_amplitude=prediction['log_amplitude'],predicted_mass=prediction['predicted_mass'],
                density=row['density'],relative=row['relative'],eligible=row['eligible'],active=row['active'],log_joint=row['log_joint'],
                unresolved=row['active'] & ~resolved,center=prediction['center'],tilt=prediction['tilt'],span=prediction['span'],frequencies=prediction['frequencies'])
            contours.append(dict(center=prediction['center'],sha256=sha(path),unresolved_cells=int(np.sum(row['active'] & ~resolved))))
            write(OUT/'contours.json',contours)
            return prediction
        n.native_prediction = recording
        try:
            result = native_terminal_update_extended(operator,logs,failure['token']['value'],spec['physical']['sigma_y'],spec['settings'],mapper=mapper)
        finally:
            n.native_prediction = original
        c.state_check(result['log_posterior_mass'])
        np.savez_compressed(OUT/'result.npz',prior_log_state=logs,log_posterior_mass=result['log_posterior_mass'])
        base = np.load(BASE/'result.npz')
        br = read(BASE/'report.json')
        b = dict(hour=24,log_state=base['log_posterior_mass'],observed=True,log_evidence=br['log_evidence'],moments=None)
        f = dict(hour=24,log_state=result['log_posterior_mass'],observed=True,log_evidence=result['log_evidence'],moments=None)
        comparison = c.compare_streams([b],[f],dict(hour24=(24,25)))
        report.update(status='passed',comparison=comparison,log_evidence=result['log_evidence'],contours=result['contours'],
            maximum_relative_conditional_guard=result['maximum_relative_conditional_guard'],maximum_overlap_log_difference=result['maximum_overlap_log_difference'],
            maximum_abs_mass_error=result['maximum_abs_mass_error'],numerical_sources_unchanged=c.source_hashes()==manifest['source_sha256'])
    except (Exception,KeyboardInterrupt) as exc:
        report.update(status='paused' if isinstance(exc,(TimeoutError,KeyboardInterrupt)) else 'failed',reason=str(exc),exception_type=type(exc).__name__)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        report.update(completed_prior_hours=len(rows),completed_contours=len(contours),elapsed_seconds=perf_counter()-BEGUN)
        write(OUT/'report.json',report)
        entry.update(status='closed',charged_seconds=perf_counter()-BEGUN)
        ledger['charged_seconds'] = PRIOR_SECONDS+ordered(e['charged_seconds'] for e in ledger['entries'])
        write(CAMPAIGN,ledger)
        watchdog.cancel()
    print(json.dumps(report),flush=True)

if __name__ == '__main__':
    main()
