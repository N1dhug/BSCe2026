"""F-only staged conditional H2 study; native kernels and legacy fits are unchanged.

See root/H2_COMPACT_STAGED_RUN.md. A prepared run is not admitted or fitted.
Every numerical job has an immutable checkpoint, including its full input log
state. Optimizer restarts replay cached calls, never completed model hours.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import threading
from time import perf_counter, time
import zipfile

import numpy as np
from scipy.special import logsumexp

from src.method.bm_1_interval_precise import digest
from src.tools.bm_1_reports import _atomic_report, _read_array_checkpoint, _RunLock

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'files/output/bm_1/runs/estimation'
INDEX = ROOT/'root/f4/h2_compact_rate_reviewed_20261007.json'
SEED = OUT/'f4_h2_native_budget_seed_20261006.json'
PRIOR_SECONDS = 425.8919726690074
LINEAR = dict(pieces=2, logits=[0., 0.])
PROBE = dict(pieces=4, logits=[.5, 0., 0., 0., 0., 0.])
HORIZONS = (1, 6, 12, 24, 48, 72, 168)
OWN_SOURCE = ('src/exe/run_model/bm_1_h2_compact.py',
    'root/f4/run_h2_compact.py', 'root/f4/run_h2_compact.ps1',
    'root/f4/check_h2_compact_runner.py', 'root/H2_COMPACT_STAGED_RUN.md')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, *, replace=False):
    _atomic_report(Path(path), value, replace_existing=replace)
    with Path(path).open('rb') as handle:
        os.fsync(handle.fileno())
    if os.name != 'nt':
        fd = os.open(str(Path(path).parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def fingerprint(value):
    """Content hash for safe real-array checkpoint payloads, including log tails."""
    def canonical(v):
        if isinstance(v,np.ndarray):
            return dict(dtype=v.dtype.str, shape=list(v.shape),
                sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest())
        if isinstance(v,dict):
            return {k:canonical(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):
            return [canonical(x) for x in v]
        if isinstance(v,np.generic):
            return v.item()
        return v
    return digest(canonical(value))


def ordered_sum(values):
    # Preserve the PC Python 3.11 ledger arithmetic on Python 3.12+ too.
    result = 0.
    for value in values:
        result += value
    return result


def validate_ledger(ledger):
    require(ledger['schema'] == 'f4-h2-campaign-budget-v1'
        and ledger['ceiling_seconds'] == 36000. and ledger['target_seconds'] == 21600.
        and math.isfinite(ledger['validation_reserve_seconds'])
        and ledger['validation_reserve_seconds'] >= 7200., 'Campaign contract changed.')
    entries = ledger['entries']
    require(all(e['status'] in ('closed', 'reserved') and math.isfinite(e['charged_seconds'])
        and e['charged_seconds'] >= 0. for e in entries), 'Invalid campaign entries.')
    # Historic chat receipts used their own interpreter's sum implementation.
    amounts = [e['charged_seconds'] for e in entries]
    require(ledger['charged_seconds'] in (PRIOR_SECONDS+ordered_sum(amounts),
        PRIOR_SECONDS+sum(amounts)), 'Campaign total does not match its entries.')
    return ledger


def reconcile_ledger(path):
    seed = validate_ledger(load(SEED))
    current = validate_ledger(load(path)) if Path(path).exists() else seed
    a, b = current['entries'], seed['entries']
    require(a[:min(len(a), len(b))] == b[:min(len(a), len(b))],
        'PC/chat ledger diverged; no budget reset or automatic reconciliation.')
    chosen = seed if len(b) > len(a) else current
    chosen['validation_reserve_seconds'] = max(current['validation_reserve_seconds'], seed['validation_reserve_seconds'])
    return chosen


@contextmanager
def reservation(path, requested, label, *, native=False, begun=None):
    """Reserve the whole invocation; a crash keeps its full charge."""
    path = Path(path); begun = perf_counter() if begun is None else begun
    require(math.isfinite(requested) and 75. < requested <= 36000., 'Invalid wall-clock cap.')
    with _RunLock(path.with_suffix('.lock')):
        ledger = reconcile_ledger(path)
        before = ledger['charged_seconds']
        native_used = ordered_sum(e['charged_seconds'] for e in ledger['entries']
            if e['label'].startswith('native-development|'))
        grant = min(requested, 36000.-before)
        if native:
            grant = min(grant, 900.-native_used,
                36000.-ledger['validation_reserve_seconds']-before)
        require(grant > 75., 'Insufficient remaining budget including 60 s stop reserve.')
        entry = dict(label=('native-development|' if native else 'compact-production|')+label,
            started_utc_seconds=time(), status='reserved', charged_seconds=grant)
        ledger['entries'].append(entry)
        ledger['charged_seconds'] = PRIOR_SECONDS+ordered_sum(e['charged_seconds'] for e in ledger['entries'])
        write(path, ledger, replace=True)
        startup = perf_counter()-begun
        watchdog = threading.Timer(max(.01, grant-startup-1.), lambda: os._exit(124))
        watchdog.daemon = True; watchdog.start()
        try:
            yield dict(ledger=ledger, before=before, grant=grant, begun=begun,
                deadline=time()+max(0., grant-startup-60.), native_before=native_used)
        finally:
            entry.update(status='closed', charged_seconds=perf_counter()-begun)
            ledger['charged_seconds'] = PRIOR_SECONDS+ordered_sum(e['charged_seconds'] for e in ledger['entries'])
            write(path, ledger, replace=True)
            watchdog.cancel()


def source_hashes():
    from root.f4.measure_h2_compact_rates import EXTRA_SOURCE
    from src.exe.run_model.bm_1_h2_native import _source
    return dict(_source(), **{p: sha(ROOT/p) for p in (*EXTRA_SOURCE, *OWN_SOURCE)})


def environment():
    from src.exe.run_model.bm_1_h2_native import _environment
    env = _environment()
    require(env['scipy'] == '1.17.0', 'The declared optimizer requires scipy==1.17.0.')
    return env


def validate_observations(observations, packet, *, final=False):
    split = packet['policy']['split']
    count = 168 if final else 72
    origin = datetime.fromisoformat(split['holdout_start' if final else 'train_start'])
    require(observations['times'] == [(origin+timedelta(hours=i)).isoformat() for i in range(count)],
        'Observation window/order/timezone differs.')
    seen = observations['observed']; values = observations['values']; group = observations['group_kW']
    require(all(isinstance(v, np.ndarray) and v.shape == (count,) for v in (seen, values, group))
        and seen.dtype.kind == 'b' and not np.isinf(values).any() and not np.isinf(group).any(),
        'Wrong observation shape/type or infinite reading.')
    require(np.array_equal(np.isfinite(values), seen) and np.array_equal(np.isfinite(group), seen)
        and np.array_equal(values[seen], group[seen]/packet['frozen_scales']['power_kW']),
        'Mask/group normalization changed; no clipping or imputation.')
    if not final:
        require(observations['final_values_decoded'] == observations['prior_values_decoded']
            == observations['imputed_values'] == 0, 'Unexpected exposed/imputed observations.')
        require(observations['train_complete'] == int(seen[:48].sum())
            and observations['selection_complete'] == int(seen[48:].sum()), 'Wrong observed counts.')


def receipt():
    """Reuse already reviewed bytes. This is not another native/source audit."""
    from root.f4.measure_h2_compact_rates import frozen_packet
    index = load(INDEX)
    require(sha(ROOT/index['review_path']) == index['review_sha256']
        and sha(ROOT/index['audit_path']) == index['audit_sha256'], 'Reviewed receipt/audit bytes changed.')
    audit = load(ROOT/index['audit_path'])
    require(audit['passed'] is True, 'Prior raw audit did not pass.')
    ready, packet, inputs = frozen_packet()
    with zipfile.ZipFile(ROOT/index['review_path']) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        raw = archive.read('observations.npz')
        require(hashlib.sha256(raw).hexdigest() == manifest['observation_sha256'], 'Receipt observations changed.')
        observations = _read_array_checkpoint(raw)
        require(manifest['packet_identity'] == packet['identity'] == index['packet_identity'], 'Different packet.')
        validate_observations(observations, packet)
        # Only numerical dependencies must match the old source; the new runner
        # and additive Git attributes are deliberately different.
        for p, expected in manifest['source_sha256'].items():
            if p.startswith(('src/method/', 'src/model/', 'src/solver/', 'src/params/', 'src/validation/')):
                current = (ROOT/p).read_bytes()
                require(expected in (hashlib.sha256(current).hexdigest(),
                    hashlib.sha256(current.replace(b'\r\n', b'\n')).hexdigest(),
                    hashlib.sha256(current.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()),
                    'Numerical dependency differs from reviewed pilot: '+p)
        jobs = {}
        for row in json.loads(archive.read('timings.json')):
            raw = archive.read(row['checkpoint'].replace('\\', '/'))
            require(hashlib.sha256(raw).hexdigest() == row['checkpoint_sha256'], 'Receipt checkpoint changed.')
            payload = _read_array_checkpoint(raw)
            jobs[(row['grid'], row['hour'], row['rate_type'])] = payload
        rates = json.loads(archive.read('report.json'))['rates']
    return dict(index=index, packet=packet, inputs=inputs, observations=observations,
        jobs=jobs, rates=rates,
        observation_sha256=manifest['observation_sha256'])


def curve(candidate):
    from src.method.bm_1_response import curve_pair_from_logits
    logits = np.asarray(candidate['logits'], dtype=float)
    require(candidate['pieces'] in (2, 4) and logits.shape == (2*(candidate['pieces']-1),)
        and np.isfinite(logits).all() and np.max(np.abs(logits)) <= 4., 'Invalid curve candidate.')
    return curve_pair_from_logits(logits, candidate['pieces'])


def curve_key(candidate):
    # Only exactly linear logits are collapsed across different knot grids.
    # Other distinct representations retain their own arithmetic trajectory.
    curve(candidate)
    return 'linear' if all(x == 0. for x in candidate['logits']) else digest(candidate)


def canonical_candidate(candidate):
    return dict(LINEAR) if curve_key(candidate) == 'linear' else candidate


def penalty(candidate):
    f, g = curve(candidate)
    return float(2.*sum(np.sum((c.values[1:-1]-(1.-2.*c.knots[1:-1]))**2) for c in (f, g)))


def state_check(logs):
    require(isinstance(logs, np.ndarray) and logs.ndim == 1 and not np.isnan(logs).any()
        and not np.isposinf(logs).any() and abs(float(logsumexp(logs))) <= 1e-10,
        'Invalid full normalized log-state.')


def same_state(a, b):
    require(np.array_equal(a, b), 'Checkpoint belongs to a different chronological prior.')


def decode_distribution(value):
    from src.method.bm_1_interval_forecast import IntervalCDFPlan, IntervalDistribution
    return IntervalDistribution(IntervalCDFPlan(**value['plan']),
        value['characteristic_real']+1j*value['characteristic_imag'])


class JobStore:
    """One atomic file per completed job; no vulnerable second index commit."""
    def __init__(self, directory, identity):
        self.directory, self.identity = Path(directory), identity
        self.completed = {}; self.new_seconds = {}; self.events = []

    def run(self, meta, prior, compute):
        token = dict(identity=self.identity, **meta)
        key = digest(token); path = self.directory/'jobs'/f'{key}.npz'
        failure = path.with_suffix('.failed.json')
        require(not failure.exists(), 'A numerical job failed previously; no automatic retry: '+key)
        state_check(prior)
        if path.exists():
            saved = _read_array_checkpoint(path.read_bytes())
            require(saved['checksum'] == fingerprint({k:v for k,v in saved.items() if k != 'checksum'}),
                'Job checkpoint content checksum failed.')
            require(saved['token'] == token, 'Job identity changed.')
            same_state(saved['prior_log_state'], prior)
            result = saved['result']; state_check(result['log_state'])
            self.completed[key] = saved
            return result
        begun = perf_counter()
        try:
            result = compute(); state_check(result['log_state'])
        except (TimeoutError, KeyboardInterrupt):
            raise
        except Exception as exc:
            write(failure, dict(token=token, error=str(exc), diagnostics=getattr(exc, 'diagnostics', {})))
            raise
        saved = dict(token=token, prior_log_state=prior, result=result, elapsed_seconds=perf_counter()-begun)
        saved['checksum'] = fingerprint(saved)
        write(path, saved)
        self.completed[key] = saved
        self.new_seconds.setdefault((meta['grid'], meta['kind']), []).append(saved['elapsed_seconds'])
        self.events.append(dict(key=key, **meta, seconds=saved['elapsed_seconds']))
        return result


class Engine:
    def __init__(self, directory, data, manifest, allocation, *, imported=True):
        self.directory, self.data, self.manifest, self.allocation = Path(directory), data, manifest, allocation
        self.store = JobStore(directory, manifest['identity'])
        self.imported = imported; self.pool = None
        self.validation_remaining = data['index']['protected_validation_seconds']
        self.validation_completed = set()
        self.source = manifest['source_sha256']
        self.rates = {g: dict(r) for g, r in data['rates'].items()}
        self.active_search = False

    def check(self, *, search=None):
        require(not (self.directory/'FAILED.json').exists(), 'This run has a preserved failure.')
        if time() >= self.allocation['deadline'] or (self.directory/'STOP').exists():
            raise TimeoutError('Paused before next native job; keep checkpoints and ledger.')
        if self.active_search if search is None else search:
            used = self.allocation['before']+perf_counter()-self.allocation['begun']
            next_job_allowance = 1.5*max(self.rates['base'].values())
            if used+self.validation_remaining+60.+next_job_allowance >= 36000.:
                raise TimeoutError('Search stopped to protect the remaining validation work.')

    def mapper(self, jobs):
        from src.exe.run_model.bm_1_h2_native import _worker_init
        from src.method.bm_1_interval_native import advance_frequency_block
        self.check()
        if self.pool is None:
            self.pool = mp.get_context('spawn').Pool(8, initializer=_worker_init,
                initargs=(self.allocation['deadline'],))
        pending = self.pool.map_async(advance_frequency_block, jobs)
        while True:
            self.check()
            try:
                return pending.get(timeout=.25)
            except mp.TimeoutError:
                pass

    def close(self):
        if self.pool is not None:
            self.pool.terminate(); self.pool.join(); self.pool = None

    def spec(self, grid):
        p = self.data['packet']; n = p['numerics']; inputs = self.data['inputs']
        return dict(settings=dict(cells=n[grid+'_cells'], max_step=n[grid+'_max_step'],
            frequencies=n['frequencies'], span=n['span'], centers=n['centers'],
            relative_guard=n['guards']['conditional'], overlap_log_tolerance=n['guards']['overlap']),
            physical=p['physical'], input_edges=inputs['input_edges'].tolist(),
            baseline=inputs['baseline'].tolist(), price=inputs['price'].tolist(),
            horizon=[0., 11.], time_unit='day', midnight_resets=0)

    def initial(self, grid, candidate):
        if curve_key(candidate) == 'linear' and self.imported:
            return self.data['jobs'][(grid, 0, 'prediction')]['prior_log_state'].copy()
        from src.exe.run_model.bm_1_history import _beta_mass
        spec = self.spec(grid); prior = self.data['packet']['prior']
        mass = _beta_mass(np.linspace(0., 1., spec['settings']['cells']+1), prior['alpha'], prior['beta'])
        with np.errstate(divide='ignore'):
            logs = np.log(mass)
        state_check(logs)
        return logs

    def job(self, candidate, grid, hour, kind, prior, *, value=None, branch='prefinal', search=False):
        from src.method.bm_1_interval_native import freeze_native_operator, advance_frequency_block, native_terminal_update
        from src.method.bm_1_interval_forecast import native_interval_moments, native_interval_distribution, interval_cdf_plan, interval_forecast_checkpoint
        candidate = canonical_candidate(candidate)
        meta = dict(candidate=curve_key(candidate), grid=grid, hour=hour, kind=kind,
            branch=branch, value=None if value is None else float(value))
        # A raw pilot import is permitted only for this exact physical curve,
        # prefinal input and prior. Its original receipt remains authoritative.
        if self.imported and meta['candidate'] == 'linear' and branch == 'prefinal' and (grid, hour, kind) in self.data['jobs']:
            raw = self.data['jobs'][(grid, hour, kind)]
            same_state(prior, raw['prior_log_state'])
            if kind == 'cdf':
                result = raw['forecast']
            else:
                result = dict(raw)
                result['log_state'] = raw['posterior_log_state'] if kind == 'observed' else raw['log_state']
            state_check(result['log_state'])
            return result
        def compute():
            self.active_search = search
            self.check(search=search)
            spec = self.spec(grid); edges = self.data['inputs']['input_edges']
            operator = freeze_native_operator(spec, candidate, dict(start=float(edges[hour]), end=float(edges[hour+1])))
            if kind == 'observed':
                result = native_terminal_update(operator, prior, float(value), spec['physical']['sigma_y'], spec['settings'], mapper=self.mapper)
                result['log_state'] = result['log_posterior_mass']
            elif kind == 'prediction':
                result = advance_frequency_block((operator, prior, np.array([0], dtype=int), 2., 0., 0.))
                result = dict(log_state=result['log_amplitude'], maximum_abs_mass_error=result['maximum_abs_mass_error'])
            elif kind == 'moments':
                result = native_interval_moments(operator, prior, spec['physical']['sigma_y'])
            elif kind == 'cdf':
                result = native_interval_distribution(operator, prior, interval_cdf_plan(spec['physical']['sigma_y']), mapper=self.mapper)
                result = interval_forecast_checkpoint(dict(hour=hour), result['log_state'], result, self.manifest['identity'])
            else:
                raise ValueError('Unknown native job kind.')
            return result
        # Scalar prediction/operator work cannot poll the pool deadline. A
        # separate search watchdog protects the validation reserve even if
        # such a job stalls; abrupt termination keeps the full reservation.
        timer = None
        if search:
            used = self.allocation['before']+perf_counter()-self.allocation['begun']
            allowance = 36000.-used-self.validation_remaining-60.
            if allowance <= 0.:
                raise TimeoutError('No search budget remains outside protected validation.')
            timer = threading.Timer(allowance,lambda:os._exit(124)); timer.daemon=True; timer.start()
        try:
            return self.store.run(meta, prior, compute)
        finally:
            if timer is not None:
                timer.cancel()

    def replay(self, candidate, grid, *, end=96, start=0, initial=None,
               observations=None, branch='prefinal', moments=False, search=False):
        from src.exe.run_model.progress import runtime_text
        obs = self.data['observations'] if observations is None else observations
        logs = self.initial(grid, candidate) if initial is None else initial.copy()
        offset = 96 if branch == 'final' else 24
        rows = []; begun = perf_counter(); initial_jobs = len(self.store.events)
        for hour in range(start, end):
            index = hour-offset
            seen = 0 <= index < len(obs['values']) and bool(obs['observed'][index])
            predicted = self.job(candidate, grid, hour, 'moments', logs, branch=branch, search=search) if moments and hour >= offset else None
            result = self.job(candidate, grid, hour, 'observed' if seen else 'prediction', logs,
                value=obs['values'][index] if seen else None, branch=branch, search=search)
            logs = result['log_state']
            rows.append(dict(hour=hour, log_state=logs, observed=seen,
                log_evidence=float(result['log_evidence']) if seen else 0., moments=predicted))
            if hour == end-1 or (hour+1)%12 == 0:
                # New work only; cached work does not inflate measured throughput.
                fresh = len(self.store.events)-initial_jobs
                print(json.dumps(dict(curve=curve_key(candidate)[:12], grid=grid, hour=hour+1,
                    new_jobs=fresh, status=runtime_text(begun, fresh, fresh+(end-hour-1)*(2 if moments else 1)))) , flush=True)
        return rows


def compare_streams(base, fine, windows):
    from root.f4.measure_h2_compact_rates import compare_states
    from src.method.bm_1_interval_forecast import compare_interval_moments
    require([r['hour'] for r in base] == [r['hour'] for r in fine], 'Base/fine chronology differs.')
    state_errors = []; moment_errors = []
    for b, f in zip(base, fine):
        require(b['observed'] == f['observed'], 'Base/fine masks differ.')
        state_errors.append(compare_states(b['log_state'], f['log_state']))
        require((b['moments'] is None) == (f['moments'] is None), 'Missing paired predictive moments.')
        if b['moments'] is not None:
            moment_errors.append(compare_interval_moments(b['moments'], f['moments']))
            compare_states(b['moments']['log_state'], f['moments']['log_state'])
    scores = {}
    for name, (start, end) in windows.items():
        b = [r for r in base if start <= r['hour'] < end]
        f = [r for r in fine if start <= r['hour'] < end]
        require(len(b) == len(f) == end-start, 'Incomplete declared window: '+name)
        sb = ordered_sum(r['log_evidence'] for r in b); sf = ordered_sum(r['log_evidence'] for r in f)
        require(abs(sb-sf) <= .05, 'Total-window log-score precision failed: '+name)
        scores[name] = dict(base=sb, fine=sf, difference=abs(sb-sf), hours=end-start)
    return dict(passed=True, scores=scores,
        maximum_state_mean=max(r['state_mean_difference'] for r in state_errors),
        maximum_state_L1=max(r['conservative_state_L1_difference'] for r in state_errors),
        paired_moment_hours=len(moment_errors))


def prepare(directory, data, source_commit):
    directory = Path(directory)
    source = source_hashes(); env = environment()
    content = dict(schema='f4-h2-compact-staged-run-v1', source_commit=source_commit,
        source_sha256=source, environment=env, packet_identity=data['packet']['identity'],
        receipt_sha256=data['index']['review_sha256'], observation_sha256=data['observation_sha256'],
        protocol='root/H2_COMPACT_STAGED_RUN.md', forecast_horizons=list(HORIZONS),
        train_hours=48, selection_hours=24, final_hours=168,
        regularization=2., bounds=[-4., 4.], family_new_pair_caps={'2':16, '4':32},
        global_pair_cap=49, optimizer='scipy-1.17.0-COBYQA',
        initial_tr_radius=.5, final_tr_radius=.02, sigma_y=data['packet']['physical']['sigma_y'],
        blind_h3_acceptance=False, automatic_retry=False)
    content['identity'] = digest(content)
    path = directory/'manifest.json'
    if path.exists():
        require(load(path) == content, 'Resume requires identical source, environment, packet and protocol.')
        for name, value in (('observations.npz',data['observations']),('inputs.npz',data['inputs']),('packet.json',data['packet'])):
            target = directory/name
            old = _read_array_checkpoint(target.read_bytes()) if target.suffix == '.npz' else load(target)
            require(fingerprint(old) == fingerprint(value), 'Frozen preparation data changed.')
    else:
        require(not (directory/'FAILED.json').exists(), 'Preserve the failed run.')
        for name, value in (('observations.npz',data['observations']),('inputs.npz',data['inputs']),('packet.json',data['packet'])):
            target = directory/name
            if target.exists():
                old = _read_array_checkpoint(target.read_bytes()) if target.suffix == '.npz' else load(target)
                require(fingerprint(old) == fingerprint(value), 'Partial preparation data differs.')
            else:
                write(target,value)
        temporary = directory/'source.zip.partial'
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for p, expected in source.items():
                raw = (ROOT/p).read_bytes()
                require(hashlib.sha256(raw).hexdigest() == expected, 'Source changed while freezing.')
                archive.writestr(p, raw)
        os.replace(temporary,directory/'source.zip')
        write(path, content)  # Publish readiness only after all preparation bytes.
    return content


def probe(engine):
    """One registered nonlinear prefix, always charged as native development."""
    from src.method.bm_1_interval_forecast import compare_interval_distributions
    rows = {g: [] for g in ('base', 'fine')}; states = {g:engine.initial(g, PROBE) for g in rows}
    obs = engine.data['observations']
    require(bool(np.all(obs['observed'][:3])), 'Registered probe requires first three complete TRAIN hours.')
    for hour in range(27):
        distributions = {}
        for grid in rows:
            prior = states[grid]; seen = hour >= 24
            moments = engine.job(PROBE, grid, hour, 'moments', prior) if seen else None
            if hour in (25, 26):
                full = engine.job(PROBE, grid, hour, 'cdf', prior)
                distributions[grid] = decode_distribution(full['distribution'])
                a, b = moments['log_state'], full['log_state']; active = np.isfinite(a)
                require(np.array_equal(active, np.isfinite(b)) and np.max(np.abs(a[active]-b[active])) <= 1e-7,
                    'Nonlinear moment/CF state mismatch.')
            result = engine.job(PROBE, grid, hour, 'observed' if seen else 'prediction', prior,
                value=obs['values'][hour-24] if seen else None)
            states[grid] = result['log_state']
            rows[grid].append(dict(hour=hour, log_state=states[grid], observed=seen,
                log_evidence=float(result['log_evidence']) if seen else 0., moments=moments))
        compare_streams(rows['base'][-1:], rows['fine'][-1:], {})
        if distributions:
            compare_interval_distributions(distributions['base'], distributions['fine'])
        print(json.dumps(dict(stage='nonlinear_probe', completed_hours=hour+1, total_hours=27)), flush=True)
    comparison = compare_streams(rows['base'], rows['fine'], {'prefix':(24,27)})
    rates = {}
    for grid in rows:
        rates[grid] = {}
        for kind in ('observed','prediction','moments','cdf'):
            # Include preserved job timings at resume, excluding only the first
            # compilation-bearing job of each type (CDF needs no own JIT).
            values = [v['elapsed_seconds'] for v in engine.store.completed.values()
                if v['token']['grid'] == grid and v['token']['kind'] == kind]
            warm = values if kind == 'cdf' else values[1:]
            require(len(warm) >= 2, 'Missing warm nonlinear rate samples.')
            rates[grid][kind] = max(engine.data['rates'][grid][kind], max(warm))
    result = dict(schema='f4-h2-nonlinear-prefix-v1', identity=engine.manifest['identity'],
        candidate=PROBE, complete=True, prefix_comparison=comparison, rates=rates,
        chronological_hours=27, full_window_precision=False, uniform_nonlinear_bound=False,
        scientific_acceptance=False, blind_h3_acceptance=False,
        job_hashes={p.name:sha(p) for p in sorted((engine.directory/'jobs').glob('*.npz'))})
    write(engine.directory/'nonlinear_probe.json', result)
    return result


def load_probe(engine):
    report = load(engine.directory/'nonlinear_probe.json')
    require(report['identity'] == engine.manifest['identity'] and report['complete'] is True
        and report['candidate'] == PROBE, 'No complete nonlinear probe for this run.')
    for name, expected in report['job_hashes'].items():
        require(sha(engine.directory/'jobs'/name) == expected, 'Nonlinear raw job changed.')
    engine.rates = report['rates']
    return report


def priced_graph(engine):
    from root.f4.measure_h2_compact_rates import work_graph
    obs = engine.data['observations']
    charged = engine.allocation['before']+perf_counter()-engine.allocation['begun']
    return work_graph(engine.rates, obs['train_complete'], obs['selection_complete'], charged)


def admission_graph(engine):
    path = engine.directory/'work_graph.json'
    if path.exists():
        graph = load(path)
        require(graph['identity'] == engine.manifest['identity'] and graph['rates'] == engine.rates,
            'Frozen admission graph/rates changed.')
        # Do not add an already priced full fit again at each resume. Actual
        # elapsed time stays in the common ledger; search protects unfinished
        # validation, reconstructed from the verified completed streams below.
    else:
        graph = priced_graph(engine)
        graph.update(identity=engine.manifest['identity'], rates=engine.rates)
        require(graph['within_ten_hour_ceiling'], 'Complete measured job graph exceeds the ten-hour campaign cap.')
        write(path, graph)
    require(graph['within_ten_hour_ceiling'], 'The complete job graph was never admitted.')
    engine.validation_remaining = graph['protected_validation_seconds']
    engine.allocation['ledger']['validation_reserve_seconds'] = max(
        engine.allocation['ledger']['validation_reserve_seconds'], graph['protected_validation_seconds'])
    return graph


def release_validation(engine, key, grid, kind, count):
    # In-memory accounting is reconstructed from completed stages on resume.
    # A deduplicated physical family cannot release the same planned work twice.
    token = (key, grid, kind)
    if token not in engine.validation_completed:
        engine.validation_remaining -= 1.5*count*engine.rates[grid][kind]
        engine.validation_completed.add(token)
    require(engine.validation_remaining >= 300.+60.-1e-7, 'Released more validation work than was priced.')


def validated_family(engine, candidate):
    streams = {}
    for grid in ('base','fine'):
        train = engine.replay(candidate, grid, end=72)
        selection = engine.replay(candidate, grid, start=72, end=96,
            initial=train[-1]['log_state'], moments=True)
        streams[grid] = train+selection
    compared = compare_streams(streams['base'], streams['fine'], {'train':(24,72), 'selection':(72,96)})
    key = curve_key(candidate); obs = engine.data['observations']
    for grid in ('base','fine'):
        release_validation(engine, key+'selection', grid, 'observed', int(obs['observed'][48:].sum()))
        release_validation(engine, key+'selection', grid, 'prediction', int((~obs['observed'][48:]).sum()))
        release_validation(engine, key+'selection', grid, 'moments', 24)
    release_validation(engine, key+'train', 'fine', 'observed', int(obs['observed'][:48].sum()))
    release_validation(engine, key+'train', 'fine', 'prediction', 24+int((~obs['observed'][:48]).sum()))
    return dict(candidate=candidate, comparison=compared, streams=streams)


def optimizer_family(engine, pieces):
    """Deterministically reconstruct COBYQA state by exact objective-call replay."""
    from scipy.optimize import minimize
    require(pieces in (2,4), 'Unknown fit family.')
    calls_path = engine.directory/f'optimizer_{pieces}_calls.json'
    result_path = engine.directory/f'optimizer_{pieces}_result.json'
    saved = load(calls_path) if calls_path.exists() else dict(identity=engine.manifest['identity'], calls=[])
    require(saved['identity'] == engine.manifest['identity'], 'Optimizer trace identity changed.')
    calls = saved['calls']; cursor = 0; cap = {2:16, 4:32}[pieces]
    registry_path = engine.directory/'candidate_registry.json'
    registry = load(registry_path) if registry_path.exists() else dict(identity=engine.manifest['identity'], families={'2':[], '4':[]})
    require(registry['identity'] == engine.manifest['identity'], 'Candidate registry changed.')
    for family, keys in registry['families'].items():
        require(len(keys) == len(set(keys)) <= {'2':16, '4':32}[family], 'Candidate budget registry invalid.')
    require(1+sum(len(k) for k in registry['families'].values()) <= 49, 'Distinct pair ceiling exceeded.')
    if result_path.exists():
        result = load(result_path)
        require(result['identity'] == engine.manifest['identity'], 'Optimizer result identity changed.')
        require(result['converged'] is True, 'Saved optimizer did not converge; no automatic new fit.')
        return result

    def objective(logits):
        nonlocal cursor
        candidate = dict(pieces=pieces, logits=np.asarray(logits, dtype=float).tolist())
        key = curve_key(candidate)
        if cursor < len(calls):
            row = calls[cursor]
            require(row['logits'] == candidate['logits'] and row['key'] == key,
                'COBYQA call order changed during deterministic resume.')
            require(math.isfinite(row['score']) and row['penalty'] == penalty(candidate)
                and row['objective'] == -row['score']+row['penalty'], 'Cached objective algebra changed.')
            cursor += 1
            return row['objective']
        engine.check(search=True)
        keys = registry['families'][str(pieces)]
        if key != 'linear' and key not in keys:
            require(len(keys) < cap and 1+sum(len(k) for k in registry['families'].values()) < 49,
                'Declared distinct curve-pair budget exhausted.')
            keys.append(key)  # Reserve candidate before any partial model work.
            write(registry_path, registry, replace=registry_path.exists())
        rows = engine.replay(candidate, 'base', end=72, search=True)
        score = ordered_sum(r['log_evidence'] for r in rows)
        reg = penalty(candidate); value = -score+reg
        require(math.isfinite(value), 'Nonfinite objective; never substitute infinity.')
        calls.append(dict(logits=candidate['logits'], key=key, score=score, penalty=reg, objective=value))
        write(calls_path, saved, replace=calls_path.exists()); cursor += 1
        print(json.dumps(dict(stage='fit', pieces=pieces, objective_calls=cursor,
            new_pairs=len(keys), objective=value)), flush=True)
        return value

    result = minimize(objective, np.zeros(2*(pieces-1)), method='COBYQA',
        bounds=[(-4.,4.)]*(2*(pieces-1)), options=dict(maxfev=cap+1,
            initial_tr_radius=.5, final_tr_radius=.02, scale=False))
    require(cursor == len(calls), 'Optimizer stopped before consuming its preserved trace.')
    converged = bool(result.success) and int(result.status) == 0 and math.isfinite(float(result.fun))
    value = dict(identity=engine.manifest['identity'], pieces=pieces,
        candidate=dict(pieces=pieces, logits=np.asarray(result.x).tolist()),
        objective=float(result.fun), nfev=int(result.nfev), status=int(result.status),
        message=str(result.message), converged=converged,
        algorithmic_local_termination_only=True, scientific_acceptance=False)
    write(result_path, value)
    require(converged, f'COBYQA family {pieces} did not reach its trust-radius termination: {result.message}')
    return value


def select_families(families):
    require(set(families) == {1,2,4}, 'All three fitted families are required.')
    scores = {g:{p:families[p]['comparison']['scores']['selection'][g] for p in families} for g in ('base','fine')}
    orders = {g:sorted(families, key=lambda p:(-scores[g][p],p)) for g in scores}
    # Any reversal is treated as ambiguous. If only a score gap is ambiguous,
    # use the union of families within .1 of either grid's best.
    reversal = orders['base'] != orders['fine']
    eligible = set(families) if reversal else {p for p in families if any(
        max(scores[g].values())-scores[g][p] <= .1 for g in scores)}
    selected = min(eligible)
    ambiguous = reversal or len(eligible)>1
    return dict(selected_family=selected, scores=scores, rankings=orders,
        ambiguity=ambiguous, ranking_reversal=reversal, eligible=sorted(eligible),
        improvement_claim=not ambiguous and selected != 1,
        rule='fewer pieces among ambiguous models; no improvement claim in ambiguity')


def selection_metrics(rows, observations):
    pairs = [(r, observations['values'][r['hour']-24]) for r in rows
        if 72 <= r['hour'] < 96 and r['observed']]
    require(len(pairs) > 0 and all(r['moments'] is not None for r,y in pairs), 'Missing predictive selection moments.')
    errors = np.array([r['moments']['observation_mean']-y for r,y in pairs])
    return dict(observed_hours=len(pairs), rmse=float(np.sqrt(np.mean(errors**2))), mae=float(np.mean(np.abs(errors))))


def fit_and_select(engine):
    load_probe(engine); graph = admission_graph(engine)
    linear = validated_family(engine, LINEAR)
    write(engine.directory/'linear_full_window.json', dict(identity=engine.manifest['identity'],
        comparison=linear['comparison'], mandatory_production_baseline=True,
        reused_in_selection=True, historical_fit_ready=True, scientific_acceptance=False),
        replace=(engine.directory/'linear_full_window.json').exists())
    families = {1:linear}
    for pieces in (2,4):
        fit = optimizer_family(engine, pieces)
        families[pieces] = validated_family(engine, fit['candidate'])
    chosen = select_families(families)
        chosen.update(identity=engine.manifest['identity'], schema='f4-h2-frozen-selection-v1',
        curves={str(p):families[p]['candidate'] for p in families},
        metrics={str(p):{g:selection_metrics(families[p]['streams'][g], engine.data['observations']) for g in ('base','fine')} for p in families},
        comparisons={str(p):families[p]['comparison'] for p in families},
        cutoff=engine.data['packet']['policy']['split']['holdout_start'],
        final_observations_decoded=0, scientific_acceptance=False, blind_h3_acceptance=False)
    path = engine.directory/'selection.json'
    if path.exists():
        require(load(path) == chosen, 'Frozen selection changed.')
    else:
        write(path, chosen)
    return families, chosen


def forecast_week(engine, families, selection):
    from root.f4.measure_h2_compact_rates import compare_states
    from src.method.bm_1_interval_forecast import compare_interval_moments, compare_interval_distributions
    selected = selection['selected_family']; chosen = [1] if selected == 1 else [1, selected]
    outputs = {}; raw_origins = {}; summaries = {}
    for family in chosen:
        info = families[family]; candidate = info['candidate']; key = curve_key(candidate)
        if key in outputs:
            continue
        streams = {}; origin = {}
        for grid in ('base','fine'):
            logs = info['streams'][grid][-1]['log_state'].copy(); origin[grid] = logs.copy()
            rows = []
            for hour in range(96,264):
                moment = engine.job(candidate, grid, hour, 'moments', logs, branch='open')
                full = engine.job(candidate, grid, hour, 'cdf', logs, branch='open') if hour-95 in HORIZONS else None
                if full is not None:
                    active = np.isfinite(moment['log_state']); other = full['log_state']
                    require(np.array_equal(active, np.isfinite(other)) and
                        np.max(np.abs(moment['log_state'][active]-other[active])) <= 1e-7,
                        'Open moment/CF state mismatch.')
                # Continue only this branch's own prior, never a final posterior.
                logs = moment['log_state']
                rows.append(dict(hour=hour, horizon=hour-95, moments=moment, distribution=full))
            streams[grid] = rows
        checks = []
        for base, fine in zip(streams['base'], streams['fine']):
            row = dict(horizon=base['horizon'], moments=compare_interval_moments(base['moments'],fine['moments']),
                state=compare_states(base['moments']['log_state'],fine['moments']['log_state']))
            if base['distribution'] is not None:
                row['distribution'] = compare_interval_distributions(
                    decode_distribution(base['distribution']['distribution']), decode_distribution(fine['distribution']['distribution']))
            checks.append(row)
        outputs[key] = streams; raw_origins[key] = origin
        summaries[key] = dict(candidate=candidate, checks=checks)
    freeze = dict(schema='f4-h2-frozen-open-week-v1', identity=engine.manifest['identity'],
        selection_sha256=sha(engine.directory/'selection.json'), origin=selection['cutoff'],
        forecast_hours=168, horizons=list(HORIZONS), assimilated_observations=0,
        final_observations_decoded=0, conditional_forecast_precision_passed=True,
        scientific_acceptance=False, blind_h3_acceptance=False,
        models=summaries, outputs=outputs, origins=raw_origins)
    freeze['checksum'] = fingerprint(freeze)
    path = engine.directory/'forecast_frozen.npz'
    if path.exists():
        old = _read_array_checkpoint(path.read_bytes())
        require(fingerprint(old) == fingerprint(freeze), 'Forecast freeze content changed.')
        for key in outputs:
            for grid in ('base','fine'):
                same_state(old['origins'][key][grid], raw_origins[key][grid])
    else:
        write(path, freeze)
    export_handoff(engine, selection, freeze)
    return freeze


def export_handoff(engine, selection, frozen):
    """Keep full log states; ordinary cell masses are only an adapter view."""
    origins = {}
    for key, grids in frozen['origins'].items():
        candidate = frozen['models'][key]['candidate']; f,g = curve(candidate)
        origins[key] = dict(curves=dict(f=f.to_record(),g=g.to_record()),
            physical=engine.data['packet']['physical'], candidate=candidate,
            grids={grid:dict(cell_edges=np.linspace(0.,1.,len(logs)+1), log_state=logs)
                for grid,logs in grids.items()})
    artifact = dict(schema='f4-h2-conditional-IJ-handoff-v1', identity=engine.manifest['identity'],
        packet_identity=engine.data['packet']['identity'], selected_family=selection['selected_family'],
        selected_curve_key=curve_key(selection['curves'][str(selection['selected_family'])]),
        origin_time=4., time_unit='day', origin_timestamp=selection['cutoff'],
        frozen_forecast_sha256=sha(engine.directory/'forecast_frozen.npz'),
        population=engine.data['packet']['policy']['population'], origins=origins,
        conditional_prefinal_ready=True, final_evaluation_complete=False,
        scientific_acceptance=False, blind_h3_acceptance=False,
        parameter_uncertainty=False, input_uncertainty=False,
        meaning='Fixed nominal physical condition and fitted shared response; not physical calibration.')
    artifact['checksum'] = fingerprint(artifact)
    path = engine.directory/'conditional_handoff.npz'
    if not path.exists():
        write(path, artifact)
    else:
        require(fingerprint(_read_array_checkpoint(path.read_bytes())) == fingerprint(artifact), 'Frozen handoff changed.')
    return artifact


def as_forecast_origin(artifact, *, model_key, grid='base'):
    """Explicit bridge to existing I/F3 control API; log state stays in artifact."""
    from src.control.bm_1 import BM1Parameters
    from src.control.bm_1_handoff import BM1ForecastOrigin
    from src.params.bm_1_response import BM1PiecewiseResponse
    require(artifact['schema'] == 'f4-h2-conditional-IJ-handoff-v1'
        and artifact['conditional_prefinal_ready'] is True, 'Not a conditional fitted handoff.')
    require(artifact['checksum'] == fingerprint({k:v for k,v in artifact.items() if k != 'checksum'}),
        'Conditional handoff checksum failed.')
    model = artifact['origins'][model_key]; state = model['grids'][grid]; p = model['physical']
    state_check(state['log_state'])
    parameters = BM1Parameters(C=p['C'], k=p['k'], Xi=p['Xi'], sigma_x=p['sigma_x'],
        state_response=BM1PiecewiseResponse.from_record(model['curves']['f']),
        price_response=BM1PiecewiseResponse.from_record(model['curves']['g']))
    return BM1ForecastOrigin(artifact['origin_time'], artifact['time_unit'], parameters,
        p['sigma_y'], state['cell_edges'], np.exp(state['log_state']))


def load_final_observations(engine, workbook):
    from src.tools.bm_1_history_io import read_raw_power_window
    frozen_path = engine.directory/'forecast_frozen.npz'
    require(frozen_path.exists(), 'Freeze and validate the open forecast before final data access.')
    frozen = _read_array_checkpoint(frozen_path.read_bytes())
    require(frozen['checksum'] == fingerprint({k:v for k,v in frozen.items() if k != 'checksum'}),
        'Frozen forecast checksum failed before final access.')
    require(frozen['identity'] == engine.manifest['identity']
        and frozen['selection_sha256'] == sha(engine.directory/'selection.json')
        and frozen['conditional_forecast_precision_passed'] is True,
        'Forecast/selection freeze failed.')
    path = engine.directory/'final_observations.npz'
    packet = engine.data['packet']
    if path.exists():
        value = _read_array_checkpoint(path.read_bytes())
        require(value['forecast_sha256'] == sha(frozen_path)
            and value['input_sha256'] == packet['input_sha256']['input'], 'Final data provenance changed.')
    else:
        require(sha(workbook) == packet['input_sha256']['input'], 'Different raw demand workbook.')
        policy = packet['policy']; split = policy['split']
        frame = read_raw_power_window(workbook, columns=policy['population']['measurement_columns'],
            start=split['holdout_start'], end=split['holdout_end'], timezone=split['timezone'],
            sheet=policy['header_evidence']['input_sheet'])
        raw = frame.to_numpy(dtype=float)
        require(raw.shape == (168,13) and not np.isinf(raw).any(), 'Wrong final fixed population/window.')
        seen = np.isfinite(raw).all(axis=1); group = np.full(168,np.nan)
        group[seen] = raw[seen].mean(axis=1)
        value = dict(times=[t.isoformat() for t in frame.index], observed=seen, group_kW=group,
            values=group/packet['frozen_scales']['power_kW'], forecast_sha256=sha(frozen_path),
            input_sha256=sha(workbook), exposure='development held-out week; prior exposure unknown')
        require(value['input_sha256'] == packet['input_sha256']['input'], 'Workbook changed while decoding final data.')
        validate_observations(value, packet, final=True)
        write(path, value)
    validate_observations(value, packet, final=True)
    return value, frozen


def final_evaluation(engine, workbook):
    observations, frozen = load_final_observations(engine, workbook)
    from src.method.bm_1_interval_forecast import compare_interval_distributions
    reports = {}
    for key, model in frozen['models'].items():
        streams = {grid:engine.replay(model['candidate'], grid, start=96, end=264,
            initial=frozen['origins'][key][grid], observations=observations, branch='final', moments=True)
            for grid in ('base','fine')}
        comparison = compare_streams(streams['base'],streams['fine'], {'final':(96,264)})
        diagnostics = {}
        for grid in ('base','fine'):
            seen = observations['observed']; values = observations['values']
            open_means = np.array([r['moments']['observation_mean'] for r in frozen['outputs'][key][grid]])
            seq_means = np.array([r['moments']['observation_mean'] for r in streams[grid]])
            require(bool(np.any(seen)), 'No observed final hours for declared diagnostics.')
            errors = {name:prediction[seen]-values[seen] for name,prediction in (('open',open_means),('prequential',seq_means))}
            diagnostics[grid] = {name:dict(rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(np.abs(e)))) for name,e in errors.items()}
        coverage = []
        for h in HORIZONS:
            if observations['observed'][h-1]:
                b = decode_distribution(frozen['outputs'][key]['base'][h-1]['distribution']['distribution'])
                f = decode_distribution(frozen['outputs'][key]['fine'][h-1]['distribution']['distribution'])
                y = float(observations['values'][h-1])
                gate = compare_interval_distributions(b,f,observed_values=[y])
                low, high = b.quantile(.05)['value'], b.quantile(.95)['value']
                coverage.append(dict(horizon=h, value=y, pit=float(b.cdf(y)), covered_90=low<=y<=high, precision=gate))
        reports[key] = dict(comparison=comparison, metrics=diagnostics, selected_horizon_diagnostics=coverage)
    result = dict(schema='f4-h2-compact-development-week-result-v1', identity=engine.manifest['identity'],
        forecast_sha256=sha(engine.directory/'forecast_frozen.npz'), final_observation_sha256=sha(engine.directory/'final_observations.npz'),
        observed_hours=int(observations['observed'].sum()), models=reports,
        conditional_historical_study_completed=True, scientific_acceptance=False, blind_h3_acceptance=False,
        scope='Conditional compact development study; seven horizons are descriptive, not calibrated coverage.')
    write(engine.directory/'final_evaluation.json', result)
    return result
