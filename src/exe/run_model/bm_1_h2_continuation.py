"""Separate nine-contour execution with immutable predecessor job adoption.

The original kernels, optimizer protocol and checkpoints remain untouched.
An old result is returned with its original token only after full verification.
"""
from pathlib import Path
from time import perf_counter, time
import math
import os
import threading

from src.exe.run_model import bm_1_h2_compact as c
from root.f4 import run_h2_same_day as s
from src.method.bm_1_interval_contour_extension import ORIGINAL_CENTERS, EXTENDED_CENTERS

SOURCES = ('src/exe/run_model/bm_1_h2_continuation.py',
           'src/method/bm_1_interval_contour_extension.py',
           'root/f4/prepare_h2_continuation.py', 'root/f4/run_h2_continuation.py')


def verify_origin(directory):
    directory = Path(directory)
    manifest = c.load(directory/'manifest.json')
    c.require(manifest['identity'] == c.digest({k:v for k,v in manifest.items() if k!='identity'}),
              'Continuation manifest identity changed.')
    c.require(manifest['source_sha256'] == c.source_hashes(), 'Frozen original numerical source changed.')
    c.require(manifest['continuation_source_sha256'] == {p:c.sha(c.ROOT/p) for p in SOURCES},
              'Continuation source changed; register a separate execution.')
    c.require(manifest['centers'] == list(EXTENDED_CENTERS), 'Contour bank changed.')
    origin = c.load(directory/'origin.json')
    c.require(c.sha(directory/'origin.json') == manifest['origin_sha256'], 'Origin record changed.')
    predecessor = c.ROOT/origin['predecessor_path']
    for name, expected in origin['metadata_sha256'].items():
        c.require(c.sha(predecessor/name) == expected, 'Preserved predecessor metadata changed: '+name)
    evidence = predecessor.parent/'f4_h2_bank_fine_control_20261007'
    for name, expected in origin['own_fine_evidence_sha256'].items():
        c.require(c.sha(evidence/name) == expected, 'Own fine-control evidence changed: '+name)
    c.require(c.sha(directory/'predecessor_job_index.json') == manifest['job_index_sha256'],
              'Predecessor raw-job index changed.')
    return manifest, origin, predecessor


class AdoptingJobStore(c.JobStore):
    def __init__(self, directory, identity, predecessor, predecessor_identity, index):
        super().__init__(directory, identity)
        self.predecessor = Path(predecessor)
        self.predecessor_identity = predecessor_identity
        self.index = index
        self.adopted = {}

    def run(self, meta, prior, compute):
        new_token = dict(identity=self.identity, **meta)
        new_key = c.digest(new_token)
        # A new failure stays terminal even if an old successful job exists.
        if (self.directory/'jobs'/f'{new_key}.npz').exists() or (self.directory/'jobs'/f'{new_key}.failed.json').exists():
            return super().run(meta, prior, compute)
        old_token = dict(identity=self.predecessor_identity, **meta)
        old_key = c.digest(old_token)
        if old_key not in self.index:
            return super().run(meta, prior, compute)
        path = self.predecessor/'jobs'/f'{old_key}.npz'
        c.require(c.sha(path) == self.index[old_key], 'Predecessor raw bytes changed.')
        saved = c._read_array_checkpoint(path.read_bytes())
        c.require(saved['token'] == old_token and saved['checksum'] == c.fingerprint(
            {k:v for k,v in saved.items() if k!='checksum'}), 'Predecessor token/content checksum changed.')
        c.state_check(prior)
        c.require(c.fingerprint(saved['prior_log_state']) == c.fingerprint(prior),
                  'Checkpoint prior bytes/dtype changed.')
        c.same_state(saved['prior_log_state'], prior)
        c.state_check(saved['result']['log_state'])
        receipt = dict(schema='f4-h2-predecessor-job-adoption-v1',
            requested_token=new_token, original_token=saved['token'], original_checksum=saved['checksum'],
            predecessor_raw_sha256=self.index[old_key], original_key=old_key,
            exact_prior_fingerprint=c.fingerprint(prior), original_result_preserved=True,
            old_checkpoint_relabelled=False, native_propagations=0)
        target = self.directory/'adopted'/f'{new_key}.json'
        if target.exists():
            c.require(c.load(target) == receipt, 'Adoption receipt changed.')
        else:
            c.write(target, receipt)
        self.adopted[new_key] = receipt
        self.completed[old_key] = saved  # Keep the original checkpoint identity.
        return saved['result']


class Engine(s.Engine):
    def __init__(self, directory, data, manifest, allocation):
        checked, origin, predecessor = verify_origin(directory)
        c.require(checked == manifest, 'Different manifest passed to engine.')
        c.require(data['packet']['identity'] == manifest['packet_identity']
            and data['observation_sha256'] == manifest['observation_sha256']
            and c.environment() == manifest['environment'], 'Packet, observations or environment changed.')
        super().__init__(directory, data, manifest, allocation)
        self.origin, self.predecessor = origin, predecessor
        self.store = AdoptingJobStore(directory, manifest['identity'], predecessor,
            origin['predecessor_identity'], c.load(Path(directory)/'predecessor_job_index.json'))

    def spec(self, grid):
        spec = super().spec(grid)
        c.require(tuple(spec['settings']['centers']) == ORIGINAL_CENTERS, 'Old contour prefix changed.')
        spec['settings'] = dict(spec['settings'], centers=list(EXTENDED_CENTERS))
        return spec

    def check(self, *, search=None):
        c.require(not (self.directory/'CONTINUATION_FAILED.json').exists(), 'Terminal continuation; no retry.')
        super().check(search=search)


def optimizer_family(engine, pieces):
    """Replay preserved objective values, then append only new calls."""
    from scipy.optimize import minimize
    c.require(pieces in (2,4), 'Unknown response family.')
    old_path = engine.predecessor/f'extension_optimizer_{pieces}_calls.json'
    original = c.load(old_path)
    c.require(original['identity'] == engine.origin['predecessor_identity'], 'Wrong old optimizer identity.')
    old_calls = original['calls']
    old_result = engine.predecessor/f'extension_optimizer_{pieces}_result.json'
    if old_result.exists():
        result = c.load(old_result)
        c.require(result['converged'] is True and result['status'] == 0
            and result['identity'] == engine.origin['predecessor_identity'], 'Old optimizer was not converged.')
        # Do not give the old optimizer result a new identity.
        return result
    path = engine.directory/f'continuation_optimizer_{pieces}_calls.json'
    extra = c.load(path) if path.exists() else dict(identity=engine.manifest['identity'],
        preserved_trace_sha256=c.sha(old_path), preserved_call_count=len(old_calls), calls=[])
    c.require(extra['identity'] == engine.manifest['identity']
        and extra['preserved_trace_sha256'] == c.sha(old_path)
        and extra['preserved_call_count'] == len(old_calls), 'Optimizer origin changed.')
    calls = old_calls+extra['calls']; cursor = 0
    result_path = engine.directory/f'continuation_optimizer_{pieces}_result.json'
    if result_path.exists():
        result = c.load(result_path)
        c.require(result['identity'] == engine.manifest['identity'] and result['converged'],
                  'Saved continuation optimizer did not converge.')
        return result
    registry_path = engine.directory/'continuation_candidate_registry.json'
    registry = c.load(registry_path) if registry_path.exists() else dict(identity=engine.manifest['identity'],
        preserved_registry_sha256=c.sha(engine.predecessor/'extension_candidate_registry.json'),
        families=c.load(engine.predecessor/'extension_candidate_registry.json')['families'])
    c.require(registry['identity'] == engine.manifest['identity'] and registry['preserved_registry_sha256'] ==
        c.sha(engine.predecessor/'extension_candidate_registry.json'), 'Candidate registry origin changed.')
    c.require(all(len(v)==len(set(v)) and len(v)<={'2':24,'4':48}[k] for k,v in registry['families'].items())
        and 1+sum(len(v) for v in registry['families'].values())<=73, 'Candidate ceiling exceeded.')
    def objective(logits):
        nonlocal cursor
        candidate = dict(pieces=pieces, logits=c.np.asarray(logits).tolist())
        key = c.curve_key(candidate)
        if cursor < len(calls):
            row = calls[cursor]
            c.require(row['logits'] == candidate['logits'] and row['key'] == key
                and row['penalty'] == c.penalty(candidate)
                and row['objective'] == -row['score']+row['penalty'], 'Cached optimizer trajectory changed.')
            cursor += 1
            return row['objective']
        engine.check(search=True)
        keys = registry['families'][str(pieces)]
        if key != 'linear' and key not in keys:
            c.require(len(keys)<{2:24,4:48}[pieces] and 1+sum(len(v) for v in registry['families'].values())<73,
                      'Declared candidate cap reached.')
            keys.append(key); c.write(registry_path,registry,replace=registry_path.exists())
        rows = engine.replay(candidate,'base',end=72,search=True)
        score = c.ordered_sum(r['log_evidence'] for r in rows)
        penalty = c.penalty(candidate); value = -score+penalty
        c.require(math.isfinite(value), 'Nonfinite objective.')
        row = dict(logits=candidate['logits'],key=key,score=score,penalty=penalty,objective=value)
        extra['calls'].append(row); calls.append(row);cursor+=1
        c.write(path,extra,replace=path.exists())
        print(c.json.dumps(dict(stage='continuation_fit',pieces=pieces,objective_calls=cursor,
            native_jobs=len(engine.store.events),adopted_jobs=len(engine.store.adopted))),flush=True)
        return value
    result = minimize(objective,c.np.zeros(2*(pieces-1)),method='COBYQA',bounds=[(-4.,4.)]*(2*(pieces-1)),
        options=dict(maxfev={2:25,4:49}[pieces],initial_tr_radius=.5,final_tr_radius=.02,scale=False))
    c.require(cursor==len(calls),'Unconsumed cached optimizer calls.')
    record = dict(identity=engine.manifest['identity'],pieces=pieces,candidate=dict(pieces=pieces,logits=result.x.tolist()),
        objective=float(result.fun),nfev=int(result.nfev),status=int(result.status),message=str(result.message),
        converged=bool(result.success) and int(result.status)==0 and math.isfinite(float(result.fun)),
        scientific_acceptance=False,algorithmic_local_termination_only=True)
    c.write(result_path,record)
    c.require(record['converged'],'COBYQA did not converge within the preserved candidate cap.')
    return record


def prefinal(engine):
    from root.f4.measure_h2_compact_rates import work_graph
    engine.rates = engine.allocation['protected_rates']
    engine.validation_remaining = work_graph(engine.rates,48,24,engine.allocation['before'])['protected_validation_seconds']
    families = {1:c.validated_family(engine,c.LINEAR)}
    for pieces in (2,4):
        fit = optimizer_family(engine,pieces)
        families[pieces] = c.validated_family(engine,fit['candidate'])
    selected = c.select_families(families)
    selected.update(schema='f4-h2-frozen-selection-v1',identity=engine.manifest['identity'],
        curves={str(p):families[p]['candidate'] for p in families},
        metrics={str(p):{g:c.selection_metrics(families[p]['streams'][g],engine.data['observations'])
                        for g in ('base','fine')} for p in families},
        comparisons={str(p):families[p]['comparison'] for p in families},
        cutoff=engine.data['packet']['policy']['split']['holdout_start'],final_observations_decoded=0,
        scientific_acceptance=False,blind_h3_acceptance=False)
    path = engine.directory/'selection.json'
    if path.exists(): c.require(c.load(path)==selected,'Frozen model selection changed.')
    else: c.write(path,selected)
    c.forecast_week(engine,families,selected)
    return dict(status='prefinal_completed',selected_family=selected['selected_family'],final_observations_decoded=0)
