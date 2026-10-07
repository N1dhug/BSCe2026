"""G-only durable native point update and continuation from its own posterior.

Native assimilate/likelihood/projection equations are delegated unchanged.
One process/thread owns this adapter; no concurrent native assimilate calls.
"""
import hashlib
from pathlib import Path
from time import perf_counter
import threading

import numpy as np
from scipy.special import logsumexp

from audited_node_control import AuditedBankControl, validate_audit
from src.method import bm_1_g7 as native
from src.method.bm_1_g7_jobs import digest, encode, NodeJobStop
from src.method.bm_1_g8 import G8BankOrigin

SOURCE_PATH = 'root/g7/audited_feedback.py'
SCHEMA = 'g7-owned-point-update-v2'
_CAPTURE_LOCK = threading.Lock()


def evaluation_record(value):
    """Full scientific result, without a process-local owner pointer."""
    result = {k: v for k, v in value.items() if k != '_owner'}
    result['predicted_origin'] = value['predicted_origin'].to_record()
    return result


def validate_update(payload, identity):
    """Authenticate algebra/ownership; numerical comparison gates are separate."""
    if payload.get('schema') != SCHEMA or payload.get('identity') != identity:
        raise ValueError('Update belongs to another forward/observation.')
    prior = G8BankOrigin.from_record(payload['predicted_origin'])
    posterior = G8BankOrigin.from_record(payload['posterior_origin'])
    left, right = prior.to_record(), posterior.to_record()
    for key in left:
        if key not in ('conditional_mass', 'conditional_coefficients', 'log_weights'):
            if digest(left[key]) != digest(right.get(key)):
                raise ValueError('Posterior changed a frozen model/grid/basis field: ' + key)
    count = len(prior.rho_values)
    rows = payload['raw_updates']; observation = identity['observation']
    if observation['time'] != prior.time:
        raise ValueError('Observation is not at the predicted endpoint.')
    if observation['observed']:
        if len(rows) != count or [r['index'] for r in rows] != list(range(count)):
            raise ValueError('Incomplete raw likelihood/posterior coverage.')
        scores = np.array([r['log_evidence'] for r in rows], float)
        if not np.isfinite(scores).all():
            raise ValueError('Observed node evidences must be finite log densities.')
        raw = np.array([r['mass'] for r in rows], float)
        if (raw.shape != prior.conditional_mass.shape or not np.isfinite(raw).all()
                or np.min(raw) < 0 or not np.allclose(raw.sum(1), 1., atol=1e-9, rtol=0.)):
            raise ValueError('Raw Bayes posterior is not normalized on its own grid.')
        log_z = float(logsumexp(prior.log_weights + scores))
        if (abs(log_z - payload['log_evidence']) > 1e-10
                or not np.allclose(posterior.log_weights,
                                   prior.log_weights + scores - log_z,
                                   atol=1e-10, rtol=0.)):
            raise ValueError('Parameter weights do not use raw pre-projection evidences.')
        if not np.array_equal(scores, np.asarray(payload['conditional_log_evidence'])):
            raise ValueError('Saved node evidences differ from captured raw update.')
        projections = payload['posterior_projections']
        if prior.backend == 'mixture':
            if len(projections) != count:
                raise ValueError('Each observed mixture posterior needs projection diagnostics.')
            l1 = np.sum(abs(posterior.conditional_mass - raw), axis=1)
            for j, diagnostics in enumerate(projections):
                measured = diagnostics.get('cell_mass_l1_error')
                if (isinstance(measured, bool) or not isinstance(measured, (int, float))
                        or not np.isfinite(measured) or measured < 0
                        or abs(measured - l1[j]) > 1e-10):
                    raise ValueError('Raw-to-projected posterior L1 differs from saved audit.')
            audit = payload['projection_audit']
            # Reuse the already registered forward-audit count/type validation.
            validate_audit(dict(projection_audit=audit), index=-1, backend='mixture')
            if audit['record']['solver_calls'] < count:
                raise ValueError('Posterior solver audit omits a hypothesis.')
        else:
            if projections or not np.allclose(raw, posterior.conditional_mass, atol=1e-10, rtol=0.):
                raise ValueError('FV posterior was unexpectedly projected.')
            validate_audit(dict(projection_audit=payload['projection_audit']), index=-1, backend='fv')
    else:
        if (rows or payload['log_evidence'] is not None
                or payload['conditional_log_evidence'] != [None] * count
                or payload['posterior_projections']
                or digest(prior.to_record()) != digest(posterior.to_record())
                or payload['projection_audit'] != dict(status='not-applicable-missing-observation')):
            raise ValueError('Missing observation must preserve the predicted law without evidence.')
    return posterior


class FeedbackBankControl(AuditedBankControl):
    def __init__(self, evaluator, store, *, source_root, source_sha256):
        root = Path(source_root).resolve()
        if (Path(__file__).resolve() != root / SOURCE_PATH
                or source_sha256.get(SOURCE_PATH) != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
            raise ValueError('Register the loaded feedback adapter.')
        super().__init__(evaluator, store, source_root=root, source_sha256=source_sha256)

    def describe(self):
        result = super().describe()
        result['point_update_receipt'] = dict(schema=SCHEMA,
            evidence='raw-before-state-projection', observations='point-demand-only',
            storage='one-native-atomic-receipt', continuation='own-saved-posterior')
        return result

    def assimilate(self, evaluation, value, *, seconds, observed=True,
                   forward_receipts=None):
        """Exactly one native bank update; no forward replay and no auto retry."""
        from src.exe.run_model.bm_1.g8_bernstein import SolverAudit
        from audited_node_control import POLICY
        self.check_sources()
        e = self.evaluator
        if digest(self.describe()) != self.contract_id or evaluation.get('_owner') is not e:
            raise ValueError('Update requires this unchanged evaluator and its own forward.')
        if not isinstance(observed, (bool, np.bool_)):
            raise ValueError('observed must be boolean.')
        present = bool(observed)
        if present and (isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value))):
            raise ValueError('Finite point observation required.')
        with self.store.lock():
            nodes, ids, hashes = [], [], []
            prices = e.supports[0].validate_prices(evaluation['prices']).tolist()
            for j in range(len(e.supports)):
                job = dict(contract=self.contract_id, index=j,
                           rho=float(e.origin.rho_values[j]), prices=prices)
                key = digest(job); path = self.store.root / 'nodes' / (key + '.json')
                if forward_receipts is None:
                    if not path.exists():
                        raise ValueError('Update cannot silently create a missing forward receipt.')
                    node = self.store.receipt(path, job)
                else:
                    # External originals are read-only. The launcher pins the
                    # complete packet and prior budget before numerical imports.
                    if len(forward_receipts) != len(e.supports):
                        raise ValueError('All adopted hypothesis receipts are required.')
                    item = forward_receipts[j]
                    if (item.get('schema') != 'g7-owned-node-v1' or item.get('job') != job
                            or item.get('payload_sha256') != digest(item.get('payload'))):
                        raise ValueError('Adopted audited receipt has another exact job/hash.')
                    node = item['payload']
                validate_audit(node, index=j, backend=e.backend)
                nodes.append(node); ids.append(key); hashes.append(digest(node))
            # Pure reduction, not a transport/likelihood/projection call.
            restored = e.combine_nodes(prices, nodes)
            fingerprint = digest(evaluation_record(restored))
            if fingerprint != digest(evaluation_record(evaluation)):
                raise ValueError('Live result differs from its durable forward receipts.')
            identity = dict(kind=SCHEMA, index=-1, contract=self.contract_id,
                forward_jobs=ids, forward_payload_sha256=hashes,
                forward_result_sha256=fingerprint,
                observation=dict(time=restored['predicted_origin'].time,
                    value=float(value) if present else None, observed=present,
                    kind='point', price_side=e.origin.price_side),
                adopted_packet_sha256=None if forward_receipts is None else digest(forward_receipts))

            def compute(deadline):
                if perf_counter() >= deadline:
                    raise NodeJobStop('Update reservation exhausted.')
                raw = []; original_update = native.bm1_observation_update
                def capture(*args, **kwargs):
                    if perf_counter() >= deadline:
                        raise NodeJobStop('Update reservation expired before hypothesis.')
                    result = original_update(*args, **kwargs)
                    raw.append(dict(index=len(raw), log_evidence=float(result.log_evidence),
                                    mass=result.mass.copy()))
                    return result
                audit = None; solver = None
                if e.backend == 'mixture' and present:
                    solver = e.projector.projection_solver
                    audit = SolverAudit(solver, e.projector.dx)
                with _CAPTURE_LOCK:
                    native.bm1_observation_update = capture
                    if audit is not None: e.projector.projection_solver = audit
                    try:
                        result = e.assimilate(restored, float(value) if present else 0., observed=present)
                        saved_audit = dict(status='not-applicable-missing-observation')
                        if present:
                            saved_audit = dict(schema=POLICY['schema'], index=-1,
                                backend=e.backend, status='complete' if audit else 'not-applicable-no-mixture-projection')
                            if audit is not None: saved_audit['record'] = audit.record()
                    finally:
                        native.bm1_observation_update = original_update
                        if audit is not None: e.projector.projection_solver = solver
                payload = dict(schema=SCHEMA, identity=identity,
                    predicted_origin=restored['predicted_origin'].to_record(),
                    posterior_origin=result['origin'].to_record(), raw_updates=raw,
                    conditional_log_evidence=result['conditional_log_evidence'] if present else [None] * len(e.supports),
                    log_evidence=float(result['log_evidence']) if present else None,
                    posterior_projections=result['projections'], projection_audit=saved_audit)
                # Validate before NodeJobStore commits the single durable envelope.
                checked = __import__('json').loads(encode(payload))
                validate_update(checked, identity)
                if perf_counter() >= deadline:
                    raise NodeJobStop('Update exceeded its reservation.')
                return checked
            payload = self.store.node(identity, seconds, compute)
            posterior = validate_update(payload, identity)
        return dict(origin=posterior, receipt=payload, job_id=digest(identity),
                    numerical_validation_passed=False, scientific_accept=False)

    def continuation(self, update, *, horizon, price_edges, numerics):
        """Build the unchanged native evaluator from the saved updated origin."""
        with self.store.lock():
            payload = self.store.receipt(self.store.root / 'nodes' / (update['job_id'] + '.json'),
                                         update['receipt']['identity'])
        origin = validate_update(payload, payload['identity'])
        if (digest(origin.to_record()) != digest(update['origin'].to_record())
                or horizon[0] != origin.time or price_edges[0] != origin.time):
            raise ValueError('Continuation must start at its own saved posterior cutoff.')
        from src.method.bm_1_distribution_panel import baseline, target
        next_evaluator = native.OwnedBankPriceEvaluator(origin, baseline=baseline,
            target=target, horizon=horizon, price_edges=price_edges, numerics=numerics,
            backend=origin.backend)
        return FeedbackBankControl(next_evaluator, self.store,
            source_root=self.root, source_sha256=self.sources)
