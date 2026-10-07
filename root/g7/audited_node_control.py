"""G-only opt-in adapter: persist projection audit inside each native receipt.

Delegates native transport, bank reduction, deadlines and durable node store.
The extra contract field distinguishes audited jobs from old unaudited jobs.
"""
import hashlib
import math
from pathlib import Path

from src.method.bm_1_g7_jobs import CheckpointedBankControl
from src.exe.run_model.bm_1.g8_bernstein import SolverAudit

SOURCE_PATH = 'root/g7/audited_node_control.py'
RECORDER_PATH = 'src/exe/run_model/bm_1/g8_bernstein.py'
POLICY = dict(schema='g7-per-hypothesis-projection-audit-v1',
              storage='inside-native-hash-authenticated-node-payload',
              mixture_coverage='fresh-per-forward-audit',
              fv_coverage='explicit-not-applicable',
              error_norm='cell-mass-L1')


def validate_audit(node, *, index, backend):
    """Check saved coverage; acceptance tolerances belong to the registered run."""
    value = node.get('projection_audit')
    if not isinstance(value, dict):
        raise ValueError('Completed result lacks its own saved projection audit.')
    if (value.get('schema') != POLICY['schema'] or value.get('index') != index
            or value.get('backend') != backend):
        raise ValueError('Saved audit does not belong to this hypothesis/backend.')
    if backend == 'fv':
        if value != dict(schema=POLICY['schema'], index=index, backend='fv',
                         status='not-applicable-no-mixture-projection'):
            raise ValueError('FV must explicitly declare no mixture projection.')
        return value
    if backend != 'mixture' or value.get('status') != 'complete':
        raise ValueError('Incomplete or unsupported projection audit.')
    record = value.get('record')
    if not isinstance(record, dict):
        raise ValueError('Saved mixture audit record is missing.')
    for name in ('solver_calls', 'checked_projections', 'factorizations', 'cache_hits'):
        number = record.get(name)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError('Invalid audit count: ' + name)
    if (record['solver_calls'] <= 0
            or record['checked_projections'] != record['solver_calls']):
        raise ValueError('Projection-fit coverage is incomplete; no zero-audit pass.')
    for name in ('active_set_seconds', 'maximum_projection_cell_mass_l1',
                 'maximum_scaled_gap', 'maximum_coefficient_mass_error'):
        number = record.get(name)
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or number < 0):
            raise ValueError('Invalid saved audit measurement: ' + name)
    return value


class AuditedBankControl(CheckpointedBankControl):
    """Opt in explicitly; legacy adapter/contracts/receipts stay untouched."""
    def __init__(self, evaluator, store, *, source_root, source_sha256):
        root = Path(source_root).resolve()
        expected = source_sha256.get(SOURCE_PATH)
        if (Path(__file__).resolve() != root / SOURCE_PATH
                or expected != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                or RECORDER_PATH not in source_sha256):
            raise ValueError('Register the loaded G audit adapter and native recorder.')
        # Prevent a numerically different import from borrowing another checkout's hashes.
        import sys
        for module, path in ((CheckpointedBankControl.__module__,
                              'src/method/bm_1_g7_jobs.py'),
                             (SolverAudit.__module__, RECORDER_PATH)):
            if Path(sys.modules[module].__file__).resolve() != root / path:
                raise ValueError('Loaded native module differs from source root: ' + path)
        super().__init__(evaluator, store, source_root=root,
                         source_sha256=source_sha256)

    def describe(self):
        contract = super().describe()
        contract['projection_audit_receipt'] = POLICY.copy()
        return contract

    def evaluate(self, prices, *, seconds_per_node, retry_ids=()):
        evaluator = self.evaluator
        original_forward = evaluator.evaluate_node
        original_combine = evaluator.combine_nodes
        forward_override = evaluator.__dict__.get('evaluate_node')
        combine_override = evaluator.__dict__.get('combine_nodes')
        had_forward = 'evaluate_node' in evaluator.__dict__
        had_combine = 'combine_nodes' in evaluator.__dict__

        def forward(prices, index):
            if evaluator.backend == 'mixture':
                projector = evaluator.projector
                original_solver = projector.projection_solver
                audit = SolverAudit(original_solver, projector.dx)
                projector.projection_solver = audit
                try:
                    result = original_forward(prices, index)
                    saved = dict(schema=POLICY['schema'], index=index,
                                 backend='mixture', status='complete',
                                 record=audit.record())
                finally:
                    projector.projection_solver = original_solver
            else:
                result = original_forward(prices, index)
                saved = dict(schema=POLICY['schema'], index=index, backend='fv',
                             status='not-applicable-no-mixture-projection')
            # Arrays and native diagnostics are unchanged. Store.node() writes
            # this envelope, including audit, before committing the ledger.
            payload = dict(result, projection_audit=saved)
            validate_audit(payload, index=index, backend=evaluator.backend)
            return payload

        def combine(prices, nodes):
            for index, node in enumerate(nodes):
                validate_audit(node, index=index, backend=evaluator.backend)
            return original_combine(prices, nodes)

        evaluator.evaluate_node = forward
        evaluator.combine_nodes = combine
        try:
            return super().evaluate(prices, seconds_per_node=seconds_per_node,
                                    retry_ids=retry_ids)
        finally:
            if had_forward:
                evaluator.evaluate_node = forward_override
            else:
                del evaluator.evaluate_node
            if had_combine:
                evaluator.combine_nodes = combine_override
            else:
                del evaluator.combine_nodes
