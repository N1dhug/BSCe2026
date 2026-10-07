"""One focused receipt-before-ledger crash/reuse check; no native forwards."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from src.method.bm_1_distribution_panel import baseline, target
from src.method.bm_1_g7 import OwnedBankPriceEvaluator
from src.method.bm_1_g7_jobs import NodeJobStore, digest, atomic_json
from audited_node_control import AuditedBankControl, SOURCE_PATH, validate_audit, POLICY


class LostProcess(BaseException):
    pass


class ProjectionStub:
    matrix = np.eye(2)
    factorizations = 1
    cache_hits = 0
    def solve(self, target, initial):
        return np.array([.4, .6])
    def optimality_gap(self, target, coefficients):
        return 0., np.zeros(2)


def main():
    start = perf_counter();root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root/'root/g7/audit_check_native_manifest.json').read_bytes())
    manifest[SOURCE_PATH] = hashlib.sha256((root/SOURCE_PATH).read_bytes()).hexdigest()
    calls = [];original_solver = ProjectionStub()
    e = object.__new__(OwnedBankPriceEvaluator);e.backend = 'mixture'
    e.projector = SimpleNamespace(projection_solver=original_solver, dx=.25)
    e.origin = SimpleNamespace(rho_values=np.array([.25]), to_record=lambda:dict(time=12.,fixture='owned-stub'))
    model = SimpleNamespace(B=baseline, A=target, _drift=lambda *_:0.)
    support = SimpleNamespace(model=model, interval_edges=np.array([12.,13.,14.]),
        solver_t_eval=np.array([12.,12.5,13.5,14.]),midpoints=np.array([12.5,13.5]),
        widths=np.ones(2),solver_max_step=1/192,breakpoints=[13.],sigma_x=.2,
        x=np.array([.25,.75]),dx=.5,target=np.array([.5,.5]),
        validate_prices=lambda p:np.asarray(p,float))
    e.supports = (support,)
    def forward(prices, index):
        calls.append(index);solver=e.projector.projection_solver
        c=solver.solve(np.array([.4,.6]),np.array([.5,.5]));solver.optimality_gap(np.array([.4,.6]),c)
        return dict(index=index, prices=prices, native_array=np.array([.4,.6]), diagnostics=dict(stub=True))
    e.evaluate_node=forward;e.combine_nodes=lambda prices,nodes:nodes
    checks=[]
    with TemporaryDirectory() as tmp:
        store=NodeJobStore(Path(tmp)/'campaign',prior_debit=0.,ceiling=10.,max_new_seconds=10.)
        adapter=AuditedBankControl(e,store,source_root=root,source_sha256=manifest)
        original_finish=store._finish
        def lose_after_receipt(*args,**kwargs):
            raise LostProcess('Simulated loss after fsynced result, before journal completion.')
        store._finish=lose_after_receipt
        try:
            adapter.evaluate([.5,.5],seconds_per_node=1.)
        except LostProcess:
            pass
        else:
            raise AssertionError('Crash injection did not interrupt ledger completion.')
        store._finish=original_finish
        assert e.projector.projection_solver is original_solver and e.evaluate_node is forward
        assert '_drift' in model.__dict__ and model._drift() == 0.
        checks.append('wrappers restored after process-loss exception')
        ledger=store.read();assert ledger['entries'][0]['status']=='running' and ledger['entries'][0]['charged_seconds']==1.
        receipt=next((store.root/'nodes').glob('*.json'));original_bytes=receipt.read_bytes();value=json.loads(original_bytes)
        assert value['payload_sha256']==digest(value['payload']) and value['payload']['projection_audit']['record']['checked_projections']==1
        checks.append('forward and complete projection audit durable before ledger commit')
        # New store/adapter simulates restart. Receipt reuse must load saved
        # measurements; a fresh live audit with zero calls is never substituted.
        restarted=NodeJobStore(store.root,prior_debit=0.,ceiling=10.,max_new_seconds=10.)
        resumed=AuditedBankControl(e,restarted,source_root=root,source_sha256=manifest)
        nodes=resumed.evaluate([.5,.5],seconds_per_node=1.)
        assert calls==[0] and nodes[0]['native_array']==[.4,.6]
        assert nodes[0]['projection_audit']==value['payload']['projection_audit']
        assert restarted.read()['entries'][0]['status']=='recovered-complete' and restarted.charged()==1.
        assert receipt.read_bytes()==original_bytes
        checks.append('restart reuses saved audit with no forward and no reservation refund')
        # Hash-valid missing-audit receipt is rejected before bank combination.
        corrupt=json.loads(original_bytes);del corrupt['payload']['projection_audit'];corrupt['payload_sha256']=digest(corrupt['payload']);atomic_json(receipt,corrupt)
        try:
            resumed.evaluate([.5,.5],seconds_per_node=1.)
        except ValueError as exc:
            assert 'lacks its own saved projection audit' in str(exc)
        else:
            raise AssertionError('Missing saved audit was accepted.')
        finally:
            atomic_json(receipt,value)
        assert calls==[0] and e.evaluate_node is forward and e.projector.projection_solver is original_solver
        checks.append('missing persisted audit blocks acceptance without replay')
        empty=dict(value['payload']);empty['projection_audit']=dict(value['payload']['projection_audit']);empty['projection_audit']['record']=dict(empty['projection_audit']['record'],solver_calls=0,checked_projections=0)
        try:validate_audit(empty,index=0,backend='mixture')
        except ValueError:pass
        else:raise AssertionError('Empty live audit accepted as completed evidence.')
        checks.append('zero-call mixture audit rejected')
        fv=dict(projection_audit=dict(schema=POLICY['schema'],index=0,backend='fv',status='not-applicable-no-mixture-projection'))
        validate_audit(fv,index=0,backend='fv');checks.append('FV declares not-applicable rather than invented projection success')
    print(json.dumps(dict(schema='g7-per-hypothesis-audit-check-v1',passed=True,checks=checks,
        native_numerical_forwards=0,stub_forwards=len(calls),source_root=str(root),
        native_sources=len(manifest)-1,body_seconds=perf_counter()-start),indent=2))


if __name__=='__main__':main()
