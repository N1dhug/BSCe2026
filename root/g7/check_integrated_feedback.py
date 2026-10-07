"""One targeted stub update/crash/reuse/continuation and budget-gate check.

Native receipt store, bank reduction, audit and Bayes algebra are used.
All transport forwards, likelihood evaluations and projection solves are stubs.
"""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from src.method.bm_1_distribution_panel import build_problem, baseline, target
from src.method.bm_1_g8 import G8BankOrigin, RHO_VALUES
from src.method.bm_1_g8_bernstein import bernstein_basis, BASIS_KIND
from src.method.bm_1_g7 import OwnedBankPriceEvaluator
from src.method import bm_1_g7 as native
from src.method.bm_1_g7_jobs import NodeJobStore, digest, atomic_json
from src.params.bm_1 import BayesUpdate, mass_to_log
from src.control.bm_1 import BM1ControlNumerics
from audited_feedback import FeedbackBankControl, validate_update
from integrated_budget import price_plan, validate_plan, digest as std_digest
from launch_integrated_feedback import preflight
from receipt_recovery import guarded_store_class


class LostProcess(BaseException): pass


class ProjectionStub:
    def __init__(self, basis, dx):
        self.matrix=basis.T*np.sqrt(dx);self.factorizations=1;self.cache_hits=0
    def solve(self, target, initial): return initial.copy()
    def optimality_gap(self, target, c): return 0.,np.zeros(len(c))


def main():
    started=perf_counter();root=Path(__file__).resolve().parents[2]
    manifest=json.loads((root/'root/g7/integration_native_manifest.json').read_bytes())
    for name in ['audited_node_control.py','audited_feedback.py','integrated_budget.py',
                 'feedback_phase.py','launch_integrated_feedback.py','receipt_recovery.py']:
        path='root/g7/'+name;manifest[path]=hashlib.sha256((root/path).read_bytes()).hexdigest()
    source_fp=std_digest(manifest)
    fixed=build_problem().parameters
    edges=np.linspace(0.,1.,4);basis=bernstein_basis(edges)
    c=np.full((5,255),1/255);m=c@basis/3
    origin=G8BankOrigin(12.,fixed,RHO_VALUES,np.full(5,-np.log(5)),edges,m,
        backend='mixture',conditional_coefficients=c,basis=basis,basis_kind=BASIS_KIND,
        basis_identity_sha256=G8BankOrigin.basis_identity(edges,basis))
    forwards=[];likelihood_calls=[]
    def stub_update(model,parameters,sigma_y,state_mass,*args,**kwargs):
        likelihood_calls.append(model.index)
        raw=np.array(state_mass).copy();raw[0]+=.005;raw[-1]-=.005
        return BayesUpdate(mass_to_log(raw),.02*model.index)

    def install_forward(e):
        def forward(prices,index):
            forwards.append((e.origin.time,index))
            s=e.supports[index]
            mass=e.origin.conditional_mass[index]
            coefficients=e.origin.conditional_coefficients
            coeff=None if coefficients is None else coefficients[index]
            if e.backend=='mixture':
                solver=e.projector.projection_solver
                coeff=solver.solve(mass/np.sqrt(s.dx),coeff)
                solver.optimality_gap(mass/np.sqrt(s.dx),coeff)
            n=len(s.solver_t_eval)
            return dict(index=index,prices=np.array(prices),times=s.solver_t_eval,
                mass=np.tile(mass,(n,1)),coefficients=None if coeff is None else np.tile(coeff,(n,1)),
                diagnostics=dict(stub=True),expected_demand=np.full(len(s.midpoints),.5),
                second_moment=np.full(len(s.midpoints),.26),
                node_total=float(np.sum(s.widths)*.01))
        e.evaluate_node=forward

    def fake_evaluator(start,backend):
        e=object.__new__(OwnedBankPriceEvaluator);e.origin=start;e.backend=backend
        e.supports=tuple(SimpleNamespace(
            model=SimpleNamespace(B=baseline,A=target,T=(12.,36.),index=j,_drift=lambda *_:0.),
            interval_edges=np.array([12.,18.,24.,30.,36.]),
            solver_t_eval=np.array([12.,24.,36.]),midpoints=np.array([24.]),
            widths=np.array([24.]),solver_max_step=1/96,breakpoints=[18.,24.,30.],
            sigma_x=.2,x=np.array([1/6,.5,5/6]),dx=1/3,target=np.array([.5]),
            validate_prices=lambda p:np.asarray(p,float),parameters_for_prices=lambda p:None)
            for j in range(5))
        e.projector=None
        if backend=='mixture':
            e.projector=SimpleNamespace(dx=1/3,basis=start.basis,
                                       projection_solver=ProjectionStub(start.basis,1/3))
            def project(density,initial_coefficients):
                trial=initial_coefficients.copy();trial[0]+=.001;trial[-1]-=.001
                solver=e.projector.projection_solver
                coeff=solver.solve(density*np.sqrt(1/3),trial)
                solver.optimality_gap(density*np.sqrt(1/3),coeff)
                projected=coeff@start.basis
                return dict(coefficients=coeff,density=projected,
                            diagnostics=dict(cell_mass_l1_error=float(np.sum(abs(projected-density))/3)))
            e.projector.project=project
        install_forward(e);return e

    checks=[]
    def rejected(action,message):
        try: action()
        except (ValueError,TypeError):checks.append(message)
        else: raise AssertionError(message+' was accepted')
    Guard=guarded_store_class(NodeJobStore,
        native_source_path=root/'src/method/bm_1_g7_jobs.py',
        native_source_sha256=manifest['src/method/bm_1_g7_jobs.py'])
    original_update=native.bm1_observation_update
    native.bm1_observation_update=stub_update
    try:
        with TemporaryDirectory() as tmp:
            tmp=Path(tmp);store=Guard(tmp/'campaign',prior_debit=0.,ceiling=30.,max_new_seconds=30.)
            e=fake_evaluator(origin,'mixture')
            control=FeedbackBankControl(e,store,source_root=root,source_sha256=manifest)
            forward=control.evaluate([.5]*4,seconds_per_node=1.)
            original_bytes={p.name:p.read_bytes() for p in (store.root/'nodes').glob('*.json')}
            baseline_calls=len(forwards)
            original_finish=store._finish
            def lose(*args,**kwargs):raise LostProcess('After complete update receipt, before ledger commit.')
            store._finish=lose
            try:control.assimilate(forward,.5,seconds=2.)
            except LostProcess:pass
            else:raise AssertionError('Crash position not reached')
            store._finish=original_finish
            assert native.bm1_observation_update is stub_update
            checks.append('raw evidence/posterior and projection audit saved before ledger completion')
            assert likelihood_calls==list(range(5)) and len(forwards)==baseline_calls
            checks.append('one native bank assimilation wrapper; no forward replay')
            restart=Guard(store.root,prior_debit=0.,ceiling=30.,max_new_seconds=30.)
            resumed=FeedbackBankControl(e,restart,source_root=root,source_sha256=manifest)
            update=resumed.assimilate(forward,.5,seconds=2.)
            assert len(likelihood_calls)==5 and len(forwards)==baseline_calls
            assert restart.read()['entries'][-1]['status']=='recovered-complete'
            assert restart.read()['entries'][-1]['charged_seconds']==2.
            checks.append('restart reuses posterior without likelihood/projection replay or refund')
            assert update['receipt']['projection_audit']['record']['checked_projections']==5
            assert not np.array_equal(update['origin'].log_weights,origin.log_weights)
            assert not np.array_equal(update['origin'].conditional_coefficients,origin.conditional_coefficients)
            checks.append('saved raw node evidences determine parameter weights independently of state projection')
            next_control=resumed.continuation(update,horizon=(36.,38.),price_edges=[36.,37.,38.],
                numerics=BM1ControlNumerics(x_steps=3,objective_max_step=2.,solver_max_step=1.))
            assert digest(next_control.evaluator.origin.to_record())==digest(update['origin'].to_record())
            next_control.evaluator.projector.projection_solver=ProjectionStub(basis,1/3)
            install_forward(next_control.evaluator)
            result=next_control.evaluate([.3,.7],seconds_per_node=1.)
            assert result['predicted_origin'].time==38 and all(t==36 for t,j in forwards[baseline_calls:])
            assert np.array_equal(result['weights'],update['origin'].weights)
            checks.append('next audited forward starts from saved updated cutoff/state/weights')
            bad=copy.deepcopy(update['receipt']);bad['conditional_log_evidence'][0]+=.1
            rejected(lambda:validate_update(bad,bad['identity']),'changed saved evidence rejected')
            bad=copy.deepcopy(update['receipt']);bad['posterior_origin']['parameters']['k']+=.1
            rejected(lambda:validate_update(bad,bad['identity']),'borrowed model parameters rejected')
            rejected(lambda:resumed.continuation(update,horizon=(12.,14.),price_edges=[12.,13.,14.],
                numerics=BM1ControlNumerics(x_steps=3)),'old cutoff continuation rejected')
            for name,raw in original_bytes.items():assert (store.root/'nodes'/name).read_bytes()==raw
            checks.append('original completed forward receipts unchanged')
            missing=resumed.assimilate(forward,None,seconds=1.,observed=False)
            assert len(likelihood_calls)==5 and digest(missing['origin'].to_record())==digest(forward['predicted_origin'].to_record())
            checks.append('missing observation preserves predicted law with no invented evidence/audit')
            fv_origin=G8BankOrigin(12.,fixed,RHO_VALUES,origin.log_weights,edges,m,backend='fv')
            fv=fake_evaluator(fv_origin,'fv')
            fvc=FeedbackBankControl(fv,restart,source_root=root,source_sha256=manifest)
            fwd=fvc.evaluate([.5]*4,seconds_per_node=1.)
            fu=fvc.assimilate(fwd,.5,seconds=2.)
            assert fu['receipt']['projection_audit']['status']=='not-applicable-no-mixture-projection'
            checks.append('FV evidence/posterior saved with explicit no-projection status')
            receipts=[]
            for j in range(5):
                job=dict(contract=fvc.contract_id,index=j,rho=float(fv_origin.rho_values[j]),prices=[.5]*4)
                receipts.append(json.loads((restart.root/'nodes'/(digest(job)+'.json')).read_bytes()))
            new_store=Guard(tmp/'new_campaign',prior_debit=0.,ceiling=10.,max_new_seconds=10.)
            external=FeedbackBankControl(fv,new_store,source_root=root,source_sha256=manifest)
            before=len(forwards);external.assimilate(fwd,.5,seconds=2.,forward_receipts=receipts)
            assert len(forwards)==before
            checks.append('pinned external audited forwards used read-only in fresh phase journal')
            bad=copy.deepcopy(receipts);bad[0]['payload_sha256']='0'*64
            rejected(lambda:external.assimilate(fwd,.6,seconds=2.,forward_receipts=bad),'changed adopted forward hash rejected')
    finally:native.bm1_observation_update=original_update
    plan=json.loads((root/'root/g7/integration_design.json').read_bytes())
    versions=dict(numpy=np.__version__,scipy=__import__('scipy').__version__)
    origin_fp=std_digest(origin.to_record())
    def priced(rates,**kwargs):
        args=dict(prior_debit=1846.3372718589885,ceiling=1910,
                  source_fingerprint=source_fp,origin_fingerprint=origin_fp,
                  versions=versions,reserve_seconds=2.)
        args.update(kwargs);return price_plan(plan,rates,**args)
    empty=priced({});assert not empty['budget_admitted'] and empty['estimated_seconds'] is None and len(empty['unknown'])==16
    checks.append('all unknown full-graph rates block admission without 2h extrapolation')
    rates={row['work']:dict(complete=True,raw_reviewed=True,context_sha256=empty['context_sha256'],
        horizon_hours=row['horizon_hours'],rate_unit=row['rate_unit'],seconds=.01,receipt_sha256='a'*64)
        for row in plan['jobs']}
    assert priced(rates)['budget_admitted']
    checks.append('positive complete synthetic rate fixture covers whole graph and reserve')
    changed=copy.deepcopy(rates);changed['search_m254']['horizon_hours']=2
    assert not priced(changed)['budget_admitted']
    changed=copy.deepcopy(rates);changed['search_m254']['context_sha256']='0'*64
    assert not priced(changed)['budget_admitted']
    checks.append('wrong horizon/source/origin context blocks rate reuse')
    rejected(lambda:priced(rates,prior_debit=True),'boolean debit rejected')
    changed=copy.deepcopy(plan);changed['jobs'][0]['count']=0
    rejected(lambda:validate_plan(changed),'changed work graph rejected')
    changed=copy.deepcopy(rates);changed['search_m254']['seconds']=10.
    assert not priced(changed)['budget_admitted']
    checks.append('whole graph and original optimizer cap enforced')
    with TemporaryDirectory() as tmp:
        tmp=Path(tmp);packet=dict(forwards={name:dict(origin=origin.to_record()) for name in
            ('m254_coarse','m254_fine','fv_coarse','fv_fine')},projection_l1_limit=.05)
        # Deliberately empty real rates; no source of numerical admission.
        values={'design.json':plan,'rates.json':{},'previous_budget.json':dict(campaign_debit=1846.3372718589885,campaign_ceiling=1910),
                'source_manifest.json':manifest,'feedback_inputs.json':packet}
        hashes={}
        for name,value in values.items():
            raw=(json.dumps(value,sort_keys=True)+'\n').encode();(tmp/name).write_bytes(raw)
            hashes[name]=hashlib.sha256(raw).hexdigest()
        reg=dict(schema='g7-feedback-phase-registration-v2',stage='feedback-only',automatic_retry=False,
            execution_authorized=True,source_root=str(root),prior_debit=1846.3372718589885,campaign_ceiling=1910,
            design_sha256=hashes['design.json'],rates_sha256=hashes['rates.json'],
            previous_budget_sha256=hashes['previous_budget.json'],source_manifest_sha256=hashes['source_manifest.json'],
            feedback_inputs_sha256=hashes['feedback_inputs.json'],source_fingerprint=source_fp,
            origin_fingerprint=std_digest({k:v['origin'] for k,v in packet['forwards'].items()}),
            versions=versions,scope_cap=5,worker_seconds=3,stop_export_review_reserve=2)
        (tmp/'registration.json').write_text(json.dumps(reg))
        rejected(lambda:preflight(tmp),'launcher blocks unadmitted graph before worker start')
        assert not (tmp/'invocation.json').exists()
    print(json.dumps(dict(schema='g7-integrated-feedback-check-v1',passed=True,checks=checks,
        native_transport_forwards=0,native_likelihood_evaluations=0,native_projection_solves=0,
        stub_forwards=len(forwards),stub_likelihood_calls=len(likelihood_calls),
        rates_fixture_only=True,body_seconds=perf_counter()-started),indent=2))


if __name__=='__main__': main()
