"""One full-graph G check: actual orchestration/store/review, stub kernels.

Explicit fixture partitions/forwards/ODE/Powell/likelihood/EM/projector.
This never measures a scientific rate or validates numerical propagation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

import day_control as day
import feedback_phase as phase
from day_review import paired_improvement
from integrated_budget import COUNTS, price_plan, digest as std_digest
from audited_feedback import FeedbackBankControl
from src.method import bm_1_g7 as native
from src.method.bm_1_g7_jobs import NodeJobStore, atomic_json, digest
from src.method.bm_1_g8 import G8BankOrigin, RHO_VALUES
from src.method.bm_1_g8_bernstein import bernstein_basis, BASIS_KIND
from src.method.bm_1_distribution_panel import build_problem, baseline, target
from src.model.bm_1 import bm_1
from src.control.bm_1 import DeterministicPriceEvaluator, BM1ControlNumerics
from src.params.bm_1 import BayesUpdate, mass_to_log
from src.tools.input_generator.input_builder import input_builder
from receipt_recovery import guarded_store_class, audit_store


class ProjectionStub:
    def __init__(self, basis, dx):
        self.matrix = basis.T*np.sqrt(dx)
        self.factorizations = 1; self.cache_hits = 0
    def solve(self, target, initial):
        return initial.copy()
    def optimality_gap(self, target, coefficients):
        return 0., np.zeros(len(coefficients))


def main(output):
    started = perf_counter(); output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((output/'source_manifest.json').read_bytes())
    for path, expected in manifest.items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == expected
    plan = json.loads((root/'root/g7/integration_design.json').read_bytes())
    fixed = build_problem().parameters; logs = np.full(5, -np.log(5))
    origins = {}
    for cells in (257,513):
        edges = np.linspace(0,1,cells+1); basis = bernstein_basis(edges)
        c = np.full((5,255), 1/255)
        mass = c@basis/cells
        for name, backend in [('m254','mixture'),('fv','fv')]:
            level = 'coarse' if cells == 257 else 'fine'
            origins[name+'_'+level] = G8BankOrigin(12, fixed, RHO_VALUES, logs, edges,
                mass, backend=backend, conditional_coefficients=c if name=='m254' else None,
                basis=basis if name=='m254' else None,
                basis_kind=BASIS_KIND if name=='m254' else None,
                basis_identity_sha256=G8BankOrigin.basis_identity(edges,basis) if name=='m254' else None)
    records = {k:v.to_record() for k,v in origins.items()}
    real_owned = native.OwnedBankPriceEvaluator
    events = []; counts = dict(forward=0, ode=0, em=0, likelihood=0, powell=0)

    def contrast(prices):
        return float(np.asarray(prices)@np.resize([1.,-1.], len(prices)))
    def variance(prices):
        return .01+.001*contrast(prices)

    def evaluator(origin, *, horizon, price_edges, numerics, backend, **kwargs):
        e = object.__new__(real_owned); e.origin = origin; e.backend = backend
        cells = len(origin.x_edges)-1; dx = 1/cells
        pieces = 8 if len(price_edges)==5 and numerics.objective_max_step==1/192 else len(price_edges)-1
        edges_t = np.linspace(horizon[0], horizon[1], pieces+1)
        midpoints = .5*(edges_t[1:]+edges_t[:-1])
        times = np.r_[horizon[0],midpoints,horizon[1]]
        supports = []
        for j, node in enumerate(origin.node_origins()):
            model = bm_1(baseline,target,horizon,2)
            model._stub_index = j; model._stub_origin = origin
            def parameters(prices, node=node):
                p = node.parameters.parameter_template(price_edges)
                p[0] = input_builder.price_intervals((price_edges,prices))
                return p
            supports.append(SimpleNamespace(model=model,
                x=.5*(origin.x_edges[:-1]+origin.x_edges[1:]), dx=dx,
                sigma_x=node.parameters.sigma_x, interval_edges=np.array(price_edges),
                solver_t_eval=times, midpoints=midpoints, widths=np.diff(edges_t),
                target=np.full(pieces,.5), solver_max_step=numerics.solver_max_step,
                breakpoints=np.array(price_edges[1:-1]),
                validate_prices=lambda p:np.asarray(p,float), parameters_for_prices=parameters))
        e.supports = tuple(supports); e.projector = None
        if backend == 'mixture':
            e.projector = SimpleNamespace(dx=dx, basis=origin.basis,
                projection_solver=ProjectionStub(origin.basis,dx))
            def project(density,initial_coefficients):
                trial = initial_coefficients.copy(); trial[0]+=.0001; trial[-1]-=.0001
                solver = e.projector.projection_solver
                c = solver.solve(density*np.sqrt(dx),trial)
                solver.optimality_gap(density*np.sqrt(dx),c)
                projected = c@origin.basis
                return dict(coefficients=c,density=projected,
                    diagnostics=dict(cell_mass_l1_error=float(np.sum(abs(projected-density))*dx)))
            e.projector.project = project
        def forward(prices,index):
            counts['forward']+=1
            events.append(dict(kind='forward',time=origin.time,index=index,backend=backend,
                cells=cells,origin_sha256=digest(origin.to_record())))
            s=e.supports[index]; mass=origin.conditional_mass[index]
            c=None if backend=='fv' else origin.conditional_coefficients[index]
            if c is not None:
                solver=e.projector.projection_solver
                c=solver.solve(mass/np.sqrt(dx),c)
                solver.optimality_gap(mass/np.sqrt(dx),c)
            v=variance(prices)
            return dict(index=index,prices=np.array(prices),times=times,
                mass=np.tile(mass,(len(times),1)),
                coefficients=None if c is None else np.tile(c,(len(times),1)),
                expected_demand=np.full(pieces,.5),second_moment=np.full(pieces,.25+v),
                node_total=float(sum(s.widths)*v), diagnostics=dict(stub=True))
        e.evaluate_node=forward
        return e

    def likelihood(model,parameters,sigma_y,state_mass,*args,**kwargs):
        counts['likelihood']+=1; origin=model._stub_origin; j=model._stub_index
        events.append(dict(kind='likelihood',time=args[1],index=j))
        raw=np.array(state_mass).copy()
        if origin.backend=='mixture':
            c=origin.conditional_coefficients[j].copy(); c[0]+=.0001; c[-1]-=.0001
            raw=c@origin.basis*np.diff(origin.x_edges)
        return BayesUpdate(mass_to_log(raw),.02*j)

    def deterministic(e, prices):
        counts['ode']+=1; prices=np.asarray(prices)
        events.append(dict(kind='ode',time=12))
        return SimpleNamespace(total=.2-.02*contrast(prices),prices=prices,
                               state=np.full(len(e.solver_t_eval),e.initial_state),
                               demand=np.full(len(e.solver_t_eval),.5))

    def em(model,initial,parameters,sigma_x,*,t_eval,brownian_increments,**kwargs):
        counts['em']+=1
        prices=parameters[0](np.array([15.,21.,27.,33.]))
        demand=np.broadcast_to(.5+np.sqrt(variance(prices)),(256,len(t_eval)))
        return (np.array(t_eval),None,None,None,None,None,None,demand,
                dict(total_violations=0))

    def powell(objective, starts, **kwargs):
        counts['powell']+=1
        assert kwargs['options']['maxfev']==24 and kwargs['lower']==0 and kwargs['upper']==1
        initial=np.array(starts[0]); objective(initial)
        objective(np.array([.7,.3,.7,.3]))
        return SimpleNamespace(best=SimpleNamespace(message='stub completion'))

    Guard=guarded_store_class(NodeJobStore,
        native_source_path=root/'src/method/bm_1_g7_jobs.py',
        native_source_sha256=manifest['src/method/bm_1_g7_jobs.py'])
    store=Guard(output/'fixture_campaign',prior_debit=0,ceiling=300,max_new_seconds=300)
    reservations={name:1. for name in COUNTS}
    checks=[]
    with patch.object(day,'OwnedBankPriceEvaluator',evaluator), \
         patch.object(phase,'OwnedBankPriceEvaluator',evaluator), \
         patch.object(native,'OwnedBankPriceEvaluator',evaluator), \
         patch.object(native,'bm1_observation_update',likelihood), \
         patch.object(DeterministicPriceEvaluator,'evaluate',deterministic), \
         patch.object(bm_1,'euler_maruyama_solver',em), \
         patch.object(day,'multistart_powell',powell):
        result=day.run_day(plan,records,store,source_root=root,manifest=manifest,
                          reservations=reservations,output=output)
        assert result['numerical_validation_passed'] and not result['scientific_accept']
        assert set(result['costs'])=={'neutral','deterministic','stochastic'}
        assert all(set(v)==set(phase.CONFIGS)|{'quadrature'} for v in result['costs'].values())
        assert counts['powell']==4 and counts['em']==30 and counts['likelihood']==20
        checks.append('actual full graph: two searches, three policies, five crosscontexts, coupled EM, four updates and continuation')
        first=next(i for i,v in enumerate(events) if v['kind']=='likelihood')
        assert not any(v['kind']=='ode' or (v['kind']=='forward' and v['time']==12) for v in events[first:])
        assert any(v['kind']=='forward' and v['time']==36 for v in events[first:])
        checks.append('observation arrives after search/cross/EM; next forwards use cutoff36')
        assert result['feedback']['selected']==1
        assert all(v['prices']==[.3,.7] for v in result['feedback']['forwards'].values())
        assert all(c['passed'] for c in result['feedback_checks'].values())
        assert all(c['passed'] for c in result['continuation_checks'].values())
        checks.append('updated coarse M254 chooses next price; all four own posterior laws pass actual comparison algebra')
        packet=json.loads((output/'generated_feedback_inputs.json').read_bytes())
        name='m254_coarse'; saved=packet['forwards'][name]
        control=FeedbackBankControl(evaluator(origins[name],horizon=(12,36),
            price_edges=[12,18,24,30,36],numerics=BM1ControlNumerics(x_steps=257),
            backend='mixture'),store,source_root=root,source_sha256=manifest)
        before=counts.copy(); old_bytes={p.name:p.read_bytes() for p in (store.root/'nodes').glob('*.json')}
        restored=control.evaluate(saved['prices'],seconds_per_node=.2)
        update=control.assimilate(restored,packet['observation']['value'],
            seconds=1,forward_receipts=saved['receipts'])
        assert counts==before
        assert update['job_id']==result['feedback']['posterior_jobs'][name]
        assert not np.array_equal(update['origin'].log_weights,origins[name].log_weights)
        assert not np.array_equal(update['origin'].conditional_coefficients,origins[name].conditional_coefficients)
        assert all((store.root/'nodes'/p).read_bytes()==raw for p,raw in old_bytes.items())
        with store.lock(): audit_store(store,full=True)
        checks.append('completed forward/update reuse changes no kernel counts or saved bytes; raw-evidence weights and own coefficients survive')
        # Pairing itself receives durable pathcost receipts.
        reference=[]
        fine_contract=day.make_control(origins['m254_fine'],'m254_fine',store,root,manifest).contract_id
        for j in range(5):
            job=dict(kind='g7-day-coupled-em-v1',contract=fine_contract,
                index=j,prices=result['policies']['neutral'],entropy=20261008,roles=[1,2],
                paths=256,coarse_step=1/192,fine_step=1/384,boundary='none')
            reference.append(store.receipt(store.root/'nodes'/(digest(job)+'.json'),job))
        changed=copy.deepcopy(reference); changed[0]['input_sha256']='0'*64
        try: paired_improvement(reference,changed,origins['m254_fine'].weights)
        except ValueError: pass
        else: raise AssertionError('Unpaired paths accepted')
        checks.append('paired MC rejects different saved Brownian/startstate input identity')
    unknown=price_plan(plan,{},prior_debit=1851.3372718589885,ceiling=1910,
        source_fingerprint=std_digest(manifest),origin_fingerprint=std_digest(records),
        versions=dict(numpy=np.__version__,scipy=__import__('scipy').__version__),reserve_seconds=2)
    assert not unknown['budget_admitted'] and unknown['estimated_seconds'] is None and len(unknown['unknown'])==16
    checks.append('all sixteen scientific rates remain unknown; stub timings cannot admit the numerical graph')
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==v for p,v in manifest.items())
    checks.append('all registered native and helper source bytes unchanged')
    report=dict(schema='g7-full-graph-stub-check-v1',passed=True,checks=checks,
        body_wall_seconds=perf_counter()-started,stub_calls=counts,
        native_transport_forwards=0,native_ode_solves=0,native_em_evolutions=0,
        native_likelihood_evaluations=0,native_projection_solves=0,
        scientific_accept=False,numerical_scientific_validation=False,
        fixture_graph_completed=True,actual_full_day_experiment_completed=False,
        note='Four/five fixture midpoint intervals; numeric kernels and Powell are stubs. Real orchestration, sampling/coupling, audit/store, Bayes weight algebra and native comparisons.')
    atomic_json(output/'check_result.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
