"""G8 adapters over unchanged BM1 filters, parameter banks and F3 objectives.

A frozen bank owns each node's conditional law. The observation contract is
point demand in hours; historical interval-energy likelihoods are separate.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from time import perf_counter

import numpy as np

from src.control.bm_1 import BM1Parameters
from src.control.bm_1_handoff import BM1ForecastOrigin
from src.control.objective import stochastic_tracking_objective
from src.method.beta_mixture import PreparedBetaProjector
from src.method.bm_1_distribution_panel import build_problem, baseline, target
from src.method.bm_1_estimation import run_parameter_bank, run_fv_filter, run_mixture_filter
from src.model.bm_1 import bm_1
from src.params.bm_1 import BM1ObservationSeries
from src.params.bm_1_response import BM1PiecewiseResponse
from src.validation import estimation as checks

RHO_VALUES = np.array([.125, .1875, .25, .375, .5])
SIGMA_Y = .03


class BudgetBM1(bm_1):
    """A deadline check before drift evaluation; numerical equations unchanged."""
    def __init__(self, *args, deadline=None):
        super().__init__(*args)
        self.deadline = deadline

    def _drift(self, time, state, parameters):
        if self.deadline is not None and perf_counter() >= self.deadline:
            raise RuntimeError('G8 computation budget exhausted.')
        return super()._drift(time, state, parameters)


class RecordingProjector(PreparedBetaProjector):
    """Retain initial/Bayes projection diagnostics; no additional projection."""
    def __init__(self, basis, dx):
        super().__init__(basis, dx)
        self.records = []

    def project(self, density, initial_coefficients=None):
        result = super().project(density, initial_coefficients)
        self.records.append(dict(result['diagnostics']))
        return result


def context(cells, degree=None, *, deadline=None, basis_kind=None):
    model = BudgetBM1(baseline, target, (0., 24.), 2, deadline=deadline)
    fixed = build_problem().parameters
    parameters = fixed.parameter_template([0., 6., 12., 18., 24.])
    from src.tools.input_generator.input_builder import input_builder
    parameters[0] = input_builder.price_intervals(([0.,6.,12.,18.,24.],[.3,.7,.3,.7]))
    edges, grid, dx = model.fokker_planc_reference.grid(cells, bounds=(0.,1.))
    from src.control.bm_1 import beta_initial_cell_mass
    initial = beta_initial_cell_mass(edges)
    projector = None
    if basis_kind is not None:
        from src.method.bm_1_g8_basis import BASIS_KIND, localized_beta_basis
        from src.method.bm_1_g8_bernstein import BASIS_KIND as M254, bernstein_basis, prior_coefficients
        if degree is not None or basis_kind not in (BASIS_KIND, M254):
            raise ValueError('Declare one supported G8 basis, without a Bernstein degree.')
        basis = bernstein_basis(edges) if basis_kind == M254 else localized_beta_basis(edges)
        projector = RecordingProjector(basis, dx)
        if basis_kind == M254:
            projector.initial_coefficients = prior_coefficients()
            represented = projector.initial_coefficients @ basis * dx
            if not np.allclose(represented, initial, atol=1e-10, rtol=0.):
                raise ValueError('Degree-elevated prior differs from original cell prior.')
    elif degree is not None:
        basis = model.fokker_planc._normalise_basis_discretely(
            model.fokker_planc.beta_basis(grid, degree), dx)
        projector = RecordingProjector(basis, dx)
    return model, fixed, parameters, edges, initial, projector


def known_filter(cells, observations, step, *, degree=None, deadline=None, basis_kind=None):
    """Use the existing whole-record state-filter API; retain its diagnostics."""
    if observations.time_unit != 'hour':
        raise ValueError('G8 point observations must explicitly use hours.')
    model, fixed, parameters, edges, initial, projector = context(
        cells, degree, deadline=deadline, basis_kind=basis_kind)
    options = dict(breakpoints=[6.,12.,18.], max_step=step, likelihood_quadrature_order=3)
    if projector is None:
        result = run_fv_filter(model, parameters, fixed.sigma_x, SIGMA_Y,
                               initial, 0., observations, **options)
    else:
        result = run_mixture_filter(model, parameters, fixed.sigma_x, SIGMA_Y,
                                    initial, 0., observations, projector=projector, cfl=.4,
                                    initial_coefficients=getattr(projector,'initial_coefficients',None), **options)
    return result, model, parameters, projector


@dataclass(frozen=True)
class G8BankOrigin:
    """Owned full rho/state posterior at one cutoff, not a MAP plug-in.

    Responses must be serializable immutable G curves. Future baseline/price
    plans are explicit caller inputs; no future observations belong here.
    """
    time: float
    fixed_parameters: BM1Parameters
    rho_values: object
    log_weights: object
    x_edges: object
    conditional_mass: object
    sigma_y: float = SIGMA_Y
    time_unit: str = 'hour'
    backend: str = 'fv'
    conditional_coefficients: object = None
    basis: object = None
    price_side: str = 'right'
    basis_kind: str | None = None
    basis_identity_sha256: str | None = None

    @staticmethod
    def basis_identity(edges, basis):
        """Exact owned float64 bytes; independent of a fresh special-function call."""
        return hashlib.sha256(np.asarray(edges,dtype='<f8').tobytes()+
                              np.asarray(basis,dtype='<f8').tobytes()).hexdigest()

    def __post_init__(self):
        time = checks.finite_scalar(self.time, 'cutoff')
        if self.basis_kind is not None:
            from src.method.bm_1_g8_bernstein import BASIS_KIND, bernstein_basis
            reference=bernstein_basis(np.asarray(self.x_edges))
            exact=np.array_equal(self.basis,reference)
            certified=False
            if self.basis_identity_sha256 is not None:
                if np.iscomplexobj(self.basis):
                    raise ValueError('Certified basis must be real.')
                owned=np.asarray(self.basis,dtype=float)
                widths=np.diff(np.asarray(self.x_edges,dtype=float))
                certified=(owned.shape==reference.shape and np.isfinite(owned).all()
                    and self.basis_identity_sha256==self.basis_identity(self.x_edges,owned)
                    and np.max(np.abs(owned-reference)@widths)<=64*np.finfo(float).eps)
                if not certified:
                    raise ValueError('Owned basis bytes or roundoff-only family certificate changed.')
            if (self.backend != 'mixture' or self.basis_kind != BASIS_KIND
                    or not (exact or certified)):
                raise ValueError('Typed M254 checkpoint must carry its declared exact basis.')
        elif self.basis_identity_sha256 is not None:
            raise ValueError('Basis byte certificate requires the typed supervisor family.')
        if self.time_unit != 'hour' or self.price_side not in ('left','right'):
            raise ValueError('Declare G8 hours and observation price side.')
        if not isinstance(self.fixed_parameters, BM1Parameters):
            raise TypeError('fixed_parameters must be BM1Parameters.')
        for response in (self.fixed_parameters.state_response, self.fixed_parameters.price_response):
            if not isinstance(response, BM1PiecewiseResponse):
                raise TypeError('G8 origins require serializable immutable response curves.')
        rho = checks.finite_vector(self.rho_values, 'rho nodes')
        logs = checks.log_probability_mass(self.log_weights, 'node log weights')
        edges = checks.increasing_grid(self.x_edges, 'cell edges')
        mass = checks.probability_rows(self.conditional_mass, 'node conditional masses')
        if (np.any(rho <= 0.) or len(np.unique(rho)) != len(rho) or logs.size != rho.size
                or mass.shape != (rho.size, edges.size-1) or edges[0] != 0. or edges[-1] != 1.):
            raise ValueError('Nodes, weights and conditional laws must share a [0,1] grid.')
        sigma_y = checks.finite_scalar(self.sigma_y, 'sigma_y', positive=True)
        coefficients, basis = None, None
        if self.backend == 'mixture':
            coefficients = checks.probability_rows(self.conditional_coefficients, 'node coefficients')
            if np.iscomplexobj(self.basis):
                raise ValueError('Mixture basis must be real.')
            basis = np.asarray(self.basis, dtype=float).copy()
            widths = np.diff(edges)
            if (basis.shape != (coefficients.shape[1],mass.shape[1])
                    or coefficients.shape[0] != rho.size or not np.all(np.isfinite(basis))
                    or np.any(basis < 0.) or not np.allclose(widths,widths[0],atol=1e-14,rtol=1e-12)
                    or not np.allclose(basis @ widths,1.,atol=1e-10,rtol=0.)
                    or not np.allclose(coefficients @ basis * widths,mass,atol=1e-10,rtol=0.)):
                raise ValueError('Mixture masses must match owned normalized basis and coefficients.')
        elif self.backend != 'fv' or self.basis is not None or self.conditional_coefficients is not None:
            raise ValueError('FV origins carry masses only; declare fv or mixture.')
        for array in (rho,logs,edges,mass,coefficients,basis):
            if array is not None:
                array.setflags(write=False)
        for name,value in [('time',time),('rho_values',rho),('log_weights',logs),
                           ('x_edges',edges),('conditional_mass',mass),('sigma_y',sigma_y),
                           ('conditional_coefficients',coefficients),('basis',basis)]:
            object.__setattr__(self,name,value)

    @property
    def weights(self):
        return np.exp(self.log_weights)

    def node_origins(self):
        return tuple(BM1ForecastOrigin(self.time,self.time_unit,
                     replace(self.fixed_parameters,C=1./rho),self.sigma_y,self.x_edges,mass)
                     for rho,mass in zip(self.rho_values,self.conditional_mass))

    def map_origin(self):
        index = int(np.argmax(self.log_weights))
        node = self.node_origins()[index]
        return BM1ForecastOrigin(node.time,node.time_unit,node.parameters,node.sigma_y,
                                  node.x_edges,node.posterior_mass,'rho_map',index)

    def to_record(self):
        p = self.fixed_parameters
        record=dict(schema='g8-rho-origin-v1',time=self.time,time_unit=self.time_unit,
                    observation_kind='point',price_side=self.price_side,backend=self.backend,basis_kind=self.basis_kind,
                    parameters=dict(C=p.C,k=p.k,Xi=p.Xi,sigma_x=p.sigma_x,
                        f=p.state_response.to_record(),g=p.price_response.to_record()),
                    sigma_y=self.sigma_y,rho_values=self.rho_values.tolist(),
                    log_weights=[None if np.isneginf(v) else float(v) for v in self.log_weights],
                    x_edges=self.x_edges.tolist(),conditional_mass=self.conditional_mass.tolist(),
                    conditional_coefficients=None if self.conditional_coefficients is None else self.conditional_coefficients.tolist(),
                    basis=None if self.basis is None else self.basis.tolist())
        if self.basis_identity_sha256 is not None:
            record['basis_identity_sha256']=self.basis_identity_sha256
        return record

    @classmethod
    def from_verified_bytes(cls, payload, expected_sha256):
        """Load registered original bytes without replacing their numerical basis.

        The default loader remains bit-exact. This explicit portability path
        verifies the externally registered whole-record hash, owned basis bytes,
        and a 64-epsilon component-mass family bound, before ordinary validation.
        """
        if hashlib.sha256(payload).hexdigest()!=expected_sha256:
            raise ValueError('Registered whole-origin bytes changed.')
        record=json.loads(payload)
        from src.method.bm_1_g8_bernstein import BASIS_KIND
        if record.get('basis_kind')!=BASIS_KIND:
            raise ValueError('Verified portability loader requires supervisor M254.')
        record['basis_identity_sha256']=cls.basis_identity(record['x_edges'],record['basis'])
        return cls.from_record(record)

    @classmethod
    def from_record(cls, record):
        if record.get('schema') != 'g8-rho-origin-v1' or record.get('observation_kind') != 'point':
            raise ValueError('Unsupported G8 origin or observation contract.')
        p = record['parameters']
        fixed = BM1Parameters(BM1PiecewiseResponse.from_record(p['f']),
                              BM1PiecewiseResponse.from_record(p['g']),p['C'],p['k'],p['Xi'],p['sigma_x'])
        return cls(record['time'],fixed,record['rho_values'],
                   [-np.inf if v is None else v for v in record['log_weights']],
                   record['x_edges'],record['conditional_mass'],record['sigma_y'],record['time_unit'],
                   record['backend'],record['conditional_coefficients'],record['basis'],record['price_side'],record.get('basis_kind'),
                   record.get('basis_identity_sha256'))


def bank_filter(cells, observations, step, *, nodes=RHO_VALUES, degree=None,
                origin=None, deadline=None, on_step=None, basis_kind=None, configure_projector=None):
    """Public bank API one new observation at a time, retaining node histories.

    Continuation passes each conditional mass AND mixture coefficient row.
    This avoids both losing node history and reprojecting a saved posterior.
    """
    if observations.time_unit != 'hour':
        raise ValueError('G8 observations use hours.')
    model,fixed,parameters,edges,initial,projector = context(
        cells,degree,deadline=deadline,basis_kind=basis_kind)
    if configure_projector is not None:
        if projector is None:
            raise ValueError('Projector instrumentation requires a mixture backend.')
        configure_projector(projector)
    nodes = checks.finite_vector(nodes,'rho nodes')
    logs = np.full(nodes.size,-np.log(nodes.size))
    current, coefficients, time = None,None,0.
    typed_kind = None
    if projector is not None and hasattr(projector, 'initial_coefficients'):
        typed_kind = basis_kind
        coefficients = np.tile(projector.initial_coefficients,(nodes.size,1))
        current = coefficients @ projector.basis * np.diff(edges)
    if origin is not None:
        expected = 'fv' if projector is None else 'mixture'
        if (not isinstance(origin,G8BankOrigin) or origin.backend != expected
                or not np.array_equal(origin.rho_values,nodes) or not np.array_equal(origin.x_edges,edges)
                or origin.price_side != observations.price_side or origin.sigma_y != SIGMA_Y
                or origin.fixed_parameters.state_response.to_record() != fixed.state_response.to_record()
                or origin.fixed_parameters.price_response.to_record() != fixed.price_response.to_record()
                or (origin.fixed_parameters.C,origin.fixed_parameters.k,origin.fixed_parameters.Xi,origin.fixed_parameters.sigma_x)
                   != (fixed.C,fixed.k,fixed.Xi,fixed.sigma_x)
                or origin.basis_kind != typed_kind
                or (projector is not None and not np.array_equal(origin.basis,projector.basis))):
            raise ValueError('Checkpoint and G8 model/nodes/grid/observation contract differ.')
        logs,current,coefficients,time = origin.log_weights,origin.conditional_mass,origin.conditional_coefficients,origin.time
    records = []
    for when,value,present in zip(observations.times,observations.values,observations.observed):
        if deadline is not None and perf_counter() >= deadline:
            raise RuntimeError('G8 computation budget exhausted.')
        one = BM1ObservationSeries([when],[value],'hour',[present],price_side=observations.price_side)
        count = 0 if projector is None else len(projector.records)
        initialized = coefficients is not None
        result = run_parameter_bank(model,parameters,fixed.sigma_x,SIGMA_Y,initial,time,one,
                    coordinate='rho',values=nodes,initial_log_weights=logs,
                    backend='fv' if projector is None else 'mixture',projector=projector,
                    breakpoints=[6.,12.,18.],max_step=step,cfl=.4,likelihood_quadrature_order=3,
                    initial_conditional_mass=current,initial_conditional_coefficients=coefficients)
        current = result['final_conditional_mass']
        coefficients = result['final_conditional_coefficients']
        old_logs,logs,time = logs,result['log_weights'][-1],float(when)
        new = [] if projector is None else projector.records[count:]
        initial_count = 0 if projector is None or initialized else nodes.size
        posterior_projection = new[initial_count:]
        if projector is not None and len(posterior_projection) != (nodes.size if present else 0):
            raise RuntimeError('Projection audit does not match node update count.')
        snapshot = G8BankOrigin(time,fixed,nodes,logs,edges,current,backend='fv' if projector is None else 'mixture',
                        conditional_coefficients=coefficients,basis=None if projector is None else projector.basis,
                        price_side=observations.price_side,basis_kind=typed_kind)
        record = dict(time=time,observed=bool(present),prior_log_weights=np.array(old_logs),
                      log_weights=np.array(logs),log_evidence=result['log_evidence'][0],
                      conditional_log_evidence=result['conditional_log_evidence'][0],
                      predicted_mass=result['predicted_mass'][0],posterior_mass=result['posterior_mass'][0],
                      conditional_mass=current.copy(),
                      conditional_coefficients=None if coefficients is None else coefficients.copy(),
                      initial_projection=new[:initial_count],posterior_projection=posterior_projection,
                      origin=snapshot)
        records.append(record)
        if on_step is not None:
            on_step(record)
    return records,model,parameters,projector


def weighted_control_evaluation(origin, *, baseline, target, horizon, price_edges, prices, numerics):
    """Fixed-weight full-bank FV objective; no optimizer or future y argument.

    Mixture origins are conservatively transferred as their represented cell
    laws to FV. This is a declared reference evaluation, not mixture control.
    """
    if not isinstance(origin,G8BankOrigin):
        raise TypeError('A full G8 bank origin is required.')
    evaluations,midpoints,widths,targets = [],None,None,None
    for node in origin.node_origins():
        problem = node.control_problem(baseline=baseline,target=target,horizon=horizon,price_edges=price_edges)
        evaluator = problem.build_evaluator(numerics,fast=False)
        evaluations.append(evaluator.evaluate(prices))
        midpoints,widths,targets = evaluator.midpoints,evaluator.widths,evaluator.target
    weights = origin.weights
    means = np.stack([e.objective.expected_demand for e in evaluations])
    seconds = np.stack([e.objective.second_moment for e in evaluations])
    mean = weights @ means
    second = weights @ seconds
    objective = stochastic_tracking_objective(widths,mean,second,targets)
    node_costs = np.array([e.total for e in evaluations])
    between = np.sum(weights[:,None]*(means-mean)**2,axis=0)
    within = weights @ np.stack([e.objective.demand_variance for e in evaluations])
    if abs(objective.total-float(weights @ node_costs)) > 1e-10:
        raise RuntimeError('Full-bank and weighted node tracking objectives differ.')
    return dict(total=objective.total,bias=objective.bias,variance=objective.variance,
                within_parameter_variance=float(widths @ within),
                between_parameter_variance=float(widths @ between),
                node_costs=node_costs,weights=weights,midpoints=midpoints,
                expected_demand=mean,demand_variance=objective.demand_variance,
                parameter_uncertainty_preserved=True,
                law_representation='owned conditional cell laws, FV forward evaluation')
