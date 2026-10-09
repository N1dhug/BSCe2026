"""G-only owned Bernstein posterior -> price forward -> point update.

No new FP/LS/Bayes kernel. Every price candidate starts from the same frozen
conditional coefficients. Geometry caches never carry a candidate's state.
"""

import numpy as np, hashlib, json, os, scipy
from src.control.bm_1 import BM1ControlNumerics
from src.control import objective as tracking
from src.method.beta_mixture import PreparedBetaProjector, evolve_projected_density
from src.method.bm_1_estimation import bm1_observation_update, parameter_bank_weights
from src.method.bm_1_g8 import G8BankOrigin, BERNSTEIN_BASIS_KIND as BASIS_KIND
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from src.method.bm_1_distribution_panel import baseline, target


class OwnedBankPriceEvaluator:
    """Fixed-weight single-shooting evaluation from an explicit M254 origin.

    The original DynamicMixturePriceEvaluator retains its legacy basis.
    This adapter supplies owned basis/c to the same dynamic transport kernel.
    Future observations are accepted only by the separate assimilate method.
    """
    def __init__(self,origin,*,baseline,target,horizon,price_edges,numerics,
                 backend='mixture',configure_projector=None):
        if not isinstance(origin,G8BankOrigin) or origin.time_unit!='hour':
            raise TypeError('An owned G8 hourly bank origin is required.')
        if backend not in ('mixture','fv'):
            raise ValueError('Declare mixture or independent FV reference.')
        if not isinstance(numerics,BM1ControlNumerics) or numerics.x_steps!=origin.conditional_mass.shape[1]:
            raise ValueError('Explicit same-grid numerics required; no implicit regridding.')
        if backend=='mixture' and (origin.backend!='mixture' or origin.basis_kind!=BASIS_KIND):
            raise ValueError('Primary G7 requires the typed cell-integrated supervisor M254 basis.')
        self.origin=origin;self.backend=backend
        self.supports=tuple(node.control_problem(baseline=baseline,target=target,
             horizon=horizon,price_edges=price_edges).build_evaluator(numerics,fast=False)
             for node in origin.node_origins())
        if any(not np.array_equal(s.model.fokker_planc_reference.grid(numerics.x_steps)[0],origin.x_edges)
               for s in self.supports):
            raise ValueError('Control grid differs from the owned checkpoint.')
        self.projector=None
        if backend=='mixture':
            self.projector=PreparedBetaProjector(origin.basis,np.diff(origin.x_edges)[0])
            if configure_projector is not None:configure_projector(self.projector)
        elif configure_projector is not None:
            raise ValueError('FV has no mixture projector.')

    def evaluate_node(self,prices,index):
        """One unchanged native forward from this node's own frozen law."""
        prices=self.supports[0].validate_prices(prices).copy()
        if isinstance(index,bool) or int(index)!=index or not 0<=index<len(self.supports):
            raise ValueError('Valid integer node index required.')
        j=int(index);s=self.supports[j]
        parameters=s.parameters_for_prices(prices)
        if self.backend=='mixture':
            evolved=evolve_projected_density(method=s.model.fokker_planc,
                t_eval=s.solver_t_eval,breakpoints=s.breakpoints,x=s.x,dx=s.dx,
                basis=self.origin.basis,initial_coefficients=self.origin.conditional_coefficients[j].copy(),
                drift=lambda t,x:s.model._drift(t,x,parameters),
                diffusion=lambda t,x:s.model.diffusion_coefficient(x,s.sigma_x),
                max_step=s.solver_max_step,cfl=.4,
                projection_data=self.projector.projection_data,
                projection_solver=self.projector.projection_solver)
            mass=evolved['density']*s.dx
            coefficients=evolved['coefficients'];diagnostics=evolved['diagnostics']
            demand=tracking.demand_grid(s.model,parameters,s.midpoints,s.x)
            mean,second,_=tracking.discrete_demand_moments(demand,mass[1:-1])
            obj=tracking.stochastic_tracking_objective(s.widths,mean,second,s.target)
        else:
            result=s.evaluate(prices);mass=result.cell_mass;obj=result.objective
            coefficients=None;diagnostics=result.diagnostics
        return dict(index=j,prices=prices,times=s.solver_t_eval,
                    mass=mass,coefficients=coefficients,diagnostics=diagnostics,
                    expected_demand=obj.expected_demand,second_moment=obj.second_moment,
                    node_total=obj.total)

    def combine_nodes(self,prices,nodes):
        """Reduce complete node receipts with the original fixed bank weights."""
        prices=self.supports[0].validate_prices(prices).copy()
        if len(nodes)!=len(self.supports):
            raise ValueError('All owned node receipts are required before bank reduction.')
        masses=[];coefficients=[];objectives=[];diagnostics=[]
        for j,(s,node) in enumerate(zip(self.supports,nodes)):
            if (node['index']!=j or not np.array_equal(node['prices'],prices)
                    or not np.array_equal(node['times'],s.solver_t_eval)):
                raise ValueError('Node index, candidate or integration partition differs.')
            mass=np.asarray(node['mass'],float)
            c=None if node['coefficients'] is None else np.asarray(node['coefficients'],float)
            if (mass.shape!=(len(s.solver_t_eval),len(s.x)) or not np.isfinite(mass).all()
                    or np.min(mass)<-1e-12 or not np.allclose(mass.sum(axis=1),1.,atol=1e-9,rtol=0.)
                    or not np.allclose(mass[0],self.origin.conditional_mass[j],atol=1e-10,rtol=0.)):
                raise ValueError('Node mass does not match its own origin or valid full trajectory.')
            if self.backend=='mixture':
                if (c is None or c.shape!=(len(s.solver_t_eval),self.origin.basis.shape[0])
                        or not np.isfinite(c).all() or not np.array_equal(c[0],self.origin.conditional_coefficients[j])
                        or not np.allclose(c@self.origin.basis*s.dx,mass,atol=1e-10,rtol=0.)):
                    raise ValueError('Node coefficients do not own the declared mixture trajectory.')
                coefficients.append(c)
            elif c is not None:
                raise ValueError('FV receipts must not carry mixture coefficients.')
            obj=tracking.stochastic_tracking_objective(s.widths,np.asarray(node['expected_demand']),
                                                       np.asarray(node['second_moment']),s.target)
            if abs(obj.total-node['node_total'])>1e-10:
                raise ValueError('Stored node cost differs from its full demand moments.')
            masses.append(mass);objectives.append(obj);diagnostics.append(node['diagnostics'])
        weights=self.origin.weights
        means=np.stack([o.expected_demand for o in objectives])
        second=np.stack([o.second_moment for o in objectives])
        s=self.supports[0]
        objective=tracking.stochastic_tracking_objective(s.widths,weights@means,weights@second,s.target)
        within=weights@np.stack([o.demand_variance for o in objectives])
        between=np.sum(weights[:,None]*(means-objective.expected_demand)**2,axis=0)
        costs=np.array([o.total for o in objectives])
        if abs(objective.total-float(weights@costs))>1e-10:
            raise RuntimeError('Bank objective differs from weighted node costs.')
        if np.max(abs(objective.demand_variance-within-between))>1e-10:
            raise RuntimeError('Law of total variance failed.')
        mass=np.stack(masses);coefficients=np.stack(coefficients) if coefficients else None
        predicted=G8BankOrigin(s.model.T[1],self.origin.fixed_parameters,self.origin.rho_values,
            self.origin.log_weights,self.origin.x_edges,mass[:,-1],sigma_y=self.origin.sigma_y,
            backend=self.backend,conditional_coefficients=None if coefficients is None else coefficients[:,-1],
            basis=self.origin.basis if self.backend=='mixture' else None,
            price_side=self.origin.price_side,basis_kind=BASIS_KIND if self.backend=='mixture' else None,
            basis_identity_sha256=self.origin.basis_identity_sha256 if self.backend=='mixture' else None)
        return dict(total=objective.total,bias=objective.bias,variance=objective.variance,
            within_parameter_variance=float(s.widths@within),between_parameter_variance=float(s.widths@between),
            expected_demand=objective.expected_demand,demand_variance=objective.demand_variance,
            within_variance=within,between_variance=between,node_costs=costs,weights=weights.copy(),
            times=s.solver_t_eval,midpoints=s.midpoints,widths=s.widths,target=s.target,
            conditional_mass=mass,conditional_coefficients=coefficients,prices=prices,
            diagnostics=diagnostics,predicted_origin=predicted,backend=self.backend,_owner=self)

    def evaluate(self,prices):
        return self.combine_nodes(prices,[self.evaluate_node(prices,j) for j in range(len(self.supports))])

    def __call__(self,prices):return self.evaluate(prices)['total']

    def assimilate(self,evaluation,value,*,observed=True):
        """Point observation at the control endpoint, under its actual price.

        No second propagation, no historical future y, no original fixture-price
        restart. State evidence is computed before its mixture LS projection.
        """
        if evaluation.get('_owner') is not self:
            raise ValueError('Assimilation requires this evaluator\'s own price forward.')
        predicted=evaluation['predicted_origin']
        if (predicted.backend!=self.backend or predicted.time!=self.supports[0].model.T[1]
            or not np.array_equal(predicted.log_weights,self.origin.log_weights)):
            raise ValueError('Evaluation does not match this frozen bank/horizon.')
        if not isinstance(observed,(bool,np.bool_)):
            raise ValueError('observed must be boolean.')
        scores=[];mass=[];coefficients=[];projections=[]
        for j,s in enumerate(self.supports):
            c=None if predicted.conditional_coefficients is None else predicted.conditional_coefficients[j]
            if observed:
                parameters=s.parameters_for_prices(evaluation['prices'])
                update=bm1_observation_update(s.model,parameters,self.origin.sigma_y,
                    predicted.conditional_mass[j],self.origin.time,predicted.time,float(value),
                    price_side=self.origin.price_side,likelihood_quadrature_order=3)
                scores.append(update.log_evidence);m=update.mass
                if self.backend=='mixture':
                    projection=self.projector.project(m/s.dx,initial_coefficients=c)
                    c=projection['coefficients'];m=projection['density']*s.dx
                    projections.append(projection['diagnostics'])
            else:
                scores.append(np.nan);m=predicted.conditional_mass[j].copy()
            mass.append(m)
            if c is not None:coefficients.append(c)
        weight_update=parameter_bank_weights(np.array([scores]),self.origin.log_weights,np.array([observed]))
        posterior=G8BankOrigin(predicted.time,predicted.fixed_parameters,predicted.rho_values,
            weight_update['log_weights'][0],predicted.x_edges,np.array(mass),sigma_y=predicted.sigma_y,
            backend=self.backend,conditional_coefficients=np.array(coefficients) if coefficients else None,
            basis=predicted.basis,price_side=predicted.price_side,basis_kind=predicted.basis_kind,
            basis_identity_sha256=predicted.basis_identity_sha256)
        return dict(origin=posterior,conditional_log_evidence=np.array(scores),
                    log_evidence=float(weight_update['log_evidence'][0]),projections=projections)

REQUIRED_SOURCE_PATHS = (
    'src/method/beta_mixture.py',
    'src/control/objective.py',
    'src/validation/model/bm.py',
    'src/model/base/bm.py',
    'src/validation/method/euler_maruyama.py',
    'src/method/euler_maruyama.py',
    'src/validation/solver/active_set.py',
    'src/solver/active_set.py',
    'src/validation/solver/finite_volume.py',
    'src/solver/finite_volume.py',
    'src/validation/method/fokker_planc.py',
    'src/method/fokker_planc.py',
    'src/validation/model/bm_1.py',
    'src/validation/tools/input_generator/input_builder.py',
    'src/tools/input_generator/input_builder.py',
    'src/validation/method/objective.py',
    'src/method/objective.py',
    'src/model/bm_1.py',
    'src/validation/control.py',
    'src/control/bm_1.py',
    'src/validation/estimation.py',
    'src/params/bm_1.py',
    'src/method/bm_1_estimation.py', 'src/method/bm_1_observation.py',
    'src/control/bm_1_handoff.py',
    'src/params/bm_1_deterministic.py',
    'src/method/bm_1_deterministic_fit.py',
    'src/params/bm_1_response.py',
    'src/method/bm_1_deterministic_response.py',
    'src/method/bm_1_linear_noise.py',
    'src/method/bm_1_distribution_panel.py',
    'src/method/bm_1_g8.py',
    'src/method/bm_1_g7.py',
)

def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError('Only explicit JSON values and real numpy arrays are serializable.')

def encode(value):
    return json.dumps(value,default=plain,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')

def digest(value): return hashlib.sha256(encode(value)).hexdigest()

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        f.write(encode(value));f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    if os.name=='posix':
        fd=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)

class NodeJobStop(RuntimeError): pass

class NodeJobStore:
    def __init__(self,root,*,prior_debit,ceiling=1800.,max_new_seconds=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        cap=float(ceiling-prior_debit if max_new_seconds is None else max_new_seconds)
        if not np.isfinite(cap) or cap<=0: raise ValueError('Positive finite study allowance required.')
        self.seed=dict(schema='g7-node-ledger-v1',prior_debit=float(prior_debit),ceiling=float(ceiling),max_new_seconds=cap)
        if not np.isfinite([prior_debit,ceiling]).all() or not 0<=prior_debit<ceiling:
            raise ValueError('A finite preserved campaign debit below the ceiling is required.')
        self.ledger_path=self.root/'ledger.json'
        with self.lock():
            if self.ledger_path.exists():
                ledger=self.read()
                if ledger['seed']!=self.seed: raise ValueError('Budget seed changed; no reset on resume.')
            else: atomic_json(self.ledger_path,dict(seed=self.seed,entries=[]))
            (self.root/'nodes').mkdir(exist_ok=True)
            (self.root/'contracts').mkdir(exist_ok=True)

    @contextmanager
    def lock(self):
        """Nonblocking OS lock; the OS releases it even after process death."""
        with (self.root/'writer.lock').open('a+b') as f:
            f.seek(0);f.write(b'0');f.flush();f.seek(0)
            if os.name=='nt':
                import msvcrt
                try: msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
                except OSError as exc: raise NodeJobStop('Another writer owns this G campaign.') from exc
            else:
                import fcntl
                try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError as exc: raise NodeJobStop('Another writer owns this G campaign.') from exc
            try: yield
            finally:
                if os.name=='nt':
                    f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
                else: fcntl.flock(f,fcntl.LOCK_UN)

    def read(self): return json.loads(self.ledger_path.read_text(encoding='utf-8'))

    def charged(self):
        ledger=self.read()
        return self.seed['prior_debit']+sum(e['charged_seconds'] for e in ledger['entries'])

    def bind(self,contract):
        key=digest(contract);path=self.root/'contracts'/(key+'.json')
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8'))!=json.loads(encode(contract)):
                raise ValueError('Contract collision or modified checkpoint.')
        else: atomic_json(path,contract)
        return key

    def receipt(self,path,job):
        value=json.loads(path.read_text(encoding='utf-8'))
        if (value.get('schema')!='g7-owned-node-v1' or value['job']!=job
                or value['payload_sha256']!=digest(value['payload'])):
            raise ValueError('Node checkpoint hash or complete job identity changed.')
        return value['payload']

    def _finish(self,ledger,entry,seconds,status,**extra):
        entry.update(status=status,charged_seconds=seconds,**extra)
        atomic_json(self.ledger_path,ledger)

    def node(self,job,reservation,compute,*,retry_ids=()):
        """Called under lock. Explicit retry adds an attempt without erasing costs."""
        key=digest(job);path=self.root/'nodes'/(key+'.json')
        ledger=self.read();attempts=[e for e in ledger['entries'] if e['job_id']==key]
        if path.exists():
            payload=self.receipt(path,job)
            if not attempts: raise ValueError('Receipt exists without a reserved ledger attempt.')
            last=attempts[-1]
            if last['status']=='running':
                # Artifact was fsynced before a crash in the ledger update.
                self._finish(ledger,last,last['reserved_seconds'],'recovered-complete')
            elif last['status'] not in ('complete','recovered-complete'):
                raise ValueError('Completed receipt conflicts with failed ledger state.')
            return payload
        if attempts:
            last=attempts[-1]
            if last['status']=='running':
                self._finish(ledger,last,last['reserved_seconds'],'interrupted')
            elif last['status'] in ('complete','recovered-complete'):
                raise ValueError('Completed node file is missing; do not silently replay it.')
            if key not in retry_ids:
                raise NodeJobStop('Incomplete node requires an explicit retry ID: '+key)
        if not np.isfinite(reservation) or reservation<=0:
            raise ValueError('Positive finite per-node reservation required.')
        total=self.seed['prior_debit']+sum(e['charged_seconds'] for e in ledger['entries'])
        limit=min(self.seed['ceiling'],self.seed['prior_debit']+self.seed['max_new_seconds'])
        if total+reservation>limit:
            raise NodeJobStop('Remaining campaign budget cannot cover this whole node.')
        entry=dict(job_id=key,node=job['index'],attempt=len(attempts)+1,status='running',
                   reserved_seconds=float(reservation),charged_seconds=float(reservation))
        ledger['entries'].append(entry);atomic_json(self.ledger_path,ledger)
        started=perf_counter()
        try:
            payload=compute(started+reservation)
            atomic_json(path,dict(schema='g7-owned-node-v1',job=job,payload=payload,payload_sha256=digest(payload)))
        except Exception as exc:
            # A genuine process loss/BaseException leaves the full running reserve.
            self._finish(ledger,entry,perf_counter()-started,'failed',error=repr(exc))
            raise
        self._finish(ledger,entry,perf_counter()-started,'complete')
        return json.loads(encode(payload))

def source_manifest(root,paths):
    root=Path(root)
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}

class CheckpointedBankControl:
    """Checkpoint only the frozen analytic G8 control fixture for now.

    Future historical B/A callbacks require their own immutable input contract.
    A declared label or sampled callback values alone are insufficient provenance.
    """
    def __init__(self,evaluator,store,*,source_root,source_sha256):
        if not isinstance(evaluator,OwnedBankPriceEvaluator) or not isinstance(store,NodeJobStore):
            raise TypeError('Explicit owned evaluator and campaign store required.')
        if any(s.model.B is not baseline or s.model.A is not target for s in evaluator.supports):
            raise ValueError('This node adapter requires the frozen analytic G8 B/A fixture.')
        self.evaluator,self.store=evaluator,store
        self.root=Path(source_root);self.sources=dict(source_sha256)
        if 'src/method/bm_1_g8_basis.py' in self.sources:
            raise ValueError('G7 source layout changed; restore the original source tree for old checkpoints.')
        required=set(REQUIRED_SOURCE_PATHS)
        if not required.issubset(self.sources): raise ValueError('Required G source hashes are missing.')
        self.check_sources()
        self.contract=self.describe()
        with store.lock(): self.contract_id=store.bind(self.contract)

    def describe(self):
        evaluator=self.evaluator;s=evaluator.supports[0]
        if any(n.model.B is not baseline or n.model.A is not target for n in evaluator.supports):
            raise ValueError('Live B/A callback differs from the frozen fixture.')
        return dict(schema='g7-node-contract-v1',origin=evaluator.origin.to_record(),
            backend=evaluator.backend,source_sha256=self.sources,
            versions=dict(numpy=np.__version__,scipy=scipy.__version__),
            baseline='G8 analytic .5+.1sin(2pi*t/24)',target='G8 fixed .5',
            price_edges=s.interval_edges,times=s.solver_t_eval,midpoints=s.midpoints,widths=s.widths,
            solver_max_step=s.solver_max_step,breakpoints=s.breakpoints,cfl=.4,
            per_node=[dict(solver_max_step=n.solver_max_step,sigma_x=n.sigma_x,
                           x=n.x,dx=n.dx,times=n.solver_t_eval,target=n.target) for n in evaluator.supports])

    def check_sources(self):
        for path,expected in self.sources.items():
            if hashlib.sha256((self.root/path).read_bytes()).hexdigest()!=expected:
                raise ValueError('Registered source changed: '+path)

    def evaluate(self,prices,*,seconds_per_node,retry_ids=()):
        self.check_sources()
        if digest(self.describe())!=self.contract_id:
            raise ValueError('Live evaluator differs from its checkpoint contract.')
        e=self.evaluator;u=e.supports[0].validate_prices(prices).copy()
        nodes=[]
        with self.store.lock():
            for j,s in enumerate(e.supports):
                job=dict(contract=self.contract_id,index=j,rho=float(e.origin.rho_values[j]),prices=u.tolist())
                def compute(deadline,j=j,s=s):
                    model=s.model;original=model._drift
                    had_override='_drift' in model.__dict__;override=model.__dict__.get('_drift')
                    def guarded(*args,**kwargs):
                        if perf_counter()>=deadline: raise NodeJobStop('Node cooperative time limit reached.')
                        return original(*args,**kwargs)
                    model._drift=guarded
                    try:
                        if perf_counter()>=deadline: raise NodeJobStop('Node reservation exhausted before forward.')
                        result=e.evaluate_node(u,j)
                        if perf_counter()>=deadline: raise NodeJobStop('Node exceeded its reservation; result is not accepted.')
                        return result
                    finally:
                        if had_override: model._drift=override
                        else: del model._drift
                nodes.append(self.store.node(job,seconds_per_node,compute,retry_ids=retry_ids))
            return e.combine_nodes(u,nodes)

WORK_COUNTS = {
    'setup':1, 'search_deterministic':48, 'search_fv':48,
    'cross_m254_coarse':3, 'cross_m254_fine':3,
    'cross_fv_coarse':3, 'cross_fv_fine':3,
    'quadrature':3, 'em_coupled':3,
    'feedback_m254_coarse':1, 'feedback_m254_fine':1,
    'feedback_fv_coarse':1, 'feedback_fv_fine':1,
    'audit_export':1,
}


