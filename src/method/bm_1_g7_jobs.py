"""G7 owned node receipts, exact job identities and conservative crash ledger.

One campaign directory has one OS-locked writer. Completed receipts survive a
crash before the ledger commit; incomplete attempts retain their reservation.
Transport equations, output partition and the primary M254 basis are unchanged.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from src.method.bm_1_g7 import OwnedBankPriceEvaluator
from src.method.bm_1_distribution_panel import baseline, target


REQUIRED_SOURCE_PATHS = ('src/method/beta_mixture.py', 'src/control/objective.py', 'src/validation/model/bm.py', 'src/model/base/bm.py', 'src/validation/method/euler_maruyama.py', 'src/method/euler_maruyama.py', 'src/validation/solver/active_set.py', 'src/solver/active_set.py', 'src/validation/solver/finite_volume.py', 'src/solver/finite_volume.py', 'src/validation/method/fokker_planc.py', 'src/method/fokker_planc.py', 'src/validation/model/bm_1.py', 'src/validation/tools/input_generator/input_builder.py', 'src/tools/input_generator/input_builder.py', 'src/validation/method/objective.py', 'src/method/objective.py', 'src/model/bm_1.py', 'src/validation/control.py', 'src/control/bm_1.py', 'src/validation/estimation.py', 'src/params/bm_1.py', 'src/method/bm_1_estimation.py', 'src/control/bm_1_handoff.py', 'src/params/bm_1_deterministic.py', 'src/method/bm_1_deterministic_fit.py', 'src/params/bm_1_response.py', 'src/method/bm_1_deterministic_response.py', 'src/method/bm_1_linear_noise.py', 'src/method/bm_1_distribution_panel.py', 'src/method/bm_1_g8.py', 'src/method/bm_1_g8_bernstein.py', 'src/method/bm_1_g7.py', 'src/method/bm_1_g8_basis.py', 'src/method/bm_1_g7_jobs.py', 'src/method/bm_1_g7_budget.py')


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
    def __init__(self,root,*,prior_debit,ceiling=1800.):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.seed=dict(schema='g7-node-ledger-v1',prior_debit=float(prior_debit),ceiling=float(ceiling))
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
        if total+reservation>self.seed['ceiling']:
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
        required=set(REQUIRED_SOURCE_PATHS)
        if not required.issubset(self.sources): raise ValueError('Required G source hashes are missing.')
        self.check_sources()
        s=evaluator.supports[0]
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
