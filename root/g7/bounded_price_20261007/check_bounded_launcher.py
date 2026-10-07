"""Focused stdlib launcher checks. No solver imports or forward replays."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace


def main():
    started=time.perf_counter()
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root))
    import launch_bounded_price as launch
    reg,plan,caps=launch.validate(root)
    checks=[]
    def check(name,value):
        if not value: raise AssertionError(name)
        checks.append(name)
    check('fixed_exact_tie_order',launch.select_candidate([1.,1.,1.])==0)
    check('candidate_one',launch.select_candidate([2.,1.,3.])==1)
    check('candidate_two',launch.select_candidate([3.,2.,1.])==2)
    for costs in ([1.,2.],[1.,float('nan'),2.],[1.,float('inf'),2.]):
        try: launch.select_candidate(costs)
        except ValueError: check('incomplete_or_nonfinite_rejected',True)
        else: raise AssertionError('bad candidate bank accepted')
    check('scope_re_admitted',reg['prior_debit']+320<=1800)
    check('maximum_reservations',sum(j['count']*j['reservation_each_seconds']
          for j in plan['proposal']['jobs'])==260)
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory)
        for name in ['registration.json','design.json','previous_budget.json',*reg['launcher_sha256']]:
            shutil.copyfile(root/name,out/name)
        launch.write(out/'invocation_started.json',{})
        try:launch.parent(out,Path('/unused'),Path('/unused'))
        except ValueError:check('started_scope_refuses_spawn',True)
        else:raise AssertionError('automatic invocation permitted')
        (out/'design.json').write_bytes(b'{}')
        try:launch.validate(out)
        except ValueError:check('changed_design_refuses_spawn',True)
        else:raise AssertionError('changed design accepted')
    class Stop(RuntimeError):pass
    try:
        with launch.job_deadline(time.perf_counter()-.1,Stop):
            raise AssertionError('expired job ran')
    except Stop:check('expired_job_before_compute',True)
    if hasattr(launch.signal,'setitimer'):
        try:
            with launch.job_deadline(time.perf_counter()+.02,Stop):time.sleep(.1)
        except Stop:check('posix_job_interrupt',True)
        else:raise AssertionError('job timer failed')

    # Execute the actual nested setup() implementation with stand-ins. Native
    # initializer would re-enter the lock; the stand-in deliberately rejects it.
    tree=ast.parse((root/'launch_bounded_price.py').read_text())
    worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    setup=next(n for n in ast.walk(worker) if isinstance(n,ast.FunctionDef) and n.name=='setup')
    class Probe:
        def __init__(self,*a,**kw):raise AssertionError('native lock reentered')
        def check_sources(self):check('setup_source_check',True)
        def describe(self):return dict(backend=self.evaluator.backend,
            objective=self.evaluator.kw['numerics'].objective_max_step)
    class Origin:
        @staticmethod
        def from_verified_bytes(raw,h):
            check('origin_bytes_authenticated',hashlib.sha256(raw).hexdigest()==h)
            return object()
    class Evaluator:
        def __init__(self,origin,**kw):self.backend=kw['backend'];self.kw=kw
        def combine_nodes(self,prices,nodes):return {'node_count':len(nodes)}
    class Store:
        def bind(self,contract):return digest(contract)
    def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        old=Path(directory);(old/'inputs').mkdir()
        for label in ('coarse','fine'):(old/'inputs'/(label+'.json')).write_bytes(b'{"owned":true}')
        original={'input_files':{label:{'sha256':hashlib.sha256(b'{"owned":true}').hexdigest()}
                   for label in ('coarse','fine')}}
        box={'owned':{'registration':original,'nodes':{'coarse':[{}]*5,'fine':[{}]*5}},'fv_nodes':[{}]*5}
        plan_stub={'reuse':{label+'_contract':digest(dict(backend='mixture',objective=1/96))
                           for label in ('coarse','fine')}}
        box['fv_contract']=digest(dict(backend='fv',objective=1/96))
        env=dict(box=box,old=old,G8BankOrigin=Origin,OwnedBankPriceEvaluator=Evaluator,
            BM1ControlNumerics=lambda **kw:SimpleNamespace(**kw),baseline=object(),target=object(),
            CheckpointedBankControl=Probe,store=Store(),native=Path('/frozen'),manifest={},
            evaluators={},contracts={},values={},audits={},plan=plan_stub)
        for name,backend,label,cells,obj,solver in [
            ('coarse','mixture','coarse',257,1/96,1/96),
            ('fine','mixture','fine',513,1/96,1/192),
            ('quad','mixture','fine',513,1/192,1/192),('fv','fv','fine',513,1/96,1/192)]:
            env.update(name=name,backend=backend,label=label,cells=cells,
                       objective_step=obj,solver_step=solver)
            exec(compile(ast.Module(body=[setup],type_ignores=[]),'<actual setup>','exec'),env)
            result=env['setup']()
            check('live_setup_'+name,result['runtime_rehydrated'] and name in env['evaluators'])
        check('quad_partition_distinct',env['contracts']['quad']!=env['contracts']['fine'])
        check('neutral_combined_without_forward',set(env['values'])=={'neutral_coarse','neutral_fine','neutral_fv'})
    check('no_numeric_imports',not any(n=='numpy' or n=='scipy' or n.startswith('src.') for n in sys.modules))
    result=dict(schema='g7-bounded-launcher-check-v1',passed=True,checks=checks,
        check_count=len(checks),seconds=time.perf_counter()-started,native_forwards=0,
        numerical_imports=0,full_solver_integration_tested=False,
        windows_concurrency_power_loss_tested=False)
    launch.write(root/'bounded_launcher_check_20261007.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
