"""One bounded zero-model shutdown control against the exact registered close AST."""
from time import perf_counter, sleep
import ast, json, multiprocessing as mp, os, sys, threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from root.f4 import run_h2_two_family_v3 as controller

def main():
    decision=controller.load(ROOT/'root/f4/h2_two_family_v3_recovery_20261008.json')
    campaign=ROOT/'files/output/bm_1/runs/estimation/f4_h2_two_family_v3_campaign_20261008.json'
    source=ROOT/controller.SOURCE
    controller.require(controller.sha(source)==decision['source_sha256'][controller.SOURCE], 'Source changed.')
    target=ROOT/'root/f4/h2_two_family_shutdown_control_20261008.json'
    controller.require(not target.exists(),'Never repeat control.')
    with controller.reservation(campaign,decision,'bounded-pool-shutdown-control',5,native=True):
        tree=ast.parse(source.read_text()); run=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run')
        engine=next(x for x in run.body if isinstance(x,ast.ClassDef) and x.name=='Engine')
        close=next(x for x in engine.body if isinstance(x,ast.FunctionDef) and x.name=='close')
        class Base:
            def close(self):
                self.pool.terminate();self.pool.join()
        scope=dict(Base=Base,os=os,threading=threading)
        model=ast.Module(body=[ast.ClassDef(name='CheckedEngine',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[close],decorator_list=[])],type_ignores=[])
        exec(compile(ast.fix_missing_locations(model),str(source),'exec'),scope)
        pool=mp.get_context('fork').Pool(2)
        pending=pool.apply_async(sleep,(30.,));sleep(.05)
        subject=scope['CheckedEngine']();subject.pool=pool
        started=perf_counter();subject.close();elapsed=perf_counter()-started
        controller.require(elapsed<2.5 and not any(p.is_alive() for p in pool._pool), 'Shutdown not bounded/worker survived.')
        controller.write(target,dict(passed=True,source_sha256=controller.sha(source),real_pool_workers=2,
            actual_close_AST_exercised=True,close_seconds=elapsed,numerical_model_propagations=0,
            pending_nonmodel_task_cancelled=True,scientific_acceptance=False))
        print(json.dumps(dict(passed=True,close_seconds=elapsed,numerical_model_propagations=0)),flush=True)
    return 0
if __name__=='__main__':
    result=main();sys.stdout.flush();sys.stderr.flush();os._exit(result)
