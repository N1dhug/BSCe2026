"""One real own-contract setup check; evaluate_node deliberately disabled."""
import argparse
import ast
import json
from pathlib import Path
import signal
import sys
import time


def main(old,seed):
    started=time.perf_counter();root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root));sys.path.insert(0,str(old/'frozen_source'))
    from launch_bounded_price import validate,write
    from receipt_recovery import checked_json,VerifiedReceiptIndex,sha
    from restore_auxiliary import read_owned
    def stop(*unused):raise TimeoutError('Preflight5.5s cap exhausted.')
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,5.5)
    checks=[]
    try:
        reg,plan,caps=validate(root)
        for label in ('coarse','fine'):
            if sha(old/'inputs'/(label+'.json'))!=plan['provenance']['original_input_files'][label]['sha256']:
                raise ValueError('Original input hash differs.')
        checks.append('both_actual_original_input_hashes')
        from src.control.bm_1 import BM1ControlNumerics
        from src.method.bm_1_distribution_panel import baseline,target
        from src.method.bm_1_g7 import OwnedBankPriceEvaluator
        from src.method.bm_1_g7_jobs import CheckpointedBankControl,digest
        from src.method.bm_1_g8 import G8BankOrigin
        from src.exe.run_model.bm_1.g8_bernstein import SolverAudit
        def forbidden(*args,**kwargs):raise AssertionError('Forward attempted in setup preflight.')
        OwnedBankPriceEvaluator.evaluate_node=forbidden
        audit=checked_json(root/'previous_audit.json',reg['previous_audit_sha256'])
        owned=read_owned(old,audit,plan['reuse'])
        index=VerifiedReceiptIndex(seed,root/'adoption_manifest.json',
            manifest_sha256=reg['adoption_manifest_sha256'],eager_payload_validation=False)
        rows=sorted((e for e in index.entries.values() if e['job']['kind']=='fv_fine_nodes'),
                    key=lambda e:e['job']['index'])
        box=dict(owned=owned,fv_nodes=[index.payload(e['job']) for e in rows],
            fv_contract=checked_json(seed/'fv_fine_setup.json',reg['fv_setup_sha256'])['contract_id'])
        class BindingStore:
            def bind(self,contract):return digest(contract)
        tree=ast.parse((root/'launch_bounded_price.py').read_text())
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        setup=next(n for n in ast.walk(worker) if isinstance(n,ast.FunctionDef) and n.name=='setup')
        env=dict(box=box,old=old,G8BankOrigin=G8BankOrigin,
            OwnedBankPriceEvaluator=OwnedBankPriceEvaluator,BM1ControlNumerics=BM1ControlNumerics,
            baseline=baseline,target=target,SolverAudit=SolverAudit,
            CheckpointedBankControl=CheckpointedBankControl,store=BindingStore(),
            native=old/'frozen_source',manifest=owned['sources'],evaluators={},contracts={},
            values={},audits={},plan=plan)
        for name,backend,label,cells,obj,solver in [
            ('coarse','mixture','coarse',257,1/96,1/96),
            ('fine','mixture','fine',513,1/96,1/192),
            ('quad','mixture','fine',513,1/192,1/192),('fv','fv','fine',513,1/96,1/192)]:
            env.update(name=name,backend=backend,label=label,cells=cells,
                       objective_step=obj,solver_step=solver)
            exec(compile(ast.Module(body=[setup],type_ignores=[]),'<actual live setup>','exec'),env)
            env['setup']();checks.append('live_complete_contract_'+name)
        if len(env['evaluators']['fine'].supports[0].midpoints)!=192 or len(env['evaluators']['quad'].supports[0].midpoints)!=384:
            raise AssertionError('Wrong midpoint refinement.')
        checks.append('192_vs384_actual_midpoints')
        # Exercise the actual parent admission before spawn with absent inputs.
        import tempfile,shutil
        from launch_bounded_price import parent
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for name in ['registration.json','design.json','previous_budget.json',*reg['launcher_sha256']]:
                shutil.copyfile(root/name,out/name)
            try:parent(out,Path('/absent_owned_inputs'),seed)
            except FileNotFoundError:checks.append('missing_input_blocks_before_spawn')
            else:raise AssertionError('Missing inputs accepted.')
            if (out/'invocation_started.json').exists():raise AssertionError('Scope started before input check.')
        result=dict(schema='g7-input-fixed-live-setup-preflight-v1',passed=True,checks=checks,
            seconds=time.perf_counter()-started,new_forwards=0,numerical_imports=True,
            scope='real four setups and neutral receipt reduction; forward method disabled',
            solver_propagation_or_windows_integration_tested=False)
    except Exception as exc:
        import traceback
        result=dict(schema='g7-input-fixed-live-setup-preflight-v1',passed=False,error=str(exc),
            traceback=traceback.format_exc(),checks=checks,seconds=time.perf_counter()-started,new_forwards=0)
    finally:signal.setitimer(signal.ITIMER_REAL,0.)
    write(root/'input_fixed_preflight_20261007.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--seed',type=Path,required=True);args=parser.parse_args()
    main(args.previous.resolve(),args.seed.resolve())
