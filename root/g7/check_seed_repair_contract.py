"""One focused seed/source/budget check; never import a numerical solver."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

def check(root, previous, report_path):
    started=time.perf_counter()
    sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location('seed_repair',root/'run_auxiliary_seed_repair.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reg=json.loads((root/'registration.json').read_text())
    plan=json.loads((root/'design.json').read_text())
    em=plan['proposed_measurement']['em'];checks=[]
    module.validate_streams(em,reg['streams']);checks.append('11 distinct role/node streams accepted')
    for poison in ('collision','changed_identity','missing_node'):
        bad=copy.deepcopy(em)
        if poison=='collision':bad['noise_seed']=bad['initial_seeds'][2]
        elif poison=='changed_identity':bad['brownian_seeds'][0]+=1
        else:bad['initial_seeds'].pop()
        try:module.validate_streams(bad,reg['streams'])
        except ValueError:checks.append(poison+' rejected')
        else:raise AssertionError(poison)
    def worker(path):
        return next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='worker')
    old=ast.unparse(worker(previous/'run_auxiliary_measurement.py'))
    new=ast.unparse(worker(root/'run_auxiliary_seed_repair.py'))
    expected=old.replace("spec['em']['initial_seed'] + j","spec['em']['initial_seeds'][j]").replace("spec['em']['brownian_seed'] + j","spec['em']['brownian_seeds'][j]")
    new=new.replace("    validate_streams(spec['em'], reg['streams'])\n",'')
    assert new==expected;checks.append('worker AST preserved except seed lookup and guard')
    for name,sha in reg['launcher_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    assert reg['prior_debit']==json.loads((root/'previous_final_budget.json').read_text())['campaign_debit']
    jobs=plan['proposed_measurement']['jobs']
    assert sum(x['count']*x['reservation_each_seconds'] for x in jobs)==68
    assert reg['worker_seconds']==68+2
    assert reg['scope_ceiling']==70+30+20==120
    assert reg['prior_debit']+120<=1800
    checks.append('source hashes and complete preserved budget verified')
    assert not any(name.startswith('src.') for name in sys.modules)
    result=dict(schema='g7-seed-repair-contract-check-v1',passed=True,checks=checks,new_forwards=0,native_solver_imports=0,seconds=time.perf_counter()-started)
    report_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':
    check(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
