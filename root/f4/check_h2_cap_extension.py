"""One cached-only control: exact adapter delta and no repeated model jobs."""
from time import perf_counter,time
BEGUN=perf_counter()-.25
import ast
import inspect
import json
from pathlib import Path
import shutil
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from root.f4 import run_h2_cap_extension as x
c=x.c;s=x.s

def check():
    old=inspect.getsource(c.optimizer_family)
    new=inspect.getsource(x.optimizer_family)
    back=new.replace('extension_optimizer_','optimizer_').replace('extension_candidate_registry','candidate_registry').replace('{2:24, 4:48}','{2:16, 4:32}').replace("{'2':24, '4':48}","{'2':16, '4':32}").replace('<= 73','<= 49').replace('< 73','< 49')
    assert ast.dump(ast.parse(old),include_attributes=False)==ast.dump(ast.parse(back),include_attributes=False)
    source=c.OUT/'f4_h2_staged_recovered_20261007'
    class BeforeNewModel(BaseException):pass
    with tempfile.TemporaryDirectory() as tmp:
        class CachedEngine:
            directory=Path(tmp)
            manifest=c.load(source/'manifest.json')
            def check(self,**kwargs):pass
            def replay(self,candidate,*args,**kwargs):
                self.pending=candidate
                raise BeforeNewModel()
        engine=CachedEngine()
        shutil.copy2(source/'optimizer_2_calls.json',engine.directory/'extension_optimizer_2_calls.json')
        shutil.copy2(source/'candidate_registry.json',engine.directory/'extension_candidate_registry.json')
        try:x.optimizer_family(engine,2)
        except BeforeNewModel:pass
        else:raise AssertionError('Expected one new point after all 17 cached calls.')
        diagnostic=c.load(c.OUT/'f4_h2_cap_extension_diagnostic_20261007.json')
        assert engine.pending['logits']==diagnostic['next_point']
        assert c.load(engine.directory/'extension_optimizer_2_calls.json')==c.load(source/'optimizer_2_calls.json')
        assert len(c.load(engine.directory/'extension_candidate_registry.json')['families']['2'])==17
        failed=x.Engine.__new__(x.Engine);failed.directory=Path(tmp)
        c.write(failed.directory/'EXTENSION_FAILED.json',{'status':'failed'})
        try:failed.check()
        except ValueError:pass
        else:raise AssertionError('Extension failure did not stop.')
    return dict(status='passed',scope='AST-equivalent optimizer except declared caps/paths; exact 17-call reuse; new candidate and terminal guard',native_propagations=0,new_model_jobs=0,next_candidate=engine.pending)

if __name__=='__main__':
    path=c.OUT/'f4_h2_same_day_campaign_20261007.json'
    budget=c.load(c.ROOT/'root/f4/h2_same_day_budget_decision_20261007.json')
    with c._RunLock(path.with_suffix('.lock')):
        ledger=s.validate_campaign(c.load(path),budget)
        native=c.ordered_sum(e['charged_seconds'] for e in ledger['entries'] if e['label'].startswith('native-development|'))
        assert 900.-native>=10.
        entry=dict(label='native-development|cap-extension-focused-control',started_utc_seconds=time(),status='reserved',charged_seconds=10.)
        ledger['entries'].append(entry);ledger['charged_seconds']=c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries']);c.write(path,ledger,replace=True)
        timer=s.threading.Timer(max(.01,9.-(perf_counter()-BEGUN)),lambda:s.os._exit(124));timer.daemon=True;timer.start()
        try:
            result=check();result.update(source_sha256=c.sha(Path(__file__)),controller_sha256=c.sha(c.ROOT/'root/f4/run_h2_cap_extension.py'),elapsed_seconds=perf_counter()-BEGUN)
            c.write(c.OUT/'f4_h2_cap_extension_control_20261007.json',result);print(json.dumps(result))
        finally:
            entry.update(status='closed',charged_seconds=perf_counter()-BEGUN)
            ledger['charged_seconds']=c.PRIOR_SECONDS+c.ordered_sum(e['charged_seconds'] for e in ledger['entries']);c.write(path,ledger,replace=True);timer.cancel()
