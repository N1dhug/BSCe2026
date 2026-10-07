"""One focused non-numerical control of the approved budget adapter."""
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from root.f4 import run_h2_same_day as s
c=s.c


def check():
    decision=c.load(ROOT/'root/f4/h2_same_day_budget_decision_20261007.json')
    original=c.load(ROOT/decision['original_campaign_path'])
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder);path=folder/'campaign.json'
        ledger=s.register_campaign(path,decision)
        assert ledger['entries']==original['entries'] and ledger['charged_seconds']==original['charged_seconds']
        assert ledger['ceiling_seconds']==64800. and original['ceiling_seconds']==36000.
        bad={**ledger,'entries':[dict(e) for e in ledger['entries']]}
        bad['entries'][-1]['charged_seconds']=0.
        try:s.validate_campaign(bad,decision)
        except ValueError:pass
        else:raise AssertionError('Old charge was allowed to change.')
        with s.reservation(path,decision,'control') as allocation:
            live=c.load(path)
            assert live['entries'][-1]['status']=='reserved'
            assert live['entries'][-1]['charged_seconds']==600.
            assert allocation['before']==original['charged_seconds']
        closed=c.load(path)
        assert closed['entries'][:-1]==original['entries']
        assert closed['entries'][-1]['status']=='closed' and 0.<=closed['entries'][-1]['charged_seconds']<600.
        s.validate_campaign(closed,decision)
        engine=s.Engine.__new__(s.Engine)
        engine.directory=folder
        engine.same_day_search=False
        engine.validation_remaining=decision['remaining_validation_seconds']
        engine.rates=decision['protected_rates']
        engine.allocation=dict(ledger=ledger,before=64800.-engine.validation_remaining-61.,
                               begun=s.perf_counter(),deadline=s.time()+500.)
        engine.check(search=False)
        try:engine.check(search=True)
        except TimeoutError:pass
        else:raise AssertionError('Search consumed protected validation.')
        engine.allocation['before']=0.
        def old_job(self,*args,**kwargs):
            assert self.same_day_search is True and kwargs['search'] is False
            return {'original_job':True}
        with patch.object(c.Engine,'job',old_job):
            assert engine.job(c.LINEAR,'base',27,'observed',None,value=.4,search=True)=={'original_job':True}
        assert engine.same_day_search is False
        with patch.object(s,'time',return_value=engine.allocation['deadline']):
            try:engine.check(search=False)
            except TimeoutError:pass
            else:raise AssertionError('Invocation deadline ignored.')
    return dict(status='passed',scope='budget prefix, reserve/close, validation protection, original-job delegation and deadline',
                native_propagations=0,model_jobs=0)


if __name__=='__main__':
    begun=perf_counter();result=check();result['elapsed_seconds']=perf_counter()-begun
    result['source_sha256']=c.sha(Path(__file__))
    result['controller_sha256']=c.sha(ROOT/'root/f4/run_h2_same_day.py')
    c.write(c.OUT/'f4_h2_same_day_budget_control_20261007.json',result)
    print(json.dumps(result))
