"""Focused orchestration controls; no native propagation, JIT or workbook read."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np


def control(output):
    from src.exe.run_model import bm_1_h2_compact as c
    passed = []
    def rejects(fn, text=None):
        try:
            fn()
        except (ValueError,TimeoutError) as exc:
            if text is not None:
                assert text in str(exc), str(exc)
        else:
            raise AssertionError('Expected fail-closed rejection.')
    data = c.receipt()
    assert len(data['observations']['values']) == 72 and len(data['jobs']) == 64
    assert data['observations']['final_values_decoded'] == 0
    poisoned = copy.deepcopy(data['observations']); poisoned['times'][-1] = '2022-02-21T00:00:00+01:00'
    rejects(lambda:c.validate_observations(poisoned,data['packet']), 'window')
    missing = copy.deepcopy(data['observations']); missing['observed'][0] = False
    rejects(lambda:c.validate_observations(missing,data['packet']), 'Mask')
    passed.append('reviewed receipt reuse, 72-hour boundary and fixed-mask rejection')
    identity = 'a'*64; logs = np.log(np.array([.2,.3,.5]))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        store = c.JobStore(root/'jobs_control',identity)
        meta = dict(candidate='linear',grid='base',hour=0,kind='observed',branch='prefinal',value=.4)
        counter = []
        def compute():
            counter.append(1)
            return dict(log_state=logs.copy(),log_evidence=1.23)
        store.run(meta,logs,compute)
        resumed = c.JobStore(root/'jobs_control',identity)
        assert resumed.run(meta,logs,compute)['log_evidence'] == 1.23 and len(counter) == 1
        rejects(lambda:resumed.run(meta,logs[::-1],compute), 'chronological')
        path = next((root/'jobs_control'/'jobs').glob('*.npz'))
        payload = c._read_array_checkpoint(path.read_bytes()); payload['result']['log_evidence'] += .01
        c.write(path,payload,replace=True)
        rejects(lambda:resumed.run(meta,logs,compute), 'checksum')
        passed.append('atomic completed-job recovery without replay; prior and payload corruption rejected')
        base = [dict(hour=h,log_state=logs,observed=True,log_evidence=.001,moments=None) for h in range(72)]
        fine = [dict(row,log_evidence=0.) for row in base]
        rejects(lambda:c.compare_streams(base,fine,{'whole':(0,72)}), 'Total-window')
        assert c.compare_streams(base,fine,{'train':(0,48),'selection':(48,72)})['passed']
        rejects(lambda:c.compare_streams(base[:-1],fine[:-1],{'whole':(0,72)}), 'Incomplete')
        passed.append('total-window .05 gate is not replaced by a per-hour check')
        # Real pinned COBYQA control flow, cheap exact quadratic objective only.
        class Fake:
            def __init__(self):
                self.directory=root/'optimizer'; self.manifest={'identity':identity}; self.evaluated=[]; self.pause=True
            def check(self,**kw):
                if self.pause and len(self.evaluated)>=3:
                    raise TimeoutError('deliberate control pause')
            def replay(self,candidate,grid,**kw):
                key=c.curve_key(candidate)
                assert key not in self.evaluated, 'Completed synthetic objective replayed.'
                self.evaluated.append(key)
                x=np.asarray(candidate['logits'])
                return [dict(log_evidence=-float(np.sum((x-.13)**2)))]
        fake=Fake()
        rejects(lambda:c.optimizer_family(fake,2),'deliberate control pause')
        prefix=list(fake.evaluated); fake.pause=False
        try:
            c.optimizer_family(fake,2)
        except ValueError as exc:
            assert 'did not reach' in str(exc)
        calls=c.load(fake.directory/'optimizer_2_calls.json')['calls']
        result=c.load(fake.directory/'optimizer_2_result.json')
        assert fake.evaluated[:3] == prefix and len(fake.evaluated)==len(set(fake.evaluated))<=17
        assert all(np.isfinite(row['objective']) for row in calls)
        assert result['converged'] == (result['status']==0)
        passed.append('actual COBYQA pause/deterministic cached call replay and cap/convergence distinction')
        def fam(b,f):
            return dict(comparison={'scores':{'selection':{'base':b,'fine':f}}})
        chosen=c.select_families({1:fam(1.,1.),2:fam(1.05,1.05),4:fam(.5,.5)})
        assert chosen['selected_family']==1 and chosen['ambiguity'] and not chosen['improvement_claim']
        chosen=c.select_families({1:fam(1.,1.),2:fam(1.3,1.3),4:fam(1.15,1.15)})
        assert chosen['selected_family']==2 and not chosen['ambiguity']
        chosen=c.select_families({1:fam(1.,1.),2:fam(1.2,1.21),4:fam(1.21,1.2)})
        assert chosen['ranking_reversal'] and chosen['selected_family']==1
        passed.append('selection uses all families, declared ambiguity and simpler-family rule')
        no_final=SimpleNamespace(directory=root/'without_forecast',data=data,manifest={'identity':identity})
        rejects(lambda:c.load_final_observations(no_final,root/'does_not_exist.xlsx'),'Freeze')
        f,g=c.curve(c.LINEAR)
        artifact=dict(schema='f4-h2-conditional-IJ-handoff-v1',conditional_prefinal_ready=True,
            origin_time=4.,time_unit='day',origins={'linear':dict(physical=data['packet']['physical'],
                curves=dict(f=f.to_record(),g=g.to_record()),grids={'base':dict(log_state=logs,cell_edges=np.linspace(0.,1.,4))})})
        artifact['checksum']=c.fingerprint(artifact)
        origin=c.as_forecast_origin(artifact,model_key='linear')
        assert origin.time==4. and np.array_equal(origin.posterior_mass,np.exp(logs))
        assert origin.parameters.sigma_x==.5 and origin.sigma_y==.04
        passed.append('final access denied before freeze; existing I/F3 handoff preserves physical/noise roles')
        seed=c.validate_ledger(c.load(c.SEED))
        broken=copy.deepcopy(seed); broken['entries'][0]['charged_seconds'] += 1.
        rejects(lambda:c.validate_ledger(broken),'total')
        passed.append('original PC/chat ledger validated; altered charge rejected')
        # Exercise the whole forecast/final/handoff boundary with an explicitly
        # synthetic algebraic engine. No result is an actual scientific run.
        from dataclasses import asdict
        from src.method.bm_1_interval_forecast import interval_cdf_plan, IntervalDistribution
        plan=interval_cdf_plan(.04); omega=2.*np.pi*np.arange(plan.frequencies+1)/plan.span
        cf=np.exp(1j*omega*.5-.5*.04**2*omega**2); law=IntervalDistribution(plan,cf)
        distribution=dict(plan=asdict(plan),characteristic_real=cf.real,characteristic_imag=cf.imag,
            quantiles=[law.quantile(q) for q in (.05,.5,.95)],cdf_audit=law.audit())
        moment=dict(log_state=logs,interval_mean=.5,observation_mean=.5,interval_variance=0.,
            observation_variance=.04**2,maximum_abs_mass_error=0.)
        class Synthetic:
            directory=root/'forecast'; manifest={'identity':'b'*64}
            def __init__(self):
                self.data=data; self.open_calls=0
            def job(self,candidate,grid,hour,kind,prior,**kw):
                assert kw['branch']=='open' and 'value' not in kw
                c.same_state(prior,logs); self.open_calls+=1
                return copy.deepcopy(moment) if kind=='moments' else dict(log_state=logs.copy(),distribution=copy.deepcopy(distribution))
            def replay(self,candidate,grid,**kw):
                assert kw['branch']=='final' and kw['start']==96 and kw['end']==264
                c.same_state(kw['initial'],logs)
                return [dict(hour=h,log_state=logs.copy(),observed=True,log_evidence=0.,moments=copy.deepcopy(moment)) for h in range(96,264)]
        synthetic=Synthetic()
        curves={1:c.LINEAR,2:dict(pieces=2,logits=[.1,0.]),4:c.PROBE}
        families={p:dict(candidate=v,streams={g:[dict(log_state=logs.copy())] for g in ('base','fine')}) for p,v in curves.items()}
        selection=dict(selected_family=2,cutoff=data['packet']['policy']['split']['holdout_start'],curves={str(k):v for k,v in curves.items()})
        c.write(synthetic.directory/'selection.json',selection)
        frozen=c.forecast_week(synthetic,families,selection)
        assert synthetic.open_calls==2*2*(168+7) and frozen['assimilated_observations']==0
        frozen_hash=c.sha(synthetic.directory/'forecast_frozen.npz')
        from datetime import datetime,timedelta
        origin_time=datetime.fromisoformat(selection['cutoff']); values=np.full(168,.5)
        final_obs=dict(times=[(origin_time+timedelta(hours=h)).isoformat() for h in range(168)],
            observed=np.ones(168,dtype=bool),values=values,group_kW=values*data['packet']['frozen_scales']['power_kW'],
            forecast_sha256=frozen_hash,input_sha256=data['packet']['input_sha256']['input'])
        c.write(synthetic.directory/'final_observations.npz',final_obs)
        final=c.final_evaluation(synthetic,root/'never_read.xlsx')
        assert final['conditional_historical_study_completed'] and not final['scientific_acceptance']
        assert c.sha(synthetic.directory/'forecast_frozen.npz')==frozen_hash
        frozen['origins']['linear']['base'][0] += .01
        c.write(synthetic.directory/'forecast_frozen.npz',frozen,replace=True)
        rejects(lambda:c.load_final_observations(synthetic,root/'never_read.xlsx'),'checksum')
        passed.append('synthetic full 168-hour open/final orchestration; separate branches, seven horizons and immutable freeze')
        env=c.environment()
        assert env['scipy']=='1.17.0' and env['numba']=='0.62.1'
        passed.append('required optimizer/native dependency versions present')
    return dict(status='passed',controls=passed, native_propagations=0, workbook_reads=0,
        receipt_sha256=data['index']['review_sha256'], conditional_ready=False,
        scientific_acceptance=False, blind_h3_acceptance=False)
