"""Before/after comparison harness; removed from the final project source."""
import dataclasses, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root))
from src.validation.bm_1 import control as c, numerics as n, mixture as m, deterministic as d, cases
from src.method import bm_1_control as core
from src.exe.run_model.bm_1 import real_data
from src.tools.bm_1_data import DaySignals
import pandas as pd

out={}
def capture(name,obj):
    if callable(obj) or obj is None or isinstance(obj,(str,bytes,Path)):return
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k not in ('elapsed_seconds','timings','seconds','reference_seconds','fast_seconds','speedup'):
                capture(name+'/'+str(k),v)
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):capture(name+'/'+f.name,getattr(obj,f.name))
    elif isinstance(obj,(list,tuple)):
        for k,v in enumerate(obj):capture(name+'/'+str(k),v)
    else:
        try:a=np.asarray(obj)
        except (TypeError,ValueError):return
        if a.dtype.kind in 'biufc':out[name]=a.copy()

print('F3-A',flush=True)
a=c.run_reference(optimize=True,progress=None);capture('contract',a)
e=a['problem'].build_evaluator(a['numerics'],fast=True)
rng=np.random.default_rng(109)
for j in range(5): capture(f'random_prices/{j}', e.evaluate(rng.uniform(size=2)))
candidates={'reference':np.array([.72,.28]),'other':np.array([.730609,.264415])}
capture('paired_mc',core.evaluate_mc_candidates_common_random_numbers(e,candidates,n_paths=128,initial_seed=5,brownian_seed=6))
print('F3-B',flush=True)
capture('refinement',n.run_f3_b_reference(mc_paths=128,mc_time_paths=64,progress=None))
print('Mixture',flush=True)
args=dict(degrees=(4,8),x_steps=31,t_eval=np.linspace(0,1,5),max_step=.125)
capture('projection',m.run_projection_study(**args))
capture('dynamic',m.run_dynamic_mixture_study(**args))
capture('conditioning',m.run_conditioning_study(**args,epsilons=(1e-4,),n_directions=1))
capture('objective',m.run_objective_reference(n_paths=128,degrees=(4,8),x_steps=31,objective_max_step=.125,solver_max_step=.125))
print('F1/complex',flush=True)
B,U=d.build_inputs()
for steps in (24,96,192):capture(f'euler/{steps}',d.run_euler(steps,B,U))
f,t,y=c.build_complex_deterministic_target();capture('complex_target',(t,y,f(np.linspace(0,1,123))))
target,_,_=c.build_deterministic_target();model=c.build_stochastic_model(target)
def make_case(k):
    ev=c._build_evaluator(model,k,fast_fv=True)
    return SimpleNamespace(evaluator=ev,optimization=SimpleNamespace(best=SimpleNamespace(x=np.linspace(.6,.4,k))))
pair={k:make_case(k) for k in (2,4)}
if hasattr(c,'_evaluate_cases_mc'):
    batch=c._evaluate_cases_mc(model,pair,c.HOURLY_EDGES[1:-1],n_paths=128,initial_seed=c.COMPLEX_MC_INITIAL_SEED,brownian_seed=c.COMPLEX_MC_BROWNIAN_SEED,max_step=c.COMPLEX_SOLVER_MAX_STEP)
else:
    with patch.object(c,'COMPLEX_MC_PATHS',128):batch=c._evaluate_mc_common_random_numbers(model,pair)
capture('multi_grid_mc',batch)
print('Synthetic real data',flush=True)
fields={f.name for f in dataclasses.fields(DaySignals)}
day=pd.Timestamp('2022-05-17');hour=np.arange(24)
vals=dict(day=day,baseline_days=tuple(day-pd.Timedelta(weeks=i) for i in range(1,6)),
          target=.3+.03*np.sin(hour/24*2*np.pi),baseline=np.full(24,.3),observed_price=.5+.1*np.cos(hour/24*2*np.pi),
          raw_target=.3+.03*np.sin(hour/24*2*np.pi),raw_price=100+20*np.cos(hour/24*2*np.pi))
signals=DaySignals(**{k:v for k,v in vals.items() if k in fields})
with patch.object(real_data,'save_day_plot',return_value=root/'dummy.png'):
    capture('historical_run',real_data.run_day(signals,cells=21,mc_paths=128,progress=None))
np.savez_compressed(sys.argv[2],**out)
print('Saved',len(out),'numerical fields',sum(a.size for a in out.values()),'values',flush=True)
