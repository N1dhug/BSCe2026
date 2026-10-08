"""G-S3 process uncertainty at fixed parameters; calibration validity is separate.

This module adds no measurement noise, parameter bank or refit of S3. FV
integrates the existing Ito SDE. A caller must explicitly supply sigma_x;
an invalid historical linear-noise fit is never silently used as that sigma.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.method.bm_1_s3_forecast import (
    validate_snapshot, predict_s3, _origin, _frame, _day_inputs, _fingerprint)
from src.method.bm_1_linear_noise import unit_variance
from src.tools.bm_1_deterministic_snapshot import restore_baseline
from src.tools.bm_1_deterministic_data import prepare_frame
from src.params.bm_1_response import BM1PiecewiseResponse
from src.model.bm_1 import bm_1
from src.control.bm_1 import point_initial_cell_mass
from src.solver.finite_volume import finite_volume_1d_cached
from src.exe.run_model.bm_1.deterministic_weekday import metrics


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('src/method/bm_1_s3_uncertainty.py', 'src/method/bm_1_linear_noise.py',
           'src/model/bm_1.py', 'src/model/base/bm.py', 'src/control/bm_1.py',
           'src/solver/finite_volume.py', 'src/validation/solver/finite_volume.py',
           'src/validation/model/bm.py', 'src/validation/model/bm_1.py')
GATES = {'demand_mean':.005, 'state_mean':.005, 'demand_variance':.002,
         'state_variance':.002, 'demand_quantiles':.005, 'state_cdf_after_start':.05}


def source_hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in SOURCES}


def demand_values(x, baseline, price_response, k, xi):
    delta = np.tanh(.5*k*(1-2*np.asarray(x)+price_response))
    return baseline+xi*delta*np.where(delta >= 0, 1-baseline, baseline)


def fit_variance_amplitude(unit_variance_profile, measured_variance):
    """One nonnegative LS amplitude q. This does not assert physical validity."""
    w, v = np.broadcast_arrays(np.asarray(unit_variance_profile,float),np.asarray(measured_variance,float))
    if not np.isfinite(w).all() or np.any(w < 0) or np.any(v[np.isfinite(v)] < 0):
        raise ValueError('Nonnegative unit/observed variances are required.')
    mask = np.isfinite(v)
    if not mask.any(): return {'status':'unidentifiable','q_per_hour':None,'samples':0}
    denominator = float(w[mask]@w[mask])
    if denominator <= np.finfo(float).tiny:
        return {'status':'unidentifiable','q_per_hour':None,'samples':int(mask.sum())}
    q = max(0.,float(w[mask]@v[mask])/denominator)
    return {'status':'estimated-not-validated','q_per_hour':q,'sigma_per_sqrt_hour':float(np.sqrt(q)),
        'sigma_per_sqrt_day':float(np.sqrt(24*q)), 'samples':int(mask.sum()),
        'variance_rmse':float(np.sqrt(np.mean((q*w[mask]-v[mask])**2)))}


def diagnose_calibration(snapshot, raw_history, *, seconds=60.):
    """Bounded training-only audit; no future demand, baseline fit or FV search.

    Raw interday dispersion is not an identified conditional process variance.
    Demand-support failure and P>1/4 independently forbid that interpretation.
    """
    started=perf_counter(); s=validate_snapshot(snapshot);origin=_origin(s['origin'])
    if not np.isfinite(seconds) or not 0 < seconds <= 60:raise ValueError('Diagnostic budget must be in (0,60] seconds.')
    _frame(raw_history,('Demand','Price','Temp'))
    train=raw_history.loc[raw_history.index<origin,['Demand','Price','Temp']].copy()
    if _fingerprint(train)!=s['training_fingerprint']:
        raise ValueError('Diagnostic history must match the fitted snapshot; no new training data/refit.')
    scales=s['scales']
    frame,_=prepare_frame(train,*s['normalization_window'],
        price_bounds=(scales['price_min'],scales['price_min']+scales['price_scale']))
    baseline=restore_baseline(s['baseline']);g=BM1PiecewiseResponse.from_record(s['price_response'])
    empirical=np.full((7,24),np.nan);counts=np.zeros((7,24),int)
    profiles=[np.full((7,24),np.nan),np.full((7,24),np.nan)]
    state_unit=[[],[]]; outside=0; measured=0; support_widths=[]
    demand_error=0.;unit_std_error=0.
    for d,days in enumerate(s['history']['days_by_weekday']):
        observations=np.full((5,24),np.nan);unit=[np.full((5,24),np.nan),np.full((5,24),np.nan)]
        for row,day in enumerate(days):
            inp,index=_day_inputs(frame,baseline,pd.Timestamp(day),history=True,steps=s['numerics']['steps_per_hour'])
            hours=index[24:].hour;observed=inp.observations[24:]
            observations[row,hours]=observed
            b=np.interp(np.arange(inp.n_hours),inp.time,inp.baseline)[24:]
            lo=demand_values(1.,b,g(inp.prices[24:]),s['parameters'][1],s['parameters'][2])
            hi=demand_values(0.,b,g(inp.prices[24:]),s['parameters'][1],s['parameters'][2])
            mask=np.isfinite(observed);measured+=int(mask.sum())
            outside+=int(np.count_nonzero(mask&((observed<lo)|(observed>hi))))
            support_widths.extend((hi-lo)[mask].tolist())
            pair=[]
            for grid,item in enumerate((inp,inp.refined())):
                out=unit_variance(item,s['parameters'],price_response=g,reset_hour=24,deadline=started+seconds)
                pair.append(out)
                unit[grid][row,hours]=out['unit_demand_variance'][24:]
                state_unit[grid].append(out['unit_state_variance'][24:])
            demand_error=max(demand_error,float(np.max(abs(pair[0]['demand']-pair[1]['demand']))))
            unit_std_error=max(unit_std_error,float(np.max(abs(pair[0]['unit_demand_std']-pair[1]['unit_demand_std']))))
        for h in range(24):
            y=observations[:,h];y=y[np.isfinite(y)];counts[d,h]=len(y)
            if len(y)>=2:empirical[d,h]=float(np.var(y,ddof=1))
        for grid in range(2):profiles[grid][d]=np.nanmean(unit[grid],axis=0)
    fits=[fit_variance_amplitude(w[:,1:].reshape(-1),empirical[:,1:].reshape(-1)) for w in profiles]
    maxima=[None if f['q_per_hour'] is None else float(f['q_per_hour']*max(np.max(v) for v in state_unit[j]))
            for j,f in enumerate(fits)]
    reasons=[]
    profile_std_error=None if any(f['q_per_hour'] is None for f in fits) else float(np.nanmax(abs(
        np.sqrt(fits[0]['q_per_hour']*profiles[0])-np.sqrt(fits[1]['q_per_hour']*profiles[1]))))
    numerics={'passed':demand_error<=1e-5 and unit_std_error<=1e-4 and
        profile_std_error is not None and profile_std_error<=1e-4,
        'max_demand_error':demand_error,'max_unit_demand_std_error':unit_std_error,
        'max_fitted_profile_std_error':profile_std_error,'demand_tolerance':1e-5,'std_tolerance':1e-4}
    if not numerics['passed']:reasons.append('Historical coarse/fine noise profile check failed or was unidentifiable.')
    if outside:reasons.append('Measured historical demand outside frozen-model state support; observation/model error is excluded.')
    if any(v is not None and v>.25 for v in maxima):reasons.append('Calibrated LNA state variance exceeds the necessary [0,1] bound 1/4.')
    if any(f['q_per_hour'] is None for f in fits):reasons.append('Process amplitude not identifiable from the supplied profiles.')
    return {'status':'rejected-process-calibration' if reasons else 'candidate-needs-nonlinear-validation',
        'reasons':reasons,'origin':s['origin'],'measured_samples':measured,
        'outside_frozen_demand_support':outside,'outside_support_fraction':outside/measured if measured else None,
        'mean_normalized_demand_support_width':float(np.mean(support_widths)) if measured else None,
        'coarse_fit':fits[0],'fine_fit':fits[1],'max_lna_state_variance':maxima,
        'numerics':numerics,
        'empirical_counts':counts.tolist(),'seconds':perf_counter()-started,
        'semantics':'raw matching-weekday variance, not residual or isolated process noise; P0=0 after warmup',
        'used_in_process_forecast':False,'source_sha256':source_hashes()}


def _fv_day(inp, parameters, curve, initial_state, sigma, cells, step, deadline):
    """Existing conservative FV, with unchanged zero-flux/no-repair contract."""
    def baseline(t):return np.interp(t,inp.time,inp.baseline)
    def price(t):
        j=np.minimum(np.floor(np.asarray(t)+1e-12).astype(int),inp.n_hours-1)
        return inp.prices[j]
    model=bm_1(baseline,baseline,(0.,float(inp.n_hours)),inp.n_hours+1)
    model.fokker_planc_reference=finite_volume_1d_cached(vectorized_inputs=True)
    edges,_,_=model.fokker_planc_reference.grid(cells)
    mass=point_initial_cell_mass(edges,initial_state)
    p=[price,BM1PiecewiseResponse.linear(pieces=2),curve,*parameters[:3]]
    original_drift=model._drift
    def guarded_drift(t,x,params):
        if perf_counter()>=deadline:raise RuntimeError('G uncertainty runtime budget stop.')
        return original_drift(t,x,params)
    model._drift=guarded_drift
    out=model.fokker_planc_reference_solver(mass,p,sigma,t_eval=np.arange(inp.n_hours+1,dtype=float),
        breakpoints=np.arange(1,inp.n_hours,dtype=float),max_step=step)
    t,x,_,weights,_,_,_,_,_,diagnostics=out
    # Read moments from probability cell masses, not FV density samples.
    state_mean=weights@x;state_var=np.sum(weights*(x[None,:]-state_mean[:,None])**2,axis=1)
    demand=np.array([demand_values(x,baseline(hour),curve(price(hour)),parameters[1],parameters[2]) for hour in t])
    mean=np.sum(weights*demand,axis=1)
    variance=np.sum(weights*(demand-mean[:,None])**2,axis=1)
    quantiles=[]
    for values,w in zip(demand,weights):
        order=np.argsort(values);cdf=np.cumsum(w[order])
        indices=np.searchsorted(cdf,[.025,.5,.975],side='left')
        quantiles.append(values[order][indices])
    cdf=np.stack([weights[:,x<=q].sum(axis=1) for q in (.25,.5,.75)],axis=1)
    return {'demand_mean':mean[:-1],'demand_variance':variance[:-1],
        'demand_quantiles':np.asarray(quantiles)[:-1],'state_mean':state_mean[:-1],
        'state_variance':state_var[:-1],'state_cdf_after_start':cdf[1:-1],
        'diagnostics':diagnostics,'initial_state_target':initial_state,
        'initial_state_grid':float(mass@x),'initial_state_discretization_error':abs(float(mass@x)-initial_state)}


def process_forecast(snapshot, raw_inputs, *, sigma_x, cells=128, max_step=1/24, seconds=120.):
    """Numerically checked process-only scenario; sigma is explicit, not fitted.

    Coarse/fine use each grid's own stored S3 daily initial state. The zero
    diffusion limit reuses RK4 exactly, avoiding FV artificial advection noise.
    """
    started=perf_counter();s=validate_snapshot(snapshot);base=predict_s3(s,raw_inputs)
    if base['status']!='complete':raise ValueError('S3 deterministic forecast failed numerical checks.')
    if (not np.isfinite(sigma_x) or sigma_x<0 or type(cells) is not int or not 32<=cells<=512
            or not np.isfinite(max_step) or not 0<max_step<=1/12
            or not np.isfinite(seconds) or not 0<seconds<=120):
        raise ValueError('Declare sigma>=0, 32..512 cells, step in (0,1/12], and budget in (0,120] s.')
    table=base['normalized_predictions'].copy()
    if sigma_x==0:
        table['process_mean']=table.prediction;table['process_variance']=0.
        table['process_lower95']=table.prediction;table['process_upper95']=table.prediction
        table['process_state_mean']=table.state;table['process_state_variance']=0.
        checks={'passed':True,'zero_diffusion':'exact original S3 RK4 forecast','daily_errors':[]}
        audits=[];solver='RK4 deterministic zero-diffusion limit'
    else:
        origin=_origin(s['origin']);scales=s['scales'];curve=BM1PiecewiseResponse.from_record(s['price_response'])
        baseline=restore_baseline(s['baseline'])
        frame=pd.DataFrame({'U':2*(raw_inputs.Price-scales['price_min'])/scales['price_scale']-1,
            'Temp':raw_inputs.Temp},index=raw_inputs.index)
        results=[];audits=[];errors=[]
        for d in range(7):
            inp,index=_day_inputs(frame,baseline,origin+pd.Timedelta(days=d),steps=s['numerics']['steps_per_hour'])
            coarse=_fv_day(inp,s['parameters'],curve,s['forecast_states'][d],sigma_x,cells,max_step,started+seconds)
            fine=_fv_day(inp,s['parameters'],curve,s['fine_forecast_states'][d],sigma_x,2*cells,max_step/2,started+seconds)
            errors.append({name:float(np.max(np.abs(coarse[name]-fine[name]))) for name in GATES})
            results.append(fine);audits.append({'coarse':coarse['diagnostics'],'fine':fine['diagnostics'],
                'initial_state_discretization_error_coarse':coarse['initial_state_discretization_error'],
                'initial_state_discretization_error_fine':fine['initial_state_discretization_error']})
        table['process_mean']=np.concatenate([r['demand_mean'] for r in results])
        table['process_variance']=np.concatenate([r['demand_variance'] for r in results])
        q=np.concatenate([r['demand_quantiles'] for r in results])
        table['process_lower95']=q[:,0];table['process_upper95']=q[:,2]
        table['process_state_mean']=np.concatenate([r['state_mean'] for r in results])
        table['process_state_variance']=np.concatenate([r['state_variance'] for r in results])
        checks={'passed':all(e[k]<=GATES[k] for e in errors for k in GATES),
            'gates':deepcopy(GATES),'daily_errors':errors,'cells':[cells,2*cells],
            'max_steps_hours':[max_step,max_step/2],
            'cdf_policy':'fixed state thresholds .25,.5,.75 after t=0; point-origin grid error reported separately'}
        solver='existing conservative finite-volume Ito Fokker-Planck reference'
    raw=table.copy();scales=s['scales']
    for name in ('process_mean','process_lower95','process_upper95'):
        raw[name]=scales['demand_min']+scales['demand_scale']*table[name]
    for name in ('prediction','baseline'):
        raw[name]=base['predictions'][name]
    raw['process_variance']=scales['demand_scale']**2*table.process_variance
    manifest=deepcopy(base['manifest'])
    manifest['stochastic_status']='implemented-conditional-process-scenario'
    manifest['deterministic_numerics']=deepcopy(base['numerics'])
    return {'status':'complete-process-scenario' if checks['passed'] else 'numerics-failed',
        'predictions':raw,'normalized_predictions':table,'manifest':manifest,
        'uncertainty':{'kind':'conditional-state-process-only','sigma_per_sqrt_hour':float(sigma_x),
            'sigma_calibration':'explicit user/scenario input; not historically estimated or accepted',
            'solver':solver,'initial_law':'point at each own frozen daily S3 state; FV cell representation error reported',
            'excluded':['measurement/model error','parameter uncertainty','baseline uncertainty','future input uncertainty'],
            'bands':'pointwise 2.5%/97.5% quantiles of D(X); no Gaussian assumption or output clipping',
            'source_sha256':source_hashes()},'numerics':checks,'solver_audits':audits,
        'scoring':{'status':'not-scored'},'seconds':perf_counter()-started}


def score_process_forecast(result, observations):
    if not isinstance(observations,pd.Series) or observations.index.has_duplicates:
        raise ValueError('Scoring requires a uniquely indexed raw-demand Series.')
    out=deepcopy(result);p=out['predictions']
    y=pd.to_numeric(observations.reindex(p.index),errors='coerce').to_numpy(float)
    y[~np.isfinite(y)]=np.nan
    mask=np.isfinite(y);p['observed']=y;p['scored']=mask
    scales=out['manifest']['scales']
    out['normalized_predictions']['observed']=(y-scales['demand_min'])/scales['demand_scale']
    out['normalized_predictions']['scored']=mask
    if not mask.any():out['scoring']={'status':'not-scored','samples':0};return out
    lo=p.process_lower95.to_numpy()[mask];hi=p.process_upper95.to_numpy()[mask];actual=y[mask]
    score=hi-lo+40*np.maximum(lo-actual,0)+40*np.maximum(actual-hi,0)
    out['scoring']={'status':'complete' if mask.all() else 'partial','samples':int(mask.sum()),
        'expected_samples':len(mask),
        'metrics':{name:metrics(y,p[name].to_numpy()) for name in ('process_mean','prediction','baseline')},
        'process_band_coverage95':float(np.mean((actual>=lo)&(actual<=hi))),
        'mean_process_band_width':float(np.mean(hi-lo)),'mean_process_interval_score95':float(np.mean(score)),
        'warning':'Observation coverage is diagnostic, not calibration acceptance for process-only bands.'}
    return out
