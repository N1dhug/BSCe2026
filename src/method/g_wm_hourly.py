"""Conditional shared-baseline comparison of G-S3 and continuous WM4.

WM4's equations/structural domain are retained. Its new interval fit uses
variable projection of historical initial states; it is not the old QP fit.
G, its baseline and optional output-error companion are never refitted.
"""
from copy import deepcopy
import hashlib
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import least_squares

from src.method import bm_1_s3_interval as interval
from src.method import bm_1_s3_residual as residual
from src.validation.model.wm_4 import wm_4_validation
from src.exe.run_model.bm_1.deterministic_weekday import metrics

PROTOCOL='G-WM4-shared-baseline-hourly-mean-v1'
BOUNDS=([np.log(1e-6),0.,0.],[np.log(1e4),np.log1p(1e4),1.])
SEEDS=([np.log(1.),np.log1p(1.),.25],[np.log(100.),np.log1p(10.),.5])


def source_hashes():
    names=(*residual.source_hashes(),'src/model/wm_4.py','src/validation/model/wm_4.py','src/method/g_wm_hourly.py')
    return {p:hashlib.sha256((interval.ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in names}


def coefficients(theta):
    a=np.exp(theta[0]);extra=np.expm1(theta[1]);rho=theta[2]
    return wm_4_validation.r_coefficients([-a,-a*rho,-a*(1+rho)-extra])


def hourly_operator(r, *, fraction=1.):
    """Exact frozen-U transition and average R on fraction/24 days.

Augmented (F,R,Z,constant-U), with Z'=R. Z starts at zero each hour.
The last column integrates forcing without inverse A or eigenvalue divisions.
"""
    r=wm_4_validation.r_coefficients(r)
    if not np.isfinite(fraction) or fraction<=0:raise ValueError('Positive physical duration required.')
    matrix=np.zeros((4,4));matrix[0,1]=1.
    matrix[1,0]=r[0];matrix[1,1]=r[2];matrix[1,3]=r[1]
    matrix[2,1]=1.;h=fraction/24;transition=expm(matrix*h)
    return transition[:2,:2],transition[:2,3],transition[2,:2]/h,transition[2,3]/h


def design(prices, r):
    prices=np.asarray(prices,float)
    if prices.ndim!=1 or not len(prices) or not np.isfinite(prices).all() or np.any(abs(prices)>1):
        raise ValueError('WM requires finite signed hourly prices in [-1,1].')
    A,b,H,j=hourly_operator(r);sensitivity=np.eye(2);offset=np.zeros(2);rows=[];forcing=[]
    for u in prices:
        rows.append(H@sensitivity);forcing.append(H@offset+j*u)
        offset=A@offset+b*u;sensitivity=A@sensitivity
    return np.asarray(rows),np.asarray(forcing)


def project_initial(H,y):
    """Exact two-column least squares with F0 in [-1,1], R0 unbounded.

The unconstrained optimum and both F-boundary optima exhaust the feasible
minimum. This also supports a rank-deficient design without inverse H'H.
"""
    H=np.asarray(H,float);y=np.asarray(y,float)
    if H.shape!=(len(y),2) or len(y)<2 or not np.isfinite(H).all() or not np.isfinite(y).all():
        raise ValueError('Need at least two finite observations and a two-column design.')
    candidate=np.linalg.lstsq(H,y,rcond=None)[0];candidates=[]
    if -1<=candidate[0]<=1:candidates.append(candidate)
    norm=float(H[:,1]@H[:,1])
    for f in (-1.,1.):
        rate=0. if norm==0 else float(H[:,1]@(y-H[:,0]*f)/norm)
        candidates.append(np.array([f,rate]))
    return min(candidates,key=lambda x:float(np.sum((H@x-y)**2))).copy()


def solve(prices, baseline, r, initial):
    initial=wm_4_validation.initial_condition(initial);prices=np.asarray(prices,float)
    H,v=design(prices,r);baseline=np.asarray(baseline,float)
    if baseline.shape!=prices.shape or not np.isfinite(baseline).all():raise ValueError('Finite matched hourly baseline required.')
    A,b,_,_=hourly_operator(r);states=[initial]
    for u in prices:states.append(A@states[-1]+b*u)
    states=np.asarray(states);prediction=baseline+H@initial+v
    if not np.isfinite(prediction).all() or not np.isfinite(states).all():raise RuntimeError('Nonfinite WM interval propagation.')
    balance=float(np.max(abs(prediction-baseline-24*np.diff(states[:,0]))))
    if balance>1e-9:raise RuntimeError('WM hourly integral/energy balance check failed.')
    return {'prediction':prediction,'states':states,'energy_balance_max_error':balance}


def history(parent, raw):
    parent=interval.validate_snapshot(parent)
    train=interval.training_frame(raw,interval.point._origin(parent['origin']),parent['demand_timestamp_label'])
    if interval.point._fingerprint(train)!=parent['training_fingerprint']:raise ValueError('Own frozen G history required.')
    scale=parent['scales'];frame,_=interval.prepare_frame(train,*parent['normalization_window'],
        price_bounds=(scale['price_min'],scale['price_min']+scale['price_scale']))
    baseline=interval.restore_interval_baseline(parent['baseline']);items=[]
    for weekday,days in enumerate(parent['history']['days_by_weekday']):
        for day in days:
            inp,index=interval.day_inputs(frame,baseline,pd.Timestamp(day),history=True,steps=parent['numerics']['steps_per_hour'])
            y=np.where(inp.observation_mask[24:],inp.observations[24:],np.nan)
            items.append({'weekday':weekday,'day':day,'index':index[24:],'price':inp.prices[24:],
                'baseline':inp.b_mean[24:],'observed':y,'mask':np.isfinite(y)})
    return items


class FitBudget(RuntimeError):pass


def fit_wm(parent,raw,*,seconds=60.,max_calls=400,checkpoint=None):
    started=perf_counter();parent=interval.validate_snapshot(parent)
    if not np.isfinite(seconds) or not 0<seconds<=120 or type(max_calls) is not int or not 1<=max_calls<=1000:
        raise ValueError('WM fit budget must be <=120 seconds and <=1000 calls.')
    items=history(parent,raw);calls=0;best=None;trace=[]
    def evaluate(theta):
        r=coefficients(theta);errors=[];initials=[]
        for item in items:
            mask=item['mask']
            if np.count_nonzero(mask)<2:initials.append(None);continue
            H,v=design(item['price'],r);y=item['observed']-item['baseline']-v
            init=project_initial(H[mask],y[mask]);initials.append(init.tolist())
            errors.extend((H@init-y)[mask])
        return np.asarray(errors),initials,r
    def objective(theta):
        nonlocal calls,best
        if calls>=max_calls or perf_counter()-started>=seconds:raise FitBudget('Bounded WM interval fit stopped.')
        calls+=1;e,initials,r=evaluate(theta)
        if not len(e):raise ValueError('No usable WM historical trajectories.')
        mse=float(np.mean(e**2))
        if best is None or mse<best['mse']:
            best={'theta':theta.tolist(),'r':r.tolist(),'mse':mse,'initial_states':initials}
        if checkpoint:checkpoint({'status':'fitting','calls':calls,'seconds':perf_counter()-started,'best_provisional':best})
        return e
    converged=[]
    for seed in SEEDS:
        try:
            result=least_squares(objective,seed,bounds=BOUNDS,max_nfev=max_calls,x_scale='jac',
                ftol=1e-8,xtol=1e-8,gtol=1e-8)
            e,initials,r=evaluate(result.x)
            candidate={'theta':result.x.tolist(),'r':r.tolist(),'mse':float(np.mean(e**2)),
                'initial_states':initials,'converged':bool(result.success),'stop':str(result.message),
                'active_bounds':result.active_mask.tolist(),'jacobian_singular_values':np.linalg.svd(result.jac,compute_uv=False).tolist()}
            trace.append(candidate)
            if result.success:converged.append(candidate)
        except FitBudget as exc:trace.append({'converged':False,'stop':str(exc)});break
    if not converged:
        return {'status':'unfinished','snapshot':None,'calls':calls,'seconds':perf_counter()-started,
            'attempts':trace,'best_provisional':best}
    selected=min(converged,key=lambda x:x['mse']);r=selected['r'];starts=[];counts=[]
    for weekday in range(7):
        values=[x for item,x in zip(items,selected['initial_states']) if item['weekday']==weekday and x is not None]
        if not values:raise ValueError('No fitted starts for a forecast weekday.')
        starts.append(np.median(values,axis=0).tolist());counts.append(len(values))
    errors=[];replays=[]
    for item in items:
        out=solve(item['price'],item['baseline'],r,starts[item['weekday']])
        errors.extend((item['observed']-out['prediction'])[item['mask']])
        replays.append(pd.DataFrame({'observed':item['observed'],'wm_prediction':out['prediction'],
            'baseline':item['baseline'],'day':item['day']},index=item['index']))
    s={'kind':'WM4-shared-baseline-hourly-fit','schema_version':1,'protocol':PROTOCOL,
        'parent_snapshot_sha256':residual.snapshot_fingerprint(parent),'training_fingerprint':parent['training_fingerprint'],
        'source_sha256':source_hashes(),'origin':parent['origin'],'r_per_day':r,'forecast_states':starts,
        'fitted_starts_per_weekday':counts,'fit':selected,'fit_method':'exact hourly free-run variable projection',
        'baseline_policy':'frozen shared parent G interval baseline; not independently optimized WM baseline',
        'history_policy':'same five weeks/masks; WM initial states fitted at target start, no WM warm-up',
        'residual_samples_normalized':np.sort(errors).tolist(),
        'residual_policy':'iid pooled frozen-median replay errors, including mean; in-sample development only',
        'parameter_bounds':BOUNDS,'fixed_seeds':SEEDS,'observed_samples':len(errors)}
    return {'status':'complete','snapshot':s,'historical_replay':pd.concat(replays).sort_index(),
        'calls':calls,'seconds':perf_counter()-started,'attempts':trace,
        'budget':{'seconds':seconds,'max_calls':max_calls}}


def validate_wm(snapshot,parent):
    parent=interval.validate_snapshot(parent);s=deepcopy(snapshot)
    try:
        if (s['kind']!='WM4-shared-baseline-hourly-fit' or s['schema_version']!=1 or s['protocol']!=PROTOCOL
                or s['parent_snapshot_sha256']!=residual.snapshot_fingerprint(parent)
                or s['source_sha256']!=source_hashes() or s['training_fingerprint']!=parent['training_fingerprint']
                or s['origin']!=parent['origin'] or not s['fit']['converged']):raise ValueError('Require own compatible WM interval fit.')
        wm_4_validation.r_coefficients(s['r_per_day'])
        starts=np.asarray(s['forecast_states']);errors=np.asarray(s['residual_samples_normalized'])
        if starts.shape!=(7,2) or not np.isfinite(starts).all() or np.any(abs(starts[:,0])>1):raise ValueError('Invalid WM starts.')
        if errors.ndim!=1 or len(errors)<2 or not np.isfinite(errors).all() or np.any(np.diff(errors)<0):raise ValueError('Invalid WM errors.')
        return s
    except (KeyError,TypeError) as exc:raise ValueError('Incomplete WM interval snapshot.') from exc


def add_empirical(table,column,errors,scale=1.):
    errors=np.asarray(errors,float);low,high=np.quantile(errors,[.025,.975],method='inverted_cdf')
    table[column+'_output_mean']=table[column]+scale*float(np.mean(errors))
    table[column+'_output_variance']=scale**2*float(np.var(errors))
    table[column+'_lower']=table[column]+scale*low;table[column+'_upper']=table[column]+scale*high


def predict_comparison(parent,wm_snapshot,raw_inputs,*,g_residual=None):
    started=perf_counter();s=validate_wm(wm_snapshot,parent)
    # This canonical G forecast supplies the shared baseline/scenario/index.
    g=interval.predict_interval(parent,raw_inputs)
    if g['status']!='complete':raise RuntimeError('G mean refinement failed.')
    t=g['normalized_predictions'].copy();t=t.rename(columns={'prediction':'g_prediction'})
    values=[];state_starts=[];state_ends=[];maximum=0.;balance=0.
    for weekday,part in t.groupby('weekday',sort=True):
        out=solve(part.price_normalized.to_numpy(),part.baseline.to_numpy(),s['r_per_day'],s['forecast_states'][weekday])
        values.extend(out['prediction']);state_starts.extend(out['states'][:-1]);state_ends.extend(out['states'][1:])
        maximum=max(maximum,float(np.max(abs(out['states'][:,0]))))
        balance=max(balance,out['energy_balance_max_error'])
    t['wm_prediction']=values;t[['wm_F','wm_R']]=np.asarray(state_starts);t[['wm_F_end','wm_R_end']]=np.asarray(state_ends)
    add_empirical(t,'wm_prediction',s['residual_samples_normalized'])
    if g_residual is not None:
        r=residual.validate_residual(g_residual,parent)
        add_empirical(t,'g_prediction',r['residual_samples_normalized'])
    raw=t.copy();scale=parent['scales'];mean_columns=[c for c in t if c in ('baseline','g_prediction','wm_prediction','point_prediction')
        or c.endswith(('_output_mean','_lower','_upper'))]
    for c in mean_columns:raw[c]=scale['demand_min']+scale['demand_scale']*t[c]
    for c in (c for c in t if c.endswith('_output_variance')):raw[c]=scale['demand_scale']**2*t[c]
    manifest=deepcopy(g['manifest']);manifest.update(comparison_protocol=PROTOCOL,
        parent_snapshot_sha256=residual.snapshot_fingerprint(parent),wm_snapshot_sha256=residual.snapshot_fingerprint(s),
        wm_model='WM4 continuous affine two-state equations; exact hourly mean operator',
        wm_fit=s['fit_method'],wm_history_policy=s['history_policy'],baseline_policy=s['baseline_policy'],
        maximum_abs_forecast_WM_F=maximum,wm_F_domain_audit='outside [-1,1]' if maximum>1+1e-10 else 'within [-1,1] at hourly endpoints',
        wm_hourly_energy_balance_max_error=balance,
        uncertainty='same iid pooled empirical output-error recipe; development, not calibrated; no state diffusion',
        g_residual_attached=g_residual is not None,source_sha256=source_hashes(),
        g_residual_snapshot_sha256=None if g_residual is None else residual.snapshot_fingerprint(g_residual))
    return {'status':'complete-development-comparison','predictions':raw,'normalized_predictions':t,
        'manifest':manifest,'numerics':g['numerics'],'scoring':{'status':'not-scored'},'seconds':perf_counter()-started}


def score_comparison(result,observations):
    out=deepcopy(result);t=out['predictions']
    y=interval.align_demand(observations,out['manifest']['demand_timestamp_label']).reindex(t.index).to_numpy(float)
    y[~np.isfinite(y)]=np.nan;mask=np.isfinite(y);t['observed']=y;t['scored']=mask
    names=['baseline','g_prediction','wm_prediction','wm_prediction_output_mean']
    if out['manifest']['g_residual_attached']:names.append('g_prediction_output_mean')
    bands={}
    for name in ('g_prediction','wm_prediction'):
        if name+'_lower' not in t:continue
        low=t[name+'_lower'].to_numpy()[mask];high=t[name+'_upper'].to_numpy()[mask];actual=y[mask]
        width=high-low;value=width+40*(np.maximum(low-actual,0)+np.maximum(actual-high,0))
        bands[name]={'samples':int(mask.sum()),'coverage':None if not mask.any() else float(np.mean((actual>=low)&(actual<=high))),
            'mean_width':None if not mask.any() else float(np.mean(width)),
            'mean_interval_score':None if not mask.any() else float(np.mean(value)),
            'status':'nominal 95% empirical marginal development band; not calibration acceptance'}
    out['scoring']={'status':'complete' if mask.all() else 'partial' if mask.any() else 'not-scored',
        'expected_samples':len(y),'scored_samples':int(mask.sum()),
        'metrics':{c:metrics(y,t[c].to_numpy()) for c in names},'bands':bands,
        'policy':'same finite measured mask, raw hourly-average units; no imputation or clipping'}
    return out
