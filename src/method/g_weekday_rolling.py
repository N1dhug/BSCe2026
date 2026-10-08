"""Five matching weekdays fitted independently, sixth week forecast input-only.

Reuses G's bounded response/RK4 fit and WM's exact hourly variable projection.
New protocol/snapshot: pooled-parameter snapshots are never relabelled.
"""
from copy import deepcopy
import hashlib
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.method import bm_1_s3_interval as interval
from src.method import bm_1_s3_residual as residual
from src.method import g_wm_hourly as wm
from src.method.bm_1_deterministic_weekday import BM1WeekdayInputs
from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.method.bm_1_s3_interval_baseline import mean_features
from src.tools.bm_1_deterministic_snapshot import baseline_settings,restore_baseline

PROTOCOL='G-WM-five-matching-weekdays-hourly-v1'
BASELINE_IMPLEMENTATION='weekday-hourly-mean-statistical-output-v1'
metrics=wm.metrics
score_comparison=wm.score_comparison


def source_hashes():
    hashes=wm.source_hashes();name='src/method/g_weekday_rolling.py'
    hashes[name]=hashlib.sha256((interval.ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return hashes


def training_data(raw,origin):
    origin=interval.point._origin(origin);start=origin-pd.Timedelta(weeks=5)
    train=interval.training_frame(raw,origin,'hour_start')
    # Only Price/Temp from the extra 24 physical hours initialize the oldest
    # Monday. Their Demand is explicitly unavailable to scales/baseline/loss.
    warmup=(start.tz_localize(interval.ZONE)-pd.Timedelta(hours=24)).tz_localize(None)
    train=train.loc[train.index>=warmup].copy()
    train.loc[train.index<start,'Demand']=np.nan
    # Physical calendar is essential when a training window crosses spring DST.
    # The older January adapter's naive reindex would invent a missing02:00.
    index=pd.date_range(warmup.tz_localize(interval.ZONE),origin.tz_localize(interval.ZONE),freq='h',inclusive='left').tz_localize(None)
    if index.has_duplicates:raise ValueError('Ambiguous autumn local-hour labels are unsupported.')
    train=train.reindex(index).apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    window=train.loc[train.index>=start]
    if (window.Demand.count()<168 or not np.isfinite(train[['Price','Temp']].to_numpy(float)).all()
            or window.Price.min()<-500 or window.Price.max()>6000):
        raise ValueError('Need measured training demand and complete finite physical-hour Price/Temp within declared bounds.')
    scales={'demand_min':float(window.Demand.min()),'demand_scale':float(window.Demand.max()-window.Demand.min()),
        'price_min':-500.,'price_scale':6500.,'price_scale_source':'declared-ex-ante'}
    if scales['demand_scale']<=0:raise ValueError('Training demand must vary.')
    frame=pd.DataFrame({'I':(train.Demand-scales['demand_min'])/scales['demand_scale'],
        'U':2*(train.Price+500)/6500-1,'Temp':train.Temp},index=train.index)
    return train,frame,scales,start,origin


def fit_baseline(frame,start,origin,weekday):
    """Same integrated features, Huber IRLS/ridge/continuation; five days only.

The old generic baseline requires >=168 samples; five matching days have
120. This explicit adapter requires all five days and >=2 measured hours
per day. No padding/duplication, no additional targets or changed ridge.
"""
    model=interval.IntervalStatisticalBaseline()
    window=frame.loc[(frame.index>=start)&(frame.index<origin)]
    selected=window.loc[window.index.dayofweek==weekday,['I','Temp']].dropna()
    expected={str((origin+pd.Timedelta(days=weekday)-pd.Timedelta(weeks=w)).date()) for w in range(1,6)}
    counts=selected.groupby(selected.index.normalize()).size()
    if {str(d.date()) for d in counts.index}!=expected or (counts<2).any():
        raise ValueError('Baseline requires five matching measured days, no filling.')
    X=model._standardize_fit(mean_features(model,selected.index,window.Temp))
    y=selected.I.to_numpy(float);penalty=model.ridge*np.eye(X.shape[1]);penalty[0,0]=0
    weights=np.ones(len(y));beta=np.zeros(X.shape[1])
    for iterations in range(1,model.max_iter+1):
        root=np.sqrt(weights);Xw=X*root[:,None]
        beta=np.linalg.solve(Xw.T@Xw+penalty,Xw.T@(y*root))
        errors=y-X@beta;center=np.median(errors);scale=1.4826*np.median(abs(errors-center))
        if not np.isfinite(scale) or scale<1e-10:break
        limit=model.huber_delta*scale;absolute=abs(errors);next_weights=np.ones_like(weights)
        outside=absolute>limit;next_weights[outside]=limit/absolute[outside]
        delta=np.max(abs(next_weights-weights));weights=next_weights
        if delta<1e-6:break
    model.coef=beta;errors=y-np.clip(X@beta,0,1);singular=np.linalg.svd(X,compute_uv=False)
    model.metadata={'fit_start':str(selected.index.min()),'fit_end':str(selected.index.max()),
        'samples':len(y),'features':X.shape[1],'iterations':iterations,'weekday':weekday,
        'training_days':sorted(expected),'ridge':model.ridge,'huber_delta':model.huber_delta,
        'base_temperature':model.base_temperature,'temperature_min':float(selected.Temp.min()),
        'temperature_max':float(selected.Temp.max()),'mse':float(np.mean(errors**2)),
        'mae':float(np.mean(abs(errors))),'design_singular_values':singular.tolist(),
        'design_rank':int(np.count_nonzero(singular>singular[0]*1e-8)),
        'feature_observation_operator':'hourly-average trapezoid on 60 samples/hour',
        'temperature_policy':'continuous five-week historical temperatures; last value held at cutoff',
        'sample_policy':'five matching weekdays only; no target filling or duplicated observations'}
    return model


def baseline_record(model):
    return interval.point._plain({'implementation':BASELINE_IMPLEMENTATION,'settings':baseline_settings(model),
        'coef':model.coef,'feature_mean':model.feature_mean,'feature_scale':model.feature_scale,'metadata':model.metadata})


def load_baseline(record):
    if record.get('implementation')!=BASELINE_IMPLEMENTATION:raise ValueError('Own weekday baseline required.')
    common=deepcopy(record);common['implementation']='statistical_output'
    old=restore_baseline(common);model=interval.IntervalStatisticalBaseline(**record['settings'])
    for name in ('coef','feature_mean','feature_scale','metadata'):setattr(model,name,deepcopy(getattr(old,name)))
    return model


def fit_g(histories,*,seconds=30.,max_calls=750,checkpoint=None):
    if not 0<seconds<=30 or type(max_calls) is not int or not 1<=max_calls<=750:raise ValueError('G weekday cap is 30 seconds/750 calls.')
    begun=perf_counter();remaining=max_calls;previous=None;attempts=[]
    for steps in (12,24,48,60):
        if remaining<1 or perf_counter()-begun>=seconds:break
        histories=[BM1ResponseInputs(h.time,h.baseline,h.prices,h.observations,observation_contract=h.contract,
            warmup_hours=h.warmup,steps_per_hour=steps,observation_mask=h.observation_mask) for h in histories]
        pool=BM1WeekdayInputs(histories)
        fitted=interval.fit_strategy(pool,[12.,2.,.5,.5],'joint',0.,max_calls=remaining,
            calls_per_start=min(750,remaining),seconds=seconds-(perf_counter()-begun),
            checkpoint=checkpoint,initial_records=[previous] if previous is not None else None)
        remaining-=fitted['counts']['residual_calls'];attempts.append({'steps_per_hour':steps,'fit':fitted})
        best=fitted['best_converged']
        if best is None:break
        curve=interval.BM1PiecewiseResponse.from_inner_values([best['g_mid']],pieces=2)
        try:
            coarse=pool.solve(best['parameters'],price_response=curve,deadline=begun+seconds)
            fine=pool.refined().solve(best['parameters'],price_response=curve,deadline=begun+seconds)
        except interval.BudgetStop:
            attempts[-1]['numerical_check_status']='budget-stop';break
        error=float(np.max(abs(coarse['demand']-fine['demand'])))
        attempts[-1]['history_max_demand_error']=error
        if error>interval.point.TOLERANCE:previous=best;continue
        return {'status':'complete','parameters':best['parameters'],'price_response':curve.to_record(),
            'forecast_state':float(np.median(coarse['historical_start_states'])),
            'fine_forecast_state':float(np.median(fine['historical_start_states'])),
            'historical_start_states':coarse['historical_start_states'].tolist(),
            'steps_per_hour':steps,'history_max_demand_error':error,'fit':fitted,
            'attempts':attempts,'calls':max_calls-remaining,'seconds':perf_counter()-begun}
    return {'status':'unfinished','attempts':attempts,'calls':max_calls-remaining,'seconds':perf_counter()-begun}


def fit_wm(items,*,seconds=10.,max_calls=400,checkpoint=None):
    if len(items)!=5 or not 0<seconds<=10 or type(max_calls) is not int or not 1<=max_calls<=400:
        raise ValueError('WM needs exactly five days within 10 seconds/400 calls.')
    begun=perf_counter();calls=0;trace=[];converged=[];last_save=begun
    def evaluate(theta):
        r=wm.coefficients(theta);errors=[];initials=[]
        for item in items:
            mask=item['mask']
            if mask.sum()<2:raise ValueError('Each of five WM days needs >=2 measured hours.')
            H,v=wm.design(item['price'],r);y=item['observed']-item['baseline']-v
            initial=wm.project_initial(H[mask],y[mask]);initials.append(initial.tolist())
            errors.extend((H@initial-y)[mask])
        return np.asarray(errors),initials,r
    def objective(theta):
        nonlocal calls,last_save
        if calls>=max_calls or perf_counter()-begun>=seconds:raise wm.FitBudget('Weekday WM budget stopped.')
        calls+=1;errors,initials,r=evaluate(theta)
        if checkpoint and perf_counter()-last_save>=2:
            checkpoint({'status':'fitting','calls':calls,'seconds':perf_counter()-begun});last_save=perf_counter()
        return errors
    for seed in wm.SEEDS:
        try:
            fit=least_squares(objective,seed,bounds=wm.BOUNDS,max_nfev=max_calls,x_scale='jac',
                ftol=1e-8,xtol=1e-8,gtol=1e-8)
            errors,initials,r=evaluate(fit.x)
            candidate={'converged':bool(fit.success),'stop':str(fit.message),'r':r.tolist(),
                'mse':float(np.mean(errors**2)),'initial_states':initials,'theta':fit.x.tolist(),
                'active_bounds':fit.active_mask.tolist(),'jacobian_singular_values':np.linalg.svd(fit.jac,compute_uv=False).tolist()}
            trace.append(candidate)
            if fit.success:converged.append(candidate)
        except wm.FitBudget as exc:trace.append({'converged':False,'stop':str(exc)});break
    if not converged:return {'status':'unfinished','attempts':trace,'calls':calls,'seconds':perf_counter()-begun}
    best=min(converged,key=lambda x:x['mse'])
    return {'status':'complete','r_per_day':best['r'],'forecast_state':np.median(best['initial_states'],axis=0).tolist(),
        'fit':best,'attempts':trace,'calls':calls,'seconds':perf_counter()-begun}


def replay_g(history,g,day):
    inp,index=history;target=BM1ResponseInputs(inp.time[1440:]-24,inp.baseline[1440:],inp.prices[24:],inp.observations[24:],
        observation_contract='hourly-mean',steps_per_hour=inp.steps,observation_mask=inp.observation_mask[24:])
    p=list(g['parameters']);p[3]=g['forecast_state'];fine_p=p.copy();fine_p[3]=g['fine_forecast_state']
    curve=interval.BM1PiecewiseResponse.from_record(g['price_response'])
    coarse=target.solve(p,price_response=curve);fine=target.refined().solve(fine_p,price_response=curve)
    error=float(np.max(abs(coarse['demand']-fine['demand'])))
    if error>interval.point.TOLERANCE:raise RuntimeError('Weekday frozen G replay refinement failed.')
    return coarse['demand'],error


def fit_rolling(raw,origin,*,baseline_policy='weekday',checkpoint=None):
    begun=perf_counter();train,frame,scales,start,origin=training_data(raw,origin)
    if baseline_policy not in ('weekday','shared'):raise ValueError('Declare weekday or shared baseline.')
    shared=None
    if baseline_policy=='shared':shared=interval.IntervalStatisticalBaseline().fit(frame.loc[frame.index>=start,['I','Temp']],origin)
    days=[];replays=[];reports=[]
    for weekday in range(7):
        baseline=shared if shared is not None else fit_baseline(frame,start,origin,weekday)
        names=[str((origin+pd.Timedelta(days=weekday)-pd.Timedelta(weeks=w)).date()) for w in range(1,6)]
        histories=[interval.day_inputs(frame,baseline,pd.Timestamp(d),history=True) for d in names]
        hook=None if checkpoint is None else lambda x:checkpoint({'weekday':weekday,'stage':'G','fit':x})
        g=fit_g([h[0] for h in histories],checkpoint=hook)
        reports.append({'weekday':weekday,'G':g})
        if g['status']!='complete':return {'status':'unfinished','reports':reports,'snapshot':None,'seconds':perf_counter()-begun}
        histories=[interval.day_inputs(frame,baseline,pd.Timestamp(d),history=True,steps=g['steps_per_hour']) for d in names]
        items=[]
        for (inp,index),day in zip(histories,names):
            y=np.where(inp.observation_mask[24:],inp.observations[24:],np.nan)
            items.append({'price':inp.prices[24:],'baseline':inp.b_mean[24:],'observed':y,'mask':np.isfinite(y),'index':index[24:],'day':day})
        hook=None if checkpoint is None else lambda x:checkpoint({'weekday':weekday,'stage':'WM','fit':x})
        w=fit_wm(items,checkpoint=hook);reports[-1]['WM']=w
        if w['status']!='complete':return {'status':'unfinished','reports':reports,'snapshot':None,'seconds':perf_counter()-begun}
        g_errors=[];w_errors=[];max_error=0
        for history,item in zip(histories,items):
            gp,error=replay_g(history,g,item['day']);max_error=max(max_error,error)
            wp=wm.solve(item['price'],item['baseline'],w['r_per_day'],w['forecast_state'])['prediction']
            mask=item['mask'];g_errors.extend((item['observed']-gp)[mask]);w_errors.extend((item['observed']-wp)[mask])
            replays.append(pd.DataFrame({'observed':item['observed'],'g_prediction':gp,'wm_prediction':wp,
                'baseline':item['baseline'],'weekday':weekday,'day':item['day']},index=item['index']))
        days.append({'weekday':weekday,'training_days':names,'baseline':baseline_record(baseline),
            'G':g,'WM':w,'target_measured_masks':[item['mask'].tolist() for item in items],
            'g_residual_samples_normalized':np.sort(g_errors).tolist(),
            'wm_residual_samples_normalized':np.sort(w_errors).tolist(),'replay_max_demand_error':max_error})
        if checkpoint:checkpoint({'weekday':weekday,'stage':'complete','day':days[-1]})
    snapshot={'kind':'G-WM-weekday-rolling-fit','schema_version':1,'protocol':PROTOCOL,'origin':str(origin),
        'training_start':str(start),'training_end_exclusive':str(origin),'normalization_window':[str(start),str(origin)],
        'training_fingerprint':interval.point._fingerprint(train),'source_sha256':source_hashes(),'scales':scales,
        'baseline_policy':baseline_policy,'days':days,'demand_timestamp_label':'hour_start',
        'population':'MED mean over available measured meters','demand_unit':'raw average-power units; physical unit unverified',
        'timestamp_interpretation_status':'caller working assumption; not independently verified',
        'fit_policy':'five matching weekdays only; seven independent parameter fits; same baseline for G/WM within weekday',
        'normalization_policy':'common scales from five training weeks; no older or target demand',
        'initialization_policy':'G neutral .5 plus 24 physical hours input-only warmup, median starts; WM target-start variable projection, median starts',
        'residual_policy':'separate weekday empirical frozen-median replay errors; in-sample development; no process diffusion',
        'budgets_per_weekday':{'G_seconds':30,'G_calls':750,'WM_seconds':10,'WM_calls':400}}
    return {'status':'complete','snapshot':validate_snapshot(snapshot),'historical_replay':pd.concat(replays).sort_index(),
        'reports':reports,'seconds':perf_counter()-begun}


def validate_snapshot(snapshot):
    try:
        s=deepcopy(snapshot);origin=interval.point._origin(s['origin']);start=origin-pd.Timedelta(weeks=5)
        if (s['kind']!='G-WM-weekday-rolling-fit' or s['schema_version']!=1 or s['protocol']!=PROTOCOL
                or s['source_sha256']!=source_hashes() or pd.Timestamp(s['training_start'])!=start
                or pd.Timestamp(s['training_end_exclusive'])!=origin or s['normalization_window']!=[str(start),str(origin)]
                or len(s['days'])!=7 or s['baseline_policy'] not in ('weekday','shared') or s['demand_timestamp_label']!='hour_start'):
            raise ValueError('Require compatible own rolling weekday fit.')
        scale=np.array([s['scales'][k] for k in ('demand_min','demand_scale','price_min','price_scale')])
        if not np.isfinite(scale).all() or scale[1]<=0 or scale[3]<=0:raise ValueError('Invalid rolling scales.')
        for weekday,day in enumerate(s['days']):
            expected=[str((origin+pd.Timedelta(days=weekday)-pd.Timedelta(weeks=w)).date()) for w in range(1,6)]
            if day['weekday']!=weekday or day['training_days']!=expected:raise ValueError('Wrong weekday training days.')
            baseline=load_baseline(day['baseline'])
            if pd.Timestamp(baseline.metadata['fit_start'])<start or pd.Timestamp(baseline.metadata['fit_end'])>=origin:
                raise ValueError('Baseline crosses five-week boundaries.')
            if s['baseline_policy']=='weekday' and baseline.metadata['training_days']!=sorted(expected):raise ValueError('Wrong baseline weekdays.')
            g=day['G'];w=day['WM'];p=interval._parameters(g['parameters']);curve=interval.BM1PiecewiseResponse.from_record(g['price_response'])
            if (g['status']!='complete' or g['fit']['status']!='converged' or w['status']!='complete' or not w['fit']['converged']
                    or not 1<=p[0]<=744 or not .05<=p[1]<=10 or p[3]!=.5
                    or not np.array_equal(p,g['fit']['best_converged']['parameters'])
                    or curve.values[1]!=g['fit']['best_converged']['g_mid'] or g['steps_per_hour'] not in (12,24,48,60)
                    or not 0<=g['history_max_demand_error']<=1e-5 or not 0<=day['replay_max_demand_error']<=1e-5
                    or not 0<=g['forecast_state']<=1 or not 0<=g['fine_forecast_state']<=1):
                raise ValueError('Incomplete or numerically invalid weekday fit.')
            wm.wm_4_validation.r_coefficients(w['r_per_day']);wm.wm_4_validation.initial_condition(w['forecast_state'])
            if (not np.array_equal(w['r_per_day'],w['fit']['r']) or
                    not np.array_equal(w['r_per_day'],wm.coefficients(w['fit']['theta']))):raise ValueError('WM coefficients disagree with own fit.')
            masks=day['target_measured_masks']
            if len(masks)!=5 or any(len(m) not in (23,24) or any(type(v) is not bool for v in m) for m in masks):raise ValueError('Five own masks required.')
            n=sum(sum(m) for m in masks)
            if n!=g['fit']['observed_samples']:raise ValueError('G samples and weekday masks disagree.')
            for model in ('g','wm'):
                errors=np.asarray(day[model+'_residual_samples_normalized'],float)
                if errors.shape!=(n,) or n<2 or not np.isfinite(errors).all() or np.any(np.diff(errors)<0) or np.var(errors)<=0:
                    raise ValueError('Own finite positive-variance weekday residual bank required.')
        return s
    except (KeyError,TypeError,OverflowError) as exc:raise ValueError('Incomplete rolling snapshot.') from exc


def predict_comparison(snapshot,raw_inputs):
    begun=perf_counter();s=validate_snapshot(snapshot);origin=pd.Timestamp(s['origin']);end=origin+pd.Timedelta(days=7)
    interval.point._frame(raw_inputs,('Price','Temp'));scenario=raw_inputs.loc[(raw_inputs.index>=origin)&(raw_inputs.index<end),['Price','Temp']].copy()
    if not np.isfinite(scenario.to_numpy(float)).all():raise ValueError('Finite declared scenario required.')
    scale=s['scales'];frame=pd.DataFrame({'U':2*(scenario.Price-scale['price_min'])/scale['price_scale']-1,'Temp':scenario.Temp},index=scenario.index)
    rows=[];errors=[];balance=0;max_F=0
    for day in s['days']:
        d=day['weekday'];g=day['G'];w=day['WM'];baseline=load_baseline(day['baseline'])
        inp,index=interval.day_inputs(frame,baseline,origin+pd.Timedelta(days=d),steps=g['steps_per_hour'])
        p=list(g['parameters']);p[3]=g['forecast_state'];q=p.copy();q[3]=g['fine_forecast_state']
        curve=interval.BM1PiecewiseResponse.from_record(g['price_response'])
        coarse=inp.solve(p,price_response=curve);fine=inp.refined().solve(q,price_response=curve)
        errors.append(float(np.max(abs(coarse['demand']-fine['demand']))))
        out=wm.solve(inp.prices,inp.b_mean,w['r_per_day'],w['forecast_state']);balance=max(balance,out['energy_balance_max_error'])
        max_F=max(max_F,float(np.max(abs(out['states'][:,0]))))
        t=pd.DataFrame({'g_prediction':coarse['demand'],'wm_prediction':out['prediction'],'baseline':inp.b_mean,
            'state':coarse['state'][:-1],'state_end':coarse['state'][1:],
            'wm_F':out['states'][:-1,0],'wm_R':out['states'][:-1,1],
            'wm_F_end':out['states'][1:,0],'wm_R_end':out['states'][1:,1],
            'weekday':d,'price':scenario.reindex(index).Price.to_numpy(),'price_normalized':inp.prices},index=index)
        wm.add_empirical(t,'g_prediction',day['g_residual_samples_normalized']);wm.add_empirical(t,'wm_prediction',day['wm_residual_samples_normalized'])
        rows.append(t)
    if max(errors)>interval.point.TOLERANCE:raise RuntimeError('Rolling forecast G mean refinement failed.')
    table=pd.concat(rows);expected=pd.date_range(origin.tz_localize(interval.ZONE),end.tz_localize(interval.ZONE),freq='h',inclusive='left').tz_localize(None)
    if not table.index.equals(expected):raise ValueError('Incomplete physical forecast week.')
    physical=table.index.tz_localize(interval.ZONE).tz_convert('UTC');table['interval_start_utc']=physical.astype(str)
    table['interval_end_utc']=(physical+pd.Timedelta(hours=1)).astype(str);raw=table.copy()
    for c in table:
        if c in ('baseline','g_prediction','wm_prediction') or c.endswith(('_output_mean','_lower','_upper')):
            raw[c]=scale['demand_min']+scale['demand_scale']*table[c]
        elif c.endswith('_output_variance'):raw[c]=scale['demand_scale']**2*table[c]
    return {'status':'complete-development-comparison','predictions':raw,'normalized_predictions':table,
        'manifest':{'protocol':PROTOCOL,'comparison_protocol':PROTOCOL,'source_sha256':source_hashes(),
            'cutoff':str(origin),'end_exclusive':str(end),'physical_hours':len(table),'timezone':interval.ZONE,
            'scales':deepcopy(scale),'demand_timestamp_label':'hour_start','output_timestamp_label':'hour_start',
            'demand_unit':s['demand_unit'],'population':s['population'],'timestamp_interpretation_status':s['timestamp_interpretation_status'],
            'demand_used':False,'g_residual_attached':True,'model_refitted':False,'baseline_refitted':False,
            'parent_snapshot_sha256':residual.snapshot_fingerprint(s),'scenario_fingerprint':interval.point._fingerprint(scenario),
            'baseline_policy':s['baseline_policy'],'fit_policy':s['fit_policy'],'normalization_policy':s['normalization_policy'],
            'training_start':s['training_start'],'training_end_exclusive':s['training_end_exclusive'],
            'state_path':'seven independent weekday paths frozen at Monday cutoff',
            'scenario':'conditional on realized future price/temperature; not weather forecasting',
            'uncertainty':s['residual_policy'],'wm_hourly_energy_balance_max_error':balance,
            'maximum_abs_forecast_WM_F':max_F,'wm_F_domain_audit':'outside [-1,1]' if max_F>1+1e-10 else 'within [-1,1] at hourly endpoints'},
        'numerics':{'passed':True,'tolerance':1e-5,'daily_max_demand_errors':errors},
        'scoring':{'status':'not-scored'},'seconds':perf_counter()-begun}
