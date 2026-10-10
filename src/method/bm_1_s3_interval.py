"""Deterministic G hourly-average fit, frozen snapshot and prediction."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.tools import bm_1_g_contract as contract
from src.method.bm_1_deterministic_fit import BudgetStop, _parameters
from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.method.bm_1_deterministic_weekday import BM1PooledInputs
from src.method.bm_1_deterministic_strategy import fit_strategy
from src.params.bm_1_response import BM1PiecewiseResponse
from src.method.bm_1_s3_interval_baseline import IntervalStatisticalBaseline,IMPLEMENTATION,restore_interval_baseline
from src.tools.bm_1_g_contract import ZONE
from src.tools.bm_1_g_baseline import baseline_settings
from src.exe.run_model.bm_1.s3_data import prepare_frame

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL='G-S3-five-weeks-Monday-frozen-hourly-mean-v1'
CONTRACT='hourly-mean-power-development-units-unverified'
SOURCES=('src/method/bm_1_s3_interval.py', 'src/method/bm_1_s3_interval_baseline.py', 'src/method/bm_1_deterministic_fit.py', 'src/method/bm_1_deterministic_response.py', 'src/method/bm_1_deterministic_weekday.py', 'src/method/bm_1_deterministic_strategy.py', 'src/tools/bm_1_g_contract.py', 'src/tools/bm_1_g_baseline.py', 'src/exe/run_model/bm_1/s3_data.py', 'src/params/bm_1_response.py', 'src/validation/estimation.py', 'src/exe/run_model/wm_4/output_model.py')


def source_hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in SOURCES}


def align_demand(observations, label):
    """Price/weather always label interval starts; demand labels are explicit.

    End labels shift by one PHYSICAL hour. Nonexistent finite observations
    cannot be repaired. Empty nonexistent calendar placeholders are ignored.
    Autumn ambiguity is rejected, consistently with the original S3 adapter.
    """
    if label not in ('hour_start','hour_end'):raise ValueError('Declare hour_start or hour_end demand labels.')
    if (not isinstance(observations,pd.Series) or not isinstance(observations.index,pd.DatetimeIndex)
            or observations.index.tz is not None or observations.index.has_duplicates
            or observations.index.hasnans or not observations.index.is_monotonic_increasing):
        raise ValueError('Demand requires sorted unique local-naive hourly labels.')
    if np.any(observations.index!=observations.index.floor('h')):raise ValueError('Demand labels must be hourly.')
    out=pd.to_numeric(observations,errors='coerce').copy()
    if label=='hour_end':
        physical=out.index.tz_localize(ZONE,ambiguous='raise',nonexistent='NaT')
        missing=physical.isna()
        if np.any(missing&np.isfinite(out.to_numpy(float))):
            raise ValueError('Finite demand at a nonexistent end timestamp is invalid.')
        out=out.loc[~missing].copy();physical=physical[~missing]
        out.index=(physical-pd.Timedelta(hours=1)).tz_localize(None)
    return out


def training_frame(raw, origin, label):
    contract._frame(raw,('Demand','Price','Temp'))
    # End-labelled observation at the cutoff describes the preceding hour;
    # future exogenous inputs and later observations remain unavailable.
    end=raw.index<=origin if label=='hour_end' else raw.index<origin
    observations=align_demand(raw.loc[end,'Demand'],label)
    train=raw.loc[raw.index<origin,['Price','Temp']].copy()
    train.insert(0,'Demand',observations.reindex(train.index))
    return train


def day_inputs(frame, baseline, day, *, history=False, steps=12):
    original,index=contract._day_inputs(frame,baseline,day,history=history,steps=steps)
    # Use the next declared temperature if it is in this historical/scenario
    # frame. At the information/window boundary hold the last known value.
    start=index[0].tz_localize(ZONE).tz_convert('UTC')
    end=(start+pd.Timedelta(hours=original.n_hours)).tz_convert(ZONE).tz_localize(None)
    temp=frame.reindex(index).Temp.to_numpy(float);hours=np.arange(len(index),dtype=float)
    if end in frame.index:
        value=float(frame.loc[end,'Temp'])
        if not np.isfinite(value):raise ValueError('Finite declared endpoint temperature required.')
        temp=np.r_[temp,value];hours=np.r_[hours,len(index)]
    fine_index=(start+pd.to_timedelta(np.arange(len(original.time)),unit='min')).tz_convert(ZONE)
    B=baseline.predict(fine_index,np.interp(original.time,hours,temp))
    return BM1ResponseInputs(original.time,B,original.prices,original.observations,
        observation_contract='hourly-mean',warmup_hours=original.warmup,
        steps_per_hour=steps,observation_mask=original.observation_mask),index


def validate_snapshot(snapshot):
    """Fail closed on incompatible fitted data; no fits or source migrations."""
    try:
        s = deepcopy(snapshot)
        if (type(s['schema_version']) is not int or s['schema_version'] != 1
                or s['kind'] != 'G-S3-interval-fit' or s['protocol'] != PROTOCOL):
            raise ValueError('Unsupported S3 snapshot version/protocol.')
        origin = contract._origin(s['origin'])
        if s['source_sha256'] != source_hashes():
            raise ValueError('S3 numerical sources changed; an explicit reviewed migration is required.')
        if (s['time_unit'] != 'hours' or s['timezone'] != ZONE
                or s['observation_contract'] != CONTRACT
                or s['demand_timestamp_label'] not in ('hour_start','hour_end')
                or s['timestamp_interpretation_status'] != 'caller working assumption; not independently verified'
                or s['fit']['objective_operator'] != 'hourly-mean'
                or s['lambda'] != 0 or s['fit']['status'] != 'converged'):
            raise ValueError('Snapshot must be a converged S3 lambda=0 fit with declared units.')
        p = _parameters(s['parameters'])
        if p[3] != .5 or not 1 <= p[0] <= 744 or not .05 <= p[1] <= 10:
            raise ValueError('Snapshot parameters disagree with S3 fit bounds/neutral history start.')
        curve = BM1PiecewiseResponse.from_record(s['price_response'])
        if not np.array_equal(curve.knots, [0., .5, 1.]):
            raise ValueError('S3 needs the selected two-piece price response.')
        best = s['fit']['best_converged']
        if not np.array_equal(p, best['parameters']) or curve.values[1] != best['g_mid']:
            raise ValueError('Saved fit and response parameters disagree.')
        n = s['numerics']
        steps = n['steps_per_hour']
        if type(steps) is not int or steps not in (12, 24, 48) or n['fine_steps_per_hour'] != 2*steps:
            raise ValueError('Invalid S3 coarse/fine grids.')
        if n['tolerance'] != contract.TOLERANCE or not n['passed'] or not 0 <= n['history_max_demand_error'] <= contract.TOLERANCE:
            raise ValueError('Snapshot requires the passed S3 historical refinement check.')
        for name in ('forecast_states', 'fine_forecast_states'):
            states = np.asarray(s[name], float)
            if states.shape != (7,) or not np.isfinite(states).all() or np.any((states < 0) | (states > 1)):
                raise ValueError('Snapshot requires seven own finite daily starts in [0,1].')
        history = s['history']
        expected = [[str((origin+pd.Timedelta(days=d)-pd.Timedelta(weeks=w)).date())
                     for w in range(1, 6)] for d in range(7)]
        if (history['days_by_weekday'] != expected or history['warmup_physical_hours'] != 24
                or history['start_state'] != .5 or history['forecast_start_rule'] != 'median of five own states after 24 physical hours'):
            raise ValueError('Snapshot history/initialization contract disagrees with S3.')
        masks = history['target_measured_masks']
        if len(masks) != 35 or any(len(mask) not in (23, 24) or any(type(v) is not bool for v in mask) for mask in masks):
            raise ValueError('Snapshot needs the 35 historical target masks.')
        if sum(sum(mask) for mask in masks) != s['fit']['observed_samples']:
            raise ValueError('Fit sample count and historical masks disagree.')
        scales = s['scales']
        v = np.array([scales[k] for k in ('demand_min', 'demand_scale', 'price_min', 'price_scale')], float)
        if not np.isfinite(v).all() or v[1] <= 0 or v[3] <= 0:
            raise ValueError('Invalid frozen scales.')
        a, b = map(pd.Timestamp, s['normalization_window'])
        if a.tz is not None or b.tz is not None or not a < b <= origin:
            raise ValueError('Normalization must use historical data only.')
        baseline = restore_interval_baseline(s['baseline'])
        if pd.Timestamp(baseline.metadata['fit_end']) >= origin:
            raise ValueError('Baseline metadata crosses forecast cutoff.')
        if not isinstance(s['training_fingerprint'], str) or len(s['training_fingerprint']) != 64:
            raise ValueError('Missing historical input fingerprint.')
        return s
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Incomplete or malformed S3 snapshot.') from exc


def save_snapshot(path,snapshot):
    from src.tools.bm_1_reports import _atomic_report
    _atomic_report(Path(path),contract._plain(validate_snapshot(snapshot)))


def load_snapshot(path):
    with Path(path).open(encoding='utf-8') as handle:return validate_snapshot(json.load(handle))


def fit_interval(raw,origin,*,demand_label,reference=(12.,2.,.5,.5),calibration_start=None,
                 calibration_end=None,price_bounds=(-500.,6000.),seconds=600.,max_calls=3000,
                 checkpoint=None,population='declared raw aggregate',demand_unit='raw average-power units; physical unit unverified'):
    started=perf_counter();origin=contract._origin(origin);reference=_parameters(reference)
    if (not np.isfinite(seconds) or not 0<seconds<=600 or type(max_calls) is not int or not 1<=max_calls<=3000):
        raise ValueError('Fit budget must be <=600 seconds and <=3000 residual calls.')
    train=training_frame(raw,origin,demand_label)
    a=origin-pd.Timedelta(weeks=5) if calibration_start is None else pd.Timestamp(calibration_start)
    b=origin if calibration_end is None else pd.Timestamp(calibration_end)
    if not a<b<=origin:raise ValueError('Normalization must precede cutoff.')
    frame,scales=prepare_frame(train,a,b,price_bounds=price_bounds)
    baseline=IntervalStatisticalBaseline().fit(frame[['I','Temp']],origin)
    histories=[];days=[]
    for weekday in range(7):
        group=[]
        for lag in range(1,6):
            day=origin+pd.Timedelta(days=weekday)-pd.Timedelta(weeks=lag)
            histories.append(day_inputs(frame,baseline,day,history=True)[0]);group.append(str(day.date()))
        days.append(group)
    attempts=[];remaining=max_calls;previous=None
    for steps in (12,24,48):
        if remaining<1 or perf_counter()-started>=seconds:break
        while histories[0].steps<steps:histories=[h.refined() for h in histories]
        pool=BM1PooledInputs(histories)
        fitted=fit_strategy(pool,reference,max_calls=remaining,calls_per_start=min(750,remaining),
            seconds=seconds-(perf_counter()-started),checkpoint=checkpoint,
            initial_records=[previous] if previous is not None else None)
        fitted['objective_operator']='hourly-mean'
        remaining-=fitted['counts']['residual_calls'];attempts.append({'steps_per_hour':steps,'fit':fitted})
        best=fitted['best_converged']
        if best is None:break
        curve=BM1PiecewiseResponse.from_inner_values([best['g_mid']],pieces=2)
        try:
            coarse=pool.solve(best['parameters'],price_response=curve,deadline=started+seconds)
            fine=pool.refined().solve(best['parameters'],price_response=curve,deadline=started+seconds)
        except BudgetStop:
            attempts[-1]['numerical_check_status']='budget-stop';break
        error=float(np.max(abs(coarse['demand']-fine['demand'])))
        attempts[-1]['history_max_demand_error']=error
        if error>contract.TOLERANCE:previous=best;continue
        record=contract._plain({'implementation':IMPLEMENTATION,'settings':baseline_settings(baseline),
            'coef':baseline.coef,'feature_mean':baseline.feature_mean,'feature_scale':baseline.feature_scale,
            'metadata':baseline.metadata})
        s={'schema_version':1,'kind':'G-S3-interval-fit','protocol':PROTOCOL,'origin':str(origin),
            'time_unit':'hours','timezone':ZONE,'lambda':0.,'observation_contract':CONTRACT,
            'demand_timestamp_label':demand_label,'parameters':best['parameters'],'price_response':curve.to_record(),
            'timestamp_interpretation_status':'caller working assumption; not independently verified',
            'forecast_states':[float(np.median(coarse['historical_start_states'][5*d:5*d+5])) for d in range(7)],
            'fine_forecast_states':[float(np.median(fine['historical_start_states'][5*d:5*d+5])) for d in range(7)],
            'baseline':record,'scales':scales,'normalization_window':[str(a),str(b)],
            'population':population,'demand_unit':demand_unit,'training_fingerprint':contract._fingerprint(train),
            'source_sha256':source_hashes(),'history':{'days_by_weekday':days,'warmup_physical_hours':24,
                'start_state':.5,'forecast_start_rule':'median of five own states after 24 physical hours',
                'target_measured_masks':[h.observation_mask[24:].tolist() for h in histories]},
            'fit':fitted,'numerics':{'steps_per_hour':steps,'fine_steps_per_hour':2*steps,
                'tolerance':contract.TOLERANCE,'history_max_demand_error':error,'passed':True}}
        return {'status':'complete','snapshot':validate_snapshot(s),'attempts':attempts,
            'seconds':perf_counter()-started,'calls_used':max_calls-remaining,
            'budget':{'seconds':seconds,'max_calls':max_calls}}
    return {'status':'unfinished','snapshot':None,'attempts':attempts,'seconds':perf_counter()-started,
        'calls_used':max_calls-remaining,'budget':{'seconds':seconds,'max_calls':max_calls}}


def predict_interval(snapshot,raw_inputs):
    started=perf_counter();s=validate_snapshot(snapshot);origin=contract._origin(s['origin']);end=origin+pd.Timedelta(days=7)
    contract._frame(raw_inputs,('Price','Temp'))
    scenario=raw_inputs.loc[(raw_inputs.index>=origin)&(raw_inputs.index<end),['Price','Temp']].copy()
    if not np.isfinite(scenario.to_numpy(float)).all():raise ValueError('Finite future price/temperature required.')
    scales=s['scales'];frame=pd.DataFrame({'U':2*(scenario.Price-scales['price_min'])/scales['price_scale']-1,
        'Temp':scenario.Temp},index=scenario.index)
    baseline=restore_interval_baseline(s['baseline']);curve=BM1PiecewiseResponse.from_record(s['price_response'])
    rows=[];errors=[]
    for d in range(7):
        inp,index=day_inputs(frame,baseline,origin+pd.Timedelta(days=d),steps=s['numerics']['steps_per_hour'])
        p=list(s['parameters']);p[3]=s['forecast_states'][d];q=p.copy();q[3]=s['fine_forecast_states'][d]
        coarse=inp.solve(p,price_response=curve);fine=inp.refined().solve(q,price_response=curve)
        errors.append(float(np.max(abs(coarse['demand']-fine['demand']))))
        rows.append(pd.DataFrame({'prediction':coarse['demand'],'baseline':inp.b_mean,
            'point_prediction':coarse['point_demand'],'state':coarse['state'][:-1],
            'state_end':coarse['state'][1:],'price':scenario.reindex(index).Price.to_numpy(),
            'price_normalized':inp.prices,'weekday':d,'horizon_hours':np.arange(len(index)),
            'cutoff':str(origin),'protocol':PROTOCOL},index=index))
    table=pd.concat(rows);physical=table.index.tz_localize(ZONE).tz_convert('UTC')
    table['interval_start_utc']=physical.astype(str)
    table['interval_end_utc']=(physical+pd.Timedelta(hours=1)).astype(str)
    table['elapsed_hours']=np.arange(len(table));raw=table.copy()
    for col in ('prediction','baseline','point_prediction'):raw[col]=scales['demand_min']+scales['demand_scale']*table[col]
    return {'status':'complete' if max(errors)<=contract.TOLERANCE else 'numerics-failed',
        'predictions':raw,'normalized_predictions':table,'manifest':{'protocol':PROTOCOL,'cutoff':str(origin),
            'end_exclusive':str(end),'timezone':ZONE,'physical_hours':len(table),'scales':deepcopy(scales),
            'population':s['population'],'demand_unit':s['demand_unit'],'observation_contract':CONTRACT,
            'demand_timestamp_label':s['demand_timestamp_label'],'output_timestamp_label':'hour_start',
            'timestamp_interpretation_status':s['timestamp_interpretation_status'],
            'observation_available':'interval_end','scenario_fingerprint':contract._fingerprint(scenario),
            'demand_used':False,'model_refitted':False,'baseline_refitted':False,
            'scenario':'conditional on future price/temperature','state_path':'seven independent daily paths frozen Monday',
            'stochastic_status':'not-attached'},'numerics':{'passed':max(errors)<=contract.TOLERANCE,
                'tolerance':contract.TOLERANCE,'daily_max_demand_errors':errors},
        'scoring':{'status':'not-scored'},'seconds':perf_counter()-started}


def score_interval(result,observations):
    aligned=align_demand(observations,result['manifest']['demand_timestamp_label'])
    out=contract.score_forecast(result,aligned)
    out['scoring']['policy']='raw average-power units; completed physical hour; same finite measured mask; no filling'
    return out
