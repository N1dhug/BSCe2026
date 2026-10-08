"""Explicit 60/120 minute-aligned numerical replay/export, no parameter refit.

Keeps original weekday fit and its receipts. Own source-bound derived record
contains the unchanged model/baselines plus numerically refined median states
and the corresponding historical empirical errors. Never reads target Y.
"""
from copy import deepcopy
import hashlib
from time import perf_counter

import numpy as np
import pandas as pd

from src.method import g_weekday_rolling as rolling
from src.method.bm_1_deterministic_weekday import BM1WeekdayInputs

PROTOCOL='G-WM-five-matching-weekdays-hourly-export60-v1'
metrics=rolling.metrics
score_comparison=rolling.score_comparison


def source_hashes():
    hashes=rolling.source_hashes();name='src/method/g_weekday_export.py'
    hashes[name]=hashlib.sha256((rolling.interval.ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return hashes


def refine_for_export(parent,raw,*,seconds=15.):
    if not np.isfinite(seconds) or not 0<seconds<=15:raise ValueError('Numerical export cap is15seconds.')
    begun=perf_counter();parent=rolling.validate_snapshot(parent);view=deepcopy(parent)
    train,frame,scales,start,origin=rolling.training_data(raw,parent['origin'])
    if rolling.interval.point._fingerprint(train)!=parent['training_fingerprint'] or scales!=parent['scales']:
        raise ValueError('Exact own fitted history/scales required for numerical export.')
    replays=[]
    for day in view['days']:
        baseline=rolling.load_baseline(day['baseline']);g=day['G'];w=day['WM']
        histories=[rolling.interval.day_inputs(frame,baseline,pd.Timestamp(d),history=True,steps=60) for d in day['training_days']]
        pool=BM1WeekdayInputs([x[0] for x in histories]);curve=rolling.interval.BM1PiecewiseResponse.from_record(g['price_response'])
        coarse=pool.solve(g['parameters'],price_response=curve,deadline=begun+seconds)
        fine=pool.refined().solve(g['parameters'],price_response=curve,deadline=begun+seconds)
        error=float(np.max(abs(coarse['demand']-fine['demand'])))
        if error>1e-5:raise RuntimeError('Minute-aligned historical G refinement failed.')
        g.update(steps_per_hour=60,forecast_state=float(np.median(coarse['historical_start_states'])),
            fine_forecast_state=float(np.median(fine['historical_start_states'])),
            historical_start_states=coarse['historical_start_states'].tolist(),history_max_demand_error=error)
        ge=[];we=[];maximum=0
        for history,name in zip(histories,day['training_days']):
            if perf_counter()-begun>=seconds:raise RuntimeError('Numerical export budget stopped.')
            inp,index=history;y=np.where(inp.observation_mask[24:],inp.observations[24:],np.nan);mask=np.isfinite(y)
            gp,e=rolling.replay_g(history,g,name);maximum=max(maximum,e)
            wp=rolling.wm.solve(inp.prices[24:],inp.b_mean[24:],w['r_per_day'],w['forecast_state'])['prediction']
            ge.extend((y-gp)[mask]);we.extend((y-wp)[mask])
            replays.append(pd.DataFrame({'observed':y,'g_prediction':gp,'wm_prediction':wp,'baseline':inp.b_mean[24:],
                'weekday':day['weekday'],'day':name},index=index[24:]))
        day.update(g_residual_samples_normalized=np.sort(ge).tolist(),wm_residual_samples_normalized=np.sort(we).tolist(),replay_max_demand_error=maximum)
    if perf_counter()-begun>seconds:raise RuntimeError('Numerical export budget exceeded.')
    record={'kind':'G-WM-weekday-numerical-export','schema_version':1,'protocol':PROTOCOL,
        'source_sha256':source_hashes(),'parent_snapshot_sha256':rolling.residual.snapshot_fingerprint(parent),
        'parent':parent,'refined_view':view,'seconds':perf_counter()-begun,'optimizer_calls':0,
        'grid':{'coarse':60,'fine':120,'tolerance':1e-5},'target_observations_used':False,
        'policy':'unchanged parameters/baselines; own-history numerical starts and residual replay; minute-aligned export'}
    return {'snapshot':validate_export(record),'historical_replay':pd.concat(replays).sort_index()}


def validate_export(record):
    try:
        r=deepcopy(record)
        if (r['kind']!='G-WM-weekday-numerical-export' or r['schema_version']!=1 or r['protocol']!=PROTOCOL
                or r['source_sha256']!=source_hashes() or r['optimizer_calls']!=0 or r['target_observations_used']
                or r['grid']!={'coarse':60,'fine':120,'tolerance':1e-5} or not 0<=r['seconds']<=15):
            raise ValueError('Own compatible numerical export required.')
        parent=rolling.validate_snapshot(r['parent']);view=rolling.validate_snapshot(r['refined_view'])
        if rolling.residual.snapshot_fingerprint(parent)!=r['parent_snapshot_sha256']:raise ValueError('Export parent changed.')
        reconstructed=deepcopy(view)
        for old,new,restored in zip(parent['days'],view['days'],reconstructed['days']):
            if new['G']['steps_per_hour']!=60:raise ValueError('Numerical export must use60grid.')
            for field in ('forecast_state','fine_forecast_state','historical_start_states','steps_per_hour','history_max_demand_error'):
                restored['G'][field]=deepcopy(old['G'][field])
            for field in ('g_residual_samples_normalized','wm_residual_samples_normalized','replay_max_demand_error'):
                restored[field]=deepcopy(old[field])
        if reconstructed!=parent:raise ValueError('Numerical export changed model parameters, baseline or fit receipts.')
        return r
    except (KeyError,TypeError) as exc:raise ValueError('Incomplete numerical export record.') from exc


def predict_comparison(record,inputs):
    r=validate_export(record);result=rolling.predict_comparison(r['refined_view'],inputs)
    result['manifest'].update(protocol=PROTOCOL,comparison_protocol=PROTOCOL,source_sha256=source_hashes(),
        parent_snapshot_sha256=r['parent_snapshot_sha256'],numerical_export_sha256=rolling.residual.snapshot_fingerprint(r),
        numerical_export_policy=r['policy'],numerical_export_optimizer_calls=0)
    return result
