"""Explicit empirical output-error companion to frozen deterministic G-S3.

Y_h = G_h + E_h, E_h iid from pooled historical replay errors. This is a
development model, not identified measurement noise or calibrated coverage.
No Gaussian replacement of nonlinear process moments is made.
"""
from copy import deepcopy
import hashlib
import json
from time import perf_counter

import numpy as np
import pandas as pd

from src.method import bm_1_s3_interval as interval
from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.method.bm_1_s3_uncertainty import demand_values
from src.params.bm_1_response import BM1PiecewiseResponse
from src.exe.run_model.bm_1.deterministic_weekday import metrics

PROTOCOL = 'G-S3-pooled-empirical-output-error-v1'


def source_hashes():
    names = (*interval.SOURCES, 'src/method/bm_1_s3_uncertainty.py', 'src/method/bm_1_s3_residual.py')
    return {p: hashlib.sha256((interval.ROOT/p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in names}


def snapshot_fingerprint(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def summarize_replay(table):
    """Finite mask, support-distance lower bound and within-day lag-one audit."""
    mask = np.isfinite(table.observed.to_numpy(float))
    t = table.loc[mask];e = (t.observed-t.prediction).to_numpy(float)
    if len(e)<2:raise ValueError('At least two finite historical errors required.')
    distance = np.maximum(np.maximum(t.lower-t.observed, t.observed-t.upper), 0).to_numpy(float)
    left=[];right=[]
    # Missing hours must not be bridged; daily starts reset the model.
    for _, day in table.groupby('day'):
        physical = pd.DatetimeIndex(pd.to_datetime(day.interval_start_utc, utc=True))
        values = (day.observed-day.prediction).to_numpy(float)
        valid = np.isfinite(values[:-1]) & np.isfinite(values[1:])
        valid &= np.diff(physical.asi8)==pd.Timedelta(hours=1).value
        left.extend(values[:-1][valid]);right.extend(values[1:][valid])
    correlation = None
    if len(left)>=2 and np.std(left)>0 and np.std(right)>0:
        correlation = float(np.corrcoef(left, right)[0,1])
    return {'observed_samples':len(e), 'error_mean':float(np.mean(e)),
        'error_variance_empirical_ddof0':float(np.var(e)),
        'error_rmse':float(np.sqrt(np.mean(e**2))),
        'baseline_rmse':metrics(t.observed.to_numpy(),t.baseline.to_numpy())['rmse'],
        'mean_abs_G_minus_baseline':float(np.mean(abs(t.prediction-t.baseline))),
        'outside_integrated_support':int(np.count_nonzero(distance>0)),
        'support_distance_rmse_lower_bound':float(np.sqrt(np.mean(distance**2))),
        'within_day_adjacent_pairs':len(left), 'within_day_lag_one_correlation':correlation,
        'interpretation':'in-sample replay; lag correlation diagnoses iid assumption, not a selection gate'}


def fit_residual(snapshot, raw_history, *, seconds=30.):
    started=perf_counter();s=interval.validate_snapshot(snapshot)
    if not np.isfinite(seconds) or not 0<seconds<=60:raise ValueError('Replay budget must be in (0,60] seconds.')
    train=interval.training_frame(raw_history,interval.point._origin(s['origin']),s['demand_timestamp_label'])
    if interval.point._fingerprint(train)!=s['training_fingerprint']:raise ValueError('Require own fitted history.')
    scale=s['scales'];frame,_=interval.prepare_frame(train,*s['normalization_window'],
        price_bounds=(scale['price_min'],scale['price_min']+scale['price_scale']))
    baseline=interval.restore_interval_baseline(s['baseline'])
    curve=BM1PiecewiseResponse.from_record(s['price_response']);rows=[];maximum=0.
    for weekday, days in enumerate(s['history']['days_by_weekday']):
        for day in days:
            inp,index=interval.day_inputs(frame,baseline,pd.Timestamp(day),history=True,steps=s['numerics']['steps_per_hour'])
            # Replay the deployed frozen median initialization, not a fitted
            # state obtained by observing the replay target day.
            n=24*60
            target=BM1ResponseInputs(inp.time[n:]-24,inp.baseline[n:],inp.prices[24:],inp.observations[24:],
                observation_contract='hourly-mean',steps_per_hour=inp.steps,observation_mask=inp.observation_mask[24:])
            p=list(s['parameters']);p[3]=s['forecast_states'][weekday]
            q=p.copy();q[3]=s['fine_forecast_states'][weekday]
            coarse=target.solve(p,price_response=curve,deadline=started+seconds)
            fine=target.refined().solve(q,price_response=curve,deadline=started+seconds)
            maximum=max(maximum,float(np.max(abs(coarse['demand']-fine['demand']))))
            lower=demand_values(1.,target.b_mean,curve(target.prices),p[1],p[2])
            upper=demand_values(0.,target.b_mean,curve(target.prices),p[1],p[2])
            rows.append(pd.DataFrame({'observed':np.where(target.observation_mask,target.observations,np.nan),'prediction':coarse['demand'],
                'baseline':target.b_mean,'lower':lower,'upper':upper,'day':day,
                'interval_start_utc':index[24:].tz_localize(interval.ZONE).tz_convert('UTC').astype(str)},index=index[24:]))
    if maximum>1e-5:raise RuntimeError('Frozen replay mean refinement failed; no residual snapshot.')
    table=pd.concat(rows).sort_index();audit=summarize_replay(table)
    errors=(table.observed-table.prediction).to_numpy(float);errors=np.sort(errors[np.isfinite(errors)])
    companion={'kind':'G-S3-empirical-output-error','schema_version':1,'protocol':PROTOCOL,
        'parent_snapshot_sha256':snapshot_fingerprint(s),'source_sha256':source_hashes(),
        'cutoff':s['origin'],'training_fingerprint':s['training_fingerprint'],
        'residual_samples_normalized':errors.tolist(),'sample_count':len(errors),
        'law':'iid pooled empirical errors, including their historical mean',
        'fit_policy':'same five-week training data as parent; frozen weekday medians replayed on all historical targets',
        'coverage_status':'development only; no out-of-sample calibration guarantee',
        'process_sigma':0.,'audit':audit,'replay_mean_refinement_error':maximum}
    return {'status':'complete-development-residual','snapshot':companion,'historical_replay':table,
        'seconds':perf_counter()-started,'optimizer_calls':0,'model_refitted':False,'baseline_refitted':False}


def validate_residual(record, parent):
    parent=interval.validate_snapshot(parent);r=deepcopy(record)
    try:
        errors=np.asarray(r['residual_samples_normalized'],float)
        if (r['kind']!='G-S3-empirical-output-error' or r['schema_version']!=1 or r['protocol']!=PROTOCOL
                or r['parent_snapshot_sha256']!=snapshot_fingerprint(parent) or r['source_sha256']!=source_hashes()
                or r['training_fingerprint']!=parent['training_fingerprint'] or r['cutoff']!=parent['origin']
                or r['process_sigma']!=0. or errors.ndim!=1 or len(errors)<2 or len(errors)!=r['sample_count']
                or not np.isfinite(errors).all() or np.any(np.diff(errors)<0)):
            raise ValueError('Require compatible own residual snapshot and unchanged parent.')
        return r
    except (KeyError,TypeError) as exc:raise ValueError('Incomplete residual snapshot.') from exc


def predict_residual(parent, companion, raw_inputs, *, confidence=.95):
    started=perf_counter();r=validate_residual(companion,parent)
    if not np.isfinite(confidence) or not 0<confidence<1:raise ValueError('Confidence must lie in (0,1).')
    out=interval.predict_interval(parent,raw_inputs)
    if out['status']!='complete':raise RuntimeError('Parent forecast refinement failed.')
    errors=np.asarray(r['residual_samples_normalized']);alpha=1-confidence
    low,high=np.quantile(errors,[alpha/2,1-alpha/2],method='inverted_cdf')
    for table,scale in ((out['normalized_predictions'],1.),(out['predictions'],parent['scales']['demand_scale'])):
        table['residual_mean']=scale*float(np.mean(errors));table['residual_variance']=scale**2*float(np.var(errors))
        table['observed_mean']=table.prediction+table.residual_mean
        table['observed_variance']=table.residual_variance
        table['observed_lower']=table.prediction+scale*low;table['observed_upper']=table.prediction+scale*high
    out['status']='complete-development-residual'
    out['manifest']['residual_model']={'protocol':PROTOCOL,'confidence':confidence,'sample_count':len(errors),
        'quantiles':'inverse empirical CDF; no Gaussian substitution',
        'coverage_status':r['coverage_status'],'fit_policy':r['fit_policy'],
        'law':r['law'],'process_sigma':0.,'process_combination':'not performed',
        'residual_refitted':False,'parent_snapshot_sha256':r['parent_snapshot_sha256'],
        'companion_snapshot_sha256':snapshot_fingerprint(r),'source_sha256':source_hashes(),
        'excluded':['identified measurement noise','parameter uncertainty','future input uncertainty','state process diffusion'],
        'negative_raw_lower_bounds':int((out['predictions'].observed_lower<0).sum()),
        'clipping':False,'scope':'output forecast only; not used in physical energy balance or price control'}
    out['seconds']=perf_counter()-started
    return out


def score_residual(result, observations):
    out=interval.score_interval(result,observations);table=out['predictions']
    mask=np.isfinite(table.observed.to_numpy(float));t=table.loc[mask]
    alpha=1-out['manifest']['residual_model']['confidence']
    widths=t.observed_upper-t.observed_lower
    score=widths+2/alpha*((t.observed_lower-t.observed).clip(lower=0)+(t.observed-t.observed_upper).clip(lower=0))
    out['scoring']['metrics']['observed_mean']=metrics(t.observed.to_numpy(),t.observed_mean.to_numpy())
    out['scoring']['residual_bands']={'scored_samples':len(t),
        'coverage':None if len(t)==0 else float(((t.observed>=t.observed_lower)&(t.observed<=t.observed_upper)).mean()),
        'mean_width':None if len(t)==0 else float(widths.mean()),
        'mean_interval_score':None if len(t)==0 else float(score.mean()),
        'status':'conditional development diagnostic; not calibrated coverage acceptance'}
    return out
