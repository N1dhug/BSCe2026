"""Fixed chronological calibration of standardized absolute forecast errors.

No diffusion/distribution is fitted. Finite-rank intervals are diagnostic
under temporal dependence and rolling refits, not an exchangeability claim.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json

import numpy as np
import pandas as pd

from src.method import g_weekday_export as comparison
from src.method import bm_1_s3_interval as interval
from src.method import bm_1_s3_residual as residual

PROTOCOL='G-weekday-rolling-prequential-bands-v2'
CALIBRATION_ORIGINS=('2022-02-28','2022-03-07','2022-03-14')
EVALUATION_ORIGINS=('2022-03-21','2022-03-28','2022-04-04')
FREEZE_CUTOFF='2022-03-21 00:00:00'
MODELS=('g_prediction','wm_prediction')


def source_hashes():
    sources=comparison.source_hashes()
    name='src/method/g_prequential_bands.py'
    sources[name]=hashlib.sha256((interval.ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return sources


def finite_rank(scores,confidence=.95):
    scores=np.asarray(scores,float)
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all() or np.any(scores<0):
        raise ValueError('Finite nonnegative calibration scores required.')
    if not np.isfinite(confidence) or not 0<confidence<1:raise ValueError('Confidence must be in (0,1).')
    # Decimal confidence rational avoids floating-point one-rank boundary changes.
    fraction=Fraction(str(float(confidence)));n=len(scores)
    rank=((n+1)*fraction.numerator+fraction.denominator-1)//fraction.denominator
    return {'samples':n,'rank':int(rank),'q':None if rank>n else float(np.sort(scores)[rank-1]),
        'deployable':rank<=n,'confidence':float(confidence)}


def forecast_fingerprint(result):
    table=result['predictions'];names=[]
    for model in MODELS:names.extend([model+'_output_mean',model+'_output_variance'])
    payload=json.dumps(result['manifest'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(payload+pd.util.hash_pandas_object(table[names],index=True).values.tobytes()).hexdigest()


def validate_forecast(result):
    manifest=result['manifest'];table=result['predictions']
    if (manifest['comparison_protocol']!=comparison.PROTOCOL or manifest['source_sha256']!=comparison.source_hashes()
            or manifest['demand_used'] or not manifest['g_residual_attached']):
        raise ValueError('Require compatible input-only shared G/WM forecast with both residual scales.')
    if (not isinstance(table.index,pd.DatetimeIndex) or table.index.tz is not None
            or table.index.has_duplicates or not table.index.is_monotonic_increasing):
        raise ValueError('Forecast must have unique sorted local hour-start labels.')
    origin=interval.point._origin(manifest['cutoff']);end=pd.Timestamp(manifest['end_exclusive'])
    if end!=origin+pd.Timedelta(days=7):raise ValueError('Exactly one physical calendar week required.')
    expected=pd.date_range(origin.tz_localize(interval.ZONE),end.tz_localize(interval.ZONE),freq='h',inclusive='left').tz_localize(None)
    if not table.index.equals(expected):raise ValueError('Require complete physical forecast week without dropped hours.')
    for model in MODELS:
        mean=table[model+'_output_mean'].to_numpy(float);variance=table[model+'_output_variance'].to_numpy(float)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance<=0):
            raise ValueError('Finite mean and positive previsible historical residual variance required; no floor.')
    return origin,end


def calibrate(packets):
    """Three earlier forecast/observation packets; evaluation Y is not an input."""
    if len(packets)!=3:raise ValueError('Require all three declared calibration weeks.')
    seen=[];references=[];score_rows=[];bank={m:[] for m in MODELS};lag_left={m:[] for m in MODELS};lag_right={m:[] for m in MODELS}
    for packet in packets:
        forecast=packet['forecast'];origin,end=validate_forecast(forecast)
        day=origin.strftime('%Y-%m-%d');seen.append(day)
        if day not in CALIBRATION_ORIGINS or end>pd.Timestamp(FREEZE_CUTOFF):
            raise ValueError('Calibration cannot include an evaluation or post-freeze week.')
        table=forecast['predictions'];aligned=interval.align_demand(packet['observations'],forecast['manifest']['demand_timestamp_label'])
        y=aligned.reindex(table.index).to_numpy(float);mask=np.isfinite(y)
        physical=table.index.tz_localize(interval.ZONE).tz_convert('UTC')
        one_hour=np.diff(physical.asi8)==pd.Timedelta(hours=1).value
        reference={'origin':str(origin),'end_exclusive':str(end),'samples':int(mask.sum()),
            'forecast_sha256':forecast_fingerprint(forecast),
            'observation_sha256':interval.point._fingerprint(pd.DataFrame({'Demand':y},index=table.index))}
        references.append(reference)
        for model in MODELS:
            mean=table[model+'_output_mean'].to_numpy(float);scale=np.sqrt(table[model+'_output_variance'].to_numpy(float))
            z=np.abs(y-mean)/scale;bank[model].extend(z[mask])
            adjacent=mask[:-1]&mask[1:]&one_hour
            lag_left[model].extend(z[:-1][adjacent]);lag_right[model].extend(z[1:][adjacent])
            score_rows.append(pd.DataFrame({'origin':str(origin),'model':model,'observed':y[mask],
                'mean':mean[mask],'historical_sd':scale[mask],'score':z[mask]},index=table.index[mask]))
    if tuple(seen)!=CALIBRATION_ORIGINS:raise ValueError('Calibration weeks must be complete, unique and chronological.')
    estimates={}
    for model in MODELS:
        scores=np.sort(bank[model]);estimate=finite_rank(scores)
        if len(scores)<300 or not estimate['deployable']:raise ValueError('At least 300 deployable scores per model required.')
        left=np.asarray(lag_left[model]);right=np.asarray(lag_right[model]);correlation=None
        if len(left)>=2 and np.std(left)>0 and np.std(right)>0:correlation=float(np.corrcoef(left,right)[0,1])
        estimate.update(scores=scores.tolist(),within_week_adjacent_pairs=len(left),score_lag1_correlation=correlation)
        estimates[model]=estimate
    record={'kind':'G-WM-prequential-calibrated-bands','schema_version':1,'protocol':PROTOCOL,
        'freeze_cutoff':FREEZE_CUTOFF,'calibration_origins':list(CALIBRATION_ORIGINS),
        'evaluation_origins':list(EVALUATION_ORIGINS),'confidence':.95,'models':estimates,
        'references':references,'source_sha256':source_hashes(),
        'mean_policy':'unchanged own empirical-output-error mean',
        'scale_policy':'unchanged own previsible historical empirical residual SD in raw units',
        'coverage_status':'prequential development calibration; temporal exchangeability not assumed or proven',
        'variance_status':'q is an interval multiplier, not diffusion or a full predictive variance',
        'refit_policy':'rolling causal parent fits allowed; calibration q frozen for all evaluation weeks',
        'evaluation_observations_used':False}
    return {'snapshot':validate_calibration(record),'scores':pd.concat(score_rows)}


def validate_calibration(record):
    try:
        r=deepcopy(record)
        if (r['kind']!='G-WM-prequential-calibrated-bands' or r['schema_version']!=1 or r['protocol']!=PROTOCOL
                or r['freeze_cutoff']!=FREEZE_CUTOFF or tuple(r['calibration_origins'])!=CALIBRATION_ORIGINS
                or tuple(r['evaluation_origins'])!=EVALUATION_ORIGINS or r['confidence']!=.95
                or r['source_sha256']!=source_hashes() or r['evaluation_observations_used']):
            raise ValueError('Require own compatible frozen calibration record.')
        for model in MODELS:
            item=r['models'][model];calculated=finite_rank(item['scores'],r['confidence'])
            if (len(item['scores'])<300 or np.any(np.diff(item['scores'])<0)
                    or any(item[key]!=calculated[key] for key in ('samples','rank','q','deployable'))):
                raise ValueError('Calibration quantile/rank does not match its score bank.')
        if (tuple(pd.Timestamp(ref['origin']).strftime('%Y-%m-%d') for ref in r['references'])!=CALIBRATION_ORIGINS
                or any(pd.Timestamp(ref['end_exclusive'])>pd.Timestamp(FREEZE_CUTOFF) for ref in r['references'])
                or any(sum(ref['samples'] for ref in r['references'])!=r['models'][m]['samples'] for m in MODELS)):
            raise ValueError('Calibration references disagree with dates or score counts.')
        return r
    except (KeyError,TypeError) as exc:raise ValueError('Incomplete calibration record.') from exc


def apply_calibration(forecast,record):
    r=validate_calibration(record);origin,end=validate_forecast(forecast)
    if origin.strftime('%Y-%m-%d') not in EVALUATION_ORIGINS or origin<pd.Timestamp(r['freeze_cutoff']):
        raise ValueError('Frozen calibration is for declared later evaluation weeks only.')
    out=deepcopy(forecast)
    for table in (out['predictions'],out['normalized_predictions']):
        for model in MODELS:
            scale=np.sqrt(table[model+'_output_variance'].to_numpy(float));q=r['models'][model]['q']
            mean=table[model+'_output_mean'].to_numpy(float)
            table[model+'_calibrated_lower']=mean-q*scale;table[model+'_calibrated_upper']=mean+q*scale
    out['status']='complete-development-calibrated-bands'
    out['manifest']['calibrated_bands']={'protocol':PROTOCOL,'snapshot_sha256':residual.snapshot_fingerprint(r),
        'freeze_cutoff':r['freeze_cutoff'],'confidence':r['confidence'],
        'multipliers':{m:r['models'][m]['q'] for m in MODELS},'mean_policy':r['mean_policy'],
        'scale_policy':r['scale_policy'],'coverage_status':r['coverage_status'],'variance_status':r['variance_status'],
        'source_sha256':source_hashes(),'clipping':False,
        'negative_raw_lower_bounds':{m:int((out['predictions'][m+'_calibrated_lower']<0).sum()) for m in MODELS},
        'evaluation_observations_used':False,'band_refitted':False}
    return out


def band_metrics(table,lower,upper):
    y=table.observed.to_numpy(float);mask=np.isfinite(y);lo=table[lower].to_numpy(float)[mask];hi=table[upper].to_numpy(float)[mask]
    actual=y[mask];width=hi-lo;score=width+40*(np.maximum(lo-actual,0)+np.maximum(actual-hi,0))
    return {'samples':int(mask.sum()),'coverage':None if not mask.any() else float(np.mean((actual>=lo)&(actual<=hi))),
        'mean_width':None if not mask.any() else float(np.mean(width)),
        'mean_interval_score':None if not mask.any() else float(np.mean(score)),
        'negative_raw_lower_bounds':int(np.count_nonzero(table[lower].to_numpy(float)<0)),
        'status':'nominal 95% marginal chronological development diagnostic'}


def score_calibrated(forecast,observations):
    if 'calibrated_bands' not in forecast['manifest']:raise ValueError('Calibrated forecast required.')
    out=comparison.score_comparison(forecast,observations);table=out['predictions']
    out['scoring']['calibrated_bands']={m:band_metrics(table,m+'_calibrated_lower',m+'_calibrated_upper') for m in MODELS}
    out['scoring']['daily_band_diagnostics']={}
    for m in MODELS:
        out['scoring']['daily_band_diagnostics'][m]=[
            {'day':str(day.date()),**band_metrics(part,m+'_calibrated_lower',m+'_calibrated_upper')}
            for day,part in table.groupby(table.index.normalize())]
    return out


def pooled_report(results):
    if tuple(pd.Timestamp(r['manifest']['cutoff']).strftime('%Y-%m-%d') for r in results)!=EVALUATION_ORIGINS:
        raise ValueError('Require all declared evaluation weeks in order.')
    fingerprints={r['manifest']['calibrated_bands']['snapshot_sha256'] for r in results}
    if len(fingerprints)!=1:raise ValueError('Every evaluation must use the same frozen calibration record.')
    table=pd.concat([r['predictions'] for r in results])
    if table.index.has_duplicates:raise ValueError('Overlapping evaluation timestamps.')
    y=table.observed.to_numpy(float)
    names=('baseline',*MODELS,*(m+'_output_mean' for m in MODELS))
    return {'protocol':PROTOCOL,'phase':'evaluation','expected_physical_hours':len(table),
        'scored_samples':int(np.isfinite(y).sum()),'metrics':{m:comparison.metrics(y,table[m].to_numpy(float)) for m in names},
        'old_empirical_bands':{m:band_metrics(table,m+'_lower',m+'_upper') for m in MODELS},
        'calibrated_bands':{m:band_metrics(table,m+'_calibrated_lower',m+'_calibrated_upper') for m in MODELS},
        'calibration_sha256':next(iter(fingerprints)),
        'weekly':[{ 'origin':r['manifest']['cutoff'],'scoring':r['scoring']} for r in results],
        'interpretation':'three chronological development weeks; no blind/general/simultaneous coverage acceptance'}
