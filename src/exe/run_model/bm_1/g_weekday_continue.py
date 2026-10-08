"""Continue unused weekday budgets, preserving all completed fits/checkpoints.

An exhausted grid-refit may still contain an optimizer-converged coarse fit.
It can be accepted ONLY after the already declared60/120 numerical check,
without new optimizer calls and within the original weekday30s wall cap.
No optimizer convergence on the exported grid is claimed for that case.
"""
from argparse import ArgumentParser
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import pandas as pd

from src.method import g_weekday_rolling as model
from src.method import g_weekday_export as numerical
from src.method.g_prequential_bands import CALIBRATION_ORIGINS
from src.tools.bm_1_deterministic_data import raw_csv
from src.tools.bm_1_reports import _atomic_report
from src.exe.run_model.bm_1.s3_forecast import write_forecast
from src.exe.run_model.bm_1.g_wm_solver_extension import fit_extension


def check_saved_converged(g,frame,baseline,names):
    if g['status']=='complete':return g
    attempts=[a for a in g['attempts'] if a['fit'].get('best_converged') is not None]
    if not attempts:raise RuntimeError('No optimizer-converged weekday fit exists; no retry permitted.')
    chosen=attempts[-1];fit=chosen['fit'];best=fit['best_converged'];begun=perf_counter()
    remaining=30.-g['seconds']
    if remaining<=0:raise RuntimeError('No remaining original weekday wall budget for numerical check.')
    histories=[model.interval.day_inputs(frame,baseline,pd.Timestamp(d),history=True,steps=60)[0] for d in names]
    pool=model.BM1WeekdayInputs(histories);curve=model.interval.BM1PiecewiseResponse.from_inner_values([best['g_mid']],pieces=2)
    coarse=pool.solve(best['parameters'],price_response=curve,deadline=begun+remaining)
    fine=pool.refined().solve(best['parameters'],price_response=curve,deadline=begun+remaining)
    error=float(np.max(abs(coarse['demand']-fine['demand'])))
    if error>1e-5 or perf_counter()-begun>remaining:raise RuntimeError('Saved converged fit fails declared precision/budget.')
    return {'status':'complete','parameters':best['parameters'],'price_response':curve.to_record(),
        'forecast_state':float(np.median(coarse['historical_start_states'])),
        'fine_forecast_state':float(np.median(fine['historical_start_states'])),
        'historical_start_states':coarse['historical_start_states'].tolist(),'steps_per_hour':60,
        'history_max_demand_error':error,'fit':fit,'attempts':g['attempts'],'calls':g['calls'],
        'seconds':g['seconds']+(perf_counter()-begun),'original_failed_fit_seconds':g['seconds'],
        'numerical_check_optimizer_calls':0,'fit_objective_steps_per_hour':chosen['steps_per_hour'],
        'fit_interpretation':'optimizer converged on declared coarse objective;60/120 export checked; no optimizer convergence claim on exported grid'}


def main(argv=None):
    parser=ArgumentParser(description=__doc__)
    for name in ('previous-output','raw-csv','inputs-csv','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--wm-extra-calls',type=int,choices=(0,1200),default=0)
    args=parser.parse_args(argv);old=Path(args.previous_output);directory=Path(args.output);directory.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((old/'invocation.json').read_text());previous=json.loads((old/'fit_report.json').read_text())
    begun=perf_counter();origin=receipt['origin']
    lineage={'previous_output':str(old),'previous_fit_report_sha256':hashlib.sha256((old/'fit_report.json').read_bytes()).hexdigest(),
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        'optimizer_retry':args.wm_extra_calls!=0,'original_per_weekday_wall_caps_preserved':True,
        'WM_extra_calls_explicit':args.wm_extra_calls,
        'WM_extension_driver_sha256':hashlib.sha256(Path(__file__).with_name('g_wm_solver_extension.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()}
    _atomic_report(directory/'invocation.json',{'utc':datetime.now(timezone.utc).isoformat(),'origin':origin,**lineage})
    try:
        if previous['status']!='unfinished' or receipt['baseline_policy']!='weekday':raise ValueError('Only an unfinished own weekday series can continue.')
        digest=hashlib.sha256(Path(args.raw_csv).read_bytes()).hexdigest()
        if digest!=receipt['input_sha256'][str(Path(args.raw_csv))]:raise ValueError('Exact original raw input file required.')
        raw=raw_csv(args.raw_csv);train,frame,scales,start,t=model.training_data(raw,origin);days=[];reports=[]
        def checkpoint(record):_atomic_report(directory/f"weekday_{record['weekday']}_{record['stage']}.json",model.interval.point._plain(record),replace_existing=True)
        for d in range(7):
            complete=old/f'weekday_{d}_complete.json'
            if complete.exists():
                item=json.loads(complete.read_text())['day'];days.append(item);reports.append({'weekday':d,'G':item['G'],'WM':item['WM'],'reused':True});continue
            names=[str((t+pd.Timedelta(days=d)-pd.Timedelta(weeks=w)).date()) for w in range(1,6)]
            baseline=model.fit_baseline(frame,start,t,d)
            saved=next((r for r in previous['reports'] if r['weekday']==d),None)
            if saved is None:
                histories=[model.interval.day_inputs(frame,baseline,pd.Timestamp(day),history=True)[0] for day in names]
                g=model.fit_g(histories,checkpoint=lambda x:checkpoint({'weekday':d,'stage':'G','fit':x}))
                checkpoint({'weekday':d,'stage':'raw_G_result','fit':g})
            else:g=saved['G']
            g=check_saved_converged(g,frame,baseline,names)
            histories=[model.interval.day_inputs(frame,baseline,pd.Timestamp(day),history=True,steps=g['steps_per_hour']) for day in names]
            items=[]
            for (inp,index),name in zip(histories,names):
                y=np.where(inp.observation_mask[24:],inp.observations[24:],np.nan)
                items.append({'price':inp.prices[24:],'baseline':inp.b_mean[24:],'observed':y,'mask':np.isfinite(y),'index':index[24:],'day':name})
            if saved is not None and 'WM' in saved:
                w=saved['WM']
                if w['status']!='complete':
                    if args.wm_extra_calls==0:raise RuntimeError('Saved WM budget is unfinished; no optimizer retry permitted.')
                    w=fit_extension(items,w,extra_calls=args.wm_extra_calls)
                    checkpoint({'weekday':d,'stage':'WM_extension','fit':w})
            else:w=model.fit_wm(items,checkpoint=lambda x:checkpoint({'weekday':d,'stage':'WM','fit':x}))
            if w['status']!='complete':raise RuntimeError('New WM weekday fit unfinished; no retry.')
            ge=[];we=[];maximum=0
            for history,item in zip(histories,items):
                gp,error=model.replay_g(history,g,item['day']);maximum=max(maximum,error)
                wp=model.wm.solve(item['price'],item['baseline'],w['r_per_day'],w['forecast_state'])['prediction'];mask=item['mask']
                ge.extend((item['observed']-gp)[mask]);we.extend((item['observed']-wp)[mask])
            day={'weekday':d,'training_days':names,'baseline':model.baseline_record(baseline),'G':g,'WM':w,
                'target_measured_masks':[item['mask'].tolist() for item in items],
                'g_residual_samples_normalized':np.sort(ge).tolist(),'wm_residual_samples_normalized':np.sort(we).tolist(),'replay_max_demand_error':maximum}
            days.append(day);reports.append({'weekday':d,'G':g,'WM':w});checkpoint({'weekday':d,'stage':'complete','day':day})
            print(f'{origin}: continued weekday {d} frozen',flush=True)
        snapshot={'kind':'G-WM-weekday-rolling-fit','schema_version':1,'protocol':model.PROTOCOL,'origin':str(t),
            'training_start':str(start),'training_end_exclusive':str(t),'normalization_window':[str(start),str(t)],
            'training_fingerprint':model.interval.point._fingerprint(train),'source_sha256':model.source_hashes(),'scales':scales,
            'baseline_policy':'weekday','days':days,'demand_timestamp_label':'hour_start',
            'population':'MED mean over available measured meters','demand_unit':'raw average-power units; physical unit unverified',
            'timestamp_interpretation_status':'caller working assumption; not independently verified',
            'fit_policy':'five matching weekdays only; seven independent parameter fits; same baseline for G/WM within weekday',
            'normalization_policy':'common scales from five training weeks; no older or target demand',
            'initialization_policy':'G neutral .5 plus24h input-only warmup, median starts; WM variable projection, median starts',
            'residual_policy':'separate weekday empirical frozen-median replay errors; in-sample development; no process diffusion',
            'budgets_per_weekday':{'G_seconds':30,'G_calls':750,'WM_seconds':10,'WM_calls':400},'continuation':lineage}
        snapshot=model.validate_snapshot(snapshot);_atomic_report(directory/'weekday_snapshot.json',snapshot)
        _atomic_report(directory/'fit_report.json',{'status':'complete','reports':reports,'seconds':perf_counter()-begun,'lineage':lineage})
        exported=numerical.refine_for_export(snapshot,raw);_atomic_report(directory/'weekday_export_snapshot.json',exported['snapshot'])
        exported['historical_replay'].to_csv(directory/'export_replay_normalized.csv',index_label='Datetime')
        forecast=numerical.predict_comparison(exported['snapshot'],raw_csv(args.inputs_csv,require_demand=False));write_forecast(directory,forecast)
        files=('weekday_snapshot.json','weekday_export_snapshot.json','forecast.json','forecast.csv','forecast_normalized.csv')
        _atomic_report(directory/'forecast_freeze.json',{'status':'frozen-input-only','origin':origin,'phase':'calibration' if origin in CALIBRATION_ORIGINS else 'evaluation',
            'utc':datetime.now(timezone.utc).isoformat(),'target_observation_file_opened':False,'total_seconds':perf_counter()-begun,
            'G_calls':sum(d['G']['calls'] for d in days),'WM_calls':sum(d['WM']['calls'] for d in days),'counts_include_reused':True,
            'artifact_sha256':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in files}})
        print(f'{origin}: own weekday forecast frozen; cumulative wall caps preserved.',flush=True);return 0
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        _atomic_report(directory/'failure.json',{'status':'failed','origin':origin,'seconds':perf_counter()-begun,'error':str(exc),'optimizer_retry':False})
        print(str(exc),file=sys.stderr,flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
