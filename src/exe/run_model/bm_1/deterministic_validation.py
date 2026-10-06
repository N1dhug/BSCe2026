"""G4: fixed S3 comparison on the next six ISO weeks, 13..18 of 2022.

Reuse the checked paired fitter; never refit S1/S2 or tune lambda on targets.
Future realized prices/temperature make this a conditional demand experiment.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import hashlib
import json
import numpy as np
import pandas as pd
from src.exe.run_model.progress import runtime_text
from src.exe.run_model.bm_1.deterministic_weekday import save_json, metrics
from src.exe.run_model.bm_1.deterministic_strategies import (
    calendar_inputs, checked_task, input_fingerprint, ORIGIN, FIRST_TEST, ZONE)
from src.exe.run_model.wm_4.output_model import statistical_output
from src.tools.bm_1_deterministic_data import prepare_frame

VALIDATION_START=pd.Timestamp('2022-03-28')
VALIDATION_END=pd.Timestamp('2022-05-09')


def week_job(frame, baseline, origin, reference, output):
    """Explicit cutoff; independent trajectories with no future demand in jobs."""
    training=frame.loc[frame.index<origin]
    histories=[];forecasts=[];positions=[];baseline_predictions=[]
    for weekday in range(7):
        day=origin+pd.Timedelta(days=weekday)
        pos=[]
        for lag in range(1,6):
            pos.append(len(histories))
            history,_=calendar_inputs(training,baseline,day-pd.Timedelta(weeks=lag),history=True)
            histories.append(history)
        positions.append(pos)
        forecast,index=calendar_inputs(frame[['U','Temp']],baseline,day)
        forecasts.append((forecast,index))
        baseline_predictions.append(pd.Series(np.interp(np.arange(len(index)),forecast.time,forecast.baseline),index=index))
    job={'output':str(output),'mode':'joint','reference':reference,
         'histories':histories,'forecasts':forecasts,'state_positions':positions}
    return job,baseline_predictions


def run(raw, reference, output, *, workers=8, resume=False):
    if not 1<=workers<=8:raise ValueError('Use one to eight workers with BLAS=1.')
    dest=Path(output);dest.mkdir(parents=True,exist_ok=resume)
    begun=perf_counter()
    frame,scales=prepare_frame(raw,ORIGIN,FIRST_TEST,price_bounds=(-500.,6000.))
    origins=list(pd.date_range(VALIDATION_START,periods=6,freq='7D'))
    source_paths=[Path(__file__),Path(__file__).with_name('deterministic_strategies.py'),
                  Path(__file__).resolve().parents[3]/'method/bm_1_deterministic_strategy.py']
    protocol={'version':1,'strategy':'S3','raw_fingerprint':input_fingerprint(raw),
        'reference_seed':list(map(float,reference)),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        'test_origins':list(map(str,origins)),'end_exclusive':str(VALIDATION_END),
        'normalization_training':[str(ORIGIN),str(FIRST_TEST)],'scales':scales,
        'penalties':[0.,.001],'training_window':'five complete calendar weeks; shared C,k,Xi,a over 35 daily histories',
        'baseline':'same WM4 statistical_output, expanding measured history before each Monday',
        'history':'neutral x=.5, exactly 24 physical warmup hours; weekday median forecast states',
        'selection':'lower pooled raw-unit forecast MSE on new block; exact tie lambda=0; no further lambda tuning',
        'numerics':{'baseline_samples_per_hour':60,'fit_grids':[12,24,48],'refinement':2,
                    'max_normalized_demand_difference':1e-5,'paired_max_calls_all_attempts':3000,
                    'calls_per_start':750,'paired_max_compute_seconds':600},
        'workers':workers,'blas_threads':1,
        'evidence':'new temporal validation for fixed G4 comparison; legacy point approximation; realized price/weather conditional',
        'not_final_test':'Selection uses this validation block; chosen-model score is not an independent final test.'}
    if resume:
        if json.loads((dest/'protocol.json').read_text())!=protocol:
            raise ValueError('Resume needs identical inputs, numerical source, seed and protocol.')
    else:save_json(dest/'protocol.json',protocol)
    jobs=[];baseline_values=[];reused=[]
    for week,origin in enumerate(origins,13):
        baseline=statistical_output().fit(frame.loc[frame.index<origin,['I','Temp']],origin)
        save_json(dest/f'baseline_week_{week}.json',baseline.metadata)
        job,values=week_job(frame,baseline,origin,reference,dest/'s3'/f'week_{week}')
        baseline_values.extend(values)
        saved=Path(job['output'])/'result.json'
        if resume and saved.exists() and json.loads(saved.read_text()).get('status')=='complete':
            reused.append(job['output'])
        else:
            job['resume']=resume;jobs.append(job)
    preparation_seconds=perf_counter()-begun
    print(f'Prepared {len(jobs)} new paired jobs; {len(reused)} saved jobs reused. Expected compute: a few minutes, fixed task budgets.',flush=True)
    parallel_start=perf_counter();completed=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(checked_task,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                future.result();completed.append(job['output'])
            except Exception as exc:
                failures.append({'job':job['output'],'error':str(exc)})
                save_json(dest/'failure.json',{'failures':failures,'completed':completed,'reused':reused})
            done=len(completed)+len(failures)
            save_json(dest/'progress.json',{'completed':completed,'failed':failures,'new_jobs':len(jobs),'reused_jobs':len(reused),
                                           'parallel_seconds':perf_counter()-parallel_start})
            print(f'{done}/{len(jobs)} {Path(job["output"]).name} {runtime_text(parallel_start,done,len(jobs))}',flush=True)
    if failures:raise RuntimeError('Incomplete paired jobs; retain checkpoints and use --resume within the same budgets.')
    predicted=pd.DataFrame({'O':pd.concat(baseline_values).sort_index()})
    for label in ('without_penalty','with_penalty'):
        pieces=[pd.read_csv(dest/'s3'/f'week_{week}'/f'forecast_{label}.csv',index_col=0,parse_dates=True).prediction for week in range(13,19)]
        predicted['s3_'+label]=pd.concat(pieces).sort_index()
    expected=pd.date_range(VALIDATION_START.tz_localize(ZONE),VALIDATION_END.tz_localize(ZONE),freq='h',inclusive='left').tz_localize(None)
    if not predicted.index.equals(expected) or not np.isfinite(predicted.to_numpy()).all():
        raise RuntimeError('Forecasts must cover the identical complete physical-hour index before scoring.')
    predicted.to_csv(dest/'forecast_normalized.csv',index_label='Datetime')
    scored=predicted*scales['demand_scale']+scales['demand_min']
    scored['observed']=raw.reindex(scored.index).Demand.to_numpy()
    scored.to_csv(dest/'scored.csv',index_label='Datetime')
    scores={name:metrics(scored.observed.to_numpy(),scored[name].to_numpy()) for name in predicted}
    weekly=[]
    for week,origin in enumerate(origins,13):
        block=scored.loc[(scored.index>=origin)&(scored.index<origin+pd.Timedelta(weeks=1))]
        weekly.append({'week':week,'origin':str(origin),'scores':{name:metrics(block.observed.to_numpy(),block[name].to_numpy()) for name in predicted}})
    chosen=min(['s3_without_penalty','s3_with_penalty'],key=lambda name:scores[name]['mse'])
    results=[json.loads((dest/'s3'/f'week_{w}'/'result.json').read_text()) for w in range(13,19)]
    summary={'status':'complete','protocol':protocol,'pooled_scores':scores,'weekly_scores':weekly,
        'selection':chosen,'selection_basis':'registered pooled MSE; robustness described separately, no subsequent tuning',
        'forecast_hours':len(predicted),'measured_hours':scores['O']['samples'],'paired_tasks':6,
        'new_tasks_this_invocation':len(jobs),'reused_tasks':len(reused),
        'preparation_seconds':preparation_seconds,'parallel_seconds_this_invocation':perf_counter()-parallel_start,
        'total_seconds_this_invocation':perf_counter()-begun,
        'sum_optimizer_worker_seconds':sum(a['fit']['seconds'] for r in results for a in r['attempts']),
        'max_normalized_numerical_error':max(c['error'] for r in results for c in r['candidates'])}
    save_json(dest/'summary.json',summary)
    return summary


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--raw-csv',required=True);p.add_argument('--reference-fit',required=True)
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();path=Path(a.reference_fit)
    manifest=json.loads((path.parent/'manifest.json').read_text())
    if pd.Timestamp(manifest['training_end_exclusive'])>FIRST_TEST:
        raise ValueError('Use the same original pre-development January reference seed.')
    reference=json.loads(path.read_text())['best_converged']['parameters']
    run(pd.read_csv(a.raw_csv,index_col=0,parse_dates=True),reference,a.output,workers=a.workers,resume=a.resume)


if __name__=='__main__':main()
