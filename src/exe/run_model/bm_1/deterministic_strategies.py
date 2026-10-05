"""G: three predeclared strategies on ISO weeks 1..12, paired penalties."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import json
import hashlib
import numpy as np
import pandas as pd
from src.exe.run_model.progress import runtime_text
from src.exe.run_model.bm_1.deterministic_weekday import save_json, metrics
from src.exe.run_model.wm_4.output_model import statistical_output
from src.method.bm_1_deterministic_response import BM1ResponseInputs
from src.method.bm_1_deterministic_weekday import BM1PooledInputs
from src.method.bm_1_deterministic_strategy import fit_strategy
from src.params.bm_1_response import BM1PiecewiseResponse
from src.tools.bm_1_deterministic_data import prepare_frame

ZONE = 'Europe/Copenhagen'
ORIGIN = pd.Timestamp('2022-01-03')
FIRST_TEST = ORIGIN+pd.Timedelta(weeks=5)
END = ORIGIN+pd.Timedelta(weeks=12)


def input_fingerprint(raw):
    """Bind resume to the exact indexed observations and scenario inputs."""
    hashed = pd.util.hash_pandas_object(raw[['Demand', 'Price', 'Temp']], index=True)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def calendar_inputs(frame, baseline, day, *, history=False, steps=12):
    """Use physical elapsed hours and local WM features; spring day has 23 h.

    The declared exogenous baseline grid stays fixed at 60 samples/h when the
    ODE grid is refined. No demand values are inspected for forecast creation.
    """
    day = pd.Timestamp(day).normalize()
    start = (day-pd.Timedelta(days=1) if history else day).tz_localize(ZONE)
    end = (day+pd.Timedelta(days=1)).tz_localize(ZONE)
    physical = pd.date_range(start, end, freq='h', inclusive='left')
    index = physical.tz_localize(None)
    source = frame.reindex(index)[['U', 'Temp']]
    if not np.isfinite(source.to_numpy()).all() or np.any(abs(source.U.to_numpy()) > 1):
        raise ValueError(f'Incomplete/domain-invalid scenario inputs for {day}.')
    n = len(index)
    elapsed = np.arange(n*60+1, dtype=float)/60
    fine_index = (start.tz_convert('UTC')+pd.to_timedelta(elapsed, unit='h')).tz_convert(ZONE)
    temp = np.interp(elapsed, np.arange(n), source.Temp.to_numpy())
    B = baseline.predict(fine_index, temp)
    obs = frame.reindex(index).I.to_numpy() if history else None
    inp = BM1ResponseInputs(elapsed, B, (source.U.to_numpy()+1)/2, obs,
        observation_mask=np.isfinite(obs) if history else None,
        warmup_hours=24 if history else 0, steps_per_hour=steps)
    return inp, index


def checked_task(job):
    """Worker receives training demand only; target demand is absent by construction."""
    dest = Path(job['output']); resume=bool(job.get('resume'))
    dest.mkdir(parents=True, exist_ok=resume)
    prior=json.loads((dest/'result.json').read_text()) if resume and (dest/'result.json').exists() else {}
    if prior.get('status')=='complete':
        return prior
    attempts=list(prior.get('attempts',[]))
    if attempts:
        save_json(dest/f'failure_before_continuation_{len(attempts)}.json',prior)
    carried_seconds=sum(a['fit']['seconds'] for a in attempts)
    began = perf_counter(); result = None
    for steps in (12,24,48):
        histories = job['histories']
        forecasts = job['forecasts']
        while histories[0].steps < steps:
            histories = [x.refined() for x in histories]
            forecasts = [(x.refined(), index) for x,index in forecasts]
        pool = BM1PooledInputs(histories)
        candidates = []; failures = []
        for strength in ((0.,) if job['mode']=='linear' else (0., .001)):
            previous_calls = sum(a['fit']['counts']['residual_calls'] for a in attempts)
            remaining = 3000-previous_calls
            if remaining < 1:
                raise RuntimeError('Numerical refit exhausted declared per-strength total budget.')
            matching=[a for a in attempts if a['steps']==steps and a['lambda']==strength]
            cached=matching[-1]['fit'] if matching else None
            if cached is not None and cached['best_converged'] is not None:
                fit=cached
            else:
                seeds=[cached['best_provisional']] if cached and cached['best_provisional'] else None
                fit = fit_strategy(pool, job['reference'], job['mode'], strength,
                    max_calls=min(750 if seeds else 1500,remaining), calls_per_start=750,
                    seconds=max(.001, 600-carried_seconds-(perf_counter()-began)), initial_records=seeds,
                    checkpoint=lambda value: save_json(dest/f'fit_steps_{steps}_lambda_{strength}_attempt_{len(attempts)}.json', value))
                attempts.append({'steps':steps,'lambda':strength,'checkpoint_continuation':bool(seeds),'fit':fit})
            if fit['best_converged'] is None:
                save_json(dest/'result.json', {'status':'unfinished','attempts':attempts})
                raise RuntimeError(f'No converged {job["mode"]} fit: {dest}.')
            best = fit['best_converged']
            g = BM1PiecewiseResponse.from_inner_values([best['g_mid']],pieces=2)
            h = pool.solve(best['parameters'],price_response=g)
            fine = pool.refined().solve(best['parameters'],price_response=g)
            error = float(np.max(abs(h['demand']-fine['demand'])))
            columns = []; starts = []
            for (forecast,index),positions in zip(forecasts,job['state_positions']):
                x0 = float(np.median(h['historical_start_states'][positions]))
                fine_x0 = float(np.median(fine['historical_start_states'][positions]))
                p = list(best['parameters']);p[3] = x0
                q = p.copy();q[3] = fine_x0
                predicted = forecast.solve(p,price_response=g)
                fine_predicted = forecast.refined().solve(q,price_response=g)
                error = max(error,float(np.max(abs(predicted['demand']-fine_predicted['demand']))))
                columns.append(pd.Series(predicted['demand'],index=index))
                starts.append(x0)
            candidates.append({'lambda':strength,'best':best,'error':error,
                               'predictions':columns,'forecast_states':starts})
            if error > 1e-5:
                failures.append(strength)
        save_json(dest/'numerics.json',{'steps':steps,'errors':[c['error'] for c in candidates],
                                       'refit_required':failures})
        if not failures:
            result = {'status':'complete','mode':job['mode'],'steps':steps,
                      'seconds':carried_seconds+perf_counter()-began,'observed_training_hours':int(pool.observation_mask.sum()),
                      'attempts':attempts,'candidates':[]}
            for c in candidates:
                label = 'without_penalty' if c['lambda']==0 else 'with_penalty'
                if c['predictions']:
                    values = pd.concat(c['predictions']).sort_index()
                    pd.DataFrame({'prediction':values}).to_csv(dest/f'forecast_{label}.csv',index_label='Datetime')
                result['candidates'].append({k:v for k,v in c.items() if k!='predictions'})
            save_json(dest/'result.json',result)
            return result
    save_json(dest/'result.json',{'status':'numerics-failed','attempts':attempts})
    raise RuntimeError('The fixed numerical tolerance failed at all predeclared grids.')


def run(raw, reference, output, *, workers=8, resume=False):
    dest=Path(output);dest.mkdir(parents=True,exist_ok=resume)
    frame,scales = prepare_frame(raw,ORIGIN,FIRST_TEST,price_bounds=(-500.,6000.))
    protocol={'raw_fingerprint':input_fingerprint(raw),'reference_seed':list(map(float,reference)),
        'weeks_start':str(ORIGIN),'end_exclusive':str(END),'test_origins':list(map(str,pd.date_range(FIRST_TEST,periods=7,freq='7D'))),
        'scales':scales,'price_domain':'fixed engineering scenario bounds [-500,6000], declared before fits; not future min/max',
        'penalties':[0.,.001],'workers':workers,'blas_threads':1,'baseline_cutoff':'Monday origin shared by all seven forecasts',
        'history':'independent daily trajectories; neutral x=.5 plus 24 h warm-up; same-weekday median forecast states',
        'numerics':{'baseline_samples_per_hour':60,'fit_grids':[12,24,48],'check':'twice the ODE resolution; max demand error <=1e-5',
                    'per_paired_task_max_calls_all_grids':3000,'calls_per_start':750,'seconds_per_task_all_grids':600},
        'strategy_1':'retrospective full-12-week C,k,Xi fit with g linear; rolling weekday a-only',
        'strategy_2':'rolling weekday C,k,Xi,a joint fits; 5 same weekdays each',
        'strategy_3':'rolling shared C,k,Xi,a joint fit; all 35 days',
        'day_2022_03_27':'23 physical hours; nonexistent raw local 02:00 row excluded, no fabricated price/weather',
        'evidence':'conditional DEVELOPMENT, legacy hourly-point approximation; physical units/population not revalidated'}
    if resume:
        if json.loads((dest/'protocol.json').read_text())!=protocol:
            raise ValueError('Resume requires the identical study protocol.')
    else:
        save_json(dest/'protocol.json',protocol)
    # Only S1's explicitly retrospective physical parameter fit sees all 12 weeks.
    global_baseline=statistical_output().fit(frame.loc[frame.index<END,['I','Temp']],END)
    global_histories=[calendar_inputs(frame,global_baseline,day,history=True)[0]
                      for day in pd.date_range(ORIGIN,periods=84,freq='D')]
    global_result=checked_task({'output':str(dest/'global_linear'),'mode':'linear',
        'reference':reference,'histories':global_histories,'forecasts':[],'state_positions':[],'resume':resume})
    fixed=global_result['candidates'][0]['best']['parameters']
    jobs=[];baseline_predictions=[]
    for w,origin in enumerate(pd.date_range(FIRST_TEST,periods=7,freq='7D'),6):
        baseline=statistical_output().fit(frame.loc[frame.index<origin,['I','Temp']],origin)
        save_json(dest/f'baseline_week_{w}.json',baseline.metadata)
        train_frame=frame.loc[frame.index<origin]
        histories=[];future=[];positions=[]
        for d in range(7):
            day=origin+pd.Timedelta(days=d)
            pos=[]
            for j in range(1,6):
                pos.append(len(histories))
                histories.append(calendar_inputs(train_frame,baseline,day-pd.Timedelta(weeks=j),history=True)[0])
            positions.append(pos)
            forecast,index=calendar_inputs(frame[['U','Temp']],baseline,day)
            future.append((forecast,index))
            baseline_predictions.append(pd.Series(np.interp(np.arange(len(index)),forecast.time,forecast.baseline),index=index))
            for strategy,mode,p in [('s1','g-only',fixed),('s2','joint',reference)]:
                jobs.append({'output':str(dest/strategy/f'week_{w}_day_{d}'),'mode':mode,'reference':p,
                    'histories':[histories[k] for k in pos],'forecasts':[(forecast,index)],
                    'state_positions':[list(range(5))]})
        jobs.append({'output':str(dest/'s3'/f'week_{w}'),'mode':'joint','reference':reference,
                     'histories':histories,'forecasts':future,'state_positions':positions})
    began=perf_counter(); completed=[]
    existing=[]
    if resume:
        pending=[]
        for job in jobs:
            saved=Path(job['output'])/'result.json'
            if saved.exists() and json.loads(saved.read_text()).get('status')=='complete':
                existing.append(job['output'])
            else:
                job['resume']=True
                pending.append(job)
        jobs=pending
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(checked_task,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                future.result()
                completed.append(job['output'])
            except Exception as exc:
                save_json(dest/'failure.json',{'job':job['output'],'error':str(exc),'completed':completed,'reused':existing})
                raise
            save_json(dest/'progress.json',{'completed':completed,'total':len(jobs),'seconds':perf_counter()-began})
            print(f'{len(completed)}/{len(jobs)} {Path(job["output"]).name} {runtime_text(began,len(completed),len(jobs))}',flush=True)
    predicted=pd.DataFrame({'O':pd.concat(baseline_predictions).sort_index()})
    for strategy in ['s1','s2','s3']:
        for label in ['without_penalty','with_penalty']:
            pieces=[pd.read_csv(p,index_col=0,parse_dates=True).prediction
                    for p in (dest/strategy).glob(f'*/forecast_{label}.csv')]
            predicted[f'{strategy}_{label}']=pd.concat(pieces).sort_index()
    if not np.isfinite(predicted.to_numpy()).all() or len(predicted)!=1175:
        raise RuntimeError('Incomplete matched forecasts; no final score accepted.')
    predicted.to_csv(dest/'forecast_normalized.csv',index_label='Datetime')
    # Target observations are opened only after the full frozen forecast is saved.
    scored=predicted*scales['demand_scale']+scales['demand_min']
    scored['observed']=raw.reindex(scored.index).Demand.to_numpy()
    scored.to_csv(dest/'scored.csv',index_label='Datetime')
    scores={name:metrics(scored.observed.to_numpy(),scored[name].to_numpy()) for name in predicted}
    weeks=[]
    for w,origin in enumerate(pd.date_range(FIRST_TEST,periods=7,freq='7D'),6):
        block=scored.loc[(scored.index>=origin)&(scored.index<origin+pd.Timedelta(weeks=1))]
        weeks.append({'week':w,'origin':str(origin),'scores':{name:metrics(block.observed.to_numpy(),block[name].to_numpy()) for name in predicted}})
    summary={'status':'complete','protocol':protocol,'pooled_scores':scores,'weekly_scores':weeks,
             'parallel_seconds':perf_counter()-began,'global_seconds':global_result['seconds'],
             'global_parameters':fixed,'tasks':105,'new_tasks_this_invocation':len(jobs),'reused_tasks':len(existing),'matched_hours':len(scored)}
    save_json(dest/'summary.json',summary)
    return summary


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('--raw-csv',required=True)
    p.add_argument('--reference-fit',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--resume',action='store_true',help='Reuse complete tasks; continue an unfinished optimizer checkpoint within its total budget.')
    a=p.parse_args()
    if not 1<=a.workers<=8:
        raise ValueError('Use one to eight independent workers with BLAS=1.')
    path=Path(a.reference_fit)
    manifest=json.loads((path.parent/'manifest.json').read_text())
    if pd.Timestamp(manifest['training_end_exclusive'])>FIRST_TEST:
        raise ValueError('Honest strategies require a pre-first-origin reference seed.')
    reference=json.loads(path.read_text())['best_converged']['parameters']
    run(pd.read_csv(a.raw_csv,index_col=0,parse_dates=True),reference,a.output,workers=a.workers,resume=a.resume)


if __name__=='__main__':
    main()
