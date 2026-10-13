"""BM1/BM3 daily and MPC comparison over every usable day in the input period.

Reuse complete, hash-checked saved BM1 days; fit missing references using the
existing BM1 week worker. Preserve all old metadata/checkpoint identities.
python -B -m src.exe.run_model.bm_3.run_all_days --workers 8 --mode both
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import argparse
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
import pandas as pd
from src.exe.run_model.bm_1 import predict_daily as daily
from src.exe.run_model.bm_1.compare_fg_week import (
    write_json, sha, job_folder, completed, load_modules, worker as bm1_worker)
from src.exe.run_model.bm_3.compare_week import (
    worker as bm3_worker, summarize, reference_root, print_daily_result, print_progress)
from src.exe.run_model.bm_3.predict_daily import add_model_options
from src.exe.run_model.wm_3.fit_weekly import aligned_data, END
from src.tools.baseline import statistical_output
from src.validation.exe.run_model.compare_model_predictions import model_comparison_validation as scale
from src.validation.model.bm_3 import configuration

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/'files'/'output'/'bm_3'/'runs'/'logistic_all_days'
REFERENCE = ROOT/'files'/'output'/'bm_1'/'runs'/'fg_week_2022-05-02_2022-05-08'


def eligible_tasks(frame, start, end, *, mode='both', hours=range(24)):
    """Preflight each task independently; never fill missing target/history hours."""
    valid = frame[['I','U','Temp']].replace([np.inf,-np.inf],np.nan).dropna().index
    available = set(valid.asi8.tolist())
    hour_ns = pd.Timedelta(hours=1).value
    tasks, skipped = [], []
    def complete_window(origin, length):
        first = origin.value
        return all(first+i*hour_ns in available for i in range(length))
    for day in pd.date_range(start,end):
        history = frame.loc[frame.index < day-pd.Timedelta(weeks=5),['I','U']]
        reason = None
        if len(frame.loc[frame.index<day,['I','Temp']].dropna())<168:
            reason = 'fewer than 168 baseline-history samples'
        elif history.empty or any(not np.isfinite(history[k].max()-history[k].min())
                                  or history[k].max()<=history[k].min() for k in ('I','U')):
            reason = 'insufficient normalization history before the oldest fit day'
        desired = ([('daily',-1)] if mode in ('daily','both') else [])
        if mode in ('mpc','both'): desired += [('mpc',hour) for hour in sorted(set(hours))]
        for kind,hour in desired:
            origin = day if kind=='daily' else day+pd.Timedelta(hours=hour)
            issue = reason
            if issue is None and not all(complete_window(origin-pd.Timedelta(weeks=k),24) for k in range(5,0,-1)):
                issue = 'incomplete historical 24h window'
            if issue is None and not complete_window(origin,24 if kind=='daily' else 9):
                issue = 'incomplete target/forecast window'
            if issue is None and kind=='daily' and len(frame.loc[frame.index<day-pd.Timedelta(weeks=5),['I','Temp']].dropna())<168:
                issue = 'fewer than 168 samples for the oldest training-day baseline'
            task = (str(day.date()),'logistic',kind,hour)
            if issue is None: tasks.append(task)
            else: skipped.append({'day':task[0],'mode':kind,'hour':hour,'reason':issue})
    return tasks, skipped


def prepare_day(frame, raw_consumption, day, *, with_daily):
    day = pd.Timestamp(day)
    normalized, calibration = scale.bm1_scale(frame,day)
    baseline = statistical_output().fit(normalized,day)
    training, training_metadata = daily.prepare_days(normalized,daily.matching_days(day)) if with_daily else (None,None)
    target,index = daily.checks.day_signals(normalized,day,baseline,require_demand=False) if with_daily else (None,None)
    historical_raw = raw_consumption.loc[raw_consumption.index<day-pd.Timedelta(weeks=5)]
    lo,hi = float(historical_raw.min()),float(historical_raw.max())
    return {'day':day,'training':training,'training_baselines':training_metadata,
        'target':target,'target_index':index,
        'target_observed':normalized.I.reindex(index).to_numpy() if with_daily else None,
        'mpc_frame':normalized,'mpc_baseline':baseline,'normalization':calibration,
        'mpc_calibration':calibration,'raw_offset':lo,'raw_scale':hi-lo,
        'original_consumption':raw_consumption}


def reuse_manifest(reference, tasks, inputs):
    """Select only complete saved days with matching raw input identities."""
    path = reference/'settings.json'
    if not path.exists(): return {},{}
    old = json.loads(path.read_text(encoding='utf-8'))
    if old.get('inputs_sha256') != inputs: raise ValueError('Saved reference raw-input hashes do not match.')
    selected,manifest = {},{'settings.json':sha(path)}
    for day in sorted({t[0] for t in tasks}):
        desired = [t for t in tasks if t[0]==day]
        prepared = reference/'prepared'/f'{day}.pkl'
        if prepared.exists() and all(completed(reference,t,old['fingerprint']) for t in desired):
            selected[day] = str(reference)
            manifest[f'prepared/{day}.pkl'] = sha(prepared)
            for task in desired:
                for name in ('fit.json','hourly.csv' if task[2]=='daily' else 'forecasts.csv'):
                    file = job_folder(reference,*task)/name
                    manifest[file.relative_to(reference).as_posix()] = sha(file)
    return selected,manifest


def period_worker(output, task):
    out = Path(output); settings = json.loads((out/'settings.json').read_text(encoding='utf-8'))
    reference = reference_root(settings,task[0])
    if task[0] not in settings['reference_by_day'] and not completed(reference,task,settings['generated_reference_fingerprint']):
        logfile = reference/'logs'/f'{task[0]}_{task[1]}_{task[2]}_{task[3]:02d}.log'
        with logfile.open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
            status = bm1_worker(ROOT,reference,task,load_modules(ROOT))
        if status:
            return task,'BM1 reference fit failed; see reference error.json'
    return bm3_worker(out,task)


def main():
    p = argparse.ArgumentParser(description='All usable days: reuse BM1 references, fit missing jobs and compare BM3.')
    p.add_argument('--start'); p.add_argument('--end')
    p.add_argument('--mode',choices=('daily','mpc','both'),default='both')
    p.add_argument('--tag',default='MED'); p.add_argument('--workers',type=int,default=8)
    p.add_argument('--hours',nargs='+',type=int,choices=range(24),default=list(range(24)))
    p.add_argument('--output',type=Path,default=OUTPUT); p.add_argument('--reference',type=Path,default=REFERENCE)
    p.add_argument('--x-steps',type=int,default=2601); p.add_argument('--fp-max-step',type=float,default=.125)
    p.add_argument('--numeric-tolerance',type=float,default=1e-4)
    p.add_argument('--daily-max-nfev',type=int,default=150); p.add_argument('--daily-max-iterations',type=int,default=6)
    p.add_argument('--mpc-max-nfev',type=int,default=120); p.add_argument('--mpc-max-iterations',type=int,default=3)
    p.add_argument('--plan-only',action='store_true',help='Check data and report date/task counts without fitting.')
    add_model_options(p); p.set_defaults(state_min=-6.,state_max=7.)
    args = p.parse_args()
    strength,bounds,tail = configuration(args.restoring_strength,(args.state_min,args.state_max),args.tail_tolerance)
    if not 1<=args.workers<=32 or args.x_steps<20 or not np.isfinite(args.fp_max_step) or args.fp_max_step<=0:
        p.error('Use valid workers/grid/time step.')
    if not np.isfinite(args.numeric_tolerance) or args.numeric_tolerance<=0 or min(args.daily_max_nfev,args.daily_max_iterations,args.mpc_max_nfev,args.mpc_max_iterations)<1:
        p.error('Use positive fit budgets and numerical tolerance.')
    out,reference = args.output.resolve(),args.reference.resolve()
    if out==reference or reference in out.parents or out in reference.parents:
        p.error('Keep BM3 output separate from the saved reference.')
    print('Indlæser MED-forbrug, priser og temperatur...' if args.tag=='MED'
          else f'Indlæser {args.tag}-forbrug, priser og temperatur...',flush=True)
    frame,measurements = aligned_data(args.tag,price_domain=(-1.,1.),price_year=2022)
    raw = measurements.mean(axis=1).reindex(frame.index)
    start = pd.Timestamp(args.start).normalize() if args.start else frame.index.min().normalize()
    end = pd.Timestamp(args.end).normalize() if args.end else min(frame.index.max().normalize(),END-pd.Timedelta(days=1))
    if start>end or start.tz is not None or end.tz is not None: p.error('Use ordered timezone-naive calendar dates.')
    tasks,skipped = eligible_tasks(frame,start,end,mode=args.mode,hours=args.hours)
    print(f'{start:%Y-%m-%d}..{end:%Y-%m-%d}: {len(tasks)} usable jobs on {len({t[0] for t in tasks})} days; '
          f'{sum(t[2]=="daily" for t in tasks)} daily, {sum(t[2]=="mpc" for t in tasks)} MPC origins; {len(skipped)} excluded jobs.',flush=True)
    if args.plan_only:
        print('Excluded reasons:',pd.DataFrame(skipped).reason.value_counts().to_dict() if skipped else {})
        return 0
    if not tasks: raise ValueError('No usable tasks in the requested period.')
    inputs = {name:sha(ROOT/'files'/'input'/'data'/name) for name in ('input.xlsx','prices.xlsx')}
    reuse,manifest = reuse_manifest(reference,tasks,inputs)
    generated = ROOT/'files'/'output'/'bm_1'/'runs'/(out.name+'_reference')
    source = {file.relative_to(ROOT).as_posix():sha(file) for file in sorted((ROOT/'src').rglob('*.py'))}
    frame_hash = hashlib.sha256(pd.util.hash_pandas_object(frame[['I','U','Temp']],index=True).values.tobytes()).hexdigest()
    core = {k:getattr(args,k) for k in ('tag','daily_max_nfev','daily_max_iterations','mpc_max_nfev','mpc_max_iterations','numeric_tolerance')}
    reference_settings = {**core,'schema':'bm1-response-period-v1','method':'alternating','days':sorted({t[0] for t in tasks}),
        'families':['logistic'],'mode':args.mode,'hours':sorted(set(args.hours)),
        'inputs_sha256':inputs,'source_sha256':source,'aligned_frame_sha256':frame_hash}
    reference_settings['fingerprint'] = hashlib.sha256(json.dumps(reference_settings,sort_keys=True).encode()).hexdigest()
    settings = {**core,'schema':'bm3-response-period-v1','model':'BM3','days':reference_settings['days'],
        'mode':args.mode,'hours':reference_settings['hours'],'reference':str(generated),'reference_by_day':reuse,
        'reference_manifest':manifest,'generated_reference_fingerprint':reference_settings['fingerprint'],
        'restoring_strength':strength,'state_bounds':list(bounds),'tail_tolerance':tail,'x_steps':args.x_steps,
        'fp_max_step':args.fp_max_step,'daily_sigma_x':0.,'refit':False,'joint_max_nfev':0,
        'inputs_sha256':inputs,'source_sha256':source,'aligned_frame_sha256':frame_hash,'BLAS_threads':1,
        'tasks':[list(t) for t in tasks],'requested_start':str(start.date()),'requested_end':str(end.date())}
    settings['fingerprint'] = hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    for directory,config in ((out,settings),(generated,reference_settings)):
        directory.mkdir(parents=True,exist_ok=True)
        file = directory/'settings.json'
        if file.exists() and json.loads(file.read_text())['fingerprint']!=config['fingerprint']:
            raise ValueError(f'Source/data/settings changed: choose a new output name ({directory}).')
        if not file.exists(): write_json(file,config)
        (directory/'prepared').mkdir(exist_ok=True); (directory/'logs').mkdir(exist_ok=True)
    pd.DataFrame(skipped).to_csv(out/'skipped_jobs.csv',index=False)
    prepared_manifest = {}
    for day in settings['days']:
        source_file = reference/'prepared'/f'{day}.pkl' if day in reuse else generated/'prepared'/f'{day}.pkl'
        if not source_file.exists():
            data = prepare_day(frame,raw,day,with_daily=any(t[0]==day and t[2]=='daily' for t in tasks))
            temporary = source_file.with_suffix('.tmp'); temporary.write_bytes(pickle.dumps(data)); temporary.replace(source_file)
        digest = sha(source_file); prepared_manifest[day] = digest
        target = out/'prepared'/f'{day}.pkl'
        if target.exists():
            if sha(target)!=digest: raise ValueError('Prepared data changed: '+day)
        else:
            import shutil
            shutil.copyfile(source_file,target)
        print(f'Data/baselines klar: {day}',flush=True)
    previous_manifest = out/'prepared_manifest.json'
    if previous_manifest.exists() and json.loads(previous_manifest.read_text())!=prepared_manifest:
        raise ValueError('Prepared source identity changed; use a new output folder.')
    write_json(previous_manifest,prepared_manifest)
    pending = [t for t in tasks if not completed(out,t,settings['fingerprint'])]
    for task in pending:
        job_folder(out,*task).mkdir(parents=True,exist_ok=True)
        if task[0] not in reuse: job_folder(generated,*task).mkdir(parents=True,exist_ok=True)
    print(f'{len(reuse)} komplette gemte BM1-dage genbruges.',flush=True)
    saved = len(tasks)-len(pending)
    print(f'{len(tasks)} jobs; {saved} gemt; {len(pending)} tilbage; workers={args.workers}; BLAS=1.',flush=True)
    started = time.perf_counter(); failures = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(period_worker,str(out),task):task for task in pending}
            for count,future in enumerate(as_completed(futures),start=1):
                task = futures[future]
                try: task,error = future.result()
                except Exception as exc:
                    error = str(exc); write_json(job_folder(out,*task)/'error.json',
                        {'task':list(task),'error':error,'traceback':traceback.format_exc()})
                if error: failures.append((list(task),error)); print('FEJL',task,error,flush=True)
                else: print_daily_result(out,task)
                print_progress(count,saved,len(tasks),failures,started)
    finally:
        summarize(out,settings,tasks)
        file = out/'runtime.json'
        runtime = json.loads(file.read_text()) if file.exists() else {'sessions':[]}
        runtime['sessions'].append({'workers':args.workers,'wall_seconds':time.perf_counter()-started,
            'completed_jobs':sum(completed(out,t,settings['fingerprint']) for t in tasks),
            'expected_jobs':len(tasks),'failed_tasks':failures})
        write_json(file,runtime)
    print('Færdig. Resultater:',out,flush=True)
    return 1 if failures else 0


if __name__=='__main__': raise SystemExit(main())
