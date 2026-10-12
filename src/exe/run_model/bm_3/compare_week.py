"""Compare BM3 with saved logistic BM1 jobs, reusing their prepared inputs and fits.

Run from the repository root; no old F/G pipeline or new raw-data format is needed.
Default fit reuse is valid only after checking identical historical BM1/BM3 drift.
The additive diffusion is ALWAYS recalibrated for BM3, never copied from BM1.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
import pandas as pd
from src.exe.run_model.bm_1.compare_fg_week import write_json, sha, job_folder, completed
from src.exe.run_model.bm_1.predict_daily import held_signal
from src.exe.run_model.bm_3.predict_daily import forecast_from_distribution, add_model_options
from src.exe.run_model.bm_3.predict_mpc import predict_origin
from src.model.bm_3 import bm_3
from src.params.bm_3 import lsq_params
from src.tools.forecast_metrics import forecast_metrics

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/'files'/'output'/'bm_3'/'runs'/'logistic_week_2022-05-02_2022-05-08'
REFERENCE = ROOT/'files'/'output'/'bm_1'/'runs'/'fg_week_2022-05-02_2022-05-08'


def reference_root(settings, day):
    """Period runs can reuse intact older reference collections by date."""
    return Path(settings.get('reference_by_day', {}).get(day, settings['reference']))


def restore_fit(metadata, strength):
    p = {**metadata['theta'], **metadata['shape_parameters'], 'restoring_strength': strength}
    if metadata['family'] != 'logistic': raise ValueError('BM3 first version supports the logistic reference.')
    p.update(f=lsq_params.response(p['alpha']), g=lsq_params.response(p['beta']),
        initial_states=np.asarray(metadata['state_X0'],float), initial_weights=np.full(5,.2),
        mse=metadata['training_rmse']**2, sse=120*metadata['training_rmse']**2,
        history=metadata['history'], iterations=len(metadata['history']),
        converged=metadata['converged'], stop_reason=metadata['stop_reason'],
        warning=None if metadata['converged'] else metadata['stop_reason'], message=metadata['stop_reason'],
        theta_nfev=metadata['history'][-1]['theta_nfev'])
    return p


def clean(value):
    if isinstance(value,dict): return {k: clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    return value


def worker(output, task):
    out = Path(output); settings = json.loads((out/'settings.json').read_text(encoding='utf-8'))
    folder = job_folder(out,*task); folder.mkdir(parents=True,exist_ok=True)
    logs = out/'logs'; logs.mkdir(exist_ok=True)
    logfile = logs/f'{task[0]}_{task[1]}_{task[2]}_{task[3]:02d}.log'
    started = time.perf_counter()
    try:
        day, family, kind, hour = task
        reference = reference_root(settings, day); oldfolder = job_folder(reference,*task)
        old = json.loads((oldfolder/'fit.json').read_text(encoding='utf-8'))
        with (out/'prepared'/f'{day}.pkl').open('rb') as handle: data = pickle.load(handle)
        fit = restore_fit(old,settings['restoring_strength'])
        if kind == 'daily': training = data['training']
        else:
            from src.validation.exe.run_model.bm_1.predict_mpc import bm_1_mpc_validation as checks
            origin = pd.Timestamp(day)+pd.Timedelta(hours=hour)
            training = [checks.window(data['mpc_frame'],origin-pd.Timedelta(weeks=k),data['mpc_baseline'])
                        for k in range(5,0,-1)]
        # Verify the exact historical equivalence required for transferring a saved fit.
        from src.params.bm_1 import lsq_params as BM1Fit
        equivalence = 0.
        for d,x0 in zip(training,fit['initial_states']):
            X,D = lsq_params._predict(d,float(x0),fit,32)
            if np.any((X < -1e-10) | (X > 1+1e-10)):
                raise ValueError('Cannot reuse a BM1 fit whose historical BM3 states leave [0,1].')
            _,oldD = BM1Fit._predict(d,float(x0),fit,32)
            equivalence = max(equivalence,float(np.max(abs(D-oldD))))
        if equivalence > settings['numeric_tolerance']:
            raise ValueError('BM1/BM3 deterministic historical dynamics do not match at the required tolerance.')
        if settings['refit']:
            model = bm_3(held_signal(training[0]['B']),lambda t:.5,(0.,24.),25,
                restoring_strength=settings['restoring_strength'], state_bounds=settings['state_bounds'],
                tail_tolerance=settings['tail_tolerance'])
            fit = lsq_params(model).fit_many(training, float(np.mean(fit['initial_states'])),
                alpha0=fit['alpha'], beta0=fit['beta'], k0=fit['k'], C0=fit['C'], Xi0=fit['Xi'],
                max_iterations=settings[f'{kind}_max_iterations'], max_nfev=settings[f'{kind}_max_nfev'],
                fit_bounds={'C':(.05,100.)} if kind=='mpc' else None,
                joint_max_nfev=settings['joint_max_nfev'])
        opts = dict(state_bounds=settings['state_bounds'],tail_tolerance=settings['tail_tolerance'])
        if kind == 'daily':
            prediction = forecast_from_distribution(data['target'],fit,
                sigma_x=settings['daily_sigma_x'],x_steps=settings['x_steps'],fp_max_step=settings['fp_max_step'],**opts)
            refined = forecast_from_distribution(data['target'],fit,substeps=32,
                sigma_x=settings['daily_sigma_x'],x_steps=2*settings['x_steps']-1,
                fp_max_step=settings['fp_max_step']/2,**opts)
            tab = pd.DataFrame({'time':data['target_index'],'I_predicted':prediction['forecast'],
                'I_observed':data['target_observed'],'B':data['target']['B'][:24],'U':data['target']['U'][:24]})
            for level,(lo,hi) in prediction['intervals'].items(): tab[f'lower_{level}'],tab[f'upper_{level}']=lo,hi
            fp = prediction['fp']; state = prediction['state_paths']
            tab['X_predicted_mean'] = fp['state_mean'][:24] if fp else state.mean(axis=0)
            tab['X_predicted_std'] = np.sqrt(fp['state_variance'][:24]) if fp else state.std(axis=0)
            tab['outside_probability'] = fp['diagnostics']['outside_probability'][:24] if fp else ((state<0)|(state>1)).mean(axis=0)
            tab['auxiliary_flow_mean'] = fp['diagnostics']['auxiliary_flow_mean'][:24] if fp else fit['C']*bm_3.restoring(state,settings['restoring_strength']).mean(axis=0)
            extra = {'process_sigma_x':settings['daily_sigma_x'], 'fp':fp['diagnostics'] if fp else None}
            gap = float(np.max(abs(prediction['forecast']-refined['forecast'])))
            csv,predcol,keys = 'hourly.csv','I_predicted',['time']
            np.savez_compressed(folder/'distribution.npz',demand_supports=prediction['demand_supports'],
                                probabilities=prediction['probabilities'])
            pd.DataFrame([{'day':str((pd.Timestamp(day)-pd.Timedelta(weeks=k)).date()),'X0':float(x),
                'rmse':math.sqrt(np.mean((lsq_params._predict(d,float(x),fit,32)[1][:24]-d['I'])**2))}
                for k,d,x in zip(range(5,0,-1),training,fit['initial_states'])]
                ).to_csv(folder/'training_days.csv',index=False)
        else:
            options = dict(fitted=fit,x_steps=settings['x_steps'],fp_max_step=settings['fp_max_step'],
                           restoring_strength=settings['restoring_strength'],**opts)
            rows,info = predict_origin(data['mpc_frame'],origin,data['mpc_baseline'],**options)
            finer,_ = predict_origin(data['mpc_frame'],origin,data['mpc_baseline'],
                **{**options,'x_steps':2*settings['x_steps']-1,'fp_max_step':settings['fp_max_step']/2,'substeps':32})
            tab = pd.DataFrame(rows); fit = info['fit']
            tab['I_observed'] = data['mpc_frame'].I.reindex(pd.DatetimeIndex(tab.target_time)).to_numpy()
            gap = float(np.max(abs(tab.I_BM3.to_numpy()-np.array([r['I_BM3'] for r in finer]))))
            extra = {'diffusion':info['diffusion'],'fp':info['fp']['diagnostics']}
            csv,predcol,keys = 'forecasts.csv','I_BM3',['origin','target_time','lead_hours']
        index = pd.DatetimeIndex(tab[keys[0] if kind=='daily' else 'target_time'])
        tab['I_observed_original'] = data['original_consumption'].reindex(index).to_numpy()
        for col in (predcol,'B'):
            tab[col+'_original'] = data['raw_offset']+data['raw_scale']*tab[col]
        oldtab = pd.read_csv(oldfolder/csv)
        for key in keys:
            if key == 'lead_hours': equal = np.array_equal(oldtab[key],tab[key])
            else: equal = np.array_equal(pd.to_datetime(oldtab[key]).to_numpy(),pd.to_datetime(tab[key]).to_numpy())
            if not equal: raise ValueError('Reference forecast timestamps differ: '+key)
        if not np.allclose(oldtab.I_observed,tab.I_observed,atol=1e-12,rtol=0):
            raise ValueError('Reference observations differ.')
        tab.to_csv(folder/csv,index=False)
        if kind=='daily':
            from src.exe.run_model.bm_3.plot_prediction import plot_saved
            plot_saved(folder)
        pd.DataFrame(fit['history']).to_csv(folder/'iterations.csv',index=False)
        bounds = {p:(tab[f'lower_{p}'].to_numpy(),tab[f'upper_{p}'].to_numpy()) for p in (68,95)}
        report = {'status':'complete','model':'BM3','family':'logistic','day':day,'kind':kind,'hour':hour,
            'settings_fingerprint':settings['fingerprint'],'fit_reused':not settings['refit'],
            'reference_folder':str(oldfolder),'theta':{k:float(fit[k]) for k in ('alpha','beta','k','Xi','C')},
            'restoring_strength':settings['restoring_strength'],'state_X0':fit['initial_states'].tolist(),
            'initial_weights':[.2]*5,'training_rmse':math.sqrt(fit['mse']),
            'converged':bool(fit['converged']),'stop_reason':fit['stop_reason'],'history':fit['history'],
            'historical_drift_equivalence_difference':equivalence,'numeric_forecast_difference':gap,
            'numerically_passed':gap<settings['numeric_tolerance'],
            'metrics':forecast_metrics(tab.I_observed,tab[predcol],bounds),
            'baseline_metrics':forecast_metrics(tab.I_observed,tab.B),
            'seconds':time.perf_counter()-started,**extra}
        write_json(folder/'fit.json',clean(report))
        logfile.write_text(json.dumps(clean(report),indent=2),encoding='utf-8')
        return task,None
    except Exception as exc:
        write_json(folder/'error.json',{'task':list(task),'error':str(exc),'traceback':traceback.format_exc()})
        logfile.write_text(traceback.format_exc(),encoding='utf-8')
        return task,str(exc)


def summarize(out, settings, tasks):
    jobs,tables = [],[]
    for task in tasks:
        if not completed(out,task,settings['fingerprint']): continue
        folder = job_folder(out,*task)
        report = json.loads((folder/'fit.json').read_text(encoding='utf-8'))
        kind = task[2]; csv = 'hourly.csv' if kind=='daily' else 'forecasts.csv'
        new = pd.read_csv(folder/csv); old = pd.read_csv(job_folder(reference_root(settings, task[0]),*task)/csv)
        p = 'I_predicted' if kind=='daily' else 'I_BM3'; oldp = 'I_predicted' if kind=='daily' else 'I_BM1'
        new['I_BM1_reference'] = old[oldp].to_numpy(); new['mode'] = kind
        tables.append(new.rename(columns={p:'prediction'}))
        jobs.append({'day':task[0],'mode':kind,'hour':task[3],'bm3_rmse':report['metrics']['rmse'],
            'bm1_rmse':math.sqrt(np.mean((old[oldp]-old.I_observed)**2)),
            'fit_reused':report['fit_reused'],'fit_converged':report['converged'],
            'numeric_forecast_difference':report['numeric_forecast_difference'],
            'numerically_passed':report['numerically_passed'],'seconds':report['seconds']})
    frame = pd.DataFrame(jobs)
    frame.to_csv(out/'fit_comparison.csv',index=False)
    for mode in ('daily','mpc'):
        (frame.loc[frame['mode']==mode] if not frame.empty else frame).to_csv(out/f'{mode}_summary.csv',index=False)
    if not tables: return
    values = pd.concat(tables,ignore_index=True); summary,leads = [],[]
    def metrics(part):
        old = forecast_metrics(part.I_observed,part.I_BM1_reference)['rmse']
        new = forecast_metrics(part.I_observed,part.prediction)['rmse']
        bounds = {p:(part[f'lower_{p}'].to_numpy(),part[f'upper_{p}'].to_numpy()) for p in (68,95)}
        return {'bm1_rmse':old,'bm3_rmse':new,'improvement_pct':100*(1-new/old) if old else None,
            'baseline_rmse':forecast_metrics(part.I_observed,part.B)['rmse'],
            **forecast_metrics(part.I_observed,part.prediction,bounds)}
    for mode,part in values.groupby('mode'):
        selected = frame[frame['mode']==mode]
        summary.append({'mode':mode,'family':'logistic',**metrics(part),
            'completed_jobs':len(selected),'expected_jobs':sum(t[2]==mode for t in tasks),
            'numeric_passed_jobs':int(selected.numerically_passed.sum()),
            'mean_outside_probability':float(part.outside_probability.mean())})
        if mode=='mpc':
            for lead,group in part.groupby('lead_hours'): leads.append({'lead_hours':int(lead),**metrics(group)})
    pd.DataFrame(summary).to_csv(out/'week_summary.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(leads).to_csv(out/'mpc_by_lead.csv',index=False)
    import matplotlib.pyplot as plt
    tab = pd.DataFrame(summary).set_index('mode')
    ax = tab[['bm1_rmse','bm3_rmse','baseline_rmse']].plot.bar(figsize=(9,5),rot=0)
    ax.set_ylabel('Normalized RMSE'); ax.grid(axis='y',alpha=.25)
    ax.figure.tight_layout(); ax.figure.savefig(out/'comparison.png',dpi=160); plt.close(ax.figure)


def main():
    p = argparse.ArgumentParser(description='BM3 vs saved logistic BM1, with existing prepared input and checkpoint layout.')
    p.add_argument('--reference',type=Path,default=REFERENCE); p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--start',default='2022-05-02'); p.add_argument('--end',default='2022-05-08')
    p.add_argument('--mode',choices=('daily','mpc','both'),default='both')
    p.add_argument('--hours',nargs='+',type=int,choices=range(24),default=list(range(24)))
    p.add_argument('--workers',type=int,default=8); p.add_argument('--x-steps',type=int,default=301)
    p.add_argument('--fp-max-step',type=float,default=.25); p.add_argument('--daily-sigma-x',type=float,default=0.)
    p.add_argument('--numeric-tolerance',type=float,default=1e-4)
    p.add_argument('--refit',action='store_true',help='Refit BM3 parameters; default reuses checked equivalent BM1 fits.')
    p.add_argument('--joint-max-nfev',type=int,default=0)
    add_model_options(p); args = p.parse_args()
    from src.validation.model.bm_3 import configuration
    strength,bounds,tail = configuration(args.restoring_strength,(args.state_min,args.state_max),args.tail_tolerance)
    if not 1<=args.workers<=32 or args.x_steps<20 or args.fp_max_step<=0 or args.numeric_tolerance<=0 or args.daily_sigma_x<0:
        p.error('Use positive finite budgets/grid sizes and nonnegative noise.')
    if not np.isfinite([args.fp_max_step,args.numeric_tolerance,args.daily_sigma_x]).all(): p.error('Options must be finite.')
    reference,out = args.reference.resolve(),args.output.resolve()
    if reference==out or reference in out.parents or out in reference.parents: p.error('Keep BM3 output separate from BM1 reference.')
    original = json.loads((reference/'settings.json').read_text(encoding='utf-8'))
    days = [str(d.date()) for d in pd.date_range(args.start,args.end)]
    if not days or not set(days)<=set(original['days']): p.error('Requested dates must exist in the BM1 reference.')
    if args.joint_max_nfev and not args.refit: p.error('--joint-max-nfev requires --refit.')
    tasks = []
    for day in days:
        if args.mode in ('daily','both'): tasks.append((day,'logistic','daily',-1))
        if args.mode in ('mpc','both'): tasks.extend((day,'logistic','mpc',h) for h in sorted(set(args.hours)))
    manifest = {'settings.json':sha(reference/'settings.json')}
    for day in days: manifest[f'prepared/{day}.pkl']=sha(reference/'prepared'/f'{day}.pkl')
    for task in tasks:
        if not completed(reference,task,original['fingerprint']): p.error(f'BM1 reference job is missing/incomplete: {task}')
        for name in ('fit.json','hourly.csv' if task[2]=='daily' else 'forecasts.csv'):
            file = job_folder(reference,*task)/name; manifest[file.relative_to(reference).as_posix()]=sha(file)
    # Prepared/reference identities are retained even when no new raw-data import is needed.
    for name,digest in original.get('inputs_sha256',{}).items():
        if sha(ROOT/'files'/'input'/'data'/name)!=digest: raise ValueError('Raw input identity changed: '+name)
    config = {'schema':'bm3-logistic-week-v1','model':'BM3','reference':str(reference),
        'days':days,'mode':args.mode,'hours':sorted(set(args.hours)),'family':'logistic',
        'restoring_strength':strength,'state_bounds':list(bounds),'tail_tolerance':tail,
        'x_steps':args.x_steps,'fp_max_step':args.fp_max_step,'daily_sigma_x':args.daily_sigma_x,
        'numeric_tolerance':args.numeric_tolerance,'refit':args.refit,'joint_max_nfev':args.joint_max_nfev,
        **{k:original[k] for k in ('daily_max_nfev','daily_max_iterations','mpc_max_nfev','mpc_max_iterations','inputs_sha256')},
        'reference_manifest':manifest,'source_sha256':{file.relative_to(ROOT).as_posix():sha(file) for file in sorted((ROOT/'src').rglob('*.py'))},
        'BLAS_threads':1}
    config['fingerprint']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'settings.json').exists():
        if json.loads((out/'settings.json').read_text())['fingerprint']!=config['fingerprint']:
            raise ValueError('Settings, sources or reference changed; use a new output folder.')
    else: write_json(out/'settings.json',config)
    import shutil
    (out/'prepared').mkdir(exist_ok=True)
    for day in days:
        target = out/'prepared'/f'{day}.pkl'
        if target.exists():
            if sha(target)!=manifest[f'prepared/{day}.pkl']: raise ValueError('Prepared checkpoint was modified.')
        else: shutil.copyfile(reference/'prepared'/f'{day}.pkl',target)
    pending = [t for t in tasks if not completed(out,t,config['fingerprint'])]
    # Create shared parents sequentially. Concurrent mkdir(parents=True) can
    # report transient PermissionError on Windows even when no ACL denies it.
    (out/'logs').mkdir(exist_ok=True)
    for task in pending: job_folder(out,*task).mkdir(parents=True,exist_ok=True)
    print(f'BM3: {len(tasks)} jobs, {len(pending)} remaining; workers={args.workers}, BLAS=1.',flush=True)
    start = time.perf_counter(); failures = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(worker,str(out),task):task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                try: task,error = future.result()
                except Exception as exc:
                    error = str(exc)
                    write_json(job_folder(out,*task)/'error.json',
                               {'task':list(task),'error':error,'traceback':traceback.format_exc()})
                if error: failures.append((task,error))
                print('ERROR' if error else 'SAVED',task,error or '',flush=True)
    finally:
        summarize(out,config,tasks)
        write_json(out/'runtime.json',{'wall_seconds':time.perf_counter()-start,'workers':args.workers,
            'failures':clean(failures),'completed_jobs':sum(completed(out,t,config['fingerprint']) for t in tasks),
            'expected_jobs':len(tasks)})
    print('Summary:',out/'comparison.csv')
    return 1 if failures else 0


if __name__=='__main__': raise SystemExit(main())
