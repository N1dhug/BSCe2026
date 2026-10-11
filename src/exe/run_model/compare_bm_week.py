"""Compare BM2 against saved BM1 forecasts on identical historical inputs.

python -B -m src.exe.run_model.compare_bm_week --workers 8
All six response families, daily mixtures and 1..8h MPC are supported.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from src.exe.run_model.bm_1.compare_fg_week import (
    FAMILIES,write_json,sha,job_folder,completed,load_modules,restore_fit,worker as bm1_worker)
from src.exe.run_model.bm_1.predict_daily import held_signal
from src.exe.run_model.bm_2.predict import forecast_daily,predict_origin,refine_forecast
from src.model.bm_2 import bm_2
from src.params.bm_2 import lsq_params

ROOT=Path(__file__).resolve().parents[3]


def reference_folder(out,task,settings):
    reference=Path(settings['reference'])
    if completed(reference,task,settings['reference_fingerprint']):return job_folder(reference,*task)
    recovery=out/'reference_completed'
    if completed(recovery,task,settings['recovery_fingerprint']):return job_folder(recovery,*task)
    return None


def worker(out,task):
    settings=json.loads((out/'settings.json').read_text(encoding='utf-8'))
    modules=load_modules(ROOT);day,family,kind,hour=task;folder=job_folder(out,*task)
    folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    try:
        oldfolder=reference_folder(out,task,settings)
        if oldfolder is None:
            if settings['bm1_variant']!='alternating':raise ValueError('Missing joint BM1 reference; complete it first.')
            if bm1_worker(ROOT,out/'reference_completed',task,modules):raise RuntimeError('BM1 reference recovery failed.')
            oldfolder=reference_folder(out,task,settings)
        old=json.loads((oldfolder/'fit.json').read_text(encoding='utf-8'))
        seed=restore_fit(old,modules)
        with (out/'prepared'/f'{day}.pkl').open('rb') as f:data=pickle.load(f)
        options={'joint_max_nfev':settings['joint_max_nfev'],
                 'joint_max_seconds':settings['joint_max_seconds'],'joint_gtol':settings['joint_gtol']}
        if kind=='daily':
            training=data['training'];target=data['target']
            model=bm_2(held_signal(training[0]['B']),lambda t:.5,(0.,24.),25)
            fit=lsq_params(model,family).fit_many(training,seed_fit=seed,
                max_iterations=settings['daily_max_iterations'],max_nfev=settings['daily_max_nfev'],
                state_maxiter=100,**options)
            pred=forecast_daily(target,fit);fine=forecast_daily(target,fit,substeps=32)
            predicted=pred['forecast'];index=data['target_index'];baseline=np.asarray(target['B'])[:24]
            tab=pd.DataFrame({'time':index,'I_observed':data['target_observed'],
                'I_predicted':predicted,'B':baseline,'U':np.asarray(target['U'])[:24],
                'I_observed_original':data['original_consumption'].reindex(index).to_numpy(),
                'I_predicted_original':data['raw_offset']+data['raw_scale']*predicted,
                'B_original':data['raw_offset']+data['raw_scale']*baseline})
            for level,(lo,hi) in pred['intervals'].items():tab[f'lower_{level}'],tab[f'upper_{level}']=lo,hi
            gap=float(np.max(abs(predicted-fine['forecast'])))
            extra={'process_sigma_x':0.,'daily_initial_pairs_only':True}
            csv='hourly.csv';predcol='I_predicted'
        else:
            origin=pd.Timestamp(day)+pd.Timedelta(hours=hour)
            rows,diag=predict_origin(data['mpc_frame'],origin,data['mpc_baseline'],family=family,
                max_iterations=settings['mpc_max_iterations'],max_nfev=settings['mpc_max_nfev'],
                state_maxiter=60,seed_fit=seed,x_steps=settings['x_steps'],r_steps=settings['r_steps'],
                fp_max_step=settings['fp_max_step'],**options)
            fit=diag['fit'];training=diag['training'];tab=pd.DataFrame(rows)
            fine,fp_fine=refine_forecast(diag,x_steps=2*settings['x_steps']-1,
                r_steps=2*settings['r_steps']-1,fp_max_step=settings['fp_max_step']/2)
            index=pd.DatetimeIndex(tab.target_time)
            tab['I_observed']=data['mpc_frame']['I'].reindex(index).to_numpy()
            tab['I_observed_original']=data['original_consumption'].reindex(index).to_numpy()
            tab['I_BM2_original']=data['raw_offset']+data['raw_scale']*tab.I_BM2
            tab['B_original']=data['raw_offset']+data['raw_scale']*tab.B
            tab['I_BM2_refined']=fine
            gap=float(np.max(abs(tab.I_BM2.to_numpy()-fine)))
            extra={'diffusion':diag['diffusion'],'fp':diag['fp'],'refined_fp':fp_fine,
                   'origin':str(origin),'posterior_mean_X':float(np.dot(bm_2.grid(*diag['posterior'].shape)[0],diag['posterior'].sum(axis=1))),
                   'posterior_mean_R':float(np.dot(bm_2.grid(*diag['posterior'].shape)[1],diag['posterior'].sum(axis=0)))}
            csv='forecasts.csv';predcol='I_BM2'
        training_gap=0.;boundary_samples=0
        for d,s in zip(training,fit['initial_states']):
            coarse,D=bm_2.predict(d,s,fit,2);_,Dfine=bm_2.predict(d,s,fit,32)
            training_gap=max(training_gap,float(np.max(abs(D-Dfine))))
            boundary_samples+=int(np.count_nonzero((coarse[:,0]<=1e-12)|(coarse[:,0]>=1-1e-12)))
        oldtab=pd.read_csv(oldfolder/csv)
        keys=['time'] if kind=='daily' else ['origin','target_time','lead_hours']
        # Compare times before writing, allowing datetime objects versus CSV strings.
        for key in keys:
            a,b=oldtab[key],tab[key]
            equal=np.array_equal(a.to_numpy(),b.to_numpy()) if key=='lead_hours' else np.array_equal(pd.to_datetime(a).to_numpy(),pd.to_datetime(b).to_numpy())
            if not equal:raise ValueError('BM1/BM2 forecast timestamps differ: '+key)
        if not np.allclose(oldtab.I_observed,tab.I_observed,rtol=0,atol=1e-12):raise ValueError('BM1/BM2 observations differ.')
        tab.to_csv(folder/csv,index=False)
        rmse=lambda p,y:float(np.sqrt(np.mean((np.asarray(p)-np.asarray(y))**2)))
        result={'status':'complete','model':'BM2','day':day,'family':family,'kind':kind,
            'bm1_variant':settings['bm1_variant'],'reference_folder':str(oldfolder),
            'theta':{n:float(fit[n]) for n in ('k','Xi','C','tau')},'gamma_fixed':1.,
            'shape_parameters':fit['shape_parameters'],'f':fit['f'].to_record(),'g':fit['g'].to_record(),
            'state_X0':fit['initial_states'][:,0].tolist(),'state_R0':fit['initial_states'][:,1].tolist(),
            'state_V0':fit['initial_velocities'].tolist(),'initial_weights':[.2]*5,
            'training_rmse':float(np.sqrt(fit['mse'])),'reference_training_rmse':old['training_rmse'],
            'rmse':rmse(tab[predcol],tab.I_observed),'rmse_original':rmse(tab[predcol+'_original'],tab.I_observed_original),
            'baseline_rmse':rmse(tab.B,tab.I_observed),'n':len(tab),'history':fit['history'],
            'converged':bool(fit['converged']),'stop_reason':fit['stop_reason'],'joint_fit':fit['joint_fit'],
            'numeric_training_difference':training_gap,'numeric_forecast_difference':gap,
            'numerically_passed':max(training_gap,gap)<settings['numeric_tolerance'],
            'training_boundary_samples':boundary_samples,'tau_near_bound':bool(fit['tau']<.0501 or fit['tau']>47.99),
            'seconds':time.perf_counter()-started,'settings_fingerprint':settings['fingerprint'],**extra}
        write_json(folder/'fit.json',result)
        print('SAVED',task,'BM2 RMSE',round(result['rmse'],6),'vs BM1',round(old['rmse'],6),flush=True)
        return 0
    except Exception as exc:
        import traceback
        write_json(folder/'error.json',{'task':list(task),'error':str(exc),'traceback':traceback.format_exc()})
        print(traceback.format_exc(),flush=True);return 1


def tasks_for(settings):
    for day in settings['days']:
        if settings['mode'] in ('daily','both'):
            for family in settings['families']:yield day,family,'daily',-1
        if settings['mode'] in ('mpc','both'):
            for hour in settings['hours']:
                for family in settings['families']:yield day,family,'mpc',hour


def summarize(out,settings,plot=False):
    records=[];tables=[]
    for task in tasks_for(settings):
        if not completed(out,task,settings['fingerprint']):continue
        newfolder=job_folder(out,*task);oldfolder=reference_folder(out,task,settings)
        old=json.loads((oldfolder/'fit.json').read_text());new=json.loads((newfolder/'fit.json').read_text())
        csv='hourly.csv' if task[2]=='daily' else 'forecasts.csv'
        a=pd.read_csv(oldfolder/csv);b=pd.read_csv(newfolder/csv)
        acol='I_predicted' if task[2]=='daily' else 'I_BM1';bcol='I_predicted' if task[2]=='daily' else 'I_BM2'
        t=pd.DataFrame({'family':task[1],'mode':task[2],'old':a[acol],'new':b[bcol],
            'observed':b.I_observed,'baseline':b.B,'old_raw':a[acol+'_original'],
            'new_raw':b[bcol+'_original'],'observed_raw':b.I_observed_original,'baseline_raw':b.B_original})
        if task[2]=='mpc':t['lead_hours']=b.lead_hours
        tables.append(t)
        records.append({'day':task[0],'family':task[1],'mode':task[2],'hour':task[3],
            'bm1_training_rmse':old['training_rmse'],'bm2_training_rmse':new['training_rmse'],
            'bm1_rmse':old['rmse'],'bm2_rmse':new['rmse'],'bm1_converged':old['converged'],
            'bm2_converged':new['converged'],'bm1_numeric_passed':old['numerically_passed'],
            'bm2_numeric_passed':new['numerically_passed'],'bm1_seconds':old['seconds'],'bm2_seconds':new['seconds'],
            'bm2_numeric_training_difference':new['numeric_training_difference'],
            'bm2_numeric_forecast_difference':new['numeric_forecast_difference'],
            'tau':new['theta']['tau'],'tau_near_bound':new['tau_near_bound'],
            'training_boundary_samples':new['training_boundary_samples'],'bm2_stop_reason':new['stop_reason']})
    def save(name,rows):
        temp=out/(name+'.tmp');pd.DataFrame(rows).to_csv(temp,index=False);temp.replace(out/name)
    save('fit_comparison.csv',records)
    if not tables:return
    data=pd.concat(tables,ignore_index=True);fits=pd.DataFrame(records);rows=[];leads=[]
    def metrics(t):
        rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
        old,new=rmse(t.old,t.observed),rmse(t.new,t.observed)
        return {'n':len(t),'bm1_rmse':old,'bm2_rmse':new,'improvement_pct':100*(1-new/old) if old else None,
            'bm1_mae':float(np.mean(abs(t.old-t.observed))),'bm2_mae':float(np.mean(abs(t.new-t.observed))),
            'baseline_rmse':rmse(t.baseline,t.observed),'bm1_rmse_original':rmse(t.old_raw,t.observed_raw),
            'bm2_rmse_original':rmse(t.new_raw,t.observed_raw),'baseline_rmse_original':rmse(t.baseline_raw,t.observed_raw)}
    for (mode,family),t in data.groupby(['mode','family'],sort=False):
        f=fits[(fits['mode']==mode)&(fits.family==family)]
        rows.append({'mode':mode,'family':family,'bm1_variant':settings['bm1_variant'],**metrics(t),
            'paired_jobs':len(f),'expected_jobs':len(settings['days'])*(1 if mode=='daily' else len(settings['hours'])),
            'bm1_converged_jobs':int(f.bm1_converged.sum()),'bm2_converged_jobs':int(f.bm2_converged.sum()),
            'bm1_numeric_passed_jobs':int(f.bm1_numeric_passed.sum()),'bm2_numeric_passed_jobs':int(f.bm2_numeric_passed.sum()),
            'all_paired_numeric_checks_passed':bool(f.bm1_numeric_passed.all() and f.bm2_numeric_passed.all()),
            'bm2_max_forecast_refinement_difference':float(f.bm2_numeric_forecast_difference.max()),
            'bm1_worker_seconds':float(f.bm1_seconds.sum()),'bm2_worker_seconds':float(f.bm2_seconds.sum())})
        if mode=='mpc':
            for lead,part in t.groupby('lead_hours'):leads.append({'family':family,'lead_hours':int(lead),**metrics(part)})
    save('comparison.csv',rows);save('comparison_by_lead.csv',leads)
    if plot:
        import matplotlib.pyplot as plt
        tab=pd.DataFrame(rows);modes=list(tab['mode'].unique())
        fig,axes=plt.subplots(1,len(modes),figsize=(7*len(modes),4.8),squeeze=False)
        for ax,mode in zip(axes[0],modes):
            t=tab[tab['mode']==mode];x=np.arange(len(t));ax.bar(x-.19,t.bm1_rmse,.38,label='BM1');ax.bar(x+.19,t.bm2_rmse,.38,label='BM2')
            ax.plot(x,t.baseline_rmse,'k_',markersize=14,label='Baseline');ax.set_xticks(x,t.family,rotation=25,ha='right')
            failed=int((~t.all_paired_numeric_checks_passed).sum())
            ax.set_title(mode+(f' ({failed} families: numerical checks incomplete)' if failed else ''))
            ax.set_ylabel('Normalized RMSE');ax.legend();ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=Path('files/output/bm_1/runs/fg_week_2022-05-02_2022-05-08'))
    parser.add_argument('--output',type=Path,default=Path('files/output/bm_2/runs/week_2022-05-02_2022-05-08'))
    parser.add_argument('--start',default='2022-05-02');parser.add_argument('--end',default='2022-05-08')
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--mode',choices=('daily','mpc','both'),default='both')
    parser.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    parser.add_argument('--hours',nargs='+',type=int,choices=range(24),default=list(range(24)))
    parser.add_argument('--x-steps',type=int,default=101);parser.add_argument('--r-steps',type=int,default=41)
    parser.add_argument('--fp-max-step',type=float,default=.25)
    parser.add_argument('--_job',nargs=4,help=argparse.SUPPRESS)
    args=parser.parse_args();out=args.output.resolve();reference=args.reference.resolve()
    if args._job:return worker(out,(args._job[0],args._job[1],args._job[2],int(args._job[3])))
    if not 1<=args.workers<=32:parser.error('workers must be 1..32.')
    bm_2.grid(args.x_steps,args.r_steps)
    if not np.isfinite(args.fp_max_step) or args.fp_max_step<=0:parser.error('fp-max-step must be positive.')
    if out==reference or reference in out.parents or out in reference.parents:parser.error('Use an independent output directory.')
    old=json.loads((reference/'settings.json').read_text(encoding='utf-8'))
    if old.get('schema') not in ('bm1-response-week-v1','bm1-joint-week-v1'):parser.error('Unsupported BM1 reference schema.')
    days=[str(d.date()) for d in pd.date_range(args.start,args.end)]
    if not days or not set(days)<=set(old['days']):parser.error('Requested dates are absent from the BM1 reference.')
    families=list(dict.fromkeys(args.families));hours=sorted(set(args.hours))
    if not set(families)<=set(old['families']):parser.error('Requested families are absent from the BM1 reference.')
    for name,digest in old['inputs_sha256'].items():
        if sha(ROOT/'files/input/data'/name)!=digest:raise ValueError('Input changed since BM1 reference: '+name)
    core={k:old[k] for k in ('tag','daily_max_nfev','daily_max_iterations','mpc_max_nfev','mpc_max_iterations','numeric_tolerance','inputs_sha256')}
    code={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'src').rglob('*.py'))}
    recovery={**core,'schema':'bm1-reference-completion-v1','method':'alternating','prepared_root':str(out/'prepared'),'source_sha256':code}
    recovery['fingerprint']=hashlib.sha256(json.dumps(recovery,sort_keys=True).encode()).hexdigest()
    settings={**core,'schema':'bm1-bm2-week-v1','days':days,'families':families,'hours':hours,'mode':args.mode,
        'reference':str(reference),'reference_fingerprint':old['fingerprint'],'source_sha256':code,
        'bm1_variant':'alternating+joint' if old['schema']=='bm1-joint-week-v1' else 'alternating',
        'joint_max_nfev':old.get('joint_max_nfev',0),'joint_max_seconds':old.get('joint_max_seconds',20.),
        'joint_gtol':old.get('joint_gtol',1e-5),'recovery_fingerprint':recovery['fingerprint'],
        'x_steps':args.x_steps,'r_steps':args.r_steps,'fp_max_step':args.fp_max_step,'BLAS_threads':1}
    manifest={'settings.json':sha(reference/'settings.json')}
    for day in days:manifest[f'prepared/{day}.pkl']=sha(reference/'prepared'/f'{day}.pkl')
    for task in tasks_for(settings):
        if completed(reference,task,old['fingerprint']):
            folder=job_folder(reference,*task)
            for name in ('fit.json','hourly.csv' if task[2]=='daily' else 'forecasts.csv'):
                path=folder/name;manifest[path.relative_to(reference).as_posix()]=sha(path)
        elif settings['bm1_variant']!='alternating':raise ValueError('Joint reference is incomplete: '+str(task))
    settings['reference_manifest_sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    settings['fingerprint']=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'settings.json').exists():
        previous=json.loads((out/'settings.json').read_text())
        if previous['fingerprint']!=settings['fingerprint']:raise ValueError('Code, reference or settings changed. Choose a new output directory.')
    else:write_json(out/'settings.json',settings)
    write_json(out/'reference_manifest.json',manifest);write_json(out/'reference_completed'/'settings.json',recovery)
    (out/'prepared').mkdir(exist_ok=True)
    for day in days:
        dest=out/'prepared'/f'{day}.pkl'
        if dest.exists():
            if sha(dest)!=manifest[f'prepared/{day}.pkl']:raise ValueError('Prepared input changed: '+str(dest))
        else:shutil.copyfile(reference/'prepared'/f'{day}.pkl',dest)
    tasks=list(tasks_for(settings));pending=[t for t in tasks if not completed(out,t,settings['fingerprint'])]
    already=len(tasks)-len(pending);finished=already;active=[];errors=[];started=time.perf_counter()
    logs=out/'logs';logs.mkdir(exist_ok=True)
    print(f'{len(tasks)} BM2 jobs; {already} saved; workers={args.workers}; BLAS=1.',flush=True)
    try:
        while pending or active:
            while pending and len(active)<args.workers:
                task=pending.pop(0);log=(logs/f'{task[0]}_{task[1]}_{task[2]}_{task[3]}.log').open('w',encoding='utf-8')
                proc=subprocess.Popen([sys.executable,'-B','-m','src.exe.run_model.compare_bm_week','--output',str(out),'--_job',*map(str,task)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active.append((proc,task,log))
            for item in list(active):
                proc,task,log=item
                if proc.poll() is None:continue
                log.close();active.remove(item);finished+=1
                if proc.returncode or not completed(out,task,settings['fingerprint']):errors.append(task);print('ERROR',task,flush=True)
                else:
                    fit=json.loads((job_folder(out,*task)/'fit.json').read_text())
                    if task[2]=='daily':print('DAILY',task[0],task[1],'BM2 RMSE',round(fit['rmse'],6),flush=True)
                if finished%100==0:summarize(out,settings)
                if finished%10==0 or not(pending or active):print(f'Status {finished}/{len(tasks)}, errors={len(errors)}, minutes={(time.perf_counter()-started)/60:.1f}',flush=True)
            time.sleep(.3)
    except KeyboardInterrupt:
        print('Stopping workers; completed results are retained.',flush=True);return 130
    finally:
        for proc,task,log in active:
            if proc.poll() is None:proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill()
            log.close()
        summarize(out,settings,plot=True)
        runtime_path=out/'runtime.json';runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {'sessions':[]}
        runtime['sessions'].append({'workers':args.workers,'wall_seconds':time.perf_counter()-started,
            'completed_at_start':already,'completed_at_end':sum(completed(out,t,settings['fingerprint']) for t in tasks),
            'failed_tasks':[list(t) for t in errors]})
        write_json(runtime_path,runtime)
    print('Comparison:',out/'comparison.csv',flush=True)
    report=out/'fit_comparison.csv'
    if report.exists() and report.stat().st_size>2:
        checks=pd.read_csv(report);failed=int((~checks.bm2_numeric_passed).sum())
        print(f'BM2 numerical checks: {len(checks)-failed}/{len(checks)} passed.',flush=True)
        if failed:print('Refinement differences remain. Inspect numerical flags before interpreting model rankings.',flush=True)
    return 1 if errors else 0


if __name__=='__main__':sys.exit(main())
