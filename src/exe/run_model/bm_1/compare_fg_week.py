"""Jointly refine saved five-window fits and compare on the same held-out week.

Run: python -m src.exe.run_model.bm_1.compare_fg_week --reference OLD_RUN --workers 8
Missing reference jobs are completed separately; existing references are read-only.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
import sys, json, pickle, hashlib, subprocess, time, math, argparse
from pathlib import Path

FAMILIES=('linear','piecewise-linear','pchip','bernstein','beta-cdf','logistic')
ROOT=Path(__file__).resolve().parents[4]

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def source_context(repo):
    branch=subprocess.run(['git','-C',str(repo),'branch','--show-current'],capture_output=True,text=True)
    if branch.returncode==0 and branch.stdout.strip()!='main':
        raise ValueError(f'Kør fra main. Nuværende branch er {branch.stdout.strip() or "detached HEAD"}.')
    commit=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True)
    return commit.stdout.strip() if commit.returncode==0 else 'source-copy-without-git'

def load_modules(repo):
    sys.path.insert(0,str(repo))
    import numpy as np
    import pandas as pd
    from src.exe.run_model.bm_1 import predict_daily as daily
    from src.exe.run_model.bm_1 import predict_mpc as mpc
    from src.exe.run_model.wm_3.fit_weekly import aligned_data
    from src.params.bm_1 import lsq_params as ThomasFit, historical_diffusion_params as ThomasDiffusion
    from src.validation.exe.run_model.compare_model_predictions import model_comparison_validation as scale
    from src.tools.baseline import statistical_output
    from src.params.bm_1_response import BM1LogisticResponse
    from src.params.bm_1_families import FamilyFit,FamilyDiffusion,shape_config,curves
    from src.params.bm_1_response import response_from_record
    return locals()


def reference_folder(out,task,settings):
    original=job_folder(Path(settings['reference']),*task)
    if completed(Path(settings['reference']),task,settings['reference_fingerprint']):return original
    recovered=out/'reference_completed'
    if completed(recovered,task,settings['recovery_fingerprint']):return job_folder(recovered,*task)
    return None


def restore_fit(metadata,modules):
    np=modules['np']
    p={**metadata['theta'],**metadata['shape_parameters']}
    p.update(f=modules['response_from_record'](metadata['f']),g=modules['response_from_record'](metadata['g']))
    if metadata['family']!='logistic':p['scalar_f']=p['f'].scalar_evaluator()
    p.update(family=metadata['family'],shape_parameters=metadata['shape_parameters'],
        initial_states=np.asarray(metadata['state_X0']),initial_weights=np.full(5,.2),
        mse=metadata['training_rmse']**2,sse=metadata['training_rmse']**2*120,
        history=metadata['history'],iterations=len(metadata['history']),
        converged=metadata['converged'],stop_reason=metadata['stop_reason'],
        warning=None if metadata['converged'] else metadata['stop_reason'],message=metadata['stop_reason'],
        theta_nfev=metadata['history'][-1]['theta_nfev'])
    return p


def joint_from_reference(model,training,fitter,out,task,settings,modules):
    folder=reference_folder(out,task,settings)
    metadata=json.loads((folder/'fit.json').read_text(encoding='utf-8'))
    family=task[1];initial=restore_fit(metadata,modules)
    options={'max_nfev':settings['joint_max_nfev'],'max_seconds':settings['joint_max_seconds'],
             'gtol':settings['joint_gtol'],'substeps':2}
    if task[2]=='mpc':options['fit_bounds']={'C':(.05,100.)}
    if family in ('piecewise-linear','bernstein'):
        linear_task=(task[0],'linear',task[2],task[3])
        linear_folder=reference_folder(out,linear_task,settings)
        if linear_folder is None:raise ValueError('The linear reference is required for the nested-family fallback.')
        options['linear_fit']=restore_fit(json.loads((linear_folder/'fit.json').read_text(encoding='utf-8')),modules)
    fitted=fitter(model).refine_joint(training,initial,**options)
    fitted['joint_fit'].update(reference_worker_seconds=metadata['seconds'],
        reference_training_rmse_recorded=metadata['training_rmse'],
        reference_converged=metadata['converged'],reference_stop_reason=metadata['stop_reason'])
    return fitted

def record(fit,family,modules):
    if family=='logistic':
        f=modules['BM1LogisticResponse'](fit['alpha']);g=modules['BM1LogisticResponse'](fit['beta'])
        shapes={'alpha':float(fit['alpha']),'beta':float(fit['beta'])}
    else:f,g,shapes=fit['f'],fit['g'],fit['shape_parameters']
    return {'family':family,'theta':{key:float(fit[key]) for key in ('C','k','Xi')},
        'shape_parameters':shapes,'f':f.to_record(),'g':g.to_record(),
        'state_X0':fit['initial_states'].tolist(),'initial_weights':[.2]*5,
        'converged':bool(fit['converged']),'stop_reason':fit['stop_reason'],
        'training_rmse':math.sqrt(fit['mse']),'history':fit['history'],
        'joint_fit':fit.get('joint_fit')}

def job_folder(out,day,family,kind,hour):
    return out/day/family/kind/(f'origin_{hour:02d}' if kind=='mpc' else 'forecast')

def worker(repo,out,task,modules):
    day,family,kind,hour=task
    settings=json.loads((out/'settings.json').read_text(encoding='utf-8'))
    np,pd=modules['np'],modules['pd'];daily=modules['daily'];mpc=modules['mpc']
    FamilyFit=modules['FamilyFit'];FamilyFit.FAMILY=family
    fitter=modules['ThomasFit'] if family=='logistic' else FamilyFit
    daily.lsq_params=fitter;mpc.lsq_params=fitter
    mpc.historical_diffusion_params=modules['ThomasDiffusion'] if family=='logistic' else modules['FamilyDiffusion']
    folder=job_folder(out,*task);folder.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    try:
        if settings['method']=='joint' and reference_folder(out,task,settings) is None:
            print('Completing missing reference:',task,flush=True)
            if worker(repo,out/'reference_completed',task,modules):raise RuntimeError('Missing reference could not be completed.')
        prepared_root=Path(settings.get('prepared_root',out/'prepared'))
        with (prepared_root/f'{day}.pkl').open('rb') as f:data=pickle.load(f)
        rmse=lambda a,b:float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
        if kind=='daily':
            target=data['target'];training=data['training']
            model=mpc.bm_1(daily.held_signal(target['B']),lambda t:.5,(0.,24.),25)
            if settings['method']=='joint':fit=joint_from_reference(model,training,fitter,out,task,settings,modules)
            else:
                fit=fitter(model).fit_many(training,max_iterations=settings['daily_max_iterations'],
                    max_nfev=settings['daily_max_nfev'],state_maxiter=100,substeps=2)
            prediction=daily.forecast_from_distribution(target,fit)
            refined=daily.forecast_from_distribution(target,fit,substeps=32)
            train_gap=max(float(np.max(abs(fitter._predict(d,float(x),fit,32)[1]-fitter._predict(d,float(x),fit,2)[1]))) for d,x in zip(training,fit['initial_states']))
            gap=float(np.max(abs(prediction['forecast']-refined['forecast'])))
            index=data['target_index'];observed=data['target_observed'];predicted=prediction['forecast'];b=target['B'][:24]
            actual_raw=data['original_consumption'].reindex(index).to_numpy()
            predicted_raw=data['raw_offset']+data['raw_scale']*predicted
            baseline_raw=data['raw_offset']+data['raw_scale']*b
            hourly=pd.DataFrame({'I_observed':observed,'I_predicted':predicted,'B':b,'U':target['U'][:24],
                'I_observed_original':actual_raw,'I_predicted_original':predicted_raw,'B_original':baseline_raw},index=index)
            for p,(lo,hi) in prediction['intervals'].items():hourly[f'lower_{p}'],hourly[f'upper_{p}']=lo,hi
            hourly.to_csv(folder/'hourly.csv',index_label='time')
            trainrows=[]
            for t,d,x in zip(daily.matching_days(pd.Timestamp(day)),training,fit['initial_states']):
                trainrows.append({'day':str(t.date()),'X0':float(x),'rmse':rmse(d['I'],fitter._predict(d,float(x),fit,2)[1][:24]),'baseline_rmse':rmse(d['I'],d['B'][:24])})
            pd.DataFrame(trainrows).to_csv(folder/'training_days.csv',index=False)
            result={**record(fit,family,modules),'day':day,'kind':kind,'n':24,
                'rmse':rmse(observed,predicted),'baseline_rmse':rmse(observed,b),
                'rmse_original':rmse(actual_raw,predicted_raw),'baseline_rmse_original':rmse(actual_raw,baseline_raw),
                'numeric_training_difference':train_gap,'numeric_forecast_difference':gap,
                'numerically_passed':max(train_gap,gap)<settings['numeric_tolerance']}
        else:
            origin=pd.Timestamp(day)+pd.Timedelta(hours=hour)
            fitted=None
            if settings['method']=='joint':
                training=[mpc.checks.window(data['mpc_frame'],origin-pd.Timedelta(weeks=k),data['mpc_baseline'],hours=24) for k in range(5,0,-1)]
                model=mpc.bm_1(daily.held_signal(training[0]['B']),lambda t:.5,(0.,24.),25)
                fitted=joint_from_reference(model,training,fitter,out,task,settings,modules)
            rows,diag=mpc.predict_origin(data['mpc_frame'],origin,data['mpc_baseline'],
                max_iterations=settings['mpc_max_iterations'],max_nfev=settings['mpc_max_nfev'],state_maxiter=60,fitted=fitted)
            fit=diag['fit'];future=mpc.checks.window(data['mpc_frame'],origin,data['mpc_baseline'],hours=9,observations=False)
            training=[mpc.checks.window(data['mpc_frame'],t,data['mpc_baseline'],hours=24) for t in diag['training_origins']]
            train_gap=max(float(np.max(abs(fitter._predict(d,float(x),fit,32)[1]-fitter._predict(d,float(x),fit,2)[1]))) for d,x in zip(training,fit['initial_states']))
            model=mpc.bm_1(daily.held_signal(future['B']),lambda t:.5,(0.,8.),9)
            model.fokker_planc_reference=mpc.finite_volume_1d_cached()
            fp=model.fokker_planc_reference_solver(diag['posterior'],[daily.held_signal(future['U']),fit['f'],fit['g'],fit['C'],fit['k'],fit['Xi']],
                diag['diffusion']['sigma_x'],t_eval=np.arange(9,dtype=float),max_step=.125)
            _,grid,_,mass,*_=fp;fine=[]
            for row in rows:
                lead=row['lead_hours'];ts=row['target_time'];row['I_observed']=float(data['mpc_frame'].loc[ts,'I']);row['B']=float(future['B'][lead])
                row['I_observed_original']=float(data['original_consumption'].loc[ts])
                row['I_BM1_original']=data['raw_offset']+data['raw_scale']*row['I_BM1']
                row['B_original']=data['raw_offset']+data['raw_scale']*row['B']
                delta=np.tanh(.5*fit['k']*(fit['f'](grid)+fit['g'](future['U'][lead])))
                output=future['B'][lead]+fit['Xi']*delta*np.where(delta>=0,1-future['B'][lead],future['B'][lead])
                fine.append(float(output@mass[lead]));row['I_BM1_finer_FP']=fine[-1]
            tab=pd.DataFrame(rows);tab.to_csv(folder/'forecasts.csv',index=False)
            fp_gap=float(np.max(abs(tab.I_BM1.to_numpy()-np.asarray(fine))))
            result={**record(fit,family,modules),'day':day,'kind':kind,'origin':str(origin),'n':len(rows),
                'diffusion':diag['diffusion'],'rmse':rmse(tab.I_observed,tab.I_BM1),'baseline_rmse':rmse(tab.I_observed,tab.B),
                'rmse_original':rmse(tab.I_observed_original,tab.I_BM1_original),'baseline_rmse_original':rmse(tab.I_observed_original,tab.B_original),
                'numeric_training_difference':train_gap,'numeric_FP_difference':fp_gap,
                'numerically_passed':max(train_gap,fp_gap)<settings['numeric_tolerance']}
        result.update(status='complete',seconds=time.perf_counter()-start,settings_fingerprint=settings['fingerprint'])
        write_json(folder/'fit.json',result)
        print('SAVED',day,family,kind,hour,'RMSE',result['rmse'],'converged',result['converged'],flush=True)
        return 0
    except Exception as exc:
        import traceback
        write_json(folder/'error.json',{'task':list(task),'error':str(exc),'traceback':traceback.format_exc()})
        print(traceback.format_exc(),flush=True);return 1

def completed(out,task,fingerprint):
    folder=job_folder(out,*task)
    try:
        result=json.loads((folder/'fit.json').read_text(encoding='utf-8'))
        return result.get('status')=='complete' and result.get('settings_fingerprint')==fingerprint and (folder/('hourly.csv' if task[2]=='daily' else 'forecasts.csv')).is_file()
    except (OSError,ValueError):return False

def summary(out,settings,modules):
    pd,np=modules['pd'],modules['np'];rows=[];daily_rows=[];mpc_rows=[]
    for day in settings['days']:
        for family in settings['families']:
            if settings['mode'] in ('daily','both'):
                folder=job_folder(out,day,family,'daily',-1)
                if completed(out,(day,family,'daily',-1),settings['fingerprint']):
                    fit=json.loads((folder/'fit.json').read_text());daily_rows.append({k:fit[k] for k in ('day','family','training_rmse','rmse','baseline_rmse','rmse_original','baseline_rmse_original','converged','numerically_passed')})
                    tab=pd.read_csv(folder/'hourly.csv');tab['family']=family;tab['day']=day
                    rows.append(('daily',family,tab))
            files=[];meta=[]
            if settings['mode'] in ('mpc','both'):
                for hour in range(24):
                    task=(day,family,'mpc',hour);folder=job_folder(out,*task)
                    if completed(out,task,settings['fingerprint']):
                        files.append(pd.read_csv(folder/'forecasts.csv'));meta.append(json.loads((folder/'fit.json').read_text()))
            if files:
                tab=pd.concat(files,ignore_index=True);rows.append(('mpc',family,tab))
                mpc_rows.append({'day':day,'family':family,'origins':len(files),'n':len(tab),'rmse':float(np.sqrt(np.mean((tab.I_BM1-tab.I_observed)**2))),
                    'rmse_original':float(np.sqrt(np.mean((tab.I_BM1_original-tab.I_observed_original)**2))),
                    'baseline_rmse':float(np.sqrt(np.mean((tab.B-tab.I_observed)**2))),
                    'converged_origins':sum(x['converged'] for x in meta),'numeric_passed_origins':sum(x['numerically_passed'] for x in meta)})
    aggregates=[];leads=[]
    for mode in ('daily','mpc'):
        for family in settings['families']:
            tables=[t for m,f,t in rows if m==mode and f==family]
            if not tables:continue
            tab=pd.concat(tables,ignore_index=True);pred='I_predicted' if mode=='daily' else 'I_BM1'
            aggregates.append({'mode':mode,'family':family,'n':len(tab),
                'rmse':float(np.sqrt(np.mean((tab[pred]-tab.I_observed)**2))),
                'rmse_original':float(np.sqrt(np.mean((tab[pred+'_original']-tab.I_observed_original)**2))),
                'baseline_rmse':float(np.sqrt(np.mean((tab.B-tab.I_observed)**2))),
                'baseline_rmse_original':float(np.sqrt(np.mean((tab.B_original-tab.I_observed_original)**2)))})
            if mode=='mpc':
                for lead,g in tab.groupby('lead_hours'):
                    leads.append({'family':family,'lead_hours':int(lead),'n':len(g),'rmse':float(np.sqrt(np.mean((g.I_BM1-g.I_observed)**2))),
                        'rmse_original':float(np.sqrt(np.mean((g.I_BM1_original-g.I_observed_original)**2)))})
    for filename,table in [('daily_summary.csv',daily_rows),('mpc_summary.csv',mpc_rows),('week_summary.csv',aggregates),('mpc_by_lead.csv',leads)]:
        temporary=out/(filename+'.tmp');pd.DataFrame(table).to_csv(temporary,index=False)
        try:temporary.replace(out/filename)
        except PermissionError:temporary.replace(out/(filename+'.pending.csv'))

def main():
    import shutil
    from src.exe.run_model.bm_1._fg_comparison import compare
    p=argparse.ArgumentParser(description='Fælles efteroptimering og parvis sammenligning med den gemte uge.')
    p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--reference',type=Path,default=Path('files/output/bm_1/runs/fg_week_2022-05-02_2022-05-08'))
    p.add_argument('--output',type=Path,default=Path('files/output/bm_1/runs/fg_week_joint_2022-05-02_2022-05-08'))
    p.add_argument('--start',default='2022-05-02');p.add_argument('--end',default='2022-05-08')
    p.add_argument('--workers',type=int,default=8);p.add_argument('--mode',choices=('daily','mpc','both'),default='both')
    p.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    p.add_argument('--hours',nargs='+',type=int,choices=range(24),default=list(range(24)))
    p.add_argument('--joint-max-nfev',type=int,default=80)
    p.add_argument('--joint-max-seconds',type=float,default=20.)
    p.add_argument('--joint-gtol',type=float,default=1e-5)
    p.add_argument('--_job',nargs=4,help=argparse.SUPPRESS)
    args=p.parse_args();repo=args.repo.resolve();out=args.output.resolve();reference=args.reference.resolve()
    if not 1<=args.workers<=32:p.error('--workers skal være 1–32.')
    if args.joint_max_nfev<1 or not math.isfinite(args.joint_max_seconds) or args.joint_max_seconds<=0 or not math.isfinite(args.joint_gtol) or args.joint_gtol<=0:
        p.error('Efteroptimeringens budgetter og tolerance skal være positive.')
    commit=source_context(repo);modules=load_modules(repo);pd=modules['pd']
    if args._job:return worker(repo,out,(args._job[0],args._job[1],args._job[2],int(args._job[3])),modules)
    if out==reference or reference in out.parents or out in reference.parents:
        p.error('Vælg en selvstændig outputmappe ved siden af referencemappen.')
    original=json.loads((reference/'settings.json').read_text(encoding='utf-8'))
    if original.get('schema')!='bm1-response-week-v1':p.error('Reference skal være fra den oprindelige f/g-ugerunner.')
    days=[str(d.date()) for d in pd.date_range(args.start,args.end,freq='D')]
    if not days or not set(days)<=set(original['days']):p.error('Datoerne skal findes i referencekørslen.')
    families=list(dict.fromkeys(args.families));hours=sorted(set(args.hours))
    if not set(families)<=set(original['families']):p.error('Familierne skal findes i referencekørslen.')
    for name,digest in original['inputs_sha256'].items():
        if sha(repo/'files/input/data'/name)!=digest:raise ValueError(f'Inputfil ændret siden referencekørslen: {name}')
    tasks=[]
    for day in days:
        if args.mode in ('daily','both'):tasks.extend((day,f,'daily',-1) for f in families)
        if args.mode in ('mpc','both'):tasks.extend((day,f,'mpc',h) for h in hours for f in families)
    needed=set(tasks)
    for day,family,kind,hour in tasks:
        if family in ('piecewise-linear','bernstein'):
            linear=(day,'linear',kind,hour)
            if not completed(reference,linear,original['fingerprint']):raise ValueError(f'Mangler lineært referencefit: {linear}')
            needed.add(linear)
    manifest={'settings.json':sha(reference/'settings.json')}
    for day in days:manifest[f'prepared/{day}.pkl']=sha(reference/'prepared'/f'{day}.pkl')
    missing_reference=[]
    for task in sorted(needed):
        if completed(reference,task,original['fingerprint']):
            folder=job_folder(reference,*task)
            for name in ('fit.json','hourly.csv' if task[2]=='daily' else 'forecasts.csv'):
                path=folder/name;manifest[path.relative_to(reference).as_posix()]=sha(path)
        elif task in tasks:missing_reference.append(task)
    core={key:original[key] for key in ('tag','daily_max_nfev','daily_max_iterations','mpc_max_nfev','mpc_max_iterations','numeric_tolerance','inputs_sha256')}
    code={path.relative_to(repo).as_posix():sha(path) for path in sorted((repo/'src').rglob('*.py'))}
    recovery={**core,'schema':'bm1-reference-completion-v1','method':'alternating',
        'days':days,'families':families,'mode':args.mode,'hours':hours,'source_sha256':code,
        'prepared_root':str(out/'prepared'),'reference_fingerprint':original['fingerprint']}
    recovery['fingerprint']=hashlib.sha256(json.dumps(recovery,sort_keys=True).encode()).hexdigest()
    config={**core,'schema':'bm1-joint-week-v1','method':'joint','days':days,'families':families,'hours':hours,
        'mode':args.mode,'reference':str(reference),'reference_fingerprint':original['fingerprint'],
        'recovery_fingerprint':recovery['fingerprint'],'reference_manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        'source_sha256':code,'joint_max_nfev':args.joint_max_nfev,'joint_max_seconds':args.joint_max_seconds,
        'joint_gtol':args.joint_gtol,'pieces':5,'BLAS_threads':1}
    config['fingerprint']=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    config['source_commit']=commit
    out.mkdir(parents=True,exist_ok=True);settings_file=out/'settings.json'
    if settings_file.exists():
        old=json.loads(settings_file.read_text(encoding='utf-8'))
        if old['fingerprint']!=config['fingerprint']:raise ValueError('Indstillinger, kode eller referencedata er ændret. Vælg en ny outputmappe.')
    else:write_json(settings_file,config)
    write_json(out/'reference_manifest.json',manifest)
    write_json(out/'reference_completed'/'settings.json',recovery)
    (out/'prepared').mkdir(exist_ok=True)
    for day in days:
        path=out/'prepared'/f'{day}.pkl'
        if path.exists():
            if sha(path)!=manifest[f'prepared/{day}.pkl']:raise ValueError(f'Forberedte data er ændret: {day}')
        else:shutil.copyfile(reference/'prepared'/f'{day}.pkl',path)
    already=sum(completed(out,t,config['fingerprint']) for t in tasks)
    pending=[t for t in tasks if not completed(out,t,config['fingerprint'])]
    print(f'{len(tasks)} jobs; {already} gemt; {len(pending)} tilbage; workers={args.workers}; BLAS=1.',flush=True)
    print(f'Færdiggør også {len(missing_reference)} manglende referencefits. Joint-budget: {args.joint_max_nfev} nfev / {args.joint_max_seconds:g} s pr. fit.',flush=True)
    active=[];errors=[];finished=already;started=time.perf_counter();logs=out/'logs';logs.mkdir(exist_ok=True)
    try:
        while pending or active:
            while pending and len(active)<args.workers:
                task=pending.pop(0);day,family,kind,hour=task
                log=(logs/f'{day}_{family}_{kind}_{hour:02d}.log').open('w',encoding='utf-8')
                proc=subprocess.Popen([sys.executable,'-B','-m','src.exe.run_model.bm_1.compare_fg_week',
                    '--repo',str(repo),'--output',str(out),'--_job',day,family,kind,str(hour)],
                    cwd=repo,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active.append((proc,task,log))
            for item in list(active):
                proc,task,log=item
                if proc.poll() is None:continue
                log.close();active.remove(item);finished+=1
                if proc.returncode or not completed(out,task,config['fingerprint']):errors.append(task);print('FEJL',task,flush=True)
                elif task[2]=='daily':
                    result=json.loads((job_folder(out,*task)/'fit.json').read_text(encoding='utf-8'))
                    print('DAG',task[0],task[1],'RMSE',round(result['rmse'],6),'joint optimalitet',result['converged'],flush=True)
                if finished%100==0:
                    summary(out,config,modules);compare(out,config,job_folder,completed,reference_folder)
                if finished%10==0 or not (pending or active):
                    print(f'Status {finished}/{len(tasks)}, fejl={len(errors)}, tid={round((time.perf_counter()-started)/60,1)} min.',flush=True)
            time.sleep(.3)
    except KeyboardInterrupt:
        print('Stopper workers; færdige checkpoints bevares.',flush=True);return 130
    finally:
        for proc,task,log in active:
            if proc.poll() is None:proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill()
            log.close()
        summary(out,config,modules);compare(out,config,job_folder,completed,reference_folder,plot=True)
        runtime_path=out/'runtime.json'
        runtime=json.loads(runtime_path.read_text(encoding='utf-8')) if runtime_path.exists() else {'sessions':[]}
        runtime['sessions'].append({'workers':args.workers,'wall_seconds':time.perf_counter()-started,
            'completed_at_start':already,'completed_at_end':sum(completed(out,t,config['fingerprint']) for t in tasks),
            'failed_tasks':[list(t) for t in errors]})
        write_json(runtime_path,runtime)
    print('Færdig. Sammenligning:',out/'comparison.csv',flush=True)
    return 1 if errors else 0


if __name__=='__main__':sys.exit(main())
