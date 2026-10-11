"""Paired scoring and runtime accounting for a saved BM1 refinement run."""
import json
import numpy as np
import pandas as pd


def save_csv(path,rows):
    temporary=path.with_name(path.name+'.tmp')
    pd.DataFrame(rows).to_csv(temporary,index=False)
    try:temporary.replace(path)
    except PermissionError:temporary.replace(path.with_name(path.name+'.pending.csv'))


def compare(out,settings,job_folder,completed,reference_folder,*,plot=False):
    records=[];tables=[]
    for day in settings['days']:
        for family in settings['families']:
            tasks=[]
            if settings['mode'] in ('daily','both'):tasks.append((day,family,'daily',-1))
            if settings['mode'] in ('mpc','both'):tasks.extend((day,family,'mpc',h) for h in settings.get('hours',range(24)))
            for task in tasks:
                if not completed(out,task,settings['fingerprint']):continue
                old_folder=reference_folder(out,task,settings)
                if old_folder is None:continue
                new_folder=job_folder(out,*task)
                old=json.loads((old_folder/'fit.json').read_text(encoding='utf-8'))
                new=json.loads((new_folder/'fit.json').read_text(encoding='utf-8'))
                report=new['joint_fit'];mode=task[2]
                csv='hourly.csv' if mode=='daily' else 'forecasts.csv'
                a=pd.read_csv(old_folder/csv);b=pd.read_csv(new_folder/csv)
                keys=['time'] if mode=='daily' else ['origin','target_time','lead_hours']
                if len(a)!=len(b) or not a[keys].equals(b[keys]):raise ValueError(f'Forecast grids differ: {task}')
                if not np.allclose(a.I_observed,b.I_observed,rtol=0,atol=1e-12):raise ValueError(f'Observed values differ: {task}')
                pred='I_predicted' if mode=='daily' else 'I_BM1'
                t=pd.DataFrame({'mode':mode,'family':family,'old':a[pred],'new':b[pred],
                    'observed':a.I_observed,'baseline':a.B,
                    'old_original':a[pred+'_original'],'new_original':b[pred+'_original'],
                    'observed_original':a.I_observed_original,'baseline_original':a.B_original})
                if mode=='mpc':t['lead_hours']=a.lead_hours
                tables.append(t)
                records.append({'day':day,'family':family,'mode':mode,'hour':task[3],
                    'old_training_rmse':old['training_rmse'],
                    'old_training_rmse_recomputed':float(np.sqrt(report['reference_sse']/120)),
                    'new_training_rmse':new['training_rmse'],
                    'training_nonworse':report['final_sse']<=report['reference_sse'],
                    'old_rmse':old['rmse'],'new_rmse':new['rmse'],
                    'old_stopping_converged':old['converged'],
                    'new_first_order_passed':report['first_order_passed'],
                    'new_optimizer_success':report['optimizer_success'],
                    'joint_stop_reason':report['stop_reason'],'joint_optimality':report['optimality'],
                    'seed_source':report['seed_source'],
                    'old_numeric_passed':old['numerically_passed'],'new_numeric_passed':new['numerically_passed'],
                    'old_worker_seconds':old['seconds'],'joint_seconds':report['seconds'],
                    'new_worker_seconds':new['seconds']})
    save_csv(out/'fit_comparison.csv',records)
    if not tables:return
    values=pd.concat(tables,ignore_index=True);fits=pd.DataFrame(records);summary=[];leads=[]
    def metrics(t):
        rmse=lambda p,y:float(np.sqrt(np.mean((p-y)**2)))
        old,new=rmse(t.old,t.observed),rmse(t.new,t.observed)
        return {'n':len(t),'old_rmse':old,'new_rmse':new,'rmse_change':new-old,
            'improvement_pct':100*(1-new/old) if old else None,
            'baseline_rmse':rmse(t.baseline,t.observed),
            'old_rmse_original':rmse(t.old_original,t.observed_original),
            'new_rmse_original':rmse(t.new_original,t.observed_original),
            'baseline_rmse_original':rmse(t.baseline_original,t.observed_original)}
    for (mode,family),t in values.groupby(['mode','family'],sort=False):
        selected=fits[(fits['mode']==mode)&(fits.family==family)]
        old_seconds=float(selected.old_worker_seconds.sum());joint_seconds=float(selected.joint_seconds.sum())
        summary.append({'mode':mode,'family':family,**metrics(t),'paired_jobs':len(selected),
            'expected_jobs':len(settings['days'])*(1 if mode=='daily' else len(settings.get('hours',range(24)))),
            'training_nonworse_jobs':int(selected.training_nonworse.sum()),
            'old_stopping_converged_jobs':int(selected.old_stopping_converged.sum()),
            'new_first_order_passed_jobs':int(selected.new_first_order_passed.sum()),
            'new_optimizer_success_jobs':int(selected.new_optimizer_success.sum()),
            'old_numeric_passed_jobs':int(selected.old_numeric_passed.sum()),
            'new_numeric_passed_jobs':int(selected.new_numeric_passed.sum()),
            'old_worker_seconds_sum':old_seconds,'joint_seconds_sum':joint_seconds,
            'continuation_worker_seconds_sum':float(selected.new_worker_seconds.sum()),
            'estimated_full_method_worker_ratio':1+joint_seconds/old_seconds if old_seconds else None})
        if mode=='mpc':
            for lead,part in t.groupby('lead_hours'):leads.append({'family':family,'lead_hours':int(lead),**metrics(part)})
    save_csv(out/'comparison.csv',summary);save_csv(out/'comparison_by_lead.csv',leads)
    if plot:
        import matplotlib.pyplot as plt
        tab=pd.DataFrame(summary);modes=list(tab['mode'].unique())
        fig,axes=plt.subplots(1,len(modes),figsize=(7*len(modes),4.8),squeeze=False)
        for ax,mode in zip(axes[0],modes):
            part=tab[tab['mode']==mode];x=np.arange(len(part))
            ax.bar(x-.19,part.old_rmse,.38,label='Alternating fit')
            ax.bar(x+.19,part.new_rmse,.38,label='Joint refinement')
            ax.plot(x,part.baseline_rmse,'k_',markersize=14,label='Baseline')
            ax.set_xticks(x,part.family,rotation=25,ha='right');ax.set_ylabel('Normalized RMSE')
            ax.set_title('Daily forecast' if mode=='daily' else 'MPC, leads 1–8 h')
            ax.legend();ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
