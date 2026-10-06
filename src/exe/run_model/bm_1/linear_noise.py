"""G6a bounded saved-G4 pilot: ISO2022 weeks 13 and 18, constant sigma only.

Reuses selected unpenalized S3 fits. Historical demand is used before Monday;
forecast inputs have no observations. Targets are scored only after export.
Already exposed G4 weeks: development diagnostics, not independent validation.
"""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.exe.run_model.bm_1.deterministic_weekday import save_json
from src.exe.run_model.bm_1.deterministic_strategies import ORIGIN, FIRST_TEST, input_fingerprint
from src.exe.run_model.bm_1.deterministic_validation import week_job
from src.exe.run_model.wm_4.output_model import statistical_output
from src.method.bm_1_linear_noise import unit_variance, fit_constant_sigma, interval_metrics
from src.params.bm_1_response import BM1PiecewiseResponse
from src.tools.bm_1_deterministic_data import prepare_frame

WEEKS = (13,18)


def empirical_profiles(histories, origin):
    """Five prior weekdays, measured samples only, ddof=1 as WM4.

    Match local clock hour rather than elapsed array slot across spring DST.
    Missing clock-hour samples remain absent; >=2 observations define a bin.
    """
    result = np.full((7,24),np.nan); counts=np.zeros((7,24),int)
    for weekday in range(7):
        rows=np.full((5,24),np.nan)
        for lag in range(1,6):
            inp=histories[weekday*5+lag-1]
            day=origin+pd.Timedelta(days=weekday)-pd.Timedelta(weeks=lag)
            physical=pd.date_range(day.tz_localize('Europe/Copenhagen'),
                (day+pd.Timedelta(days=1)).tz_localize('Europe/Copenhagen'),freq='h',inclusive='left')
            values=inp.observations[24:]
            if len(physical)!=len(values):raise ValueError('Historical physical-hour mismatch.')
            rows[lag-1,physical.hour]=values
        for hour in range(24):
            observed=rows[:,hour];observed=observed[np.isfinite(observed)]
            counts[weekday,hour]=len(observed)
            if len(observed)>=2:result[weekday,hour]=np.std(observed,ddof=1)
    return result,counts


def run(raw, saved_g4, output):
    started=perf_counter();deadline=started+300
    dest=Path(output);dest.mkdir(parents=True,exist_ok=False)
    saved=Path(saved_g4)
    prior=json.loads((saved/'protocol.json').read_text())
    if prior['raw_fingerprint']!=input_fingerprint(raw):raise ValueError('Use the exact saved G4 input frame.')
    frame,scales=prepare_frame(raw,ORIGIN,FIRST_TEST,price_bounds=(-500.,6000.))
    if scales!=prior['scales']:raise ValueError('G4 normalization must be identical.')
    source=Path(__file__).resolve().parents[3]/'method/bm_1_linear_noise.py'
    protocol={'version':1,'weeks':list(WEEKS),'family':'constant sigma >= 0; no further family search',
        'raw_fingerprint':input_fingerprint(raw),'scales':scales,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),source)},
        'g4_protocol_sha256':hashlib.sha256((saved/'protocol.json').read_bytes()).hexdigest(),
        'g4_fit_sha256':{str(w):hashlib.sha256((saved/'s3'/f'week_{w}'/'result.json').read_bytes()).hexdigest() for w in WEEKS},
        'calibration':'one shared sigma per Monday; LS to measured same-weekday/hour std, ddof=1; exclude hour0; no targets',
        'profile':'sqrt(mean five conditional unit variances); no between-input/mean variance added',
        'initialization':'known conditional x, P0=0 at each target midnight after 24 h physical state warmup',
        'reference':'empirical training std bands about identical G4 forecast center; not WM4 stochastic bands',
        'numerics':'saved G4 grid and twice its resolution; sigma/profile std max error <=1e-4 normalized; center <=1e-5',
        'pilot_limits':{'seconds':60,'propagations':30},'whole_limits':{'seconds':300,'propagations':500},
        'scope':'conditional linear-noise point approximation; G4-exposed weeks; no independent calibration/coverage acceptance',
        'exclusions':['parameter','baseline','future inputs','initial state','separate measurement variance']}
    save_json(dest/'protocol.json',protocol)
    counts=0;pilot_start=perf_counter();pilot_finished=False;weeks=[];forecasts=[]
    def propagate(inp,p,g,reset):
        nonlocal counts,pilot_finished
        if counts>=500 or perf_counter()>=deadline:raise RuntimeError('G6a study budget stop, incomplete evidence.')
        if not pilot_finished and (counts>=30 or perf_counter()-pilot_start>=60):
            raise RuntimeError('G6a runtime pilot budget stop.')
        result=unit_variance(inp,p,price_response=g,reset_hour=reset,deadline=deadline)
        counts+=1
        if not pilot_finished and counts==5:
            save_json(dest/'runtime_pilot.json',{'status':'complete','propagations':counts,'seconds':perf_counter()-pilot_start,
                'note':'These five histories are retained in the study, no repeated pilot.'})
            pilot_finished=True
        return result
    for week in WEEKS:
        origin=ORIGIN+pd.Timedelta(weeks=week-1)
        fit=json.loads((saved/'s3'/f'week_{week}'/'result.json').read_text())
        if fit['status']!='complete':raise ValueError('Completed G4 fit required.')
        chosen=next(c for c in fit['candidates'] if c['lambda']==0)
        p=chosen['best']['parameters'];g=BM1PiecewiseResponse.from_inner_values([chosen['best']['g_mid']],pieces=2)
        baseline=statistical_output().fit(frame.loc[frame.index<origin,['I','Temp']],origin)
        job,_=week_job(frame,baseline,origin,p,dest/f'week_{week}')
        history=job['histories'];future=job['forecasts']
        while history[0].steps<fit['steps']:
            history=[i.refined() for i in history];future=[(i.refined(),idx) for i,idx in future]
        empirical,n=empirical_profiles(history,origin)
        np.savez(dest/f'training_dispersion_week_{week}.npz',std=empirical,counts=n)
        grid_data=[]
        for grid in range(2):
            hs=[propagate(i,p,g,24) for i in history]
            profiles=np.full((7,24),np.nan)
            for weekday,pos in enumerate(job['state_positions']):
                variance_rows=np.full((5,24),np.nan)
                for row,position in enumerate(pos):
                    day=origin+pd.Timedelta(days=weekday)-pd.Timedelta(weeks=row+1)
                    idx=pd.date_range(day.tz_localize('Europe/Copenhagen'),(day+pd.Timedelta(days=1)).tz_localize('Europe/Copenhagen'),freq='h',inclusive='left')
                    variance_rows[row,idx.hour]=hs[position]['unit_demand_variance'][24:]
                profiles[weekday]=np.sqrt(np.nanmean(variance_rows,axis=0))
            calibrated=fit_constant_sigma(profiles[:,1:].reshape(-1),empirical[:,1:].reshape(-1))
            if calibrated['status']!='complete':raise RuntimeError('Noise amplitude not identifiable for this fixed mean model.')
            sigma=calibrated['sigma_per_sqrt_hour'];day_outputs=[]
            for weekday,((inp,index),pos) in enumerate(zip(future,job['state_positions'])):
                params=list(p);params[3]=float(np.median([hs[j]['state'][24] for j in pos]))
                out=propagate(inp,params,g,0)
                sd=sigma*out['unit_demand_std']
                day_outputs.append(pd.DataFrame({'center':out['demand'],'process_std':sd,
                    'empirical_std':empirical[weekday,index.hour],'local_hour':index.hour,
                    'horizon_hours':np.arange(len(index)),'state_variance':sigma**2*out['unit_state_variance'][:-1],
                    'state':out['state'][:-1],'weekday':weekday},index=index))
            grid_data.append({'fit':calibrated,'profile':sigma*profiles,'days':pd.concat(day_outputs),'histories':hs})
            if grid==0:
                history=[i.refined() for i in history];future=[(i.refined(),idx) for i,idx in future]
        coarse,fine=grid_data
        center_error=float(np.max(abs(coarse['days'].center-fine['days'].center)))
        std_error=float(np.max(abs(coarse['days'].process_std-fine['days'].process_std)))
        profile_error=float(np.nanmax(abs(coarse['profile']-fine['profile'])))
        passing=center_error<=1e-5 and max(std_error,profile_error)<=1e-4
        frozen=coarse['days'];previous=pd.read_csv(saved/'s3'/f'week_{week}'/'forecast_without_penalty.csv',index_col=0,parse_dates=True).prediction
        replay_error=float(np.max(abs(frozen.center-previous)))
        if replay_error>1e-12:raise RuntimeError('The G4 forecast center changed.')
        frozen.to_csv(dest/f'forecast_normalized_week_{week}.csv',index_label='Datetime')
        normalized_state_sd=np.sqrt(frozen.state_variance.to_numpy())
        state_sd_boundary_fraction=float(np.mean(1.959963984540054*normalized_state_sd>
            np.minimum(frozen.state.to_numpy(),1-frozen.state.to_numpy())))
        record={'week':week,'origin':str(origin),'steps':fit['steps'],'noise_fit':coarse['fit'],
            'numerics':{'pass':passing,'center_error':center_error,'std_error':std_error,'training_profile_error':profile_error,'g4_center_replay_error':replay_error},
            'training_bins':int(np.isfinite(empirical[:,1:]).sum()),'training_bin_counts_min':int(n[:,1:].min()),
            'state_95_band_outside_domain_fraction':state_sd_boundary_fraction,
            'max_state_std':float(normalized_state_sd.max()),'max_state_variance':float(frozen.state_variance.max()),
            'state_variance_bound_violation_fraction':float(np.mean(frozen.state_variance>.25)),
            'response_sign_changes_histories':sum(h['response_sign_changes'] for h in coarse['histories'])}
        save_json(dest/f'week_{week}_diagnostics.json',record)
        if not passing:raise RuntimeError('G6a temporal refinement failed; retain diagnostic files, no automatic retuning.')
        weeks.append(record);forecasts.append(frozen)
        print(f'Week {week}: sigma={coarse["fit"]["sigma_per_sqrt_hour"]:.6g}, std-grid error={std_error:.3g}; {counts}/500 propagations; {perf_counter()-started:.1f}s elapsed',flush=True)
    # Both sets of parameters and predictions are already exported before targets.
    prediction=pd.concat(forecasts).sort_index();scored=prediction.copy()
    for c in ('center',):scored[c]=scored[c]*scales['demand_scale']+scales['demand_min']
    for c in ('process_std','empirical_std'):scored[c]=scored[c]*scales['demand_scale']
    scored['observed']=raw.reindex(scored.index).Demand.to_numpy()
    z=norm.ppf(.975)
    for name,col in (('process','process_std'),('empirical','empirical_std')):
        scored[name+'_lower']=scored.center-z*scored[col];scored[name+'_upper']=scored.center+z*scored[col]
    scored.to_csv(dest/'scored.csv',index_label='Datetime')
    def scores(block):
        return {name:{scope:interval_metrics(b.observed,b.center,b[col]) for scope,b in
            (('all',block),('exclude_daily_origin',block.loc[block.horizon_hours>0]))} for name,col in
            (('process','process_std'),('empirical','empirical_std'))}
    summary={'status':'complete','protocol':protocol,'weeks':weeks,'scores':scores(scored),
        'weekly_scores':{str(w):scores(scored.loc[scored.index.isocalendar().week==w]) for w in WEEKS},
        'by_horizon':{str(h):{name:interval_metrics(b.observed,b.center,b[col]) for name,col in
            (('process','process_std'),('empirical','empirical_std'))} for h,b in scored.groupby('horizon_hours')},
        'propagations':counts,'seconds':perf_counter()-started,
        'interpretation':'Numerical reference complete only; inspect dispersion, coverage and boundary validity before scientific acceptance.'}
    save_json(dest/'summary.json',summary)
    return summary


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--raw-csv',required=True);parser.add_argument('--saved-g4',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(pd.read_csv(args.raw_csv,index_col=0,parse_dates=True),args.saved_g4,args.output)

if __name__=='__main__':main()
