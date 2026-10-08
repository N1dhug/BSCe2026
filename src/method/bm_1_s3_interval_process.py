"""Hourly integrated process moments, including within-hour time covariance.

No observation noise or Gaussian/point-marginal bands are substituted. The
three moment fields share the existing conservative FV Markov transition.
"""
from copy import deepcopy
import hashlib
from time import perf_counter

import numpy as np
import pandas as pd

from src.method import bm_1_s3_interval as interval
from src.method.bm_1_s3_uncertainty import demand_values
from src.method.bm_1_s3_uncertainty import fit_variance_amplitude
from src.method.bm_1_linear_noise import demand_state_derivative
from src.model.bm_1 import bm_1
from src.params.bm_1_response import BM1PiecewiseResponse
from src.control.bm_1 import point_initial_cell_mass
from src.solver.finite_volume import finite_volume_1d_cached
from src.validation.solver.finite_volume import finite_volume_1d_validation
from src.tools.bm_1_deterministic_snapshot import restore_baseline
from src.exe.run_model.bm_1.deterministic_weekday import metrics

GATES={'process_mean':.005,'process_variance':.002,'state_mean':.005,'state_variance':.002}
SOURCES=(*interval.SOURCES,'src/method/bm_1_s3_interval_process.py',
    'src/method/bm_1_s3_uncertainty.py','src/method/bm_1_linear_noise.py','src/model/bm_1.py','src/model/base/bm.py',
    'src/control/bm_1.py','src/solver/finite_volume.py','src/validation/solver/finite_volume.py',
    'src/validation/model/bm.py','src/validation/model/bm_1.py')


def source_hashes():
    return {p:hashlib.sha256((interval.ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in SOURCES}


def reward_step(fields,half_reward,advance,time,step):
    """Positive transition applied to mass and both additive-reward moments.

    R += half_reward(x) before and after the Markov transition. r=E[R 1_X]
    and s=E[R^2 1_X]; r may be signed. Old r is used in each second moment
    shift. Mixing all three fields retains the X/R dependence and time covariance.
    """
    out=fields.copy();a=np.asarray(half_reward,float)
    out[:,2]+=2*a*out[:,1]+a*a*out[:,0]
    out[:,1]+=a*out[:,0]
    out,diagnostics=advance(out,time,step)
    out[:,2]+=2*a*out[:,1]+a*a*out[:,0]
    out[:,1]+=a*out[:,0]
    return out,diagnostics


def interval_day_moments(inp,parameters,curve,start,sigma,cells,step,deadline):
    if inp.contract!='hourly-mean':raise ValueError('Integrated process moments require hourly-mean inputs.')
    solver=finite_volume_1d_cached();edges,x,dx=solver.grid(cells)
    def B(t):return np.interp(t,inp.time,inp.baseline)
    def U(t):return inp.prices[min(int(np.floor(t)),inp.n_hours-1)]
    model=bm_1(B,B,(0.,float(inp.n_hours)),inp.n_hours+1)
    params=[U,BM1PiecewiseResponse.linear(pieces=2),curve,*parameters[:3]]
    advance=solver._stepper(dx,edges,x,lambda t,z:model._drift(t,z,params),
        lambda t,z:model.diffusion_coefficient(z,sigma),lambda t:(float(B(t)),float(U(t))))
    mass=point_initial_cell_mass(edges,start)
    means=[];variances=[];state_mean=[float(mass@x)]
    state_variance=[float(np.sum(mass*(x-state_mean[-1])**2))]
    max_mass_error=0.;max_column_error=0.;minimum_mass=0.;steps=0
    for hour in range(inp.n_hours):
        # The process state carries over within the day. Only the CURRENT
        # hourly reward resets; previous-hour observations are never used.
        fields=np.column_stack((mass,np.zeros_like(mass),np.zeros_like(mass)))
        grid,_=solver.build_time_grid([float(hour),float(hour+1)],max_step=step)
        for left,right in zip(grid[:-1],grid[1:]):
            if perf_counter()>=deadline:raise RuntimeError('G interval-process runtime budget stop.')
            h=float(right-left);mid=float((left+right)/2)
            # Integrate flexibility; the deterministic baseline integral is
            # added exactly once per hour. This reduces moment cancellation.
            reward=demand_values(x,B(mid),curve(U(mid)),parameters[1],parameters[2])-B(mid)
            fields,diagnostics=reward_step(fields,.5*h*reward,advance,mid,h)
            if not np.isfinite(fields).all():raise RuntimeError('Nonfinite interval moment field; no repair.')
            finite_volume_1d_validation.evolved_mass(fields[:,0])
            max_mass_error=max(max_mass_error,abs(float(fields[:,0].sum())-1.))
            max_column_error=max(max_column_error,diagnostics['maximum_abs_column_sum'])
            minimum_mass=min(minimum_mass,float(fields[:,0].min()));steps+=1
        mass=fields[:,0].copy();flex=float(fields[:,1].sum())
        variance=float(fields[:,2].sum()-flex*flex)
        mean=float(inp.b_mean[hour]+flex)
        if not np.isfinite([mean,variance]).all() or variance<0 or not -1e-10<=mean<=1+1e-10:
            raise RuntimeError('Invalid integrated demand moments; no variance clipping.')
        means.append(mean);variances.append(variance)
        state_mean.append(float(mass@x));state_variance.append(float(np.sum(mass*(x-state_mean[-1])**2)))
    return {'process_mean':np.asarray(means),'process_variance':np.asarray(variances),
        'state_mean':np.asarray(state_mean),'state_variance':np.asarray(state_variance),
        'diagnostics':{'n_steps':steps,'maximum_abs_mass_error':max_mass_error,
            'maximum_abs_generator_column_sum':max_column_error,'minimum_cell_mass':minimum_mass,
            'used_clipping':False,'used_renormalization':False,
            'initial_state_discretization_error':abs(float(state_mean[0])-start)}}


def process_interval(snapshot,raw_inputs,*,sigma_x,cells=128,max_step=1/24,seconds=120.):
    started=perf_counter();s=interval.validate_snapshot(snapshot)
    if (not np.isfinite(sigma_x) or sigma_x<0 or type(cells) is not int or not 32<=cells<=512
            or not np.isfinite(max_step) or not 0<max_step<=1/12 or not np.isfinite(seconds) or not 0<seconds<=120):
        raise ValueError('Declare sigma>=0,32..512 cells,step<=(1/12) hour and budget<=120s.')
    result=interval.predict_interval(s,raw_inputs)
    deterministic_numerics=deepcopy(result['numerics'])
    if result['status']!='complete':raise ValueError('Deterministic interval refinement failed.')
    table=result['normalized_predictions'].copy();audits=[];errors=[]
    if sigma_x==0:
        table['process_mean']=table.prediction;table['process_variance']=0.
        table['process_state_mean']=table.state;table['process_state_end_mean']=table.state_end
        table['process_state_variance']=0.;table['process_state_end_variance']=0.
        checks={'passed':True,'zero_diffusion':'exact own hourly-mean RK4 forecast'}
    else:
        origin=interval.point._origin(s['origin']);scale=s['scales']
        frame=pd.DataFrame({'U':2*(raw_inputs.Price-scale['price_min'])/scale['price_scale']-1,
            'Temp':raw_inputs.Temp},index=raw_inputs.index)
        baseline=restore_baseline(s['baseline']);curve=BM1PiecewiseResponse.from_record(s['price_response'])
        fine_results=[]
        for d in range(7):
            inp,_=interval.day_inputs(frame,baseline,origin+pd.Timedelta(days=d),steps=s['numerics']['steps_per_hour'])
            coarse=interval_day_moments(inp,s['parameters'],curve,s['forecast_states'][d],sigma_x,cells,max_step,started+seconds)
            fine=interval_day_moments(inp,s['parameters'],curve,s['fine_forecast_states'][d],sigma_x,2*cells,max_step/2,started+seconds)
            errors.append({k:float(np.max(abs(coarse[k]-fine[k]))) for k in GATES})
            audits.append({'coarse':coarse['diagnostics'],'fine':fine['diagnostics']});fine_results.append(fine)
        for key in ('process_mean','process_variance'):table[key]=np.concatenate([r[key] for r in fine_results])
        for field,column in (('state_mean','process_state'),('state_variance','process_state')):
            suffix='mean' if field=='state_mean' else 'variance'
            table[f'{column}_{suffix}']=np.concatenate([r[field][:-1] for r in fine_results])
            table[f'{column}_end_{suffix}']=np.concatenate([r[field][1:] for r in fine_results])
        checks={'passed':all(e[k]<=GATES[k] for e in errors for k in GATES),
            'gates':deepcopy(GATES),'daily_errors':errors,'cells':[cells,2*cells],
            'max_steps_hours':[max_step,max_step/2]}
    raw=table.copy();scale=s['scales']
    for col in ('prediction','baseline','point_prediction','process_mean'):
        raw[col]=scale['demand_min']+scale['demand_scale']*table[col]
    raw['process_variance']=scale['demand_scale']**2*table.process_variance
    result.update(status='complete-interval-process' if checks['passed'] else 'numerics-failed',
        predictions=raw,normalized_predictions=table,seconds=perf_counter()-started,
        numerics=checks,solver_audits=audits,
        uncertainty={'kind':'conditional-hourly-integrated-state-process-only','sigma_per_sqrt_hour':float(sigma_x),
            'sigma_calibration':'explicit scenario input; no historical calibration acceptance',
            'time_covariance':'joint state/additive reward first and second moments; not averaged marginal variances',
            'solver':'conservative FV implicit Euler, midpoint-frozen coefficients and symmetric reward shifts',
            'excluded':['measurement/model error','parameter uncertainty','baseline uncertainty','future input uncertainty'],
            'bands_status':'not-computed; interval moments do not determine exact quantiles',
            'source_sha256':source_hashes()})
    result['manifest']['deterministic_numerics']=deterministic_numerics
    result['manifest']['stochastic_status']='implemented-hourly-integrated-process-moments'
    return result


def score_interval_process(result,observations):
    out=interval.score_interval(result,observations)
    out['scoring']['metrics']['process_mean']=metrics(out['predictions'].observed.to_numpy(),out['predictions'].process_mean.to_numpy())
    out['scoring']['bands_status']='not-computed; no point-band score substituted for an hourly integral'
    return out


def covariance_step(covariance,rate,readout,noise,step):
    """Frozen linearized (X,Z) covariance, positive 3-node noise quadrature.

    dX=rate X dt+noise dW; dZ=readout X dt. Z integrates the CURRENT hour.
    The 2x2 transition is exact for frozen coefficients. The noise covariance
    integrates positive outer products, not a subtraction of exponentials.
    """
    def integral(s):return np.expm1(rate*s)/rate if rate!=0 else s
    transition=np.array([[np.exp(rate*step),0.],[readout*integral(step),1.]])
    nodes=step*np.array([.5-np.sqrt(15)/10,.5,.5+np.sqrt(15)/10])
    weights=np.array([5/18,4/9,5/18])*step
    vectors=noise*np.column_stack((np.exp(rate*nodes),readout*integral(nodes)))
    return transition@covariance@transition.T+vectors.T@(weights[:,None]*vectors)


def unit_interval_variance(inp,parameters,curve,*,reset_hour=24,deadline):
    if inp.contract!='hourly-mean':raise ValueError('Interval covariance requires hourly-mean inputs.')
    forward=inp.solve(parameters,price_response=curve,keep_fine=True,deadline=deadline)
    x=forward['fine_state'];mid=.5*(x[:-1]+x[1:])
    g=np.repeat(curve(inp.prices),inp.steps)
    c=demand_state_derivative(mid,g,inp.b_mid,parameters[1],parameters[2])
    covariance=np.zeros((2,2));variance=np.zeros(inp.n_hours);state_variance=np.zeros(inp.n_hours+1)
    for j in range(reset_hour*inp.steps,len(mid)):
        if perf_counter()>=deadline:raise RuntimeError('G interval calibration runtime budget stop.')
        if j%inp.steps==0:
            covariance=np.diag([covariance[0,0],0.])
        covariance=covariance_step(covariance,c[j]/parameters[0],c[j],mid[j]*(1-mid[j]),inp.h)
        if not np.isfinite(covariance).all() or np.any(np.diag(covariance)<0):
            raise RuntimeError('Invalid interval covariance; no clipping.')
        if (j+1)%inp.steps==0:
            hour=(j+1)//inp.steps;variance[hour-1]=covariance[1,1];state_variance[hour]=covariance[0,0]
    return {'unit_interval_variance':variance,'unit_state_variance':state_variance,'demand':forward['demand']}


def diagnose_interval(snapshot,raw_history,*,seconds=60.):
    started=perf_counter();s=interval.validate_snapshot(snapshot);origin=interval.point._origin(s['origin'])
    if not np.isfinite(seconds) or not 0<seconds<=60:raise ValueError('Diagnostic budget must be <=60s.')
    train=interval.training_frame(raw_history,origin,s['demand_timestamp_label'])
    if interval.point._fingerprint(train)!=s['training_fingerprint']:raise ValueError('Own fitted history is required.')
    scale=s['scales'];frame,_=interval.prepare_frame(train,*s['normalization_window'],
        price_bounds=(scale['price_min'],scale['price_min']+scale['price_scale']))
    baseline=restore_baseline(s['baseline']);curve=BM1PiecewiseResponse.from_record(s['price_response'])
    empirical=np.full((7,24),np.nan);counts=np.zeros((7,24),int)
    profiles=[np.full((7,24),np.nan),np.full((7,24),np.nan)]
    max_state_unit=[0.,0.];outside=0;measured=0;demand_error=0.;unit_std_error=0.
    for d,days in enumerate(s['history']['days_by_weekday']):
        observed=np.full((5,24),np.nan);units=[np.full((5,24),np.nan),np.full((5,24),np.nan)]
        for row,day in enumerate(days):
            inp,index=interval.day_inputs(frame,baseline,pd.Timestamp(day),history=True,steps=s['numerics']['steps_per_hour'])
            hours=index[24:].hour;y=inp.observations[24:];observed[row,hours]=y
            lower=demand_values(1.,inp.b_mean[24:],curve(inp.prices[24:]),s['parameters'][1],s['parameters'][2])
            upper=demand_values(0.,inp.b_mean[24:],curve(inp.prices[24:]),s['parameters'][1],s['parameters'][2])
            mask=np.isfinite(y);measured+=int(mask.sum())
            outside+=int(np.count_nonzero(mask&((y<lower)|(y>upper))))
            pair=[]
            for grid,item in enumerate((inp,inp.refined())):
                out=unit_interval_variance(item,s['parameters'],curve,deadline=started+seconds)
                pair.append(out);units[grid][row,hours]=out['unit_interval_variance'][24:]
                max_state_unit[grid]=max(max_state_unit[grid],float(out['unit_state_variance'].max()))
            demand_error=max(demand_error,float(np.max(abs(pair[0]['demand']-pair[1]['demand']))))
            unit_std_error=max(unit_std_error,float(np.max(abs(np.sqrt(pair[0]['unit_interval_variance'])-
                np.sqrt(pair[1]['unit_interval_variance'])))))
        for h in range(24):
            y=observed[:,h];y=y[np.isfinite(y)];counts[d,h]=len(y)
            if len(y)>=2:empirical[d,h]=float(np.var(y,ddof=1))
        for grid in range(2):profiles[grid][d]=np.nanmean(units[grid],axis=0)
    fits=[fit_variance_amplitude(w.ravel(),empirical.ravel()) for w in profiles]
    maxima=[None if f['q_per_hour'] is None else f['q_per_hour']*max_state_unit[j] for j,f in enumerate(fits)]
    std_error=None if any(f['q_per_hour'] is None for f in fits) else float(np.nanmax(abs(
        np.sqrt(fits[0]['q_per_hour']*profiles[0])-np.sqrt(fits[1]['q_per_hour']*profiles[1]))))
    passed=demand_error<=1e-5 and unit_std_error<=1e-4 and std_error is not None and std_error<=1e-4
    reasons=[]
    if not passed:reasons.append('Interval mean/covariance refinement failed or was unidentifiable.')
    if outside:reasons.append('Historical measured averages outside the frozen integrated-demand support; model/measurement error excluded.')
    if any(p is not None and p>.25 for p in maxima):reasons.append('Calibrated LNA state variance exceeds 1/4 on [0,1].')
    return {'status':'rejected-process-calibration' if reasons else 'candidate-needs-nonlinear-validation',
        'reasons':reasons,'observed_samples':measured,'outside_integrated_demand_support':outside,
        'coarse_fit':fits[0],'fine_fit':fits[1],'max_lna_state_variance':maxima,
        'numerics':{'passed':passed,'max_demand_error':demand_error,'max_unit_interval_std_error':unit_std_error,
            'max_fitted_profile_std_error':std_error,'demand_tolerance':1e-5,'std_tolerance':1e-4},
        'empirical_counts':counts.tolist(),'time0_policy':'included: the first full-hour integral has positive process variance despite P0=0',
        'semantics':'matching-weekday raw average-power sample variance; not isolated physical process noise',
        'used_in_process_forecast':False,'seconds':perf_counter()-started,'source_sha256':source_hashes()}
