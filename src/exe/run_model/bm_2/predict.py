"""BM2 daily mixtures and hourly MPC using BM1's historical information set."""
import numpy as np
import pandas as pd
from src.model.bm_2 import bm_2
from src.params.bm_2 import lsq_params,historical_diffusion_params
from src.exe.run_model.bm_1.predict_daily import held_signal,matching_days,prepare_days
from src.validation.exe.run_model.bm_1.predict_mpc import bm_1_mpc_validation as checks
from src.tools.forecast_metrics import predictive_intervals
from src.tools.baseline import statistical_output


def posterior_mass(states,x,r,p,b,u,observed,observation_sd,bandwidth=.025,r_bandwidth=.05):
    """KDE mixture of the five PAIRED historical states, updated by I(origin).

    Demand observes X. The conditional distribution of R is updated through
    its dependence on X in the joint prior; pairs are never cross-combined.
    """
    states=np.asarray(states,dtype=float)
    if states.shape!=(5,2) or not np.all(np.isfinite(states)):
        raise ValueError('Expected five finite BM2 state pairs.')
    if min(observation_sd,bandwidth,r_bandwidth)<=0 or not np.isfinite(observed+observation_sd+bandwidth+r_bandwidth):
        raise ValueError('Invalid BM2 posterior controls.')
    sx=max(bandwidth,float(np.std(states[:,0]))/2)
    sr=max(r_bandwidth,float(np.std(states[:,1]))/2)
    prior=np.zeros((len(x),len(r)))
    for x0,r0 in states:
        kernel=np.exp(-.5*((x[:,None]-x0)/sx)**2-.5*((r[None,:]-r0)/sr)**2)
        prior+=kernel/kernel.sum()/5
    D=bm_2.demand(x,b,u,p)
    logp=np.log(np.maximum(prior,1e-300))-.5*((D[:,None]-observed)/observation_sd)**2
    mass=np.exp(logp-logp.max());return mass/mass.sum()


def empirical_mass(states,x,r):
    """Deposit five paired states on the nearest joint grid cells."""
    mass=np.zeros((len(x),len(r)))
    for x0,r0 in states:mass[np.argmin(abs(x-x0)),np.argmin(abs(r-r0))]+=.2
    return mass


def forecast_distribution(data,fit,initial_mass,sigma_x,*,fp_max_step=.25,substeps=2):
    """Return demand supports/masses at the supplied hourly sample times."""
    x,r=bm_2.grid(*initial_mass.shape); supports=[];weights=[]
    if sigma_x>0:
        fp=bm_2.evolve(data,fit,initial_mass,sigma_x,max_step=fp_max_step)
        for i,t in enumerate(data['Time']):
            supports.append(bm_2.demand(x,float(data['B'][i]),float(data['U'][i]),fit))
            weights.append(fp['cell_mass'][i].sum(axis=1))
        diagnostics=fp['diagnostics']
    else:
        # Preserve a zero-noise law without adding FV numerical diffusion.
        X,R=np.meshgrid(x,r,indexing='ij');X=X.ravel();R=R.ravel();mass=initial_mass.ravel()
        for i,t in enumerate(data['Time']):
            b,u=float(data['B'][i]),float(data['U'][i]);supports.append(bm_2.demand(X,b,u,fit));weights.append(mass)
            if i+1<len(data['Time']):
                X,R=bm_2.advance(X,R,float(data['Time'][i+1]-t),b,u,fit,substeps)
        diagnostics={'method':'deterministic paired-grid trajectories','maximum_mass_error':abs(float(mass.sum())-1),'minimum_mass':float(mass.min())}
    return supports,weights,diagnostics


def forecast_daily(data,fit,*,substeps=2):
    """Same daily convention as BM1: five equal-weight deterministic paths."""
    paths=np.array([bm_2.predict(data,s,fit,substeps)[1][:24] for s in fit['initial_states']])
    supports=list(paths.T);weights=[np.full(5,.2) for _ in supports]
    return {'forecast':paths.mean(axis=0),'intervals':predictive_intervals(supports,weights),'ensemble':paths}


def predict_day(day,*,frame,family='linear',**fit_options):
    day=checks.origin(pd.Timestamp(day).normalize())
    training,_=prepare_days(frame,matching_days(day))
    baseline=statistical_output().fit(frame,day)
    target=checks.window(frame,day,baseline,hours=24,observations=False)
    model=bm_2(held_signal(training[0]['B']),lambda t:.5,(0.,24.),25)
    fit=lsq_params(model,family).fit_many(training,**fit_options)
    return forecast_daily(target,fit),{'fit':fit,'training':training,'future':target}


def predict_origin(frame,origin,output_model,*,family='linear',horizon=8,x_steps=101,r_steps=41,
                   fp_max_step=.25,prior_bandwidth=.025,r_prior_bandwidth=.05,substeps=2,
                   max_iterations=3,max_nfev=120,state_maxiter=60,seed_fit=None,fitted=None,
                   joint_max_nfev=0,joint_max_seconds=20.,joint_gtol=1e-5):
    origin=checks.origin(origin)
    checks.options(horizon,x_steps,fp_max_step,prior_bandwidth,substeps,5)
    x,r=bm_2.grid(x_steps,r_steps)
    training_origins=[origin-pd.Timedelta(weeks=k) for k in range(5,0,-1)]
    training=[checks.window(frame,t,output_model,hours=24) for t in training_origins]
    if fitted is None:
        model=bm_2(held_signal(training[0]['B']),lambda t:.5,(0.,24.),25)
        fitted=lsq_params(model,family).fit_many(training,max_iterations=max_iterations,
            max_nfev=max_nfev,state_maxiter=state_maxiter,substeps=substeps,
            fit_bounds={'C':(.05,100.)},seed_fit=seed_fit,joint_max_nfev=joint_max_nfev,
            joint_max_seconds=joint_max_seconds,joint_gtol=joint_gtol)
    diffusion=historical_diffusion_params().fit_many(training,fitted,substeps=substeps)
    future=checks.window(frame,origin,output_model,hours=horizon+1,observations=False)
    # No target-window demand is supplied to forecasting; only I(origin) enters Bayes.
    observed=checks.observation(frame,origin)
    mass=posterior_mass(fitted['initial_states'],x,r,fitted,float(future['B'][0]),
        float(future['U'][0]),observed,diffusion['observation_sd'],prior_bandwidth,r_prior_bandwidth)
    # The extra input sample at horizon+1 is unnecessary for the forecast itself.
    future={k:np.asarray(v)[:horizon+1] for k,v in future.items() if k in ('Time','B','U')}
    support,weights,fp=forecast_distribution(future,fitted,mass,diffusion['sigma_x'],fp_max_step=fp_max_step,substeps=substeps)
    intervals=predictive_intervals(support[1:],weights[1:]);rows=[]
    for lead in range(1,horizon+1):
        row={'origin':origin,'target_time':origin+pd.Timedelta(hours=lead),'lead_hours':lead,
             'I_BM2':float(np.dot(support[lead],weights[lead])),'B':float(future['B'][lead]),
             'sigma_x':diffusion['sigma_x'],'observation_sd':diffusion['observation_sd'],
             'fit_converged':bool(fitted['converged']),'fit_stop_reason':fitted['stop_reason']}
        for level,(lo,hi) in intervals.items():row[f'lower_{level}'],row[f'upper_{level}']=float(lo[lead-1]),float(hi[lead-1])
        rows.append(row)
    return rows,{'fit':fitted,'diffusion':diffusion,'posterior':mass,'fp':fp,
        'training':training,'training_origins':training_origins,'future':future,
        'origin_observed_I':observed,'prior_bandwidth':prior_bandwidth,'r_prior_bandwidth':r_prior_bandwidth}


def refine_forecast(diag,*,x_steps=201,r_steps=81,fp_max_step=.125,substeps=32):
    """Refine both state dimensions and time using the same fitted model/data."""
    fit=diag['fit'];future=diag['future'];diffusion=diag['diffusion']
    x,r=bm_2.grid(x_steps,r_steps)
    mass=posterior_mass(fit['initial_states'],x,r,fit,float(future['B'][0]),float(future['U'][0]),
        diag['origin_observed_I'],diffusion['observation_sd'],diag['prior_bandwidth'],diag['r_prior_bandwidth'])
    support,weights,fp=forecast_distribution(future,fit,mass,diffusion['sigma_x'],fp_max_step=fp_max_step,substeps=substeps)
    return np.array([float(s@w) for s,w in zip(support[1:],weights[1:])]),fp
