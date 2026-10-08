"""One explicit rawcall extension of an unfinished WM solver, same wall cap.

Same equations, bounds, seeds, variable-projection objective and tolerances.
Not automatically enabled; cumulative calls/time and earlier failure retained.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import least_squares
from src.method import g_weekday_rolling as model


def fit_extension(items,previous,*,extra_calls=1200):
    if previous['status']!='unfinished' or previous['calls']!=400 or extra_calls!=1200:
        raise ValueError('Only the explicit400+1200 rawcall extension is supported.')
    remaining=10.-previous['seconds'];begun=perf_counter();calls=0;trace=[];converged=[]
    if remaining<=0:raise RuntimeError('Original WM10second wall cap is exhausted.')
    def evaluate(theta):
        r=model.wm.coefficients(theta);errors=[];initials=[]
        for item in items:
            mask=item['mask'];H,v=model.wm.design(item['price'],r);y=item['observed']-item['baseline']-v
            if mask.sum()<2:raise ValueError('Five own measured target days required.')
            initial=model.wm.project_initial(H[mask],y[mask]);initials.append(initial.tolist());errors.extend((H@initial-y)[mask])
        return np.asarray(errors),initials,r
    def objective(theta):
        nonlocal calls
        if calls>=extra_calls or perf_counter()-begun>=remaining:raise model.wm.FitBudget('Explicit cumulative WM extension stopped.')
        calls+=1;return evaluate(theta)[0]
    for seed in model.wm.SEEDS:
        try:
            result=least_squares(objective,seed,bounds=model.wm.BOUNDS,max_nfev=extra_calls,x_scale='jac',ftol=1e-8,xtol=1e-8,gtol=1e-8)
            e,initials,r=evaluate(result.x)
            candidate={'converged':bool(result.success),'stop':str(result.message),'r':r.tolist(),'mse':float(np.mean(e**2)),
                'initial_states':initials,'theta':result.x.tolist(),'active_bounds':result.active_mask.tolist(),
                'jacobian_singular_values':np.linalg.svd(result.jac,compute_uv=False).tolist()}
            trace.append(candidate)
            if result.success:converged.append(candidate)
        except model.wm.FitBudget as exc:trace.append({'converged':False,'stop':str(exc)});break
    budget={'seconds_cumulative':10.,'calls_cumulative':1600,'extra_calls_cap':1200,'single_explicit_extension':True}
    shared={'attempts':trace,'calls':previous['calls']+calls,'seconds':previous['seconds']+(perf_counter()-begun),
        'previous_failed_fit':previous,'new_calls':calls,'budget':budget,'model_or_objective_changed':False}
    if not converged:return {'status':'unfinished',**shared}
    best=min(converged,key=lambda x:x['mse'])
    return {'status':'complete','r_per_day':best['r'],'forecast_state':np.median(best['initial_states'],axis=0).tolist(),'fit':best,**shared}
