"""Bounded joint refinement of pooled BM parameters and five initial states.

The original fit remains a fallback. The reported convergence flag requires
first-order optimality of the joint, discretized least-squares problem; it is
not a global-minimum certificate.
"""
import time
import numpy as np
from scipy.optimize import least_squares


class _TimeLimit(Exception):
    pass


def refine_joint(datasets, fitted, *, predict, unpack, names, lower, upper,
                 max_nfev=80, max_seconds=20.0, gtol=1e-5, substeps=2,
                 alternative=None, alternative_name='linear_reference'):
    """Improve an existing fit without increasing its unpenalized training SSE.

`unpack` maps free response/physical parameters to the dictionary consumed by
`predict`. Each of the five starting states is an additional bounded unknown.
The wall-time limit is checked between residual evaluations (a soft limit).
"""
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev<1:
        raise ValueError('joint max_nfev must be a positive integer.')
    if not np.isfinite(max_seconds) or max_seconds<=0 or not np.isfinite(gtol) or gtol<=0:
        raise ValueError('joint max_seconds and gtol must be finite and positive.')
    started=time.perf_counter();n=len(names)
    lo=np.r_[np.asarray(lower,dtype=float),np.zeros(len(datasets))]
    hi=np.r_[np.asarray(upper,dtype=float),np.ones(len(datasets))]
    def vector(fit):return np.r_[[fit[key] for key in names],fit['initial_states']].astype(float)
    def errors(point):
        p=unpack(point[:n]);states=point[n:]
        return np.concatenate([predict(d,float(x),p,substeps)[1][:len(d['I'])]-d['I']
                               for d,x in zip(datasets,states)])
    initial=vector(fitted)
    if initial.shape!=lo.shape or np.any(initial<lo) or np.any(initial>hi):
        raise ValueError('Saved joint starting point is outside the fit bounds.')
    initial_errors=errors(initial);initial_sse=float(initial_errors@initial_errors)
    selected=initial.copy();selected_sse=initial_sse;seed_source='saved_fit'
    if alternative is not None:
        other=vector(alternative)
        if other.shape!=lo.shape or np.any(other<lo) or np.any(other>hi):
            raise ValueError('Alternative joint starting point is outside the fit bounds.')
        e=errors(other);other_sse=float(e@e)
        if other_sse<selected_sse:selected,selected_sse,seed_source=other.copy(),other_sse,alternative_name
    else:other_sse=None
    seed_sse=selected_sse;best=selected.copy();best_sse=selected_sse
    calls=0;result=None;error=None;termination='max_nfev'
    def residual(point):
        nonlocal calls,best,best_sse
        if time.perf_counter()-started>=max_seconds:raise _TimeLimit()
        values=errors(point);calls+=1
        sse=float(values@values)
        if np.isfinite(sse) and sse<best_sse:
            best,best_sse=point.copy(),sse
        return values
    try:
        result=least_squares(residual,selected,bounds=(lo,hi),method='trf',
            x_scale='jac',max_nfev=max_nfev,gtol=gtol,ftol=1e-8,xtol=1e-8)
        termination={0:'max_nfev',1:'gtol',2:'ftol',3:'xtol',4:'ftol_and_xtol'}.get(int(result.status),'optimizer_failure')
        candidate_errors=errors(result.x);candidate_sse=float(candidate_errors@candidate_errors)
        # On a normal exit select the optimizer iterate, not a Jacobian probe.
        if np.isfinite(candidate_sse) and candidate_sse<=selected_sse:
            selected,selected_sse=result.x.copy(),candidate_sse
    except _TimeLimit:
        termination='time_limit';selected,selected_sse=best,best_sse
    except (ValueError,RuntimeError,FloatingPointError,OverflowError) as exc:
        # A failed proposal must not discard a usable reference forecast.
        termination='evaluation_error';error=str(exc);selected,selected_sse=best,best_sse
    final_errors=errors(selected);final_sse=float(final_errors@final_errors)
    if not np.isfinite(final_sse) or final_sse>initial_sse:
        selected,final_sse=initial,initial_sse
    uses_optimizer=(result is not None and np.array_equal(selected,result.x))
    optimality=float(result.optimality) if uses_optimizer and np.isfinite(result.optimality) else None
    stationary=bool(uses_optimizer and optimality is not None and optimality<=gtol)
    report={'method':'joint_theta_and_five_initial_states','seed_source':seed_source,
        'reference_sse':initial_sse,'linear_candidate_sse':other_sse,'seed_sse':seed_sse,
        'final_sse':final_sse,'accepted_improvement':bool(final_sse<initial_sse),
        'optimizer_success':bool(uses_optimizer and result.success),'first_order_passed':stationary,
        'optimality':optimality,'gtol':float(gtol),'stop_reason':termination,
        'nfev':int(result.nfev) if result is not None else None,'residual_calls':calls,
        'max_nfev':max_nfev,'max_seconds':float(max_seconds),'seconds':time.perf_counter()-started,
        'error':error}
    p=unpack(selected[:n]);states=selected[n:]
    return {**fitted,**p,'initial_states':states.copy(),'gain':float(p['Xi']/p['C']),
        'sse':final_sse,'mse':final_sse/sum(len(d['I']) for d in datasets),
        'alternating_converged':bool(fitted['converged']),
        'alternating_stop_reason':fitted['stop_reason'],'joint_fit':report,
        'converged':stationary,'stop_reason':'joint_'+termination,
        'warning':None if stationary else 'Joint fit did not meet first-order tolerance: '+termination,
        'message':'Joint refinement: '+termination,'fit_method':'pooled_demand_alternating_lsq+joint'}
