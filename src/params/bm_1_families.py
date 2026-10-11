"""Response-family fits following Thomas' five-window estimation recipe.

All model fitting, state fitting, baselines and scoring follow Thomas' recipe.
Only the response-family parameterization and a slope-aware RK4 safety step
are adapted. Target demand never enters the daily fit. MPC uses I(origin).
"""
import math
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from src.params.bm_1 import lsq_params as ThomasFit
from src.params.bm_1_response import BM1PiecewiseResponse, BM1BetaResponse, BM1LogisticResponse
from src.validation.params.bm_1 import bm_1_params_validation as validation
PIECES=5

def curves(family, values):
    if family=='linear':
        return BM1PiecewiseResponse.linear(),BM1PiecewiseResponse.linear()
    if family=='logistic':
        return BM1LogisticResponse(values[0]),BM1LogisticResponse(values[1])
    if family=='beta-cdf':
        return BM1BetaResponse(*values[:2]),BM1BetaResponse(*values[2:4])
    n=PIECES-1
    interp='linear' if family=='piecewise-linear' else family
    return (BM1PiecewiseResponse.from_drop_fractions(values[:n],interpolation=interp),
            BM1PiecewiseResponse.from_drop_fractions(values[n:2*n],interpolation=interp))

def shape_config(family):
    if family=='linear': return [],[],[],[]
    if family=='logistic': return ['alpha','beta'],[2.,2.],[.05,.05],[20.,20.]
    if family=='beta-cdf': return ['f_a','f_b','g_a','g_b'],[1.,1.,1.,1.],[1.]*4,[10.]*4
    seed=list(1./np.arange(PIECES,1,-1))
    names=[f'{name}_drop_{i}' for name in ('f','g') for i in range(1,PIECES)]
    return names,seed+seed,[0.]*len(names),[1.]*len(names)

class FamilyFit(ThomasFit):
    FAMILY='linear'

    def __init__(self,model,family=None):
        super().__init__(model)
        if family is not None:self.FAMILY=family
        if self.FAMILY not in ('linear','piecewise-linear','pchip','bernstein','beta-cdf','logistic'):
            raise ValueError('Unknown response family.')

    @classmethod
    def _demand(cls,x,b,u,p):
        f,g=p['f'],p['g']
        delta=math.tanh(.5*p['k']*(float(f(x))+float(g(u))))
        return b+p['Xi']*delta*(1-b if delta>=0 else b)

    @classmethod
    def _advance(cls,x,dt,b,u,p,substeps):
        f=p['scalar_f']; gv=float(p['g'](u))
        # h*L<=1 preserves the response domain at RK4 stages; no state clipping.
        steps=max(substeps,int(math.ceil(dt*p['Xi']/(.2*p['C']))),
                  int(math.ceil(dt*.5*p['k']*p['Xi']*p['f'].max_abs_slope/p['C'])))
        h=dt/steps
        def rate(z):
            delta=math.tanh(.5*p['k']*(f(z)+gv))
            return p['Xi']*delta*(1-b if delta>=0 else b)/p['C']
        for _ in range(steps):
            v1=rate(x); v2=rate(x+h*v1/2); v3=rate(x+h*v2/2); v4=rate(x+h*v3)
            x += h*(v1+2*v2+2*v3+v4)/6
        if not math.isfinite(x) or not 0<=x<=1: raise ValueError('State left [0,1].')
        return x

    @classmethod
    def _predict(cls,data,X0,p,substeps):
        t,B,U=data['Time'],data['B'],data['U']
        X=np.empty(len(t)); D=np.empty(len(t)); X[0]=X0
        for i in range(len(t)):
            D[i]=cls._demand(float(X[i]),float(B[i]),float(U[i]),p)
            if i+1<len(t): X[i+1]=cls._advance(float(X[i]),float(t[i+1]-t[i]),float(B[i]),float(U[i]),p,substeps)
        return X,D

    def fit_many(self,datasets,X0=.5,*,C=None,Xi=None,alpha0=2.,beta0=2.,k0=1.,C0=1.,Xi0=.5,
                 max_iterations=6,min_improvement=1e-6,max_nfev=150,state_maxiter=100,
                 substeps=2,loss='linear',fit_bounds=None,joint_max_nfev=0,joint_max_seconds=20.,joint_gtol=1e-5):
        # Reuse the source's input/control checks before setting family shapes.
        prepared,*_=validation.fit_many_inputs(self.model,datasets,X0,C=C,Xi=Xi,
            alpha0=alpha0,beta0=beta0,k0=k0,C0=C0,Xi0=Xi0,max_iterations=max_iterations,
            min_improvement=min_improvement,max_nfev=max_nfev,state_maxiter=state_maxiter,
            substeps=substeps,names=ThomasFit.NAMES,bounds=ThomasFit.BOUNDS,fit_bounds=fit_bounds)
        sn,ss,sl,su=shape_config(self.FAMILY)
        names=sn+['k','Xi','C']
        seed=np.array(ss+[k0,Xi0,C0],float)
        bounds={**{n:(lo,hi) for n,lo,hi in zip(sn,sl,su)},
                **{n:ThomasFit.BOUNDS[n] for n in ('k','Xi','C')}}
        if fit_bounds: bounds.update(fit_bounds)
        fixed={k:float(v) for k,v in [('C',C),('Xi',Xi)] if v is not None}
        free=[n for n in names if n not in fixed]
        seed=np.array([seed[names.index(n)] for n in free])
        lower=np.array([bounds[n][0] for n in free]); upper=np.array([bounds[n][1] for n in free])
        seed=np.clip(seed,lower+1e-9,upper-1e-9)
        def unpack(theta):
            p={**fixed,**dict(zip(free,theta))}
            f,g=curves(self.FAMILY,[p[n] for n in sn])
            p.update(f=f,g=g,scalar_f=f.scalar_evaluator())
            return p
        def residual(theta,states):
            p=unpack(theta)
            return np.concatenate([self._predict(d,float(x),p,substeps)[1][:len(d['I'])]-d['I'] for d,x in zip(prepared,states)])
        def refine_states(p,initial):
            states,reports=[],[]
            for d,x in zip(prepared,initial):
                def error(z):
                    e=self._predict(d,float(z),p,substeps)[1][:len(d['I'])]-d['I']
                    return float(e@e)
                result=minimize_scalar(error,bounds=(0.,1.),method='bounded',options={'xatol':1e-6,'maxiter':state_maxiter})
                reports.append(validation.capped_optimizer_result(result,'initial-state',limit_status=1))
                states.append(float(result.x) if error(result.x)<=error(x) else float(x))
            return np.asarray(states),reports
        states=np.full(5,X0); theta=seed.copy(); history=[]; best=None; stop='max_iterations'
        for iteration in range(1,max_iterations+1):
            opt=least_squares(lambda v:residual(v,states),theta,bounds=(lower,upper),loss=loss,max_nfev=max_nfev)
            report=validation.capped_optimizer_result(opt,'pooled demand',limit_status=0)
            candidate=opt.x.copy(); p=unpack(candidate)
            newstates,statereports=refine_states(p,states)
            e=residual(candidate,newstates); sse=float(e@e)
            previous=np.inf if best is None else best['sse']
            improved=sse<previous; state_limit=any(r['limit_reached'] for r in statereports)
            history.append({'iteration':iteration,'sse':sse,'mse':float(np.mean(e**2)),
                'improvement':None if best is None else previous-sse,'accepted':improved,
                'theta_converged':report['converged'],'theta_nfev':report['nfev'],
                'theta_status':report['status'],'state_converged':not state_limit})
            if not improved:
                stop='max_nfev' if report['limit_reached'] else ('state_maxiter' if state_limit else 'no_improvement'); break
            best={'params':p,'states':newstates,'sse':sse,'mse':float(np.mean(e**2)),'report':report}
            theta,states=candidate,newstates
            if report['limit_reached']: stop='max_nfev'; break
            if state_limit: stop='state_maxiter'; break
            if iteration>1 and previous-sse<=min_improvement*max(previous,1e-10): stop='min_improvement'; break
        converged=stop in ('min_improvement','no_improvement')
        fitted={**best['params'],'gain':best['params']['Xi']/best['params']['C'],
            'initial_states':best['states'],'initial_weights':np.full(5,.2),'mse':best['mse'],'sse':best['sse'],
            'history':history,'iterations':len(history),'converged':converged,'stop_reason':stop,
            'warning':None if converged else stop,'message':stop,'theta_nfev':best['report']['nfev'],
            'family':self.FAMILY,'shape_parameters':{n:float(best['params'][n]) for n in sn}}
        if joint_max_nfev:
            if loss!='linear':raise ValueError('Joint refinement currently uses unpenalized squared residuals.')
            return self.refine_joint(prepared,fitted,C=C,Xi=Xi,fit_bounds=fit_bounds,
                max_nfev=joint_max_nfev,max_seconds=joint_max_seconds,gtol=joint_gtol,substeps=substeps)
        return fitted

    def refine_joint(self,datasets,fitted,*,C=None,Xi=None,fit_bounds=None,max_nfev=80,
                     max_seconds=20.,gtol=1e-5,substeps=2,linear_fit=None):
        from src.params.bm_1_joint import refine_joint
        family=fitted.get('family',self.FAMILY)
        sn,seed,sl,su=shape_config(family)
        bounds={**{n:(lo,hi) for n,lo,hi in zip(sn,sl,su)},
                **{n:ThomasFit.BOUNDS[n] for n in ('k','Xi','C')},**(fit_bounds or {})}
        fixed={key:float(value) for key,value in (('C',C),('Xi',Xi)) if value is not None}
        names=[key for key in sn+['k','Xi','C'] if key not in fixed]
        def unpack(values):
            p={**fixed,**dict(zip(names,values))}
            f,g=curves(family,[p[key] for key in sn]);p.update(f=f,g=g,scalar_f=f.scalar_evaluator())
            return p
        candidate=None
        if linear_fit is not None and family in ('piecewise-linear','bernstein'):
            candidate={**linear_fit,**dict(zip(sn,seed))}
        result=refine_joint(datasets,fitted,predict=self._predict,unpack=unpack,names=names,
            lower=[bounds[key][0] for key in names],upper=[bounds[key][1] for key in names],
            max_nfev=max_nfev,max_seconds=max_seconds,gtol=gtol,substeps=substeps,alternative=candidate)
        result['shape_parameters']={key:float(result[key]) for key in sn}
        return result

class FamilyDiffusion:
    def fit_many(self,datasets,fit,*,substeps=2,derivative_step=1e-4,minimum_observation_sd=.005):
        kernels,squares,initial,errors=[],[],[],[]
        for data,x0 in zip(datasets,fit['initial_states']):
            X,D=FamilyFit._predict(data,float(x0),fit,substeps)
            e=data['I']-D[:len(data['I'])]; initial.append(float(e[0])); errors.extend(e.tolist()); variance=0.
            for k in range(len(e)-1):
                x=float(X[k]); lo=max(0.,x-derivative_step); hi=min(1.,x+derivative_step)
                args=(float(data['Time'][k+1]-data['Time'][k]),float(data['B'][k]),float(data['U'][k]),fit,substeps)
                A=(FamilyFit._advance(hi,*args)-FamilyFit._advance(lo,*args))/(hi-lo)
                variance=A*A*variance+(x*(1-x))**2*args[0]
                j=k+1; x=float(X[j]); lo=max(0.,x-derivative_step); hi=min(1.,x+derivative_step)
                args=(float(data['B'][j]),float(data['U'][j]),fit)
                J=(FamilyFit._demand(hi,*args)-FamilyFit._demand(lo,*args))/(hi-lo)
                kernels.append(J*J*variance); squares.append(float(e[j]**2))
        sd=max(minimum_observation_sd,float(np.sqrt(np.mean(np.square(initial)))))
        K,E=validation.diffusion_variance_fit(kernels,squares,sd*sd)
        target=np.maximum(E-sd*sd,0); q=max(0.,float(K@target/(K@K)))
        return {'sigma_x':math.sqrt(q),'q':q,'observation_sd':sd,'historical_rmse':float(np.sqrt(np.mean(np.square(errors)))),
                'variance_rmse':float(np.sqrt(np.mean((K*q-target)**2))),'samples':len(K),
                'method':'Thomas historical linear-noise regression; one-sided derivatives at domain endpoints'}

