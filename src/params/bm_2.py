"""Five historical windows, common BM2 parameters and five paired states."""
import math
import time
import numpy as np
from scipy.optimize import least_squares
from src.model.bm_2 import bm_2
from src.params.bm_1 import lsq_params as BM1Fit
from src.params.bm_1_families import curves,shape_config
from src.validation.params.bm_1 import bm_1_params_validation as checks


class lsq_params:
    BOUNDS={**BM1Fit.BOUNDS,'tau':(.05,48.)}
    _predict=staticmethod(bm_2.predict)

    def __init__(self,model,family='linear'):
        self.model=model; self.family=family
        if family not in ('linear','piecewise-linear','pchip','bernstein','beta-cdf','logistic'):
            raise ValueError('Unknown response family.')

    def fit_many(self,datasets,X0=.5,*,R0=0.,C=None,Xi=None,tau=None,tau0=1.,
                 max_iterations=6,max_nfev=150,state_maxiter=100,substeps=2,
                 min_improvement=1e-6,fit_bounds=None,seed_fit=None,
                 joint_max_nfev=0,joint_max_seconds=20.,joint_gtol=1e-5):
        prepared,*_=checks.fit_many_inputs(self.model,datasets,X0,C=C,Xi=Xi,
            alpha0=2.,beta0=2.,k0=1.,C0=1.,Xi0=.5,max_iterations=max_iterations,
            min_improvement=min_improvement,max_nfev=max_nfev,state_maxiter=state_maxiter,
            substeps=substeps,names=BM1Fit.NAMES,bounds=BM1Fit.BOUNDS,fit_bounds=fit_bounds)
        if not np.isfinite(R0) or not -1<=R0<=1:raise ValueError('R0 must be in [-1,1].')
        if not np.isfinite(tau0) or not .05<=tau0<=48:raise ValueError('tau0 must be in [.05,48] hours.')
        sn,ss,sl,su=shape_config(self.family)
        bounds={**self.BOUNDS,**dict(zip(sn,zip(sl,su))),**(fit_bounds or {})}
        fixed={key:float(value) for key,value in (('C',C),('Xi',Xi),('tau',tau)) if value is not None}
        if any(not np.isfinite(v) or not bounds[k][0]<=v<=bounds[k][1] for k,v in fixed.items()):
            raise ValueError('Fixed BM2 parameter outside its bounds.')
        free=[n for n in sn+['k','Xi','C','tau'] if n not in fixed]
        values={**dict(zip(sn,ss)),'k':1.,'Xi':.5,'C':1.,'tau':tau0}
        if seed_fit is not None:
            values.update({n:float(seed_fit[n]) for n in free if n in seed_fit})
        lower=np.array([bounds[n][0] for n in free]); upper=np.array([bounds[n][1] for n in free])
        theta=np.clip([values[n] for n in free],lower+1e-10,upper-1e-10)
        def unpack(v):
            p={**fixed,**dict(zip(free,v))}; f,g=curves(self.family,[p[n] for n in sn])
            return {**p,'f':f,'g':g,'scalar_f':f.scalar_evaluator()}
        states=np.tile([X0,R0],(5,1)).astype(float)
        if seed_fit is not None:
            states[:,0]=seed_fit['initial_states']
            p=unpack(theta)
            for j,d in enumerate(prepared):
                states[j,1]=bm_2.response_rate(states[j,0],float(d['B'][0]),float(d['U'][0]),p)
        def residual(v,z):
            p=unpack(v)
            return np.concatenate([bm_2.predict(d,s,p,substeps)[1][:len(d['I'])]-d['I'] for d,s in zip(prepared,z)])
        initial_error=residual(theta,states); best_sse=float(initial_error@initial_error)
        best_theta=theta.copy();best_states=states.copy();history=[];stop='max_iterations'
        for iteration in range(1,max_iterations+1):
            opt=least_squares(lambda v:residual(v,states),theta,bounds=(lower,upper),max_nfev=max_nfev,x_scale='jac')
            report=checks.capped_optimizer_result(opt,'BM2 pooled demand',limit_status=0)
            # Both block updates retain their starting point if a proposal is worse.
            before=residual(theta,states); after=residual(opt.x,states)
            candidate=opt.x.copy() if after@after<=before@before else theta.copy()
            p=unpack(candidate); newstates=states.copy(); state_success=True
            for j,(d,s) in enumerate(zip(prepared,states)):
                def error(z):return bm_2.predict(d,z,p,substeps)[1][:len(d['I'])]-d['I']
                result=least_squares(error,s,bounds=([0.,-1.],[1.,1.]),max_nfev=state_maxiter,x_scale='jac')
                sr=checks.capped_optimizer_result(result,'BM2 initial pair',limit_status=0)
                old_error=error(s);new_error=error(result.x)
                if new_error@new_error<=old_error@old_error:newstates[j]=result.x
                state_success=state_success and sr['converged']
            e=residual(candidate,newstates); sse=float(e@e); improvement=best_sse-sse
            accepted=sse<=best_sse
            history.append({'iteration':iteration,'sse':sse,'mse':sse/120,
                'improvement':improvement,'accepted':accepted,'theta_converged':report['converged'],
                'theta_nfev':report['nfev'],'theta_status':report['status'],'state_converged':state_success})
            if accepted:best_theta,best_states,best_sse=candidate.copy(),newstates.copy(),sse
            theta,states=best_theta.copy(),best_states.copy()
            if report['limit_reached']:stop='max_nfev';break
            if not state_success:stop='state_maxiter';break
            if not accepted:stop='no_improvement';break
            if iteration>1 and improvement<=min_improvement*max(sse+improvement,1e-10):stop='min_improvement';break
        converged=stop in ('min_improvement','no_improvement');joint=None
        if joint_max_nfev:
            if joint_max_nfev<1 or joint_max_seconds<=0 or joint_gtol<=0:raise ValueError('Invalid joint controls.')
            started=time.perf_counter(); initial_sse=best_sse;calls=0;opt=None
            best_point=np.r_[theta,states.ravel()]
            class BudgetReached(Exception):pass
            def joint_error(z):
                nonlocal best_point,best_sse,calls
                if time.perf_counter()-started>=joint_max_seconds:raise BudgetReached()
                e=residual(z[:len(free)],z[len(free):].reshape(5,2));calls+=1;sse=float(e@e)
                if sse<best_sse:best_point,best_sse=z.copy(),sse
                return e
            try:
                opt=least_squares(joint_error,best_point,
                    bounds=(np.r_[lower,np.tile([0.,-1.],5)],np.r_[upper,np.tile([1.,1.],5)]),
                    max_nfev=joint_max_nfev,x_scale='jac',gtol=joint_gtol,ftol=1e-8,xtol=1e-8)
                stop={0:'max_nfev',1:'gtol',2:'ftol',3:'xtol',4:'ftol_and_xtol'}[int(opt.status)]
                e=residual(opt.x[:len(free)],opt.x[len(free):].reshape(5,2));sse=float(e@e)
                if sse<=initial_sse:best_point,best_sse=opt.x.copy(),sse
            except BudgetReached:stop='time_limit'
            uses_opt=opt is not None and np.array_equal(best_point,opt.x)
            optimality=float(opt.optimality) if uses_opt else None
            converged=bool(uses_opt and optimality<=joint_gtol)
            joint={'reference_sse':initial_sse,'final_sse':best_sse,'first_order_passed':converged,
                'optimizer_success':bool(uses_opt and opt.success),'optimality':optimality,
                'stop_reason':stop,'seconds':time.perf_counter()-started,'residual_calls':calls,
                'gtol':joint_gtol,'max_nfev':joint_max_nfev,'max_seconds':joint_max_seconds}
            best_theta=best_point[:len(free)];best_states=best_point[len(free):].reshape(5,2)
            stop='joint_'+stop
        p=unpack(best_theta)
        return {**p,'family':self.family,'shape_parameters':{n:float(p[n]) for n in sn},
            'initial_states':best_states,'initial_velocities':best_states[:,1]*p['Xi']/p['C'],
            'initial_weights':np.full(5,.2),'sse':best_sse,'mse':best_sse/120,
            'history':history,'iterations':len(history),'converged':converged,'stop_reason':stop,
            'warning':None if converged else stop,'theta_nfev':history[-1]['theta_nfev'],
            'joint_fit':joint,'fit_method':'pooled_alternating_BM2'+('+joint' if joint is not None else ''),
            'gamma_fixed':1.}


class historical_diffusion_params:
    """BM1's historical variance regression extended to a 2x2 state covariance.

    P_next = A P A.T + diag(X^2(1-X)^2 dt, 0) for unit sigma^2.
    A is the Jacobian of the reflected deterministic one-hour transition;
    demand sensitivity is [dD/dX,0]. This is the same discrete linear-noise
    approximation as BM1, not exact SDE maximum likelihood.
    """
    def fit_many(self,datasets,fit,*,substeps=2,derivative_step=1e-4,minimum_observation_sd=.005):
        kernels=[];squares=[];initial=[];errors=[]
        for d,s in zip(datasets,fit['initial_states']):
            states,D=bm_2.predict(d,s,fit,substeps);e=np.asarray(d['I'])-D[:len(d['I'])]
            initial.append(float(e[0]));errors.extend(e);cov=np.zeros((2,2))
            for k in range(len(e)-1):
                x,r=states[k];dt=float(d['Time'][k+1]-d['Time'][k]);A=np.empty((2,2))
                for axis,(lo,hi) in enumerate(((0.,1.),(-1.,1.))):
                    plus=states[k].copy();minus=plus.copy()
                    plus[axis]=min(hi,plus[axis]+derivative_step);minus[axis]=max(lo,minus[axis]-derivative_step)
                    args=(dt,float(d['B'][k]),float(d['U'][k]),fit,substeps)
                    A[:,axis]=(np.asarray(bm_2.advance(*plus,*args))-np.asarray(bm_2.advance(*minus,*args)))/(plus[axis]-minus[axis])
                cov=A@cov@A.T+np.diag([(x*(1-x))**2*dt,0.])
                j=k+1;x=states[j,0];lo=max(0.,x-derivative_step);hi=min(1.,x+derivative_step)
                J=(bm_2.demand(hi,float(d['B'][j]),float(d['U'][j]),fit)-bm_2.demand(lo,float(d['B'][j]),float(d['U'][j]),fit))/(hi-lo)
                kernels.append(J*J*cov[0,0]);squares.append(float(e[j]**2))
        sd=max(minimum_observation_sd,float(np.sqrt(np.mean(np.square(initial)))))
        K=np.asarray(kernels);E=np.asarray(squares);target=np.maximum(E-sd*sd,0.)
        if len(K)<20 or not np.all(np.isfinite(K)):raise ValueError('Invalid BM2 variance kernels.')
        denominator=float(K@K);identified=denominator>1e-30
        q=max(0.,float(K@target/denominator)) if identified else 0.
        return {'sigma_x':math.sqrt(q),'q':q,'observation_sd':sd,'samples':len(K),
            'historical_rmse':float(np.sqrt(np.mean(np.square(errors)))),
            'variance_rmse':float(np.sqrt(np.mean((K*q-target)**2))),
            'diffusion_identified':identified,'method':'BM2 paired-state historical linear-noise variance regression'}
