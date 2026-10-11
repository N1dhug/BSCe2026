"""Bounded BM2: inertia with the same state-process noise as BM1.

Interior equations: dX = (Xi/C) R dt + sigma X(1-X) dW,
dR = (q(X,B,U)-R)/tau dt, where D=B+Xi*q and R=C*V/Xi.
X is reflected at 0 and 1 (the deterministic limit is a projected flow).
R stays in [-1,1]. tau=m/gamma and C=gamma*C_original remove the
unidentifiable common scale of the historical m, gamma, C parameters.
"""
import math
import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu
from scipy.linalg import solve_banded
from src.model.base.bm import bm
from src.solver.finite_volume import finite_volume_1d


class bm_2(bm):
    @staticmethod
    def response_rate(x, b, u, p):
        if np.ndim(x) == 0:
            f = p.get('scalar_f') or p['f'].scalar_evaluator()
            delta = math.tanh(.5*p['k']*(f(float(x))+float(p['g'](u))))
            return delta*(1-b if delta >= 0 else b)
        delta = np.tanh(.5*p['k']*(p['f'](x)+p['g'](u)))
        return delta*np.where(delta >= 0, 1-b, b)

    @classmethod
    def demand(cls, x, b, u, p):
        return b+p['Xi']*cls.response_rate(x,b,u,p)

    @classmethod
    def advance(cls, x, r, dt, b, u, p, substeps=2):
        """Adaptive RK4 with projected reflection and a relaxation stability cap.

        Step doubling controls errors, including the kink at the X boundary.
        The deterministic projected flow is the zero-noise reflected model.
        """
        gain=p['Xi']/p['C']; tau=p['tau']
        lipschitz=.5*p['k']*p['f'].max_abs_slope
        steps=max(int(substeps),int(math.ceil(2*dt/tau)),int(math.ceil(dt*gain/.2)),
                  int(math.ceil(2*dt*math.sqrt(gain*lipschitz/tau))))
        scalar=np.ndim(x)==0
        clip=(lambda z:min(1.,max(0.,z))) if scalar else (lambda z:np.clip(z,0.,1.))
        gv=float(p['g'](u));half_k=.5*p['k']
        if scalar:
            fcurve=p.get('scalar_f') or p['f'].scalar_evaluator()
            def force_at(z):
                delta=math.tanh(half_k*(fcurve(z)+gv))
                return delta*(1-b if delta>=0 else b)
        else:
            def force_at(z):
                delta=np.tanh(half_k*(p['f'](z)+gv))
                return delta*np.where(delta>=0,1-b,b)
        def rate(x,r):
            force=force_at(clip(x))
            dx=gain*r
            if scalar:
                if (x<=0 and r<0) or (x>=1 and r>0):dx=0.
            else:dx=np.where(((x<=0)&(r<0))|((x>=1)&(r>0)),0.,dx)
            return dx,(force-r)/tau
        def step(x,r,h):
            a,b1=rate(x,r);c,d=rate(x+h*a/2,r+h*b1/2)
            e,f=rate(x+h*c/2,r+h*d/2);g,j=rate(x+h*e,r+h*f)
            return clip(x+h*(a+2*c+2*e+g)/6),r+h*(b1+2*d+2*f+j)/6
        # Step doubling controls interior error and the kink at reflection.
        # Finer substeps also tighten the tolerance, so the numerical check
        # is independent of this baseline error target.
        tolerance=1e-8*min(2.,float(substeps))/substeps
        xscale=max(1.,p['Xi']*lipschitz);rscale=max(1.,gain)
        elapsed=0.;hmax=dt/steps;h=hmax;attempts=0
        while elapsed<dt-1e-14:
            h=min(h,dt-elapsed);attempts+=1
            if attempts>100000 or h<1e-12:raise RuntimeError('BM2 adaptive integration failed to progress.')
            full_x,full_r=step(x,r,h)
            mid_x,mid_r=step(x,r,h/2);fine_x,fine_r=step(mid_x,mid_r,h/2)
            if scalar:error=max(xscale*abs(full_x-fine_x),rscale*abs(full_r-fine_r))
            else:error=max(xscale*float(np.max(abs(full_x-fine_x))),rscale*float(np.max(abs(full_r-fine_r))))
            if error<=tolerance:
                x,r=fine_x,fine_r;elapsed+=h
                if error<tolerance/8:h=min(hmax,2*h)
            else:h/=2
        return x,r

    @classmethod
    def predict(cls, data, initial, p, substeps=2):
        time=np.asarray(data['Time']); states=np.empty((len(time),2)); demand=np.empty(len(time))
        initial=np.asarray(initial,dtype=float)
        if initial.shape!=(2,) or not np.all(np.isfinite(initial)) or not 0<=initial[0]<=1 or not -1<=initial[1]<=1:
            raise ValueError('BM2 requires initial [X0,R0] in [0,1] x [-1,1].')
        states[0]=initial
        for i,t in enumerate(time):
            b,u=float(data['B'][i]),float(data['U'][i])
            demand[i]=cls.demand(float(states[i,0]),b,u,p)
            if i+1<len(time):
                states[i+1]=cls.advance(*map(float,states[i]),float(time[i+1]-t),b,u,p,substeps)
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(demand)):
            raise FloatingPointError('Nonfinite BM2 trajectory.')
        return states,demand

    @staticmethod
    def grid(x_steps=101,r_steps=41):
        if any(isinstance(n,bool) or not isinstance(n,int) or n<3 for n in (x_steps,r_steps)):
            raise ValueError('BM2 grid sizes must be integers >=3.')
        return (np.arange(x_steps)+.5)/x_steps, -1+2*(np.arange(r_steps)+.5)/r_steps

    @classmethod
    def generator(cls,x,r,b,u,p,sigma_x):
        """Conservative upwind/centered Ito generator; zero exterior flux.

        Flattening is C order: index=x_index*n_r+r_index. Diffusion acts
        only in X. Its coefficient and boundary convention match BM1 FV.
        """
        nx,nr=len(x),len(r); dx=1/nx; dr=2/nr
        ids=np.arange(nx*nr).reshape(nx,nr)
        q=cls.response_rate(x,b,u,p)
        ax=p['Xi']/p['C']*r
        diffusion=.5*(sigma_x*x*(1-x))**2
        right=np.maximum(ax,0)[None,:]/dx+diffusion[:-1,None]/dx**2
        left=np.maximum(-ax,0)[None,:]/dx+diffusion[1:,None]/dx**2
        rf=(r[:-1]+r[1:])/2
        ar=(q[:,None]-rf[None,:])/p['tau']
        sources=np.concatenate((ids[:-1].ravel(),ids[1:].ravel(),ids[:,:-1].ravel(),ids[:,1:].ravel()))
        targets=np.concatenate((ids[1:].ravel(),ids[:-1].ravel(),ids[:,1:].ravel(),ids[:,:-1].ravel()))
        rates=np.concatenate((right.ravel(),left.ravel(),(np.maximum(ar,0)/dr).ravel(),(np.maximum(-ar,0)/dr).ravel()))
        diagonal=-np.bincount(sources,weights=rates,minlength=nx*nr)
        return coo_matrix((np.r_[rates,diagonal],(np.r_[targets,ids.ravel()],np.r_[sources,ids.ravel()])),shape=(nx*nr,nx*nr)).tocsc()

    @classmethod
    def evolve_reference(cls,data,p,initial_mass,sigma_x,*,max_step=.25):
        """Joint (X,R) law by implicit FV, without clipping or renormalization."""
        mass=np.asarray(initial_mass,dtype=float)
        if mass.ndim!=2 or not np.all(np.isfinite(mass)) or np.any(mass<0) or abs(mass.sum()-1)>1e-10:
            raise ValueError('Initial BM2 cell masses must be nonnegative and sum to one.')
        if not np.isfinite(sigma_x) or sigma_x<0 or not np.isfinite(max_step) or max_step<=0:
            raise ValueError('Invalid diffusion amplitude or FP time step.')
        x,r=cls.grid(*mass.shape); time=np.asarray(data['Time'],dtype=float)
        grid,output_indices=finite_volume_1d.build_time_grid(time,max_step=max_step)
        history=np.empty((len(time),*mass.shape)); history[0]=mass
        mass=mass.ravel().copy(); cache={}; output=1
        identity=eye(len(mass),format='csc')
        max_mass_error=0.; min_mass=float(mass.min())
        for i,(left,right) in enumerate(zip(grid[:-1],grid[1:]),start=1):
            j=min(len(time)-1,int(np.searchsorted(time,left,side='right')-1))
            b,u=float(data['B'][j]),float(data['U'][j]); dt=float(right-left)
            key=(b,u,dt)
            if key not in cache:
                cache[key]=splu(identity-dt*cls.generator(x,r,b,u,p,sigma_x))
            mass=cache[key].solve(mass)
            error=abs(float(mass.sum())-1); minimum=float(mass.min())
            max_mass_error=max(max_mass_error,error); min_mass=min(min_mass,minimum)
            if not np.all(np.isfinite(mass)) or minimum < -1e-12 or error>1e-8:
                raise FloatingPointError('BM2 FP lost positivity or probability mass.')
            if output<len(time) and i==output_indices[output]:
                history[output]=mass.reshape(len(x),len(r)); output+=1
        return {'x_grid':x,'r_grid':r,'cell_mass':history,
                'diagnostics':{'maximum_mass_error':max_mass_error,'minimum_mass':min_mass,
                               'x_steps':len(x),'r_steps':len(r),'max_step':max_step,
                               'boundary':'zero flux, reflected X; invariant R interval'}}

    @classmethod
    def evolve(cls,data,p,initial_mass,sigma_x,*,max_step=.25):
        """Limited second-order transport and implicit Ito diffusion splitting.

        Minmod reconstruction + SSPRK2 reduce the first-order reference's
        artificial velocity diffusion. A conservative CFL bound preserves
        positivity; zero exterior flux implements the same reflected model.
        Diffusion half steps use backward Euler for unconditional positivity.
        """
        mass=np.asarray(initial_mass,dtype=float).copy()
        if mass.ndim!=2 or not np.all(np.isfinite(mass)) or np.any(mass<0) or abs(mass.sum()-1)>1e-10:
            raise ValueError('Initial BM2 cell masses must be nonnegative and sum to one.')
        if not np.isfinite(sigma_x) or sigma_x<0 or not np.isfinite(max_step) or max_step<=0:
            raise ValueError('Invalid diffusion amplitude or FP time step.')
        x,r=cls.grid(*mass.shape);nx,nr=mass.shape;dx=1/nx;dr=2/nr
        times=np.asarray(data['Time'],dtype=float)
        if len(times)<2 or not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0):raise ValueError('Invalid BM2 time grid.')
        history=np.empty((len(times),nx,nr));history[0]=mass
        ax=p['Xi']/p['C']*r
        lower,diagonal,upper,_=finite_volume_1d.generator_tridiagonal(dx,np.zeros(nx-1),sigma_x*x*(1-x))
        def diffuse(m,h):
            if sigma_x==0:return m
            bands=np.zeros((3,nx));bands[0,1:]=-h*upper;bands[1]=1-h*diagonal;bands[2,:-1]=-h*lower
            return solve_banded((1,1),bands,m,check_finite=False)
        def minmod(a,b):return np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.)
        def transport(m,ar):
            sx=np.zeros_like(m);sr=np.zeros_like(m)
            sx[1:-1]=minmod(m[1:-1]-m[:-2],m[2:]-m[1:-1])
            sr[:,1:-1]=minmod(m[:,1:-1]-m[:,:-2],m[:,2:]-m[:,1:-1])
            fx=np.zeros((nx+1,nr));fr=np.zeros((nx,nr+1))
            fx[1:-1]=ax[None,:]*np.where(ax[None,:]>=0,m[:-1]+sx[:-1]/2,m[1:]-sx[1:]/2)
            fr[:,1:-1]=ar*np.where(ar>=0,m[:,:-1]+sr[:,:-1]/2,m[:,1:]-sr[:,1:]/2)
            return -np.diff(fx,axis=0)/dx-np.diff(fr,axis=1)/dr
        maximum_error=0.;minimum=float(mass.min());steps=0
        for i in range(len(times)-1):
            b,u=float(data['B'][i]),float(data['U'][i])
            q=cls.response_rate(x,b,u,p);ar=(q[:,None]-(r[:-1]+r[1:])[None,:]/2)/p['tau']
            rate=float(np.max(abs(ax)))/dx+float(np.max(abs(ar)))/dr
            limit=min(max_step,.4/rate) if rate>0 else max_step
            n=max(1,int(math.ceil((times[i+1]-times[i])/limit)));h=(times[i+1]-times[i])/n
            for _ in range(n):
                first=diffuse(mass,h/2)
                stage=first+h*transport(first,ar)
                mass=diffuse(.5*first+.5*(stage+h*transport(stage,ar)),h/2)
                error=abs(float(mass.sum())-1);smallest=float(mass.min())
                maximum_error=max(maximum_error,error);minimum=min(minimum,smallest);steps+=1
                if not np.all(np.isfinite(mass)) or smallest < -1e-12 or error>1e-8:
                    raise FloatingPointError('BM2 limited transport lost positivity or probability mass.')
            history[i+1]=mass
        return {'x_grid':x,'r_grid':r,'cell_mass':history,
            'diagnostics':{'method':'minmod SSPRK2 transport / implicit diffusion splitting',
                'maximum_mass_error':maximum_error,'minimum_mass':minimum,'internal_steps':steps,
                'x_steps':nx,'r_steps':nr,'max_step':max_step,'boundary':'zero flux'}}
