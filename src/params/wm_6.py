"""Joint drift and stochastic variance convex QPs for WM6."""
from time import perf_counter
import numpy as np

from src.solver.active_set import active_set
from src.validation.model.wm_4 import wm_4_validation


class joint_qp_params:
    """Shared [theta_F,theta_U,theta_R,theta_B], independent beta0_n=theta_F*F0_n."""
    def __init__(self,model): self.model=model

    def _prepare_data(self,item):
        data=wm_4_validation.fit_data(item,self.model.t_steps)
        O=np.asarray(item.get("O",[self.model.O(t) for t in data["Time"]]),dtype=float).reshape(-1)
        if O.size!=self.model.t_steps or not np.isfinite(O).all(): raise ValueError("WM6 O needs one finite sample per timestep.")
        self.model.mapped_baseline(O)
        return {**data,"O":O}

    def fit_many(self,input_data,initial_conditions=None,*,stabilize=True,enforce_path=True,
                 f_bounds=(-1.,1.),verbose=False):
        if not isinstance(input_data,(tuple,list)) or not input_data: raise ValueError("WM6 requires nonempty historical intervals.")
        started=perf_counter()
        data=[self._prepare_data(item) for item in input_data]
        n=len(data)
        dt=self.model.dt
        m=n+4
        rows,targets,constr,rhs=[],[],[],[]
        eps=self.model._FIT_EPS
        strict=10*eps

        # Structural constraints, coefficient columns [theta_F, theta_U, theta_R, theta_B].
        # theta_B >= 0, theta_U <= 0 => |theta_B|+|theta_U|=theta_B-theta_U.
        coeff=np.array([[1.,-1.,0.,1.],[-1.,-1.,1.,1.]])
        bounds=np.array([0.,0.])
        if stabilize:
            coeff=np.vstack((coeff,[dt*dt,0.,-2*dt,0.],[-dt,0.,1.,0.]))
            bounds=np.r_[bounds,4.-eps,-strict]
        constr.append(np.column_stack((np.zeros((len(coeff),n)),coeff)))
        rhs.append(bounds)

        for i,item in enumerate(data):
            R=item["I"]-item["O"]
            C=np.r_[0.,np.cumsum(dt*R[:-1])]
            Bmapped=self.model.mapped_baseline(item["O"])
            X=np.zeros((len(R)-1,m),dtype=float)
            X[:,i]=1.
            X[:,n:]=np.column_stack((C[:-1],item["U"][:-1],R[:-1],Bmapped[:-1]))
            mask=item["transition_mask"]
            rows.append(X[mask]);targets.append((np.diff(R)/dt)[mask])
            if enforce_path:
                lower,upper=map(float,f_bounds)
                if lower>=upper: raise ValueError("Invalid preferred flexibility bounds.")
                a=np.zeros((2,m))
                a[0,i]=-1.; a[0,n]=upper-float(np.max(C))
                a[1,i]=1.; a[1,n]=float(np.min(C))-lower
                constr.append(a);rhs.append(np.zeros(2))

        X=np.vstack(rows);y=np.concatenate(targets)
        if len(y)<1: raise ValueError("WM6 has no valid rate transitions.")
        A=np.vstack(constr);b=np.concatenate(rhs)
        lb=np.r_[np.full(n,-np.inf),[-np.inf,-np.inf,-np.inf,0.]]
        ub=np.r_[np.full(n,np.inf),[-strict,0.,-strict,np.inf]]
        scale=max(1,len(y))
        H=2.*X.T@X/scale+self.model._FIT_RIDGE*np.eye(m)
        q=-2.*X.T@y/scale
        beta=np.asarray(active_set.solve(H,q,A_ub=A,b_ub=b,lb=lb,ub=ub,
                          tol=1e-9,max_iter=10000),dtype=float)
        theta=self.model.coefficients(beta[n:])
        F0=beta[:n]/theta[0]
        initial=np.column_stack((F0,[item["I"][0]-item["O"][0] for item in data]))
        if np.any(A@beta>b+1e-6): raise RuntimeError("WM6 QP structural constraint violation.")
        if enforce_path and np.any((F0<f_bounds[0]-1e-6)|(F0>f_bounds[1]+1e-6)):
            raise RuntimeError("WM6 QP preferred state violation.")
        states=[self.model._solve_deterministic(init,item["U"],item["PE"],
                      item["Time"],item["O"],theta) for init,item in zip(initial,data)]
        error=np.concatenate([(s["I"]-item["I"])[item["observation_mask"]]
                              for s,item in zip(states,data)])
        result={"r":theta,"beta":beta,"init_cond":initial,
                "deterministic_states":states,"deterministic_loss":float(np.mean(error**2)),
                "qp_loss":float(np.mean((X@beta-y)**2)),
                "design_rank":int(np.linalg.matrix_rank(X)),"design_columns":m,
                "transition_count":len(y),"fit_method":"wm6_joint_qp_many",
                "fit_trajectories":n,"fit_time":perf_counter()-started,"success":True}
        if verbose: print(f"[wm6 joint] days={n} transitions={len(y)} QP={result['qp_loss']:.8g}")
        return result


def fit_variances(errors,kernel):
    """Nonnegative variance-matching QP with active_set: tau² + sigma²*K."""
    e=np.asarray(errors,dtype=float).reshape(-1)
    K=np.asarray(kernel,dtype=float).reshape(-1)
    if len(e)!=len(K) or len(e)==0 or not np.isfinite(e).all() or not np.isfinite(K).all() or np.any(K<0):
        raise ValueError("WM6 variance fit requires matching finite errors and nonnegative kernel.")
    y=e*e
    if np.ptp(K)<1e-12: return float(np.mean(y)),0.,False
    X=np.column_stack((np.ones_like(K),K))
    H=2.*X.T@X/len(y)+1e-12*np.eye(2)
    q=-2.*X.T@y/len(y)
    x=np.asarray(active_set.solve(H,q,lb=np.zeros(2),tol=1e-10,max_iter=10000),dtype=float)
    return float(max(0.,x[0])),float(max(0.,x[1])),True
