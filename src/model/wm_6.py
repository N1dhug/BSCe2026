"""WM6: baseline-forced affine flexibility-rate SDE (ARM extension)."""
import numpy as np

from src.model.wm_4 import wm_4
from src.validation.model.wm_4 import wm_4_validation


class wm_6(wm_4):
    """D=B+R, dF=R dt, dR=(theta_F*F+theta_U*U+theta_R*R+theta_B*(2B-1))dt+sigma dW.

    F and U have preferred signed range [-1,1]. The baseline B itself is
    normalized to [0,1], so 2B-1 is a fixed mapping across all fit days.
    Coefficient order: [theta_F, theta_U, theta_R, theta_B].
    """
    @staticmethod
    def coefficients(r, tol=1e-7):
        r = np.asarray(r, dtype=float).reshape(-1)
        if r.shape != (4,) or not np.isfinite(r).all(): raise ValueError("WM6 requires four finite drift coefficients.")
        f, u, damping, baseline = r
        if f >= 0 or damping >= 0 or u > tol or baseline < -tol: raise ValueError("WM6 requires theta_F<0, theta_R<0, theta_U<=0, theta_B>=0.")
        if f+baseline-u > tol: raise ValueError("WM6 preferred equilibrium requires theta_B-theta_U <= -theta_F.")
        if f+u+baseline-damping > tol: raise ValueError("WM6 rate-equilibrium restriction violated.")
        return r

    @staticmethod
    def mapped_baseline(B):
        B = np.asarray(B, dtype=float)
        if not np.isfinite(B).all() or np.any(B < -1e-8) or np.any(B > 1+1e-8):
            raise ValueError("WM6 baseline must lie in [0,1].")
        return 2.0*B-1.0

    @staticmethod
    def drift(F, R, U, B, r):
        f, u, damping, baseline = wm_6.coefficients(r)
        return np.array([R, f*F+u*U+damping*R+baseline*(2*B-1)],dtype=float)

    @staticmethod
    def expected_equilibrium(U, B, r):
        f,u,_,baseline=wm_6.coefficients(r)
        return -(u*np.asarray(U)+baseline*wm_6.mapped_baseline(B))/f

    def _solve_deterministic(self, init_cond, U, PE, Time, O, r, gamma=None, r_0=0.0):
        r=self.coefficients(r)
        if abs(r_0)>1e-12: raise ValueError("WM6 has no free intercept; baseline forcing replaces it.")
        F,R=np.empty(self.t_steps),np.empty(self.t_steps)
        R_dot=np.empty(self.t_steps)
        F[0],R[0]=wm_4_validation.initial_condition(init_cond)
        O=np.asarray(O,dtype=float).reshape(-1)
        U=np.asarray(U,dtype=float).reshape(-1)
        Bmapped=self.mapped_baseline(O)
        if len(O)!=self.t_steps or len(U)!=self.t_steps: raise ValueError("WM6 signal lengths must match t_steps.")
        if np.any(U < -1) or np.any(U > 1): raise ValueError("WM6 price must lie in [-1,1].")
        P_R=np.empty((self.t_steps,4),dtype=float)
        for k in range(self.t_steps):
            P_R[k]=[F[k],U[k],R[k],Bmapped[k]]
            R_dot[k]=float(r@P_R[k])
            if k+1<self.t_steps:
                F[k+1]=F[k]+self.dt*R[k]
                R[k+1]=R[k]+self.dt*R_dot[k]
        I=O+R
        return {"t":self.t.copy(),"F":F,"R":R,"R_dot":R_dot,
                "U":U.copy(),"PE":np.asarray(PE).copy(),"Time":np.asarray(Time).copy(),
                "O":O.copy(),"B_mapped":Bmapped,"I":I,"P_R":P_R,
                "r":r.copy(),"A":np.full(self.t_steps,np.nan)}

    def deterministic_solver(self,init_cond,parameters):
        if not isinstance(parameters,dict) or "r" not in parameters: raise ValueError("WM6 requires parameters['r'].")
        for key in ("U","Time"):
            if not callable(parameters.get(key)): raise ValueError(f"WM6 requires callable '{key}'.")
        values={**parameters,"PE":parameters.get("PE",lambda t:0.),"r":self.coefficients(parameters["r"])}
        U,PE,Time,O=self._sample_signals(values)
        return self._solve_deterministic(init_cond,U,PE,Time,O,values["r"])

    def sensitivity(self,parameters):
        r=self.coefficients(parameters["r"] if isinstance(parameters,dict) else parameters)
        transition=np.array([[1.,self.dt],[self.dt*r[0],1.+self.dt*r[2]]])
        state=np.empty((self.t_steps,2,2))
        state[0]=np.eye(2)
        for k in range(self.t_steps-1): state[k+1]=transition@state[k]
        return {"state":state,"input":state[:,1,:].copy(),"transition":transition}

    def process_variance_kernel(self,state,sample_indices):
        r=self.coefficients(state["r"])
        return super().process_variance_kernel({"r":r[:3]},sample_indices)

    def fit_params_joint_qp(self,input_data,initial_conditions=None,**kwargs):
        from src.params.wm_6 import joint_qp_params
        return joint_qp_params(self).fit_many(input_data,initial_conditions,**kwargs)

    def stochastic_solver(self,init_cond,parameters,sigma=0.0,tau=0.0,seed=None):
        """Euler-Maruyama with independent Gaussian observation noise.

        Returns one sample path, deterministic mean, and exact discrete
        unconditional demand variance for fixed initial state and inputs.
        """
        sigma,tau=float(sigma),float(tau)
        if not np.isfinite([sigma,tau]).all() or min(sigma,tau)<0:
            raise ValueError("WM6 sigma and tau must be finite and nonnegative.")
        deterministic=self.deterministic_solver(init_cond,parameters)
        r=deterministic["r"]
        rng=np.random.default_rng(seed)
        F,R=np.empty(self.t_steps),np.empty(self.t_steps)
        F[0],R[0]=wm_4_validation.initial_condition(init_cond)
        for k in range(self.t_steps-1):
            rate=r[0]*F[k]+r[1]*deterministic["U"][k]+r[2]*R[k]+r[3]*deterministic["B_mapped"][k]
            F[k+1]=F[k]+self.dt*R[k]
            R[k+1]=R[k]+self.dt*rate+sigma*np.sqrt(self.dt)*rng.standard_normal()
        variance_kernel=self.process_variance_kernel({"r":r},np.arange(self.t_steps)).sum(axis=1)
        variance=tau*tau+sigma*sigma*variance_kernel
        observation=deterministic["O"]+R+tau*rng.standard_normal(self.t_steps)
        return {**deterministic,"sample_F":F,"sample_R":R,
                "sample_demand":observation,"mean_demand":deterministic["I"].copy(),
                "variance":variance,"std":np.sqrt(variance),
                "sigma":sigma,"tau":tau}
