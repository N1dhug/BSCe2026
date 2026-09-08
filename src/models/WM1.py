from src.models.WM import WM
import numpy as np
from copy import deepcopy
from models.Simulate import Simulate
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


class WM1(WM):
    def __init__(self, T, t_steps):
        self.simulate = Simulate()
        self.T        = T
        self.t_steps  = t_steps
        self.t        = np.linspace(T[0], T[1], t_steps)
        self.dt       = self.t[1]-self.t[0]

    def It(self, Vt, Pt, Bt, alpha, beta, gamma):
        return alpha * Bt + beta*(1- Vt) + gamma*(1 - Pt)

    def Ot(self, Vt, Pt, Bt, It, delta, epsilon, zeta, eta):
        return delta * Bt + epsilon * Vt + zeta*It + eta*(1-Pt)
    
    def At(self, Pt, It, PEt, Vt, iota, kappa, mu, nu):
        return iota*(1-Pt) + kappa*It + mu*Vt + nu*(1 - PEt)

    def dVdt(self, t, state, m_params, opt_params, obj_params):
        V_prev, dV_prev = state
        P_func, B_func, alpha, beta, gamma,delta, epsilon, zeta, eta, theta = m_params    

        Pt = P_func(t)
        Bt = B_func(t)

        # inflow & outflow at the current step
        I_t = self.It(V_prev, Pt, Bt, alpha, beta, gamma)
        O_t = self.Ot(V_prev, Pt, Bt, I_t, delta, epsilon, zeta, eta)

        dV = I_t - O_t + theta * (dV_prev)  
        V =  V_prev + dV*self.dt             
        return [V, dV]

    def deterministic_solver(self, init_cond, parameters):
        m_params, opt_params, obj_params = parameters
        V  = np.empty(self.t_steps, dtype=float)
        dV = np.empty(self.t_steps, dtype=float)

        # k = 0 : store the supplied initial state
        V_prev, dV_prev = init_cond
        V[0], dV[0] = V_prev, dV_prev

        # March forward on the grid
        for k in range(1, self.t_steps):
            t_prev = self.t[k-1]
            V[k], dV[k] = self.dVdt(t_prev, (V_prev, dV_prev), m_params, opt_params, obj_params)
            V_prev, dV_prev = V[k], dV[k]
        return V

    def phi(self, P, init_cond, parameters):

        m_params, opt_params, obj_params = parameters

        P_func = self.simulate.price((self.t, P))
        parameters[0][0] = P_func

        V = self.deterministic_solver(init_cond, parameters)
        I = self.It(V, P_func(self.t), m_params[1](self.t), m_params[2], m_params[3], m_params[4])
        A = self.At(P_func(self.t), I, obj_params[0](self.t), V, opt_params[0], opt_params[1], opt_params[2], opt_params[3])
        return np.sum((A - I)**2)
    
    def optimize(self, P0, init_cond, parameters):
        bounds = [(0.0, 1.0)] * len(P0)
        solution = minimize(lambda P: self.phi(P, init_cond, parameters), P0, method="L-BFGS-B", bounds=bounds)
        params_final      = deepcopy(parameters)
        params_final[0][0]= self.simulate.price((self.t, solution.x))
        V_opt             = self.deterministic_solver(init_cond, params_final)

        return self.t, V_opt, solution.x, solution

