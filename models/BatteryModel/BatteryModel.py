from models.Simulate import Simulate
import numpy as np
import torch 

class BatteryModel:
    # Constructor
    def __init__(self, B, A, T, t_steps):
        self.simulate = Simulate()
        self.B = B
        self.A = A
        self.T = T
        self.t_steps =  t_steps
        self.dt = (self.T[1]-self.T[0])/t_steps
        self.t = np.linspace(T[0], T[1], t_steps)

    #Internal functions (private)
    #Eval helper function
    def eval(self, t, Xt, U, f, g, k, Xi):
        Ut = U(t)
        Bt = self.B(t)
        fXt = f(Xt)
        gUt = g(Ut)
        dt = self.l(fXt, gUt, k)
        Dt = self.D(Bt, Xi, dt)
        return Ut, Bt, fXt, gUt, dt, Dt 
    
    #Model functions:
    #Logistic function that outputs specific value of L(f(Xt; A) + g(u;B); k) (Eq. 2)
    def l(self, fXt, gUt, k):
        return 2 / (1 + np.exp(-k * (fXt+gUt))) - 1

    #Demand based on time function (Eq. 3)
    def D(self, Bt, Xi, dt):
        return Bt + dt*Xi*((dt >= 0)*(1 - Bt) + (dt < 0)*(Bt))
    
    #dXdt pseudo-overloaded to output either deterministic or stochastic version (Eq. 1)
    def dXdt(self, t, state, U, f, g, C, k, Xi, stochastic = False):
        Xt = state[0]
        if(stochastic):
            X = Xt[..., 0].item() 
            return torch.tensor([[self.dXdt(t, [X], U, f, g, C, k, Xi)[0]]], dtype=torch.float32)
        
        _, Bt, _, _, _, Dt = self.eval(t, Xt, U, f, g, k, Xi)
        dXdt = (Dt - Bt) / C
        return [dXdt]
    
    #Stochastic part of (Eq. 1)
    def XtOx_dWt(self, t, X, sigma_x):
        diff = sigma_x * X * (1 - X)
        return diff.unsqueeze(-1)
    
    #Abstract funcctions
    def ode(self):
        raise NotImplementedError("Implement the ode in specific subclass") 
    
    def phi(self):
        raise NotImplementedError("Implement the minimaztion function in specific subclass")    
           
    def deterministic_solver(self):
        raise NotImplementedError("Implement the deterministic solver for the specific subclass")    

    #Stochastic solver (cannot handle multiple versions yet)
    def stochastic_solver(self):
        raise NotImplementedError("Implement the stochastic solver for the specific subclass")    
    
    def optimize(self):
        raise NotImplementedError("Implement the optimizer for the specific subclass")    

