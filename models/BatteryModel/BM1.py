from models.BatteryModel.BatteryModel import BatteryModel
import numpy as np
import torch 
import torchsde
from scipy.integrate import solve_ivp
from scipy.integrate import simpson as simps
from scipy.optimize import minimize

class BM1(BatteryModel):
    def __init__(self, B, A, T, t_steps) :
        super().__init__(B, A, T, t_steps)

    def __phi(self, U, parameters, init_cond):
        simulated_U = self.simulate.price((self.t, U))
        parameters[0] = simulated_U
        t, _, _, _, _, _, _, Dt = self.deterministic_solver(init_cond, parameters)
        At = self.A(t)
        Phi = np.square(Dt - At)
        phi = simps(Phi, t)
        return phi
    
    def deterministic_solver(self, init_cond, parameters):
        solution = solve_ivp(self.dXdt,
                  self.T,
                  init_cond,
                  args=parameters,
                  t_eval=self.t
        )
        Xt = solution.y[0]
        Ut, Bt, fXt, gUt, dt, Dt = self.eval(solution.t, solution.y[0], parameters[0], parameters[1], parameters[2], parameters[4], parameters[5])
        return solution.t, Xt, Ut, Bt, fXt, gUt, dt, Dt
    
    def optimize(self, initial_u, init_cond, parameters):
        bounds = [(0, 1) for _ in range(len(initial_u))] 
        phi = lambda U: self.__phi(U, parameters, init_cond)
        solution = minimize(phi, initial_u, bounds=bounds)
        return solution
    
    def stochastic_solver(self, init_cond, parameters, sigma_x):
        t = torch.linspace(self.T[0], self.T[1], self.t_steps, dtype=torch.float32)

        # Create lambda wrappers that supply the required parameters to the drift and diffusion methods.
        drift = lambda t, X: self.dXdt(t, X, *parameters, stochastic = True)
        diffusion = lambda t, X: self.XtOx_dWt(t, X, sigma_x)
        X0 = torch.tensor([[init_cond[0]]], dtype=torch.float32)

        
        # Create a wrapper object with the required `noise_type` attribute
        sde = lambda: None  
        setattr(sde, "f", drift)
        setattr(sde, "g", diffusion)
        setattr(sde, "noise_type", "scalar") 
        setattr(sde, "sde_type", "ito")         

        # Convert initial condition to tensor  
        solution = torchsde.sdeint(sde, X0, t, method="srk")
        Xt = solution.squeeze().numpy()

        t = t.numpy()
        Ut, Bt, fXt, gUt, dt, Dt = self.eval(t, Xt, parameters[0], parameters[1], parameters[2], parameters[4], parameters[5])

        return t, Xt, Ut, Bt, fXt, gUt, dt, Dt




