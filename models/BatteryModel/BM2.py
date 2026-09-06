from .BatteryModel import BatteryModel
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import simpson as simps
from scipy.optimize import minimize
import torch 
import torchsde

class BM2(BatteryModel):
    def __init__(self, B, A, T, t_steps) :
        super().__init__(B, A, T, t_steps)  # Initialize the base class

    def __ode(self, t, state, U, f, g, C, Xi, k, y, m, stochastic = False):
        if stochastic:
            state = state[0]
        x2, x1 = state
        dx1dt = (self.dXdt(t, [x2], U, f, g, C, k, Xi, stochastic)[0] - y*x1)/m
        dx2dt = x1
        if stochastic:
            dx1dt = torch.tensor([dx1dt], dtype=torch.float32) if not isinstance(dx1dt, torch.Tensor) else dx1dt.reshape(-1)
            dx2dt = torch.tensor([dx2dt], dtype=torch.float32) if not isinstance(dx2dt, torch.Tensor) else dx2dt.reshape(-1)
            return torch.cat([dx1dt, dx2dt], dim=0).unsqueeze(0)
        return [dx2dt, dx1dt]
    
    def __phi(self, U, init_cond, parameters):
        simulated_U = self.simulate.price((self.t, U))
        parameters[0] = simulated_U
        t, _, _, _, _, _, _, Dt = self.deterministic_solver(init_cond, parameters)
        At = self.A(t)
        Phi = np.square(Dt - At)
        phi = simps(Phi, t)
        return phi
    
    def deterministic_solver(self, init_cond, parameters):
        solution = solve_ivp(self.__ode,
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
        phi = lambda U: self.__phi(U, init_cond, parameters)
        solution = minimize(phi, initial_u, bounds = bounds)
        return solution
    

    def diffusion(self, t, X, sigma_x):
        # X is expected to be a tensor with two elements.
        noise_x2 = self.XtOx_dWt(t, X[:,0:1], sigma_x)
        noise_x2 = noise_x2.squeeze(-1)  # Remove the last dimension to get the correct shape.
        noise_x1 = torch.zeros_like(noise_x2)
        # Return a stacked tensor; stacking ensures the correct shape.
        return torch.stack([noise_x2, noise_x1], dim=1)
    
    def stochastic_solver(self, init_cond, parameters, sigma_x):
        t = torch.linspace(self.T[0], self.T[1], self.t_steps, dtype=torch.float32)

        # Create lambda wrappers that supply the required parameters to the drift and diffusion methods.

        drift = lambda t, X: self.__ode(t, X, *parameters, stochastic = True)
        diffusion = lambda t, X: self.diffusion(t, X, sigma_x)
        X0 = torch.tensor(init_cond, dtype=torch.float32).unsqueeze(0)
        
        # Create a wrapper object with the required `noise_type` attribute
        sde = lambda: None  
        setattr(sde, "f", drift)
        setattr(sde, "g", diffusion)
        setattr(sde, "noise_type", "scalar") 
        setattr(sde, "sde_type", "ito")         

        # Convert initial condition to tensor  
        solution = torchsde.sdeint(sde, X0, t, method="srk")
        Xt = solution.squeeze().numpy()

        Xt = Xt[:, 0]
        t = t.numpy()
        Ut, Bt, fXt, gUt, dt, Dt = self.eval(t, Xt, parameters[0], parameters[1], parameters[2], parameters[4], parameters[5])

        return t, Xt, Ut, Bt, fXt, gUt, dt, Dt
    

