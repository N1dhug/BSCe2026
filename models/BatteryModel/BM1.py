from models.BatteryModel.BatteryModel import BatteryModel
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import simpson as simps
from scipy.optimize import minimize


class BM1(BatteryModel):
    def __init__(self, B, A, T, t_steps):
        super().__init__(B, A, T, t_steps)

    @staticmethod
    def _apply_boundary(values, boundary):
        values = np.asarray(values, dtype=float)
        if boundary == "reflect":
            reflected = np.mod(values, 2.0)
            return np.where(reflected <= 1.0, reflected, 2.0 - reflected)
        elif boundary == "clip":
            return np.clip(values, 0.0, 1.0)
        elif boundary == "none":
            return values

        raise ValueError("boundary must be 'reflect', 'clip', or 'none'.")

    def __phi(self, U, parameters, init_cond):
        simulated_U = self.simulate.price((self.t, U))
        parameters[0] = simulated_U
        t, _, _, _, _, _, _, Dt = self.deterministic_solver(init_cond, parameters)
        At = self.A(t)
        Phi = np.square(Dt - At)
        phi = simps(Phi, t)
        return phi

    def _drift(self, t, X, parameters):
        U, f, g, C, k, Xi = parameters
        _, Bt, _, _, _, Dt = self.eval(t, X, U, f, g, k, Xi)
        return np.asarray((Dt - Bt) / C, dtype=float)

    def _euler_maruyama_paths(self, init_cond, parameters, sigma_x, n_paths=1, seed=None, boundary="reflect"):
        x0 = init_cond
        n_paths = n_paths
        rng = np.random.default_rng(seed)

        paths = np.empty((n_paths, self.t_steps), dtype=float)
        paths[:, 0] = x0

        for i in range(self.t_steps - 1):
            t_i = float(self.t[i])
            h = float(self.t[i + 1] - self.t[i])
            X_i = paths[:, i]

            drift = self._drift(t_i, X_i, parameters)
            diffusion = np.asarray(self.diffusion_coefficient(X_i, sigma_x),dtype=float,)
            dW = np.sqrt(h) * rng.standard_normal(n_paths)

            proposal = X_i + drift * h + diffusion * dW
            paths[:, i + 1] = self._apply_boundary(proposal, boundary)

        return paths

    def euler_maruyama_solver(self, init_cond, parameters, sigma_x, n_paths=1, seed=None,boundary=" reflect"):
        paths = self._euler_maruyama_paths(init_cond, parameters, sigma_x, n_paths, seed, boundary)

        Xt = paths[0] if n_paths == 1 else paths
        Ut = parameters[0](self.t)
        Bt = self.B(self.t)
        fXt = parameters[1](Xt)
        gUt = parameters[2](Ut)
        dt = self.l(fXt, gUt, parameters[4])
        Dt = self.D(Bt, parameters[5], dt)

        return self.t.copy(), Xt, Ut, Bt, fXt, gUt, dt, Dt

    def deterministic_solver(self, init_cond, parameters):
        solution = solve_ivp(
        self.dXdt,self.T,init_cond,args=parameters, t_eval=self.t,)
        Xt = solution.y[0]
        Ut, Bt, fXt, gUt, dt, Dt = self.eval(solution.t,Xt,parameters[0],parameters[1], parameters[2],parameters[4],parameters[5],)
        return solution.t, Xt, Ut, Bt, fXt, gUt, dt, Dt

    def optimize(self, initial_u, init_cond, parameters):
        bounds = [(0, 1) for _ in range(len(initial_u))]
        phi = lambda U: self.__phi(U, parameters, init_cond)
        solution = minimize(phi, initial_u, bounds=bounds)
        return solution
