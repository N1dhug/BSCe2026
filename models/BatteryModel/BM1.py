from models.BatteryModel.BatteryModel import BatteryModel
from models.Solvers.EulerMaruyama import EulerMaruyama
from models.Solvers.FokkerPlanc import FokkerPlanc
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import simpson as simps
from scipy.optimize import minimize


class BM1(BatteryModel):
    def __init__(self, B, A, T, t_steps):
        super().__init__(B, A, T, t_steps)
        self.euler_maruyama = EulerMaruyama()
        self.fokker_planc = FokkerPlanc()

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

    def euler_maruyama_solver(
        self,
        init_cond,
        parameters,
        sigma_x,
        n_paths=1,
        seed=None,
        boundary="reflect",
    ):
        def drift(t, X):
            return self._drift(t, X, parameters)

        def diffusion(t, X):
            return self.diffusion_coefficient(X, sigma_x)

        paths = self.euler_maruyama.solve_paths(
            t=self.t,
            init_cond=init_cond,
            drift=drift,
            diffusion=diffusion,
            n_paths=n_paths,
            seed=seed,
            boundary=boundary,
        )

        Xt = paths[0] if n_paths == 1 else paths
        Ut = parameters[0](self.t)
        Bt = self.B(self.t)
        fXt = parameters[1](Xt)
        gUt = parameters[2](Ut)
        dt = self.l(fXt, gUt, parameters[4])
        Dt = self.D(Bt, parameters[5], dt)

        return self.t.copy(), Xt, Ut, Bt, fXt, gUt, dt, Dt

    def fokker_planc_solver(
        self,
        init_cond,
        parameters,
        sigma_x,
        degree=12,
        x_steps=201,
        cfl=0.45,
        max_substeps=100000,
    ):
        """Solve the BM1 probability density with a Beta-mixture Fokker-Planck model.

        Returns
        -------
        t : ndarray
            BM1 output time grid.
        x : ndarray
            State-of-charge grid on (0, 1).
        p : ndarray
            Probability density p(t, x), shape (len(t), x_steps).
        coefficients : ndarray
            Beta-mixture coefficients, shape (len(t), degree + 1).
        EXt : ndarray
            Expected state of charge E[X_t].
        Ut : ndarray
            Price signal evaluated on the output time grid.
        Bt : ndarray
            Baseline demand evaluated on the output time grid.
        EDt : ndarray
            Expected demand E[D_t].
        """
        def drift(t, X):
            return self._drift(t, X, parameters)

        def diffusion(t, X):
            return self.diffusion_coefficient(X, sigma_x)

        x, p, coefficients = self.fokker_planc.solve(
            t=self.t,
            init_cond=init_cond,
            drift=drift,
            diffusion=diffusion,
            degree=degree,
            x_steps=x_steps,
            cfl=cfl,
            max_substeps=max_substeps,
        )

        dx = 1.0 / len(x)
        EXt = np.sum(p * x[None, :], axis=1) * dx
        Ut = np.asarray(parameters[0](self.t), dtype=float)
        Bt = np.asarray(self.B(self.t), dtype=float)
        EDt = np.empty(len(self.t), dtype=float)

        for i, t_i in enumerate(self.t):
            Ut_i = parameters[0](t_i)
            Bt_i = self.B(t_i)
            fXt_i = parameters[1](x)
            gUt_i = parameters[2](Ut_i)
            delta_i = self.l(fXt_i, gUt_i, parameters[4])
            Dt_i = self.D(Bt_i, parameters[5], delta_i)
            EDt[i] = np.sum(Dt_i * p[i]) * dx

        return self.t.copy(), x, p, coefficients, EXt, Ut, Bt, EDt

    def deterministic_solver(self, init_cond, parameters):
        solution = solve_ivp(
            self.dXdt,
            self.T,
            init_cond,
            args=parameters,
            t_eval=self.t,
        )
        Xt = solution.y[0]
        Ut, Bt, fXt, gUt, dt, Dt = self.eval(
            solution.t,
            Xt,
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[4],
            parameters[5],
        )
        return solution.t, Xt, Ut, Bt, fXt, gUt, dt, Dt

    def optimize(self, initial_u, init_cond, parameters):
        bounds = [(0, 1) for _ in range(len(initial_u))]
        phi = lambda U: self.__phi(U, parameters, init_cond)
        solution = minimize(phi, initial_u, bounds=bounds)
        return solution
