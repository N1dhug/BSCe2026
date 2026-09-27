from src.model.base.bm import bm
from src.method.euler_maruyama import euler_maruyama
from src.method.fokker_planc import bm_1_fokker_planc
from src.validation.model.bm_1 import bm_1_validation
from src.tools.input_generator.input_builder import input_builder
from src.method.objective import tracking_error
from src.solver.finite_volume import finite_volume_1d
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class bm_1(bm):

    def __init__(self, B, A, T, t_steps):
        super().__init__(B, A, T, t_steps)
        self.euler_maruyama = euler_maruyama()
        self.fokker_planc = bm_1_fokker_planc()
        self.fokker_planc_reference = finite_volume_1d()

    def _evaluate_tracking_objective(self, interval_prices, interval_edges, parameters, init_cond, t_eval, rtol, atol,
        max_step):
        simulated_U = input_builder.price_intervals((interval_edges, interval_prices))
        simulated_parameters = list(parameters)
        simulated_parameters[0] = simulated_U
        t, _, _, _, _, _, _, Dt = self.deterministic_solver(init_cond, simulated_parameters, t_eval=t_eval,
            breakpoints=interval_edges[1:-1], rtol=rtol, atol=atol, max_step=max_step)
        return tracking_error(t, Dt, self.A(t))

    def _drift(self, t, X, parameters):
        U, f, g, C, k, Xi = parameters
        _, Bt, _, _, _, Dt = self.eval(t, X, U, f, g, k, Xi)
        return np.asarray((Dt - Bt) / C, dtype=float)

    def euler_maruyama_solver(self, init_cond, parameters, sigma_x, n_paths=1, seed=None, boundary='reflect', *,
        t_eval=None, breakpoints=None, max_step=np.inf, brownian_increments=None, return_diagnostics=False):
        """Solve stochastic bm_1 with controlled Euler-Maruyama integration.

        ``t_eval`` is the returned output grid. ``breakpoints`` and
        ``max_step`` define additional internal integration points, so an EM
        step never crosses a known input discontinuity. ``brownian_increments``
        may be supplied explicitly for coupled convergence experiments.
        """
        if t_eval is None:
            t_eval = self.t.copy()
        init_cond, parameters, sigma_x = bm_1_validation.stochastic_path_inputs(init_cond, parameters, sigma_x)
        t_eval, breakpoints, max_step = bm_1_validation.stochastic_time_inputs(t_eval, breakpoints, self.T, max_step)
        integration_t, output_indices = self.euler_maruyama.build_time_grid(t_eval, breakpoints=breakpoints,
            max_step=max_step)

        def drift(t, X):
            return self._drift(t, X, parameters)

        def diffusion(t, X):
            return self.diffusion_coefficient(X, sigma_x)
        result = self.euler_maruyama.solve_paths(t=integration_t, init_cond=init_cond, drift=drift,
            diffusion=diffusion, n_paths=n_paths, seed=seed, boundary=boundary,
            brownian_increments=brownian_increments, return_diagnostics=return_diagnostics)
        if return_diagnostics:
            integration_paths, diagnostics = result
        else:
            integration_paths = result
        paths = integration_paths[:, output_indices]
        Xt = paths[0] if paths.shape[0] == 1 else paths
        Ut = parameters[0](t_eval)
        Bt = self.B(t_eval)
        fXt = parameters[1](Xt)
        gUt = parameters[2](Ut)
        dt = self.l(fXt, gUt, parameters[4])
        Dt = self.D(Bt, parameters[5], dt)
        output = (t_eval.copy(), Xt, Ut, Bt, fXt, gUt, dt, Dt)
        if not return_diagnostics:
            return output
        diagnostics = diagnostics.copy()
        diagnostics.update({'output_t': t_eval.copy(), 'output_indices': output_indices.copy(),
            'breakpoints': breakpoints.copy(), 'requested_max_step': max_step})
        return (*output, diagnostics)

    def fokker_planc_solver(self, init_cond, parameters, sigma_x, degree=12, x_steps=201, cfl=0.45):
        init_cond, parameters, sigma_x = bm_1_validation.stochastic_inputs(init_cond, parameters, sigma_x)

        def drift(t, X):
            return self._drift(t, X, parameters)

        def diffusion(t, X):
            return self.diffusion_coefficient(X, sigma_x)
        x, p, coefficients = self.fokker_planc.solve(t=self.t, init_cond=init_cond, drift=drift, diffusion=diffusion,
            degree=degree, x_steps=x_steps, cfl=cfl)
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
        return (self.t.copy(), x, p, coefficients, EXt, Ut, Bt, EDt)

    def fokker_planc_reference_solver(self, initial_mass, parameters, sigma_x, *, t_eval=None, breakpoints=None,
        max_step=np.inf):
        """Solve stochastic bm_1 with the pure finite-volume Fokker-Planck reference.

        The state distribution is represented by cell probability masses on
        [0, 1]. The supplied initial_mass must already be normalized; this
        reference does not clip or renormalize probability during evolution.
        """
        if t_eval is None:
            t_eval = self.t.copy()
        _, parameters, sigma_x = bm_1_validation.stochastic_inputs([0.5], parameters, sigma_x)

        def drift(t, X):
            return self._drift(t, X, parameters)

        def diffusion(t, X):
            return self.diffusion_coefficient(X, sigma_x)
        result = self.fokker_planc_reference.solve(t_eval, initial_mass, drift, diffusion, bounds=(0.0, 1.0),
            breakpoints=breakpoints, max_step=max_step)
        x = result['x_grid']
        cell_mass = result['cell_mass']
        EXt = np.sum(cell_mass * x[None, :], axis=1)
        EX2t = np.sum(cell_mass * np.square(x[None, :]), axis=1)
        VarXt = np.maximum(EX2t - np.square(EXt), 0.0)
        t = result['t']
        Ut = np.asarray(parameters[0](t), dtype=float)
        Bt = np.asarray(self.B(t), dtype=float)
        EDt = np.empty(t.size, dtype=float)
        for i, t_i in enumerate(t):
            Ut_i = parameters[0](t_i)
            Bt_i = self.B(t_i)
            fXt_i = parameters[1](x)
            gUt_i = parameters[2](Ut_i)
            delta_i = self.l(fXt_i, gUt_i, parameters[4])
            Dt_i = self.D(Bt_i, parameters[5], delta_i)
            EDt[i] = np.sum(Dt_i * cell_mass[i])
        return (t.copy(), x.copy(), result['density'].copy(), cell_mass.copy(), EXt, VarXt, Ut, Bt, EDt,
            result['diagnostics'].copy())

    def deterministic_solver(self, init_cond, parameters, *, t_eval=None, breakpoints=None, rtol=0.001, atol=1e-06,
        max_step=np.inf):
        """Solve deterministic bm_1, restarting at known input discontinuities.

        ``t_eval`` controls only the returned output grid. ``breakpoints`` are
        internal integration boundaries and need not occur in ``t_eval``.
        Existing calls without optional arguments retain ``self.t`` and
        SciPy's default tolerances.
        """
        if t_eval is None:
            t_eval = self.t.copy()
        init_cond, parameters, t_eval, breakpoints, rtol, atol, max_step = bm_1_validation.deterministic_inputs(init_cond,
            parameters, t_eval, breakpoints, self.T, rtol, atol, max_step)
        interval_edges = np.concatenate(([self.T[0]], breakpoints, [self.T[1]]))
        Xt = np.empty(t_eval.size, dtype=float)
        current_state = init_cond
        for index, (left, right) in enumerate(zip(interval_edges[:-1], interval_edges[1:])):
            first_interval = index == 0
            final_interval = index == len(interval_edges) - 2

            def segment_rhs(t, state, left=left, right=right, first=first_interval, final=final_interval):
                if not first and t <= left:
                    model_time = np.nextafter(left, right)
                elif not final and t >= right:
                    model_time = np.nextafter(right, left)
                else:
                    model_time = t
                return self.dXdt(model_time, state, *parameters)
            solution = solve_ivp(segment_rhs, (left, right), current_state, dense_output=True, rtol=rtol, atol=atol,
                max_step=max_step)
            if not solution.success:
                raise RuntimeError(f'solve_ivp failed on interval [{left}, {right}]: {solution.message}')
            if final_interval:
                output_mask = (t_eval >= left) & (t_eval <= right)
            else:
                output_mask = (t_eval >= left) & (t_eval < right)
            if np.any(output_mask):
                Xt[output_mask] = solution.sol(t_eval[output_mask])[0]
            current_state = solution.y[:, -1]
        Ut, Bt, fXt, gUt, dt, Dt = self.eval(t_eval, Xt, parameters[0], parameters[1], parameters[2], parameters[4],
            parameters[5])
        return (t_eval.copy(), Xt, Ut, Bt, fXt, gUt, dt, Dt)

    def evaluate_tracking_objective(self, interval_prices, init_cond, parameters, *, interval_edges, t_eval=None,
        rtol=1e-06, atol=1e-08, max_step=np.inf):
        """Evaluate the deterministic demand-tracking objective for prices."""
        if t_eval is None:
            t_eval = self.t.copy()
        interval_prices, interval_edges, init_cond, parameters, t_eval, rtol, atol, max_step = bm_1_validation.optimize_inputs(interval_prices,
            interval_edges, init_cond, parameters, t_eval, self.T, rtol, atol, max_step)
        return self._evaluate_tracking_objective(interval_prices, interval_edges, parameters, init_cond, t_eval, rtol,
            atol, max_step)

    def optimize(self, initial_u, init_cond, parameters, *, interval_edges, t_eval=None, rtol=1e-06, atol=1e-08,
        max_step=np.inf, options=None):
        """Optimize one normalized price value per supplied time interval."""
        if t_eval is None:
            t_eval = self.t.copy()
        initial_u, interval_edges, init_cond, parameters, t_eval, rtol, atol, max_step = bm_1_validation.optimize_inputs(initial_u,
            interval_edges, init_cond, parameters, t_eval, self.T, rtol, atol, max_step)
        bounds = [(0.0, 1.0) for _ in range(initial_u.size)]

        def phi(interval_prices):
            return self._evaluate_tracking_objective(interval_prices, interval_edges, parameters, init_cond, t_eval,
                rtol, atol, max_step)
        return minimize(phi, initial_u, method='L-BFGS-B', bounds=bounds, options=options)
