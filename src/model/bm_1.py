"""BM1 equations and forward solvers. Control/validation experiments live outside this model."""

import numpy as np
from scipy.integrate import solve_ivp

from src.model.base.bm import bm
from src.solver.finite_volume import finite_volume_1d, finite_volume_1d_cached
from src.validation.model.bm_1 import bm_1_validation


class bm_1(bm):
    """Battery Model 1 with normalized state X in [0, 1].

    dX/dt = (D(X,U,B) - B) / C.
    The separate Ito diffusion coefficient is sigma_X * X * (1 - X).
    A(t) is the tracking target, B(t) the baseline and U(t) the price input.
    Response functions and physical parameters are supplied to each solve.
    """

    def __init__(self, B, A, T, t_steps):
        super().__init__(B, A, T, t_steps)
        self.fokker_planc_reference = finite_volume_1d()

    def _drift(self, t, X, parameters):
        """Net charging rate (demand minus baseline) divided by capacity."""
        U, f, g, C, k, Xi = parameters
        _, Bt, _, _, _, Dt = self.eval(t, X, U, f, g, k, Xi)
        return np.asarray((Dt - Bt) / C, dtype=float)

    @staticmethod
    def _diffusion(X, sigma_X):
        """Return b(X) = sigma_X * X * (1 - X) for scalar or array states.

        This is the Ito coefficient, without a time-step factor or Wiener draw.
        sigma_X has units of inverse square root of the model time unit.
        The deterministic solver does not use this coefficient.
        """
        X, sigma_X = bm_1_validation.diffusion_inputs(X, sigma_X)
        return sigma_X * X * (1.0 - X)

    def deterministic_solver(
        self,
        init_cond,
        parameters,
        *,
        t_eval=None,
        breakpoints=None,
        rtol=0.001,
        atol=1e-06,
        max_step=np.inf,
    ):
        """Integrate the drift ODE; restart at specified price/input discontinuities."""
        if t_eval is None:
            t_eval = self.t.copy()
        init_cond, parameters, t_eval, breakpoints, rtol, atol, max_step = bm_1_validation.deterministic_inputs(
            init_cond,
            parameters,
            t_eval,
            breakpoints,
            self.T,
            rtol,
            atol,
            max_step,
        )
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
            solution = solve_ivp(
                segment_rhs,
                (left, right),
                current_state,
                dense_output=True,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            if not solution.success:
                raise RuntimeError(f'solve_ivp failed on interval [{left}, {right}]: {solution.message}')
            if final_interval:
                output_mask = (t_eval >= left) & (t_eval <= right)
            else:
                output_mask = (t_eval >= left) & (t_eval < right)
            if np.any(output_mask):
                Xt[output_mask] = solution.sol(t_eval[output_mask])[0]
            current_state = solution.y[:, -1]
        Ut, Bt, fXt, gUt, delta, Dt = self.eval(
            t_eval,
            Xt,
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[4],
            parameters[5],
        )
        return (t_eval.copy(), Xt, Ut, Bt, fXt, gUt, delta, Dt)

    def diffusion_coefficient(self, X, sigma_x):
        """BM1 process amplitude used by the transferred forecast pipeline."""
        return self._diffusion(X, sigma_x)

    def _fv_evolution_fields(self, parameters, sigma_x, t_eval, breakpoints, max_step):
        """Build the shared FV fields and exact cache keys for fixed parameters."""
        def drift(t, X):
            return self._drift(t, X, parameters)

        def diffusion(t, X):
            return self.diffusion_coefficient(X, sigma_x)
        options = {}
        # In the standard BM1 equations time enters only through B(t), U(t).
        # f/g and physical parameters are fixed pure inputs within this solve.
        # Overrides keep the ordinary callback path; no stationarity is guessed.
        standard = all(getattr(getattr(self, name), '__func__', None) is implementation for name, implementation in (('_drift', bm_1._drift), ('eval', bm.eval), ('l', bm.l), ('D', bm.D), ('diffusion_coefficient', bm_1.diffusion_coefficient)))
        if isinstance(self.fokker_planc_reference, finite_volume_1d_cached) and standard:
            if self.fokker_planc_reference.vectorized_inputs:
                # Opt-in, like F3's blocked evaluator: both input callbacks
                # must obey ordinary NumPy broadcasting. No fallback hides an
                # invalid field. The grid is identical to the solver's grid.
                grid, _ = self.fokker_planc_reference.build_time_grid(t_eval, breakpoints, max_step)
                left = grid[:-1]
                baseline = np.broadcast_to(self.B(left), left.shape)
                price = np.broadcast_to(parameters[0](left), left.shape)
                keys = dict(zip(left, zip(map(float, baseline), map(float, price))))
                coefficient_key = keys.__getitem__
            else:
                def coefficient_key(time):
                    return (float(self.B(time)), float(parameters[0](time)))
            options['coefficient_key'] = coefficient_key
        return drift, diffusion, options

    def fokker_planc_reference_solver(self, initial_mass, parameters, sigma_x, *, t_eval=None, breakpoints=None, max_step=np.inf,):
        """Solve stochastic bm_1 with the pure finite-volume Fokker-Planck reference.

        The state distribution is represented by cell probability masses on
        [0, 1]. The supplied initial_mass must already be normalized; this
        reference does not clip or renormalize probability during evolution.
        """
        t_eval, parameters, sigma_x = bm_1_validation.reference_solver_inputs(self.t, t_eval, parameters, sigma_x)

        drift, diffusion, options = self._fv_evolution_fields(parameters, sigma_x, t_eval, breakpoints, max_step,)
        result = self.fokker_planc_reference.solve(t_eval, initial_mass, drift, diffusion, bounds=(0.0, 1.0), breakpoints=breakpoints, max_step=max_step, **options,)
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
        return (t.copy(), x.copy(), result['density'].copy(), cell_mass.copy(), EXt, VarXt, Ut, Bt, EDt, result['diagnostics'].copy(),)

