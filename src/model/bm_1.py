"""BM1 model dynamics and Fokker-Planck density solvers."""

import numpy as np

from src.model.base.bm import bm
from src.method.fokker_planc import bm_1_fokker_planc
from src.validation.model.bm_1 import bm_1_validation
from src.solver.finite_volume import finite_volume_1d, finite_volume_1d_cached


class bm_1(bm):
    """Battery Model 1 with normalized state X in [0, 1].

    dX = (D(X,U,B) - B) / C dt + sigma_x X(1-X) dW.
    A(t) is the tracking target, B(t) the baseline and U(t) the price input.
    Response functions and physical parameters are supplied to each solve.
    """

    def __init__(self, B, A, T, t_steps):
        super().__init__(B, A, T, t_steps)
        self.fokker_planc = bm_1_fokker_planc()
        self.fokker_planc_reference = finite_volume_1d()

    def _drift(self, t, X, parameters):
        """Net charging rate (demand minus baseline) divided by capacity."""
        U, f, g, C, k, Xi = parameters
        _, Bt, _, _, _, Dt = self.eval(t, X, U, f, g, k, Xi)
        return np.asarray((Dt - Bt) / C, dtype=float)

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
        standard = all(getattr(getattr(self, name), '__func__', None) is implementation for name, implementation in (('_drift', bm_1._drift), ('eval', bm.eval), ('l', bm.l), ('D', bm.D), ('diffusion_coefficient', bm.diffusion_coefficient)))
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
        if t_eval is None: t_eval = self.t.copy()
        _, parameters, sigma_x = bm_1_validation.stochastic_inputs([0.5], parameters, sigma_x)

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

    def fokker_planc_solver(self, init_cond, parameters, sigma_x, degree=12, x_steps=201, cfl=0.45):
        """Return a dynamic Beta-mixture approximation and its demand/state moments."""
        init_cond, parameters, sigma_x = bm_1_validation.stochastic_inputs(init_cond, parameters, sigma_x)

        def drift(t, X):
            return self._drift(t, X, parameters)

        def diffusion(t, X):
            return self.diffusion_coefficient(X, sigma_x)
        x, p, coefficients = self.fokker_planc.solve(t=self.t, init_cond=init_cond, drift=drift, diffusion=diffusion, degree=degree, x_steps=x_steps, cfl=cfl,)
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
