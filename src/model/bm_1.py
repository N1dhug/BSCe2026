"""BM1 equations and forward solvers. Control/validation experiments live outside this model."""

import numpy as np
from scipy.integrate import solve_ivp

from src.model.base.bm import bm
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
