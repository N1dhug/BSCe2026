from src.model.base.bm import BM
from src.method.euler_maruyama import EulerMaruyama
from src.method.fokker_planc import BM1FokkerPlanc
from src.validation.model.bm_1 import BM1Validation
from src.exe.tool.InputBuilder import InputBuilder
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import simpson as simps
from scipy.optimize import minimize

class BM1(BM):
    def __init__(self, B, A, T, t_steps):
        # Initialize variables
        super().__init__(B, A, T, t_steps)
        self.euler_maruyama = EulerMaruyama()
        self.fokker_planc = BM1FokkerPlanc()

    def __phi(
        self,
        interval_prices,
        interval_edges,
        parameters,
        init_cond,
        t_eval,
        rtol,
        atol,
        max_step,
    ):
        # Build one right-continuous price value per decision interval.
        simulated_U = InputBuilder.price_intervals(
            (interval_edges, interval_prices)
        )
        simulated_parameters = list(parameters)
        simulated_parameters[0] = simulated_U

        t, _, _, _, _, _, _, Dt = self.deterministic_solver(
            init_cond,
            simulated_parameters,
            t_eval=t_eval,
            breakpoints=interval_edges[1:-1],
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        At = np.asarray(self.A(t), dtype=float)
        try:
            At = np.broadcast_to(At, Dt.shape)
        except ValueError as exc:
            raise ValueError(
                "A(t) must return one target value per objective-grid point."
            ) from exc
        if not np.all(np.isfinite(At)):
            raise ValueError("A(t) must return finite target values.")

        return float(simps(np.square(Dt - At), x=t))

    def _drift(self, t, X, parameters):
        # Evaluate drift
        U, f, g, C, k, Xi = parameters
        _, Bt, _, _, _, Dt = self.eval(t, X, U, f, g, k, Xi)
        return np.asarray((Dt - Bt) / C, dtype=float)

    def euler_maruyama_solver(self, init_cond, parameters, sigma_x, n_paths=1, seed=None, boundary="reflect"):
        # Solve stochastic model
        init_cond, parameters, sigma_x = BM1Validation.stochastic_inputs(init_cond, parameters, sigma_x)

        def drift(t, X):
            # Evaluate drift
            return self._drift(t, X, parameters)

        def diffusion(t, X):
            # Evaluate diffusion
            return self.diffusion_coefficient(X, sigma_x)

        paths = self.euler_maruyama.solve_paths(t=self.t,init_cond=init_cond,drift=drift,diffusion=diffusion,n_paths=n_paths,seed=seed,boundary=boundary)

        Xt = paths[0] if paths.shape[0] == 1 else paths
        Ut = parameters[0](self.t)
        Bt = self.B(self.t)
        fXt = parameters[1](Xt)
        gUt = parameters[2](Ut)
        dt = self.l(fXt, gUt, parameters[4])
        Dt = self.D(Bt, parameters[5], dt)
        return self.t.copy(), Xt, Ut, Bt, fXt, gUt, dt, Dt

    def fokker_planc_solver(self, init_cond, parameters, sigma_x, degree=12, x_steps=201, cfl=0.45):
        # Solve Fokker-Planck model
        init_cond, parameters, sigma_x = BM1Validation.stochastic_inputs(init_cond, parameters, sigma_x)

        def drift(t, X):
            # Evaluate drift
            return self._drift(t, X, parameters)

        def diffusion(t, X):
            # Evaluate diffusion
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

        return self.t.copy(), x, p, coefficients, EXt, Ut, Bt, EDt

    def deterministic_solver(
        self,
        init_cond,
        parameters,
        *,
        t_eval=None,
        breakpoints=None,
        rtol=1.0e-3,
        atol=1.0e-6,
        max_step=np.inf,
    ):
        """Solve deterministic BM1, restarting at known input discontinuities.

        ``t_eval`` controls only the returned output grid. ``breakpoints`` are
        internal integration boundaries and need not occur in ``t_eval``.
        Existing calls without optional arguments retain ``self.t`` and
        SciPy's default tolerances.
        """
        # Solve deterministic model
        if t_eval is None:
            t_eval = self.t.copy()

        (
            init_cond,
            parameters,
            t_eval,
            breakpoints,
            rtol,
            atol,
            max_step,
        ) = BM1Validation.deterministic_inputs(
            init_cond,
            parameters,
            t_eval,
            breakpoints,
            self.T,
            rtol,
            atol,
            max_step,
        )

        interval_edges = np.concatenate(
            ([self.T[0]], breakpoints, [self.T[1]])
        )
        Xt = np.empty(t_eval.size, dtype=float)
        current_state = init_cond

        for index, (left, right) in enumerate(
            zip(interval_edges[:-1], interval_edges[1:])
        ):
            first_interval = index == 0
            final_interval = index == len(interval_edges) - 2

            def segment_rhs(
                t,
                state,
                left=left,
                right=right,
                first=first_interval,
                final=final_interval,
            ):
                # Use the correct one-sided input at each internal boundary.
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
                raise RuntimeError(
                    f"solve_ivp failed on interval [{left}, {right}]: "
                    f"{solution.message}"
                )

            if final_interval:
                output_mask = (t_eval >= left) & (t_eval <= right)
            else:
                output_mask = (t_eval >= left) & (t_eval < right)
            if np.any(output_mask):
                Xt[output_mask] = solution.sol(t_eval[output_mask])[0]

            current_state = solution.y[:, -1]

        Ut, Bt, fXt, gUt, dt, Dt = self.eval(
            t_eval,
            Xt,
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[4],
            parameters[5],
        )
        return t_eval.copy(), Xt, Ut, Bt, fXt, gUt, dt, Dt

    def optimize(
        self,
        initial_u,
        init_cond,
        parameters,
        *,
        interval_edges,
        t_eval=None,
        rtol=1.0e-6,
        atol=1.0e-8,
        max_step=np.inf,
        options=None,
    ):
        """Optimize one normalized price value per supplied time interval."""
        if t_eval is None:
            t_eval = self.t.copy()

        (
            initial_u,
            interval_edges,
            init_cond,
            parameters,
            t_eval,
            rtol,
            atol,
            max_step,
        ) = BM1Validation.optimize_inputs(
            initial_u,
            interval_edges,
            init_cond,
            parameters,
            t_eval,
            self.T,
            rtol,
            atol,
            max_step,
        )

        bounds = [(0.0, 1.0) for _ in range(initial_u.size)]

        def phi(interval_prices):
            return self.__phi(
                interval_prices,
                interval_edges,
                parameters,
                init_cond,
                t_eval,
                rtol,
                atol,
                max_step,
            )

        return minimize(
            phi,
            initial_u,
            method="L-BFGS-B",
            bounds=bounds,
            options=options,
        )
