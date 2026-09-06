from models.BatteryModel.BatteryModel import BatteryModel
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import simpson as simps
from scipy.optimize import minimize


class BM1(BatteryModel):
    def __init__(self, B, A, T, t_steps):
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

    @staticmethod
    def _validate_stochastic_inputs(init_cond, sigma_x, n_paths=1):
        if len(init_cond) == 0:
            raise ValueError("init_cond must contain the initial state of charge.")

        x0 = float(init_cond[0])
        if not 0.0 <= x0 <= 1.0:
            raise ValueError("The initial state of charge must lie in [0, 1].")

        if sigma_x < 0:
            raise ValueError("sigma_x must be non-negative.")

        if int(n_paths) != n_paths or n_paths < 1:
            raise ValueError("n_paths must be a positive integer.")

        return x0

    @staticmethod
    def _apply_boundary(values, boundary):
        values = np.asarray(values, dtype=float)

        if boundary == "reflect":
            # Repeated reflection into [0, 1]. This also handles proposals
            # that overshoot by more than one interval in a single step.
            reflected = np.mod(values, 2.0)
            return np.where(reflected <= 1.0, reflected, 2.0 - reflected)

        if boundary == "clip":
            return np.clip(values, 0.0, 1.0)

        if boundary == "none":
            return values

        raise ValueError("boundary must be 'reflect', 'clip', or 'none'.")

    def _drift(self, t, X, parameters):
        U, f, g, C, k, Xi = parameters

        if C <= 0:
            raise ValueError("C must be strictly positive.")

        _, Bt, _, _, _, Dt = self.eval(t, X, U, f, g, k, Xi)
        return np.asarray((Dt - Bt) / C, dtype=float)

    def _euler_maruyama_paths(
        self,
        init_cond,
        parameters,
        sigma_x,
        n_paths=1,
        seed=None,
        boundary="reflect",
    ):
        x0 = self._validate_stochastic_inputs(init_cond, sigma_x, n_paths)
        n_paths = int(n_paths)
        rng = np.random.default_rng(seed)

        paths = np.empty((n_paths, self.t_steps), dtype=float)
        paths[:, 0] = x0

        for i in range(self.t_steps - 1):
            t_i = float(self.t[i])
            h = float(self.t[i + 1] - self.t[i])
            X_i = paths[:, i]

            drift = self._drift(t_i, X_i, parameters)
            diffusion = np.asarray(
                self.diffusion_coefficient(X_i, sigma_x),
                dtype=float,
            )
            dW = np.sqrt(h) * rng.standard_normal(n_paths)

            proposal = X_i + drift * h + diffusion * dW
            paths[:, i + 1] = self._apply_boundary(proposal, boundary)

        return paths

    def euler_maruyama_solver(
        self,
        init_cond,
        parameters,
        sigma_x,
        n_paths=1,
        seed=None,
        boundary="reflect",
    ):
        """
        Simulate BM1 with the Euler-Maruyama method.

        The Ito SDE is
            dX = a(t, X) dt + b(X) dW
        with
            a(t, X) = (D(t, X) - B(t)) / C
            b(X) = sigma_x * X * (1 - X).

        Parameters
        ----------
        n_paths : int
            Number of Monte-Carlo paths. For n_paths=1 the returned arrays
            preserve the legacy stochastic_solver shape.
        seed : int or None
            Optional NumPy random seed for reproducible simulations.
        boundary : {"reflect", "clip", "none"}
            State handling after each Euler-Maruyama proposal. "reflect" is
            the default and is consistent with the zero-flux boundary used
            by fokker_planck_solver.
        """
        paths = self._euler_maruyama_paths(
            init_cond,
            parameters,
            sigma_x,
            n_paths=n_paths,
            seed=seed,
            boundary=boundary,
        )

        Xt = paths[0] if int(n_paths) == 1 else paths
        Ut = parameters[0](self.t)
        Bt = self.B(self.t)
        fXt = parameters[1](Xt)
        gUt = parameters[2](Ut)
        dt = self.l(fXt, gUt, parameters[4])
        Dt = self.D(Bt, parameters[5], dt)

        return self.t.copy(), Xt, Ut, Bt, fXt, gUt, dt, Dt

    def stochastic_solver(
        self,
        init_cond,
        parameters,
        sigma_x,
        seed=None,
        boundary="reflect",
    ):
        """
        Backwards-compatible stochastic BM1 solver.

        BM1 no longer uses torchsde. This wrapper returns one manually
        generated Euler-Maruyama trajectory with the same tuple layout as
        the previous stochastic_solver.
        """
        return self.euler_maruyama_solver(
            init_cond,
            parameters,
            sigma_x,
            n_paths=1,
            seed=seed,
            boundary=boundary,
        )

    @staticmethod
    def _initial_fokker_planck_density(x_grid, dx, x0):
        """
        Construct a discrete approximation of delta(x - x0) on cell centres.
        Linear deposition between neighbouring cells preserves x0 whenever it
        lies between the first and last cell centres.
        """
        density = np.zeros_like(x_grid, dtype=float)

        if x0 <= x_grid[0]:
            density[0] = 1.0 / dx
            return density

        if x0 >= x_grid[-1]:
            density[-1] = 1.0 / dx
            return density

        upper = int(np.searchsorted(x_grid, x0))
        lower = upper - 1
        weight_upper = (
            (x0 - x_grid[lower])
            / (x_grid[upper] - x_grid[lower])
        )

        density[lower] = (1.0 - weight_upper) / dx
        density[upper] = weight_upper / dx
        return density

    def fokker_planck_solver(
        self,
        init_cond,
        parameters,
        sigma_x,
        n_x=401,
        cfl=0.45,
    ):
        """
        Solve the BM1 Fokker-Planck equation on X in [0, 1].

        The probability density p(x, t) satisfies

            dp/dt = -d(a p)/dx + 1/2 d^2(b^2 p)/dx^2,

        where a is the BM1 drift and b(x) = sigma_x*x*(1-x).

        A conservative finite-volume discretization is used with:
          * upwind advection,
          * centred diffusion of b^2 p,
          * zero probability flux at x=0 and x=1,
          * adaptive internal time sub-stepping from a CFL bound.

        Returns
        -------
        t : ndarray, shape (t_steps,)
        x : ndarray, shape (n_x,)
            Cell-centre state grid.
        density : ndarray, shape (t_steps, n_x)
            Probability density at every model output time.
        mean : ndarray, shape (t_steps,)
            E[X_t] calculated from the density.
        variance : ndarray, shape (t_steps,)
            Var[X_t] calculated from the density.
        """
        x0 = self._validate_stochastic_inputs(init_cond, sigma_x, 1)

        if int(n_x) != n_x or n_x < 3:
            raise ValueError("n_x must be an integer greater than or equal to 3.")

        if not 0.0 < cfl < 1.0:
            raise ValueError("cfl must lie strictly between 0 and 1.")

        n_x = int(n_x)
        dx = 1.0 / n_x
        x = (np.arange(n_x, dtype=float) + 0.5) * dx

        density = np.empty((self.t_steps, n_x), dtype=float)
        p = self._initial_fokker_planck_density(x, dx, x0)
        density[0] = p

        for n in range(self.t_steps - 1):
            current_t = float(self.t[n])
            remaining = float(self.t[n + 1] - self.t[n])

            while remaining > 1e-15:
                drift = self._drift(current_t, x, parameters)
                diffusion = np.asarray(
                    self.diffusion_coefficient(x, sigma_x),
                    dtype=float,
                )
                diffusion_sq = diffusion * diffusion

                # A sufficient positivity/stability restriction for the
                # explicit upwind + diffusion finite-volume update.
                max_rate = (
                    np.max(np.abs(drift)) / dx
                    + np.max(diffusion_sq) / (dx * dx)
                )

                if max_rate > 0.0:
                    h = min(remaining, cfl / max_rate)
                else:
                    h = remaining

                drift_face = 0.5 * (drift[:-1] + drift[1:])
                upwind_p = np.where(
                    drift_face >= 0.0,
                    p[:-1],
                    p[1:],
                )

                flux = np.zeros(n_x + 1, dtype=float)
                flux[1:-1] = (
                    drift_face * upwind_p
                    - 0.5
                    * (
                        diffusion_sq[1:] * p[1:]
                        - diffusion_sq[:-1] * p[:-1]
                    )
                    / dx
                )

                # flux[0] = flux[-1] = 0 imposes reflecting/no-flux boundaries.
                p_new = p - (h / dx) * (flux[1:] - flux[:-1])

                min_density = float(np.min(p_new))
                if min_density < -1e-10:
                    raise FloatingPointError(
                        "Fokker-Planck density became negative. "
                        "Reduce cfl or increase n_x."
                    )

                # Remove only roundoff-scale negative values.
                p_new[p_new < 0.0] = 0.0

                mass = float(np.sum(p_new) * dx)
                if not np.isfinite(mass) or mass <= 0.0:
                    raise FloatingPointError(
                        "Fokker-Planck probability mass became invalid."
                    )

                # The finite-volume update is conservative. This normalization
                # only removes floating-point accumulation error (and any tiny
                # values clipped above).
                p = p_new / mass
                current_t += h
                remaining -= h

            density[n + 1] = p

        mean = np.sum(density * x[None, :], axis=1) * dx
        variance = (
            np.sum(
                density * (x[None, :] - mean[:, None]) ** 2,
                axis=1,
            )
            * dx
        )

        return self.t.copy(), x, density, mean, variance

    def compare_stochastic_solvers(
        self,
        init_cond,
        parameters,
        sigma_x,
        n_paths=10000,
        n_x=401,
        seed=None,
    ):
        """
        Compare Euler-Maruyama Monte-Carlo moments with Fokker-Planck moments.

        The Euler-Maruyama paths use reflecting boundaries, matching the
        zero-flux boundary condition of the Fokker-Planck solver.
        """
        paths = self._euler_maruyama_paths(
            init_cond,
            parameters,
            sigma_x,
            n_paths=n_paths,
            seed=seed,
            boundary="reflect",
        )

        em_mean = np.mean(paths, axis=0)
        em_variance = np.var(paths, axis=0)

        fp_t, x, density, fp_mean, fp_variance = self.fokker_planck_solver(
            init_cond,
            parameters,
            sigma_x,
            n_x=n_x,
        )

        dx = 1.0 / int(n_x)
        terminal_histogram, _ = np.histogram(
            paths[:, -1],
            bins=int(n_x),
            range=(0.0, 1.0),
            density=True,
        )
        terminal_l1_error = (
            np.sum(np.abs(terminal_histogram - density[-1])) * dx
        )

        return {
            "t": fp_t,
            "x": x,
            "euler_maruyama_paths": paths,
            "euler_maruyama_mean": em_mean,
            "euler_maruyama_variance": em_variance,
            "fokker_planck_density": density,
            "fokker_planck_mean": fp_mean,
            "fokker_planck_variance": fp_variance,
            "terminal_density_l1_error": terminal_l1_error,
        }
