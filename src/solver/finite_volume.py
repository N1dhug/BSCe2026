from collections import OrderedDict

import numpy as np

from src.validation.solver.finite_volume import finite_volume_1d_validation, finite_volume_2d_validation

class finite_volume_1d:
    """Conservative one-dimensional finite-volume Fokker-Planck reference.

    Cell probability masses are the primary unknowns. The spatial operator is
    assembled as a tridiagonal continuous-time Markov generator. Zero boundary
    flux is enforced by omitting transitions outside the supplied interval.
    Time integration uses implicit Euler with coefficients frozen at the left
    endpoint of each internal time interval. No clipping or renormalization is
    performed during evolution.
    """

    @staticmethod
    def grid(x_steps, bounds=(0.0, 1.0)):
        # Build cell-centered finite-volume grid
        x_steps = finite_volume_1d_validation.integer(x_steps)
        lower, upper = finite_volume_1d_validation.bounds(bounds)
        dx = (upper - lower) / x_steps
        edges = np.linspace(lower, upper, x_steps + 1, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return edges, centers, dx

    @staticmethod
    def build_time_grid(t_eval, breakpoints=None, max_step=np.inf):
        # Build output-aware internal integration grid
        t_eval, breakpoints, max_step = finite_volume_1d_validation.integration_grid_inputs(
            t_eval,
            breakpoints,
            max_step,
        )
        knots = np.unique(np.concatenate((t_eval, breakpoints)))
        pieces = []

        for index, (left, right) in enumerate(zip(knots[:-1], knots[1:])):
            width = right - left
            if np.isinf(max_step):
                n_substeps = 1
            else:
                ratio = width / max_step
                n_substeps = max(
                    1,
                    int(np.ceil(np.nextafter(ratio, -np.inf))),
                )
            segment = np.linspace(left, right, n_substeps + 1, dtype=float)
            segment[0], segment[-1] = left, right
            pieces.append(segment if index == 0 else segment[1:])

        integration_t = np.concatenate(pieces)
        output_indices = np.searchsorted(integration_t, t_eval)
        if np.any(output_indices >= integration_t.size) or not np.allclose(
            integration_t[output_indices],
            t_eval,
            atol=1.0e-14,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Failed to embed every finite-volume output time in the integration grid."
            )
        return integration_t, output_indices

    @staticmethod
    def generator_tridiagonal(dx, drift_faces, diffusion_centers):
        """Build Q for dm/dt = Q m from conservative BM/Fokker-Planck fluxes.

        For D=0.5*b^2 and the conservative flux
            J = a*p - d_x(D*p),
        upwind advection and a centered difference of D*p give non-negative
        nearest-neighbor transition rates. The returned diagonals describe Q,
        with columns summing to zero.
        """
        # Build Markov generator
        dx, drift_faces, diffusion_centers = (
            finite_volume_1d_validation.generator_inputs(
                dx,
                drift_faces,
                diffusion_centers,
            )
        )
        n_cells = diffusion_centers.size
        diffusion_term = 0.5 * np.square(diffusion_centers)

        rate_right = np.zeros(n_cells, dtype=float)
        rate_left = np.zeros(n_cells, dtype=float)

        rate_right[:-1] = (
            np.maximum(drift_faces, 0.0) / dx
            + diffusion_term[:-1] / (dx * dx)
        )
        rate_left[1:] = (
            np.maximum(-drift_faces, 0.0) / dx
            + diffusion_term[1:] / (dx * dx)
        )

        diagonal = -(rate_left + rate_right)
        lower = rate_right[:-1].copy()
        upper = rate_left[1:].copy()

        column_sum = diagonal.copy()
        column_sum[:-1] += lower
        column_sum[1:] += upper
        diagnostics = {
            "maximum_abs_column_sum": float(np.max(np.abs(column_sum))),
            "minimum_off_diagonal_rate": float(
                min(np.min(lower), np.min(upper))
            ),
        }
        return lower, diagonal, upper, diagnostics

    @staticmethod
    def _solve_tridiagonal(lower, diagonal, upper, rhs):
        # Solve one tridiagonal linear system by the Thomas algorithm
        n = len(diagonal)
        a = np.asarray(lower, dtype=float).copy()
        b = np.asarray(diagonal, dtype=float).copy()
        c = np.asarray(upper, dtype=float).copy()
        d = np.asarray(rhs, dtype=float).copy()

        for i in range(1, n):
            if b[i - 1] == 0.0:
                raise RuntimeError("Finite-volume tridiagonal solve encountered a zero pivot.")
            factor = a[i - 1] / b[i - 1]
            b[i] -= factor * c[i - 1]
            d[i] -= factor * d[i - 1]

        if b[-1] == 0.0:
            raise RuntimeError("Finite-volume tridiagonal solve encountered a zero pivot.")

        solution = np.empty(n, dtype=float)
        solution[-1] = d[-1] / b[-1]
        for i in range(n - 2, -1, -1):
            if b[i] == 0.0:
                raise RuntimeError("Finite-volume tridiagonal solve encountered a zero pivot.")
            solution[i] = (d[i] - c[i] * solution[i + 1]) / b[i]
        return solution

    def solve(
        self,
        t_eval,
        initial_mass,
        drift,
        diffusion,
        *,
        bounds=(0.0, 1.0),
        breakpoints=None,
        max_step=np.inf,
        coefficient_key=None,
    ):
        """Solve a one-dimensional Fokker-Planck equation without post-correction."""
        # Validate and construct grids
        if not callable(drift):
            raise TypeError("drift must be callable.")
        if not callable(diffusion):
            raise TypeError("diffusion must be callable.")

        initial_mass = finite_volume_1d_validation.initial_mass(initial_mass)
        edges, centers, dx = self.grid(initial_mass.size, bounds=bounds)
        integration_t, output_indices = self.build_time_grid(
            t_eval,
            breakpoints=breakpoints,
            max_step=max_step,
        )

        mass = initial_mass.copy()
        # Only requested outputs are retained. Every internal step is still
        # solved and checked; filtering usually requests only two endpoints.
        mass_history = np.empty((len(output_indices), mass.size), dtype=float)
        mass_history[0] = mass
        next_output = 1
        advance = self._stepper(dx, edges, centers, drift, diffusion, coefficient_key)

        maximum_abs_column_sum = 0.0
        minimum_off_diagonal_rate = np.inf
        minimum_cell_mass = float(np.min(mass))
        maximum_mass_error = abs(float(np.sum(mass)) - 1.0)

        for n in range(integration_t.size - 1):
            left = float(integration_t[n])
            right = float(integration_t[n + 1])
            h = right - left

            mass, generator_diagnostics = advance(mass, left, h)
            mass = finite_volume_1d_validation.evolved_mass(mass)
            if next_output < len(output_indices) and n + 1 == output_indices[next_output]:
                mass_history[next_output] = mass
                next_output += 1

            maximum_abs_column_sum = max(
                maximum_abs_column_sum,
                generator_diagnostics["maximum_abs_column_sum"],
            )
            minimum_off_diagonal_rate = min(
                minimum_off_diagonal_rate,
                generator_diagnostics["minimum_off_diagonal_rate"],
            )
            minimum_cell_mass = min(minimum_cell_mass, float(np.min(mass)))
            maximum_mass_error = max(
                maximum_mass_error,
                abs(float(np.sum(mass)) - 1.0),
            )

        density_history = mass_history / dx
        diagnostics = {
            "integration_t": integration_t.copy(),
            "output_indices": output_indices.copy(),
            "n_steps": integration_t.size - 1,
            "maximum_abs_generator_column_sum": maximum_abs_column_sum,
            "minimum_off_diagonal_rate": minimum_off_diagonal_rate,
            "minimum_cell_mass": minimum_cell_mass,
            "maximum_abs_mass_error": maximum_mass_error,
            "used_clipping": False,
            "used_renormalization": False,
        }
        return {
            "t": np.asarray(t_eval, dtype=float).copy(),
            "x_grid": centers,
            "x_edges": edges,
            "dx": dx,
            "cell_mass": mass_history,
            "density": density_history,
            "mass": np.sum(mass_history, axis=1),
            "diagnostics": diagnostics,
        }

    def _stepper(self, dx, edges, centers, drift, diffusion, coefficient_key):
        if coefficient_key is not None:
            raise ValueError('Coefficient reuse requires the cached LAPACK backend.')
        faces = edges[1:-1]

        def advance(mass, time, step):
            a = finite_volume_1d_validation.field(drift(time, faces), faces.shape, 'drift')
            b = finite_volume_1d_validation.field(diffusion(time, centers), centers.shape, 'diffusion')
            lower, diagonal, upper, diagnostics = self.generator_tridiagonal(dx, a, b)
            mass = self._solve_tridiagonal(-step * lower, 1.0 - step * diagonal, -step * upper, mass)
            return mass, diagnostics

        return advance


class finite_volume_1d_lapack(finite_volume_1d):
    """Opt-in LAPACK linear solve for the identical conservative FV scheme.

    Grid, coefficients, internal times and all probability diagnostics are
    inherited unchanged. The independent Python/Thomas reference is retained.
    """
    @staticmethod
    def _solve_tridiagonal(lower, diagonal, upper, rhs):
        from scipy.linalg.lapack import dgtsv
        rhs = np.array(rhs, dtype=float, copy=True, order='F')
        vector = rhs.ndim == 1
        _, _, _, solved, info = dgtsv(
            np.array(lower,dtype=float,copy=True),
            np.array(diagonal,dtype=float,copy=True),
            np.array(upper,dtype=float,copy=True),
            rhs[:, None] if vector else rhs,
            overwrite_dl=1,overwrite_d=1,overwrite_du=1,overwrite_b=1,
        )
        if info != 0:
            raise RuntimeError(f'LAPACK FV tridiagonal solve failed (info={info}).')
        return solved[:, 0] if vector else solved


class finite_volume_1d_cached(finite_volume_1d_lapack):
    """Exact coefficient/factor reuse for declared time-independent fields.

    ``coefficient_key(t)`` is a caller contract: equal keys must imply identical
    drift and diffusion arrays on this fixed spatial grid. With no such contract
    the ordinary LAPACK path evaluates both fields at every original left time.
    Caches are bounded and local to one solve, so parameters, grids and inputs
    cannot leak between calls. Step lengths are exact floats, never rounded.
    """

    def __init__(self, *, vectorized_inputs=False):
        if not isinstance(vectorized_inputs, (bool, np.bool_)):
            raise TypeError('vectorized_inputs must be boolean.')
        self.vectorized_inputs = bool(vectorized_inputs)

    def _stepper(self, dx, edges, centers, drift, diffusion, coefficient_key):
        if coefficient_key is None:
            return super()._stepper(dx, edges, centers, drift, diffusion, None)
        if not callable(coefficient_key):
            raise TypeError('coefficient_key must be callable.')
        from scipy.linalg.lapack import dgttrf, dgttrs

        faces = edges[1:-1]
        generators, factors = OrderedDict(), OrderedDict()

        def advance(mass, time, step):
            key = coefficient_key(time)
            if key not in generators:
                a = finite_volume_1d_validation.field(drift(time, faces), faces.shape, 'drift')
                b = finite_volume_1d_validation.field(diffusion(time, centers), centers.shape, 'diffusion')
                generators[key] = self.generator_tridiagonal(dx, a, b)
                if len(generators) > 8:
                    generators.popitem(last=False)
            generators.move_to_end(key)
            lower, diagonal, upper, diagnostics = generators[key]
            factor_key = (key, step)
            if factor_key not in factors:
                system = (-step * lower, 1.0 - step * diagonal, -step * upper)
                if not all(np.all(np.isfinite(v)) for v in system):
                    raise ValueError('Implicit step matrix must be finite.')
                *factor, info = dgttrf(*system, overwrite_dl=1, overwrite_d=1, overwrite_du=1)
                if info != 0:
                    raise RuntimeError(f'LAPACK FV factorization failed (info={info}).')
                factors[factor_key] = factor
                if len(factors) > 32:
                    factors.popitem(last=False)
            factors.move_to_end(factor_key)
            rhs = np.array(mass, copy=True, order='F')
            vector = rhs.ndim == 1
            solved, info = dgttrs(*factors[factor_key], rhs[:, None] if vector else rhs, overwrite_b=1)
            if info != 0:
                raise RuntimeError(f'LAPACK FV factored solve failed (info={info}).')
            return (solved[:, 0] if vector else solved), diagnostics

        return advance

    def solve_many(self, t_eval, initial_mass, drift, diffusion, *,
                   bounds=(0.0, 1.0), breakpoints=None, max_step=np.inf,
                   coefficient_key=None):
        """Evolve independent probability rows under one common operator.

        This is the same implicit Euler system with multiple right-hand sides.
        Every column retains its own state; diagnostics check every internal
        step and every column. Initial rows are validated without repair.
        Output masses have shape (requested times, hypotheses, cells).
        """
        if not callable(drift) or not callable(diffusion):
            raise TypeError('drift and diffusion must be callable.')
        if np.iscomplexobj(initial_mass):
            raise ValueError('initial_mass must be real.')
        initial = np.asarray(initial_mass, dtype=float)
        if initial.ndim != 2 or initial.shape[0] < 1:
            raise ValueError('initial_mass must contain one or more probability rows.')
        initial = np.stack([finite_volume_1d_validation.initial_mass(row) for row in initial])
        edges, centers, dx = self.grid(initial.shape[1], bounds=bounds)
        integration_t, output_indices = self.build_time_grid(t_eval, breakpoints, max_step)
        mass = np.array(initial.T, order='F', copy=True)
        history = np.empty((len(output_indices), *initial.shape), dtype=float)
        history[0] = initial
        next_output = 1
        advance = self._stepper(dx, edges, centers, drift, diffusion, coefficient_key)
        minima = initial.min(axis=1)
        mass_errors = abs(initial.sum(axis=1)-1.)
        column_error = 0.
        minimum_rate = np.inf
        for index, (left, right) in enumerate(zip(integration_t[:-1], integration_t[1:])):
            mass, diagnostics = advance(mass, float(left), float(right-left))
            if mass.shape != (initial.shape[1], initial.shape[0]) or not np.all(np.isfinite(mass)):
                raise RuntimeError('Batched finite-volume evolution returned invalid masses.')
            if np.any(mass < 0.):
                raise RuntimeError('Batched finite-volume evolution returned negative masses; no repair.')
            row_errors = abs(mass.T.sum(axis=1)-1.)
            if np.any(row_errors > 1e-10):
                raise RuntimeError('Batched finite-volume probability mass drifted; no repair.')
            minima = np.minimum(minima, mass.min(axis=0))
            mass_errors = np.maximum(mass_errors, row_errors)
            column_error = max(column_error, diagnostics['maximum_abs_column_sum'])
            minimum_rate = min(minimum_rate, diagnostics['minimum_off_diagonal_rate'])
            if next_output < len(output_indices) and index+1 == output_indices[next_output]:
                history[next_output] = mass.T
                next_output += 1
        shared = dict(integration_t=integration_t.copy(), output_indices=output_indices.copy(),
            n_steps=len(integration_t)-1, maximum_abs_generator_column_sum=column_error,
            minimum_off_diagonal_rate=minimum_rate, used_clipping=False, used_renormalization=False)
        per_column = tuple(dict(shared, minimum_cell_mass=float(minima[j]),
                                maximum_abs_mass_error=float(mass_errors[j]))
                           for j in range(len(initial)))
        return dict(t=np.asarray(t_eval, dtype=float).copy(), x_grid=centers, x_edges=edges,
                    cell_mass=history, prediction_diagnostics=per_column)


class finite_volume_2d:
    """Generic two-dimensional finite-volume PDE solver.

    Solves p_t = -d_v(a_v p) - d_r(a_r p) + 1/2 d_rr(g^2 p)
    on a truncated rectangular computational domain. Drift uses conservative
    first-order upwinding; r-diffusion is backward-Euler implicit. Boundary
    faces admit outflow and no artificial inflow. Diffusion uses zero exterior
    density so mass reaching the computational boundary may leave the box.
    """

    @staticmethod
    def _upwind_flux_v(density, drift_v):
        # Compute volume flux
        nv, nr = density.shape
        flux = np.zeros((nv + 1, nr), dtype=float)
        face = 0.5 * (drift_v[:-1] + drift_v[1:])
        flux[1:-1] = np.where(face >= 0.0, face * density[:-1], face * density[1:])
        left = drift_v[0]
        right = drift_v[-1]
        flux[0] = np.where(left < 0.0, left * density[0], 0.0)
        flux[-1] = np.where(right > 0.0, right * density[-1], 0.0)
        return flux

    @staticmethod
    def _upwind_flux_r(density, drift_r):
        # Compute rate flux
        nv, nr = density.shape
        flux = np.zeros((nv, nr + 1), dtype=float)
        face = 0.5 * (drift_r[:, :-1] + drift_r[:, 1:])
        flux[:, 1:-1] = np.where(face >= 0.0, face * density[:, :-1], face * density[:, 1:])
        lower = drift_r[:, 0]
        upper = drift_r[:, -1]
        flux[:, 0] = np.where(lower < 0.0, lower * density[:, 0], 0.0)
        flux[:, -1] = np.where(upper > 0.0, upper * density[:, -1], 0.0)
        return flux

    @staticmethod
    def _solve_tridiagonal(lower, diagonal, upper, rhs):
        # Solve tridiagonal system
        n = len(diagonal)
        c = upper.copy()
        d = rhs.copy()
        b = diagonal.copy()
        for j in range(1, n):
            w = lower[j - 1] / b[j - 1]
            b[j] -= w * c[j - 1]
            d[j] -= w * d[j - 1]
        x = np.empty(n, dtype=float)
        x[-1] = d[-1] / b[-1]
        for j in range(n - 2, -1, -1):
            x[j] = (d[j] - c[j] * x[j + 1]) / b[j]
        return x

    @classmethod
    def _implicit_r_diffusion(cls, density, diffusion, h, dr):
        # Apply implicit diffusion
        nv, nr = density.shape
        result = np.empty_like(density)
        row_diffusion_squared = np.mean(np.square(diffusion), axis=1)
        for i in range(nv):
            lam = 0.5 * h * row_diffusion_squared[i] / (dr * dr)
            if lam == 0.0:
                result[i] = density[i]
                continue
            diagonal = np.full(nr, 1.0 + 2.0 * lam)
            off = np.full(nr - 1, -lam)
            result[i] = cls._solve_tridiagonal(off, diagonal, off, density[i])
        return result

    @staticmethod
    def _stable_step(dv, dr, drift_v, drift_r, cfl):
        # Compute stable time step
        max_v = float(np.max(np.abs(drift_v)))
        max_r = float(np.max(np.abs(drift_r)))
        limit_v = np.inf if max_v == 0.0 else dv / max_v
        limit_r = np.inf if max_r == 0.0 else dr / max_r
        limit = min(limit_v, limit_r)
        return np.inf if not np.isfinite(limit) else cfl * limit

    def solve(self, t, initial_density, drift_v, drift_r, diffusion_r, v_bounds=(-0.5, 1.5), r_bounds=(-1.0, 1.0), v_steps=161, r_steps=161, cfl=0.8):
        # Solve finite-volume system
        t = finite_volume_2d_validation.time_grid(t)
        v_min, v_max = finite_volume_2d_validation.bounds(v_bounds, "v_bounds")
        r_min, r_max = finite_volume_2d_validation.bounds(r_bounds, "r_bounds")
        v_steps = finite_volume_2d_validation.integer(v_steps, "v_steps")
        r_steps = finite_volume_2d_validation.integer(r_steps, "r_steps")
        cfl = finite_volume_2d_validation.cfl(cfl)

        dv = (v_max - v_min) / v_steps
        dr = (r_max - r_min) / r_steps
        v = v_min + (np.arange(v_steps) + 0.5) * dv
        r = r_min + (np.arange(r_steps) + 0.5) * dr
        V, R = np.meshgrid(v, r, indexing="ij")

        density0 = initial_density(V, R) if callable(initial_density) else initial_density
        density = finite_volume_2d_validation.density(density0, (v_steps, r_steps), dv, dr)

        history = np.empty((len(t), v_steps, r_steps), dtype=float)
        mass_history = np.empty(len(t), dtype=float)
        history[0] = density
        mass_history[0] = np.sum(density) * dv * dr

        for n in range(len(t) - 1):
            current = float(t[n])
            target = float(t[n + 1])
            while current < target - 1e-15:
                av = finite_volume_2d_validation.field(drift_v(current, V, R), density.shape, "drift_v")
                ar = finite_volume_2d_validation.field(drift_r(current, V, R), density.shape, "drift_r")
                g = finite_volume_2d_validation.field(diffusion_r(current, V, R), density.shape, "diffusion_r", nonnegative=True,)
                stable_h = self._stable_step(dv, dr, av, ar, cfl)
                remaining = target - current
                h = remaining if not np.isfinite(stable_h) else min(remaining, stable_h)
                if h <= 0.0:
                    raise RuntimeError("Fokker-Planck substep became non-positive.")

                flux_v = self._upwind_flux_v(density, av)
                flux_r = self._upwind_flux_r(density, ar)
                advected = density - h * (np.diff(flux_v, axis=0) / dv + np.diff(flux_r, axis=1) / dr)
                advected = np.maximum(advected, 0.0)
                density = self._implicit_r_diffusion(advected, g, h, dr)
                density = np.maximum(density, 0.0)
                current += h

            history[n + 1] = density
            mass_history[n + 1] = np.sum(density) * dv * dr

        return {"v_grid": v, "r_grid": r, "density": history, "mass": mass_history, "dv": dv, "dr": dr,}
