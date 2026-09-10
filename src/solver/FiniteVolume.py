import numpy as np

from src.validation.solver.FiniteVolume import FiniteVolume2DValidation


class FiniteVolume2D:
    """Generic two-dimensional finite-volume PDE solver.

    Solves p_t = -d_v(a_v p) - d_r(a_r p) + 1/2 d_rr(g^2 p)
    on a truncated rectangular computational domain. Drift uses conservative
    first-order upwinding; r-diffusion is backward-Euler implicit. Boundary
    faces admit outflow and no artificial inflow. Diffusion uses zero exterior
    density so mass reaching the computational boundary may leave the box.
    """

    @staticmethod
    def _upwind_flux_v(density, drift_v):
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
        max_v = float(np.max(np.abs(drift_v)))
        max_r = float(np.max(np.abs(drift_r)))
        limit_v = np.inf if max_v == 0.0 else dv / max_v
        limit_r = np.inf if max_r == 0.0 else dr / max_r
        limit = min(limit_v, limit_r)
        return np.inf if not np.isfinite(limit) else cfl * limit

    def solve(self, t, initial_density, drift_v, drift_r, diffusion_r, v_bounds=(-0.5, 1.5), r_bounds=(-1.0, 1.0), v_steps=161, r_steps=161, cfl=0.8):
        t = FiniteVolume2DValidation.time_grid(t)
        v_min, v_max = FiniteVolume2DValidation.bounds(v_bounds, "v_bounds")
        r_min, r_max = FiniteVolume2DValidation.bounds(r_bounds, "r_bounds")
        v_steps = FiniteVolume2DValidation.integer(v_steps, "v_steps")
        r_steps = FiniteVolume2DValidation.integer(r_steps, "r_steps")
        cfl = FiniteVolume2DValidation.cfl(cfl)

        dv = (v_max - v_min) / v_steps
        dr = (r_max - r_min) / r_steps
        v = v_min + (np.arange(v_steps) + 0.5) * dv
        r = r_min + (np.arange(r_steps) + 0.5) * dr
        V, R = np.meshgrid(v, r, indexing="ij")

        density0 = initial_density(V, R) if callable(initial_density) else initial_density
        density = FiniteVolume2DValidation.density(density0, (v_steps, r_steps), dv, dr)

        history = np.empty((len(t), v_steps, r_steps), dtype=float)
        mass_history = np.empty(len(t), dtype=float)
        history[0] = density
        mass_history[0] = np.sum(density) * dv * dr

        for n in range(len(t) - 1):
            current = float(t[n])
            target = float(t[n + 1])
            while current < target - 1e-15:
                av = FiniteVolume2DValidation.field(
                    drift_v(current, V, R), density.shape, "drift_v"
                )
                ar = FiniteVolume2DValidation.field(
                    drift_r(current, V, R), density.shape, "drift_r"
                )
                g = FiniteVolume2DValidation.field(
                    diffusion_r(current, V, R), density.shape, "diffusion_r",
                    nonnegative=True,
                )
                stable_h = self._stable_step(dv, dr, av, ar, cfl)
                remaining = target - current
                h = remaining if not np.isfinite(stable_h) else min(remaining, stable_h)
                if h <= 0.0:
                    raise RuntimeError("Fokker-Planck substep became non-positive.")

                flux_v = self._upwind_flux_v(density, av)
                flux_r = self._upwind_flux_r(density, ar)
                advected = density - h * (
                    np.diff(flux_v, axis=0) / dv + np.diff(flux_r, axis=1) / dr
                )
                advected = np.maximum(advected, 0.0)
                density = self._implicit_r_diffusion(advected, g, h, dr)
                density = np.maximum(density, 0.0)
                current += h

            history[n + 1] = density
            mass_history[n + 1] = np.sum(density) * dv * dr

        return {
            "v_grid": v,
            "r_grid": r,
            "density": history,
            "mass": mass_history,
            "dv": dv,
            "dr": dr,
        }
