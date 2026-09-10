import numpy as np
from scipy.special import gammaln

from src.solver.ActiveSet import ActiveSet
from src.solver.FiniteVolume import FiniteVolume2D
from src.validation.method.FokkerPlanc import (
    BM1FokkerPlancValidation,
    WM2FokkerPlancValidation,
)


class BM1FokkerPlanc:
    """One-dimensional Beta-mixture Fokker-Planck method used by BM1/WM1.

    PDE time integration belongs to the method layer. Constrained least-squares
    recovery of Beta-mixture weights is delegated to the generic ActiveSet
    solver.
    """

    @staticmethod
    def beta_basis(x, degree):
        x, degree = BM1FokkerPlancValidation.beta_basis_inputs(x, degree)
        m = np.arange(degree + 1, dtype=float)[:, None]
        x_row = x[None, :]
        log_coefficient = (
            np.log(degree + 1.0)
            + gammaln(degree + 1.0)
            - gammaln(m + 1.0)
            - gammaln(degree - m + 1.0)
        )
        log_basis = (
            log_coefficient
            + m * np.log(x_row)
            + (degree - m) * np.log1p(-x_row)
        )
        return np.exp(log_basis)

    @staticmethod
    def _normalise_basis_discretely(basis, dx):
        basis, dx, masses = BM1FokkerPlancValidation.basis_normalisation_inputs(basis, dx)
        return basis / masses[:, None]

    @staticmethod
    def initial_coefficients(init_cond, degree):
        x0, degree = BM1FokkerPlancValidation.initial_coefficients_inputs(init_cond, degree)
        component_means = (np.arange(degree + 1, dtype=float) + 1.0) / (degree + 2.0)
        coefficients = np.zeros(degree + 1, dtype=float)

        if x0 <= component_means[0]:
            coefficients[0] = 1.0
            return coefficients
        if x0 >= component_means[-1]:
            coefficients[-1] = 1.0
            return coefficients

        right = int(np.searchsorted(component_means, x0, side="right"))
        left = right - 1
        mu_left = component_means[left]
        mu_right = component_means[right]
        weight_right = (x0 - mu_left) / (mu_right - mu_left)
        coefficients[left] = 1.0 - weight_right
        coefficients[right] = weight_right
        return coefficients

    @staticmethod
    def mixture_density(coefficients, basis):
        coefficients, basis = BM1FokkerPlancValidation.mixture_inputs(coefficients, basis)
        return coefficients @ basis

    @staticmethod
    def project_density(density, basis, dx):
        """Project p onto a nonnegative, unit-mass Beta mixture."""
        density, basis, dx = BM1FokkerPlancValidation.projection_inputs(density, basis, dx)
        scale = np.sqrt(dx)
        matrix = basis.T * scale
        target = density * scale
        coefficients = ActiveSet.simplex_least_squares(matrix, target)
        projected = coefficients @ basis
        return coefficients, projected

    @staticmethod
    def _normalise_density(density, dx):
        density, dx = BM1FokkerPlancValidation.density_inputs(density, dx)
        density = np.maximum(density, 0.0)
        mass = BM1FokkerPlancValidation.positive_mass(np.sum(density) * dx)
        return density / mass

    @staticmethod
    def _flux_rhs(x, density, drift_values, diffusion_values):
        x, density, drift_values, diffusion_values = BM1FokkerPlancValidation.flux_inputs(
            x, density, drift_values, diffusion_values
        )
        dx = float(x[1] - x[0])
        diffusion_squared = np.square(diffusion_values)

        face_drift = 0.5 * (drift_values[:-1] + drift_values[1:])
        upwind_density = np.where(face_drift >= 0.0, density[:-1], density[1:])
        advective_flux = face_drift * upwind_density
        diffusive_quantity = diffusion_squared * density
        diffusive_flux = -0.5 * np.diff(diffusive_quantity) / dx

        flux = np.zeros(len(x) + 1, dtype=float)
        flux[1:-1] = advective_flux + diffusive_flux
        return -np.diff(flux) / dx

    @staticmethod
    def _stable_step(dx, drift_values, diffusion_values, cfl):
        dx = BM1FokkerPlancValidation.positive_scalar(dx, "dx")
        cfl = BM1FokkerPlancValidation.cfl(cfl)
        drift_values = BM1FokkerPlancValidation.field_values(
            drift_values, np.asarray(drift_values).shape, "drift"
        )
        diffusion_values = BM1FokkerPlancValidation.field_values(
            diffusion_values, np.asarray(diffusion_values).shape, "diffusion"
        )
        max_drift = float(np.max(np.abs(drift_values)))
        max_diffusion_squared = float(np.max(np.square(diffusion_values)))
        advective_limit = np.inf if max_drift == 0.0 else dx / max_drift
        diffusive_limit = (
            np.inf
            if max_diffusion_squared == 0.0
            else dx * dx / max_diffusion_squared
        )
        return cfl * min(advective_limit, diffusive_limit)

    def solve(self, t, init_cond, drift, diffusion, degree=12, x_steps=201, cfl=0.45):
        """Solve the one-dimensional Fokker-Planck equation."""
        t, x0, drift, diffusion, degree, x_steps, cfl = BM1FokkerPlancValidation.solve_inputs(
            t, init_cond, drift, diffusion, degree, x_steps, cfl
        )

        dx = 1.0 / x_steps
        x = (np.arange(x_steps, dtype=float) + 0.5) * dx

        basis = self._normalise_basis_discretely(self.beta_basis(x, degree), dx)
        coefficients = self.initial_coefficients(x0, degree)
        density = self._normalise_density(self.mixture_density(coefficients, basis), dx)

        density_history = np.empty((len(t), x_steps), dtype=float)
        coefficient_history = np.empty((len(t), degree + 1), dtype=float)
        density_history[0] = density
        coefficient_history[0] = coefficients

        for n in range(len(t) - 1):
            current_time = float(t[n])
            target_time = float(t[n + 1])

            while current_time < target_time:
                drift_values = BM1FokkerPlancValidation.field_values(
                    drift(current_time, x), x.shape, "drift"
                )
                diffusion_values = BM1FokkerPlancValidation.field_values(
                    diffusion(current_time, x), x.shape, "diffusion"
                )
                stable_h = self._stable_step(dx, drift_values, diffusion_values, cfl)
                remaining = target_time - current_time
                h = remaining if not np.isfinite(stable_h) else min(remaining, stable_h)
                h = BM1FokkerPlancValidation.positive_substep(h)

                density = density + h * self._flux_rhs(
                    x, density, drift_values, diffusion_values
                )
                density = self._normalise_density(density, dx)
                current_time += h

            coefficients, density = self.project_density(density, basis, dx)
            coefficient_history[n + 1] = coefficients
            density_history[n + 1] = density

        return x, density_history, coefficient_history


class WM2FokkerPlanc:
    """WM2-specific 2D Fokker-Planck construction and post-processing."""

    def __init__(self, solver=None):
        self.solver = FiniteVolume2D() if solver is None else solver

    @staticmethod
    def _reduced_coefficients(B_O, B_I, alpha, beta):
        o_bar = beta[0] * B_O + beta[2]
        i_bar = alpha[0] * B_I + alpha[1] + alpha[2] * o_bar + alpha[3]
        d0 = i_bar - o_bar
        dV = -alpha[1] - (1.0 - alpha[2]) * beta[1]
        dU = (1.0 - alpha[2]) * beta[2] - alpha[3]
        return d0, dV, dU

    @staticmethod
    def gaussian_initial_density(V0, R0, std_V, std_R):
        norm = 1.0 / (2.0 * np.pi * std_V * std_R)

        def density(V, R):
            exponent = -0.5 * (
                np.square((V - V0) / std_V) + np.square((R - R0) / std_R)
            )
            return norm * np.exp(exponent)

        return density

    def solve(self, t, init_cond, parameters, initial_density=None, initial_std=(0.02, 0.02), v_bounds=(-0.5, 1.5), r_bounds=(-1.0, 1.0), v_steps=161, r_steps=161, cfl=0.8):
        t, init_cond, parameters, v_bounds, r_bounds, v_steps, r_steps, cfl = WM2FokkerPlancValidation.solve_inputs(t, init_cond, parameters, v_bounds, r_bounds, v_steps, r_steps, cfl)
        alpha = parameters["alpha"]
        beta = parameters["beta"]
        kappa_R = parameters["kappa_R"]
        sigma_0 = parameters["sigma_0"]
        sigma_O = parameters["sigma_O"]

        if initial_density is None:
            std_V, std_R = WM2FokkerPlancValidation.initial_std(initial_std)
            initial_density = self.gaussian_initial_density(
                init_cond[0], init_cond[1], std_V, std_R
            )

        def fields(time, V):
            U = float(parameters["U"](time))
            B_O = float(parameters["B_O"](time))
            B_I = float(parameters["B_I"](time))
            d0, dV, dU = self._reduced_coefficients(B_O, B_I, alpha, beta)
            O = beta[0] * B_O + beta[1] * V + beta[2] * (1.0 - U)
            return U, d0, dV, dU, O

        def drift_v(time, V, R):
            return R

        def drift_r(time, V, R):
            U, d0, dV, dU, _ = fields(time, V)
            return kappa_R * (d0 + dV * V + dU * U - R)

        def diffusion_r(time, V, R):
            _, _, _, _, O = fields(time, V)
            g = sigma_0 + sigma_O * O
            if np.any(g < 0.0):
                raise ValueError(
                    "WM2 produced negative g(O) on the selected numerical domain; "
                    "choose an admissible domain/parameterization with O >= 0."
                )
            return np.broadcast_to(g, R.shape)

        result = self.solver.solve(
            t,
            initial_density,
            drift_v,
            drift_r,
            diffusion_r,
            v_bounds=v_bounds,
            r_bounds=r_bounds,
            v_steps=v_steps,
            r_steps=r_steps,
            cfl=cfl,
        )

        p = result["density"]
        v = result["v_grid"]
        r = result["r_grid"]
        dv = result["dv"]
        dr = result["dr"]
        marginal_v = np.sum(p, axis=2) * dr
        marginal_r = np.sum(p, axis=1) * dv
        mass = result["mass"]
        safe_mass = np.where(mass > 0.0, mass, np.nan)
        mean_v = np.sum(marginal_v * v[None, :], axis=1) * dv / safe_mass
        mean_r = np.sum(marginal_r * r[None, :], axis=1) * dr / safe_mass
        var_v = (
            np.sum(marginal_v * np.square(v[None, :] - mean_v[:, None]), axis=1)
            * dv
            / safe_mass
        )
        var_r = (
            np.sum(marginal_r * np.square(r[None, :] - mean_r[:, None]), axis=1)
            * dr
            / safe_mass
        )

        result.update(
            {
                "t": t.copy(),
                "marginal_v": marginal_v,
                "marginal_r": marginal_r,
                "mean_v": mean_v,
                "mean_r": mean_r,
                "var_v": var_v,
                "var_r": var_r,
            }
        )
        return result
