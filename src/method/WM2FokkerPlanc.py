import numpy as np

from src.solver.FokkerPlanc2D import FokkerPlanc2D
from src.validation.method.WM2FokkerPlanc import WM2FokkerPlancValidation


class WM2FokkerPlanc:
    """WM2-specific Fokker-Planck construction and post-processing."""

    def __init__(self, solver=None):
        self.solver = FokkerPlanc2D() if solver is None else solver

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

    def solve(
        self, t, init_cond, parameters, initial_density=None,
        initial_std=(0.02, 0.02), v_bounds=(-0.5, 1.5),
        r_bounds=(-1.0, 1.0), v_steps=161, r_steps=161, cfl=0.8,
    ):
        (
            t, init_cond, parameters, v_bounds, r_bounds,
            v_steps, r_steps, cfl,
        ) = WM2FokkerPlancValidation.solve_inputs(
            t, init_cond, parameters, v_bounds, r_bounds, v_steps, r_steps, cfl
        )
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
            t, initial_density, drift_v, drift_r, diffusion_r,
            v_bounds=v_bounds, r_bounds=r_bounds,
            v_steps=v_steps, r_steps=r_steps, cfl=cfl,
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
            * dv / safe_mass
        )
        var_r = (
            np.sum(marginal_r * np.square(r[None, :] - mean_r[:, None]), axis=1)
            * dr / safe_mass
        )

        result.update({
            "t": t.copy(),
            "marginal_v": marginal_v,
            "marginal_r": marginal_r,
            "mean_v": mean_v,
            "mean_r": mean_r,
            "var_v": var_v,
            "var_r": var_r,
        })
        return result
