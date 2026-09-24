import numpy as np

from src.solver.FiniteVolume import FiniteVolume2D
from src.validation.model.WM3 import WM3Validation
from src.validation.method.FokkerPlanc import WM2FokkerPlancValidation


class WM3FokkerPlanc:
    """Two-dimensional finite-volume Fokker-Planck solver for WM3."""

    def __init__(self, solver=None):
        self.solver = FiniteVolume2D() if solver is None else solver

    def solve(self, t, init_cond, parameters, initial_density, v_bounds=(-0.5, 1.5),
              r_bounds=(-1.0, 1.0), v_steps=161, r_steps=161, cfl=0.8):
        # Reuse the generic 2D finite-volume input checks used by WM2.
        t, init_cond, parameters, initial_density, v_bounds, r_bounds, v_steps, r_steps, cfl = (
            WM2FokkerPlancValidation.solve_inputs(
                t, init_cond, parameters, initial_density, v_bounds, r_bounds,
                v_steps, r_steps, cfl
            )
        )

        beta = WM3Validation.beta_coefficients(parameters["beta"])
        d = np.asarray(parameters["d_coeff"], dtype=float).reshape(-1)
        if d.size != 5: raise ValueError("WM3 d_coeff must contain [d0, dL, dV, dU, dH].")

        kappa_R = float(parameters["kappa_R"])
        sigma_0 = float(parameters["sigma_0"])
        sigma_O = float(parameters["sigma_O"])
        L_fun = parameters["L"]
        H_fun = parameters["H"]

        def fields(time, V):
            U = float(parameters["U"](time))
            time_feature = float(parameters["Time"](time))
            temp_feature = float(parameters["Temp"](time))
            L = float(L_fun(time_feature))
            H = float(H_fun(temp_feature))

            D = d[0] + d[1] * L + d[2] * V + d[3] * U + d[4] * H
            O = beta[0] * V + beta[1] * (1.0 - U) + beta[2] * L + beta[3] * H
            return D, O

        def drift_v(time, V, R):
            return R

        def drift_r(time, V, R):
            D, _ = fields(time, V)
            return kappa_R * (D - R)

        def diffusion_r(time, V, R):
            _, O = fields(time, V)
            g = sigma_0 + sigma_O * O
            if np.any(~np.isfinite(g)): raise ValueError("WM3 diffusion scale must be finite.")
            if np.any(g < 0.0): raise ValueError("WM3 produced negative g(O) on the selected numerical domain.")
            return np.broadcast_to(g, R.shape)

        result = self.solver.solve(
            t, initial_density, drift_v, drift_r, diffusion_r,
            v_bounds=v_bounds, r_bounds=r_bounds,
            v_steps=v_steps, r_steps=r_steps, cfl=cfl
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
        var_v = np.sum(marginal_v * np.square(v[None, :] - mean_v[:, None]), axis=1) * dv / safe_mass
        var_r = np.sum(marginal_r * np.square(r[None, :] - mean_r[:, None]), axis=1) * dr / safe_mass

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
