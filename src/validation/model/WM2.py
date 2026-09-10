import itertools
import numpy as np

from src.validation.model.WM1 import WM1Validation


class WM2Validation(WM1Validation):
    """Validation rules for Water Model 2."""

    @classmethod
    def parameters(cls, parameters):
        validated = super().parameters(parameters)
        validated.pop("theta", None)
        validated["kappa_R"] = cls._finite_scalar(parameters.get("kappa_R"), "kappa_R")
        validated["sigma_0"] = cls._finite_scalar(parameters.get("sigma_0"), "sigma_0")
        validated["sigma_O"] = cls._finite_scalar(parameters.get("sigma_O"), "sigma_O")
        if validated["kappa_R"] <= 0.0:
            raise ValueError("kappa_R must be positive.")
        if validated["sigma_0"] <= 0.0:
            raise ValueError("sigma_0 must be positive.")
        if validated["sigma_O"] <= 0.0:
            raise ValueError("sigma_O must be positive.")
        cls.expected_equilibrium_parameters(validated["alpha"], validated["beta"])
        return validated

    @classmethod
    def initial_condition(cls, init_cond, allow_rate=True):
        values = np.asarray(init_cond, dtype=float).reshape(-1)
        if values.size != 2:
            raise ValueError("WM2 init_cond must contain exactly [V0, R0].")
        if not np.all(np.isfinite(values)):
            raise ValueError("init_cond must contain finite values.")
        if not 0.0 <= values[0] <= 1.0:
            raise ValueError("Initial expected volume V0 must lie in [0, 1].")
        return values

    @classmethod
    def deterministic_inputs(cls, init_cond, parameters):
        return cls.initial_condition(init_cond), cls.parameters(parameters)

    @classmethod
    def expected_equilibrium_parameters(cls, alpha, beta, tol=1e-10):
        alpha = cls.simplex(alpha, 4, "alpha")
        beta = cls.simplex(beta, 3, "beta")
        dV = -alpha[1] - (1.0 - alpha[2]) * beta[1]
        if dV >= -tol:
            raise ValueError("WM2 requires d_V < 0 so the expected equilibrium is stable.")

        equilibria = []
        for B_O, B_I, U in itertools.product((0.0, 1.0), repeat=3):
            o_bar = beta[0] * B_O + beta[2]
            i_bar = alpha[0] * B_I + alpha[1] + alpha[2] * o_bar + alpha[3]
            d0 = i_bar - o_bar
            dU = (1.0 - alpha[2]) * beta[2] - alpha[3]
            V_star = -(d0 + dU * U) / dV
            equilibria.append((B_O, B_I, U, V_star))
            if V_star < -tol or V_star > 1.0 + tol:
                raise ValueError(
                    "WM2 parameterization is inadmissible: expected equilibrium "
                    f"V*(B_O={B_O}, B_I={B_I}, U={U})={V_star:.6g} lies outside [0, 1]."
                )
        return dV, equilibria

    @classmethod
    def fp_inputs(cls, init_cond, parameters):
        return cls.initial_condition(init_cond), cls.parameters(parameters)
