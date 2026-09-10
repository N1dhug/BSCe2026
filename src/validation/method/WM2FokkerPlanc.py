import numpy as np

from src.validation.model.WM2 import WM2Validation
from src.validation.solver.FokkerPlanc2D import FokkerPlanc2DValidation


class WM2FokkerPlancValidation:
    @classmethod
    def solve_inputs(
        cls, t, init_cond, parameters, v_bounds, r_bounds, v_steps, r_steps, cfl
    ):
        t = FokkerPlanc2DValidation.time_grid(t)
        init_cond, parameters = WM2Validation.fp_inputs(init_cond, parameters)
        v_bounds = FokkerPlanc2DValidation.bounds(v_bounds, "v_bounds")
        r_bounds = FokkerPlanc2DValidation.bounds(r_bounds, "r_bounds")
        v_steps = FokkerPlanc2DValidation.integer(v_steps, "v_steps")
        r_steps = FokkerPlanc2DValidation.integer(r_steps, "r_steps")
        cfl = FokkerPlanc2DValidation.cfl(cfl)
        return t, init_cond, parameters, v_bounds, r_bounds, v_steps, r_steps, cfl

    @staticmethod
    def initial_std(initial_std):
        values = np.asarray(initial_std, dtype=float).reshape(-1)
        if values.size != 2 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("initial_std must contain positive finite [std_V, std_R].")
        return float(values[0]), float(values[1])
