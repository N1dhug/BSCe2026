import unittest

import numpy as np

from src.exe.run_model.wm_5.moving_horizon import wm4_fixed_rate_qp
from src.model.wm_4 import wm_4
from src.model.wm_5 import response_library, wm_5
from src.params.wm_5 import qp_params


class wm_5_tests(unittest.TestCase):
    @staticmethod
    def _case(r_names=("linear",), u_names=("linear",)):
        library = response_library()
        model = wm_5((0.0, 1.0), 49, lambda _: 0.5,
                     [library[n] for n in r_names], [library[n] for n in u_names])
        U = np.linspace(-0.75, 0.75, model.t_steps)
        Time = model.t.copy()
        O = np.full(model.t_steps, 0.5)
        obs = np.zeros(model.t_steps, dtype=bool)
        obs[::2] = True
        transitions = np.zeros(model.t_steps-1, dtype=bool)
        transitions[:-2] = True
        return model, U, Time, O, obs, transitions

    def test_linear_reproduces_wm4(self):
        model, U, Time, O, _, _ = self._case()
        p = np.array([-2.0, -3.0, -0.5])
        initial = [0.12, -0.08]
        actual = model._solve_deterministic(initial, U, Time, O, p)
        baseline = wm_4((0.0, 1.0), model.t_steps, lambda _: 0.5)
        reference = baseline._solve_deterministic(initial, U, U, Time, O, [-2.0, -0.5, -3.0])
        np.testing.assert_allclose(actual["F"], reference["F"], atol=2e-14)
        np.testing.assert_allclose(actual["R"], reference["R"], atol=2e-14)
        np.testing.assert_allclose(actual["I"], reference["I"], atol=2e-14)

    def test_nonlinear_rate_price_fit_and_oracle(self):
        model, U, Time, O, mask, transitions = self._case(
            ("linear", "cubic"), ("linear", "exponential"))
        true_p = np.array([-2.0, -3.0, -0.1, -0.5, -0.1])
        model.coefficients(true_p)
        true_init = [0.12, 0.08]
        actual = model._solve_deterministic(true_init, U, Time, O, true_p)
        data = {"I": actual["I"], "O": O, "U": U, "Time": Time, "PE": U,
                "observation_mask": mask, "transition_mask": transitions}
        fit = qp_params(model).fit(data, [0.0, true_init[1]], max_state_iter=40)
        self.assertTrue(np.isfinite(fit["qp_loss"]))
        self.assertAlmostEqual(fit["init_cond"][1], true_init[1], places=11)
        self.assertGreaterEqual(fit["init_cond"][0], -1.0)
        self.assertLessEqual(fit["init_cond"][0], 1.0)
        model.coefficients(fit["coefficients"])
        self.assertLess(fit["qp_loss"], 1e-6)
        oracle = qp_params(model).optimize_prediction_initial_state(
            data, true_p, starts=2, max_nfev=50)
        self.assertLess(oracle["loss"], 1e-10)

    def test_closed_form_fixed_rate_matches_wm4(self):
        model, U, Time, O, mask, transitions = self._case()
        p = np.array([-2.0, -3.0, -0.5])
        initial = [0.12, -0.08]
        simulated = model._solve_deterministic(initial, U, Time, O, p)
        data = {"I": simulated["I"], "O": O, "U": U, "Time": Time, "PE": U,
                "observation_mask": mask, "transition_mask": transitions}
        fit5 = qp_params(model).fit(data, [0.0, initial[1]], max_state_iter=40)
        reference = wm_4((0.0, 1.0), model.t_steps, lambda _: 0.5)
        fit4 = wm4_fixed_rate_qp(reference).fit(
            data, [0.0, initial[1]], max_state_iter=40, state_tol=1e-8,
            f_bounds=(-1.0, 1.0), r_bounds=(initial[1], initial[1]))
        np.testing.assert_allclose(fit5["coefficients"], fit4["r"][[0, 2, 1]], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(fit5["init_cond"], fit4["init_cond"], atol=1e-4)


if __name__ == "__main__": unittest.main()
