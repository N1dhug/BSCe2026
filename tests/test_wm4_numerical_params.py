import unittest
from unittest.mock import patch

import numpy as np

from src.model.wm_4 import wm_4
from src.params.wm_4 import numerical_params, qp_params


class test_wm4_numerical_params(unittest.TestCase):
    @staticmethod
    def example():
        model = wm_4((0.0, 1.0), 97, lambda time: 0.5)
        elapsed = np.arange(97, dtype=float)/4
        U = 0.15*np.sin(2.0*np.pi*elapsed/24)
        O = np.full(97, 0.5)
        truth = np.array([-3.0, -0.2, -4.0])
        initial = np.array([0.03, -0.04])
        result = model._solve_deterministic(initial, U, U, elapsed, O, truth)
        mask = np.zeros(97, dtype=bool)
        mask[::4] = True
        data = {"I": result["I"], "O": O, "U": U, "PE": U,
                "Time": elapsed, "observation_mask": mask}
        return model, data, initial, truth

    def test_numerical_gradient_without_and_with_intercept(self):
        model, data, initial, truth = self.example()
        fitter = numerical_params(model)
        prepared = qp_params(model)._prepare_data(data)
        for fit_intercept, params in ((False, truth), (True, np.r_[truth, 0.02])):
            _, analytic = fitter._loss_and_gradient(params, prepared, initial, fit_intercept)
            differences = []
            for k in range(len(params)):
                offset = np.zeros(len(params))
                offset[k] = 1e-5
                left = fitter._loss_and_gradient(params-offset, prepared, initial, fit_intercept)[0]
                right = fitter._loss_and_gradient(params+offset, prepared, initial, fit_intercept)[0]
                differences.append((right-left)/2e-5)
            np.testing.assert_allclose(analytic, differences, atol=1e-7, rtol=1e-5)

    def test_model_facade_uses_relocated_parameter_fitter(self):
        model, data, initial, _ = self.example()
        direct = numerical_params(model).fit(data, initial, max_iter=100)
        with patch("src.model.wm_4.numerical_params", wraps=numerical_params) as construct:
            facade = model.fit_params_numerical(data, initial, max_iter=100)
            self.assertEqual(construct.call_count, 1)
        self.assertEqual(facade["fit_method"], "numerical")
        self.assertFalse(facade["fit_initial_state"])
        self.assertEqual(facade["init_cond"].tolist(), initial.tolist())
        np.testing.assert_allclose(facade["r"], direct["r"], atol=1e-8)
        self.assertAlmostEqual(facade["deterministic_loss"], direct["deterministic_loss"])
        self.assertLessEqual(facade["deterministic_loss"], direct["qp_seed_loss"]+1e-7)
        self.assertIn("numerical_solve_time", facade)
        self.assertIn("qp_seed_time", facade)


if __name__ == "__main__": unittest.main()
