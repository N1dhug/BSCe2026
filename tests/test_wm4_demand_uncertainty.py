"""Checks for WM4 constant demand-observation and accumulating process variance."""
import unittest

import numpy as np

from src.model.wm_4 import wm_4
from src.params.wm_4 import constant_diffusion_qp


class wm4_uncertainty_tests(unittest.TestCase):
    def setUp(self):
        self.fitter = constant_diffusion_qp(wm_4((0.0, 1.0), 25, lambda _: 0.5))

    def test_two_variance_components_recovered_from_demand_residuals(self):
        n, m = 24, 20000
        response = np.linspace(0.0, 1.5, n)
        kernel = response[:, None]
        process_q, observation_q = 0.08, 0.015
        rng = np.random.default_rng(7)
        residuals = rng.normal(size=(m, n)) * np.sqrt(observation_q + process_q * response)
        fit = self.fitter.fit(kernel, np.zeros(n), residuals=residuals)
        self.assertAlmostEqual(fit["q"], process_q, delta=0.005)
        self.assertAlmostEqual(fit["observation_q"], observation_q, delta=0.002)
        np.testing.assert_allclose(
            fit["propagated_variance"],
            fit["process_variance"] + fit["observation_variance"],
        )
        self.assertAlmostEqual(fit["process_variance"][0], 0.0)
        self.assertGreater(fit["observation_variance"][0], 0.0)

    def test_one_step_zero_observation_error(self):
        e, dt = np.array([0.1, -0.2, 0.3]), 0.25
        result = constant_diffusion_qp.one_step_diffusion(e, dt)
        self.assertAlmostEqual(result["q"], np.mean(e ** 2) / dt)

    def test_variance_components_need_distinct_horizons(self):
        with self.assertRaises(ValueError):
            self.fitter.fit(np.ones((4, 3)), np.ones(4))


if __name__ == "__main__":
    unittest.main()
