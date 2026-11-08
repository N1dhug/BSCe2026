import unittest

import numpy as np

from src.exe.run_model.wm_4.fit_moving_horizon_diffusion import Z_SCORE, variance_kernel
from src.exe.run_model.wm_4.methodology_uncertainty import (
    demand_interval, fine_variance_kernel, fit_historical_uncertainty,
)


class test_wm4_methodology_uncertainty(unittest.TestCase):
    def test_fine_kernel_agrees_with_hourly_implementation(self):
        theta_f, theta_r, sph, k = -.4, -.9, 4, 12
        fine = fine_variance_kernel(theta_f, theta_r, sph, k)
        hourly = variance_kernel(theta_f, theta_r, sph, k)
        self.assertEqual(len(fine), k*sph+1)
        np.testing.assert_allclose(fine[np.arange(k)*sph], hourly, rtol=1e-12)
        self.assertAlmostEqual(fine[0], 0.)
        self.assertTrue(np.all(np.diff(fine)>=0.))

    def test_historical_only_variance_fit_and_demand_bands(self):
        theta_f, theta_r, sph, k, n = -.4, -.9, 4, 6, 3
        sigma, tau = .04, .02
        K = variance_kernel(theta_f, theta_r, sph, k)
        residual = np.sqrt(tau**2+sigma**2*K)
        obs_idx = np.arange(k)*sph
        inputs, states = [], []
        for i in range(n):
            fitted = np.full(k*sph+1, .5)
            actual = fitted.copy()
            actual[obs_idx] += (-1.)**i*residual
            mask = np.zeros(len(actual), dtype=bool)
            mask[obs_idx] = True
            inputs.append({"I":actual, "observation_mask":mask})
            states.append({"I":fitted})
        fit = {"r":np.array([theta_f, -.1, theta_r]),
               "deterministic_states":states}
        uncertainty = fit_historical_uncertainty(fit, inputs, sph)
        self.assertTrue(uncertainty["identifiable"])
        self.assertAlmostEqual(uncertainty["tau"], tau, places=10)
        self.assertAlmostEqual(uncertainty["sigma"], sigma, places=10)
        self.assertEqual(uncertainty["historical_observations"], n*k)
        np.testing.assert_allclose(uncertainty["sd_hourly"], residual)
        predicted_target = np.full(k*sph+1, .65)
        lower, upper = demand_interval(predicted_target, uncertainty)
        np.testing.assert_allclose((lower+upper)/2., predicted_target)
        np.testing.assert_allclose(upper-lower, 2*Z_SCORE*uncertainty["sd_fine"])
        self.assertAlmostEqual(uncertainty["historical_coverage_95"], 1.)

    def test_zero_residuals_produce_zero_uncertainty(self):
        k, sph = 3, 4
        actual = np.full(k*sph+1, .5)
        mask = np.zeros(len(actual), dtype=bool)
        mask[np.arange(k)*sph] = True
        fit = {"r":np.array([-.4,-.1,-.9]),
               "deterministic_states":[{"I":actual.copy()}]}
        result = fit_historical_uncertainty(fit,
            [{"I":actual, "observation_mask":mask}], sph)
        self.assertEqual(result["sigma"], 0.)
        self.assertEqual(result["tau"], 0.)
        np.testing.assert_array_equal(result["sd_fine"], 0.)

    def test_reject_nonmatching_observation_mask(self):
        fitted = np.full(17, .5)
        fit = {"r":np.array([-.4,-.1,-.9]),
               "deterministic_states":[{"I":fitted}]}
        mask = np.zeros(17, dtype=bool)
        mask[[0, 1, 4, 8]] = True
        with self.assertRaisesRegex(ValueError, "hourly observations"):
            fit_historical_uncertainty(fit, [{"I":fitted,"observation_mask":mask}], 4)


if __name__ == "__main__": unittest.main()
