"""Regression tests for direct oracle-residual variance calibration across MPC origins."""

import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_oracle_mpc_diffusion import (
    fit_variance, rolling_variance_fits, summarise, unit_process_covariance,
)
from src.model.wm_4 import wm_4


class WM4OracleMPCDiffusionTests(unittest.TestCase):
    @staticmethod
    def synthetic(q=0.09, bias=0.15):
        origin_index = pd.date_range("2022-05-01", periods=5, freq="h")
        standardized = np.arange(5, dtype=float) - 2.0
        standardized /= np.std(standardized, ddof=1)
        rows = []
        origin_rows = []
        for i, origin in enumerate(origin_index):
            origin_rows.append({"origin": origin, "origin_day": origin.normalize()})
            for lead in range(1, 25):
                K = 0.001 * lead
                residual = bias + standardized[i] * np.sqrt(q * K)
                rows.append({
                    "origin": origin, "origin_day": origin.normalize(),
                    "lead_hours": lead, "oracle_error": residual,
                    "unit_process_variance": K,
                    "I_observed": residual, "I_oracle": 0.0, "O": 0.0,
                })
        return pd.DataFrame(origin_rows), pd.DataFrame(rows)

    def test_direct_fit_matches_empirical_variance_not_mse(self):
        _, forecasts = self.synthetic()
        q, lead = fit_variance(forecasts)
        self.assertAlmostEqual(q, 0.09, places=12)
        self.assertTrue(np.allclose(
            lead["empirical_oracle_variance"],
            0.09 * lead["mean_unit_process_variance"],
        ))
        self.assertTrue(np.allclose(lead["oracle_bias"], 0.15))
        self.assertTrue(np.all(lead["oracle_rmse"] ** 2 > lead["empirical_oracle_variance"]))

    def test_summary_propagates_single_variance_fitted_q_to_every_origin(self):
        origins, forecasts = self.synthetic()
        fitted, rows, lead, summary, rolling = summarise(
            origins, forecasts, rolling_days=1, min_rolling_origins=3,
        )
        self.assertAlmostEqual(summary.iloc[0]["sigma"], 0.3, places=12)
        self.assertAlmostEqual(summary.iloc[0]["variance_rmse"], 0.0, places=12)
        self.assertTrue(np.allclose(
            rows["process_variance"], 0.09 * rows["unit_process_variance"],
        ))
        self.assertTrue(np.allclose(fitted["global_sigma"], 0.3))
        self.assertTrue(np.allclose(fitted["rolling_sigma"], 0.3))
        self.assertTrue(np.allclose(lead["fitted_process_std"] ** 2, lead["empirical_oracle_variance"]))
        self.assertEqual(len(rolling), 1)

    def test_one_origin_cannot_estimate_empirical_variance(self):
        _, forecasts = self.synthetic()
        one = forecasts.loc[forecasts["origin"] == forecasts["origin"].min()]
        with self.assertRaisesRegex(ValueError, "distinct complete MPC origins"):
            fit_variance(one)

    def test_unit_kernel_matches_model_propagation(self):
        steps_per_hour = 4
        r = np.array([-2.0, -0.1, -3.0])
        model = wm_4((0.0, 1.0), 24 * steps_per_hour + 1, lambda _t: 0.5)
        sample_indices = np.arange(1, 25) * steps_per_hour
        kernel = model.process_variance_kernel({"r": r}, sample_indices)
        expected = kernel @ np.ones(model.t_steps)
        actual = np.diag(unit_process_covariance(r, steps_per_hour))
        self.assertTrue(np.allclose(actual, expected, rtol=1e-10, atol=1e-12))


if __name__ == "__main__": unittest.main()
