"""Regression tests for cumulative 24-hour oracle diffusion fitting at individual MPC origins."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_oracle_mpc_cumulative_diffusion import (
    fit_all_origins, fit_cumulative_origin, horizon_summary, load_predictions, save_plot,
)


class WM4OracleMPCCumulativeDiffusionTests(unittest.TestCase):
    @staticmethod
    def forecast(origin="2022-05-01 14:00", q=0.04, bias=0.0):
        origin = pd.Timestamp(origin)
        h = np.arange(1, 25, dtype=int)
        k = 0.0005 + 0.0003 * h
        # With zero bias each squared error equals q times the model unit variance.
        error = np.sqrt(q * k) + bias
        return pd.DataFrame({
            "origin": origin, "lead_hours": h, "I_observed": 0.5 + error,
            "I_oracle": np.full(h.size, 0.5),
            "oracle_error": error, "unit_process_variance": k,
            "r_F": -2.0, "r_U": -0.1, "r_R": -3.0,
            "F0_oracle": 0.1, "R0_oracle": 0.02,
        })

    def test_per_origin_cumulative_curve_recovers_diffusion(self):
        frame = self.forecast(q=0.04)
        fit, result = fit_cumulative_origin(frame)
        self.assertAlmostEqual(fit["q"], 0.04, places=12)
        self.assertAlmostEqual(fit["sigma"], 0.2, places=12)
        self.assertAlmostEqual(fit["cumulative_curve_mse"], 0.0, places=12)
        self.assertTrue(np.allclose(
            result["cumulative_error_squared"],
            result["fitted_cumulative_process_variance"],
        ))
        self.assertAlmostEqual(fit["observed_24h_rmse"], fit["model_24h_rms_std"], places=12)
        self.assertAlmostEqual(fit["r_F"], -2.0)
        self.assertAlmostEqual(fit["F0_oracle"], 0.1)

    def test_fit_is_per_origin_and_requires_no_cross_origin_sample_variance(self):
        combined = pd.concat([
            self.forecast("2022-05-01 14:00", q=0.04),
            self.forecast("2022-05-01 15:00", q=0.09),
        ], ignore_index=True)
        origin_fits, hourly = fit_all_origins(combined)
        self.assertEqual(len(origin_fits), 2)
        self.assertEqual(len(hourly), 48)
        self.assertTrue(np.allclose(origin_fits["q"], [0.04, 0.09]))
        self.assertTrue(np.allclose(origin_fits["sigma"], [0.2, 0.3]))
        summary = horizon_summary(hourly)
        self.assertEqual(summary["lead_hours"].tolist(), list(range(1, 25)))
        self.assertTrue(np.allclose(
            summary["mean_cumulative_squared_error"],
            summary["mean_cumulative_model_variance"],
        ))

    def test_missing_or_duplicate_horizon_is_rejected(self):
        frame = self.forecast()
        with self.assertRaisesRegex(ValueError, "exactly the 24 leads"):
            fit_cumulative_origin(frame[frame["lead_hours"] != 13])
        with self.assertRaisesRegex(ValueError, "exactly the 24 leads"):
            fit_cumulative_origin(pd.concat([frame.iloc[:23], frame.iloc[[22]]]))

    def test_zero_kernel_is_rejected(self):
        frame = self.forecast()
        frame["unit_process_variance"] = 0.0
        with self.assertRaisesRegex(ValueError, "kernel is zero"):
            fit_cumulative_origin(frame)

    def test_cached_data_avoids_any_refitting_and_plot_is_generated(self):
        frame = self.forecast()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "hourly.csv"
            frame.to_csv(source, index=False)
            loaded = load_predictions(source)
            self.assertEqual(len(loaded), 24)
            self.assertTrue(np.allclose(loaded["unit_process_variance"], frame["unit_process_variance"]))
            origins, hourly = fit_all_origins(loaded)
            summary = horizon_summary(hourly)
            fig = Path(folder) / "cumulative.png"
            save_plot(origins, summary, fig)
            self.assertTrue(fig.is_file())
            self.assertGreater(fig.stat().st_size, 0)


if __name__ == "__main__": unittest.main()
