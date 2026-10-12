"""Tests for saved WM4 receding interval and cumulative diffusion analysis."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_mpc_intervals import analyze, load_forecasts, scores


class MPCIntervalAnalysisTests(unittest.TestCase):
    @staticmethod
    def simulated_forecasts(days=2):
        rows = []
        for day in pd.date_range("2022-03-01", periods=days, freq="D"):
            for hour in range(24):
                origin = day + pd.Timedelta(hours=hour)
                sigma = 0.15 + hour / 1000
                for lead in range(1, 9):
                    initial = 0.003 * np.exp(-0.3 * lead)
                    process = 0.002 * lead
                    variance = initial + process
                    mean = 0.5 + hour / 400
                    observed = mean + 0.01 * ((hour + lead) % 3 - 1)
                    sd = np.sqrt(variance)
                    rows.append({
                        "origin": origin, "target_time": origin + pd.Timedelta(hours=lead),
                        "lead_hours": lead, "I_observed": observed, "I_mpc": mean,
                        "initial_variance": initial, "process_variance": process,
                        "predicted_variance": variance, "diffusion_sigma": sigma,
                        "sigma_fit_mode": "mpc_replay",
                        "calibration_mpc_count": 1,
                        "calibration_oracle_count": 4,
                        "calibration_horizon": 8,
                        "process_kernel_unit": process / sigma ** 2,
                        "r0_variance_mode": "oracle_reconstruction",
                        "R_observed_origin": 0.15,
                        "R_initialized_origin": 0.13,
                        "R_reconstruction_bias": 0.02,
                        "R_reconstruction_variance": 0.008,
                        "R_reconstruction_rmse": 0.095,
                        "R_reconstruction_samples": 5,
                        "R_bias_correction": 0.02,
                        "initial_F_variance": 0.003,
                        "initial_R_variance": 0.008,
                        "initial_F_component_variance": 0.2 * initial,
                        "initial_R_component_variance": 0.8 * initial,
                        "baseline_origin_residual_variance": 0.008,
                        "lower_68": mean - sd, "upper_68": mean + sd,
                        "lower_95": mean - 1.96 * sd, "upper_95": mean + 1.96 * sd,
                    })
        return pd.DataFrame(rows)

    def test_exact_and_cumulative_metrics_are_pooled(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.csv"
            self.simulated_forecasts().to_csv(path, index=False)
            frame, discarded, mode = load_forecasts(path)
            self.assertEqual(discarded, 0)
            self.assertEqual(mode, "mpc_replay")
            self.assertEqual(len(frame), 2 * 24 * 8)
            result = scores(frame)
            self.assertAlmostEqual(result["mse"], float(np.mean(frame["error"] ** 2)))
            self.assertAlmostEqual(result["predicted_rms_std"] ** 2,
                float(frame["predicted_variance"].mean()))
            self.assertEqual(result["coverage_95_pct"], 100.0)

    def test_partial_origins_removed_and_inconsistent_variance_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.csv"
            full = self.simulated_forecasts(1)
            incomplete = full.iloc[:-1]
            incomplete.to_csv(path, index=False)
            frame, discarded, _ = load_forecasts(path)
            self.assertEqual(discarded, 1)
            self.assertEqual(frame["origin"].nunique(), 23)
            invalid = full.copy()
            invalid.loc[0, "predicted_variance"] = 100
            invalid.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_forecasts(path)

    def test_stale_mpc_csv_is_rejected_instead_of_silently_plotting(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.csv"
            old = self.simulated_forecasts(1).drop(columns=[
                "r0_variance_mode", "initial_F_variance", "initial_R_variance",
                "initial_F_component_variance", "initial_R_component_variance",
                "baseline_origin_residual_variance",
            ])
            old["sigma_fit_mode"] = "historical_p0"
            old.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "OUTDATED WM4 MPC FORECAST DATA"):
                load_forecasts(path)
            accepted, _, _ = load_forecasts(path, allow_legacy=True)
            self.assertEqual(len(accepted), 192)

    def test_zero_r0_and_inconsistent_covariance_are_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.csv"
            data = self.simulated_forecasts(1)
            data["r0_variance_mode"] = "zero"
            data["R_bias_correction"] = 0.0
            data["R_initialized_origin"] = data["R_observed_origin"]
            data["initial_R_variance"] = 0.0
            data["initial_R_component_variance"] = 0.0
            data["initial_F_component_variance"] = data["initial_variance"]
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "expects"):
                load_forecasts(path)
            accepted, _, _ = load_forecasts(path, expected_r0_variance_mode="zero")
            self.assertEqual(len(accepted), 192)

            data["r0_variance_mode"] = "oracle_reconstruction"
            data["R_bias_correction"] = data["R_reconstruction_bias"]
            data["R_initialized_origin"] = data["R_observed_origin"] - data["R_bias_correction"]
            data["initial_R_variance"] = 0.008
            data["initial_R_component_variance"] = 0.001
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "propagated F0"):
                load_forecasts(path)

    def test_oracle_reconstruction_inconsistency_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.csv"
            data = self.simulated_forecasts(1)
            data["initial_R_variance"] = 0.5
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Oracle R0 covariance"):
                load_forecasts(path)
            data["initial_R_variance"] = data["R_reconstruction_variance"]
            data["R_initialized_origin"] = data["R_observed_origin"]
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Initial R0 center"):
                load_forecasts(path)

    def test_old_oracle_sigma_fit_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.csv"
            data = self.simulated_forecasts(1)
            data["sigma_fit_mode"] = "historical_p0"
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "expects 'mpc_replay'"):
                load_forecasts(path)
            accepted, _, mode = load_forecasts(path, expected_sigma_fit_mode="historical_p0")
            self.assertEqual(mode, "historical_p0")
            self.assertEqual(len(accepted), 192)

    def test_all_csv_and_plots_are_created_without_refitting(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forecasts.csv"
            directory = Path(temp) / "plots"
            self.simulated_forecasts(1).to_csv(path, index=False)
            result = analyze(path, directory, day="2022-03-01", horizon=8)
            self.assertEqual(result["example_day"], pd.Timestamp("2022-03-01"))
            self.assertEqual(len(result["exact"]), 8)
            self.assertEqual(int(result["cumulative"]["n"].iloc[-1]), 192)
            for name in (
                "exact_lead_metrics.csv", "cumulative_horizon_metrics.csv",
                "origin_hour_lead_metrics.csv", "origin_diffusion_estimates.csv",
                "daily_diffusion_summary.csv", "diffusion_variance_by_horizon.png",
                "interval_coverage_by_horizon.png", "origin_hour_lead_heatmaps.png",
                "fitted_sigma_over_time.png", "day_2022-03-01_8h_fans.png",
                "day_2022-03-01_rolling_1h.png", "summary.json",
            ):
                self.assertTrue((directory / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
