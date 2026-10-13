"""Paired quintile comparison regression tests for WM4 sigma estimators."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.compare_sigma_quantiles import (
    analyze, compare_metrics, matched_forecasts, plot_independent,
    plot_paired, quantile_curves, summarize_origins,
)


class SigmaComparisonTests(unittest.TestCase):
    @staticmethod
    def forecasts(n=20, horizon=8):
        old, new = [], []
        for i, origin in enumerate(pd.date_range("2022-07-01", periods=n, freq="h")):
            for h in range(1, horizon + 1):
                truth = 0.4 + 0.01 * h
                pred = truth - 0.01 * (i - 7) - 0.004 * h
                init = 0.0025 + 0.0002 * h
                old_variance = init + 0.0005 * h * (1 + i / 4)
                new_variance = init + 0.00025 * h * (1 + i / 5)
                common = {"origin": origin, "lead_hours": h, "I_observed": truth,
                    "I_mpc": pred, "initial_variance": init}
                old.append({**common, "process_variance": old_variance - init,
                    "predicted_variance": old_variance, "diffusion_sigma": 0.11})
                new.append({**common, "process_variance": new_variance - init,
                    "predicted_variance": new_variance, "diffusion_sigma": 0.07,
                    "sigma_rolling_count": 24 if i >= 4 else 4,
                    "sigma_rolling_days": 24,
                    "calibration_horizon": 24 if i >= 4 else 8,
                    "sigma_rolling_complete": i >= 4})
        return pd.DataFrame(old), pd.DataFrame(new)

    def test_only_mature_same_origins_are_compared(self):
        old, new = self.forecasts()
        paired = matched_forecasts(old, new, require_full_history=True)
        self.assertEqual(paired["origin"].nunique(), 16)
        self.assertEqual(len(paired), 16 * 8)
        self.assertTrue(np.allclose(paired["error"],
            paired["I_observed_old"] - paired["I_mpc_old"]))
        self.assertTrue(np.allclose(paired["I_mpc_old"], paired["I_mpc_new"]))
        invalid = new.copy()
        invalid.loc[invalid["lead_hours"] == 2, "I_mpc"] += 0.3
        with self.assertRaisesRegex(ValueError, "deterministic fit"):
            matched_forecasts(old, invalid)
        with self.assertRaisesRegex(ValueError, "maturity columns"):
            matched_forecasts(old, new.drop(columns="sigma_rolling_complete"))

    def test_old_quantiles_use_identical_observed_errors_for_both_methods(self):
        old, new = self.forecasts()
        paired = matched_forecasts(old, new)
        summary = summarize_origins(paired, quantiles=4)
        self.assertEqual(summary["old_quantile"].value_counts().tolist(), [4] * 4)
        old_bins = quantile_curves(paired, summary, quantiles=4, independent=False)
        own_bins = quantile_curves(paired, summary, quantiles=4, independent=True)
        self.assertEqual(len(old_bins), 4 * 8 * 2)
        self.assertEqual(len(own_bins), 4 * 8 * 2)
        for q in range(1, 5):
            g = old_bins[(old_bins["quantile"] == q) & (old_bins["horizon"] == 8)]
            self.assertAlmostEqual(g.iloc[0]["realized_mse"], g.iloc[1]["realized_mse"])
        metric = compare_metrics(paired, summary)
        self.assertEqual(metric.shape[0], 16)
        self.assertTrue(np.all(np.isfinite(metric["mean_gaussian_nll"])))

    def test_six_hour_mode_checks_rolling_calibration_horizon_and_is_paired(self):
        old, new = self.forecasts()
        new = new.copy()
        new["calibration_horizon"] = np.where(
            new["sigma_rolling_complete"], 6, 8)
        paired = matched_forecasts(old, new, calibration_horizon=6)
        self.assertEqual(paired["origin"].nunique(), 16)
        self.assertEqual(len(paired), 16 * 8)
        with self.assertRaisesRegex(ValueError, "No fully mature"):
            matched_forecasts(old, new, calibration_horizon=24)

        with tempfile.TemporaryDirectory() as temp:
            with patch("src.exe.run_model.wm_4.compare_sigma_quantiles.load_forecasts",
                side_effect=[(old, 0, "mpc_replay"), (new, 0, "mpc_replay_24d_6h")]) as reader:
                result = analyze(old_path=Path(temp) / "old.csv",
                    new_path=Path(temp) / "six.csv",
                    output=Path(temp) / "six_hour", quantiles=4,
                    new_mode="mpc_replay_24d_6h")
                self.assertEqual(reader.call_count, 2)
                self.assertEqual(reader.call_args_list[1].kwargs["expected_sigma_fit_mode"],
                    "mpc_replay_24d_6h")
            self.assertEqual(len(result["origins"]), 16)
            self.assertTrue(result["figure"].exists())
            self.assertTrue(result["reliability"].exists())

    def test_both_rolling_estimators_must_be_mature_when_comparing_6h_vs_24h(self):
        old, new = self.forecasts()
        old = old.copy()
        old["sigma_rolling_complete"] = np.arange(len(old)) // 8 >= 7
        old["sigma_rolling_count"] = np.where(old["sigma_rolling_complete"], 24, 4)
        old["sigma_rolling_days"] = 24
        old["calibration_horizon"] = np.where(old["sigma_rolling_complete"], 24, 8)
        new["calibration_horizon"] = np.where(new["sigma_rolling_complete"], 6, 8)
        paired = matched_forecasts(old, new, calibration_horizon=6,
            old_calibration_horizon=24)
        self.assertEqual(paired["origin"].nunique(), 13)
        self.assertEqual(len(paired), 13 * 8)

        with tempfile.TemporaryDirectory() as temp:
            with patch("src.exe.run_model.wm_4.compare_sigma_quantiles.load_forecasts",
                side_effect=[(old, 0, "mpc_replay_24d_24h"),
                             (new, 0, "mpc_replay_24d_6h")]) as reader:
                result = analyze(old_path=Path(temp) / "full.csv",
                    new_path=Path(temp) / "six.csv",
                    output=Path(temp) / "six_vs_full", quantiles=4,
                    old_mode="mpc_replay_24d_24h", new_mode="mpc_replay_24d_6h")
                self.assertEqual(reader.call_args_list[0].kwargs["expected_sigma_fit_mode"],
                    "mpc_replay_24d_24h")
            self.assertEqual(len(result["origins"]), 13)
            self.assertTrue(result["figure"].exists())

    def test_thirty_day_six_hour_against_24_day_six_hour(self):
        old, new = self.forecasts()
        old = old.copy()
        old["sigma_rolling_complete"] = np.arange(len(old)) // 8 >= 4
        old["sigma_rolling_count"] = np.where(old["sigma_rolling_complete"], 24, 4)
        old["sigma_rolling_days"] = 24
        old["calibration_horizon"] = np.where(old["sigma_rolling_complete"], 6, 8)

        new = new.copy()
        new["sigma_rolling_complete"] = np.arange(len(new)) // 8 >= 7
        new["sigma_rolling_count"] = np.where(new["sigma_rolling_complete"], 30, 4)
        new["sigma_rolling_days"] = 30
        new["calibration_horizon"] = np.where(new["sigma_rolling_complete"], 6, 8)
        paired = matched_forecasts(old, new, calibration_horizon=6, days=30,
            old_calibration_horizon=6, old_days=24)
        self.assertEqual(paired["origin"].nunique(), 13)
        self.assertEqual(len(paired), 13 * 8)

        with tempfile.TemporaryDirectory() as temp:
            with patch("src.exe.run_model.wm_4.compare_sigma_quantiles.load_forecasts",
                side_effect=[(old, 0, "mpc_replay_24d_6h"),
                    (new, 0, "mpc_replay_30d_6h")]) as reader:
                result = analyze(old_path=Path(temp) / "old.csv",
                    new_path=Path(temp) / "new.csv", output=Path(temp) / "report",
                    quantiles=4, old_mode="mpc_replay_24d_6h",
                    new_mode="mpc_replay_30d_6h")
                self.assertEqual(reader.call_count, 2)
                self.assertEqual(reader.call_args_list[1].kwargs["expected_sigma_fit_mode"],
                    "mpc_replay_30d_6h")
            self.assertEqual(len(result["origins"]), 13)
            self.assertTrue(result["figure"].is_file())
            self.assertTrue(result["reliability"].is_file())

    def test_figure_csv_and_nll_summary_generated_without_refit(self):
        old, new = self.forecasts()
        with tempfile.TemporaryDirectory() as temp:
            with patch("src.exe.run_model.wm_4.compare_sigma_quantiles.load_forecasts",
                side_effect=[(old, 0, "mpc_replay"), (new, 0, "mpc_replay_24d_24h")]) as loader:
                result = analyze(old_path=Path(temp) / "old.csv",
                    new_path=Path(temp) / "new.csv",
                    output=Path(temp) / "plots", quantiles=4)
                self.assertEqual(loader.call_count, 2)
            self.assertTrue(result["figure"].is_file())
            self.assertTrue(result["reliability"].is_file())
            self.assertEqual(len(result["origins"]), 16)
            for filename in ("paired_baseline_quintile_curves.csv",
                "independent_quintile_curves.csv", "paired_cumulative_nll_mse.csv",
                "matched_origin_quintiles.csv", "matched_mpc_forecasts.csv"):
                self.assertTrue((result["output"] / filename).exists())


if __name__ == "__main__": unittest.main()
