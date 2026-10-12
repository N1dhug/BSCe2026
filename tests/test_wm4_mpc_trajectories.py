"""Regression checks for full-eight-hour WM4 MPC trajectory diagnostics."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_mpc_trajectories import (
    analyze, origin_scores, parameter_associations, plot_reliability,
    plot_trajectory, prepare_trajectories, reliability_bins, select_examples,
)


class WM4MPCTrajectoryTests(unittest.TestCase):
    @staticmethod
    def history(n=12, horizon=8):
        rows = []
        for i, origin in enumerate(pd.date_range("2022-05-01 10:00", periods=n, freq="h")):
            for h in range(1, horizon + 1):
                F = 0.002 + h * 0.00025
                R = 0.004 / (1 + h / 8)
                FR = -0.00035
                W = 0.00085 * h * (1.0 + i / 20)
                variance = F + R + FR + W
                error = 0.012 * (i - 4) + (h - 4.5) * 0.009 + 0.002 * (i % 3)
                prediction = 0.4 + 0.015 * h + i * 0.003
                rows.append({
                    "origin": origin, "lead_hours": h,
                    "target_time": origin + pd.Timedelta(hours=h),
                    "I_observed": prediction + error, "I_mpc": prediction,
                    "predicted_variance": variance,
                    "initial_variance": F + R + FR,
                    "initial_F_component_variance": F,
                    "initial_R_component_variance": R,
                    "initial_FR_component_variance": FR,
                    "process_variance": W,
                    "r_F": -1.0 - i * 0.01, "r_U": -0.02,
                    "r_R": -2.0 - i * 0.02, "qp_loss": i * 0.001,
                    "diffusion_sigma": 0.1 + i * 0.001,
                    "F_initialized_origin": 0.1, "R_initialized_origin": 0.025,
                    "initial_F_variance": 0.004, "initial_R_variance": 0.005,
                    "initial_FR_covariance": -0.001,
                    "calibration_mpc_count": 5 if i > 2 else 4,
                    "calibration_oracle_count": 0 if i > 2 else 1,
                    "F_calibration_mpc_count": 5 if i > 2 else 4,
                    "F_calibration_oracle_count": 0 if i > 2 else 1,
                })
        return pd.DataFrame(rows)

    def test_cumulative_curves_are_means_within_each_origin(self):
        raw = self.history()
        data = prepare_trajectories(raw)
        self.assertEqual(len(data), len(raw))
        first = data[data["origin"] == raw.iloc[0]["origin"]]
        self.assertEqual(first["lead_hours"].tolist(), list(range(1, 9)))
        self.assertAlmostEqual(first.iloc[0]["cumulative_realized_mse"],
            first.iloc[0]["squared_error"])
        self.assertAlmostEqual(first.iloc[-1]["cumulative_realized_mse"],
            np.mean(first["squared_error"]))
        self.assertAlmostEqual(first.iloc[-1]["cumulative_predicted_variance"],
            np.mean(first["predicted_variance"]))
        self.assertAlmostEqual(first.iloc[-1]["cumulative_initial_variance"],
            np.mean(first["initial_variance"]))
        self.assertAlmostEqual(first.iloc[-1]["cumulative_wiener_variance"],
            np.mean(first["process_variance"]))
        self.assertAlmostEqual(first.iloc[-1]["cumulative_F0_variance"] +
            first.iloc[-1]["cumulative_R0_variance"] +
            first.iloc[-1]["cumulative_FR_covariance"] +
            first.iloc[-1]["cumulative_wiener_variance"],
            first.iloc[-1]["cumulative_predicted_variance"])
        self.assertAlmostEqual(first.iloc[-1]["cumulative_mse_ratio"],
            first.iloc[-1]["cumulative_predicted_variance"] /
            first.iloc[-1]["cumulative_realized_mse"])
        self.assertLess(first.iloc[-1]["cumulative_FR_covariance"], 0)

    def test_origin_scores_are_for_complete_eight_hour_paths(self):
        raw = self.history()
        data = prepare_trajectories(raw)
        scores = origin_scores(data)
        self.assertEqual(len(scores), 12)
        self.assertTrue((scores["origin_hour"] == scores["origin"].dt.hour).all())
        self.assertTrue(np.allclose(scores["overconfidence_ratio"],
            scores["realized_mse"] / scores["predicted_variance"]))
        self.assertIn("r_F", scores)
        self.assertIn("r_R", scores)
        self.assertGreater(len(select_examples(scores, top_k=1)), 0)
        self.assertEqual(select_examples(scores, origin=str(scores.iloc[3]["origin"])),
            [("requested", scores.iloc[3]["origin"])])
        with self.assertRaisesRegex(ValueError, "No complete"):
            select_examples(scores, origin="2030-01-01")
        partial = raw[~((raw["origin"] == raw.iloc[0]["origin"]) & (raw["lead_hours"] == 5))]
        processed = prepare_trajectories(partial)
        self.assertEqual(processed["origin"].nunique(), 11)
        broken = raw.copy()
        broken.loc[0, "process_variance"] += 0.02
        with self.assertRaisesRegex(ValueError, "Wiener"):
            prepare_trajectories(broken)

    def test_conditional_bins_average_realized_mse_not_variance_of_single_path(self):
        raw = self.history()
        data = prepare_trajectories(raw)
        scores = origin_scores(data)
        scored, grouped = reliability_bins(scores, data, quantiles=3)
        self.assertEqual(scored["uncertainty_quantile"].value_counts().sort_index().tolist(), [4, 4, 4])
        self.assertEqual(grouped.shape[0], 3 * 8)
        self.assertTrue((grouped["n_origins"] == 4).all())
        for q in range(1, 4):
            stamps = scored.loc[scored["uncertainty_quantile"] == q, "origin"]
            subset = data.loc[(data["origin"].isin(stamps)) & (data["lead_hours"] == 8)]
            row = grouped.loc[(grouped["uncertainty_quantile"] == q) &
                (grouped["horizon_hours"] == 8)].iloc[0]
            self.assertAlmostEqual(row["mean_cumulative_predicted_variance"],
                subset["cumulative_predicted_variance"].mean())
            self.assertAlmostEqual(row["mean_cumulative_realized_mse"],
                subset["cumulative_realized_mse"].mean())
            self.assertAlmostEqual(row["ratio_realized_to_predicted"],
                row["mean_cumulative_realized_mse"] / row["mean_cumulative_predicted_variance"])
        associations = parameter_associations(scored)
        self.assertIn("r_F", associations["parameter"].values)
        self.assertIn("rmse", associations["metric"].values)

    def test_trajectory_and_reliability_plots_save_and_no_refitting(self):
        raw = self.history(n=12)
        with tempfile.TemporaryDirectory() as temp:
            with patch("src.exe.run_model.wm_4.analyze_mpc_trajectories.load_forecasts",
                return_value=(raw, 0, "mpc_replay")) as reader:
                result = analyze(forecasts=Path(temp) / "existing.csv",
                    output=Path(temp) / "report", horizon=8,
                    quantiles=3, top_k=1, fully_mpc=False)
                reader.assert_called_once()
            self.assertEqual(result["scores"].shape[0], 12)
            self.assertTrue(result["reliability"].is_file())
            self.assertEqual(len(result["selected"]), 4)
            for record in result["selected"]:
                self.assertTrue(Path(record["figure"]).is_file())
            for name in ("origin_cumulative_curves.csv", "origin_summary.csv",
                "conditional_variance_reliability.csv", "selected_origins.csv",
                "parameter_associations.csv", "conditional_variance_reliability.png"):
                self.assertTrue((result["output"] / name).exists(), name)
            with patch("src.exe.run_model.wm_4.analyze_mpc_trajectories.load_forecasts",
                return_value=(raw, 0, "mpc_replay")):
                result = analyze(forecasts=Path(temp) / "existing.csv",
                    output=Path(temp) / "specific", horizon=8,
                    quantiles=3, origin="2022-05-01 14:00")
            self.assertEqual(len(result["selected"]), 1)
            self.assertEqual(result["selected"][0]["category"], "requested")


if __name__ == "__main__": unittest.main()
