"""Regression tests for predicted versus empirical WM4 MPC error variance."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.plot_mpc_variance import (
    analyze, select_origins, summarize_variance, variance_scores,
)


class WM4MPCVariancePlotTests(unittest.TestCase):
    @staticmethod
    def forecasts():
        origins = pd.date_range("2022-06-01", periods=6, freq="h")
        rows = []
        for i, origin in enumerate(origins):
            for lead in range(1, 9):
                error = 0.1 * (i - 2) + 0.025 * lead
                rows.append({"origin": origin, "lead_hours": lead, "error": error,
                    "predicted_variance": 0.02 + 0.001 * lead,
                    "calibration_mpc_count": 5 if i >= 2 else 4,
                    "calibration_oracle_count": 0 if i >= 2 else 1,
                    "F_calibration_mpc_count": 5 if i >= 2 else 4,
                    "F_calibration_oracle_count": 0 if i >= 2 else 1})
        return pd.DataFrame(rows)

    def test_observed_variance_is_error_variance_not_mse(self):
        origin = pd.Timestamp("2022-06-01")
        frame = pd.DataFrame({"origin": [origin] * 3,
            "error": [1.0, 2.0, 3.0],
            "predicted_variance": [0.7, 0.8, 0.9]})
        result = variance_scores(frame)
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["empirical_error_variance"], 1.0)
        self.assertAlmostEqual(result["empirical_error_variance_population"], 2 / 3)
        self.assertAlmostEqual(result["forecast_bias"], 2.0)
        self.assertAlmostEqual(result["bias_squared"], 4.0)
        self.assertAlmostEqual(result["observed_mse"], 14 / 3)
        self.assertAlmostEqual(result["predicted_variance"], 0.8)
        self.assertAlmostEqual(result["variance_ratio_predicted_to_empirical"], 0.8)
        self.assertAlmostEqual(result["observed_mse"],
            result["empirical_error_variance_population"] + result["bias_squared"])
        self.assertNotAlmostEqual(result["observed_mse"], result["empirical_error_variance"])

    def test_variance_by_lead_and_cumulative_pooling(self):
        frame = self.forecasts()
        lead, cumulative = summarize_variance(frame, 8)
        self.assertEqual(lead["n"].tolist(), [6] * 8)
        self.assertEqual(cumulative["n"].tolist(), [6 * h for h in range(1, 9)])
        self.assertTrue(np.allclose(lead["empirical_error_variance"].to_numpy(),
            np.var(0.1 * (np.arange(6) - 2), ddof=1)))
        self.assertAlmostEqual(cumulative.iloc[-1]["empirical_error_variance"],
            np.var(frame["error"], ddof=1))
        self.assertAlmostEqual(cumulative.iloc[-1]["predicted_variance"],
            frame["predicted_variance"].mean())
        self.assertAlmostEqual(cumulative.iloc[-1]["observed_mse"],
            np.mean(frame["error"] ** 2))

    def test_fully_mpc_removes_bootstrap_origins_without_mixing_leads(self):
        frame = self.forecasts()
        selected = select_origins(frame, fully_mpc=True)
        self.assertEqual(selected["origin"].nunique(), 4)
        self.assertEqual(len(selected), 4 * 8)
        self.assertEqual(selected.groupby("origin")["lead_hours"].nunique().tolist(), [8] * 4)
        incomplete = frame.copy()
        incomplete["F_calibration_mpc_count"] = 4
        with self.assertRaisesRegex(ValueError, "No origins"):
            select_origins(incomplete, fully_mpc=True)
        with self.assertRaisesRegex(ValueError, "calibration columns"):
            select_origins(frame.drop(columns="calibration_mpc_count"), fully_mpc=True)

    def test_plot_and_csv_are_saved_from_existing_forecasts_without_refitting(self):
        frame = self.forecasts()
        with tempfile.TemporaryDirectory() as temp, patch(
            "src.exe.run_model.wm_4.plot_mpc_variance.load_forecasts",
            return_value=(frame, 0, "mpc_replay")) as loader:
            out = analyze(forecasts=Path(temp) / "historical.csv",
                output=Path(temp) / "results", fully_mpc=True)
            loader.assert_called_once()
            self.assertTrue(out["figure"].is_file())
            self.assertGreater(out["figure"].stat().st_size, 0)
            lead = pd.read_csv(out["output"] / "variance_by_exact_lead.csv")
            cumulative = pd.read_csv(out["output"] / "variance_cumulative_1_to_H.csv")
            self.assertEqual(len(lead), 8)
            self.assertEqual(len(cumulative), 8)
            self.assertEqual(lead["n"].tolist(), [4] * 8)
            self.assertEqual(int(cumulative.iloc[-1]["n"]), 32)


if __name__ == "__main__": unittest.main()
