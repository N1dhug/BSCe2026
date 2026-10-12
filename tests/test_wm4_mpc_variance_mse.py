"""WM4 variance/MSE 2x2 grid regression checks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.plot_mpc_variance_mse import (
    analyze, budget_scores, plot_budget, summarize_budgets,
)


class WM4MPCVarianceMSETests(unittest.TestCase):
    @staticmethod
    def forecasts():
        rows = []
        for i, origin in enumerate(pd.date_range("2022-07-01", periods=5, freq="h")):
            for h in range(1, 9):
                # Deliberately nonzero bias and signed F/R cross covariance.
                error = 0.1 * (i - 1.5) + 0.008 * h
                F = 0.002 + 0.0001 * h
                R = 0.003
                cross = -0.0005
                process = 0.0008 * h
                variance = F + R + cross + process
                rows.append({
                    "origin": origin, "lead_hours": h, "error": error,
                    "predicted_variance": variance,
                    "initial_variance": F + R + cross,
                    "initial_F_component_variance": F,
                    "initial_R_component_variance": R,
                    "initial_FR_component_variance": cross,
                    "process_variance": process,
                    "calibration_mpc_count": 5 if i > 0 else 4,
                    "calibration_oracle_count": 0 if i > 0 else 1,
                    "F_calibration_mpc_count": 5 if i > 0 else 4,
                    "F_calibration_oracle_count": 0 if i > 0 else 1,
                })
        return pd.DataFrame(rows)

    def test_budget_matches_signed_state_and_wiener_components(self):
        data = self.forecasts().query("lead_hours == 3")
        score = budget_scores(data)
        self.assertAlmostEqual(score["predicted_variance"],
            score["F0_variance"] + score["R0_variance"]
            + score["FR_covariance_contribution"] + score["Wiener_variance"])
        self.assertAlmostEqual(score["FR_covariance_contribution"], -0.0005)
        self.assertAlmostEqual(score["predicted_mse_unbiased"], score["predicted_variance"])
        self.assertAlmostEqual(score["actual_mse"],
            score["actual_population_error_variance"] + score["actual_bias_squared"])
        self.assertAlmostEqual(score["empirical_error_variance"],
            np.var(data["error"].to_numpy(), ddof=1))
        self.assertNotAlmostEqual(score["empirical_error_variance"], score["actual_mse"])

    def test_cumulative_is_pool_not_variance_sum(self):
        data = self.forecasts()
        exact, cumulative = summarize_budgets(data, 8)
        self.assertEqual(exact["n"].to_list(), [5] * 8)
        self.assertEqual(cumulative["n"].to_list(), [5 * h for h in range(1, 9)])
        self.assertAlmostEqual(cumulative.iloc[-1]["predicted_variance"],
            data["predicted_variance"].mean())
        self.assertAlmostEqual(cumulative.iloc[-1]["actual_mse"],
            np.mean(data["error"] ** 2))
        self.assertAlmostEqual(cumulative.iloc[-1]["empirical_error_variance"],
            np.var(data["error"], ddof=1))
        self.assertAlmostEqual(exact.iloc[-1]["actual_mse"],
            np.mean(data.loc[data["lead_hours"] == 8, "error"] ** 2))

    def test_inconsistent_variance_contributions_are_rejected(self):
        data = self.forecasts().query("lead_hours == 1").copy()
        data["predicted_variance"] += 0.01
        with self.assertRaisesRegex(ValueError, "do not sum"):
            budget_scores(data)

    def test_one_plot_has_four_axes_and_saves_csvs_without_refitting(self):
        data = self.forecasts()
        exact, cumulative = summarize_budgets(data, 8)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "figure.png"
            saved = plot_budget(exact, cumulative, path)
            self.assertTrue(saved.is_file())
            self.assertGreater(saved.stat().st_size, 0)

            with patch("src.exe.run_model.wm_4.plot_mpc_variance_mse.load_forecasts",
                return_value=(data, 0, "mpc_replay")) as reader:
                result = analyze(forecasts=Path(temp) / "source.csv",
                    output=Path(temp) / "budget", fully_mpc=True)
                reader.assert_called_once()
            self.assertEqual(result["exact"]["n"].to_list(), [4] * 8)
            self.assertEqual(result["cumulative"]["n"].to_list(), [4 * h for h in range(1, 9)])
            self.assertTrue((result["output"] / "variance_mse_exact_lead.csv").exists())
            self.assertTrue((result["output"] / "variance_mse_cumulative_1_to_H.csv").exists())
            self.assertTrue((result["output"] / "variance_mse_2x2.png").exists())


if __name__ == "__main__": unittest.main()
