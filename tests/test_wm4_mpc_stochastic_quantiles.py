"""Regression checks for per-quintile WM4 stochastic RMSE and coverage plots."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.plot_mpc_stochastic_quantiles import (
    analyze, plot_quantiles, select_mpc_origins, summarize_quantiles,
)


class WM4StochasticQuintileTests(unittest.TestCase):
    @staticmethod
    def forecasts(origins=20, horizon=8):
        rows = []
        for i, origin in enumerate(pd.date_range("2022-06-01 09:00", periods=origins, freq="h")):
            for h in range(1, horizon + 1):
                std = 0.045 + i * 0.003 + h * 0.001
                error = std * (0.45 if i % 2 else 2.1) * (1 if h % 2 else -1)
                mean = 0.5 + h * 0.01
                complete = i >= 4
                rows.append({"origin": origin, "lead_hours": h,
                    "target_time": origin + pd.Timedelta(hours=h),
                    "I_observed": mean + error, "I_mpc": mean,
                    "predicted_variance": std ** 2,
                    "sigma_rolling_complete": complete,
                    "sigma_rolling_days": 24,
                    "sigma_rolling_count": 24 if complete else 4,
                    "calibration_horizon": 6 if complete else 8,
                })
        return pd.DataFrame(rows)

    def test_each_quintile_contains_the_same_origins_at_every_lead(self):
        data, ranks = select_mpc_origins(self.forecasts(),
            quantiles=4, mode="mpc_replay_24d_6h", mature_only=True)
        self.assertEqual(len(ranks), 16)
        self.assertEqual(len(data), 16 * 8)
        self.assertEqual(ranks["uncertainty_quantile"].value_counts().sort_index().tolist(),
            [4, 4, 4, 4])
        summary = summarize_quantiles(data)
        self.assertEqual(len(summary), 4 * 8)
        self.assertTrue((summary["n_origins"] == 4).all())
        for q in range(1, 5):
            subset = data[data["uncertainty_quantile"] == q]
            for h in range(1, 9):
                self.assertEqual(set(subset[subset["lead_hours"] == h]["origin"]),
                    set(subset[subset["lead_hours"] == 1]["origin"]))

    def test_sample_rmse_std_and_interval_coverage_use_exact_leads(self):
        data, _ = select_mpc_origins(self.forecasts(),
            quantiles=4, mode="mpc_replay_24d_6h")
        scores = summarize_quantiles(data)
        subset = data[(data["uncertainty_quantile"] == 1) & (data["lead_hours"] == 2)]
        row = scores[(scores["uncertainty_quantile"] == 1) &
            (scores["lead_hours"] == 2)].iloc[0]
        error = subset["I_observed"] - subset["I_mpc"]
        variance = subset["predicted_variance"]
        std = np.sqrt(variance)
        self.assertAlmostEqual(row["observed_rmse"], np.sqrt(np.mean(error ** 2)))
        self.assertAlmostEqual(row["predicted_rms_std"], np.sqrt(variance.mean()))
        self.assertAlmostEqual(row["coverage_68_pct"],
            100 * np.mean(np.abs(error) <= std))
        self.assertAlmostEqual(row["coverage_95_pct"],
            100 * np.mean(np.abs(error) <= 1.96 * std))
        self.assertAlmostEqual(row["predicted_to_observed_mse"],
            variance.mean() / np.mean(error ** 2))

    def test_mature_only_excludes_bootstrap_not_other_modes(self):
        data = self.forecasts()
        mature, _ = select_mpc_origins(data, quantiles=4,
            mode="mpc_replay_24d_6h", mature_only=True)
        mixed, _ = select_mpc_origins(data, quantiles=4,
            mode="mpc_replay_24d_6h", mature_only=False)
        old, _ = select_mpc_origins(data, quantiles=4,
            mode="mpc_replay", mature_only=True)
        self.assertEqual(mature["origin"].nunique(), 16)
        self.assertEqual(mixed["origin"].nunique(), 20)
        self.assertEqual(old["origin"].nunique(), 20)
        changed = data.copy()
        changed.loc[0, "predicted_variance"] = 0
        with self.assertRaisesRegex(ValueError, "Positive predicted variance"):
            select_mpc_origins(changed, quantiles=4, mode="mpc_replay",
                mature_only=False)

    def test_incomplete_origins_excluded_and_too_few_rejected(self):
        data = self.forecasts()
        changed = data[~((data["origin"] == pd.Timestamp("2022-06-01 18:00")) &
            (data["lead_hours"] == 5))]
        selected, _ = select_mpc_origins(changed, quantiles=4,
            mode="mpc_replay_24d_6h")
        self.assertEqual(selected["origin"].nunique(), 15)
        with self.assertRaisesRegex(ValueError, "quantiles required"):
            select_mpc_origins(changed, quantiles=17, mode="mpc_replay_24d_6h")

    def test_chart_is_reproducible_from_archived_csv_without_refitting(self):
        frame = self.forecasts()
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.exe.run_model.wm_4.plot_mpc_stochastic_quantiles.load_forecasts",
                return_value=(frame, 0, "mpc_replay_24d_6h")) as reader:
                result = analyze(Path(tmp) / "saved.csv", Path(tmp) / "plots",
                    quantiles=4, mode="mpc_replay_24d_6h")
                reader.assert_called_once()
                self.assertEqual(reader.call_args.kwargs["expected_sigma_fit_mode"],
                    "mpc_replay_24d_6h")
            self.assertEqual(len(result["origins"]), 16)
            self.assertEqual(len(result["summary"]), 32)
            self.assertTrue(result["figure"].is_file())
            self.assertGreater(result["figure"].stat().st_size, 1000)
            self.assertTrue((result["output"] /
                "stochastic_calibration_by_quantile_and_lead.csv").exists())
            self.assertTrue((result["output"] /
                "mpc_origin_uncertainty_quantiles.csv").exists())


if __name__ == "__main__": unittest.main()
