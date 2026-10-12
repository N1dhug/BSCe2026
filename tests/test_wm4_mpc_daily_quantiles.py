"""Regression checks for full-day WM4 MPC RMSE/calibration/sharpness quantiles."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_daily_mpc_quantiles import (
    analyze, assign_quantiles, daily_metrics, origin_metrics, overlap_summary, quantile_overlap,
    quantile_spread, plot_quantiles,
)


class WM4MPCDailyQuantilesTests(unittest.TestCase):
    @staticmethod
    def forecasts(days=15):
        rows = []
        for i, day in enumerate(pd.date_range("2022-05-01", periods=days, freq="D")):
            scale = 0.025 + 0.004 * i
            # Deliberately varied calibration, not a monotone function of RMSE.
            variance_multiplier = [0.65, 1.0, 1.55, 0.83, 1.25][i % 5]
            for hour in range(24):
                origin = day + pd.Timedelta(hours=hour)
                for lead in range(1, 9):
                    e = scale * ((hour - 11.5) / 7 + 0.07 * (lead - 4.5)) + 0.004
                    rows.append({
                        "origin": origin, "origin_day": day,
                        "lead_hours": lead, "error": e,
                        "predicted_variance": scale ** 2 * variance_multiplier,
                        "calibration_mpc_count": 4 if i == 0 else 5,
                        "calibration_oracle_count": 1 if i == 0 else 0,
                        "F_calibration_mpc_count": 4 if i == 0 else 5,
                        "F_calibration_oracle_count": 1 if i == 0 else 0,
                    })
        return pd.DataFrame(rows)

    def test_full_day_pools_exactly_24_times_8_and_rejects_partial(self):
        forecasts = self.forecasts(6)
        daily, excluded = daily_metrics(forecasts)
        self.assertEqual(len(daily), 6)
        self.assertTrue(excluded.empty)
        self.assertTrue((daily["n_origins"] == 24).all())
        self.assertTrue((daily["n_predictions"] == 192).all())
        sample = forecasts.loc[forecasts["origin_day"] == pd.Timestamp("2022-05-02")]
        values = sample["error"].to_numpy(dtype=float)
        second = daily.loc[daily["day"] == pd.Timestamp("2022-05-02")].iloc[0]
        self.assertAlmostEqual(second["rmse"], np.sqrt(np.mean(values ** 2)))
        self.assertAlmostEqual(second["actual_error_variance"], np.var(values, ddof=1))
        self.assertAlmostEqual(second["predicted_variance"], sample["predicted_variance"].mean())
        self.assertAlmostEqual(second["variance_calibration_error"],
            abs(np.log(second["predicted_variance"] / second["actual_error_variance"])))

        partial = forecasts[~(
            (forecasts["origin_day"] == pd.Timestamp("2022-05-03")) &
            (forecasts["origin"].dt.hour == 16))]
        daily, excluded = daily_metrics(partial)
        self.assertEqual(len(daily), 5)
        self.assertEqual(excluded.iloc[0]["complete_origins"], 23)

    def test_individual_origin_scatter_uses_eight_leads_not_daily_averages(self):
        frame = self.forecasts(2)
        origins = origin_metrics(frame)
        self.assertEqual(len(origins), 2 * 24)
        self.assertTrue((origins["n_predictions"] == 8).all())
        first_origin = frame.loc[frame["origin"] == pd.Timestamp("2022-05-01")]
        first = origins.loc[origins["origin"] == pd.Timestamp("2022-05-01")].iloc[0]
        self.assertAlmostEqual(first["rmse"],
            np.sqrt(np.mean(first_origin["error"] ** 2)))
        self.assertAlmostEqual(first["predicted_rms_std"],
            np.sqrt(np.mean(first_origin["predicted_variance"])))
        self.assertAlmostEqual(first["mse_calibration_error"],
            abs(np.log(first["predicted_variance"] / first["observed_mse"])))
        daily, _ = daily_metrics(frame)
        self.assertNotAlmostEqual(first["rmse"], daily.iloc[0]["rmse"])
        self.assertEqual(len(daily), 2)
        missing_lead = frame[~((frame["origin"] == pd.Timestamp("2022-05-01")) &
            (frame["lead_hours"] == 5))]
        with self.assertRaisesRegex(ValueError, "Expected one full"):
            origin_metrics(missing_lead)
        with self.assertRaisesRegex(ValueError, "requires columns"):
            origin_metrics(frame.drop(columns="predicted_variance"))

    def test_all_three_quantile_rankings_and_overlap_tables(self):
        days, _ = daily_metrics(self.forecasts())
        ranked = assign_quantiles(days, quantiles=5)
        for col in ("rmse_quantile", "calibration_quantile", "uncertainty_quantile"):
            self.assertEqual(ranked[col].value_counts().sort_index().tolist(), [3] * 5)
        self.assertTrue(ranked.sort_values("rmse")["rmse_quantile"].is_monotonic_increasing)
        self.assertTrue(ranked.sort_values("predicted_rms_std")["uncertainty_quantile"].is_monotonic_increasing)
        self.assertTrue(ranked.sort_values("variance_calibration_error")["calibration_quantile"].is_monotonic_increasing)
        spread = quantile_spread(ranked, "rmse_quantile")
        self.assertEqual(spread["n_days"].tolist(), [3] * 5)
        self.assertTrue((spread["rmse_p25"] <= spread["rmse_median"]).all())
        self.assertTrue((spread["rmse_median"] <= spread["rmse_p75"]).all())
        counts = quantile_overlap(ranked, "rmse_quantile", "calibration_quantile", 5)
        self.assertEqual(counts.shape, (5, 5))
        self.assertEqual(int(counts.to_numpy().sum()), 15)
        info = overlap_summary(ranked, 5)
        self.assertEqual(info["days"], 15)
        self.assertEqual(info["best_rmse_days"], 3)
        self.assertAlmostEqual(info["expected_best_calibration_overlap_independent"], 3 * 3 / 15)
        self.assertEqual(info["best_rmse_and_best_calibration_days"], int(counts.loc[1, 1]))
        self.assertNotIn("calibration_error_quantile", ranked.columns)
        self.assertRaisesRegex(ValueError, "At least 25 full days",
            assign_quantiles, days, 25)

    def test_plot_and_csv_from_existing_forecasts_with_full_mpc_filter(self):
        frame = self.forecasts()
        with tempfile.TemporaryDirectory() as temp:
            with patch("src.exe.run_model.wm_4.analyze_daily_mpc_quantiles.load_forecasts",
                return_value=(frame, 0, "mpc_replay")) as loader, patch(
                "src.exe.run_model.wm_4.analyze_daily_mpc_quantiles.plot_quantiles",
                wraps=plot_quantiles) as plotter:
                result = analyze(forecasts=Path(temp) / "saved.csv",
                    output=Path(temp) / "reports", fully_mpc=True)
                loader.assert_called_once()
                plotter.assert_called_once()
                self.assertEqual(len(plotter.call_args.args[3]), 14 * 24)
            self.assertEqual(len(result["daily"]), 14)
            self.assertEqual(len(result["origins"]), 14 * 24)
            self.assertTrue((result["origins"]["n_predictions"] == 8).all())
            self.assertTrue((result["daily"]["n_predictions"] == 192).all())
            self.assertTrue(result["figure"].is_file())
            self.assertGreater(result["figure"].stat().st_size, 1000)
            output = result["output"]
            origins_csv = pd.read_csv(output / "mpc_origin_quality.csv")
            self.assertEqual(len(origins_csv), 14 * 24)
            self.assertEqual(len(origins_csv["origin"].unique()), 14 * 24)
            for name in ("daily_quality_quantiles.csv", "mpc_origin_quality.csv",
                "excluded_incomplete_days.csv",
                "rmse_vs_calibration_quantiles.csv", "rmse_vs_uncertainty_quantiles.csv",
                "rmse_quantile_spread.csv", "calibration_quantile_spread.csv",
                "uncertainty_quantile_spread.csv", "quantile_overlap_summary.json"):
                self.assertTrue((output / name).exists(), name)


if __name__ == "__main__": unittest.main()
