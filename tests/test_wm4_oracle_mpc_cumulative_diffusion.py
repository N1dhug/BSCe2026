"""Regression tests for cumulative 24-hour oracle diffusion fitting at individual MPC origins."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_oracle_mpc_cumulative_diffusion import (
    NOMINAL_68_PCT, NOMINAL_95_PCT, fit_all_origins, fit_cumulative_origin,
    fit_nonnegative_variances, fit_historical_variances, fit_all_origins_historical,
    day_error_quintiles, save_day_error_quintiles,
    horizon_summary, load_predictions, save_calibration_plot, save_plot,
)


class WM4OracleMPCCumulativeDiffusionTests(unittest.TestCase):
    @staticmethod
    def forecast(origin="2022-05-01 14:00", q=0.04, bias=0.0, q0=0.0):
        origin = pd.Timestamp(origin)
        h = np.arange(1, 25, dtype=int)
        k = 0.0005 + 0.0003 * h
        # With zero bias squared errors match observation variance plus q*K.
        error = np.sqrt(q0 + q * k) + bias
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

    def test_separates_observation_and_wiener_diffusion(self):
        fit, hourly = fit_cumulative_origin(self.forecast(q0=0.003, q=0.09))
        self.assertAlmostEqual(fit["q_0"], 0.003, places=10)
        self.assertAlmostEqual(fit["q_p"], 0.09, places=10)
        self.assertAlmostEqual(fit["sigma_0"], np.sqrt(0.003), places=10)
        self.assertAlmostEqual(fit["sigma_p"], 0.3, places=10)
        self.assertTrue(np.allclose(
            hourly["observation_variance"] + hourly["process_variance"],
            hourly["total_variance"],
        ))
        self.assertTrue(np.allclose(hourly["total_std"] ** 2, hourly["total_variance"]))
        self.assertTrue(np.allclose(
            hourly["fitted_cumulative_total_variance"], hourly["cumulative_error_squared"],
        ))
        self.assertTrue(np.allclose(
            hourly["upper_95"] - hourly["I_oracle"], 1.959963984540054 * hourly["total_std"],
        ))
        self.assertAlmostEqual(fit["cumulative_curve_mse"], 0.0, places=12)
        self.assertAlmostEqual(fit["total_95_coverage_pct"], fit["process_95_coverage_pct"])

    def test_QP_axis_optima(self):
        frame = self.forecast()
        cum_k = np.cumsum(frame["unit_process_variance"].to_numpy(dtype=float))
        q0, qp = fit_nonnegative_variances(np.arange(1, 25) * 0.002, cum_k)
        self.assertAlmostEqual(q0, 0.002, places=12)
        self.assertAlmostEqual(qp, 0.0, places=12)
        q0, qp = fit_nonnegative_variances(0.06 * cum_k, cum_k)
        self.assertAlmostEqual(q0, 0.0, places=12)
        self.assertAlmostEqual(qp, 0.06, places=12)

    def test_unidentifiable_two_component_kernel_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "separately identified"):
            fit_nonnegative_variances(np.arange(1, 25), np.arange(1, 25) * 0.04)

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
        self.assertTrue(np.allclose(origin_fits["sigma_0"], 0.0))
        summary = horizon_summary(hourly)
        self.assertEqual(summary["lead_hours"].tolist(), list(range(1, 25)))
        self.assertTrue(np.allclose(
            summary["mean_cumulative_squared_error"],
            summary["mean_cumulative_model_variance"],
        ))

    def test_coverage_and_empirical_variance_are_aggregated_by_lead(self):
        # Use four origins with known process variance (not re-estimated from
        # their own errors) to test numerical calibration diagnostics directly.
        errors = [0.0, 0.5, 1.5, 3.0]
        chunks = []
        for i, error in enumerate(errors):
            origin = pd.Timestamp("2022-05-01") + pd.Timedelta(hours=i)
            hours = np.arange(1, 25)
            chunks.append(pd.DataFrame({
                "origin": origin, "lead_hours": hours,
                "oracle_error": np.full(24, error),
                "cumulative_error_squared": hours * error ** 2,
                "fitted_cumulative_process_variance": hours.astype(float),
                "fitted_cumulative_total_variance": hours.astype(float),
                "observation_variance": np.zeros(24),
                "total_variance": np.ones(24),
                "process_variance": np.ones(24),
                "covered_68": np.full(24, abs(error) <= 1.0),
                "covered_95": np.full(24, abs(error) <= 1.959963984540054),
            }))
        summary = horizon_summary(pd.concat(chunks, ignore_index=True))
        self.assertEqual(len(summary), 24)
        self.assertTrue(np.allclose(summary["empirical_oracle_variance"], np.var(errors, ddof=1)))
        self.assertTrue(np.allclose(summary["mean_predicted_process_variance"], 1.0))
        self.assertTrue(np.allclose(summary["mean_predicted_observation_variance"], 0.0))
        self.assertTrue(np.allclose(summary["mean_predicted_total_variance"], 1.0))
        self.assertTrue(np.allclose(summary["process_68_coverage_pct"], 50.0))
        self.assertTrue(np.allclose(summary["process_95_coverage_pct"], 75.0))
        self.assertTrue(np.allclose(summary["coverage_68_ratio"], 50.0 / NOMINAL_68_PCT))
        self.assertTrue(np.allclose(summary["coverage_95_ratio"], 75.0 / NOMINAL_95_PCT))
        self.assertTrue(np.allclose(summary["coverage_68_error_pp"], 50.0 - NOMINAL_68_PCT))
        self.assertTrue(np.allclose(summary["coverage_95_error_pp"], 75.0 - NOMINAL_95_PCT))
        self.assertEqual(summary["covered_68_origins"].tolist(), [2] * 24)
        self.assertEqual(summary["covered_95_origins"].tolist(), [3] * 24)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "variance_coverage.png"
            save_calibration_plot(summary, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_fitted_intervals_include_68_and_95_coverage_flags(self):
        fit, hourly = fit_cumulative_origin(self.forecast())
        self.assertTrue(np.all(hourly["covered_95"]))
        self.assertIn("covered_68", hourly.columns)
        self.assertAlmostEqual(fit["process_68_coverage_pct"],
            100.0 * float(hourly["covered_68"].mean()))
        self.assertAlmostEqual(fit["process_95_coverage_pct"], 100.0)
        self.assertTrue(np.allclose(
            np.abs(hourly["oracle_error"]), hourly["process_std"], atol=1e-12,
        ))
        self.assertTrue(np.allclose(
            hourly["upper_68"] - hourly["I_oracle"], hourly["process_std"],
        ))
        self.assertTrue(np.all(
            hourly["upper_95"] - hourly["I_oracle"] > hourly["upper_68"] - hourly["I_oracle"]
        ))

    @staticmethod
    def daily_quintile_forecasts(include_noon=False):
        predicted = np.array([0.11, 0.05, 0.13, 0.07, 0.09, 0.03, 0.15, 0.12, 0.06, 0.10])
        actual = np.array([0.04, 0.15, 0.05, 0.12, 0.08, 0.16, 0.03, 0.09, 0.14, 0.11])
        rows = []
        for index, day in enumerate(pd.date_range("2022-05-01", periods=10, freq="D")):
            for hour in ([0, 12] if include_noon else [0]):
                origin = day + pd.Timedelta(hours=hour)
                for lead in range(1, 25):
                    # Midday origins deliberately have different metrics.
                    predicted_std = predicted[index] if hour == 0 else 0.20
                    observed_err = actual[index] if hour == 0 else 0.01
                    rows.append({
                        "origin": origin, "lead_hours": lead,
                        "oracle_error": observed_err,
                        "total_variance": predicted_std ** 2,
                    })
        return pd.DataFrame(rows)

    def test_predicted_quintiles_rank_days_not_actual_errors(self):
        daily, quintiles = day_error_quintiles(self.daily_quintile_forecasts(), mode="midnight")
        self.assertEqual(daily["quintile_name"].unique().tolist(),
            ["Lowest", "Low", "Neutral", "High", "Highest"])
        self.assertEqual(quintiles["days"].tolist(), [2] * 5)
        self.assertEqual(daily["origins"].tolist(), [1] * 10)
        self.assertEqual(daily["evaluated_hours"].tolist(), [24] * 10)
        self.assertTrue(daily["predicted_24h_rms_std"].is_monotonic_increasing)
        self.assertAlmostEqual(daily.iloc[0]["predicted_24h_rms_std"], 0.03)
        self.assertAlmostEqual(daily.iloc[0]["actual_24h_rmse"], 0.16)
        self.assertAlmostEqual(quintiles.iloc[0]["predicted_24h_rms_std"],
            np.sqrt(np.mean([0.03 ** 2, 0.05 ** 2])))
        self.assertAlmostEqual(quintiles.iloc[0]["actual_24h_rmse"],
            np.sqrt(np.mean([0.16 ** 2, 0.15 ** 2])))
        self.assertLess(quintiles.iloc[0]["predicted_24h_rms_std"],
            quintiles.iloc[-1]["predicted_24h_rms_std"])
        self.assertGreater(quintiles.iloc[0]["actual_24h_rmse"],
            quintiles.iloc[-1]["actual_24h_rmse"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "day_error_quintiles.png"
            save_day_error_quintiles(daily, quintiles, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_midnight_vs_all_origin_calendar_day_aggregation(self):
        hourly = self.daily_quintile_forecasts(include_noon=True)
        midnight, _ = day_error_quintiles(hourly, mode="midnight")
        all_origins, quintiles = day_error_quintiles(hourly, mode="all-origins")
        self.assertEqual(midnight["origins"].tolist(), [1] * 10)
        self.assertEqual(all_origins["origins"].tolist(), [2] * 10)
        self.assertEqual(all_origins["evaluated_hours"].tolist(), [48] * 10)
        self.assertNotAlmostEqual(
            midnight["predicted_24h_rms_std"].mean(),
            all_origins["predicted_24h_rms_std"].mean(),
        )
        self.assertEqual(quintiles["days"].sum(), 10)

    def test_calendar_day_uses_previous_day_23h_forecast(self):
        rows = []
        for i, day in enumerate(pd.date_range("2022-05-01", periods=10, freq="D")):
            origin = day - pd.Timedelta(hours=1)
            for h in range(1, 25):
                rows.append({
                    "origin": origin, "lead_hours": h,
                    "oracle_error": (i + 1) * 0.01,
                    "total_variance": ((10 - i) * 0.01) ** 2,
                })
        daily, quintiles = day_error_quintiles(pd.DataFrame(rows))
        self.assertEqual(len(daily), 10)
        self.assertEqual(daily["day"].dt.strftime("%Y-%m-%d").min(), "2022-05-01")
        self.assertEqual(daily["day"].dt.strftime("%Y-%m-%d").max(), "2022-05-10")
        self.assertEqual(daily["origins"].unique().tolist(), [1])
        self.assertEqual(quintiles["days"].tolist(), [2] * 5)

    def test_day_quintiles_reject_too_few_days_and_partial_forecasts(self):
        hourly = self.daily_quintile_forecasts()
        with self.assertRaisesRegex(ValueError, "At least five"):
            day_error_quintiles(hourly[hourly["origin"] < pd.Timestamp("2022-05-05")], mode="midnight")
        partial = hourly.drop(hourly.index[0])
        with self.assertRaisesRegex(ValueError, "complete 24-hour"):
            day_error_quintiles(partial, mode="midnight")


    def test_walk_forward_uses_only_prior_completed_days(self):
        rows = []
        start = pd.Timestamp("2022-05-01 23:00")
        for i in range(9):
            origin = start + pd.Timedelta(days=i)
            # Historical q values are constant, while validation days are deliberately extreme.
            q0, qp = (0.002, 0.05) if i < 7 else (0.20, 0.80)
            source = self.forecast(origin=origin, q0=q0, q=qp)
            rows.append(source)
        frames = pd.concat(rows, ignore_index=True)
        fits, hourly, skipped = fit_all_origins_historical(
            frames, lookback_days=7, min_training_days=5,
            start_day="2022-05-09", end_day="2022-05-09",
        )
        self.assertEqual(len(fits), 2)
        self.assertEqual(skipped, 0)
        self.assertAlmostEqual(fits.iloc[0]["q_0"], 0.002)
        self.assertAlmostEqual(fits.iloc[0]["q_p"], 0.05)
        self.assertGreater(fits.iloc[1]["q_0"] + fits.iloc[1]["q_p"],
            fits.iloc[0]["q_0"] + fits.iloc[0]["q_p"])
        self.assertTrue(np.all(fits["training_last_observation"] <= fits["origin"]))
        self.assertEqual(fits["fit_mode"].unique().tolist(), ["historical"])
        self.assertTrue(np.all(hourly["fit_mode"].eq("historical")))
        self.assertTrue(np.all(hourly["training_last_observation"] <= hourly["origin"]))
        # The validation day's squared errors may be large, but MUST NOT change its q.
        self.assertGreater(fits.iloc[-1]["observed_24h_rmse"],
            fits.iloc[-1]["model_24h_rms_std"])

    def test_walk_forward_rejects_target_error_leakage(self):
        start = pd.Timestamp("2022-05-01 23:00")
        past = [self.forecast(start + pd.Timedelta(days=i), q0=0.003, q=0.04)
            for i in range(6)]
        target = self.forecast(start + pd.Timedelta(days=6), q0=0.1, q=0.8)
        frames = pd.concat(past + [target], ignore_index=True)
        first, _, _ = fit_all_origins_historical(frames, lookback_days=6,
            min_training_days=5, start_day="2022-05-08", end_day="2022-05-08")
        modified = frames.copy()
        mask = modified["origin"].eq(start + pd.Timedelta(days=6))
        modified.loc[mask, "oracle_error"] *= 20
        modified.loc[mask, "I_observed"] = modified.loc[mask, "I_oracle"] + modified.loc[mask, "oracle_error"]
        second, _, _ = fit_all_origins_historical(modified, lookback_days=6,
            min_training_days=5, start_day="2022-05-07", end_day="2022-05-07")
        self.assertAlmostEqual(first.iloc[0]["q_0"], second.iloc[0]["q_0"])
        self.assertAlmostEqual(first.iloc[0]["q_p"], second.iloc[0]["q_p"])
        self.assertNotAlmostEqual(first.iloc[0]["observed_24h_rmse"],
            second.iloc[0]["observed_24h_rmse"])

    def test_historical_diffusion_pooled_QP_recovers_both_components(self):
        histories = [self.forecast("2022-05-01 23:00", q0=0.005, q=0.04),
            self.forecast("2022-05-02 23:00", q0=0.005, q=0.04)]
        q0, qp = fit_historical_variances(histories)
        self.assertAlmostEqual(q0, 0.005, places=10)
        self.assertAlmostEqual(qp, 0.04, places=10)

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
