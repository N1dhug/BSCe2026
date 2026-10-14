"""Daily 24-origin x six-lead WM4 MPC aggregation from saved diffusion QPs."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_daily_mpc_sigma_quantiles import (
    _summarize, analyze, daily_metrics, load_inputs,
)


class DailyWM4SigmaQuantilesTests(unittest.TestCase):
    @staticmethod
    def synthetic(days=12, windows=(7, 8, 9)):
        """A complete six-hour MPC CSV and independent historical QP results."""
        rng = np.random.default_rng(1234)
        daily = pd.date_range("2022-04-01", periods=days, freq="D")
        records, fits = [], []
        for d, day in enumerate(daily):
            for hour in range(24):
                origin = day + pd.Timedelta(hours=hour)
                error = rng.normal(0, 0.09 + 0.01 * np.sin(d), 6)
                B = np.zeros(6)
                K = np.arange(1, 7, dtype=float) * (0.05 + 0.005 * d)
                I = 0.43 + .005 * d + .001 * hour
                for lead in range(1, 7):
                    records.append({"origin": origin, "lead_hours": lead,
                        "target_time": origin + pd.Timedelta(hours=lead),
                        "I_observed": I + error[lead - 1], "I_mpc": I,
                        "initial_variance": B[lead - 1],
                        "process_kernel_unit": K[lead - 1],
                        "parameter_fit_hours": 24})
                for n in windows:
                    # Distinct nonnegative fit per origin and lookback.
                    if d < n - min(windows): continue
                    q = 0.018 + 0.004 * d + 0.002 * (n - min(windows)) + .0001 * hour
                    row = {"origin": origin, "history_days": n, "diffusion_q": q}
                    row.update({f"variance_h{lead}": q * K[lead - 1]
                        for lead in range(1, 7)})
                    fits.append(row)
        return pd.DataFrame(records), pd.DataFrame(fits)

    @staticmethod
    def save(tmp, forecasts, fits):
        path = Path(tmp)
        path.mkdir(parents=True, exist_ok=True)
        forecast_path, fits_path = path / "mpc_forecasts.csv", path / "fits.csv"
        forecasts.to_csv(forecast_path, index=False)
        fits.to_csv(fits_path, index=False)
        return fits_path, forecast_path

    def test_daily_means_are_144_forecast_squared_errors_and_variances(self):
        forecasts, fitted = self.synthetic(days=12)
        with tempfile.TemporaryDirectory() as tmp:
            source, path = self.save(tmp, forecasts, fitted)
            fits, archive, raw = load_inputs(source, path)
            daily, dropped = daily_metrics(fits, archive, raw)
        self.assertTrue(dropped.empty)
        self.assertTrue((daily["n_origins"] == 24).all())
        self.assertTrue((daily["n_forecasts"] == 144).all())
        self.assertEqual(daily.groupby("history_days").size().to_dict(),
            {7: 12, 8: 11, 9: 10})
        day = pd.Timestamp("2022-04-03")
        example = daily[(daily["history_days"] == 7) & (daily["date"] == day)].iloc[0]
        subset = forecasts.loc[forecasts["origin"].dt.normalize() == day]
        expected_errors = subset["I_observed"] - subset["I_mpc"]
        variance = fitted[(fitted["history_days"] == 7)
            & (fitted["origin"].dt.normalize() == day)]
        self.assertAlmostEqual(example["observed_mse"], np.mean(expected_errors ** 2))
        self.assertAlmostEqual(example["predicted_variance"],
            variance[[f"variance_h{h}" for h in range(1, 7)]].to_numpy().mean())
        self.assertAlmostEqual(example["mean_predicted_consumption"], subset["I_mpc"].mean())
        self.assertAlmostEqual(example["mean_observed_consumption"], subset["I_observed"].mean())
        # Pooling MSE != square of mean error or variance of the daily mean.
        self.assertGreater(example["observed_mse"], example["mean_forecast_error"] ** 2)

    def test_incomplete_origin_day_is_dropped_not_incorrectly_normalized(self):
        forecasts, fitted = self.synthetic(days=12)
        removed = (fitted["history_days"] == 7) & (
            fitted["origin"] == pd.Timestamp("2022-04-03 10:00"))
        fitted = fitted.loc[~removed].copy()
        with tempfile.TemporaryDirectory() as tmp:
            source, path = self.save(tmp, forecasts, fitted)
            daily, skipped = daily_metrics(*load_inputs(source, path))
        self.assertEqual(daily.groupby("history_days").size().to_dict(),
            {7: 11, 8: 11, 9: 10})
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped.iloc[0]["available_origins"], 23)
        self.assertFalse(((daily["date"] == pd.Timestamp("2022-04-03"))
            & (daily["history_days"] == 7)).any())

    def test_daily_quintiles_bin_calendar_days_not_hours_or_leads(self):
        forecasts, fitted = self.synthetic(days=12)
        with tempfile.TemporaryDirectory() as tmp:
            source, path = self.save(tmp, forecasts, fitted)
            daily, _ = daily_metrics(*load_inputs(source, path))
        overall, bins = _summarize(daily, 5)
        self.assertEqual(len(overall), 3)
        self.assertEqual(len(bins), 15)
        self.assertEqual(bins.groupby("history_days")["n_days"].sum().to_dict(),
            {7: 12, 8: 11, 9: 10})
        self.assertTrue((bins["n_origins"] == bins["n_days"] * 24).all())
        self.assertTrue((bins["n_forecasts"] == bins["n_days"] * 144).all())
        for n, part in bins.groupby("history_days"):
            part = part.sort_values("uncertainty_quantile")
            self.assertTrue(np.all(np.diff(part["predicted_variance"]) >= 0))
            global_mean = overall.loc[overall["history_days"] == n, "observed_mse"].iloc[0]
            self.assertAlmostEqual(np.average(part["observed_mse"], weights=part["n_days"]),
                global_mean)

    def test_reanalysis_saves_daily_quintiles_scatter_and_matches_same_days(self):
        forecasts, fitted = self.synthetic(days=12)
        with tempfile.TemporaryDirectory() as tmp:
            source, path = self.save(tmp, forecasts, fitted)
            output = Path(tmp) / "daily"
            result = analyze(source, path, output, quantiles=5)
            self.assertEqual(result["daily_metrics_all"]["date"].nunique(), 12)
            self.assertEqual(result["daily_metrics_matched"]["date"].nunique(), 10)
            self.assertEqual(len(result["daily_quantiles_all"]), 15)
            self.assertEqual(len(result["daily_quantiles_matched"]), 15)
            self.assertTrue((result["daily_summary_matched"]["n_days"] == 10).all())
            for name in ("daily_metrics_all.csv", "daily_metrics_matched.csv",
                "daily_quantiles_all.csv", "daily_quantiles_matched.csv",
                "daily_summary_all.csv", "daily_summary_matched.csv",
                "incomplete_days.csv", "all_daily_scatter.png", "all_daily_quintile_sweep.png",
                "all_daily_quantiles_reference.png", "all_daily_calibration.png",
                "matched_daily_scatter.png", "matched_daily_quintile_sweep.png",
                "matched_daily_quantiles_reference.png", "matched_daily_calibration.png"):
                self.assertTrue((output / name).exists(), name)

    def test_saved_fit_must_match_wiener_kernel_and_original_origin(self):
        forecasts, fitted = self.synthetic(days=12)
        fitted.loc[fitted.index[0], "variance_h1"] += 0.2
        with tempfile.TemporaryDirectory() as tmp:
            source, path = self.save(tmp, forecasts, fitted)
            with self.assertRaisesRegex(ValueError, "do not match diffusion fit"):
                daily_metrics(*load_inputs(source, path))
        fitted.loc[fitted.index[0], "variance_h1"] -= 0.2
        fitted.loc[fitted.index[0], "origin"] = pd.Timestamp("2020-01-01")
        with tempfile.TemporaryDirectory() as tmp:
            source, path = self.save(tmp, forecasts, fitted)
            with self.assertRaisesRegex(ValueError, "lack complete saved"):
                load_inputs(source, path)


if __name__ == "__main__": unittest.main()
