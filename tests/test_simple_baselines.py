"""Tests for the simple statistical baselines sharing the WM4 output interface."""
import unittest

import numpy as np
import pandas as pd

from src.tools.baseline import statistical_output, weekly_average, weekly_temperature


class SimpleBaselineTests(unittest.TestCase):

    @staticmethod
    def data():
        index = pd.date_range("2022-03-07", periods=28 * 24, freq="h")
        hour = index.hour.to_numpy()
        day = index.dayofweek.to_numpy()
        week = np.arange(len(index)) // (7 * 24)
        temp = 10.0 + 2.0 * np.sin(2 * np.pi * hour / 24) + 1.2 * week
        chill = np.maximum(0.0, 18.0 - temp)
        weekly = 0.26 + 0.018 * day + 0.004 * hour
        observed = weekly + 0.012 * (chill - 8.0)
        return pd.DataFrame({"I": observed, "Temp": temp}, index=index), weekly

    def test_average_is_weekday_hour_mean_and_ignores_temperature(self):
        frame, _ = self.data()
        cutoff = frame.index[21 * 24]
        model = weekly_average().fit(frame, cutoff)
        train = frame.loc[frame.index < cutoff]
        next_week = frame.loc[frame.index >= cutoff]
        group = (train.index.dayofweek * 24 + train.index.hour).to_numpy()
        expected = np.array([train["I"].to_numpy()[group == g].mean() for g in range(168)])
        self.assertTrue(np.allclose(model.coef, expected, atol=1e-12))
        first = model.predict(next_week.index, next_week["Temp"].to_numpy())
        second = model.predict(next_week.index, np.full(len(next_week), -20.0))
        self.assertTrue(np.allclose(first, second))
        self.assertTrue(np.allclose(first, model.levels[model._groups(next_week.index)]))
        self.assertEqual(model.metadata["features"], 168)
        self.assertGreaterEqual(model.metadata["mse"], 0.0)

    def test_temperature_fit_recovers_weekly_effects_and_linear_chill(self):
        frame, weekly = self.data()
        cutoff = frame.index[21 * 24]
        model = weekly_temperature().fit(frame, cutoff)
        prediction = model.predict(frame.index, frame["Temp"].to_numpy())
        self.assertAlmostEqual(model.beta_C, 0.012, places=11)
        self.assertTrue(np.allclose(prediction, frame["I"].to_numpy(), atol=1e-11))
        self.assertEqual(model.metadata["features"], 169)
        self.assertAlmostEqual(model.metadata["beta_C"], 0.012, places=11)
        self.assertLess(model.metadata["mse"], 1e-20)

    def test_fit_excludes_future_samples(self):
        frame, _ = self.data()
        cutoff = frame.index[21 * 24]
        changed = frame.copy()
        changed.loc[changed.index >= cutoff, "I"] = 0.99
        for cls in (weekly_average, weekly_temperature):
            reference = cls().fit(frame, cutoff)
            modified = cls().fit(changed, cutoff)
            self.assertTrue(np.allclose(reference.coef, modified.coef))
            self.assertEqual(reference.metadata["fit_end"], modified.metadata["fit_end"])
            self.assertLess(reference.metadata["fit_end"], cutoff)

    def test_subhour_prediction_is_valid_and_clipped(self):
        frame, _ = self.data()
        model = weekly_temperature().fit(frame, frame.index[21 * 24])
        index = pd.date_range("2022-03-28 06:00", periods=61, freq="min")
        prediction = model.predict(index, np.full(len(index), -1000.0))
        self.assertEqual(prediction.shape, (61,))
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertTrue(np.all((prediction >= 0.0) & (prediction <= 1.0)))

    def test_missing_weekday_hour_uses_hourly_fallback(self):
        frame, _ = self.data()
        cutoff = frame.index[21 * 24]
        partial = frame.loc[(frame.index < cutoff) & (frame.index.dayofweek != 0)]
        for cls in (weekly_average, weekly_temperature):
            model = cls().fit(partial, cutoff)
            index = pd.date_range("2022-03-28", periods=24, freq="h")
            result = model.predict(index, np.full(24, 10.0))
            self.assertTrue(np.all(np.isfinite(result)))
            self.assertEqual(result.shape, (24,))

    def test_constant_chill_has_zero_weakly_identified_coefficient(self):
        frame, _ = self.data()
        frame["Temp"] = 10.0
        model = weekly_temperature().fit(frame, frame.index[21 * 24])
        self.assertEqual(model.beta_C, 0.0)
        self.assertTrue(np.all(np.isfinite(model.predict(frame.index, frame["Temp"]))))

    def test_validation_and_common_fit_predict_metadata(self):
        frame, _ = self.data()
        cutoff = frame.index[21 * 24]
        for cls in (weekly_average, weekly_temperature):
            model = cls()
            with self.assertRaises(RuntimeError): model.predict(frame.index[:1], [12.0])
            with self.assertRaises(ValueError): model.fit(frame.iloc[:24], cutoff)
            with self.assertRaises(ValueError): model.fit(frame.drop(columns="Temp"), cutoff)
            model.fit(frame, cutoff)
            with self.assertRaises(ValueError): model.predict(frame.index[:2], [12.0])
            for key in ("fit_start", "fit_end", "samples", "features", "iterations", "ridge",
                        "huber_delta", "base_temperature", "temperature_min",
                        "temperature_max", "mse", "mae"):
                self.assertIn(key, model.metadata)

        # Existing advanced baseline remains exposed in the same module.
        self.assertTrue(callable(statistical_output().fit))


if __name__ == "__main__": unittest.main()
