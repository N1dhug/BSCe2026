"""Regression tests: last 24 days, full 24-hour WM4 MPC sigma calibration."""

import unittest

import numpy as np
import pandas as pd

from src.model.wm_4 import wm_4
from src.exe.run_model.wm_4.predict_all_days_mpc import (
    _archive_mpc_origin, _archive_mpc_calibration, _fit_sigma_rolling_qp, _fit_stochastic_variance,
    _rolling_sigma_training, ROLLING_SIGMA_HORIZONS, ROLLING_SIGMA_DAYS,
)


class WM4RollingSigmaTests(unittest.TestCase):
    @staticmethod
    def archive(origin, model, r, days=24, steps_per_hour=2, q=0.035):
        idx = steps_per_hour * np.arange(1, 25)
        K = model.process_variance_kernel({"r": r}, idx) @ np.ones(model.t_steps)
        J = model.sensitivity({"r": r})["input"][idx]
        P0 = np.diag([0.004, 0.003])
        initial = np.einsum("ni,ij,nj->n", J, P0, J)
        z = np.arange(days, dtype=float)
        z = (z - np.mean(z)) / np.std(z, ddof=1)
        samples = {}
        for i in range(days):
            history = origin - pd.Timedelta(days=i + 1)
            e = z[i] * np.sqrt(initial + q * K)
            samples[history] = np.column_stack([e, initial, K])
        return samples, P0

    @staticmethod
    def model():
        model = wm_4((0.0, 1.0), 49, lambda t: 0.0)
        r = np.array([-2.0, -0.1, -3.0], dtype=float)
        return model, r

    def test_qp_fits_24_variances_from_24_daily_errors(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive, _ = self.archive(origin, model, r)
        training = _rolling_sigma_training(archive, origin)
        self.assertEqual(len(training["origins"]), 24)
        self.assertEqual(training["values"].shape, (24, 24, 3))
        fit, observed, initial, kernel = _fit_sigma_rolling_qp(model, training)
        self.assertEqual(observed.size, 24)
        self.assertTrue(np.allclose(observed, initial + 0.035 * kernel))
        self.assertAlmostEqual(fit["q"], 0.035, delta=0.0001)
        self.assertAlmostEqual(fit["sigma"], np.sqrt(0.035), delta=0.0002)

    def test_current_origin_or_incomplete_24h_path_cannot_enter_training(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive, _ = self.archive(origin, model, r)
        archive[origin] = np.zeros((24, 3))
        archive[origin - pd.Timedelta(hours=2)] = np.ones((24, 3))
        archive[origin - pd.Timedelta(days=1)] = archive[
            origin - pd.Timedelta(days=1)][:8]
        selected = _rolling_sigma_training(archive, origin)
        self.assertEqual(len(selected["origins"]), 23)
        self.assertNotIn(origin, selected["origins"])
        self.assertNotIn(origin - pd.Timedelta(hours=2), selected["origins"])
        self.assertNotIn(origin - pd.Timedelta(days=1), selected["origins"])

    def test_short_eight_hour_archive_cannot_replace_full_day(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive, _ = self.archive(origin, model, r)
        full = archive[origin - pd.Timedelta(days=1)].copy()
        rows = [{"lead_hours": h, "target_time": origin - pd.Timedelta(days=1) +
            pd.Timedelta(hours=h), "forecast_error": float(full[h - 1, 0]),
            "initial_variance": float(full[h - 1, 1]),
            "process_kernel_unit": float(full[h - 1, 2])} for h in range(1, 9)]
        _archive_mpc_origin(archive, origin - pd.Timedelta(days=1), rows, 8)
        self.assertTrue(np.array_equal(archive[origin - pd.Timedelta(days=1)], full))

    def test_full_sigma_fit_uses_daily_archive_but_weekly_P0(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive, _ = self.archive(origin, model, r)
        states = np.array([[0.04, 0.07], [-0.02, 0.09], [0.12, 0.0],
            [0.17, -0.03], [0.02, 0.15]])
        datasets = [{"I": np.ones(49) * 0.5, "O": np.ones(49) * 0.5} for _ in range(5)]
        fit = {"r": r, "prediction_origin": origin,
            "training_origins": [origin - pd.Timedelta(weeks=i) for i in range(1, 6)],
            "init_cond": states,
            "deterministic_states": [{"I": np.ones(49) * 0.5} for _ in range(5)]}
        result = _fit_stochastic_variance(model, fit, datasets, 2,
            sigma_fit_mode="mpc_replay_24d_24h", mpc_archive=archive,
            r0_variance_mode="oracle_reconstruction", max_horizon=8)
        self.assertEqual(result["calibration_horizon"], 24)
        self.assertEqual(result["calibration_mpc_count"], 24)
        self.assertEqual(result["calibration_oracle_count"], 0)
        self.assertEqual(result["sigma_rolling_count"], 24)
        self.assertTrue(result["sigma_rolling_complete"])
        self.assertAlmostEqual(result["q"], 0.035, delta=0.0001)
        self.assertEqual(result["mpc_initial_covariance"].shape, (2, 2))

    def test_six_hour_rolling_fit_uses_24_days_and_six_sample_variances(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive, _ = self.archive(origin, model, r)
        training = _rolling_sigma_training(archive, origin, horizon=6)
        self.assertEqual(len(training["origins"]), 24)
        self.assertEqual(training["values"].shape, (24, 6, 3))
        fit, observed, initial, response = _fit_sigma_rolling_qp(
            model, training, horizon=6)
        self.assertEqual(len(observed), 6)
        self.assertTrue(np.allclose(observed, initial + 0.035 * response))
        self.assertAlmostEqual(fit["q"], 0.035, delta=0.0001)
        self.assertEqual(ROLLING_SIGMA_HORIZONS["mpc_replay_24d_6h"], 6)

    def test_six_hour_mode_retains_eight_hour_forecast_and_weekly_P0(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive, _ = self.archive(origin, model, r)
        states = np.array([[0.04, 0.07], [-0.02, 0.09], [0.12, 0.0],
            [0.17, -0.03], [0.02, 0.15]])
        datasets = [{"I": np.ones(49) * 0.5, "O": np.ones(49) * 0.5} for _ in range(5)]
        fit = {"r": r, "prediction_origin": origin,
            "training_origins": [origin - pd.Timedelta(weeks=i) for i in range(1, 6)],
            "init_cond": states,
            "deterministic_states": [{"I": np.ones(49) * 0.5} for _ in range(5)]}
        result = _fit_stochastic_variance(model, fit, datasets, 2,
            sigma_fit_mode="mpc_replay_24d_6h", mpc_archive=archive,
            r0_variance_mode="oracle_reconstruction", max_horizon=8)
        self.assertEqual(result["calibration_horizon"], 6)
        self.assertEqual(result["calibration_mpc_count"], 24)
        self.assertEqual(result["sigma_rolling_count"], 24)
        self.assertEqual(result["calibration_oracle_count"], 0)
        self.assertTrue(result["sigma_rolling_complete"])
        self.assertAlmostEqual(result["q"], 0.035, delta=0.0001)
        self.assertEqual(result["mpc_initial_covariance"].shape, (2, 2))

    def test_six_hour_archive_does_not_require_horizon_24(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive, _ = self.archive(origin, model, r)
        for stamp in archive: archive[stamp] = archive[stamp][:6].copy()
        selected = _rolling_sigma_training(archive, origin, horizon=6)
        self.assertEqual(len(selected["origins"]), 24)
        long = _rolling_sigma_training(archive, origin, horizon=24)
        self.assertEqual(len(long["origins"]), 0)
        incomplete = archive.copy()
        first = origin - pd.Timedelta(days=1)
        incomplete[first] = archive[first][:5].copy()
        self.assertEqual(len(_rolling_sigma_training(incomplete, origin, horizon=6)["origins"]), 23)
        self.assertNotIn(first, _rolling_sigma_training(incomplete, origin, horizon=6)["origins"])

    def test_six_hour_variance_fit_excludes_later_error_misspecification(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive, _ = self.archive(origin, model, r)
        for val in archive.values():
            val[6:, 0] *= 5.0
        six, _, _, _ = _fit_sigma_rolling_qp(model,
            _rolling_sigma_training(archive, origin, horizon=6), horizon=6)
        full, _, _, _ = _fit_sigma_rolling_qp(model,
            _rolling_sigma_training(archive, origin, horizon=24), horizon=24)
        self.assertAlmostEqual(six["q"], 0.035, delta=0.0001)
        self.assertGreater(full["q"], six["q"])

    def test_32_day_clip4_limits_four_extreme_squared_errors_at_each_lead(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive, _ = self.archive(origin, model, r, days=32)
        for i, values in enumerate(archive.values()):
            values = values[:6].copy()
            if i == 0: values[:, 0] += 30.0
            if i == 1: values[:, 0] -= 20.0
            archive[origin - pd.Timedelta(days=i + 1)] = values
        rolling = _rolling_sigma_training(archive, origin, days=32, horizon=6)
        self.assertEqual(rolling["values"].shape, (32, 6, 3))
        normal, naive, base, kernel = _fit_sigma_rolling_qp(model, rolling, horizon=6)
        robust, clipped, robust_base, robust_kernel = _fit_sigma_rolling_qp(
            model, rolling, horizon=6, clip_each_tail=4)
        errors = rolling["values"][:, :, 0]
        sq = (errors - errors.mean(axis=0)) ** 2
        ordered = np.sort(sq, axis=0)
        expected = np.clip(sq, ordered[4], ordered[-5]).sum(axis=0) / 31.0
        self.assertTrue(np.allclose(clipped, expected))
        self.assertTrue(np.all(clipped < naive))
        self.assertTrue(np.allclose(base, robust_base))
        self.assertTrue(np.allclose(kernel, robust_kernel))
        self.assertLess(robust["q"], normal["q"])
        self.assertEqual(ROLLING_SIGMA_HORIZONS["mpc_replay_32d_6h_clip4"], 6)
        self.assertEqual(ROLLING_SIGMA_DAYS["mpc_replay_32d_6h_clip4"], 32)
        with self.assertRaisesRegex(ValueError, "clip_each_tail"):
            _fit_sigma_rolling_qp(model, rolling, horizon=6, clip_each_tail=16)

    def test_32_day_mode_requires_full_history_and_reports_clip_metadata(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive, _ = self.archive(origin, model, r, days=32)
        for stamp in archive: archive[stamp] = archive[stamp][:6].copy()
        states = np.array([[0.04, 0.07], [-0.02, 0.09], [0.12, 0.0],
            [0.17, -0.03], [0.02, 0.15]])
        datasets = [{"I": np.ones(49) * 0.5, "O": np.ones(49) * 0.5} for _ in range(5)]
        fit = {"r": r, "prediction_origin": origin,
            "training_origins": [origin - pd.Timedelta(weeks=i) for i in range(1, 6)],
            "init_cond": states,
            "deterministic_states": [{"I": np.ones(49) * 0.5} for _ in range(5)]}
        sigma = _fit_stochastic_variance(model, fit, datasets, 2,
            sigma_fit_mode="mpc_replay_32d_6h_clip4", mpc_archive=archive,
            max_horizon=6)
        self.assertEqual(sigma["calibration_horizon"], 6)
        self.assertEqual(sigma["calibration_mpc_count"], 32)
        self.assertEqual(sigma["sigma_rolling_days"], 32)
        self.assertTrue(sigma["sigma_rolling_complete"])
        self.assertFalse(sigma["sigma_rolling_bootstrap"])
        self.assertEqual(sigma["sigma_variance_clip_each_tail"], 4)
        missing = archive.copy()
        missing.pop(origin - pd.Timedelta(days=1))
        boot = _fit_stochastic_variance(model, fit, datasets, 2,
            sigma_fit_mode="mpc_replay_32d_6h_clip4", mpc_archive=missing,
            max_horizon=6)
        self.assertEqual(boot["sigma_rolling_count"], 31)
        self.assertFalse(boot["sigma_rolling_complete"])
        self.assertTrue(boot["sigma_rolling_bootstrap"])
        self.assertEqual(boot["sigma_variance_clip_each_tail"], 0)

    def test_28_day_sigma_uses_only_real_six_hour_mpc_errors(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive, _ = self.archive(origin, model, r, days=28)
        # After hour 6, artificially magnify errors. The six-hour QP
        # must be unaffected (the 24-hour estimator would not be).
        for values in archive.values(): values[6:, 0] *= 25.0
        training = _rolling_sigma_training(archive, origin, days=28, horizon=6)
        self.assertEqual(len(training["origins"]), 28)
        self.assertEqual(training["values"].shape, (28, 6, 3))
        fitted, observed, initial, kernel = _fit_sigma_rolling_qp(
            model, training, horizon=6)
        self.assertEqual(observed.shape, (6,))
        self.assertAlmostEqual(fitted["q"], 0.035, delta=0.0001)
        self.assertTrue(np.allclose(observed, initial + 0.035 * kernel))
        self.assertEqual(ROLLING_SIGMA_HORIZONS["mpc_replay_28d_6h"], 6)
        self.assertEqual(ROLLING_SIGMA_DAYS["mpc_replay_28d_6h"], 28)

        states = np.array([[0.04, 0.07], [-0.02, 0.09], [0.12, 0.0],
            [0.17, -0.03], [0.02, 0.15]])
        datasets = [{"I": np.ones(49) * 0.5, "O": np.ones(49) * 0.5}
            for _ in range(5)]
        fit = {"r": r, "prediction_origin": origin,
            "training_origins": [origin - pd.Timedelta(weeks=i) for i in range(1, 6)],
            "init_cond": states,
            "deterministic_states": [{"I": np.ones(49) * 0.5} for _ in range(5)]}
        sigma = _fit_stochastic_variance(model, fit, datasets, 2,
            sigma_fit_mode="mpc_replay_28d_6h", mpc_archive=archive,
            max_horizon=6)
        self.assertEqual(sigma["calibration_horizon"], 6)
        self.assertEqual(sigma["calibration_mpc_count"], 28)
        self.assertEqual(sigma["sigma_rolling_days"], 28)
        self.assertTrue(sigma["sigma_rolling_complete"])
        self.assertAlmostEqual(sigma["q"], 0.035, delta=0.0001)

        incomplete_archive = dict(archive)
        incomplete_archive.pop(origin - pd.Timedelta(days=1))
        boot = _fit_stochastic_variance(model, fit, datasets, 2,
            sigma_fit_mode="mpc_replay_28d_6h", mpc_archive=incomplete_archive,
            max_horizon=6)
        self.assertEqual(boot["sigma_rolling_count"], 27)
        self.assertTrue(boot["sigma_rolling_bootstrap"])
        self.assertFalse(boot["sigma_rolling_complete"])
        self.assertEqual(boot["calibration_horizon"], 6)

    def test_thirty_day_six_hour_mode_uses_only_six_mpc_leads(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive, _ = self.archive(origin, model, r, days=30)
        # The old model may forecast 24h, but the new fit ONLY sees 1..6h.
        for row in archive.values(): row[6:, 0] *= 20.0
        samples = _rolling_sigma_training(archive, origin, days=30, horizon=6)
        self.assertEqual(samples["values"].shape, (30, 6, 3))
        qp, observed, initial, kernel = _fit_sigma_rolling_qp(model, samples, horizon=6)
        self.assertEqual(len(observed), 6)
        self.assertTrue(np.allclose(observed, initial + 0.035 * kernel))
        self.assertAlmostEqual(qp["q"], 0.035, delta=0.0001)
        self.assertEqual(ROLLING_SIGMA_HORIZONS["mpc_replay_30d_6h"], 6)
        # A 31st older day is never selected; the current origin is never used.
        archive[origin - pd.Timedelta(days=31)] = np.full((6, 3), 500.0)
        archive[origin] = np.full((6, 3), 500.0)
        fresh = _rolling_sigma_training(archive, origin, days=30, horizon=6)
        self.assertEqual(len(fresh["origins"]), 30)
        self.assertNotIn(origin, fresh["origins"])
        self.assertNotIn(origin - pd.Timedelta(days=31), fresh["origins"])

    def test_six_hour_archive_retains_only_six_forecast_leads(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive = {}
        state = {"r": r, "I": np.linspace(0.3, 0.8, model.t_steps)}
        prepared = {"I": np.linspace(0.4, 0.9, model.t_steps)}
        fit = {"r": r}
        diffusion = {"q": 0.035, "mpc_initial_covariance": np.diag([0.004, 0.003])}
        _archive_mpc_calibration(archive, origin, model, state,
            prepared, fit, diffusion, steps_per_hour=2, horizon=6)
        self.assertEqual(archive[origin].shape, (6, 3))
        self.assertTrue(np.allclose(archive[origin][:, 0], 0.1))
        self.assertTrue(np.all(archive[origin][:, 1:] >= 0))

    def test_thirty_day_mode_defaults_to_thirty_complete_historical_mpcs(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        archive, _ = self.archive(origin, model, r, days=30)
        states = np.array([[0.04, 0.07], [-0.02, 0.09], [0.12, 0.0],
            [0.17, -0.03], [0.02, 0.15]])
        datasets = [{"I": np.ones(49) * 0.5, "O": np.ones(49) * 0.5} for _ in range(5)]
        fit = {"r": r, "prediction_origin": origin,
            "training_origins": [origin - pd.Timedelta(weeks=i) for i in range(1, 6)],
            "init_cond": states,
            "deterministic_states": [{"I": np.ones(49) * 0.5} for _ in range(5)]}
        fitted = _fit_stochastic_variance(model, fit, datasets, 2,
            sigma_fit_mode="mpc_replay_30d_6h", mpc_archive=archive,
            r0_variance_mode="oracle_reconstruction", max_horizon=8)
        self.assertEqual(fitted["calibration_horizon"], 6)
        self.assertEqual(fitted["calibration_mpc_count"], 30)
        self.assertEqual(fitted["calibration_oracle_count"], 0)
        self.assertEqual(fitted["sigma_rolling_days"], 30)
        self.assertEqual(fitted["sigma_rolling_count"], 30)
        self.assertTrue(fitted["sigma_rolling_complete"])
        self.assertFalse(fitted["sigma_rolling_bootstrap"])
        self.assertAlmostEqual(fitted["q"], 0.035, delta=0.0001)
        self.assertEqual(fitted["mpc_initial_covariance"].shape, (2, 2))

    def test_insufficient_history_bootstraps_eight_hour_weekly_fit(self):
        origin = pd.Timestamp("2022-08-01 14:00")
        model, r = self.model()
        states = np.array([[0.04, 0.07], [-0.02, 0.09], [0.12, 0.0],
            [0.17, -0.03], [0.02, 0.15]])
        datasets = [{"I": np.ones(49) * 0.5, "O": np.ones(49) * 0.5} for _ in range(5)]
        fit = {"r": r, "prediction_origin": origin,
            "training_origins": [origin - pd.Timedelta(weeks=i) for i in range(1, 6)],
            "init_cond": states,
            "deterministic_states": [{"I": np.ones(49) * 0.5} for _ in range(5)]}
        result = _fit_stochastic_variance(model, fit, datasets, 2,
            sigma_fit_mode="mpc_replay_24d_24h", mpc_archive={})
        self.assertTrue(result["sigma_rolling_bootstrap"])
        self.assertEqual(result["calibration_horizon"], 8)
        self.assertFalse(result["sigma_rolling_complete"])


if __name__ == "__main__": unittest.main()
