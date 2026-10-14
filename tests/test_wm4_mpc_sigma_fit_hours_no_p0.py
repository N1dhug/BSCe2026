"""Regression tests: 25-day, 24h historical MPC and 6..24h Wiener QP sweep."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.model.wm_4 import wm_4
from src.exe.run_model.wm_4.predict_all_days_mpc import _archive_mpc_calibration
from src.exe.run_model.wm_4.sweep_mpc_sigma_fit_hours_no_p0 import (
    ARCHIVE_HOURS, BACKLOG_DAYS, FORECAST_HOURS, MODE,
    fit_diffusion_horizons, replay, run, six_hour_evaluation, summarize,
)


class WM4SixToTwentyFourHourSigmaTests(unittest.TestCase):
    @staticmethod
    def fake_data(n_days=40):
        """Each day has the same hourly origin, with observed 24h error paths."""
        rng = np.random.default_rng(2026)
        archive, rows = {}, []
        for origin in pd.date_range("2022-03-01 12:00", periods=n_days, freq="D"):
            K = .001 + .0005 * np.arange(1, ARCHIVE_HOURS + 1)
            e = rng.normal(size=ARCHIVE_HOURS) * np.sqrt(.025 * K)
            archive[origin] = np.column_stack((e, np.zeros(ARCHIVE_HOURS), K))
            for lead in range(1, FORECAST_HOURS + 1):
                rows.append({"origin": origin, "lead_hours": lead,
                    "I_observed": .5 + e[lead - 1], "I_mpc": .5,
                    "initial_variance": 0., "process_kernel_unit": K[lead - 1],
                    "parameter_fit_hours": 24, "sigma_fit_mode": MODE})
        return archive, pd.DataFrame(rows)

    def test_actual_24h_calibration_archive_contains_24_historical_mpc_leads(self):
        origin = pd.Timestamp("2022-07-01 14:00")
        model = wm_4((0., 1.), 49, lambda t: 0.)
        r = np.array([-2., -0.1, -3.], dtype=float)
        state = {"r": r, "I": np.full(49, 0.5)}
        prepared = {"I": np.full(49, 0.52)}
        fit = {"r": r}
        diffusion = {"mpc_initial_covariance": np.zeros((2, 2)), "q": 0.04}
        archive = {}
        _archive_mpc_calibration(archive, origin, model, state, prepared,
            fit, diffusion, steps_per_hour=2, horizon=24)
        self.assertEqual(archive[origin].shape, (24, 3))
        self.assertTrue(np.allclose(archive[origin][:, 0], .02))
        self.assertTrue(np.allclose(archive[origin][:, 1], 0.))
        self.assertGreater(float(archive[origin][-1, 2]), 0.)
        self.assertGreater(float(archive[origin][-1, 2]),
            float(archive[origin][5, 2]))

    def test_6_to_24_hour_qp_matches_explicit_lead_limited_least_squares(self):
        archive, _ = self.fake_data(40)
        origin = list(archive)[-1]
        history = np.stack([archive[origin - pd.Timedelta(days=d)]
            for d in range(1, 26)])
        hours, qs, S, K = fit_diffusion_horizons(history)
        self.assertEqual(len(history), BACKLOG_DAYS)
        self.assertEqual(hours.tolist(), list(range(6, 25)))
        self.assertTrue((qs >= 0.).all())
        self.assertTrue(np.all(np.isfinite(qs)))
        for h, q in zip(hours, qs):
            expected = max(0., np.dot(S[:h], K[:h]) /
                (np.dot(K[:h], K[:h]) + 1e-10 * h / 2.))
            self.assertAlmostEqual(q, expected, places=12)
        with self.assertRaisesRegex(ValueError, "initial-state variance"):
            altered = history.copy()
            altered[:, 0, 1] = .02
            fit_diffusion_horizons(altered)

    def test_25_previous_day_24h_paths_are_used_for_every_horizon(self):
        archive, forecasts = self.fake_data(40)
        evaluations = six_hour_evaluation(forecasts)
        fits = replay(archive, evaluations, days=25)
        self.assertEqual(fits["calibration_hours"].unique().tolist(),
            list(range(6, 25)))
        self.assertEqual(fits["origin"].nunique(), 15)
        self.assertEqual(len(fits), 15 * 19)
        self.assertTrue((fits["history_days"] == 25).all())
        self.assertTrue((fits["n_history_mpcs"] == 25).all())
        self.assertTrue((fits["diffusion_q"] >= 0.).all())
        self.assertTrue((fits["predicted_variance"] >= 0.).all())
        for _, row in fits.iterrows():
            origin = row["origin"]
            q = row["diffusion_q"]
            for h in range(1, 7):
                self.assertAlmostEqual(row[f"variance_h{h}"],
                    q * evaluations[origin][h - 1, 1], places=12)
        score, quintiles, lead = summarize(fits, evaluations)
        self.assertEqual(len(score), 19)
        self.assertEqual(len(quintiles), 19 * 5)
        self.assertEqual(len(lead), 19 * 6)
        self.assertTrue((score["n_origins"] == 15).all())
        self.assertEqual(score["observed_rmse"].nunique(), 1)
        self.assertTrue((quintiles["n_origins"] == 3).all())

    def test_missing_24_hour_history_does_not_use_partial_mpc_errors(self):
        archive, forecasts = self.fake_data(40)
        evaluations = six_hour_evaluation(forecasts)
        target = list(archive)[-1]
        broken = archive.copy()
        broken.pop(target - pd.Timedelta(days=1))
        fits = replay(broken, evaluations, days=25)
        self.assertNotIn(target, fits["origin"].tolist())
        short = archive.copy()
        short[target - pd.Timedelta(days=1)] = short[
            target - pd.Timedelta(days=1)][:6]
        with self.assertRaisesRegex(ValueError, "Expected 24 historical MPC leads"):
            replay(short, evaluations, days=25)

    def test_one_mpc_generation_then_all_diffusion_horizons_and_figures(self):
        history, forecasts = self.fake_data(40)
        states = pd.DataFrame([{"origin": o} for o in history])
        archive = {key: value.copy() for key, value in history.items()}

        def fake_mpc(frame, start, end, **kwargs):
            self.assertEqual(kwargs["max_horizon"], 6)
            self.assertEqual(kwargs["fit_hours"], 24)
            self.assertEqual(kwargs["archive_horizon"], 24)
            self.assertEqual(kwargs["sigma_rolling_days"], 25)
            self.assertEqual(kwargs["sigma_fit_mode"], MODE)
            kwargs["mpc_archive"].update(archive)
            return forecasts, states, []
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.exe.run_model.wm_4.sweep_mpc_sigma_fit_hours_no_p0.run_all_days",
                side_effect=fake_mpc) as mpc:
                result = run(pd.DataFrame(), output=Path(tmp))
            self.assertEqual(mpc.call_count, 1)
            self.assertEqual(len(result["fits"]), 15 * 19)
            self.assertEqual(len(result["quantiles"]), 19 * 5)
            self.assertEqual(len(result["leads"]), 19 * 6)
            self.assertTrue((result["mpc_forecasts"]["initial_variance"] == 0).all())
            for name in ("fits.csv", "summary.csv", "quantiles.csv", "leads.csv",
                "mpc_forecasts.csv", "mpc_states.csv", "mpc_calibration_24h.csv",
                "fit_hours_sweep.png",
                "fit_hours_quintile_sweep.png", "fit_hours_lead_heatmap.png",
                "fit_hours_scatter.png"):
                self.assertTrue((Path(tmp) / name).is_file(), name)

    def test_invalid_calibration_config_rejected_before_refitting(self):
        with self.assertRaisesRegex(ValueError, "2 <= min_hours"):
            run(pd.DataFrame(), min_hours=25, max_hours=24)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            run(pd.DataFrame(), history_days=1)


if __name__ == "__main__": unittest.main()
