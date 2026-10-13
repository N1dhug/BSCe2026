"""Regression checks for fresh 7–28 day WM4 Wiener fits with no P0."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.predict_all_days_mpc_no_p0 import MODE
from src.exe.run_model.wm_4.sweep_mpc_sigma_days import fit_window_variances
from src.exe.run_model.wm_4.sweep_mpc_sigma_days_no_p0 import run


class WM4NoP0SweepTests(unittest.TestCase):
    @staticmethod
    def fake_mpc(frame, start, end, *, baseline_weeks, steps_per_hour,
        max_horizon, fit_hours, fail_fast, sigma_fit_mode, sigma_rolling_days,
        mpc_archive):
        assert max_horizon == 6
        assert fit_hours == 24
        assert sigma_fit_mode == MODE
        assert sigma_rolling_days == 24
        gen = np.random.default_rng(2026)
        rows, states = [], []
        # Each origin has exactly six hourly forecast leads. Every 24-hour
        # lag at the same clock hour has completed before current origin.
        for origin in pd.date_range("2022-03-01 09:00", periods=39, freq="D"):
            h = np.arange(1, 7, dtype=float)
            error = gen.normal(size=6) * np.sqrt(0.002 + h * .0008)
            kernel = 0.003 * h * (1 + gen.random() * 0.05)
            mpc_archive[origin] = np.column_stack((error, np.zeros(6), kernel))
            states.append({"origin": origin, "diffusion_q": 999.})
            for lead in range(1, 7):
                rows.append({"origin": origin, "lead_hours": lead,
                    "I_observed": 0.5 + error[lead - 1], "I_mpc": 0.5,
                    "initial_variance": 0., "sigma_fit_mode": MODE,
                    "diffusion_q": 999.})
        return pd.DataFrame(rows), pd.DataFrame(states), []

    def test_new_six_lead_sigma_is_fit_for_each_window_and_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new_no_p0"
            with patch("src.exe.run_model.wm_4.sweep_mpc_sigma_days_no_p0.run_all_days",
                side_effect=self.fake_mpc) as mpc:
                result = run(pd.DataFrame(), output=output, min_days=7, max_days=28)
            self.assertEqual(mpc.call_count, 1)  # 22 QPs, no 22 deterministic refits
            self.assertEqual(len(result["archive"]), 39)
            self.assertEqual(len(result["fits"]), sum(39 - n for n in range(7, 29)))
            self.assertEqual(len(result["matched_fits"]), (39 - 28) * 22)
            self.assertEqual(len(result["matched_quantiles"]), 22 * 5)
            self.assertTrue((result["matched_summary"]["n_origins"] == 39 - 28).all())
            self.assertTrue((result["fits"]["diffusion_q"] >= 0.).all())
            self.assertTrue((result["fits"]["diffusion_q"] < 999.).all())
            self.assertEqual(result["matched_leads"]["lead_hours"].max(), 6)
            for filename in ("fits.csv", "matched_fits.csv",
                "all_summary.csv", "all_quantiles.csv",
                "matched_summary.csv", "matched_quantiles.csv",
                "all_leads.csv", "matched_leads.csv", "mpc_forecasts.csv",
                "mpc_states.csv", "matched_sigma_sweep.png",
                "matched_quintile_sweep.png", "matched_lead_calibration_heatmap.png",
                "fitted_origins_by_window.png", "matched_scatter_all_windows.png"):
                self.assertTrue((output / filename).is_file(), filename)

            # Recompute individual QPs from generated historical MPC errors,
            # checking both 7d and 28d and ensuring B is identically zero.
            for window in (7, 24, 28):
                origin = sorted(result["archive"])[-1]
                history = np.stack([result["archive"][origin - pd.Timedelta(days=k)]
                    for k in range(1, window + 1)])
                ns, q, _, B, K = fit_window_variances(history, window, window)
                self.assertEqual(ns.tolist(), [window])
                self.assertTrue(np.allclose(B, 0.))
                matched = result["fits"].loc[
                    (result["fits"]["origin"] == origin)
                    & (result["fits"]["history_days"] == window)].iloc[0]
                self.assertAlmostEqual(float(matched["diffusion_q"]), float(q[0]), places=12)
                for lead in range(1, 7):
                    self.assertAlmostEqual(matched[f"variance_h{lead}"],
                        q[0] * result["archive"][origin][lead - 1, 2],
                        places=12)

    def test_rejects_archived_initial_variance_propagation(self):
        def bad_forecast(*args, **kwargs):
            archive = kwargs["mpc_archive"]
            forecasts, states, skipped = self.fake_mpc(*args, **kwargs)
            next(iter(archive.values()))[0, 1] = 0.1
            return forecasts, states, skipped

        with patch("src.exe.run_model.wm_4.sweep_mpc_sigma_days_no_p0.run_all_days",
            side_effect=bad_forecast):
            with self.assertRaisesRegex(RuntimeError, "Initial-state variance entered"):
                run(pd.DataFrame())

    def test_invalid_fit_days_rejected_before_forecast(self):
        with self.assertRaisesRegex(ValueError, "Require 2"):
            run(pd.DataFrame(), min_days=28, max_days=7)


if __name__ == "__main__": unittest.main()
