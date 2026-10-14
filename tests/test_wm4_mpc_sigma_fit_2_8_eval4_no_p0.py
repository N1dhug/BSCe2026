"""Regress the 2..8h no-P0 diffusion fit against identical 4h WM4 MPC errors."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.sweep_mpc_sigma_fit_hours_eval4_no_p0 import (
    EVALUATION_HOURS, MODE, fit_per_origin, four_hour_evaluation,
    load_archive, run, save_archive, summarize,
)
from src.exe.run_model.wm_4.sweep_mpc_sigma_fit_hours_no_p0 import (
    ARCHIVE_HOURS, BACKLOG_DAYS, fit_diffusion_horizons,
)


class FourHourWM4DiffusionSweepTests(unittest.TestCase):
    @staticmethod
    def example(days=42):
        rng = np.random.default_rng(2026)
        archive, rows = {}, []
        for i, origin in enumerate(pd.date_range("2022-04-01 10:00",
            periods=days, freq="D")):
            lead = np.arange(1, ARCHIVE_HOURS + 1, dtype=float)
            kernel = 0.002 + 0.003 * lead * (1 + 0.04 * np.sin(i / 8))
            error = rng.normal(size=ARCHIVE_HOURS) * np.sqrt(
                (0.025 + 0.004 * np.cos(i / 6)) * kernel)
            archive[origin] = np.column_stack((
                error, np.zeros(ARCHIVE_HOURS), kernel))
            for h in range(1, 7):
                rows.append({"origin": origin, "lead_hours": h,
                    "I_observed": .5 + error[h - 1], "I_mpc": .5,
                    "initial_variance": 0., "process_kernel_unit": kernel[h - 1],
                    "parameter_fit_hours": 24, "sigma_fit_mode": MODE,
                    "diffusion_q": 999.})  # Must NOT reuse an old q.
        return archive, pd.DataFrame(rows)

    def test_2_to_8_hour_qps_and_four_hour_variances_are_independent(self):
        archive, frame = self.example()
        evaluations = four_hour_evaluation(frame)
        self.assertEqual(len(evaluations), 42)
        self.assertTrue(all(v.shape == (4, 2) for v in evaluations.values()))
        fitted = fit_per_origin(archive, evaluations)
        self.assertEqual(fitted["calibration_hours"].unique().tolist(), list(range(2, 9)))
        self.assertEqual(len(fitted), (42 - BACKLOG_DAYS) * 7)
        self.assertEqual(fitted["origin"].nunique(), 17)
        self.assertTrue((fitted["history_days"] == BACKLOG_DAYS).all())
        self.assertTrue((fitted["evaluation_hours"] == 4).all())
        self.assertTrue((fitted["diffusion_q"] < 999).all())
        self.assertTrue((fitted["diffusion_q"] >= 0).all())

        origin = max(archive)
        historical = np.stack([archive[origin - pd.Timedelta(days=k)]
            for k in range(1, BACKLOG_DAYS + 1)])
        hs, qs, S, K = fit_diffusion_horizons(historical, 2, 8)
        for h, q in zip(hs, qs):
            expected = max(0., np.dot(S[:h], K[:h]) /
                (np.dot(K[:h], K[:h]) + 1e-10 * h / 2.))
            self.assertAlmostEqual(q, expected, places=12)
            current = fitted.loc[(fitted["origin"] == origin)
                & (fitted["calibration_hours"] == h)].iloc[0]
            self.assertAlmostEqual(current["diffusion_q"], q, places=12)
            for lead in range(1, 5):
                self.assertAlmostEqual(current[f"variance_h{lead}"],
                    q * archive[origin][lead - 1, 2], places=12)
        self.assertEqual(EVALUATION_HOURS, 4)
        scores, quintiles, by_lead = summarize(fitted, evaluations)
        self.assertEqual(len(scores), 7)
        self.assertEqual(len(quintiles), 7 * 5)
        self.assertEqual(len(by_lead), 7 * 4)
        self.assertTrue((scores["n_origins"] == 17).all())
        self.assertTrue((scores["n_forecasts"] == 17 * 4).all())
        self.assertEqual(scores["observed_rmse"].nunique(), 1)
        self.assertTrue(np.allclose(quintiles.groupby("calibration_hours")[
            "n_origins"].sum(), 17))

    def test_archive_roundtrip_preserves_exact_historical_six_to_24_hour_kernel(self):
        archive, frame = self.example(30)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mpc_calibration_24h.csv"
            save_archive(archive, path)
            recovered = load_archive(path)
            self.assertEqual(set(recovered), set(archive))
            for origin in archive:
                self.assertTrue(np.array_equal(recovered[origin], archive[origin]))
            broken = pd.read_csv(path)
            broken.loc[0, "initial_variance"] = .1
            broken.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "propagated P0"):
                load_archive(path)

    def test_missing_historical_mpc_path_does_not_use_current_or_future(self):
        archive, frame = self.example()
        evaluations = four_hour_evaluation(frame)
        target = max(archive)
        removed = archive.copy()
        removed.pop(target - pd.Timedelta(days=12))
        result = fit_per_origin(removed, evaluations)
        self.assertNotIn(target, result["origin"].tolist())
        broken = archive.copy()
        broken[target - pd.Timedelta(days=1)] = broken[
            target - pd.Timedelta(days=1)][:6]
        with self.assertRaisesRegex(ValueError, "Invalid 24-hour historical MPC path"):
            fit_per_origin(broken, evaluations)

    def test_fresh_run_once_then_reuse_saved_mpc_without_refitting(self):
        archive, forecasts = self.example()
        states = pd.DataFrame([{"origin": origin} for origin in archive])

        def fake_mpc(frame, start, end, **kwargs):
            self.assertEqual(kwargs["max_horizon"], 6)
            self.assertEqual(kwargs["fit_hours"], 24)
            self.assertEqual(kwargs["archive_horizon"], 24)
            self.assertEqual(kwargs["sigma_fit_mode"], MODE)
            self.assertEqual(kwargs["sigma_rolling_days"], 25)
            kwargs["mpc_archive"].update(archive)
            return forecasts, states, []

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "four_hour"
            source = Path(tmp) / "missing"
            with patch("src.exe.run_model.wm_4.sweep_mpc_sigma_fit_hours_eval4_no_p0.run_all_days",
                side_effect=fake_mpc) as mpc:
                original = run(pd.DataFrame(), source=source, output=output)
                self.assertEqual(mpc.call_count, 1)
            self.assertEqual(len(original["fits"]), 17 * 7)
            for name in ("mpc_calibration_24h.csv", "mpc_forecasts.csv",
                "fits.csv", "summary.csv", "quantiles.csv", "leads.csv",
                "4h_evaluation_fit_2_8_sweep.png", "4h_evaluation_quintiles.png",
                "4h_evaluation_lead_calibration.png", "4h_evaluation_scatter.png"):
                self.assertTrue((output / name).is_file(), name)

            with patch("src.exe.run_model.wm_4.sweep_mpc_sigma_fit_hours_eval4_no_p0.run_all_days",
                side_effect=AssertionError("Must reuse saved historical MPC")):
                cached = run(source=source, output=output, reuse_only=True)
            self.assertTrue(np.allclose(cached["fits"]["diffusion_q"],
                original["fits"]["diffusion_q"]))
            self.assertTrue(np.allclose(cached["summary"]["gaussian_nll"],
                original["summary"]["gaussian_nll"]))

    def test_reuse_only_never_silently_runs_deterministic_wm4(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "No saved 24h"):
                run(source=Path(tmp), output=Path(tmp) / "out",
                    reuse_only=True)
        with self.assertRaisesRegex(ValueError, "2 <= min_hours"):
            run(frame=pd.DataFrame(), min_hours=9, max_hours=8)


if __name__ == "__main__": unittest.main()
