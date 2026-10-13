"""Regression tests for fast, same-origin 7..28 day WM4 diffusion variance sweep."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.model.wm_4 import wm_4
from src.exe.run_model.wm_4.predict_all_days_mpc import _fit_sigma_rolling_qp
from src.exe.run_model.wm_4.sweep_mpc_sigma_days import (
    analyze, fit_window_variances, read_mpc_archive, replay, sweep,
)


class SigmaDaysSweepTests(unittest.TestCase):
    @staticmethod
    def synthetic(days=45):
        archive = {}
        generator = np.random.default_rng(2026)
        for n, origin in enumerate(pd.date_range("2022-04-01 09:00", periods=days, freq="D")):
            h = np.arange(1, 7, dtype=float)
            B = 0.003 + 0.0005 * h
            K = 0.005 * h
            err = generator.normal(size=6) * np.sqrt(B + K * 0.04)
            archive[pd.Timestamp(origin)] = np.column_stack((err, B, K))
        return archive

    @staticmethod
    def csv(frame):
        rows = []
        for origin, samples in frame.items():
            for h in range(1, 7):
                rows.append({"origin": origin, "lead_hours": h,
                    "target_time": origin + pd.Timedelta(hours=h),
                    "I_observed": 0.5 + samples[h - 1, 0], "I_mpc": 0.5,
                    "initial_variance": samples[h - 1, 1],
                    "process_kernel_unit": samples[h - 1, 2],
                    "parameter_fit_hours": 24})
        return pd.DataFrame(rows)

    def test_fast_scalar_qp_matches_original_constrained_qp_for_all_lengths(self):
        model = wm_4((0.0, 1.0), 49, lambda t: 0.0)
        archive = self.synthetic(40)
        history = np.stack([archive[pd.Timestamp("2022-05-10 09:00") -
            pd.Timedelta(days=d)] for d in range(1, 29)])
        ns, qs, S, B, K = fit_window_variances(history)
        self.assertEqual(ns.tolist(), list(range(7, 29)))
        self.assertEqual(len(qs), 22)
        self.assertTrue((qs >= 0).all())
        for n, q, obs, prior, response in zip(ns, qs, S, B, K):
            reference, ref_S, ref_B, ref_K = _fit_sigma_rolling_qp(model,
                {"values": history[:n]}, horizon=6)
            self.assertAlmostEqual(q, reference["q"], delta=1e-8)
            self.assertTrue(np.allclose(obs, ref_S, atol=1e-12))
            self.assertTrue(np.allclose(prior, ref_B, atol=1e-12))
            self.assertTrue(np.allclose(response, ref_K, atol=1e-12))

    def test_full_sweep_and_common_cohort_have_correct_fitting_counts(self):
        archive = self.synthetic(45)
        result = sweep(archive, min_days=7, max_days=28)
        fits = result["fits"]
        self.assertEqual(fits["history_days"].nunique(), 22)
        self.assertEqual(len(fits), sum(45 - d for d in range(7, 29)))
        self.assertEqual(len(result["matched_fits"]), (45 - 28) * 22)
        self.assertTrue((fits["diffusion_q"] >= 0).all())
        self.assertTrue((fits[[f"variance_h{h}" for h in range(1, 7)]] >= 0).all().all())
        self.assertTrue((result["matched_summary"]["n_origins"] == 45 - 28).all())
        self.assertTrue(np.allclose(result["matched_summary"]["observed_rmse"],
            result["matched_summary"]["observed_rmse"].iloc[0]))
        self.assertEqual(len(result["matched_quantiles"]), 22 * 5)
        self.assertEqual(len(result["matched_leads"]), 22 * 6)
        self.assertEqual(len(result["all_leads"]), 22 * 6)
        self.assertTrue((result["matched_leads"]["n_origins"] == 45 - 28).all())
        self.assertEqual(int(result["all_summary"].iloc[0]["n_origins"]), 45 - 7)
        self.assertEqual(int(result["all_summary"].iloc[-1]["n_origins"]), 45 - 28)
        self.assertTrue(result["all_quantiles"]["coverage_95_pct"].between(0, 100).all())

    def test_missing_intermediate_day_breaks_causal_consecutive_history(self):
        archive = self.synthetic(45)
        target = pd.Timestamp("2022-05-15 09:00")
        archive.pop(target - pd.Timedelta(days=12))
        fits = replay(archive, min_days=7, max_days=28)
        eligible = fits[fits["origin"] == target]
        self.assertEqual(eligible["history_days"].tolist(), list(range(7, 12)))
        self.assertNotIn(target, fits.loc[fits["history_days"] == 28, "origin"].tolist())
        self.assertTrue((fits["history_days"] <= 28).all())

    def test_import_slices_first_six_leads_and_checks_metadata(self):
        archive = self.synthetic(8)
        csv = self.csv(archive)
        origin = next(iter(archive))
        extra = csv[csv["lead_hours"] == 6].iloc[0].copy()
        extra["lead_hours"] = 7
        extra["target_time"] = origin + pd.Timedelta(hours=7)
        csv = pd.concat([csv, pd.DataFrame([extra])], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mpc_forecasts.csv"
            csv.to_csv(path, index=False)
            loaded, data = read_mpc_archive(path)
            self.assertEqual(len(loaded), 8)
            self.assertEqual(len(data), 8 * 6)
            self.assertTrue(np.allclose(loaded[origin], archive[origin]))
            csv.loc[0, "parameter_fit_hours"] = 12
            csv.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "not all fitted"):
                read_mpc_archive(path)

    def test_all_fit_csvs_and_plots_without_rerunning_wm4(self):
        archive = self.synthetic(40)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "forecasts.csv"
            output = Path(tmp) / "out"
            self.csv(archive).to_csv(source, index=False)
            result = analyze(source, output, min_days=7, max_days=28)
            self.assertEqual(len(result["matched_summary"]), 22)
            self.assertEqual(len(result["matched_quantiles"]), 110)
            for name in ("matched_sigma_sweep.png", "matched_quintile_sweep.png",
                "fitted_origins_by_window.png", "matched_lead_calibration_heatmap.png",
                "fits.csv", "matched_fits.csv", "all_summary.csv", "all_quantiles.csv",
                "matched_summary.csv", "matched_quantiles.csv", "all_leads.csv",
                "matched_leads.csv"):
                self.assertTrue((output / name).is_file(), msg=name)


if __name__ == "__main__": unittest.main()
