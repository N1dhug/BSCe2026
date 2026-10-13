"""WM4 P0 propagation ablation: same origins and deterministic MPC errors."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.compare_mpc_p0_ablation import (
    VARIANTS, _no_p0_fit_q, ablate, analyze, summarize,
)
from src.exe.run_model.wm_4.sweep_mpc_sigma_days import (
    fit_window_variances, sweep,
)


class P0AblationTests(unittest.TestCase):
    @staticmethod
    def synthetic(n=41):
        rng = np.random.default_rng(9082)
        archive = {}
        for i, stamp in enumerate(pd.date_range("2022-05-01 10:00", periods=n, freq="D")):
            hours = np.arange(1, 7, dtype=float)
            B = (0.003 + 0.001 * hours) * (1.0 + 0.15 * np.sin(i / 8))
            K = 0.003 * hours * (1 + 0.1 * np.cos(i / 11))
            error = rng.normal(size=6) * np.sqrt(B + .06 * K)
            archive[stamp] = np.column_stack((error, B, K))
        return archive

    @staticmethod
    def save_forecasts(archive):
        rows = []
        for stamp, matrix in archive.items():
            for h in range(1, 7):
                rows.append({"origin": stamp, "target_time": stamp + pd.Timedelta(hours=h),
                    "lead_hours": h, "I_observed": 0.5 + matrix[h - 1, 0],
                    "I_mpc": 0.5, "initial_variance": matrix[h - 1, 1],
                    "process_kernel_unit": matrix[h - 1, 2],
                    "parameter_fit_hours": 24})
        return pd.DataFrame(rows)

    def test_no_p0_refit_matches_qp_with_zero_initial_variance(self):
        archive = self.synthetic()
        origin = list(archive)[-1]
        history = np.stack([archive[origin - pd.Timedelta(days=d)]
            for d in range(1, 25)])
        history[:, :, 1] = 0.0
        _, qs, S, B, K = fit_window_variances(history, 24, 24)
        actual = _no_p0_fit_q(archive, origin, 24)
        self.assertEqual(float(B.max()), 0.0)
        self.assertAlmostEqual(actual, qs[0], places=12)
        self.assertTrue(actual >= 0.0)
        self.assertTrue((S >= 0).all())

    def test_all_variants_use_identical_errors_origins_and_valid_decomposition(self):
        archive = self.synthetic()
        matched = sweep(archive)["matched_fits"]
        origins, forecasts = ablate(matched, archive, history_days=24, quantiles=5)
        score = summarize(forecasts)
        n = len(archive) - 28
        self.assertEqual(len(origins), n * 3)
        self.assertEqual(len(forecasts), n * 6 * 3)
        self.assertEqual(set(origins["variant"]), set(VARIANTS))
        self.assertTrue((score["n_origins"] >= 1).all())
        self.assertEqual(int((score["binning"] == "own").sum()), 15)
        self.assertEqual(int((score["binning"] == "fixed").sum()), 15)
        self.assertEqual(int((score["binning"] == "own_lead").sum()), 90)
        pivot = origins.pivot(index="origin", columns="variant", values="observed_mse")
        for variant in VARIANTS[1:]:
            self.assertTrue(np.allclose(pivot[variant], pivot["full_p0"]))
        original_bins = origins.pivot(index="origin", columns="variant",
            values="original_quantile")
        self.assertTrue((original_bins.nunique(axis=1) == 1).all())
        f = forecasts.pivot(index=["origin", "lead_hours"], columns="variant",
            values="forecast_error")
        self.assertTrue((f.nunique(axis=1) == 1).all())
        for _, part in origins.groupby("origin"):
            a = part.set_index("variant")
            self.assertAlmostEqual(a.loc["full_p0", "predicted_variance"],
                a.loc["no_p0_fixed_q", "predicted_variance"] +
                a.loc["full_p0", "original_initial_variance"])
            self.assertGreaterEqual(a.loc["no_p0_refit_q", "diffusion_q"],
                a.loc["full_p0", "diffusion_q"] - 1e-11)
            self.assertAlmostEqual(a.loc["no_p0_refit_q", "predicted_variance"],
                a.loc["no_p0_refit_q", "wiener_variance"])
        for _, part in score[score["binning"] == "overall"].groupby("variant"):
            self.assertEqual(int(part.iloc[0]["n_origins"]), n)
        overall = score.loc[score["binning"] == "overall"]
        self.assertTrue(np.allclose(overall["observed_mse"], overall["observed_mse"].iloc[0]))

    def test_outputs_without_wm4_refitting(self):
        archive = self.synthetic()
        matched = sweep(archive)["matched_fits"]
        with tempfile.TemporaryDirectory() as temp:
            forecast_path = Path(temp) / "forecasts.csv"
            fits_path = Path(temp) / "matched_fits.csv"
            output = Path(temp) / "p0_ablation"
            self.save_forecasts(archive).to_csv(forecast_path, index=False)
            matched.to_csv(fits_path, index=False)
            result = analyze(fits_path, forecast_path, output, history_days=24)
            self.assertEqual(result["origins"]["origin"].nunique(), 13)
            self.assertEqual(result["forecasts"]["lead_hours"].max(), 6)
            self.assertEqual(set(result["scores"]["variant"]), set(VARIANTS))
            for filename in ("p0_ablation_origins.csv", "p0_ablation_forecasts.csv",
                "p0_ablation_scores.csv", "p0_ablation_quantiles.png",
                "p0_ablation_scatter.png"):
                self.assertTrue((output / filename).is_file(), filename)


if __name__ == "__main__": unittest.main()
