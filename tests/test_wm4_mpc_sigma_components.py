"""Validate origin-level P0/Wiener contributions to WM4 MPC uncertainty quintiles."""

import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.plot_mpc_sigma_components import (
    INITIAL_TERMS, _initial_details, analyze, load_matched_fits,
    summarize_components,
)
from src.exe.run_model.wm_4.sweep_mpc_sigma_days import sweep


class WM4MPCVarianceComponentTests(unittest.TestCase):
    @staticmethod
    def samples(count=42):
        generator = np.random.default_rng(1234)
        archive = {}
        terms = {}
        for origin in pd.date_range("2022-05-01 12:00", periods=count, freq="D"):
            h = np.arange(1, 7, dtype=float)
            ff, rr, fr = 0.002 + 0.0004 * h, 0.002 + 0.0008 * h, -0.0002 * h
            B = ff + rr + fr
            K = 0.004 * h
            noise = generator.normal(size=6) * np.sqrt(B + K * .045)
            archive[origin] = np.column_stack((noise, B, K))
            terms[origin] = np.column_stack((ff, rr, fr))
        return archive, terms

    @staticmethod
    def forecast_csv(archive, terms):
        rows = []
        for origin, arr in archive.items():
            for lead in range(1, 7):
                row = {"origin": origin, "lead_hours": lead,
                    "target_time": origin + pd.Timedelta(hours=lead),
                    "I_observed": 0.5 + arr[lead - 1, 0], "I_mpc": 0.5,
                    "initial_variance": arr[lead - 1, 1],
                    "process_kernel_unit": arr[lead - 1, 2],
                    "parameter_fit_hours": 24}
                row.update({field: terms[origin][lead - 1, j]
                    for j, field in enumerate(INITIAL_TERMS)})
                rows.append(row)
        return pd.DataFrame(rows)

    def test_contributions_exactly_rebuild_predicted_variance_in_each_quantile(self):
        archive, terms = self.samples()
        sweep_result = sweep(archive)
        fits = sweep_result["matched_fits"]
        origins, summary = summarize_components(fits, archive, initial_terms=terms)
        self.assertEqual(len(origins), (42 - 28) * 22)
        self.assertEqual(len(summary), 22 * 5)
        self.assertTrue(np.allclose(origins["initial_variance"] +
            origins["wiener_variance"], origins["predicted_variance"]))
        self.assertTrue(np.allclose(summary["initial_variance"] +
            summary["wiener_variance"], summary["predicted_variance"]))
        self.assertTrue(np.allclose(summary["initial_share_pct"] +
            summary["wiener_share_pct"], 100.))
        self.assertTrue(np.allclose(summary[list(INITIAL_TERMS)].sum(axis=1),
            summary["initial_variance"]))
        self.assertTrue((summary["initial_FR_component_variance"] < 0).all())
        self.assertTrue(np.allclose(summary["predicted_to_observed_mse"],
            summary["initial_to_observed_mse"] + summary["wiener_to_observed_mse"]))
        check = sweep_result["matched_quantiles"]
        paired = summary.merge(check, on=["history_days", "uncertainty_quantile"],
            validate="one_to_one", suffixes=("_components", "_sweep"))
        self.assertTrue(np.allclose(paired["predicted_to_observed_mse_components"],
            paired["predicted_to_observed_mse_sweep"]))
        self.assertTrue((paired["n_origins_components"] == paired["n_origins_sweep"]).all())

    def test_csv_replay_produces_all_diagnostic_figures_without_refits(self):
        archive, terms = self.samples()
        fits = sweep(archive)["matched_fits"]
        with tempfile.TemporaryDirectory() as tmp:
            source, history = Path(tmp) / "mpc.csv", Path(tmp) / "fits.csv"
            output = Path(tmp) / "components"
            self.forecast_csv(archive, terms).to_csv(source, index=False)
            fits.to_csv(history, index=False)
            loaded = load_matched_fits(history)
            self.assertEqual(len(loaded), len(fits))
            result = analyze(history, source, output, reference_days=24)
            self.assertEqual(len(result["quantiles"]), 110)
            self.assertEqual(result["origins"]["origin"].nunique(), 14)
            self.assertTrue((result["quantiles"]["history_days"] == 24).any())
            self.assertTrue(_initial_details(pd.read_csv(source, parse_dates=["origin"]),
                archive) is not None)
            for name in ("variance_components_by_quantile.csv",
                "origin_variance_components.csv",
                "reference_variance_components_by_quantile.png",
                "variance_components_across_backlogs.png",
                "variance_component_scatter_all_backlogs.png",
                "reference_initial_covariance_terms.png"):
                self.assertTrue((output / name).is_file(), name)

    def test_missing_covariance_terms_and_inconsistent_variances(self):
        archive, terms = self.samples()
        frame = self.forecast_csv(archive, terms)
        no_terms = frame.drop(columns=list(INITIAL_TERMS))
        self.assertIsNone(_initial_details(no_terms, archive))
        invalid = frame.copy()
        invalid.loc[0, "initial_FR_component_variance"] += 0.05
        with self.assertRaisesRegex(ValueError, "do not reproduce initial variance"):
            _initial_details(invalid, archive)

        matched = sweep(archive)["matched_fits"].copy()
        matched.loc[matched.index[0], "variance_h1"] += 0.05
        with self.assertRaisesRegex(ValueError, "decomposition failed"):
            summarize_components(matched, archive)


if __name__ == "__main__": unittest.main()
