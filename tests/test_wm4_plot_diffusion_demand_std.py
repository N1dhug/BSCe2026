import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.fit_moving_horizon_diffusion import variance_kernel
from src.exe.run_model.wm_4.plot_diffusion_demand_std import (
    aggregate, interval_std, matched_rows, plot, run,
)
from src.exe.run_model.wm_4.sweep_weekday_horizon import EXPERIMENT_VERSION


class test_wm4_plot_diffusion_demand_std(unittest.TestCase):
    @staticmethod
    def drift(day, baseline, n=2, k=3, theta_F=-0.8, theta_R=-1.4):
        return {"day":day, "baseline":baseline, "training_weekdays":n,
                "horizon_hours":k, "theta_F":theta_F, "theta_R":theta_R}

    @staticmethod
    def noise(drift, tau2, q):
        return {name:drift[name] for name in
                ("day", "baseline", "training_weekdays", "horizon_hours")} | {
                    "observation_variance":tau2, "process_variance_rate":q}

    def test_demand_std_includes_both_variance_components(self):
        drift = self.drift("2022-04-04 04:00:00", "weekly_average")
        noise = self.noise(drift, 0.0004, 0.01)
        K = variance_kernel(drift["theta_F"], drift["theta_R"], 4, 3)
        expected = np.sqrt(0.0004+0.01*K)
        np.testing.assert_allclose(interval_std(drift, noise, 4), expected)
        self.assertAlmostEqual(expected[0], 0.02)
        self.assertGreater(expected[2], expected[0])

    def test_negative_variance_rejected(self):
        drift = self.drift("2022-04-04 04:00:00", "weekly_average")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            interval_std(drift, self.noise(drift, -0.001, 0.01), 4)

    def test_streaming_merge_and_mean_of_std(self):
        day1, day2 = "2022-04-04 04:00:00", "2022-04-05 04:00:00"
        first = self.drift(day1, "weekly_average")
        skipped = self.drift(day1, "statistical_output")
        second = self.drift(day1, "weekly_temperature")
        third = self.drift(day2, "weekly_average")
        fourth = self.drift(day2, "weekly_temperature")
        drifts = [first, skipped, second, third, fourth]
        noises = [self.noise(first, 0.01**2, 0.01), self.noise(second, 0.02**2, 0.01),
                  self.noise(third, 0.03**2, 0.01), self.noise(fourth, 0.04**2, 0.01)]
        with tempfile.TemporaryDirectory() as tmp:
            source, diffusion = Path(tmp)/"parameter_estimates.csv", Path(tmp)/"diffusion_interval_parameters.csv"
            pd.DataFrame(drifts).to_csv(source, index=False)
            pd.DataFrame(noises).to_csv(diffusion, index=False)
            self.assertEqual(len(list(matched_rows(source, diffusion))), 4)
            selection = (("weekly_average", "weekly_temperature"), (2,), (3,), None, None)
            summary = aggregate(source, diffusion, 4, selection, progress_every=100)
        self.assertEqual(len(summary), 2)
        self.assertTrue((summary["origins"] == 2).all())
        avg = summary.loc[summary["baseline"] == "weekly_average"].iloc[0]
        expected_std = np.r_[interval_std(first, noises[0], 4),
                             interval_std(third, noises[2], 4)]
        self.assertAlmostEqual(avg["mean_demand_std"], np.mean(expected_std))
        self.assertAlmostEqual(avg["rms_demand_std"], np.sqrt(np.mean(expected_std**2)))
        self.assertLess(avg["mean_demand_std"], avg["rms_demand_std"])
        self.assertEqual(avg["observations"], 6)

    def test_missing_or_misaligned_noise_row_rejected(self):
        day = "2022-04-04 04:00:00"
        drift = self.drift(day, "weekly_average")
        other = self.drift(day, "weekly_temperature")
        with tempfile.TemporaryDirectory() as tmp:
            source, diffusion = Path(tmp)/"drift.csv", Path(tmp)/"noise.csv"
            pd.DataFrame([drift]).to_csv(source, index=False)
            pd.DataFrame([self.noise(other, .01, .01)]).to_csv(diffusion, index=False)
            with self.assertRaisesRegex(ValueError, "could not be matched"):
                list(matched_rows(source, diffusion))

    def test_plot_2x1_heatmap_and_saved_sources(self):
        day = "2022-04-04 04:00:00"
        drifts = [self.drift(day, "weekly_average"), self.drift(day, "weekly_temperature")]
        noises = [self.noise(drifts[0], 0.001, 0.02), self.noise(drifts[1], 0.002, 0.03)]
        with tempfile.TemporaryDirectory() as tmp:
            source, diffusion = Path(tmp)/"original", Path(tmp)/"diffusion"
            source.mkdir()
            diffusion.mkdir()
            pd.DataFrame(drifts).to_csv(source/"parameter_estimates.csv", index=False)
            pd.DataFrame(noises).to_csv(diffusion/"diffusion_interval_parameters.csv", index=False)
            (source/"configuration.json").write_text(json.dumps({
                "version":EXPERIMENT_VERSION, "group":"ALL", "steps_per_hour":4,
                "baselines":["weekly_average", "weekly_temperature"],
                "min_weekdays":2, "max_weekdays":2, "min_horizon":3, "max_horizon":3,
            }), encoding="utf-8")
            (diffusion/"diffusion_configuration.json").write_text(json.dumps({
                "steps_per_hour":4
            }), encoding="utf-8")
            args = Namespace(group="ALL", source=str(source), diffusion_source=str(diffusion),
                             output=None, baselines=["weekly_average", "weekly_temperature"],
                             min_weekdays=None, max_weekdays=None,
                             min_horizon=None, max_horizon=None,
                             start_day=None, end_day=None, progress_every=100)
            summary = run(args)
            self.assertEqual(len(summary), 2)
            self.assertTrue((diffusion/"diffusion_demand_std_heatmaps.png").is_file())
            saved = pd.read_csv(diffusion/"diffusion_demand_std_summary.csv")
            self.assertTrue((saved["observations"] == 3).all())
            self.assertTrue((saved["mean_demand_std"] > 0).all())

    def test_discretization_mismatch_rejected(self):
        day = "2022-04-04 04:00:00"
        drift = self.drift(day, "weekly_average")
        with tempfile.TemporaryDirectory() as tmp:
            source, diffusion = Path(tmp)/"original", Path(tmp)/"diffusion"
            source.mkdir()
            diffusion.mkdir()
            pd.DataFrame([drift]).to_csv(source/"parameter_estimates.csv", index=False)
            pd.DataFrame([self.noise(drift, 0.01, 0.01)]).to_csv(
                diffusion/"diffusion_interval_parameters.csv", index=False)
            (source/"configuration.json").write_text(json.dumps({
                "version":EXPERIMENT_VERSION, "group":"ALL", "steps_per_hour":4,
                "baselines":["weekly_average", "weekly_temperature"],
                "min_weekdays":2, "max_weekdays":2, "min_horizon":3, "max_horizon":3,
            }), encoding="utf-8")
            (diffusion/"diffusion_configuration.json").write_text(json.dumps({
                "steps_per_hour":2
            }), encoding="utf-8")
            args = Namespace(group="ALL", source=str(source), diffusion_source=str(diffusion),
                             output=None, baselines=["weekly_average", "weekly_temperature"],
                             min_weekdays=None, max_weekdays=None,
                             min_horizon=None, max_horizon=None,
                             start_day=None, end_day=None, progress_every=100)
            with self.assertRaisesRegex(ValueError, "different steps_per_hour"):
                run(args)


if __name__ == "__main__": unittest.main()
