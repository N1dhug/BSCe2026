import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src.exe.run_model.wm_4.fit_moving_horizon_diffusion import (
    fit_variances, grid_summary, ks_similarity, oracle_errors, pooled_record,
    run, update_pool, variance_kernel,
)
from src.exe.run_model.wm_4.sweep_weekday_horizon import EXPERIMENT_VERSION
from src.exe.run_model.wm_4.sweep_weekday_horizon_diffusion import unit_process_variance
from src.model.wm_4 import wm_4
from src.tools.baseline import output_callable


class test_wm4_moving_horizon_diffusion(unittest.TestCase):
    def test_variance_kernel_matches_original_euler_maruyama_solver(self):
        for theta_F, theta_R, steps in ((-1., -0.3, 2), (-0.8, -1.4, 4), (-0.2, -1., 8)):
            kernel = variance_kernel(theta_F, theta_R, steps, 12)
            expected = np.array([
                unit_process_variance(theta_F, theta_R, steps, j) for j in range(12)
            ])
            np.testing.assert_allclose(kernel, expected, atol=1e-13, rtol=1e-13)
            self.assertEqual(kernel[0], 0.)

    def test_analytic_nonnegative_least_squares_matches_nnls(self):
        rng = np.random.default_rng(3)
        for tau2, q in ((0.001, 0.004), (0., 0.1), (0.1, 0.), (0., 0.)):
            K = np.tile(variance_kernel(-0.8, -1.4, 4, 12), 15)
            errors = rng.normal(0., np.sqrt(tau2+q*K))
            expected_tau, expected_q = nnls(np.column_stack((np.ones(len(K)), K)),
                                           errors*errors)[0]
            estimate_tau, estimate_q, identifiable = fit_variances(errors, K)
            self.assertTrue(identifiable)
            np.testing.assert_allclose([estimate_tau, estimate_q],
                                       [expected_tau, expected_q], rtol=1e-9, atol=1e-12)
        t, q, identifiable = fit_variances([0.2, -0.4], [0., 0.])
        self.assertFalse(identifiable)
        self.assertAlmostEqual(t, 0.1)
        self.assertEqual(q, 0.)

    def test_pooled_metrics_weight_all_observations(self):
        pool = {}
        key = ("weekly_average", 2, 3)
        update_pool(pool, key, [0., 1., -2., 1., 0., 3.],
                    [0.01]*6, 0.01, 0., True, 2)
        update_pool(pool, key, [0., 0., 0., 0., 0., 0.],
                    [0.02]*6, 0.02, 0., True, 2)
        row = grid_summary(pool).iloc[0]
        self.assertEqual(row["origins"], 2)
        self.assertEqual(row["observations"], 12)
        self.assertAlmostEqual(row["observed_rmse"], np.sqrt(15/12))
        self.assertAlmostEqual(row["expected_rmse"], np.sqrt(0.015))
        self.assertAlmostEqual(row["bias_observed_minus_predicted"], 3/12)

    def test_ks_distribution_similarity_prefers_normalized_gaussian_errors(self):
        rng = np.random.default_rng(17)
        from src.exe.run_model.wm_4.fit_moving_horizon_diffusion import HIST_BINS
        gaussian = rng.standard_normal(50000)
        strange = np.full(50000, 4.)
        a = np.histogram(gaussian, bins=HIST_BINS)[0]
        b = np.histogram(strange, bins=HIST_BINS)[0]
        self.assertGreater(ks_similarity(a, a.sum()), .98)
        self.assertLess(ks_similarity(b, b.sum()), .15)
        self.assertTrue(np.isnan(ks_similarity(np.zeros(len(a)), 0)))

    @staticmethod
    def source_data():
        index = pd.date_range("2022-01-01", periods=24*105, freq="h")
        hourly = index.hour.to_numpy(dtype=float)
        frame = pd.DataFrame({
            "I":0.52+0.04*np.sin(2*np.pi*hourly/24.),
            "U":0.1+0.06*np.cos(2*np.pi*hourly/24.),
            "Temp":np.full(len(index), 8.),
        }, index=index)
        origin = pd.Timestamp("2022-04-02 21:00")
        history = pd.to_datetime(["2022-03-19 21:00", "2022-03-26 21:00"])
        k, steps = 6, 4
        theta = np.array([-0.8, -0.2, -1.4])
        fit = {"day":origin, "baseline":"weekly_average", "training_weekdays":2,
               "horizon_hours":k, "theta_F":theta[0], "theta_U":theta[1],
               "theta_R":theta[2], "baseline_coefficients":json.dumps([0.5]*168),
               "baseline_beta_C":0., "baseline_chill_center":0.,
               "training_days":";".join(f"{past:%Y-%m-%dT%H:%M:%S}" for past in history)}
        initial = [0.1, 0.03]
        elapsed = np.arange(k*steps+1, dtype=float)/steps
        fine_index = origin+pd.to_timedelta(elapsed, unit="h")
        hourly_U = frame.reindex(pd.date_range(origin, periods=k, freq="h"))["U"].to_numpy()
        U = hourly_U[np.minimum(np.floor(elapsed+1e-10).astype(int), k-1)]
        O = np.full(len(U), 0.5)
        model = wm_4((0., k/24.), len(U), output_callable(fine_index, O))
        pred = model._solve_deterministic(initial, U, U, elapsed, O, theta)["I"][np.arange(k)*steps]
        actual = frame.reindex(pd.date_range(origin, periods=k, freq="h"))["I"].to_numpy()
        mse = float(np.mean((actual-pred)**2))
        score = {"day":origin, "baseline":"weekly_average", "training_weekdays":2,
                 "horizon_hours":k, "oracle_F0":initial[0], "oracle_R0":initial[1],
                 "mse":mse}
        states = [
            {"day":origin, "baseline":"weekly_average", "training_weekdays":2,
             "horizon_hours":k, "historical_day":past,
             "historical_F0":0.1, "historical_R0":0.02}
            for past in history
        ]
        return frame, fit, states, score

    def test_oracle_replay_checks_saved_prediction_mse(self):
        frame, fit, states, score = self.source_data()
        error = oracle_errors(frame, fit, score, 4)
        self.assertEqual(len(error), 6)
        self.assertAlmostEqual(float(np.mean(error**2)), score["mse"])
        bad_score = dict(score, mse=score["mse"]+0.1)
        with self.assertRaisesRegex(ValueError, "Oracle replay differs"):
            oracle_errors(frame, fit, bad_score, 4)

    def test_end_to_end_from_saved_csv_without_refitting(self):
        frame, fit, states, score = self.source_data()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root/"sweep", root/"diffusion"
            source.mkdir()
            pd.DataFrame([fit]).to_csv(source/"parameter_estimates.csv", index=False)
            pd.DataFrame(states).to_csv(source/"historical_initial_states.csv", index=False)
            pd.DataFrame([score]).to_csv(source/"oracle_horizon_scores.csv", index=False)
            config = {"version":EXPERIMENT_VERSION, "group":"ALL", "steps_per_hour":4,
                      "baselines":["weekly_average"], "min_weekdays":2,
                      "max_weekdays":2, "min_horizon":6, "max_horizon":6}
            (source/"configuration.json").write_text(json.dumps(config), encoding="utf-8")
            args = Namespace(group="ALL", source=str(source), output=str(output),
                             evaluation="both", cohort="all", baselines=None,
                             start_day=None, end_day=None, min_weekdays=None,
                             max_weekdays=None, min_horizon=None, max_horizon=None,
                             progress_every=1, max_configurations=0)
            with patch("src.exe.run_model.wm_4.fit_moving_horizon_diffusion.group_frame",
                       return_value=frame):
                result = run(args)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["observations"], 12)
            self.assertTrue((output/"diffusion_historical_heatmaps.png").is_file())
            self.assertTrue((output/"diffusion_oracle_heatmaps.png").is_file())
            self.assertTrue((output/"diffusion_parameter_heatmaps.png").is_file())
            self.assertTrue((output/"diffusion_interval_parameters.csv").is_file())
            self.assertTrue((output/"diffusion_oracle_interval_scores.csv").is_file())
            historic = pd.read_csv(output/"diffusion_historical_summary.csv")
            target = pd.read_csv(output/"diffusion_oracle_summary.csv")
            self.assertEqual(historic.loc[0, "observations"], 12)
            self.assertEqual(target.loc[0, "observations"], 6)
            self.assertAlmostEqual(target.loc[0, "observed_rmse"], np.sqrt(score["mse"]))
            metadata = json.loads((output/"diffusion_configuration.json").read_text())
            self.assertEqual(metadata["drift_parameter_refits"], 0)

    def test_saved_row_alignment_is_checked(self):
        from src.exe.run_model.wm_4.fit_moving_horizon_diffusion import iter_saved
        _, fit, states, score = self.source_data()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            pd.DataFrame([fit]).to_csv(folder/"parameter_estimates.csv", index=False)
            pd.DataFrame([dict(row, horizon_hours=7) for row in states]).to_csv(
                folder/"historical_initial_states.csv", index=False)
            pd.DataFrame([score]).to_csv(folder/"oracle_horizon_scores.csv", index=False)
            with self.assertRaisesRegex(ValueError, "not aligned"):
                list(iter_saved(folder, include_scores=True))


if __name__ == "__main__": unittest.main()
