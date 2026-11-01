import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.analyze_oracle_8weekday import (
    HORIZON, N_WEEKDAYS, analyze, check_tables, consolidate, cumulative_table,
    hac_test, load_from_sweep, parameter_tests, quantile_table,
)
from src.exe.run_model.wm_4.sweep_weekday_horizon import evaluate_configuration
from unittest.mock import patch


class fake_joint_fitter:
    def __init__(self, model): self.model = model

    def fit_many(self, datasets):
        n = len(datasets)
        return {
            "r": np.array([-0.8, -0.15, -1.4]),
            "init_cond": np.zeros((n, 2)),
            "qp_loss": 0.01, "deterministic_loss": 0.02,
            "design_rank": n+3, "beta": np.zeros(n+3),
        }


class test_wm4_oracle_8weekday(unittest.TestCase):
    @staticmethod
    def frame():
        index = pd.date_range("2022-01-01", periods=24*100, freq="h")
        h = index.hour.to_numpy(dtype=float)
        d = np.arange(len(index), dtype=float)/24
        I = 0.5 + 0.05*np.sin(h*2*np.pi/24) + 0.03*np.sin(d/8)
        U = 0.25*np.sin(h*2*np.pi/24+d/2)
        Temp = 6+3*np.cos(d/6)
        return pd.DataFrame({"I": I, "U": U, "Temp": Temp}, index=index)

    def sample(self):
        day = pd.Timestamp("2022-03-22")
        with patch("src.exe.run_model.wm_4.sweep_weekday_horizon.joint_qp_params", fake_joint_fitter):
            fit, scores, component, states = evaluate_configuration(
                self.frame(), day, N_WEEKDAYS, 4, HORIZON, HORIZON, "weekly_temperature")
        return fit, scores[0], component, states

    def test_reconstructed_dynamics_agree_with_oracle_scores(self):
        fit, score, component, states = self.sample()
        daily, hourly, history = consolidate(fit, score, component, states, 4)
        self.assertEqual(len(hourly), 7)
        self.assertEqual(len(history), 8)
        self.assertAlmostEqual(np.sqrt(np.mean([h["error"]**2 for h in hourly])), daily["rmse"])
        for hour in hourly:
            self.assertAlmostEqual(hour["restoring"]+hour["damping"]+hour["price"], hour["R_dot"])
            self.assertAlmostEqual(hour["predicted"]-hour["baseline_prediction"], hour["R"])
            self.assertTrue(np.isfinite([hour["F"], hour["R"], hour["price"]]).all())
        self.assertGreaterEqual(daily["restoring_rms"], 0)
        self.assertGreaterEqual(daily["damping_rms"], 0)
        self.assertGreaterEqual(daily["price_rms"], 0)

    def test_quantiles_and_parameter_tests_preserve_raw_mse(self):
        fit, score, component, states = self.sample()
        days, hours = [], []
        for i in range(4):
            date = pd.Timestamp(f"2022-03-{20+i}")
            row, hourly, _ = consolidate(fit, score, component, states, 4)
            row["day"] = date
            row["rmse"] = .06+i*.01
            for h in hourly:
                h["day"] = date
                h["time"] = date+pd.Timedelta(hours=h["hour"])
                h["error"] = .01*(h["hour"]+i)
                h["absolute_error"] = abs(h["error"])
            days.append(row); hours.extend(hourly)
        daily, hourly = check_tables(pd.DataFrame(days), pd.DataFrame(hours))
        quantile = quantile_table(daily, hourly)
        cumulative = cumulative_table(hourly)
        self.assertEqual(len(quantile), HORIZON)
        self.assertEqual(len(cumulative), HORIZON)
        self.assertAlmostEqual(quantile.iloc[0]["abs_error_q20"], .006)
        self.assertAlmostEqual(cumulative.iloc[0]["daily_cumulative_rmse_q80"], .024)
        stats, corr, weekday = parameter_tests(daily, lag=2)
        self.assertIn("theta_F", set(stats["variable"]))
        self.assertIn("price_rms", set(stats["variable"]))
        self.assertIn("spearman_rho", corr.columns)
        self.assertIn("statistic", weekday.columns)

    def test_hac_handles_insufficient_or_constant_series(self):
        self.assertTrue(np.isnan(hac_test([1., 2.])["p_value"]))
        result = hac_test([1., 1., 1., 1.])
        self.assertTrue(np.isnan(result["p_value"]))
        normal = hac_test([1., 2., 0., 3., -1., 2., 1.], lag=3)
        self.assertTrue(np.isfinite(normal["hac_se"]))

    def test_sweep_reuse_and_report_without_refitting(self):
        fit, score, components, history = self.sample()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root/"source", root/"report"
            source.mkdir()
            pd.DataFrame([fit]).to_csv(source/"parameter_estimates.csv", index=False)
            pd.DataFrame([score]).to_csv(source/"oracle_horizon_scores.csv", index=False)
            pd.DataFrame(components).to_csv(source/"forecast_components.csv", index=False)
            pd.DataFrame(history).to_csv(source/"historical_initial_states.csv", index=False)
            (source/"configuration.json").write_text('{"steps_per_hour":4}', encoding="utf-8")
            self.assertEqual(load_from_sweep(source, "weekly_temperature", output, 4), 1)
            daily, hourly, parameters = analyze(output)
            self.assertEqual(len(daily), 1)
            self.assertEqual(len(hourly), 7)
            self.assertEqual(len(parameters), 9)
            self.assertIn("baseline_beta_C", set(parameters["variable"]))
            for name in ("forecast_error_quintiles.png", "theta_time_quantiles.png",
                         "contribution_hourly_quantiles.png", "prediction_demand_quintiles.png",
                         "daily_rmse_quintiles.png", "parameter_distribution_and_hac_tests.csv",
                         "parameter_spearman_tests.csv"):
                self.assertTrue((output/name).is_file(), name)
            with self.assertRaisesRegex(ValueError, "numerical resolution differs"):
                load_from_sweep(source, "weekly_temperature", root/"mismatch", 3)


if __name__ == "__main__": unittest.main()
