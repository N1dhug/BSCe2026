import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.sweep_weekday_horizon_diffusion import (
    fit_variance, load_sweep, plot_grid, same_period, summarize_grid, unit_process_variance,
)
from src.exe.run_model.wm_4.sweep_weekday_horizon import COLUMNS


class wm4_weekday_diffusion_tests(unittest.TestCase):
    def test_process_kernel_zero_at_origin_and_grows(self):
        K = [unit_process_variance(-1.0, -0.25, 4, h) for h in range(5)]
        self.assertAlmostEqual(K[0], 0.0)
        self.assertTrue(np.isfinite(K).all())
        self.assertTrue(all(K[i+1] > K[i] for i in range(4)))

    def test_nonnegative_variance_recovery(self):
        rng = np.random.default_rng(123)
        kernel = np.repeat(np.arange(8, dtype=float)/10, 2000)
        residuals = rng.normal(0, np.sqrt(0.02+0.05*kernel))
        part = pd.DataFrame({"error":residuals, "unit_process_variance":kernel})
        observation, process, identifiable = fit_variance(part)
        self.assertTrue(identifiable)
        self.assertAlmostEqual(observation, 0.02, delta=0.003)
        self.assertAlmostEqual(process, 0.05, delta=0.008)

    def test_common_period_matches_both_baselines_and_backlogs(self):
        rows = []
        for day in ("2022-04-08", "2022-04-09", "2022-04-10", "2022-10-31", "2022-11-01"):
            for baseline in ("weekly_average", "weekly_temperature"):
                for n in (2, 3):
                    for k in (1, 2):
                        if day == "2022-04-10" and baseline == "weekly_temperature" and n == 3 and k == 2: continue
                        rows.append({"day":pd.Timestamp(day), "baseline":baseline, "training_weekdays":n, "horizon_hours":k})
        matched, days = same_period(pd.DataFrame(rows), backlogs=(2, 3), horizons=(1, 2))
        self.assertEqual([d.strftime("%Y-%m-%d") for d in days], ["2022-04-09", "2022-10-31"])
        self.assertEqual(len(matched), 16)

    def test_historical_fit_schema_without_oracle_components(self):
        day, history = pd.Timestamp("2022-05-01"), pd.Timestamp("2022-04-24")
        fit = [{"day":day, "baseline":"weekly_average", "training_weekdays":1,
                "theta_F":-1., "theta_U":-0.3, "theta_R":-0.5, "training_days":"2022-04-24"}]
        states = [{"day":day, "baseline":"weekly_average", "training_weekdays":1,
                   "historical_day":history, "historical_F0":0.2, "historical_R0":-0.1}]
        scores = [{"day":day, "baseline":"weekly_average", "training_weekdays":1, "horizon_hours":1}]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            pd.DataFrame(fit).to_csv(source/COLUMNS["fit"], index=False)
            pd.DataFrame(states).to_csv(source/COLUMNS["state"], index=False)
            pd.DataFrame(scores).to_csv(source/COLUMNS["score"], index=False)
            (source/"configuration.json").write_text('{"steps_per_hour": 4}', encoding="utf-8")
            actual_fit, actual_states, actual_scores, config = load_sweep(source, ("weekly_average",))
            self.assertEqual(len(actual_fit), 1)
            self.assertEqual(len(actual_states), 1)
            self.assertEqual(len(actual_scores), 1)
            self.assertEqual(config["steps_per_hour"], 4)

    def test_full_historical_trajectory_variance_fit_and_hourly_rmse_difference(self):
        rng = np.random.default_rng(203)
        rows = []
        for baseline in ("weekly_average", "weekly_temperature"):
            for d in range(8):
                day = pd.Timestamp("2022-04-09")+pd.Timedelta(days=d)
                for h in range(24):
                    K = unit_process_variance(-1., -0.5, 4, h)
                    residual = rng.normal(0, np.sqrt(0.002+0.05*K))
                    rows.append({"day":day, "historical_day":day-pd.Timedelta(days=7),
                                 "baseline":baseline, "training_weekdays":3, "hour":h,
                                 "observed":0.3+residual, "predicted":0.3, "error":residual,
                                 "unit_process_variance":K})
        summary, fitted = summarize_grid(pd.DataFrame(rows), horizons=(1, 4, 24))
        self.assertEqual(len(summary), 2*3)
        self.assertEqual(len(fitted), 2*8*24)
        self.assertTrue((summary["rmse_absolute_error"] >= 0).all())
        self.assertTrue((summary["rmse_pooled_absolute_error"] >= 0).all())
        self.assertTrue((summary["distribution_similarity"] <= 1.0).all())
        # The pooled moment is matched by the fitted intercept, but the hourly
        # RMSE profile is not forced to agree and remains a meaningful diagnostic.
        complete = summary.loc[summary["horizon_hours"] == 24]
        self.assertTrue((complete["rmse_absolute_error"] > 1e-6).all())
        self.assertTrue((complete["rmse_pooled_absolute_error"] < 1e-10).all())
        with tempfile.TemporaryDirectory() as tmp:
            plot = Path(tmp)/"diffusion.png"
            plot_grid(summary, plot, N_values=(3,), k_values=(1, 4, 24))
            self.assertTrue(plot.exists())


if __name__ == "__main__":
    unittest.main()
