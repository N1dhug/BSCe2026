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
        q = [unit_process_variance(-1.0, -0.25, 4, h) for h in range(5)]
        self.assertAlmostEqual(q[0], 0.0)
        self.assertTrue(np.isfinite(q).all())
        self.assertTrue(all(q[i+1] > q[i] for i in range(4)))

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

    def test_actual_sweep_schema_and_2x2_plot(self):
        rng = np.random.default_rng(123)
        fits, scores, components = [], [], []
        for baseline in ("weekly_average", "weekly_temperature"):
            for i in range(12):
                day = pd.Timestamp("2022-01-01")+pd.Timedelta(days=i)
                fits.append({"day":day, "baseline":baseline, "training_weekdays":3,
                             "theta_F":-1.0, "theta_R":-0.25})
                for hour in range(4):
                    component = {"day":day, "baseline":baseline, "training_weekdays":3,
                                 "hour":hour, "prediction_at_initial_zero":0.3,
                                 "sensitivity_F0":0.0, "sensitivity_R0":1.0,
                                 "observed":0.3+rng.normal(0,0.08)}
                    components.append(component)
                scores.append({"day":day, "baseline":baseline, "training_weekdays":3,
                               "horizon_hours":4, "oracle_F0":0.0, "oracle_R0":0.0})
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            pd.DataFrame(fits).to_csv(source/COLUMNS["fit"], index=False)
            pd.DataFrame(scores).to_csv(source/COLUMNS["score"], index=False)
            pd.DataFrame(components).to_csv(source/COLUMNS["component"], index=False)
            (source/"configuration.json").write_text('{"steps_per_hour": 4}', encoding="utf-8")
            frames, _ = load_sweep(source)
            self.assertEqual(len(frames), 12*2*4)
            summary, fitted = summarize_grid(frames)
            self.assertEqual(len(summary), 2)
            self.assertTrue((summary["reported_rmse"] >= 0.0).all())
            self.assertTrue((summary["process_sigma"] >= 0.0).all())
            self.assertEqual(len(fitted), 12*2*4)
            self.assertEqual(int(summary["days"].min()), 12)
            self.assertEqual(int(summary["forecasts"].min()), 12*4)
            output = source/"comparison.png"
            plot_grid(summary, output, N_values=(3,), k_values=(4,))
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
