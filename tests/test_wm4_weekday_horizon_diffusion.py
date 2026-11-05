import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.sweep_weekday_horizon_diffusion import (
    fit_variance, load_sweep, plot_grid, summarize_grid, unit_process_variance,
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
            summary, validation = summarize_grid(frames, min_train_days=5, min_test_days=2)
            self.assertEqual(len(summary), 2)
            self.assertTrue((summary["reported_rmse"] >= 0.0).all())
            self.assertTrue((summary["process_sigma"] >= 0.0).all())
            self.assertEqual(len(validation), 2*4*4)
            output = source/"comparison.png"
            plot_grid(summary, output, N_values=(3,), k_values=(4,))
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
