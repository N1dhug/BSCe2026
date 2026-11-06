import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.sweep_weekday_horizon import COLUMNS, interval_window
from src.exe.run_model.wm_4.sweep_weekday_horizon_diffusion import (
    fit_variance, load_sweep, plot_grid, same_period, summarize_grid, unit_process_variance,
)


class wm4_weekday_diffusion_tests(unittest.TestCase):
    def test_kernel_zero_at_origin_and_grows(self):
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

    def test_common_period_matches_all_backlogs_and_horizons(self):
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

    def test_sweep_interval_is_exactly_k_hours_not_24(self):
        class baseline:
            @staticmethod
            def predict(index, temperature): return np.full(len(index), 0.45)

        day = pd.Timestamp("2022-04-04")
        index = pd.date_range(day, periods=24, freq="h")
        frame = pd.DataFrame({"I":0.5+0.01*np.sin(np.arange(24)/5),
                              "U":np.zeros(24), "Temp":np.full(24, 12.)}, index=index)
        for k in (1, 6, 24):
            model, data, _, observed = interval_window(frame, day, baseline(), 4, k)
            self.assertEqual(model.t_steps, 4*k+1)
            self.assertAlmostEqual(model.dt, 1./96.)
            self.assertEqual(len(observed), k)
            self.assertEqual(int(np.sum(data["transition_mask"])), min(k, 23)*4)
            self.assertEqual(int(np.sum(data["observation_mask"])), min(k+1, 24))

    def test_horizon_specific_saved_fit_states_required(self):
        day, history = pd.Timestamp("2022-05-01"), pd.Timestamp("2022-04-24")
        fit = [{"day":day, "baseline":"weekly_average", "training_weekdays":1, "horizon_hours":6,
                "theta_F":-1., "theta_U":-0.3, "theta_R":-0.5, "training_days":"2022-04-24"}]
        states = [{"day":day, "baseline":"weekly_average", "training_weekdays":1,
                   "horizon_hours":6, "historical_day":history,
                   "historical_F0":0.2, "historical_R0":-0.1}]
        scores = [{"day":day, "baseline":"weekly_average", "training_weekdays":1, "horizon_hours":6}]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            pd.DataFrame(fit).to_csv(source/COLUMNS["fit"], index=False)
            pd.DataFrame(states).to_csv(source/COLUMNS["state"], index=False)
            pd.DataFrame(scores).to_csv(source/COLUMNS["score"], index=False)
            (source/"configuration.json").write_text('{"steps_per_hour": 4}', encoding="utf-8")
            fits, historical, _, cfg = load_sweep(source, ("weekly_average",))
            self.assertEqual(len(fits), 1)
            self.assertEqual(len(historical), 1)
            self.assertEqual(int(fits.iloc[0]["horizon_hours"]), 6)
            self.assertEqual(int(historical.iloc[0]["horizon_hours"]), 6)
            self.assertEqual(cfg["steps_per_hour"], 4)
            pd.DataFrame([{k:v for k, v in fit[0].items() if k != "horizon_hours"}]).to_csv(source/COLUMNS["fit"], index=False)
            with self.assertRaises(ValueError): load_sweep(source, ("weekly_average",))

    def test_diffusion_variances_use_each_matching_N_times_k_window(self):
        rows = []
        for d in range(2):
            day = pd.Timestamp("2022-04-09")+pd.Timedelta(days=d)
            for n in range(2):
                for k in (2, 4):
                    for h, e in enumerate((0.1, 0.1, 0.1, 0.6)[:k]):
                        rows.append({
                            "day":day, "historical_day":day-pd.Timedelta(days=7*(n+1)),
                            "baseline":"weekly_average", "training_weekdays":2,
                            "horizon_hours":k, "hour":h, "observed":e, "predicted":0.,
                            "error":e, "unit_process_variance":float(h),
                        })
        summary, fitted = summarize_grid(pd.DataFrame(rows), horizons=(2, 4))
        self.assertEqual(len(summary), 2)
        self.assertEqual(len(fitted), 4)
        short = fitted.loc[fitted["horizon_hours"] == 2].sort_values("day")
        full = fitted.loc[fitted["horizon_hours"] == 4].sort_values("day")
        self.assertTrue((short["observations"] == 2*2).all())
        self.assertTrue((full["observations"] == 2*4).all())
        self.assertTrue((short["process_variance_rate"] < 1e-12).all())
        self.assertTrue((full["process_variance_rate"] > 0).all())
        self.assertFalse(np.allclose(short["observation_variance"], full["observation_variance"]))

    def test_each_baseline_uses_exact_historical_traj_and_horizon(self):
        rng, rows = np.random.default_rng(203), []
        for baseline in ("weekly_average", "weekly_temperature"):
            for d in range(8):
                day = pd.Timestamp("2022-04-09")+pd.Timedelta(days=d)
                for k in (1, 4, 24):
                    for n in range(3):
                        for h in range(k):
                            K = unit_process_variance(-1., -0.5, 4, h)
                            error = rng.normal(0, np.sqrt(0.002+0.05*K))
                            rows.append({
                                "day":day, "historical_day":day-pd.Timedelta(days=7*(n+1)),
                                "baseline":baseline, "training_weekdays":3, "horizon_hours":k,
                                "hour":h, "observed":0.3+error, "predicted":0.3,
                                "error":error, "unit_process_variance":K,
                            })
        summary, fitted = summarize_grid(pd.DataFrame(rows), horizons=(1, 4, 24))
        self.assertEqual(len(summary), 2*3)
        self.assertEqual(len(fitted), 2*8*3)
        self.assertEqual(set(fitted["horizon_hours"].unique()), {1, 4, 24})
        self.assertTrue((fitted["observations"] == 3*fitted["horizon_hours"]).all())
        self.assertTrue((summary["rmse_absolute_error"] >= 0).all())
        self.assertTrue((summary["rmse_pooled_absolute_error"] >= 0).all())
        self.assertTrue((summary["distribution_similarity"] <= 1.0).all())
        complete = summary.loc[summary["horizon_hours"] == 24]
        self.assertTrue((complete["rmse_absolute_error"] > 1e-6).all())
        self.assertTrue((complete["rmse_pooled_absolute_error"] < 1e-10).all())
        with tempfile.TemporaryDirectory() as tmp:
            plot = Path(tmp)/"diffusion.png"
            plot_grid(summary, plot, N_values=(3,), k_values=(1, 4, 24))
            self.assertTrue(plot.exists())


if __name__ == "__main__":
    unittest.main()
