"""Regression tests for the 12h parameter/O trajectory and genuine 6h MPC preset."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.predict_all_days_mpc import (
    DAY_HOURS, _fit_shifted_history, _window_input, run_all_days,
)


class Baseline:
    def predict(self, index, temperature):
        return np.full(len(index), 0.45, dtype=float)


class WM4SixHourTwelveHourTests(unittest.TestCase):
    @staticmethod
    def data():
        timestamps = pd.date_range("2022-01-01", periods=125 * 24 + 1, freq="h")
        i = np.arange(len(timestamps))
        return pd.DataFrame({"I": 0.5 + 0.04 * np.sin(i / 24),
            "U": np.zeros(len(i)), "Temp": np.full(len(i), 6.)}, index=timestamps)

    def test_twelve_hour_baseline_trajectory_and_observation_grid(self):
        origin = pd.Timestamp("2022-03-09 14:00")
        source = self.data()
        data, index = _window_input(source, origin, 2, Baseline(), hours=12)
        self.assertEqual(len(index), 25)
        self.assertEqual(index[0], origin)
        self.assertEqual(index[-1], origin + pd.Timedelta(hours=12))
        self.assertEqual(int(data["observation_mask"].sum()), 12)
        self.assertEqual(len(data["O"]), 25)
        self.assertTrue(np.allclose(data["O"], 0.45))
        self.assertEqual(len(data["transition_mask"]), 24)
        self.assertTrue(np.all(data["transition_mask"]))

    def test_five_weekday_parameter_fit_uses_only_twelve_hour_windows(self):
        origin = pd.Timestamp("2022-03-09 14:00")
        seen = []

        class FakeFit:
            def __init__(self, model): self.model = model
            def fit_many(self, datasets, seeds, **kwargs):
                seen.extend(datasets)
                self_model = self.model
                assert self_model.t_steps == 25
                return {"init_cond": seeds, "r": np.array([-1., -0.2, -3.]),
                    "x_hat": np.mean(seeds, axis=0),
                    "deterministic_states": [{"I": d["I"]} for d in datasets]}

        with patch("src.exe.run_model.wm_4.predict_all_days_mpc.optimized_qp_params",
            FakeFit):
            fit, model, histories = _fit_shifted_history(
                self.data(), origin, 5, 2, Baseline(), fit_hours=12)
        self.assertEqual(model.t_steps, 25)
        self.assertEqual(fit["fit_hours"], 12)
        self.assertEqual(len(histories), 5)
        self.assertEqual(len(seen), 5)
        self.assertTrue(all(len(a["I"]) == 25 and len(a["O"]) == 25 for a in seen))
        self.assertTrue(all(int(a["observation_mask"].sum()) == 12 for a in seen))
        self.assertTrue(all(origin - h == pd.Timedelta(weeks=k)
            for k, h in enumerate(fit["training_origins"], 1)))

    def test_run_all_days_preset_uses_six_lead_mpc_and_12h_fit(self):
        calls = []

        def fake_origin(frame, origin, baseline_weeks, steps_per_hour, output_model,
            max_horizon, *, fit_hours, **kwargs):
            calls.append((max_horizon, fit_hours, kwargs["sigma_fit_mode"]))
            rows = [{"lead_hours": h, "target_time": origin + pd.Timedelta(hours=h),
                "forecast_error": float(h * 0.01), "initial_variance": 0.01,
                "process_kernel_unit": 0.03} for h in range(1, max_horizon + 1)]
            return rows, {"origin": origin}

        with patch("src.exe.run_model.wm_4.predict_all_days_mpc.fit_expected_output",
            return_value=(Baseline(), {})), patch(
            "src.exe.run_model.wm_4.predict_all_days_mpc._mpc_origin",
            side_effect=fake_origin), patch(
            "src.exe.run_model.wm_4.predict_all_days_mpc._archive_reconstruction_origin"):
            forecast, states, skipped = run_all_days(
                self.data(), start="2022-03-09", end="2022-03-09",
                steps_per_hour=2, sigma_fit_mode="mpc_replay_28d_6h")
        self.assertEqual(len(calls), 24)
        self.assertTrue(all(h == 6 and f == 12 and mode == "mpc_replay_28d_6h"
            for h, f, mode in calls))
        self.assertEqual(len(forecast), 24 * 6)
        self.assertTrue((forecast["lead_hours"] <= 6).all())
        self.assertEqual(len(states), 24)
        self.assertFalse(skipped)

        with self.assertRaisesRegex(ValueError, "6-hour MPC"):
            run_all_days(self.data(), "2022-03-09", "2022-03-09",
                max_horizon=8, sigma_fit_mode="mpc_replay_28d_6h")


if __name__ == "__main__": unittest.main()
