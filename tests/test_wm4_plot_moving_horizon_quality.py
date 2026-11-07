import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.plot_moving_horizon_quality import (
    accumulate, empty_pool, historical_errors, historical_summary, plot_quality,
    pooled_row, prediction_summary, saved_baseline,
)


class test_wm4_plot_moving_horizon_quality(unittest.TestCase):
    @staticmethod
    def frame():
        times = pd.date_range("2022-03-01", periods=24*35, freq="h")
        demand = 0.5+0.015*np.sin(np.arange(len(times))*2*np.pi/24)
        return pd.DataFrame({"I":demand, "U":np.full(len(times), 0.1),
                             "Temp":np.full(len(times), 8.0)}, index=times)

    @staticmethod
    def selection():
        return ("weekly_average",), (2,), (3,), None, None

    def test_pooled_rmse_is_sqrt_mean_mse_not_mean_rmse(self):
        pool = empty_pool()
        key = ("weekly_average", 2, 3)
        accumulate(pool, key, [0., 0., 3.])
        accumulate(pool, key, [0., 0., 1.])
        result = pooled_row(key, pool[key])
        self.assertAlmostEqual(result["rmse"], np.sqrt(10./6.))
        self.assertAlmostEqual(result["mae"], 4./6.)
        self.assertAlmostEqual(result["bias"], 4./6.)
        self.assertEqual(result["observations"], 6)
        self.assertEqual(result["origins"], 2)

    def test_prediction_uses_cumulative_saved_k_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            records = [
                {"day":"2022-04-09", "baseline":"weekly_average", "training_weekdays":2,
                 "horizon_hours":3, "mse":1/3, "mae":1/3, "bias":1/3},
                {"day":"2022-04-09 01:00:00", "baseline":"weekly_average",
                 "training_weekdays":2, "horizon_hours":3,
                 "mse":4/3, "mae":2/3, "bias":-2/3},
                {"day":"2022-04-09 01:00:00", "baseline":"weekly_average",
                 "training_weekdays":2, "horizon_hours":4,
                 "mse":0, "mae":0, "bias":0},
            ]
            pd.DataFrame(records).to_csv(Path(directory)/"oracle_horizon_scores.csv", index=False)
            result = prediction_summary(directory, self.selection()).iloc[0]
            self.assertAlmostEqual(result["rmse"], np.sqrt(5./6.))
            self.assertAlmostEqual(result["mae"], 0.5)
            self.assertAlmostEqual(result["bias"], -1./6.)
            self.assertEqual(result["origins"], 2)
            self.assertEqual(result["observations"], 6)

    def test_historical_replay_preserves_saved_baseline_and_uses_k_hours_only(self):
        frame = self.frame()
        coef = np.full(168, .5)
        original = {"day":"2022-04-09 21:00:00", "baseline":"weekly_average",
                    "training_weekdays":"2", "horizon_hours":"3",
                    "theta_F":"-1", "theta_U":"0", "theta_R":"-1",
                    "baseline_coefficients":json.dumps(coef.tolist()),
                    "baseline_beta_C":"0", "baseline_chill_center":"0"}
        states = [{"day":original["day"], "baseline":"weekly_average", "training_weekdays":"2",
                   "horizon_hours":"3", "historical_day":f"2022-03-{day:02d} 21:00:00",
                   "historical_F0":"0", "historical_R0":"0"} for day in (26, 19)]
        baseline = saved_baseline(original)
        self.assertEqual(float(baseline.predict(pd.DatetimeIndex(["2022-03-26 22:00"]),
                                                 np.array([8.0]))[0]), 0.5)
        residuals = historical_errors(frame, original, states, 4)
        self.assertEqual(len(residuals), 6)
        expected = np.concatenate([
            0.5-frame.loc[pd.date_range(pd.Timestamp(r["historical_day"]), periods=3, freq="h"), "I"].to_numpy()
            for r in states
        ])
        np.testing.assert_allclose(residuals, expected, atol=1e-14)

    def test_vectorized_replay_matches_wm4_euler_with_nonzero_price_and_initial_states(self):
        from src.model.wm_4 import wm_4
        from src.tools.baseline import output_callable
        frame = self.frame()
        origins = pd.to_datetime(["2022-03-19 21:00:00", "2022-03-26 21:00:00"])
        k, steps = 6, 4
        # Include the 21:00 -> next calendar day crossing.
        frame.loc[:, "U"] = 0.2*np.sin(np.arange(len(frame))/24)
        row = {"day":"2022-04-09 21:00:00", "baseline":"weekly_average",
               "training_weekdays":"2", "horizon_hours":str(k),
               "theta_F":"-0.8", "theta_U":"-0.2", "theta_R":"-1.4",
               "baseline_coefficients":json.dumps(np.linspace(.3, .6, 168).tolist()),
               "baseline_beta_C":"0", "baseline_chill_center":"0"}
        states = [{"day":row["day"], "baseline":"weekly_average", "training_weekdays":"2",
                   "horizon_hours":str(k), "historical_day":str(origin),
                   "historical_F0":str(f0), "historical_R0":str(r0)}
                  for origin, f0, r0 in zip(origins, [0.1, -0.2], [0.02, -0.04])]
        residual = historical_errors(frame, row, states, steps).reshape(2, k)
        baseline = saved_baseline(row)
        elapsed = np.arange(k*steps+1)/steps
        sample = np.arange(k)*steps
        for i, origin in enumerate(origins):
            index = origin+pd.to_timedelta(elapsed, unit="h")
            raw = frame.reindex(index)
            U = frame.reindex(pd.date_range(origin, periods=k, freq="h"))["U"].to_numpy()
            U_fine = U[np.minimum(np.floor(elapsed+1e-10).astype(int), k-1)]
            T_fine = np.full(len(index), 8.)
            O_fine = baseline.predict(index, T_fine)
            model = wm_4((0., k/24.), len(index), output_callable(index, O_fine))
            trajectory = model._solve_deterministic([float(states[i]["historical_F0"]),
                                                     float(states[i]["historical_R0"])],
                         U_fine, U_fine, elapsed, O_fine,
                         np.array([-0.8, -0.2, -1.4]))
            observed = frame.reindex(pd.date_range(origin, periods=k, freq="h"))["I"].to_numpy()
            np.testing.assert_allclose(residual[i], trajectory["I"][sample]-observed,
                                       atol=1e-12, rtol=1e-12)

    def test_replay_streams_saved_initial_states_without_refitting(self):
        frame = self.frame()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            origin = "2022-04-09 21:00:00"
            row = {"day":origin, "baseline":"weekly_average", "training_weekdays":2,
                   "horizon_hours":3, "theta_F":-1, "theta_U":0, "theta_R":-1,
                   "baseline_coefficients":json.dumps(np.full(168, 0.5).tolist()),
                   "baseline_beta_C":0., "baseline_chill_center":0.}
            states = [{"day":origin, "baseline":"weekly_average", "training_weekdays":2,
                       "horizon_hours":3, "historical_day":f"2022-03-{day:02d} 21:00:00",
                       "historical_F0":0., "historical_R0":0.} for day in (19, 26)]
            pd.DataFrame([row]).to_csv(output/"parameter_estimates.csv", index=False)
            pd.DataFrame(states).to_csv(output/"historical_initial_states.csv", index=False)
            with patch("src.exe.run_model.wm_4.moving_horizon_common.group_frame", return_value=frame):
                result = historical_summary(output, self.selection(), 4, "ALL")
            expected = historical_errors(frame, row, [
                {key:str(value) for key, value in state.items()} for state in states
            ], 4)
            summary = result.iloc[0]
            self.assertAlmostEqual(summary["rmse"], np.sqrt(np.mean(expected**2)))
            self.assertAlmostEqual(summary["mae"], np.mean(np.abs(expected)))
            self.assertAlmostEqual(summary["bias"], np.mean(expected))
            self.assertEqual(summary["observations"], 6)
            self.assertEqual(summary["origins"], 1)

    def test_plot_creates_three_cumulative_metrics_on_requested_grid(self):
        summary = pd.DataFrame([{"baseline":"weekly_average", "training_weekdays":2,
                                 "horizon_hours":3, "rmse":0.1, "mae":0.05, "bias":-0.02}])
        with tempfile.TemporaryDirectory() as directory:
            path = plot_quality(summary, "prediction", ("weekly_average",), (2,), (3,), Path(directory))
            self.assertTrue(path.is_file())


if __name__ == "__main__": unittest.main()
