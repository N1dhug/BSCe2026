import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.predict_all_moving_horizon import historical_weekdays, horizon_analysis, oracle_f0, report_saved
from src.model.wm_4 import wm_4
from src.params.wm_4 import joint_qp_params


class test_wm4_joint_moving_horizon(unittest.TestCase):
    def test_history_uses_five_preceding_matching_weekdays(self):
        days = historical_weekdays(pd.Timestamp("2022-03-07"))
        self.assertEqual([day.day_name() for day in days], ["Monday"]*5)
        self.assertEqual(days, sorted(days))
        self.assertEqual(days[-1], pd.Timestamp("2022-02-28"))

    def test_joint_multi_day_qp_shared_coefficients_and_bounded_states(self):
        n = 24*3+1
        dt = 1.0/(n-1)
        tau = np.arange(n)*dt
        theta = np.array([-1.2, -0.3, -2.0])
        model = wm_4((0.0, 1.0), n, lambda t: 0.5)
        inputs = []
        for i in range(5):
            U = 0.7*np.sin(2*np.pi*(tau + i/7.0))
            F0, R0 = -0.3+0.12*i, 0.05*(i-2)
            state = model._solve_deterministic(
                [F0, R0], U, U, tau*24, np.full(n, 0.5), theta)
            mask = np.zeros(n, dtype=bool)
            mask[np.arange(24)*3] = True
            transition = np.zeros(n-1, dtype=bool)
            transition[:23*3] = True
            inputs.append({"I": state["I"], "O": state["O"], "U": U, "PE": U,
                           "Time": tau*24, "observation_mask": mask, "transition_mask": transition})
        fit = joint_qp_params(model).fit_many(inputs)
        self.assertEqual(fit["fit_trajectories"], 5)
        self.assertEqual(fit["beta"].shape, (8,))
        self.assertEqual(fit["init_cond"].shape, (5, 2))
        self.assertLess(fit["qp_loss"], 1e-6)
        self.assertLess(fit["deterministic_loss"], 1e-5)
        self.assertTrue(np.all(fit["r"][[0, 2]] < 0))
        self.assertLessEqual(fit["r"][1], 0)
        for reconstructed in fit["reconstructed_F"]:
            self.assertGreaterEqual(np.min(reconstructed), -1.0-1e-6)
            self.assertLessEqual(np.max(reconstructed), 1.0+1e-6)

    def test_oracle_f0_only_reconstructs_known_state(self):
        sph, n = 3, 24*3+1
        tau = np.linspace(0, 1, n)
        U = 0.5*np.cos(4*np.pi*tau)
        O = np.full(n, 0.45)
        theta = np.array([-1.2, -0.3, -2.0])
        model = wm_4((0.0, 1.0), n, lambda t: 0.45)
        f0, r0 = 0.35, 0.07
        truth = model._solve_deterministic([f0, r0], U, U, tau*24, O, theta)
        forecast = {"U": U, "PE": U, "Time": tau*24, "O": O}
        observed = truth["I"][np.arange(24)*sph]
        recovered, predicted = oracle_f0(model, forecast, observed, theta, r0, sph)
        self.assertAlmostEqual(recovered, f0, places=7)
        np.testing.assert_allclose(predicted["I"][np.arange(24)*sph], observed, atol=1e-9)

    def test_report_only_from_saved_files_includes_median_theta(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            daily = pd.DataFrame([
                {"day": f"2022-03-0{i+1}", "weekday": "Monday", "rmse": 0.01*i,
                 "theta_F": -0.4-i/10, "theta_R": -0.8-i/10, "theta_U": -0.1-i/20,
                 "design_rank": 8, "design_columns": 8, "damping_branch": "under",
                 "damping_ratio": 0.5, "fit_qp_loss": 0.01, "fit_seconds": 0.1,
                 "mse": 0.01, "bias": 0.0} for i in (0, 1)])
            hourly = pd.DataFrame({"error": [0.1, -0.1], "baseline_error": [0.2, -0.2]})
            daily.to_csv(output / "daily_metrics.csv", index=False)
            hourly.to_csv(output / "hourly_predictions.csv", index=False)
            pd.DataFrame(columns=["day", "reason"]).to_csv(output / "skipped_days.csv", index=False)
            buffer = io.StringIO()
            with redirect_stdout(buffer): result = report_saved(output)
            self.assertEqual(len(result), 2)
            self.assertIn("50%", buffer.getvalue())
            self.assertIn("Theta correlations:", buffer.getvalue())

    def test_aggregate_rmse_and_cumulative_quintiles(self):
        days = pd.to_datetime(["2022-03-01", "2022-03-02"])
        data = pd.DataFrame([{"day": day, "hour": h, "error": float(i*(h+1))}
                             for i, day in enumerate(days, 1) for h in range(24)])
        table = horizon_analysis(data)
        self.assertEqual(len(table), 24)
        self.assertAlmostEqual(table.loc[0, "aggregate_rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(table.loc[0, "aggregate_cumulative_rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(table.loc[0, "cumulative_rmse_q20"], 1.2)


if __name__ == "__main__": unittest.main()
