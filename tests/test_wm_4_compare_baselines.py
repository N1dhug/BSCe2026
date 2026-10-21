import unittest

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.compare_baselines import (
    BASELINES, METHODS, baseline_before, fit_and_predict, matched_cohort, summaries,
)
from src.model.wm_4 import wm_4


class constant_baseline:
    metadata = {"samples": 200, "mse": 0.0}

    def predict(self, index, temperature):
        return np.full(len(index), 0.5, dtype=float)


class wm_4_baseline_comparison_tests(unittest.TestCase):
    def test_baseline_models_registered(self):
        self.assertEqual(set(BASELINES), {"statistical_output", "weekly_average", "weekly_temperature"})

    def test_strict_matched_cohort(self):
        rows, errors = [], []
        for origin, methods in ((pd.Timestamp("2022-03-01"), ("A", "B", "C")),
                                (pd.Timestamp("2022-03-02"), ("A", "C"))):
            for name in methods:
                rows.append({"group": "ALL", "origin": origin, "baseline": name})
                errors.append({"group": "ALL", "origin": origin, "baseline": name})
        matched, scored = matched_cohort(pd.DataFrame(rows), pd.DataFrame(errors), ("A", "B", "C"))
        self.assertEqual(len(matched), 3)
        self.assertEqual(len(scored), 3)
        self.assertEqual(matched["origin"].nunique(), 1)

    def test_synthetic_standard_and_oracle(self):
        steps_per_hour, hours = 4, 48
        time = np.arange(hours * steps_per_hour+1) / (24.0 * steps_per_hour)
        prices_hourly = -0.35 + 0.2 * np.sin(np.arange(hours) * 0.3)
        prices = np.r_[prices_hourly, prices_hourly[-1]][np.minimum(
            np.arange(len(time)) // steps_per_hour, hours)]
        model = wm_4((0.0, 2.0), len(time), lambda _t: 0.5)
        truth = model._solve_deterministic(
            [0.12, 0.06], prices, prices, time,
            np.full(len(time), 0.5), np.array([-2.0, -0.5, -3.0]))
        index = pd.date_range("2022-02-01", periods=hours, freq="h")
        frame = pd.DataFrame({
            "I": truth["I"][np.arange(hours) * steps_per_hour],
            "U": prices_hourly,
            "Temp": np.full(hours, 5.0)}, index=index)
        row, errors, example = fit_and_predict(
            frame, pd.Timestamp("2022-02-02"), "constant", constant_baseline(),
            steps_per_hour, "observed", 12, 1e-8)
        self.assertEqual(set(example), {"origin", "reference", *METHODS})
        self.assertTrue(np.isfinite(row["standard_rmse"]))
        self.assertLessEqual(row["oracle_rmse"], row["standard_rmse"] + 1e-6)
        self.assertEqual(len([k for k in errors if k.startswith("standard_error_")]), 24)
        row["group"], errors["group"] = "ALL", "ALL"
        summary, horizon = summaries(pd.DataFrame([row]), pd.DataFrame([errors]))
        self.assertEqual(summary["windows"].iloc[0], 1)
        self.assertEqual(len(horizon), 24 * len(METHODS))


if __name__ == "__main__": unittest.main()
