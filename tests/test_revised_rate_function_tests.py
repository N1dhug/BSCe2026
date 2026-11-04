import unittest

import numpy as np
import pandas as pd

from src.exe.data.revised_rate_function_tests import (
    _holm_adjust, analyze, reconstruct_day
)


class test_revised_rate_function_tests(unittest.TestCase):
    def test_reconstruction_matches_hourly_transitions(self):
        class zero_baseline:
            def predict(self, index, temperature):
                return np.zeros(len(index))
        day = pd.Timestamp("2022-05-02")
        idx = pd.date_range(day, periods=24, freq="h")
        rate = np.arange(24, dtype=float)/100.0
        frame = pd.DataFrame({"I": rate, "Temp": np.full(24, 12.0),
                              "U": np.linspace(-1, 1, 24)}, index=idx)
        forward = reconstruct_day(day, frame, zero_baseline())
        central = reconstruct_day(day, frame, zero_baseline(), difference="central")
        self.assertEqual(len(forward), 23)
        self.assertEqual(len(central), 22)
        self.assertTrue(np.allclose(forward["R_dot"], .01))
        self.assertTrue(np.allclose(central["R_dot"], .01))
        self.assertAlmostEqual(float(forward["C"].iloc[2]), rate[0]+rate[1])
        self.assertEqual(int(central["hour"].iloc[0]), 1)

    def test_nonlinearity_detectable_on_chronological_holdout(self):
        generator = np.random.default_rng(41)
        blocks = []
        days = pd.date_range("2022-02-01", periods=75, freq="D")
        for name in ("weekly_average", "weekly_temperature"):
            for day in days:
                r = generator.normal(0, .1, 23)
                u = generator.uniform(-1, 1, 23)
                c = np.r_[0, np.cumsum(r[:-1])]
                y = .1*c-.25*r+.06*u+.8*(r/.13)**2
                y += generator.normal(0, .018, len(y))+generator.normal(0, .08)
                blocks.append(pd.DataFrame({
                    "day": day, "hour": np.arange(23), "baseline": name,
                    "C": c, "R": r, "U": u, "R_dot": y,
                }))
        scores, coefficients, residuals, split = analyze(pd.concat(blocks, ignore_index=True))
        for name in ("weekly_average", "weekly_temperature"):
            selected = scores.loc[(scores["baseline"] == name) &
                                  (scores["candidate"] == "rate_quadratic")].iloc[0]
            self.assertTrue(selected["significant_holm"])
            self.assertTrue(selected["material_heldout_gain"])
            self.assertGreater(float(selected["heldout_gain_pct"]), 40)
            self.assertEqual(split[name]["train_days"], 56)
        self.assertFalse(coefficients.empty)
        self.assertEqual(len(residuals), 2*19*23)

    def test_holm_correction(self):
        self.assertTrue(np.allclose(_holm_adjust([.01, .03, .04]), [.03, .06, .06]))


if __name__ == "__main__": unittest.main()
