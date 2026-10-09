"""Regression tests for BM1 drift/diffusion least-squares fitting."""

import unittest

import numpy as np

from src.model.bm_1 import bm_1
from src.params.bm_1 import lsq_params


class BM1ParameterTests(unittest.TestCase):
    @staticmethod
    def case(*, sigma=0.0, n=5001, end=10.0, seed=42):
        t = np.linspace(0.0, end, n)
        B = lambda s: 0.35 + 0.2 * np.cos(0.3 * np.asarray(s))
        U = 0.5 + 0.45 * np.sin(1.2 * t)
        f = lambda x: 1.0 - 2.0 * np.asarray(x)
        g = lambda u: 1.0 - 2.0 * np.asarray(u)
        C, Xi, k = 2.0, 0.65, 1.6
        X = np.empty(n)
        X[0] = 0.25
        rng = np.random.default_rng(seed)
        for j in range(n - 1):
            dt = t[j + 1] - t[j]
            delta = np.tanh(0.5 * k * (f(X[j]) + g(U[j])))
            b = B(t[j])
            mu = Xi / C * delta * (1.0 - b if delta >= 0 else b)
            X[j + 1] = X[j] + mu * dt + sigma * X[j] * (1.0 - X[j]) * np.sqrt(dt) * rng.standard_normal()
        model = bm_1(B, lambda s: np.asarray(s) * 0.0 + 0.5, (0.0, end), n)
        return model, {"Time": t, "X": X, "U": U}, f, g

    def test_exact_euler_drift_and_zero_diffusion(self):
        model, data, f, g = self.case()
        fitted = lsq_params(model).fit(data, f, g, C=2.0)
        self.assertAlmostEqual(fitted["Xi"], 0.65, places=5)
        self.assertAlmostEqual(fitted["k"], 1.6, places=5)
        self.assertLess(fitted["sigma_x"], 0.005)
        self.assertAlmostEqual(fitted["parameters"][0](data["Time"][100]), data["U"][100], places=12)
        self.assertEqual(fitted["diffusion_samples"], data["X"].size - 1)

    def test_only_gain_is_identified_without_capacity_or_flexibility(self):
        model, data, f, g = self.case()
        fitted = lsq_params(model).fit(data, f, g)
        self.assertAlmostEqual(fitted["gain"], 0.65 / 2.0, places=5)
        self.assertIsNone(fitted["C"])
        self.assertIsNone(fitted["Xi"])
        self.assertIsNone(fitted["parameters"])

    def test_estimate_positive_diffusion_with_fixed_gain(self):
        model, data, f, g = self.case(sigma=0.25, n=20001, end=100.0)
        self.assertTrue(np.all((data["X"] >= 0.0) & (data["X"] <= 1.0)))
        fitted = lsq_params(model).fit(data, f, g, C=2.0, Xi=0.65)
        self.assertAlmostEqual(fitted["sigma_x"], 0.25, delta=0.06)
        self.assertGreater(fitted["k"], 0.0)
        self.assertAlmostEqual(fitted["gain"], 0.65 / 2.0, places=12)

    def test_requires_observed_state_and_interior_diffusion_samples(self):
        model, data, f, g = self.case(n=21)
        with self.assertRaisesRegex(ValueError, "Observed BM1 state"):
            lsq_params(model).fit({"Time": data["Time"], "U": data["U"]}, f, g)
        with self.assertRaisesRegex(ValueError, "Xi=0"):
            lsq_params(model).fit(data, f, g, C=2.0, Xi=0.0)
        boundary = dict(data, X=np.zeros_like(data["X"]))
        with self.assertRaisesRegex(ValueError, "Too few interior"):
            lsq_params(model).fit(boundary, f, g, C=2.0)


if __name__ == "__main__": unittest.main()
