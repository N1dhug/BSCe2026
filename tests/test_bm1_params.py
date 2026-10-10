"""Regression tests for demand-only BM1 nonlinear least squares."""

import unittest

import numpy as np

from src.model.bm_1 import bm_1
from src.params.bm_1 import lsq_params


class BM1ParameterTests(unittest.TestCase):
    @staticmethod
    def case(n=181):
        time = np.linspace(0.0, 12.0, n)
        B = lambda t: 0.45 + 0.15 * np.sin(0.9 * np.asarray(t))
        U = 0.5 + 0.45 * np.cos(1.7 * time)
        model = bm_1(B, lambda t: np.asarray(t) * 0.0 + 0.5, (0.0, 12.0), n)
        fitter = lsq_params(model)
        truth = {"alpha": 3.0, "beta": 5.0, "k": 2.0, "C": 1.5, "Xi": 0.7, "X0": 0.23}
        basis = {"Time": time, "U": U, "B": B(time)}
        X, D = fitter._predict({**basis, "dt": np.diff(time)}, truth["X0"], truth, 3)
        return fitter, {**basis, "I": D}, truth

    def test_logistic_has_exact_normalized_endpoints(self):
        for slope in (0.05, 1.0, 5.0, 20.0):
            fun = lsq_params.response(slope)
            self.assertAlmostEqual(fun(0.0), 1.0, places=12)
            self.assertAlmostEqual(fun(1.0), -1.0, places=12)
            self.assertAlmostEqual(fun(0.5), 0.0, places=12)
            self.assertTrue(np.all(np.diff(fun(np.linspace(0.0, 1.0, 50))) < 0.0))

    def test_demand_fit_from_unobserved_state_and_X0_guess(self):
        fitter, data, true = self.case()
        result = fitter.fit(data, 0.38, C=true["C"], Xi=true["Xi"],
            alpha0=2.5, beta0=4.5, k0=1.8, substeps=3)
        self.assertLess(result["mse"], 1e-7)
        self.assertAlmostEqual(result["X0"], true["X0"], delta=0.03)
        self.assertEqual(result["parameters"][3], true["C"])
        self.assertAlmostEqual(result["parameters"][1](0.0), 1.0)
        self.assertIsNone(result["sigma_x"])
        self.assertNotIn("X", data)

    def test_fixed_X0_and_interval_sample_data(self):
        fitter, data, true = self.case(n=81)
        data["I"] = data["I"][:-1]
        data["U"] = data["U"][:-1]
        result = fitter.fit(data, true["X0"], C=true["C"], Xi=true["Xi"],
            alpha0=true["alpha"], beta0=true["beta"], k0=true["k"], fit_X0=False)
        self.assertAlmostEqual(result["X0"], true["X0"])
        self.assertEqual(result["X"].size, data["Time"].size)
        self.assertEqual(result["D"].size, data["I"].size)
        self.assertLess(result["mse"], 1e-6)

    def test_unknown_capacity_and_flexibility_are_fitted(self):
        fitter, data, true = self.case()
        result = fitter.fit(data, true["X0"], alpha0=true["alpha"], beta0=true["beta"],
            k0=true["k"], C0=true["C"], Xi0=true["Xi"], fit_X0=False, max_nfev=20)
        self.assertLess(result["mse"], 1e-10)
        self.assertIn("C", result["free_parameters"])
        self.assertIn("Xi", result["free_parameters"])

    def test_diffusion_requires_invertible_noiseless_demand(self):
        fitter, data, true = self.case(n=251)
        rng = np.random.default_rng(7)
        x = np.empty(len(data["Time"]))
        x[0] = true["X0"]
        for k, dt in enumerate(np.diff(data["Time"])):
            b, u = data["B"][k], data["U"][k]
            mean = fitter._advance(x[k], dt, b, u, true, 3)
            x[k + 1] = mean + 0.12 * x[k] * (1.0 - x[k]) * np.sqrt(dt) * rng.standard_normal()
        self.assertTrue(np.all((x > 0.0) & (x < 1.0)))
        measured = dict(data, I=np.array([fitter._demand(x[i], data["B"][i], data["U"][i], true) for i in range(len(x))]))
        out = fitter._diffusion(fitter._prepare(measured), true, 0.1, 3, 1e-3)
        self.assertAlmostEqual(out["sigma_x"], 0.12, delta=0.04)

    def test_rejects_invalid_demand_and_state_guess(self):
        fitter, data, _ = self.case(n=41)
        with self.assertRaisesRegex(ValueError, "requires observed demand"):
            fitter.fit({"Time": data["Time"], "U": data["U"]}, 0.5)
        with self.assertRaisesRegex(ValueError, "X0 must"):
            fitter.fit(data, 1.2)
        with self.assertRaisesRegex(ValueError, "normalized"):
            fitter.fit({**data, "I": data["I"] + 2.0}, 0.5)


if __name__ == "__main__": unittest.main()
