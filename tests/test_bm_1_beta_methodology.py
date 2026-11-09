"""Fast synthetic checks for the BM1 adaptive beta-mixture chapter-four methodology.

These do not load the real ALL workbook or run expensive real-data likelihood fits.
"""
import unittest

import numpy as np
import pandas as pd

from src.method.bm_1_beta import bm_1_beta
from src.method.bm_1_fv_reference import bm_1_fv_reference
from src.params.bm_1_beta import bm_1_beta_params
from src.exe.run_model.bm_1.fit_beta_moving_horizon import (
    historical_origins, historical_training, score, timed)
from src.validation.exe.run_model.bm_1.fit_beta_moving_horizon import (
    bm_1_beta_moving_horizon_validation as checks)


class BM1BetaMethodologyTests(unittest.TestCase):
    @staticmethod
    def parameters():
        return {"alpha": 2.0, "beta": 2.0, "k": 1.0, "Xi": 0.3,
                "C": 1.2, "sigma_x": 0.15, "sigma_y": 0.03}

    @staticmethod
    def data():
        return {"Time":np.arange(4,dtype=float),
                "B":np.array([0.3,0.34,0.32,0.31]),
                "U":np.array([0.5,0.6,0.6,0.4]),
                "I":np.array([0.3,0.35,0.3,0.33])}

    def test_concentrated_beta_quadrature_preserves_moments_without_overflow(self):
        """Normalized Jacobi eigenweights must work when scipy roots_jacobi overflows."""
        for mean, variance in ((0.25,1e-6),(0.5,1e-5),(0.6,1e-8),(0.96,1e-6)):
            points, weights = bm_1_beta._component_nodes(mean, variance, 8)
            self.assertTrue(np.all(np.isfinite(points)))
            self.assertTrue(np.all(np.isfinite(weights)))
            self.assertAlmostEqual(float(weights.sum()),1.0,places=12)
            self.assertAlmostEqual(float(weights @ points),mean,places=11)
            self.assertLess(abs(float(weights @ np.square(points - mean))-variance),1e-11)

    def test_concentrated_beta_remains_admissible_for_sustained_price_extremes(self):
        """Old raw Gauss-Jacobi weights overflowed within a few hours here."""
        data = {"B":np.full(14,0.25), "I":np.full(14,0.25)}
        p = self.parameters()
        solver = bm_1_beta(components=8,quadrature=8,reconstruction_quadrature=48,max_step=0.25)
        for price in (0.01,0.99):
            data["U"] = np.full(14,price)
            result = solver.simulate(data,p,0.5,observations=None)
            self.assertTrue(np.all(np.isfinite(result["mean"])))
            self.assertTrue(all(abs(float(w.sum())-1.0)<1e-12 for w in result["weights"]))
            fit = solver.simulate(data,p,0.5,observations=data["I"])
            self.assertTrue(np.isfinite(fit["nll"]))

    def test_moment_preserving_equal_mass_reconstruction(self):
        method = bm_1_beta(components=4, quadrature=6, reconstruction_quadrature=12, max_step=0.25)
        mixture = (np.array([0.5,0.5]),np.array([0.25,0.75]),np.array([0.01,0.02]))
        w, mu, var = method._reconstruct(mixture)
        self.assertAlmostEqual(float(w.sum()),1.0,places=12)
        self.assertTrue(np.all(var >= 0.0))
        self.assertAlmostEqual(float(w @ mu), 0.5, places=9)
        original_second = float(np.sum(mixture[0] * (mixture[2]+mixture[1]**2)))
        self.assertAlmostEqual(float(w @ (var+mu**2)),original_second,places=9)

    def test_stochastic_open_forecast_and_historical_bayes_likelihood(self):
        solver = bm_1_beta(components=4,quadrature=4,reconstruction_quadrature=8,max_step=0.25)
        data, p = self.data(), self.parameters()
        open_result = solver.simulate(data,p,0.45)
        historical = solver.simulate(data,p,0.45,observations=data["I"])
        self.assertEqual(open_result["mean"].shape,(4,))
        self.assertTrue(np.all(np.isfinite(open_result["mean"])))
        self.assertTrue(np.isfinite(historical["nll"]))
        self.assertTrue(all(abs(float(m.sum())-1.0)<1e-9 for m in historical["weights"]))
        intervals = solver.intervals(open_result,p["sigma_y"])
        self.assertTrue(np.all(intervals["lower_95"] < intervals["upper_95"]))
        self.assertTrue(np.all(intervals["lower_68"] < intervals["upper_68"]))

    def test_exact_initial_observation_likelihood(self):
        solver = bm_1_beta(components=2,quadrature=2,reconstruction_quadrature=4,max_step=0.5)
        p = self.parameters()
        data = {"B":np.array([0.3]),"U":np.array([0.5]),"I":np.array([0.3])}
        result = solver.simulate(data,p,0.5,observations=data["I"])
        self.assertAlmostEqual(result["nll"],float(np.log(p["sigma_y"])+0.5*np.log(2*np.pi)),places=10)

    def test_fv_reference_is_independent_and_normalized(self):
        reference = bm_1_fv_reference(cells=61,max_step=0.25)
        data, p = self.data(), self.parameters()
        data = {name:value[:3] for name,value in data.items()}
        forecast = reference.simulate(data,p,0.45)
        fit = reference.simulate(data,p,0.45,observations=data["I"])
        self.assertTrue(np.isfinite(fit["nll"]))
        self.assertTrue(np.all(np.isfinite(forecast["mean"])))
        self.assertTrue(all(abs(float(m.sum())-1.0)<1e-7 for m in fit["weights"]))
        self.assertLess(abs(forecast["mean"][0]-bm_1_beta._demand(
            np.array([0.45]),data["B"][0],data["U"][0],p)[0]),0.01)

    def test_arbitrary_backlog_in_deterministic_warm_start(self):
        fit = bm_1_beta_params()
        p = self.parameters()
        a, b = self.data(), self.data()
        b = {key: value.copy() for key,value in b.items()}
        b["U"] = np.array([0.4,0.3,0.5,0.6])
        a["I"] = fit._deterministic(a,p,0.4,2)
        b["I"] = fit._deterministic(b,p,0.6,2)
        result = fit.fit_deterministic([a,b],cycles=2,max_nfev=10,state_maxiter=20,substeps=2)
        self.assertEqual(len(result["initial_states"]),2)
        self.assertTrue(np.isfinite(result["mse"]))
        self.assertGreaterEqual(result["seconds"],0.0)

    def test_weekly_history_and_target_do_not_overlap(self):
        origin = pd.Timestamp("2022-03-21 06:00")
        dates = historical_origins(origin,3)
        self.assertEqual(dates, [origin-pd.Timedelta(weeks=w) for w in (3,2,1)])
        idx = pd.date_range("2022-02-01",periods=60*24,freq="h")
        frame = pd.DataFrame({"I":0.3,"U":0.5,"Temp":8.0},index=idx)
        selected = historical_training(frame,dates,12)
        self.assertEqual(len(selected),36)
        self.assertLess(selected.index.max(),origin)
        checks.settings(3,12,5,2,10)
        with self.assertRaises(ValueError): checks.settings(3,169,5,2,10)

    def test_metrics_and_timing(self):
        metrics = score(np.array([1.,2.,3.]),np.array([2.,2.,2.]))
        self.assertAlmostEqual(metrics["bias"],0.0)
        seconds = {}
        self.assertEqual(timed(seconds,"fit",lambda:17),17)
        self.assertGreaterEqual(seconds["fit"],0.0)


if __name__ == "__main__":
    unittest.main()
