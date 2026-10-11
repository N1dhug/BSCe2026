"""WM4 MPC P0 regression: baseline-origin uncertainty enters sigma QP and forecast."""

import unittest

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.predict_all_days_mpc import (
    _fit_stochastic_variance, _mpc_initial_covariance, _stochastic_forecast_variance,
)
from src.model.wm_4 import wm_4
from src.validation.exe.run_model.compare_mpc_predictions import compare_mpc_validation as checks


class HistoricalP0SigmaTests(unittest.TestCase):
    def setup_history(self, q):
        steps_per_hour = 2
        model = wm_4((0.0, 1.0), 49, lambda t: 0.0)
        r = np.array([-2.0, 0.1, -1.0], dtype=float)
        init = np.array([[-0.3, 0.1], [-0.1, -0.1], [0.0, 0.0],
                         [0.1, -0.1], [0.3, 0.1]], dtype=float)
        sample = 2 * np.arange(24, dtype=int)
        kernel = model.process_variance_kernel({"r": r}, sample)
        kernel_response = kernel @ np.ones(model.t_steps)

        # Orthogonal synthetic process residuals isolate the known q.
        design = np.column_stack([np.ones(len(init)), init - np.mean(init, axis=0)])
        noise = np.array([0.3, -0.9, 1.5, -0.8, 0.2], dtype=float)
        noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]
        noise /= np.std(noise, ddof=1)

        sensitivity = model.sensitivity({"r": r})["input"]
        datasets, fitted_states = [], []
        for i, x in enumerate(init):
            baseline = 0.4 + 0.01 * i
            model_I = baseline + sensitivity @ x
            observed = model_I.copy()
            observed[sample] += noise[i] * np.sqrt(q * kernel_response)
            datasets.append({"I": observed, "O": np.full_like(observed, baseline)})
            fitted_states.append({"I": model_I})

        return model, {"init_cond": init, "r": r, "deterministic_states": fitted_states,
                       "deterministic_state": fitted_states[0]}, datasets

    def test_historical_sigma_fit_uses_same_diagonal_mpc_p0(self):
        q_true = 0.07
        model, fit, datasets = self.setup_history(q_true)
        result = _fit_stochastic_variance(model, fit, datasets, 2, "historical_p0")
        f_var = np.var(fit["init_cond"][:, 0], ddof=1)
        r_var = np.var([float(d["I"][0] - d["O"][0]) for d in datasets], ddof=1)
        self.assertAlmostEqual(result["F_variance"], f_var)
        self.assertAlmostEqual(result["R_variance"], r_var)
        P0 = np.diag([f_var, r_var])
        J = model.sensitivity({"r": fit["r"]})["input"][2 * np.arange(24)]
        prior_variance = np.einsum("ni,ij,nj->n", J, P0, J)
        self.assertTrue(np.allclose(result["mpc_initial_covariance"], P0))
        self.assertTrue(np.allclose(result["training_initial_variance"], prior_variance))
        self.assertTrue(np.allclose(result["initial_variance"], prior_variance))
        self.assertAlmostEqual(result["q"], q_true, delta=1e-4)
        self.assertTrue(np.allclose(
            result["training_centered_residual_variance"] - prior_variance,
            result["training_fit_residual_variance"], atol=1e-12))

    def test_initial_R_variance_propagates_and_can_be_ablated(self):
        model, fit, datasets = self.setup_history(0.05)
        full = _fit_stochastic_variance(model, fit, datasets, 2, "historical_p0", "baseline_residual")
        no_rate = _fit_stochastic_variance(model, fit, datasets, 2, "historical_p0", "zero")
        self.assertEqual(no_rate["R_variance"], 0.0)
        self.assertGreater(full["R_variance"], 0.0)
        self.assertAlmostEqual(no_rate["training_initial_variance"][0], 0.0)
        self.assertAlmostEqual(full["training_initial_variance"][0], full["R_variance"])
        # Compare forecast variance while holding sigma fixed, isolating P0's effect.
        no_rate = {**no_rate, "q": full["q"]}
        full_prediction = _stochastic_forecast_variance(model, {"r": fit["r"]},
            fit, full, 8, 2)
        ablated_prediction = _stochastic_forecast_variance(model, {"r": fit["r"]},
            fit, no_rate, 8, 2)
        self.assertTrue(np.all(full_prediction["initial_variance"] >
                               ablated_prediction["initial_variance"]))
        self.assertTrue(np.all(full_prediction["variance"] >
                               ablated_prediction["variance"]))

    def test_legacy_sigma_mode_preserves_zero_P0_in_fit(self):
        model, fit, datasets = self.setup_history(0.05)
        legacy = _fit_stochastic_variance(model, fit, datasets, 2, "zero_p0", "zero")
        self.assertTrue(np.allclose(legacy["initial_variance"], 0.0))
        self.assertTrue(np.allclose(legacy["training_residual_variance"],
                                    legacy["training_fit_residual_variance"]))
        self.assertEqual(legacy["sigma_fit_mode"], "zero_p0")
        self.assertEqual(legacy["r0_variance_mode"], "zero")

    def test_saved_wm4_comparison_requires_correct_variance_mode(self):
        old = pd.DataFrame([{
            "origin": "2022-03-04 08:00:00", "target_time": "2022-03-04 09:00:00",
            "lead_hours": 1, "I_mpc": 0.5, "lower_68": 0.4, "upper_68": 0.6,
            "lower_95": 0.3, "upper_95": 0.7,
        }])
        with self.assertRaisesRegex(ValueError, "predate historical-P0"):
            checks.wm4_saved(old, sigma_fit_mode="historical_p0")
        with self.assertRaisesRegex(ValueError, "predate baseline-derived"):
            checks.wm4_saved(old, sigma_fit_mode="zero_p0", r0_variance_mode="baseline_residual")
        recovered = checks.wm4_saved(old, sigma_fit_mode="zero_p0", r0_variance_mode="zero")
        self.assertEqual(recovered.iloc[0]["sigma_fit_mode"], "zero_p0")
        new = old.copy()
        new["sigma_fit_mode"] = "historical_p0"
        new["r0_variance_mode"] = "baseline_residual"
        loaded = checks.wm4_saved(new)
        self.assertEqual(loaded.iloc[0]["r0_variance_mode"], "baseline_residual")


if __name__ == "__main__":
    unittest.main()
