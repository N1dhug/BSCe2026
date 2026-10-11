"""WM4 MPC diffusion regression tests: P0 is included during sigma estimation."""

import unittest

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.predict_all_days_mpc import _fit_stochastic_variance
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
        J = model.sensitivity({"r": r})["input"][sample]
        kernel = model.process_variance_kernel({"r": r}, sample)
        kernel_response = kernel @ np.ones(model.t_steps)

        # Construct sample noise orthogonal to the two historical state
        # deviations. This makes the variance decomposition exactly additive.
        design = np.column_stack([np.ones(len(init)), init - np.mean(init, axis=0)])
        noise = np.array([0.3, -0.9, 1.5, -0.8, 0.2], dtype=float)
        noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]
        noise /= np.std(noise, ddof=1)

        sensitivity = model.sensitivity({"r": r})["input"]
        datasets, fitted_states = [], []
        for i, x in enumerate(init):
            center = 0.4 + 0.01 * i
            model_I = center + sensitivity @ x
            observed = model_I.copy()
            observed[sample] += noise[i] * np.sqrt(q * kernel_response)
            datasets.append({"I": observed})
            fitted_states.append({"I": model_I})

        return model, {"init_cond": init, "r": r, "deterministic_states": fitted_states,
                       "deterministic_state": fitted_states[0]}, datasets

    def test_historical_p0_subtracts_full_F_R_covariance(self):
        q_true = 0.07
        model, fit, datasets = self.setup_history(q_true)
        estimated = _fit_stochastic_variance(model, fit, datasets, 2, "historical_p0")
        prior_cov = np.cov(fit["init_cond"], rowvar=False, ddof=1)
        J = model.sensitivity({"r": fit["r"]})["input"][2 * np.arange(24)]
        prior_variance = np.einsum("ni,ij,nj->n", J, prior_cov, J)
        self.assertTrue(np.allclose(estimated["training_initial_variance"], prior_variance))
        self.assertTrue(np.allclose(estimated["initial_variance"], prior_variance))
        self.assertAlmostEqual(estimated["q"], q_true, delta=1e-4)
        self.assertTrue(np.allclose(
            estimated["training_centered_residual_variance"] - prior_variance,
            estimated["training_fit_residual_variance"], atol=1e-12))

    def test_legacy_mode_preserves_zero_prior_fit(self):
        model, fit, datasets = self.setup_history(0.05)
        legacy = _fit_stochastic_variance(model, fit, datasets, 2, "zero_p0")
        self.assertTrue(np.allclose(legacy["initial_variance"], 0.0))
        self.assertTrue(np.allclose(legacy["training_residual_variance"],
                                    legacy["training_fit_residual_variance"]))
        self.assertEqual(legacy["sigma_fit_mode"], "zero_p0")

    def test_new_comparison_rejects_old_saved_wm4_intervals(self):
        old = pd.DataFrame([{
            "origin": "2022-03-04 08:00:00", "target_time": "2022-03-04 09:00:00",
            "lead_hours": 1, "I_mpc": 0.5, "lower_68": 0.4, "upper_68": 0.6,
            "lower_95": 0.3, "upper_95": 0.7,
        }])
        with self.assertRaisesRegex(ValueError, "predate historical-P0"):
            checks.wm4_saved(old, sigma_fit_mode="historical_p0")
        recovered = checks.wm4_saved(old, sigma_fit_mode="zero_p0")
        self.assertEqual(recovered.iloc[0]["sigma_fit_mode"], "zero_p0")


if __name__ == "__main__":
    unittest.main()
