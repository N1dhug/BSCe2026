"""WM4 24-day six-hour re-estimation with zero initial-state covariance."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.model.wm_4 import wm_4
from src.exe.run_model.wm_4.predict_all_days_mpc import (
    _archive_mpc_calibration, _fit_sigma_rolling_qp, _fit_stochastic_variance,
    _rolling_sigma_training, _stochastic_forecast_variance,
    ROLLING_SIGMA_DAYS, ROLLING_SIGMA_HORIZONS,
)
from src.exe.run_model.wm_4.analyze_mpc_intervals import load_forecasts
from src.exe.run_model.wm_4.predict_all_days_mpc_no_p0 import MODE
from tests.test_wm4_mpc_intervals import MPCIntervalAnalysisTests


class WM4NoP0RefittingTests(unittest.TestCase):
    @staticmethod
    def sample(days=24):
        origin = pd.Timestamp("2022-08-01 14:00")
        model = wm_4((0., 1.), 49, lambda t: 0.)
        r = np.array([-2., -0.1, -3.], dtype=float)
        h = 2 * np.arange(1, 7)
        kernel = model.process_variance_kernel({"r": r}, h) @ np.ones(model.t_steps)
        J = model.sensitivity({"r": r})["input"][h]
        base_p0 = np.diag([0.005, 0.008])
        B = np.einsum("ni,ij,nj->n", J, base_p0, J)
        z = np.arange(days, dtype=float)
        z = (z - z.mean()) / z.std(ddof=1)
        archive = {}
        for i in range(days):
            when = origin - pd.Timedelta(days=i + 1)
            err = z[i] * np.sqrt(B + 0.025 * kernel)
            archive[when] = np.column_stack((err, B, kernel))
        initial = np.array([[0.04, 0.07], [-0.02, 0.09], [0.12, 0.],
            [0.17, -0.03], [0.02, 0.15]])
        fit = {"r": r, "prediction_origin": origin, "init_cond": initial,
            "training_origins": [origin - pd.Timedelta(weeks=i) for i in range(1, 6)],
            "deterministic_states": [{"I": np.full(49, 0.5)} for _ in range(5)]}
        dataset = [{"I": np.full(49, 0.5), "O": np.full(49, 0.5)} for _ in range(5)]
        return model, fit, dataset, archive, B

    def test_six_hour_24_day_sigma_refits_without_subtracting_any_p0(self):
        model, fit, dataset, archive, _ = self.sample()
        usual = _fit_stochastic_variance(model, fit, dataset, 2,
            sigma_fit_mode="mpc_replay_24d_6h", mpc_archive=archive, max_horizon=6)
        zero = _fit_stochastic_variance(model, fit, dataset, 2,
            sigma_fit_mode=MODE, mpc_archive=archive, max_horizon=6)
        history = _rolling_sigma_training(archive, fit["prediction_origin"],
            days=24, horizon=6)
        expected = {"values": history["values"].copy()}
        expected["values"][:, :, 1] = 0.
        ref, observed, initial, kernel = _fit_sigma_rolling_qp(model, expected, horizon=6)
        self.assertEqual(zero["calibration_mpc_count"], 24)
        self.assertEqual(zero["calibration_horizon"], 6)
        self.assertEqual(zero["sigma_rolling_days"], 24)
        self.assertEqual(ROLLING_SIGMA_DAYS[MODE], 24)
        self.assertEqual(ROLLING_SIGMA_HORIZONS[MODE], 6)
        self.assertTrue(zero["sigma_rolling_complete"])
        self.assertTrue(np.array_equal(zero["training_initial_variance"], np.zeros(6)))
        self.assertTrue(np.allclose(zero["mpc_initial_covariance"], 0.))
        self.assertTrue(np.allclose(initial, 0.))
        self.assertAlmostEqual(zero["q"], ref["q"], delta=1e-8)
        self.assertGreater(zero["q"], usual["q"])
        self.assertTrue(np.allclose(zero["training_residual_variance"], observed))
        # Original F/R reconstruction statistics are retained for the
        # deterministic initial-state bias correction, not propagated.
        self.assertTrue(np.max(np.abs(zero["covariance"])) > 0.)

    def test_forward_six_hour_variance_has_only_refitted_wiener(self):
        model, fit, dataset, archive, _ = self.sample()
        diffusion = _fit_stochastic_variance(model, fit, dataset, 2,
            sigma_fit_mode=MODE, mpc_archive=archive, max_horizon=6)
        forward = _stochastic_forecast_variance(model, {"r": fit["r"]}, fit,
            diffusion, max_horizon=6, steps_per_hour=2)
        self.assertTrue(np.allclose(forward["initial_variance"], 0.))
        self.assertTrue(np.allclose(forward["initial_F_component"], 0.))
        self.assertTrue(np.allclose(forward["initial_R_component"], 0.))
        self.assertTrue(np.allclose(forward["initial_FR_component"], 0.))
        self.assertTrue(np.allclose(forward["variance"], forward["process_variance"]))
        self.assertTrue(np.allclose(forward["variance"],
            diffusion["q"] * forward["process_kernel_unit"]))
        new_archive = {}
        prepared = {"I": np.full(49, 0.5)}
        state = {"r": fit["r"], "I": np.full(49, 0.5)}
        _archive_mpc_calibration(new_archive, fit["prediction_origin"],
            model, state, prepared, fit, diffusion, steps_per_hour=2, horizon=6)
        self.assertEqual(new_archive[fit["prediction_origin"]].shape, (6, 3))
        self.assertTrue(np.allclose(new_archive[fit["prediction_origin"]][:, 1], 0.))

    def test_bootstrap_does_not_inject_initial_state_uncertainty(self):
        model, fit, dataset, archive, _ = self.sample()
        result = _fit_stochastic_variance(model, fit, dataset, 2,
            sigma_fit_mode=MODE, mpc_archive={}, max_horizon=6)
        self.assertTrue(result["sigma_rolling_bootstrap"])
        self.assertTrue(np.allclose(result["training_initial_variance"], 0.))
        self.assertTrue(np.allclose(result["mpc_initial_covariance"], 0.))

    def test_quantile_input_loader_checks_no_p0_without_rejecting_reconstruction(self):
        data = MPCIntervalAnalysisTests.simulated_forecasts(1)
        data = data.loc[data["lead_hours"] <= 6].copy()
        data["sigma_fit_mode"] = MODE
        data["calibration_horizon"] = 6
        for key in ("initial_variance", "initial_F_variance",
            "initial_R_variance", "initial_FR_covariance", "initial_F_component_variance",
            "initial_R_component_variance", "initial_FR_component_variance"):
            data[key] = 0.
        data["predicted_variance"] = data["process_variance"]
        std = np.sqrt(data["predicted_variance"].to_numpy())
        for confidence, factor in ((68, 1.), (95, 1.96)):
            data[f"lower_{confidence}"] = data["I_mpc"] - factor * std
            data[f"upper_{confidence}"] = data["I_mpc"] + factor * std
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "forecasts.csv"
            data.to_csv(csv, index=False)
            actual, dropped, mode = load_forecasts(csv, horizon=6,
                expected_sigma_fit_mode=MODE)
            self.assertEqual(mode, MODE)
            self.assertFalse(dropped)
            self.assertTrue((actual["initial_variance"] == 0).all())
            data.loc[data.index[0], "initial_F_variance"] = 0.1
            data.to_csv(csv, index=False)
            with self.assertRaises(ValueError):
                load_forecasts(csv, horizon=6, expected_sigma_fit_mode=MODE)


if __name__ == "__main__": unittest.main()
