"""WM4 MPC P0 regression: oracle rate reconstruction and causal diffusion calibration."""

import unittest

import numpy as np
import pandas as pd

from src.exe.run_model.wm_4.predict_all_days_mpc import (
    _archive_mpc_origin, _archive_reconstruction_origin, _fit_stochastic_variance, _mpc_initial_covariance,
    _mpc_origin, _stochastic_forecast_variance,
)
from unittest.mock import patch
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
        result = _fit_stochastic_variance(model, fit, datasets, 2, "historical_p0", "baseline_residual")
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

    def test_historical_mpc_state_errors_replace_oracle_bootstrap_and_fit_full_P0(self):
        model, fit, datasets = self.setup_history(0.05)
        origin = pd.Timestamp("2022-07-12 08:00")
        training = [origin - pd.Timedelta(weeks=k) for k in range(1, 6)]
        fit["training_origins"] = training
        fit["prediction_origin"] = origin
        boot = _mpc_initial_covariance(fit, datasets)
        self.assertEqual(boot["F_calibration_oracle_count"], 5)
        self.assertEqual(boot["F_calibration_mpc_count"], 0)
        self.assertAlmostEqual(boot["F_bias_correction"], 0.0)

        eF = np.array([-0.12, -0.04, 0.03, 0.08, 0.16])
        eR = np.array([-0.06, -0.03, 0.01, 0.04, 0.09])
        archive = {}
        for i, stamp in enumerate(training):
            _archive_reconstruction_origin(archive, stamp, {
                "origin": stamp, "F_reconstruction_error_origin": float(eF[i]),
                "R_reconstruction_error_origin": float(eR[i])})
        cal = _mpc_initial_covariance(fit, datasets, state_archive=archive)
        expected = np.cov(np.column_stack([eF, eR]), rowvar=False, ddof=1)
        self.assertEqual(cal["F_calibration_mpc_count"], 5)
        self.assertEqual(cal["F_calibration_oracle_count"], 0)
        self.assertTrue(np.allclose(cal["covariance"], expected))
        self.assertAlmostEqual(cal["F_bias_correction"], np.mean(eF))
        self.assertAlmostEqual(cal["R_bias_correction"], np.mean(eR))
        self.assertGreater(cal["FR_covariance"], 0.0)
        self.assertGreater(cal["F_variance"], 0.0)
        self.assertGreater(cal["R_variance"], 0.0)
        diffusion = _fit_stochastic_variance(model, fit, datasets, 2,
            state_archive=archive, max_horizon=8)
        self.assertTrue(np.allclose(diffusion["mpc_initial_covariance"], expected))
        prediction = _stochastic_forecast_variance(model, {"r": fit["r"]},
            fit, diffusion, 8, 2)
        J = model.sensitivity({"r": fit["r"]})["input"][2 * np.arange(1, 9)]
        propagated = np.einsum("ni,ij,nj->n", J, expected, J)
        self.assertTrue(np.allclose(prediction["initial_variance"], propagated))
        self.assertTrue(np.allclose(prediction["initial_variance"],
            prediction["initial_F_component"] + prediction["initial_R_component"]
            + prediction["initial_FR_component"]))
        self.assertTrue(np.any(abs(prediction["initial_FR_component"]) > 1e-10))

        archive.pop(training[0])
        partial = _mpc_initial_covariance(fit, datasets, state_archive=archive)
        self.assertEqual(partial["F_calibration_mpc_count"], 4)
        self.assertEqual(partial["F_calibration_oracle_count"], 1)

    def test_f_oracle_spread_ablation_retains_diagonal_P0(self):
        model, fit, datasets = self.setup_history(0.05)
        origin = pd.Timestamp("2022-07-12 08:00")
        fit["training_origins"] = [origin - pd.Timedelta(weeks=k) for k in range(1, 6)]
        archive = {stamp: np.array([0.03, 0.08])
            for stamp in fit["training_origins"]}
        old = _mpc_initial_covariance(fit, datasets, f0_variance_mode="oracle_spread",
            state_archive=archive)
        self.assertAlmostEqual(old["F_variance"], np.var(fit["init_cond"][:, 0], ddof=1))
        self.assertEqual(old["F_bias_correction"], 0.0)
        self.assertEqual(old["FR_covariance"], 0.0)
        self.assertEqual(old["F_calibration_mpc_count"], 0)

    def test_incomplete_historical_origin_does_not_enter_state_calibration(self):
        model, fit, datasets = self.setup_history(0.05)
        origin = pd.Timestamp("2022-07-12 08:00")
        fit["prediction_origin"] = origin
        fit["training_origins"] = [origin - pd.Timedelta(weeks=k) for k in range(1, 6)]
        training = fit["training_origins"]
        archive = {training[0]: np.array([0.12, -0.07])}
        usable = _mpc_initial_covariance(fit, datasets, state_archive=archive)
        self.assertEqual(usable["F_calibration_mpc_count"], 1)
        fit["prediction_origin"] = training[0] + pd.Timedelta(hours=23)
        excluded = _mpc_initial_covariance(fit, datasets, state_archive=archive)
        self.assertEqual(excluded["F_calibration_mpc_count"], 0)

    def test_oracle_reconstruction_error_controls_R0_variance_and_bias(self):
        model, fit, datasets = self.setup_history(0.05)
        # Historical R rates vary widely, but observed-minus-oracle errors do not.
        errors = np.array([0.02, 0.04, -0.01, 0.03, -0.03])
        for i, data in enumerate(datasets):
            data["I"][0] = data["O"][0] + fit["init_cond"][i, 1] + errors[i]
        result = _mpc_initial_covariance(fit, datasets)
        self.assertEqual(result["r0_variance_mode"], "oracle_reconstruction")
        self.assertAlmostEqual(result["R_variance"], np.var(errors, ddof=1))
        self.assertAlmostEqual(result["R_reconstruction_bias"], np.mean(errors))
        self.assertAlmostEqual(result["R_bias_correction"], np.mean(errors))
        self.assertAlmostEqual(result["R_reconstruction_rmse"],
            np.sqrt(np.mean(errors ** 2)))
        self.assertEqual(result["R_reconstruction_samples"], len(errors))
        self.assertTrue(np.allclose(result["R_reconstruction_errors"], errors))
        self.assertGreater(result["baseline_residual_variance"], result["R_variance"])
        self.assertAlmostEqual(result["covariance"][0, 1], 0.0)
        old = _mpc_initial_covariance(fit, datasets, "baseline_residual")
        zero = _mpc_initial_covariance(fit, datasets, "zero")
        self.assertAlmostEqual(old["R_variance"], old["baseline_residual_variance"])
        self.assertEqual(old["R_bias_correction"], 0.0)
        self.assertEqual(zero["R_variance"], 0.0)
        self.assertEqual(zero["R_bias_correction"], 0.0)

    def test_zero_oracle_reconstruction_error_creates_no_rate_uncertainty(self):
        model, fit, datasets = self.setup_history(0.05)
        # At t=0 the synthetic observation agrees exactly with oracle R0.
        actual = _mpc_initial_covariance(fit, datasets)
        self.assertAlmostEqual(actual["R_variance"], 0.0, delta=1e-25)
        self.assertAlmostEqual(actual["R_bias_correction"], 0.0, delta=1e-12)
        self.assertGreater(actual["baseline_residual_variance"], 0.0)

    def test_mpc_origin_corrects_observed_R_center_before_forecast(self):
        model, fit, datasets = self.setup_history(0.05)
        correction = np.array([0.01, 0.03, -0.02, 0.04, -0.01])
        for i, dataset in enumerate(datasets):
            dataset["I"][0] = dataset["O"][0] + fit["init_cond"][i, 1] + correction[i]
        origin = pd.Timestamp("2022-06-01 12:00")
        fit["training_origins"] = [origin - pd.Timedelta(weeks=k) for k in range(1, 6)]
        fit["x_hat"] = np.mean(fit["init_cond"], axis=0)
        fit["state_iterations"] = 1
        fit["qp_loss"] = 0.0
        initial_R_observed = 0.23
        current = {"I": np.full(model.t_steps, 0.4 + initial_R_observed),
            "O": np.full(model.t_steps, 0.4), "U": np.zeros(model.t_steps),
            "PE": np.zeros(model.t_steps), "Time": np.linspace(0, 24, model.t_steps),
            "observation_mask": np.zeros(model.t_steps, dtype=bool),
            "transition_mask": np.ones(model.t_steps - 1, dtype=bool)}
        times = pd.date_range(origin, periods=model.t_steps, freq="30min")

        class FakeHelper:
            def __init__(self, model): self.model = model
            def _prepare_data(self, data): return data
            def optimize_prediction_initial_state(self, data, r, **kwargs):
                return {"init_cond": np.array([0.05, 0.19])}
            def _state_from_parameters(self, data, initial, r, full=True):
                sensitivity = self.model.sensitivity({"r": r})["state"]
                trajectory = sensitivity @ np.asarray(initial, dtype=float)
                return {"I": data["O"] + trajectory[:, 1], "O": data["O"],
                    "R": trajectory[:, 1], "F": trajectory[:, 0]}

        with patch("src.exe.run_model.wm_4.predict_all_days_mpc._fit_shifted_history",
                return_value=(fit, model, datasets)), patch(
                "src.exe.run_model.wm_4.predict_all_days_mpc._window_input",
                return_value=(current, times)), patch(
                "src.exe.run_model.wm_4.predict_all_days_mpc.optimized_qp_params",
                FakeHelper):
            rows, state = _mpc_origin(None, origin, 5, 2, None, 8)
        b = float(np.mean(correction))
        self.assertAlmostEqual(state["R_observed"], initial_R_observed)
        self.assertAlmostEqual(state["R_reconstruction_bias"], b)
        self.assertAlmostEqual(state["R_initialized"], initial_R_observed - b)
        self.assertAlmostEqual(state["mpc_P0_RR"], np.var(correction, ddof=1))
        self.assertAlmostEqual(rows[0]["R_bias_correction"], b)
        J = model.sensitivity({"r": fit["r"]})["input"][2]
        prediction = 0.4 + J @ np.array([fit["x_hat"][0], initial_R_observed - b])
        self.assertAlmostEqual(rows[0]["I_mpc"], prediction)

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

    def test_mpc_replay_bootstraps_then_replaces_oracle_errors(self):
        model, fit, datasets = self.setup_history(0.05)
        origins = [pd.Timestamp("2022-06-01 12:00") - pd.Timedelta(weeks=k)
                   for k in range(1, 6)]
        fit["training_origins"] = origins
        initial = _mpc_initial_covariance(fit, datasets)
        indices = 2 * np.arange(1, 9)
        J = model.sensitivity({"r": fit["r"]})["input"][indices]
        V0 = np.einsum("ni,ij,nj->n", J, initial["covariance"], J)
        K = model.process_variance_kernel({"r": fit["r"]}, indices) @ np.ones(model.t_steps)
        oracle = _fit_stochastic_variance(model, fit, datasets, 2, max_horizon=8)
        self.assertEqual(oracle["sigma_fit_mode"], "mpc_replay")
        self.assertEqual(oracle["calibration_mpc_count"], 0)
        self.assertEqual(oracle["calibration_oracle_count"], 5)

        # Populate five genuine MPC forecast records; for each lead their
        # sample errors have Var = propagated P0 + known q*K exactly.
        q_true = 0.15
        standardized = np.array([-2., -1., 0., 1., 2.])
        standardized /= np.std(standardized, ddof=1)
        archive = {}
        for i, origin in enumerate(origins):
            std = np.sqrt(V0 + q_true * K)
            rows = [{
                "lead_hours": h, "target_time": origin + pd.Timedelta(hours=h),
                "forecast_error": float(standardized[i] * std[h - 1]),
                "initial_variance": float(V0[h - 1]),
                "process_kernel_unit": float(K[h - 1]),
            } for h in range(1, 9)]
            _archive_mpc_origin(archive, origin, rows)
        updated = _fit_stochastic_variance(model, fit, datasets, 2, mpc_archive=archive)
        self.assertEqual(updated["calibration_mpc_count"], 5)
        self.assertEqual(updated["calibration_oracle_count"], 0)
        self.assertEqual(updated["calibration_origin_types"], ["mpc"] * 5)
        self.assertAlmostEqual(updated["q"], q_true, delta=1e-4)
        self.assertTrue(np.allclose(updated["initial_variance"], V0, atol=1e-10))

        # Losing a single weekly record restores exactly one oracle bootstrap.
        archive.pop(origins[0])
        partial = _fit_stochastic_variance(model, fit, datasets, 2, mpc_archive=archive)
        self.assertEqual(partial["calibration_mpc_count"], 4)
        self.assertEqual(partial["calibration_oracle_count"], 1)
        self.assertEqual(partial["calibration_origin_types"].count("oracle_bootstrap"), 1)

    def test_mpc_replay_does_not_use_nonhistorical_origin(self):
        model, fit, datasets = self.setup_history(0.05)
        fit["training_origins"] = [
            pd.Timestamp("2022-06-01") - pd.Timedelta(weeks=k) for k in range(1, 6)]
        archive = {pd.Timestamp("2022-06-02"): np.ones((8, 3))}
        fit_result = _fit_stochastic_variance(model, fit, datasets, 2, mpc_archive=archive)
        self.assertEqual(fit_result["calibration_mpc_count"], 0)
        self.assertEqual(fit_result["calibration_oracle_count"], 5)

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
        recovered = checks.wm4_saved(old, sigma_fit_mode="zero_p0", r0_variance_mode="zero",
            f0_variance_mode="oracle_spread")
        self.assertEqual(recovered.iloc[0]["sigma_fit_mode"], "zero_p0")
        new = old.copy()
        new["sigma_fit_mode"] = "historical_p0"
        new["r0_variance_mode"] = "baseline_residual"
        loaded = checks.wm4_saved(new, sigma_fit_mode="historical_p0",
            f0_variance_mode="oracle_spread")
        self.assertEqual(loaded.iloc[0]["r0_variance_mode"], "baseline_residual")
        with self.assertRaisesRegex(ValueError, "sigma_fit_mode"):
            checks.wm4_saved(new, sigma_fit_mode="mpc_replay", f0_variance_mode="oracle_spread")
        new["sigma_fit_mode"] = "mpc_replay"
        with self.assertRaisesRegex(ValueError, "MPC-replay forecasts missing"):
            checks.wm4_saved(new, sigma_fit_mode="mpc_replay", f0_variance_mode="oracle_spread")


if __name__ == "__main__":
    unittest.main()
