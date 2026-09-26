"""F2-F tests for dynamic Beta-mixture evolution."""

import unittest

import numpy as np

from src.exe.run_model.bm_1.beta_dynamic_reference import (
    _evolve_projected_density,
    run_dynamic_mixture_study,
)
from src.exe.run_model.bm_1.f2_reference_case import build_reference_case


class DynamicMixturePropagationTests(unittest.TestCase):
    def test_zero_operator_preserves_an_existing_mixture(self):
        model, _ = build_reference_case()
        edges, x, dx = model.fokker_planc_reference.grid(41)
        del edges

        degree = 10
        basis = model.fokker_planc._normalise_basis_discretely(
            model.fokker_planc.beta_basis(x, degree),
            dx,
        )
        coefficients = np.zeros(degree + 1)
        coefficients[4] = 0.35
        coefficients[5] = 0.65

        result = _evolve_projected_density(
            method=model.fokker_planc,
            t_eval=np.array([0.0, 0.5, 1.0]),
            breakpoints=np.array([0.37]),
            x=x,
            dx=dx,
            basis=basis,
            initial_coefficients=coefficients,
            drift=lambda time, state: np.zeros_like(state),
            diffusion=lambda time, state: np.zeros_like(state),
            max_step=0.1,
            cfl=0.4,
        )

        expected_density = coefficients @ basis
        np.testing.assert_allclose(
            result["density"],
            np.tile(expected_density, (3, 1)),
            atol=2.0e-10,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            result["coefficients"],
            np.tile(coefficients, (3, 1)),
            atol=2.0e-10,
            rtol=0.0,
        )
        diagnostics = result["diagnostics"]
        self.assertFalse(diagnostics["used_clipping"])
        self.assertFalse(diagnostics["used_renormalization"])
        self.assertLess(
            diagnostics["maximum_abs_raw_prediction_mass_error"],
            1.0e-12,
        )


class DynamicMixtureReferenceTests(unittest.TestCase):
    def test_small_dynamic_study_preserves_probability_and_separates_errors(self):
        result = run_dynamic_mixture_study(
            degrees=(20,),
            x_steps=51,
            max_step=1.0 / 96.0,
            t_eval=np.array([0.0, 0.5, 1.0]),
            cfl=0.35,
        )

        item = result["dynamic"][20]
        diagnostics = item["diagnostics"]

        np.testing.assert_allclose(
            np.sum(item["cell_mass"], axis=1),
            1.0,
            atol=1.0e-9,
            rtol=0.0,
        )
        self.assertGreater(diagnostics["n_steps"], 0)
        self.assertTrue(np.isfinite(diagnostics["hessian_condition_number"]))
        self.assertGreaterEqual(diagnostics["hessian_condition_number"], 1.0)
        self.assertGreaterEqual(
            diagnostics["minimum_raw_prediction"],
            -1.0e-10,
        )
        self.assertLess(
            diagnostics["maximum_abs_raw_prediction_mass_error"],
            1.0e-9,
        )
        self.assertLess(
            diagnostics["maximum_abs_projected_mass_error"],
            1.0e-9,
        )
        self.assertLess(
            diagnostics["maximum_abs_coefficient_sum_error"],
            1.0e-9,
        )
        self.assertGreaterEqual(diagnostics["minimum_coefficient"], -1.0e-9)
        self.assertFalse(diagnostics["used_clipping"])
        self.assertFalse(diagnostics["used_renormalization"])

        # F2-F starts from exactly the same degree-M projection used by F2-E.
        self.assertAlmostEqual(item["dynamic_gap"]["l2"][0], 0.0, places=12)
        self.assertAlmostEqual(
            item["total_error"]["l2"][0],
            result["projection"][20]["l2"][0],
            places=12,
        )

        self.assertTrue(np.all(np.isfinite(item["total_error"]["l2"])))
        self.assertTrue(np.all(np.isfinite(item["dynamic_gap"]["l2"])))
        self.assertTrue(np.all(np.isfinite(item["mean"])))
        self.assertTrue(np.all(np.isfinite(item["variance"])))
        self.assertTrue(np.all(np.isfinite(item["expected_demand"])))


if __name__ == "__main__":
    unittest.main()
