"""F2-G tests for Beta-mixture conditioning and sensitivity diagnostics."""

import unittest

import numpy as np

from src.exe.run_model.bm_1.beta_conditioning_reference import (
    _positive_mass_preserving_perturbation,
    _simplex_tangent_condition,
    run_conditioning_study,
)


class ConditioningTheoryTests(unittest.TestCase):
    def test_simplex_tangent_condition_matches_one_dimensional_free_direction(self):
        hessian = np.diag([1.0, 4.0])
        coefficients = np.array([0.5, 0.5])
        condition, free_size, tangent_dimension = _simplex_tangent_condition(
            hessian,
            coefficients,
            active_tol=1.0e-12,
        )
        self.assertAlmostEqual(condition, 1.0)
        self.assertEqual(free_size, 2)
        self.assertEqual(tangent_dimension, 1)

    def test_positive_perturbation_preserves_mass_and_nonnegativity(self):
        density = np.array([0.5, 1.0, 1.5, 1.0])
        dx = 0.25
        density /= np.sum(density) * dx
        direction = np.array([-2.0, -0.5, 0.7, 1.8])

        perturbed = _positive_mass_preserving_perturbation(
            density,
            dx,
            1.0e-6,
            direction,
        )
        self.assertGreaterEqual(np.min(perturbed), 0.0)
        self.assertAlmostEqual(np.sum(perturbed) * dx, 1.0, places=14)
        self.assertGreater(np.linalg.norm(perturbed - density), 0.0)


class ConditioningIntegrationTests(unittest.TestCase):
    def test_small_study_returns_finite_sensitivity_diagnostics(self):
        result = run_conditioning_study(
            degrees=(10,),
            epsilons=(1.0e-8,),
            n_directions=1,
            seed=44,
            x_steps=41,
            max_step=1.0 / 48.0,
            t_eval=np.array([0.0, 1.0]),
            cfl=0.35,
        )

        item = result["conditioning"][10]
        self.assertTrue(np.isfinite(item["matrix_condition_number"]))
        self.assertTrue(np.isfinite(item["hessian_condition_number"]))
        self.assertGreaterEqual(item["matrix_condition_number"], 1.0)
        self.assertGreaterEqual(item["hessian_condition_number"], 1.0)
        self.assertTrue(
            np.all(np.isfinite(item["static_local_condition_numbers"]))
        )
        self.assertTrue(
            np.all(np.isfinite(item["dynamic_local_condition_numbers"]))
        )

        alt = item["alternative_start"]
        self.assertEqual(alt["failures"], 0)
        self.assertTrue(np.isfinite(alt["maximum_coefficient_l1"]))
        self.assertTrue(np.isfinite(alt["maximum_density_l2"]))

        pert = item["perturbations"][1.0e-8]
        self.assertEqual(pert["failures"], 0)
        self.assertGreater(pert["maximum_input_l2"], 0.0)
        self.assertTrue(np.isfinite(pert["maximum_coefficient_l1"]))
        self.assertTrue(np.isfinite(pert["maximum_projected_density_l2"]))
        self.assertTrue(np.isfinite(pert["maximum_density_amplification"]))
        self.assertTrue(np.isfinite(pert["maximum_coefficient_amplification"]))


if __name__ == "__main__":
    unittest.main()
