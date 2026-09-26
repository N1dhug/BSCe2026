"""F2-E tests for static Beta-mixture projection against pure finite volume."""

import unittest

import numpy as np

from src.diagnostics.bm_1 import (
    beta_mixture_variance_floor,
    cell_mass_l1,
    density_l2,
)
from src.exe.run_model.bm_1.beta_projection_reference import run_projection_study
from src.exe.run_model.bm_1.f2_reference_case import (
    INITIAL_ALPHA,
    INITIAL_BETA,
    beta_initial_cell_mass,
    build_reference_case,
)


class BetaProjectionTheoryTests(unittest.TestCase):
    def test_variance_floor_matches_beta_basis_formula(self):
        mean = 0.5
        expected_initial_variance = (
            INITIAL_ALPHA
            * INITIAL_BETA
            / (
                (INITIAL_ALPHA + INITIAL_BETA) ** 2
                * (INITIAL_ALPHA + INITIAL_BETA + 1.0)
            )
        )
        floor_20 = beta_mixture_variance_floor(mean, 20)
        floor_38 = beta_mixture_variance_floor(mean, 38)

        self.assertAlmostEqual(floor_20, 0.25 / 23.0)
        self.assertGreater(floor_20, expected_initial_variance)
        self.assertAlmostEqual(floor_38, expected_initial_variance)

    def test_density_l2_matches_definition(self):
        left = np.array([1.0, 2.0, 3.0])
        right = np.array([1.0, 1.0, 1.0])
        dx = 0.25
        expected = np.sqrt(dx * (0.0 + 1.0 + 4.0))
        self.assertAlmostEqual(density_l2(left, right, dx), expected)


class BetaProjectionRepresentationTests(unittest.TestCase):
    def test_degree_38_represents_beta_20_20_far_better_than_degree_20(self):
        model, _ = build_reference_case()
        edges, x, dx = model.fokker_planc_reference.grid(201)
        target_mass = beta_initial_cell_mass(edges)
        target_density = target_mass / dx

        errors = {}
        for degree in (20, 38):
            basis = model.fokker_planc._normalise_basis_discretely(
                model.fokker_planc.beta_basis(x, degree),
                dx,
            )
            coefficients, projected_density = model.fokker_planc.project_density(
                target_density,
                basis,
                dx,
            )
            projected_mass = projected_density * dx
            errors[degree] = cell_mass_l1(projected_mass, target_mass)

            self.assertLess(abs(np.sum(coefficients) - 1.0), 1.0e-9)
            self.assertGreaterEqual(np.min(coefficients), -1.0e-9)
            self.assertLess(abs(np.sum(projected_mass) - 1.0), 1.0e-9)

        self.assertGreater(errors[20], 0.05)
        self.assertLess(errors[38], 1.0e-3)
        self.assertLess(errors[38], errors[20])

    def test_small_projection_study_preserves_probability(self):
        result = run_projection_study(
            degrees=(10, 20),
            x_steps=101,
            max_step=1.0 / 96.0,
            t_eval=np.array([0.0, 1.0]),
        )

        for degree in result["degrees"]:
            item = result["projection"][degree]
            self.assertLess(item["maximum_coefficient_sum_error"], 1.0e-9)
            self.assertGreaterEqual(item["minimum_coefficient"], -1.0e-9)
            self.assertLess(item["maximum_mass_error"], 1.0e-9)
            self.assertTrue(np.all(np.isfinite(item["l2"])))
            self.assertTrue(np.all(np.isfinite(item["l1"])))
            self.assertTrue(np.all(np.isfinite(item["wasserstein_1"])))
            self.assertEqual(item["density"].shape, result["fp_density"].shape)


if __name__ == "__main__":
    unittest.main()
