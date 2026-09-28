"""BM1 regression tests grouped by responsibility."""

import numpy as np
import unittest
from unittest.mock import patch
from src.diagnostics.bm_1 import (beta_mixture_variance_floor, cell_mass_l1, density_l2)
from src.control.bm_1 import beta_initial_cell_mass
from src.method.beta_mixture import (
    _project_with_warm_start,
    _projection_matrices,
    evolve_projected_density as _evolve_projected_density,
)
from src.validation.bm_1.cases import (INITIAL_ALPHA, INITIAL_BETA, build_reference_case)
from src.validation.bm_1.mixture import (_positive_mass_preserving_perturbation, _simplex_tangent_condition,
    run_conditioning_study, run_dynamic_mixture_study, run_projection_study)

class BetaProjectionTheoryTests(unittest.TestCase):

    def test_variance_floor_matches_beta_basis_formula(self):
        mean = 0.5
        expected_initial_variance = INITIAL_ALPHA * INITIAL_BETA / ((INITIAL_ALPHA + INITIAL_BETA) ** 2 * (INITIAL_ALPHA + INITIAL_BETA + 1.0))
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
            basis = model.fokker_planc._normalise_basis_discretely(model.fokker_planc.beta_basis(x, degree), dx)
            coefficients, projected_density = model.fokker_planc.project_density(target_density, basis, dx)
            projected_mass = projected_density * dx
            errors[degree] = cell_mass_l1(projected_mass, target_mass)
            self.assertLess(abs(np.sum(coefficients) - 1.0), 1e-09)
            self.assertGreaterEqual(np.min(coefficients), -1e-09)
            self.assertLess(abs(np.sum(projected_mass) - 1.0), 1e-09)
        self.assertGreater(errors[20], 0.05)
        self.assertLess(errors[38], 0.001)
        self.assertLess(errors[38], errors[20])

    def test_small_projection_study_preserves_probability(self):
        result = run_projection_study(degrees=(10, 20), x_steps=101, max_step=1.0 / 96.0, t_eval=np.array([0.0, 1.0]))
        for degree in result['degrees']:
            item = result['projection'][degree]
            self.assertLess(item['maximum_coefficient_sum_error'], 1e-09)
            self.assertGreaterEqual(item['minimum_coefficient'], -1e-09)
            self.assertLess(item['maximum_mass_error'], 1e-09)
            self.assertTrue(np.all(np.isfinite(item['l2'])))
            self.assertTrue(np.all(np.isfinite(item['l1'])))
            self.assertTrue(np.all(np.isfinite(item['wasserstein_1'])))
            self.assertEqual(item['density'].shape, result['fp_density'].shape)

class DynamicMixturePropagationTests(unittest.TestCase):

    def test_zero_operator_preserves_an_existing_mixture(self):
        model, _ = build_reference_case()
        edges, x, dx = model.fokker_planc_reference.grid(41)
        del edges
        degree = 10
        basis = model.fokker_planc._normalise_basis_discretely(model.fokker_planc.beta_basis(x, degree), dx)
        coefficients = np.zeros(degree + 1)
        coefficients[4] = 0.35
        coefficients[5] = 0.65
        result = _evolve_projected_density(method=model.fokker_planc, t_eval=np.array([0.0, 0.5, 1.0]),
            breakpoints=np.array([0.37]), x=x, dx=dx, basis=basis, initial_coefficients=coefficients,
            drift=lambda time, state: np.zeros_like(state), diffusion=lambda time, state: np.zeros_like(state),
            max_step=0.1, cfl=0.4)
        expected_density = coefficients @ basis
        np.testing.assert_allclose(result['density'], np.tile(expected_density, (3, 1)), atol=2e-10, rtol=0.0)
        np.testing.assert_allclose(result['coefficients'], np.tile(coefficients, (3, 1)), atol=2e-10, rtol=0.0)
        diagnostics = result['diagnostics']
        self.assertEqual(diagnostics['projection_fallback_count'], 0)
        self.assertFalse(diagnostics['used_clipping'])
        self.assertFalse(diagnostics['used_renormalization'])
        self.assertLess(diagnostics['maximum_abs_raw_prediction_mass_error'], 1e-12)

    def test_projection_retries_same_simplex_problem_when_active_set_stalls(self):
        model, _ = build_reference_case()
        _, x, dx = model.fokker_planc_reference.grid(31)
        degree = 8
        basis = model.fokker_planc._normalise_basis_discretely(
            model.fokker_planc.beta_basis(x, degree),
            dx,
        )
        matrix, hessian, _ = _projection_matrices(basis, dx)
        original = np.zeros(degree + 1, dtype=float)
        original[3] = 0.4
        original[4] = 0.6
        density = original @ basis
        x0 = np.full(degree + 1, 1.0 / (degree + 1.0))

        with patch(
            'src.method.beta_mixture._SimplexProjector.solve',
            side_effect=RuntimeError('forced active-set stall'),
        ):
            coefficients, projected, used_fallback = _project_with_warm_start(
                density,
                basis,
                matrix,
                hessian,
                dx,
                x0,
                return_fallback=True,
            )

        self.assertTrue(used_fallback)
        self.assertGreaterEqual(np.min(coefficients), -1e-9)
        self.assertAlmostEqual(float(np.sum(coefficients)), 1.0, places=9)
        self.assertAlmostEqual(float(np.sum(projected) * dx), 1.0, places=9)
        self.assertLess(density_l2(projected, density, dx), 1e-6)


class DynamicMixtureReferenceTests(unittest.TestCase):

    def test_small_dynamic_study_preserves_probability_and_separates_errors(self):
        result = run_dynamic_mixture_study(degrees=(20,), x_steps=51, max_step=1.0 / 96.0,
            t_eval=np.array([0.0, 0.5, 1.0]), cfl=0.35)
        item = result['dynamic'][20]
        diagnostics = item['diagnostics']
        np.testing.assert_allclose(np.sum(item['cell_mass'], axis=1), 1.0, atol=1e-09, rtol=0.0)
        self.assertGreater(diagnostics['n_steps'], 0)
        self.assertTrue(np.isfinite(diagnostics['hessian_condition_number']))
        self.assertGreaterEqual(diagnostics['hessian_condition_number'], 1.0)
        self.assertGreaterEqual(diagnostics['minimum_raw_prediction'], -1e-10)
        self.assertLess(diagnostics['maximum_abs_raw_prediction_mass_error'], 1e-09)
        self.assertLess(diagnostics['maximum_abs_projected_mass_error'], 1e-09)
        self.assertLess(diagnostics['maximum_abs_coefficient_sum_error'], 1e-09)
        self.assertGreaterEqual(diagnostics['minimum_coefficient'], -1e-09)
        self.assertFalse(diagnostics['used_clipping'])
        self.assertFalse(diagnostics['used_renormalization'])
        self.assertAlmostEqual(item['dynamic_gap']['l2'][0], 0.0, places=12)
        self.assertAlmostEqual(item['total_error']['l2'][0], result['projection'][20]['l2'][0], places=12)
        self.assertTrue(np.all(np.isfinite(item['total_error']['l2'])))
        self.assertTrue(np.all(np.isfinite(item['dynamic_gap']['l2'])))
        self.assertTrue(np.all(np.isfinite(item['mean'])))
        self.assertTrue(np.all(np.isfinite(item['variance'])))
        self.assertTrue(np.all(np.isfinite(item['expected_demand'])))

class ConditioningTheoryTests(unittest.TestCase):

    def test_simplex_tangent_condition_matches_one_dimensional_free_direction(self):
        hessian = np.diag([1.0, 4.0])
        coefficients = np.array([0.5, 0.5])
        condition, free_size, tangent_dimension = _simplex_tangent_condition(hessian, coefficients, active_tol=1e-12)
        self.assertAlmostEqual(condition, 1.0)
        self.assertEqual(free_size, 2)
        self.assertEqual(tangent_dimension, 1)

    def test_positive_perturbation_preserves_mass_and_nonnegativity(self):
        density = np.array([0.5, 1.0, 1.5, 1.0])
        dx = 0.25
        density /= np.sum(density) * dx
        direction = np.array([-2.0, -0.5, 0.7, 1.8])
        perturbed = _positive_mass_preserving_perturbation(density, dx, 1e-06, direction)
        self.assertGreaterEqual(np.min(perturbed), 0.0)
        self.assertAlmostEqual(np.sum(perturbed) * dx, 1.0, places=14)
        self.assertGreater(np.linalg.norm(perturbed - density), 0.0)

class ConditioningIntegrationTests(unittest.TestCase):

    def test_small_study_returns_finite_sensitivity_diagnostics(self):
        result = run_conditioning_study(degrees=(10,), epsilons=(1e-08,), n_directions=1, seed=44, x_steps=41,
            max_step=1.0 / 48.0, t_eval=np.array([0.0, 1.0]), cfl=0.35)
        item = result['conditioning'][10]
        self.assertTrue(np.isfinite(item['matrix_condition_number']))
        self.assertTrue(np.isfinite(item['hessian_condition_number']))
        self.assertGreaterEqual(item['matrix_condition_number'], 1.0)
        self.assertGreaterEqual(item['hessian_condition_number'], 1.0)
        self.assertTrue(np.all(np.isfinite(item['static_local_condition_numbers'])))
        self.assertTrue(np.all(np.isfinite(item['dynamic_local_condition_numbers'])))
        alt = item['alternative_start']
        self.assertEqual(alt['failures'], 0)
        self.assertTrue(np.isfinite(alt['maximum_coefficient_l1']))
        self.assertTrue(np.isfinite(alt['maximum_density_l2']))
        pert = item['perturbations'][1e-08]
        self.assertEqual(pert['failures'], 0)
        self.assertGreater(pert['maximum_input_l2'], 0.0)
        self.assertTrue(np.isfinite(pert['maximum_coefficient_l1']))
        self.assertTrue(np.isfinite(pert['maximum_projected_density_l2']))
        self.assertTrue(np.isfinite(pert['maximum_density_amplification']))
        self.assertTrue(np.isfinite(pert['maximum_coefficient_amplification']))


class PreparedSimplexAuditTests(unittest.TestCase):

    def test_identity_problem_matches_exact_simplex_projection(self):
        from src.method.beta_mixture import _SimplexProjector
        rng = np.random.default_rng(431)
        solver = _SimplexProjector(np.eye(8), cache_size=2)
        for target in rng.normal(size=(30, 8)):
            ordered = np.sort(target)[::-1]
            cumulative = np.cumsum(ordered) - 1.0
            rho = np.flatnonzero(ordered - cumulative / np.arange(1, 9) > 0.0)[-1]
            # Independent closed-form Euclidean simplex projection, test only.
            exact = np.maximum(target - cumulative[rho] / (rho + 1), 0.0)
            start = np.full(8, 1.0 / 8)
            actual = solver.solve(target, start)
            np.testing.assert_allclose(actual, exact, atol=2e-11, rtol=0.0)
            np.testing.assert_array_equal(start, np.full(8, 1.0 / 8))
            self.assertLessEqual(len(solver._faces), 2)
            self.assertLess(solver.optimality_gap(target, actual)[0], 1e-10)

    def test_rank_deficient_basis_preserves_density_not_unique_weights(self):
        from src.method.beta_mixture import _SimplexProjector
        matrix = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        target = np.array([0.3, 0.7])
        for initial in (np.array([1., 0., 0.]), np.full(3, 1/3)):
            result = _SimplexProjector(matrix).solve(target, initial)
            np.testing.assert_allclose(matrix @ result, target, atol=1e-10, rtol=0.0)
            self.assertAlmostEqual(float(np.sum(result)), 1.0, places=12)

    def test_cache_changes_neither_solution_nor_input_matrix(self):
        from src.method.beta_mixture import _SimplexProjector
        matrix = np.random.default_rng(432).normal(size=(25, 9))
        original = matrix.copy()
        warm = _SimplexProjector(matrix)
        cold = _SimplexProjector(matrix, cache_size=0)
        start = np.full(9, 1/9)
        for target in np.random.default_rng(433).normal(size=(5, 25)):
            np.testing.assert_array_equal(warm.solve(target, start), cold.solve(target, start))
        np.testing.assert_array_equal(matrix, original)
        matrix[:] = 0.0
        np.testing.assert_array_equal(warm.matrix, original)
        self.assertFalse(warm.matrix.flags.writeable)

    def test_projection_does_not_accept_success_flag_without_optimality(self):
        from types import SimpleNamespace
        from src.method.beta_mixture import _SimplexProjector
        matrix = np.eye(2)
        with patch.object(_SimplexProjector, 'solve', side_effect=RuntimeError('forced')), \
             patch('src.method.beta_mixture.minimize', return_value=SimpleNamespace(
                 x=np.array([.5, .5]), success=True, message='false success')):
            with self.assertRaisesRegex(RuntimeError, 'optimality'):
                _project_with_warm_start(np.array([1., 0.]), matrix, matrix, matrix, 1., np.array([.5, .5]))

    def test_invalid_start_is_not_hidden_by_fallback(self):
        from src.method.beta_mixture import _SimplexProjector
        with patch('src.method.beta_mixture.minimize', side_effect=AssertionError('unexpected fallback')):
            with self.assertRaises(ValueError):
                _SimplexProjector(np.eye(2)).solve(np.ones(2), np.array([.9, .9]))

    def test_fractional_degree_is_rejected_by_theory_and_studies(self):
        for call in (
            lambda: beta_mixture_variance_floor(.5, 8.5),
            lambda: run_projection_study(degrees=(8.5,)),
            lambda: run_dynamic_mixture_study(degrees=(8.5,)),
        ):
            with self.assertRaises(ValueError):
                call()


if __name__ == '__main__':
    unittest.main()
