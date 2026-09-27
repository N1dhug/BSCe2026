"""BM1 regression tests grouped by responsibility."""

import numpy as np
import unittest
from scipy.special import betainc
from src.diagnostics.bm_1 import (cell_mass_cdf_sup, cell_mass_l1, cell_mass_wasserstein_1, empirical_cell_mass)
from src.method.bm_1_control import (FVStochasticPriceEvaluator, FastFVStochasticPriceEvaluator,
    beta_initial_cell_mass, multistart_powell, sample_beta_initial_states)
from src.model.bm_1 import bm_1
from src.solver.finite_volume import finite_volume_1d
from src.tools.input_generator.input_builder import input_builder
from src.validation.bm_1.cases import (BASELINE, INITIAL_ALPHA, INITIAL_BETA, PRICE_DROP_TIME, PRICE_VALUES, T_EVAL,
    build_reference_case)
from src.validation.bm_1.stochastic import run_comparison
from unittest.mock import patch

def linear_response(x):
    return 1.0 - 2.0 * np.asarray(x, dtype=float)

class FiniteVolumeGeneratorTests(unittest.TestCase):

    def setUp(self):
        self.solver = finite_volume_1d()

    def test_generator_is_conservative_markov_generator(self):
        x_steps = 21
        edges, centers, dx = self.solver.grid(x_steps)
        faces = edges[1:-1]
        drift_faces = 0.4 - 0.8 * faces
        diffusion = 0.5 * centers * (1.0 - centers)
        lower, diagonal, upper, diagnostics = self.solver.generator_tridiagonal(dx, drift_faces, diffusion)
        generator = np.diag(diagonal) + np.diag(lower, k=-1) + np.diag(upper, k=1)
        np.testing.assert_allclose(np.sum(generator, axis=0), 0.0, atol=2e-14, rtol=0.0)
        self.assertGreaterEqual(np.min(lower), 0.0)
        self.assertGreaterEqual(np.min(upper), 0.0)
        self.assertLessEqual(np.max(diagonal), 0.0)
        self.assertLessEqual(diagnostics['maximum_abs_column_sum'], 2e-14)
        self.assertGreaterEqual(diagnostics['minimum_off_diagonal_rate'], 0.0)

    def test_zero_operator_leaves_cell_mass_unchanged(self):
        initial_mass = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        result = self.solver.solve(np.array([0.0, 0.3, 1.0]), initial_mass, lambda time, x: np.zeros_like(x),
            lambda time, x: np.zeros_like(x), max_step=0.1)
        np.testing.assert_allclose(result['cell_mass'], np.tile(initial_mass, (3, 1)), atol=1e-15, rtol=0.0)
        np.testing.assert_allclose(result['mass'], 1.0, atol=1e-15, rtol=0.0)

    def test_uniform_density_is_stationary_for_reflecting_constant_diffusion(self):
        x_steps = 41
        initial_mass = np.full(x_steps, 1.0 / x_steps)
        diffusion = np.sqrt(0.2)
        result = self.solver.solve(np.array([0.0, 0.5]), initial_mass, lambda time, x: np.zeros_like(x), lambda time,
            x: diffusion * np.ones_like(x), max_step=0.01)
        np.testing.assert_allclose(result['cell_mass'][-1], initial_mass, atol=2e-14, rtol=0.0)
        self.assertGreaterEqual(result['diagnostics']['minimum_cell_mass'], 0.0)
        self.assertFalse(result['diagnostics']['used_clipping'])
        self.assertFalse(result['diagnostics']['used_renormalization'])

    def test_heat_equation_benchmark_improves_under_spatial_refinement(self):
        final_time = 0.1
        diffusion_coefficient = 0.05
        amplitude = 0.5
        errors = []
        for x_steps in (40, 80):
            edges, _, dx = self.solver.grid(x_steps)
            initial_mass = dx + amplitude / np.pi * (np.sin(np.pi * edges[1:]) - np.sin(np.pi * edges[:-1]))
            result = self.solver.solve(np.array([0.0, final_time]), initial_mass, lambda time, x: np.zeros_like(x),
                lambda time, x: np.sqrt(2.0 * diffusion_coefficient) * np.ones_like(x), max_step=0.0005)
            exact_amplitude = amplitude * np.exp(-diffusion_coefficient * np.pi ** 2 * final_time)
            exact_mass = dx + exact_amplitude / np.pi * (np.sin(np.pi * edges[1:]) - np.sin(np.pi * edges[:-1]))
            errors.append(np.sum(np.abs(result['cell_mass'][-1] - exact_mass)))
            self.assertLess(result['diagnostics']['maximum_abs_mass_error'], 2e-13)
            self.assertGreaterEqual(result['diagnostics']['minimum_cell_mass'], -1e-13)
        self.assertLess(errors[1], errors[0])

    def test_solver_rejects_unnormalized_initial_mass(self):
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            self.solver.solve(np.array([0.0, 1.0]), np.array([0.2, 0.2, 0.2]), lambda time, x: np.zeros_like(x),
                lambda time, x: np.zeros_like(x))

class BM1FiniteVolumeReferenceTests(unittest.TestCase):

    def build_case(self):
        T = (0.0, 1.0)
        B = input_builder.energy((np.array(T), np.array([0.3, 0.3])))
        price_edges = np.array([0.0, 0.37, 1.0])
        U = input_builder.price_intervals((price_edges, np.array([0.75, 0.25])))
        parameters = [U, linear_response, linear_response, 1.0, 2.0, 1.0]
        return (bm_1(B, B, T, 3), parameters, price_edges[1:-1])

    def test_bm1_reference_preserves_probability_without_post_correction(self):
        model, parameters, breakpoints = self.build_case()
        x_steps = 101
        edges, _, _ = model.fokker_planc_reference.grid(x_steps)
        initial_mass = np.diff(betainc(20.0, 20.0, edges))
        result = model.fokker_planc_reference_solver(initial_mass, parameters, 0.5, t_eval=np.array([0.0, 0.5, 1.0]),
            breakpoints=breakpoints, max_step=1.0 / 96.0)
        t, x, density, cell_mass, mean, variance, price, baseline, expected_demand, diagnostics = result
        np.testing.assert_array_equal(t, [0.0, 0.5, 1.0])
        self.assertEqual(density.shape, (3, x_steps))
        self.assertEqual(cell_mass.shape, (3, x_steps))
        np.testing.assert_allclose(np.sum(cell_mass, axis=1), 1.0, atol=2e-13)
        self.assertTrue(np.all(mean >= 0.0))
        self.assertTrue(np.all(mean <= 1.0))
        self.assertTrue(np.all(variance >= 0.0))
        self.assertTrue(np.all(np.isfinite(expected_demand)))
        self.assertFalse(diagnostics['used_clipping'])
        self.assertFalse(diagnostics['used_renormalization'])
        self.assertLess(diagnostics['maximum_abs_mass_error'], 2e-13)
        self.assertGreaterEqual(diagnostics['minimum_cell_mass'], -1e-13)
        self.assertTrue(np.any(diagnostics['integration_t'] == breakpoints[0]))
        self.assertEqual(price.shape, t.shape)
        self.assertEqual(baseline.shape, t.shape)
        self.assertEqual(x.shape, (x_steps,))

class ReferenceCaseTests(unittest.TestCase):

    def test_shared_case_builds_expected_bm_inputs(self):
        model, parameters = build_reference_case()
        self.assertEqual(model.T, (0.0, 1.0))
        np.testing.assert_array_equal(model.t, T_EVAL)
        self.assertAlmostEqual(float(model.B(0.25)), BASELINE)
        self.assertAlmostEqual(float(parameters[0](0.25)), PRICE_VALUES[0])
        self.assertAlmostEqual(float(parameters[0](PRICE_DROP_TIME)), PRICE_VALUES[1])

    def test_beta_cell_mass_is_normalized_and_centered(self):
        model, _ = build_reference_case()
        edges, centers, _ = model.fokker_planc_reference.grid(201)
        mass = beta_initial_cell_mass(edges)
        self.assertAlmostEqual(np.sum(mass), 1.0, places=14)
        self.assertGreaterEqual(np.min(mass), 0.0)
        self.assertAlmostEqual(np.sum(mass * centers), 0.5, places=12)

    def test_beta_sampling_is_reproducible_and_matches_initial_law(self):
        left = sample_beta_initial_states(20000, seed=42)
        right = sample_beta_initial_states(20000, seed=42)
        np.testing.assert_array_equal(left, right)
        self.assertTrue(np.all((left > 0.0) & (left < 1.0)))
        self.assertLess(abs(np.mean(left) - 0.5), 0.005)
        expected_variance = INITIAL_ALPHA * INITIAL_BETA / ((INITIAL_ALPHA + INITIAL_BETA) ** 2 * (INITIAL_ALPHA + INITIAL_BETA + 1.0))
        self.assertLess(abs(np.var(left, ddof=1) - expected_variance), 0.001)

class DistributionMetricTests(unittest.TestCase):

    def test_empirical_mass_and_distances_match_simple_case(self):
        edges = np.array([0.0, 0.5, 1.0])
        empirical = empirical_cell_mass(np.array([0.1, 0.2, 0.7, 0.8]), edges)
        np.testing.assert_array_equal(empirical, [0.5, 0.5])
        other = np.array([0.75, 0.25])
        self.assertAlmostEqual(cell_mass_l1(empirical, other), 0.5)
        self.assertAlmostEqual(cell_mass_cdf_sup(empirical, other), 0.25)
        self.assertAlmostEqual(cell_mass_wasserstein_1(empirical, other, 0.5), 0.125)

class MCvsFVIntegrationTests(unittest.TestCase):

    def test_small_controlled_comparison_is_consistent(self):
        result = run_comparison(n_paths=5000, x_steps=101, max_step=1.0 / 96.0, initial_seed=77, brownian_seed=78)
        self.assertEqual(result['em_diagnostics']['total_violations'], 0)
        self.assertLess(result['fp_diagnostics']['maximum_abs_mass_error'], 1e-10)
        self.assertLess(np.max(np.abs(result['mc_mean'] - result['fp_mean'])), 0.02)
        self.assertLess(np.max(np.abs(result['mc_expected_demand'] - result['fp_expected_demand'])), 0.04)
        self.assertLess(result['cdf_sup'][-1], 0.1)
        self.assertLess(result['l1'][-1], 0.4)
        np.testing.assert_allclose(np.sum(result['mc_cell_mass'], axis=1), 1.0, atol=1e-14)

class FastFiniteVolumeTests(unittest.TestCase):

    def pair(self, n=51, sigma=0.5, counts=2, **fast_options):
        model, params = build_reference_case()
        edges = np.linspace(0, 1, counts + 1)
        grid, _, _ = model.fokker_planc_reference.grid(n)
        initial = beta_initial_cell_mass(grid)
        args = (model, params, edges, sigma, initial)
        kwargs = dict(objective_max_step=1 / 24, solver_max_step=1 / 96, breakpoints=np.array([0.37, 0.71]))
        return (FVStochasticPriceEvaluator(*args, **kwargs),
            FastFVStochasticPriceEvaluator(*args, **kwargs, **fast_options))

    def assert_equivalent(self, slow, fast, prices):
        left, right = (slow.evaluate(prices), fast.evaluate(prices))
        np.testing.assert_allclose(left.cell_mass, right.cell_mass, atol=2e-12, rtol=2e-11)
        for name in ('total', 'bias', 'variance', 'expected_demand', 'second_moment'):
            np.testing.assert_allclose(getattr(left.objective, name), getattr(right.objective, name), atol=2e-12,
                rtol=2e-11)
        np.testing.assert_array_equal(left.diagnostics['integration_t'], right.diagnostics['integration_t'])
        self.assertLess(right.diagnostics['maximum_abs_mass_error'], 1e-09)
        self.assertGreaterEqual(right.diagnostics['minimum_cell_mass'], -1e-12)
        self.assertFalse(right.diagnostics['used_clipping'])
        self.assertFalse(right.diagnostics['used_renormalization'])

    def test_grid_sizes_noise_and_extreme_prices_match_reference(self):
        for n, sigma in ((3, 0.0), (31, 0.5), (201, 0.5), (1001, 1.0)):
            with self.subTest(n=n, sigma=sigma):
                slow, fast = self.pair(n=n, sigma=sigma)
                for prices in ([0, 1], [1, 0], [0.73, 0.26]):
                    self.assert_equivalent(slow, fast, prices)

    def test_random_24_price_schedules_match_reference(self):
        slow, fast = self.pair(counts=24)
        for seed in range(3):
            self.assert_equivalent(slow, fast, np.random.default_rng(seed).uniform(0, 1, 24))

    def test_time_varying_baseline_and_target_jumps(self):
        slow, fast = self.pair()

        def baseline(t):
            t = np.asarray(t)
            return 0.2 + 0.1 * np.sin(7 * t) ** 2 + 0.2 * (t >= 0.37)
        slow.model.B = baseline
        fast.model.B = baseline
        self.assert_equivalent(slow, fast, [0.64, 0.33])

    def test_zero_flexibility_and_small_capacity(self):
        for c, xi in ((0.01, 1.0), (2.0, 0.0)):
            slow, fast = self.pair()
            slow.parameter_template[3] = c
            fast.parameter_template[3] = c
            slow.parameter_template[5] = xi
            fast.parameter_template[5] = xi
            self.assert_equivalent(slow, fast, [0.01, 0.98])

    def test_batch_size_does_not_change_results(self):
        for size in (1, 7, 128, 1000):
            slow, fast = self.pair(batch_size=size)
            self.assert_equivalent(slow, fast, [0.72, 0.28])

    def test_cost_only_mode_matches_full_history_mode(self):
        slow, fast = self.pair()
        full = fast.evaluate([0.72, 0.28])
        cost = fast.evaluate_cost([0.72, 0.28])
        self.assertEqual(full.objective.total, cost.objective.total)
        self.assertFalse(hasattr(cost, 'cell_mass'))
        self.assertFalse(cost.diagnostics['saved_density_history'])
        self.assertTrue(full.diagnostics['saved_density_history'])

    def test_scalar_time_callbacks_remain_supported_explicitly(self):
        slow, fast = self.pair(vectorized=False)

        def baseline(t):
            t = np.asarray(t)
            if t.ndim > 1:
                raise ValueError('No 2D inputs')
            return 0.3 + 0.05 * t
        slow.model.B = baseline
        fast.model.B = baseline
        self.assert_equivalent(slow, fast, [0.7, 0.3])

    def test_initial_mass_not_mutated_by_lapack(self):
        slow, fast = self.pair()
        before = fast.initial_mass.copy()
        fast([0.8, 0.2])
        fast([0.1, 0.9])
        np.testing.assert_array_equal(before, fast.initial_mass)

    def test_bad_controls_and_prices_are_rejected(self):
        for size in (0, 1.5, True):
            with self.assertRaises((ValueError, TypeError)):
                self.pair(batch_size=size)
        _, fast = self.pair()
        for prices in ([0.3], [0.1, np.nan], [1.1, 0.3]):
            with self.assertRaises(ValueError):
                fast(prices)

    def test_invalid_physical_response_is_not_hidden(self):
        _, fast = self.pair()
        fast.model.B = lambda t: np.zeros_like(t) + 1.1
        with self.assertRaises(ValueError):
            fast([0.5, 0.5])

    def test_lapack_failure_is_reported(self):
        _, fast = self.pair()
        with patch('src.method.bm_1_control.dgtsv', return_value=(None, None, None, None, 2)):
            with self.assertRaisesRegex(RuntimeError, 'LAPACK'):
                fast([0.5, 0.5])

    def test_raw_mass_loss_is_not_normalized_away(self):
        from src.method.bm_1_control import dgtsv
        _, fast = self.pair()

        def lose_mass(*args, **kwargs):
            out = list(dgtsv(*args, **kwargs))
            out[3] = 0.99 * out[3]
            return tuple(out)
        with patch('src.method.bm_1_control.dgtsv', side_effect=lose_mass):
            with self.assertRaisesRegex(RuntimeError, 'probability'):
                fast([0.5, 0.5])

    def test_small_powell_solution_matches_reference(self):
        slow, fast = self.pair(n=21)
        starts = [[0.7, 0.3]]
        options = dict(xtol=1e-05, ftol=1e-09, maxiter=30)
        a = multistart_powell(slow, starts, options=options)
        b = multistart_powell(fast, starts, options=options)
        np.testing.assert_allclose(a.best.x, b.best.x, atol=2e-06, rtol=0)
        self.assertLess(abs(a.best.fun - b.best.fun), 1e-10)


if __name__ == '__main__':
    unittest.main()
