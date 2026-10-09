"""Generic numerical controls; no stochastic BM model or parameter bank."""
import unittest
import numpy as np
from src.method.euler_maruyama import euler_maruyama
from src.diagnostics.stochastic import loglog_convergence_order,rms_strong_error,successive_convergence_orders

class TimeGridTests(unittest.TestCase):

    def setUp(self):
        self.method = euler_maruyama()

    def test_grid_contains_outputs_breakpoints_and_respects_max_step(self):
        t_eval = np.array([0.0, 0.4, 1.0])
        integration_t, output_indices = self.method.build_time_grid(t_eval, breakpoints=np.array([0.55]), max_step=0.1)
        np.testing.assert_array_equal(integration_t[output_indices], t_eval)
        self.assertTrue(np.any(integration_t == 0.55))
        self.assertLessEqual(np.max(np.diff(integration_t)), 0.1 + 1e-15)

    def test_no_extra_controls_preserve_the_output_grid(self):
        t_eval = np.array([0.0, 0.2, 0.7, 1.0])
        integration_t, output_indices = self.method.build_time_grid(t_eval)
        np.testing.assert_array_equal(integration_t, t_eval)
        np.testing.assert_array_equal(output_indices, np.arange(t_eval.size))

    def test_invalid_grid_controls_are_rejected(self):
        cases = ((np.array([0.0, 0.5, 0.4]), None, np.inf), (np.array([0.0, 1.0]), np.array([0.0]), np.inf),
            (np.array([0.0, 1.0]), np.array([0.5, 0.4]), np.inf), (np.array([0.0, 1.0]), None, 0.0))
        for t_eval, breakpoints, max_step in cases:
            with self.subTest(t_eval=t_eval, breakpoints=breakpoints, max_step=max_step):
                with self.assertRaises(ValueError):
                    self.method.build_time_grid(t_eval, breakpoints, max_step)

class BrownianIncrementTests(unittest.TestCase):

    def setUp(self):
        self.method = euler_maruyama()

    def test_generated_increments_use_each_step_variance(self):
        t = np.array([0.0, 0.1, 0.5, 1.0])
        increments = self.method.brownian_increments(t, n_paths=3, seed=12)
        expected = np.random.default_rng(12).standard_normal((3, 3)).T * np.sqrt(np.diff(t))[None, :]
        np.testing.assert_array_equal(increments, expected)

    def test_supplied_increments_drive_the_exact_one_step_formula(self):
        t = np.array([0.0, 0.25])
        dW = np.array([[0.1], [-0.2], [0.05]])
        paths = self.method.solve_paths(t, 0.5, lambda time, state: 2.0, lambda time, state: 3.0, n_paths=3,
            boundary='none', brownian_increments=dW)
        expected = 0.5 + 2.0 * 0.25 + 3.0 * dW[:, 0]
        np.testing.assert_array_equal(paths[:, -1], expected)

    def test_seed_and_supplied_increments_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, 'seed must be None'):
            self.method.solve_paths(np.array([0.0, 1.0]), 0.5, lambda time, state: 0.0, lambda time, state: 1.0,
                n_paths=1, seed=1, boundary='none', brownian_increments=np.array([[0.0]]))

    def test_coarsening_sums_fine_brownian_increments(self):
        fine = np.array([[0.1, -0.2, 0.3, 0.4], [-0.4, 0.5, 0.2, -0.1]])
        coarse = self.method.coarsen_brownian_increments(fine, factor=2)
        np.testing.assert_allclose(coarse, [[-0.1, 0.7], [0.1, 0.1]], atol=1e-15)

    def test_coupled_constant_diffusion_paths_match_at_coarse_times(self):
        fine_t = np.linspace(0.0, 1.0, 5)
        coarse_t = fine_t[::2]
        fine_dW = np.array([[0.1, -0.2, 0.3, 0.4]])
        coarse_dW = self.method.coarsen_brownian_increments(fine_dW, factor=2)
        drift = lambda time, state: np.zeros_like(state)
        diffusion = lambda time, state: np.ones_like(state)
        fine_paths = self.method.solve_paths(fine_t, 0.5, drift, diffusion, boundary='none',
            brownian_increments=fine_dW)
        coarse_paths = self.method.solve_paths(coarse_t, 0.5, drift, diffusion, boundary='none',
            brownian_increments=coarse_dW)
        np.testing.assert_allclose(fine_paths[:, ::2], coarse_paths, atol=1e-15)

class BoundaryDiagnosticTests(unittest.TestCase):

    def setUp(self):
        self.method = euler_maruyama()
        self.t = np.array([0.0, 1.0])
        self.initial = np.array([0.1, 0.9])
        self.dW = np.array([[-0.2], [0.2]])
        self.drift = lambda time, state: np.zeros_like(state)
        self.diffusion = lambda time, state: np.ones_like(state)

    def test_raw_violations_are_reported_before_reflection(self):
        paths, diagnostics = self.method.solve_paths(self.t, self.initial, self.drift, self.diffusion, n_paths=2,
            boundary='reflect', brownian_increments=self.dW, return_diagnostics=True)
        np.testing.assert_allclose(paths[:, -1], [0.1, 0.9], atol=1e-15)
        self.assertAlmostEqual(diagnostics['minimum_raw_proposal'], -0.1)
        self.assertAlmostEqual(diagnostics['maximum_raw_proposal'], 1.1)
        self.assertEqual(diagnostics['lower_violations'], 1)
        self.assertEqual(diagnostics['upper_violations'], 1)
        self.assertEqual(diagnostics['total_violations'], 2)
        self.assertEqual(diagnostics['violating_paths'], 2)
        self.assertEqual(diagnostics['boundary_corrections'], 2)

    def test_boundary_none_leaves_the_same_raw_proposals_uncorrected(self):
        paths, diagnostics = self.method.solve_paths(self.t, self.initial, self.drift, self.diffusion, n_paths=2,
            boundary='none', brownian_increments=self.dW, return_diagnostics=True)
        np.testing.assert_allclose(paths[:, -1], [-0.1, 1.1], atol=1e-15)
        self.assertEqual(diagnostics['total_violations'], 2)
        self.assertEqual(diagnostics['boundary_corrections'], 0)

    def test_one_initial_state_per_path_is_supported(self):
        paths = self.method.solve_paths(np.array([0.0, 0.5, 1.0]), np.array([0.2, 0.8]), self.drift, lambda time,
            state: np.zeros_like(state), n_paths=2, boundary='none', brownian_increments=np.zeros((2, 2)))
        np.testing.assert_array_equal(paths[0], [0.2, 0.2, 0.2])
        np.testing.assert_array_equal(paths[1], [0.8, 0.8, 0.8])

class ConvergenceDiagnosticTests(unittest.TestCase):

    def test_rms_strong_error_matches_definition(self):
        approximation = np.array([1.0, 2.0, 4.0])
        reference = np.array([0.0, 2.0, 2.0])
        expected = np.sqrt((1.0 + 0.0 + 4.0) / 3.0)
        self.assertAlmostEqual(rms_strong_error(approximation, reference), expected)

    def test_known_power_law_recovers_its_order(self):
        step_sizes = np.array([1.0, 0.5, 0.25, 0.125])
        errors = 3.0 * step_sizes ** 0.5
        np.testing.assert_allclose(successive_convergence_orders(errors), [0.5, 0.5, 0.5], atol=1e-14)
        self.assertAlmostEqual(loglog_convergence_order(step_sizes, errors), 0.5, places=14)

    def test_invalid_convergence_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            rms_strong_error([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            successive_convergence_orders([1.0])
        with self.assertRaises(ValueError):
            successive_convergence_orders([1.0, 0.0])
        with self.assertRaises(ValueError):
            successive_convergence_orders([1.0, 0.5], refinement_factor=1.0)
        with self.assertRaises(ValueError):
            loglog_convergence_order([0.5, 1.0], [1.0, 0.5])

    def test_exact_logistic_benchmark_exhibits_em_half_order(self):
        method = euler_maruyama()
        sigma = 0.5
        x0 = 0.5
        n_paths = 2000
        step_counts = np.array([24, 48, 96, 192], dtype=int)
        finest_steps = int(step_counts[-1])
        finest_t = np.linspace(0.0, 1.0, finest_steps + 1)
        fine_dW = method.brownian_increments(finest_t, n_paths=n_paths, seed=12345)
        exact_terminal = 1.0 / (1.0 + np.exp(-(np.log(x0 / (1.0 - x0)) + sigma * np.sum(fine_dW, axis=1))))

        def drift(time, state):
            del time
            state = np.asarray(state, dtype=float)
            return 0.5 * sigma ** 2 * state * (1.0 - state) * (1.0 - 2.0 * state)

        def diffusion(time, state):
            del time
            state = np.asarray(state, dtype=float)
            return sigma * state * (1.0 - state)
        errors = []
        for n_steps in step_counts:
            factor = finest_steps // int(n_steps)
            dW = fine_dW if factor == 1 else method.coarsen_brownian_increments(fine_dW, factor=factor)
            t = np.linspace(0.0, 1.0, int(n_steps) + 1)
            paths, diagnostics = method.solve_paths(t, x0, drift, diffusion, n_paths=n_paths, boundary='none',
                brownian_increments=dW, return_diagnostics=True)
            self.assertEqual(diagnostics['total_violations'], 0)
            errors.append(rms_strong_error(paths[:, -1], exact_terminal))
        errors = np.asarray(errors)
        self.assertTrue(np.all(np.diff(errors) < 0.0))
        order = loglog_convergence_order(1.0 / step_counts.astype(float), errors)
        self.assertGreater(order, 0.35)
        self.assertLess(order, 0.7)

if __name__=='__main__':unittest.main()
