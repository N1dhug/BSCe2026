"""BM1 regression tests grouped by responsibility."""

import numpy as np
import unittest
from src.diagnostics.stochastic import (loglog_convergence_order, rms_strong_error, successive_convergence_orders)
from src.method.euler_maruyama import euler_maruyama
from src.model.base.bm import bm
from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder
from src.validation.model.bm import bm_validation
from src.validation.model.bm_1 import bm_1_validation

def linear_response(x):
    """Canonical F1 response, also useful for F2-A boundary tests."""
    return 1.0 - 2.0 * np.asarray(x, dtype=float)

def build_case(price=0.25, baseline=0.3, C=1.0, Xi=1.0, T=(0.0, 1.0)):
    """Reuse the existing input builder and BM primitives."""
    B = input_builder.energy((np.array(T), np.array([baseline, baseline])))
    U = input_builder.price_intervals((np.array(T), np.array([price])))
    return (bm(B, B, T, 3), [U, linear_response, linear_response, C, 2.0, Xi])

def solve_small_case(model, parameters, t, sigma_x, x0=0.5, n_paths=4, seed=7):
    """Exercise the actual BM drift and generic EM method without corrections."""
    initial, parameters, sigma_x = bm_1_validation.stochastic_inputs([x0], parameters, sigma_x)

    def drift(time, states):
        return np.array([model.dXdt(time, [state], *parameters)[0] for state in states])
    return euler_maruyama().solve_paths(t, initial, drift, lambda time,
        states: model.diffusion_coefficient(states, sigma_x), n_paths=n_paths, seed=seed, boundary='none')

class StochasticContractTests(unittest.TestCase):

    def setUp(self):
        self.model, self.parameters = build_case()

    def test_valid_input_preserves_parameter_values_and_callables(self):
        original = self.parameters.copy()
        x0, parameters, sigma_x = bm_1_validation.stochastic_inputs([0.5], self.parameters, 0.4)
        np.testing.assert_array_equal(x0, [0.5])
        self.assertEqual(sigma_x, 0.4)
        self.assertIsNot(parameters, self.parameters)
        for index in range(3):
            self.assertIs(parameters[index], original[index])
        self.assertEqual(parameters[3:], original[3:])
        self.assertEqual(self.parameters, original)

    def test_zero_process_noise_is_valid(self):
        self.assertEqual(bm_1_validation.stochastic_inputs([0.5], self.parameters, 0.0)[2], 0.0)

    def test_invalid_process_noise_is_rejected(self):
        for sigma_x in (-0.1, np.nan, np.inf, -np.inf, [0.1, 0.2]):
            with self.subTest(sigma_x=sigma_x), self.assertRaises(ValueError):
                bm_1_validation.stochastic_inputs([0.5], self.parameters, sigma_x)

    def test_invalid_initial_state_is_rejected(self):
        for x0 in ([-0.01], [1.01], [np.nan], [np.inf], [0.4, 0.5], []):
            with self.subTest(x0=x0), self.assertRaises(ValueError):
                bm_1_validation.stochastic_inputs(x0, self.parameters, 0.4)

    def test_boundary_initial_states_are_accepted(self):
        for x0 in (0.0, 1.0):
            initial, _, _ = bm_1_validation.stochastic_inputs([x0], self.parameters, 0.4)
            self.assertEqual(initial[0], x0)

    def test_wrong_endpoints_are_rejected_for_f_and_g(self):
        invalid = (lambda x: 0.5 - np.asarray(x), lambda x: 2.0 * np.asarray(x) - 1.0,
            lambda x: np.zeros_like(np.asarray(x)))
        for index, name in ((1, 'f'), (2, 'g')):
            for response in invalid:
                parameters = self.parameters.copy()
                parameters[index] = response
                with self.subTest(index=index, response=response):
                    with self.assertRaisesRegex(ValueError, f'F2 requires {name}'):
                        bm_1_validation.stochastic_inputs([0.5], parameters, 0.4)

    def test_nonfinite_or_nonscalar_endpoints_are_rejected(self):
        for response in (lambda x: np.nan, lambda x: np.inf, lambda x: [1.0, -1.0]):
            parameters = self.parameters.copy()
            parameters[1] = response
            with self.subTest(response=response), self.assertRaises(ValueError):
                bm_1_validation.stochastic_inputs([0.5], parameters, 0.4)

    def test_noncallable_response_is_rejected(self):
        for index in (1, 2):
            parameters = self.parameters.copy()
            parameters[index] = 0.5
            with self.subTest(index=index), self.assertRaises(TypeError):
                bm_1_validation.stochastic_inputs([0.5], parameters, 0.4)

    def test_roundoff_at_endpoints_is_accepted_without_modification(self):
        response = lambda x: linear_response(x) + 2e-12
        f, g = bm_validation.response_endpoints(response, linear_response)
        self.assertIs(f, response)
        self.assertIs(g, linear_response)
        self.assertGreater(f(0.0), 1.0)

    def test_existing_pchip_builder_satisfies_endpoint_contract(self):
        for kappa in (1.0, 5.0, 10.0):
            response = input_builder.md(*input_builder.md_vals(kappa, 10))
            f, _ = bm_validation.response_endpoints(response, response)
            self.assertIs(f, response)
            values = response(np.linspace(0.0, 1.0, 201))
            self.assertTrue(np.all(np.diff(values) <= 1e-12))

    def test_deterministic_entry_is_not_given_the_new_endpoint_restriction(self):
        parameters = self.parameters.copy()
        parameters[1] = lambda x: 0.5 - np.asarray(x)
        _, validated = bm_1_validation.model_inputs([0.5], parameters)
        self.assertIs(validated[1], parameters[1])
        with self.assertRaises(ValueError):
            bm_1_validation.stochastic_inputs([0.5], parameters, 0.0)

    def test_endpoint_check_does_not_claim_global_range_validation(self):
        response = lambda x: linear_response(x) + 10.0 * np.asarray(x) * (1.0 - np.asarray(x))
        bm_validation.response_endpoints(response, linear_response)
        parameters = self.parameters.copy()
        parameters[1] = response
        with self.assertRaisesRegex(ValueError, 'f\\(X\\)'):
            self.model.dXdt(0.0, [0.5], *parameters)

    def test_parameter_restrictions_are_still_enforced(self):
        for index, value in ((3, 0.0), (3, -1.0), (4, 0.0), (5, -0.1), (5, 1.1)):
            parameters = self.parameters.copy()
            parameters[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                bm_1_validation.stochastic_inputs([0.5], parameters, 0.4)

class DriftAndDiffusionTests(unittest.TestCase):

    def test_drift_points_inward_at_both_boundaries(self):
        for baseline in (0.0, 0.3, 1.0):
            for price in (0.0, 0.25, 0.75, 1.0):
                for Xi in (0.0, 0.5, 1.0):
                    model, parameters = build_case(price=price, baseline=baseline, Xi=Xi)
                    with self.subTest(baseline=baseline, price=price, Xi=Xi):
                        self.assertGreaterEqual(model.dXdt(0.0, [0.0], *parameters)[0], 0.0)
                        self.assertLessEqual(model.dXdt(0.0, [1.0], *parameters)[0], 0.0)

    def test_diffusion_vanishes_at_boundaries(self):
        model, _ = build_case()
        for sigma_x in (0.0, 0.5, 10.0):
            np.testing.assert_array_equal(model.diffusion_coefficient(np.array([0.0, 1.0]), sigma_x), [0.0, 0.0])

    def test_diffusion_is_the_amplitude_not_its_square(self):
        model, _ = build_case()
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(model.diffusion_coefficient(x, 0.8), [0.0, 0.15, 0.2, 0.15, 0.0], atol=1e-15,
            rtol=0.0)

    def test_diffusion_preserves_array_shape(self):
        model, _ = build_case()
        states = np.full((3, 4), 0.5)
        self.assertEqual(model.diffusion_coefficient(states, 0.4).shape, states.shape)

    def test_zero_noise_returns_zero_diffusion(self):
        model, _ = build_case()
        np.testing.assert_array_equal(model.diffusion_coefficient(np.linspace(0.0, 1.0, 9), 0.0), np.zeros(9))

    def test_zero_flexibility_eliminates_drift_but_not_diffusion(self):
        model, parameters = build_case(Xi=0.0)
        for x in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertEqual(model.dXdt(0.0, [x], *parameters)[0], 0.0)
            self.assertEqual(self.eval_demand(model, parameters, x), 0.3)
        self.assertGreater(model.diffusion_coefficient(0.5, 0.4), 0.0)

    @staticmethod
    def eval_demand(model, parameters, x):
        return model.eval(0.0, x, *parameters[:3], *parameters[4:])[-1]

    def test_stochastic_keyword_does_not_add_noise_to_the_drift(self):
        model, parameters = build_case()
        self.assertEqual(model.dXdt(0.0, [0.4], *parameters), model.dXdt(0.0, [0.4], *parameters, stochastic=True))

class EulerLimitAndUnitsTests(unittest.TestCase):

    def test_zero_noise_em_matches_the_euler_recursion(self):
        model, parameters = build_case()
        t = np.linspace(0.0, 0.1, 9)
        paths = solve_small_case(model, parameters, t, sigma_x=0.0)
        expected = [0.5]
        for left, h in zip(t[:-1], np.diff(t)):
            expected.append(expected[-1] + h * model.dXdt(left, [expected[-1]], *parameters)[0])
        np.testing.assert_allclose(paths, np.tile(expected, (4, 1)), atol=1e-15, rtol=0.0)

    def test_zero_noise_and_zero_flexibility_keep_state_constant(self):
        model, parameters = build_case(Xi=0.0)
        paths = solve_small_case(model, parameters, np.linspace(0.0, 1.0, 11), 0.0)
        np.testing.assert_array_equal(paths, np.full((4, 11), 0.5))

    def test_zero_flexibility_with_noise_has_a_random_one_step_displacement(self):
        model, parameters = build_case(Xi=0.0)
        paths = solve_small_case(model, parameters, np.array([0.0, 0.01]), 0.4)
        self.assertGreater(np.var(paths[:, -1]), 0.0)
        self.assertFalse(np.all(paths[:, -1] == 0.5))

    def test_em_one_step_uses_sqrt_time_and_existing_bm_drift(self):
        model, parameters = build_case()
        h, sigma_x, seed, n_paths = (0.01, 0.4, 12, 5)
        paths = solve_small_case(model, parameters, np.array([0.0, h]), sigma_x, n_paths=n_paths, seed=seed)
        xi = np.random.default_rng(seed).standard_normal(n_paths)
        expected = 0.5 + h * model.dXdt(0.0, [0.5], *parameters)[0] + model.diffusion_coefficient(0.5,
            sigma_x) * np.sqrt(h) * xi
        np.testing.assert_array_equal(paths[:, -1], expected)

    def test_hours_and_days_give_the_same_em_states(self):
        hours = np.linspace(0.0, 2.0, 17)
        model_h, parameters_h = build_case(C=24.0, T=(0.0, 2.0))
        model_d, parameters_d = build_case(C=1.0, T=(0.0, 2.0 / 24.0))
        paths_h = solve_small_case(model_h, parameters_h, hours, 0.1)
        paths_d = solve_small_case(model_d, parameters_d, hours / 24.0, np.sqrt(24.0) * 0.1)
        np.testing.assert_allclose(paths_h, paths_d, atol=2e-15, rtol=0.0)

    def test_price_intervals_are_right_continuous_over_the_full_demo_horizon(self):
        price = input_builder.price_intervals((np.array([0.0, 0.5, 4.0]), np.array([0.75, 0.25])))
        query = np.array([0.0, np.nextafter(0.5, 0.0), 0.5, np.nextafter(0.5, 1.0), 4.0])
        np.testing.assert_array_equal(price(query), [0.75, 0.75, 0.25, 0.25, 0.25])

def f2_em_linear_response(x):
    """Canonical benchmark response f(x)=g(x)=1-2x."""
    return 1.0 - 2.0 * np.asarray(x, dtype=float)

def build_model(price_edges=None, price_values=None, Xi=1.0, t_steps=5):
    T = (0.0, 1.0)
    B = input_builder.energy((np.array(T), np.array([0.3, 0.3])))
    if price_edges is None:
        price_edges = np.array(T)
        price_values = np.array([0.25])
    U = input_builder.price_intervals((price_edges, price_values))
    parameters = [U, f2_em_linear_response, f2_em_linear_response, 1.0, 2.0, Xi]
    return (bm_1(B, B, T, t_steps), parameters)

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

class BM1EulerReferenceTests(unittest.TestCase):

    def test_output_grid_is_separate_from_internal_integration_grid(self):
        model, parameters = build_model(t_steps=5)
        t_eval = np.array([0.0, 0.5, 1.0])
        result = model.euler_maruyama_solver([0.5], parameters, 0.0, boundary='none', t_eval=t_eval, max_step=0.1,
            return_diagnostics=True)
        t, state, *_, diagnostics = result
        np.testing.assert_array_equal(t, t_eval)
        self.assertEqual(state.shape, t_eval.shape)
        self.assertGreater(diagnostics['integration_t'].size, t_eval.size)
        np.testing.assert_array_equal(diagnostics['integration_t'][diagnostics['output_indices']], t_eval)
        self.assertLessEqual(np.max(np.diff(diagnostics['integration_t'])), 0.1 + 1e-15)

    def test_unaligned_price_breakpoint_is_an_internal_em_boundary(self):
        step = 0.37
        model, parameters = build_model(price_edges=np.array([0.0, step, 1.0]), price_values=np.array([0.75, 0.25]),
            t_steps=3)
        t_eval = np.array([0.0, 0.5, 1.0])
        integration_t, _ = model.euler_maruyama.build_time_grid(t_eval, breakpoints=np.array([step]))
        dW = np.zeros((1, integration_t.size - 1))
        result = model.euler_maruyama_solver([0.5], parameters, 0.0, boundary='none', t_eval=t_eval,
            breakpoints=np.array([step]), brownian_increments=dW, return_diagnostics=True)
        state = result[1]
        diagnostics = result[-1]
        self.assertTrue(np.any(diagnostics['integration_t'] == step))
        manual = [0.5]
        internal_states = [0.5]
        for left, right in zip(integration_t[:-1], integration_t[1:]):
            x = internal_states[-1]
            x = x + (right - left) * model._drift(left, np.array([x]), parameters)[0]
            internal_states.append(x)
        internal_states = np.asarray(internal_states)
        output_indices = np.searchsorted(integration_t, t_eval)
        np.testing.assert_allclose(state, internal_states[output_indices], atol=1e-15)

    def test_default_output_grid_remains_the_model_grid(self):
        model, parameters = build_model(t_steps=7)
        legacy = model.euler_maruyama_solver([0.5], parameters, 0.0, n_paths=2, seed=4, boundary='none')
        explicit = model.euler_maruyama_solver([0.5], parameters, 0.0, n_paths=2, seed=4, boundary='none',
            t_eval=model.t, max_step=np.inf)
        for left, right in zip(legacy, explicit):
            np.testing.assert_array_equal(left, right)

    def test_vector_initial_states_are_allowed_for_em_only(self):
        model, parameters = build_model(Xi=0.0, t_steps=3)
        initial = np.array([0.2, 0.5, 0.8])
        result = model.euler_maruyama_solver(initial, parameters, 0.0, n_paths=3, boundary='none',
            brownian_increments=np.zeros((3, 2)))
        np.testing.assert_array_equal(result[1], np.tile(initial[:, None], (1, 3)))
        with self.assertRaises(ValueError):
            bm_1_validation.stochastic_inputs(initial, parameters, 0.0)

    def test_em_output_grid_must_span_the_model_interval(self):
        model, parameters = build_model()
        for t_eval in (np.array([0.1, 1.0]), np.array([0.0, 0.9])):
            with self.subTest(t_eval=t_eval), self.assertRaisesRegex(ValueError, 'span'):
                model.euler_maruyama_solver([0.5], parameters, 0.0, boundary='none', t_eval=t_eval)

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


if __name__ == '__main__':
    unittest.main()
