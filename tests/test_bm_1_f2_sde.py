"""F2-A regression tests for the BM1 model contract (no figures or data files).

Run from the repository root:
    python -B -m unittest discover -s tests -p "test_bm_1_f2_sde.py" -v

These are model/entry-validation tests, not a stochastic convergence study.
The existing EM method is exercised with boundary="none" on small test steps.
"""

import unittest

import numpy as np

from src.method.euler_maruyama import euler_maruyama
from src.model.base.bm import bm
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
    return bm(B, B, T, 3), [U, linear_response, linear_response, C, 2.0, Xi]


def solve_small_case(model, parameters, t, sigma_x, x0=0.5, n_paths=4, seed=7):
    """Exercise the actual BM drift and generic EM method without corrections."""
    initial, parameters, sigma_x = bm_1_validation.stochastic_inputs(
        [x0], parameters, sigma_x
    )

    def drift(time, states):
        return np.array([
            model.dXdt(time, [state], *parameters)[0] for state in states
        ])

    return euler_maruyama().solve_paths(
        t, initial, drift,
        lambda time, states: model.diffusion_coefficient(states, sigma_x),
        n_paths=n_paths, seed=seed, boundary="none",
    )


class StochasticContractTests(unittest.TestCase):
    def setUp(self):
        self.model, self.parameters = build_case()

    def test_valid_input_preserves_parameter_values_and_callables(self):
        original = self.parameters.copy()
        x0, parameters, sigma_x = bm_1_validation.stochastic_inputs(
            [0.5], self.parameters, 0.4
        )
        np.testing.assert_array_equal(x0, [0.5])
        self.assertEqual(sigma_x, 0.4)
        self.assertIsNot(parameters, self.parameters)
        for index in range(3):
            self.assertIs(parameters[index], original[index])
        self.assertEqual(parameters[3:], original[3:])
        self.assertEqual(self.parameters, original)

    def test_zero_process_noise_is_valid(self):
        self.assertEqual(
            bm_1_validation.stochastic_inputs([0.5], self.parameters, 0.0)[2], 0.0
        )

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
            initial, _, _ = bm_1_validation.stochastic_inputs(
                [x0], self.parameters, 0.4
            )
            self.assertEqual(initial[0], x0)

    def test_wrong_endpoints_are_rejected_for_f_and_g(self):
        invalid = (
            lambda x: 0.5 - np.asarray(x),
            lambda x: 2.0 * np.asarray(x) - 1.0,
            lambda x: np.zeros_like(np.asarray(x)),
        )
        for index, name in ((1, "f"), (2, "g")):
            for response in invalid:
                parameters = self.parameters.copy()
                parameters[index] = response
                with self.subTest(index=index, response=response):
                    with self.assertRaisesRegex(ValueError, f"F2 requires {name}"):
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
        response = lambda x: linear_response(x) + 2.0e-12
        f, g = bm_validation.response_endpoints(response, linear_response)
        self.assertIs(f, response)
        self.assertIs(g, linear_response)
        self.assertGreater(f(0.0), 1.0)

    def test_existing_pchip_builder_satisfies_endpoint_contract(self):
        for kappa in (1.0, 5.0, 10.0):
            response = input_builder.md(*input_builder.md_vals(kappa, 10))
            f, _ = bm_validation.response_endpoints(response, response)
            self.assertIs(f, response)
            # A sampled check of the supplied builder, not a proof for all callables.
            values = response(np.linspace(0.0, 1.0, 201))
            self.assertTrue(np.all(np.diff(values) <= 1.0e-12))

    def test_deterministic_entry_is_not_given_the_new_endpoint_restriction(self):
        parameters = self.parameters.copy()
        parameters[1] = lambda x: 0.5 - np.asarray(x)
        _, validated = bm_1_validation.model_inputs([0.5], parameters)
        self.assertIs(validated[1], parameters[1])
        # This same noncanonical response must fail at the stochastic entry.
        with self.assertRaises(ValueError):
            bm_1_validation.stochastic_inputs([0.5], parameters, 0.0)

    def test_endpoint_check_does_not_claim_global_range_validation(self):
        response = lambda x: linear_response(x) + 10.0 * np.asarray(x) * (1.0 - np.asarray(x))
        bm_validation.response_endpoints(response, linear_response)
        parameters = self.parameters.copy()
        parameters[1] = response
        with self.assertRaisesRegex(ValueError, "f\\(X\\)"):
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
            np.testing.assert_array_equal(
                model.diffusion_coefficient(np.array([0.0, 1.0]), sigma_x), [0.0, 0.0]
            )

    def test_diffusion_is_the_amplitude_not_its_square(self):
        model, _ = build_case()
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(
            model.diffusion_coefficient(x, 0.8), [0.0, 0.15, 0.2, 0.15, 0.0],
            atol=1.0e-15, rtol=0.0,
        )

    def test_diffusion_preserves_array_shape(self):
        model, _ = build_case()
        states = np.full((3, 4), 0.5)
        self.assertEqual(model.diffusion_coefficient(states, 0.4).shape, states.shape)

    def test_zero_noise_returns_zero_diffusion(self):
        model, _ = build_case()
        np.testing.assert_array_equal(
            model.diffusion_coefficient(np.linspace(0.0, 1.0, 9), 0.0), np.zeros(9)
        )

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
        self.assertEqual(
            model.dXdt(0.0, [0.4], *parameters),
            model.dXdt(0.0, [0.4], *parameters, stochastic=True),
        )


class EulerLimitAndUnitsTests(unittest.TestCase):
    def test_zero_noise_em_matches_the_euler_recursion(self):
        model, parameters = build_case()
        t = np.linspace(0.0, 0.1, 9)
        paths = solve_small_case(model, parameters, t, sigma_x=0.0)
        expected = [0.5]
        for left, h in zip(t[:-1], np.diff(t)):
            expected.append(expected[-1] + h * model.dXdt(left, [expected[-1]], *parameters)[0])
        np.testing.assert_allclose(paths, np.tile(expected, (4, 1)), atol=1.0e-15, rtol=0.0)

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
        h, sigma_x, seed, n_paths = 0.01, 0.4, 12, 5
        paths = solve_small_case(model, parameters, np.array([0.0, h]), sigma_x, n_paths=n_paths, seed=seed)
        xi = np.random.default_rng(seed).standard_normal(n_paths)
        expected = (0.5 + h * model.dXdt(0.0, [0.5], *parameters)[0]
                    + model.diffusion_coefficient(0.5, sigma_x) * np.sqrt(h) * xi)
        np.testing.assert_array_equal(paths[:, -1], expected)

    def test_hours_and_days_give_the_same_em_states(self):
        hours = np.linspace(0.0, 2.0, 17)
        model_h, parameters_h = build_case(C=24.0, T=(0.0, 2.0))
        model_d, parameters_d = build_case(C=1.0, T=(0.0, 2.0 / 24.0))
        paths_h = solve_small_case(model_h, parameters_h, hours, 0.1)
        paths_d = solve_small_case(model_d, parameters_d, hours / 24.0, np.sqrt(24.0) * 0.1)
        np.testing.assert_allclose(paths_h, paths_d, atol=2.0e-15, rtol=0.0)

    def test_price_intervals_are_right_continuous_over_the_full_demo_horizon(self):
        price = input_builder.price_intervals((np.array([0.0, 0.5, 4.0]), np.array([0.75, 0.25])))
        query = np.array([0.0, np.nextafter(0.5, 0.0), 0.5, np.nextafter(0.5, 1.0), 4.0])
        np.testing.assert_array_equal(price(query), [0.75, 0.75, 0.25, 0.25, 0.25])


if __name__ == "__main__":
    unittest.main()
