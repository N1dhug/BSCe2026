"""F2-B regression tests for the controlled Euler-Maruyama reference.

Run from the repository root:
    python -B -m unittest discover -s tests -p "test_bm_1_f2_em.py" -v

These tests cover time-grid construction, supplied/coupled Wiener increments,
raw boundary diagnostics and the bm_1 EM wrapper. They do not validate the
Fokker-Planck or beta-mixture implementations.
"""

import unittest

import numpy as np

from src.method.euler_maruyama import euler_maruyama
from src.model.bm_1 import bm_1
from src.tools.input_generator.input_builder import input_builder
from src.validation.model.bm_1 import bm_1_validation


def linear_response(x):
    """Canonical benchmark response f(x)=g(x)=1-2x."""
    return 1.0 - 2.0 * np.asarray(x, dtype=float)


def build_model(price_edges=None, price_values=None, Xi=1.0, t_steps=5):
    T = (0.0, 1.0)
    B = input_builder.energy((np.array(T), np.array([0.3, 0.3])))
    if price_edges is None:
        price_edges = np.array(T)
        price_values = np.array([0.25])
    U = input_builder.price_intervals((price_edges, price_values))
    parameters = [U, linear_response, linear_response, 1.0, 2.0, Xi]
    return bm_1(B, B, T, t_steps), parameters


class TimeGridTests(unittest.TestCase):
    def setUp(self):
        self.method = euler_maruyama()

    def test_grid_contains_outputs_breakpoints_and_respects_max_step(self):
        t_eval = np.array([0.0, 0.4, 1.0])
        integration_t, output_indices = self.method.build_time_grid(
            t_eval, breakpoints=np.array([0.55]), max_step=0.1
        )
        np.testing.assert_array_equal(integration_t[output_indices], t_eval)
        self.assertTrue(np.any(integration_t == 0.55))
        self.assertLessEqual(np.max(np.diff(integration_t)), 0.1 + 1.0e-15)

    def test_no_extra_controls_preserve_the_output_grid(self):
        t_eval = np.array([0.0, 0.2, 0.7, 1.0])
        integration_t, output_indices = self.method.build_time_grid(t_eval)
        np.testing.assert_array_equal(integration_t, t_eval)
        np.testing.assert_array_equal(output_indices, np.arange(t_eval.size))

    def test_invalid_grid_controls_are_rejected(self):
        cases = (
            (np.array([0.0, 0.5, 0.4]), None, np.inf),
            (np.array([0.0, 1.0]), np.array([0.0]), np.inf),
            (np.array([0.0, 1.0]), np.array([0.5, 0.4]), np.inf),
            (np.array([0.0, 1.0]), None, 0.0),
        )
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
        expected = (
            np.random.default_rng(12).standard_normal((3, 3)).T
            * np.sqrt(np.diff(t))[None, :]
        )
        np.testing.assert_array_equal(increments, expected)

    def test_supplied_increments_drive_the_exact_one_step_formula(self):
        t = np.array([0.0, 0.25])
        dW = np.array([[0.10], [-0.20], [0.05]])
        paths = self.method.solve_paths(
            t,
            0.5,
            lambda time, state: 2.0,
            lambda time, state: 3.0,
            n_paths=3,
            boundary="none",
            brownian_increments=dW,
        )
        expected = 0.5 + 2.0 * 0.25 + 3.0 * dW[:, 0]
        np.testing.assert_array_equal(paths[:, -1], expected)

    def test_seed_and_supplied_increments_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "seed must be None"):
            self.method.solve_paths(
                np.array([0.0, 1.0]),
                0.5,
                lambda time, state: 0.0,
                lambda time, state: 1.0,
                n_paths=1,
                seed=1,
                boundary="none",
                brownian_increments=np.array([[0.0]]),
            )

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

        fine_paths = self.method.solve_paths(
            fine_t, 0.5, drift, diffusion, boundary="none",
            brownian_increments=fine_dW,
        )
        coarse_paths = self.method.solve_paths(
            coarse_t, 0.5, drift, diffusion, boundary="none",
            brownian_increments=coarse_dW,
        )
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
        paths, diagnostics = self.method.solve_paths(
            self.t,
            self.initial,
            self.drift,
            self.diffusion,
            n_paths=2,
            boundary="reflect",
            brownian_increments=self.dW,
            return_diagnostics=True,
        )
        np.testing.assert_allclose(paths[:, -1], [0.1, 0.9], atol=1e-15)
        self.assertAlmostEqual(diagnostics["minimum_raw_proposal"], -0.1)
        self.assertAlmostEqual(diagnostics["maximum_raw_proposal"], 1.1)
        self.assertEqual(diagnostics["lower_violations"], 1)
        self.assertEqual(diagnostics["upper_violations"], 1)
        self.assertEqual(diagnostics["total_violations"], 2)
        self.assertEqual(diagnostics["violating_paths"], 2)
        self.assertEqual(diagnostics["boundary_corrections"], 2)

    def test_boundary_none_leaves_the_same_raw_proposals_uncorrected(self):
        paths, diagnostics = self.method.solve_paths(
            self.t,
            self.initial,
            self.drift,
            self.diffusion,
            n_paths=2,
            boundary="none",
            brownian_increments=self.dW,
            return_diagnostics=True,
        )
        np.testing.assert_allclose(paths[:, -1], [-0.1, 1.1], atol=1e-15)
        self.assertEqual(diagnostics["total_violations"], 2)
        self.assertEqual(diagnostics["boundary_corrections"], 0)

    def test_one_initial_state_per_path_is_supported(self):
        paths = self.method.solve_paths(
            np.array([0.0, 0.5, 1.0]),
            np.array([0.2, 0.8]),
            self.drift,
            lambda time, state: np.zeros_like(state),
            n_paths=2,
            boundary="none",
            brownian_increments=np.zeros((2, 2)),
        )
        np.testing.assert_array_equal(paths[0], [0.2, 0.2, 0.2])
        np.testing.assert_array_equal(paths[1], [0.8, 0.8, 0.8])


class BM1EulerReferenceTests(unittest.TestCase):
    def test_output_grid_is_separate_from_internal_integration_grid(self):
        model, parameters = build_model(t_steps=5)
        t_eval = np.array([0.0, 0.5, 1.0])
        result = model.euler_maruyama_solver(
            [0.5], parameters, 0.0,
            boundary="none",
            t_eval=t_eval,
            max_step=0.1,
            return_diagnostics=True,
        )
        t, state, *_, diagnostics = result
        np.testing.assert_array_equal(t, t_eval)
        self.assertEqual(state.shape, t_eval.shape)
        self.assertGreater(diagnostics["integration_t"].size, t_eval.size)
        np.testing.assert_array_equal(
            diagnostics["integration_t"][diagnostics["output_indices"]], t_eval
        )
        self.assertLessEqual(np.max(np.diff(diagnostics["integration_t"])), 0.1 + 1e-15)

    def test_unaligned_price_breakpoint_is_an_internal_em_boundary(self):
        step = 0.37
        model, parameters = build_model(
            price_edges=np.array([0.0, step, 1.0]),
            price_values=np.array([0.75, 0.25]),
            t_steps=3,
        )
        t_eval = np.array([0.0, 0.5, 1.0])
        integration_t, _ = model.euler_maruyama.build_time_grid(
            t_eval, breakpoints=np.array([step])
        )
        dW = np.zeros((1, integration_t.size - 1))

        result = model.euler_maruyama_solver(
            [0.5], parameters, 0.0,
            boundary="none",
            t_eval=t_eval,
            breakpoints=np.array([step]),
            brownian_increments=dW,
            return_diagnostics=True,
        )
        state = result[1]
        diagnostics = result[-1]
        self.assertTrue(np.any(diagnostics["integration_t"] == step))

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
        legacy = model.euler_maruyama_solver(
            [0.5], parameters, 0.0, n_paths=2, seed=4, boundary="none"
        )
        explicit = model.euler_maruyama_solver(
            [0.5], parameters, 0.0, n_paths=2, seed=4, boundary="none",
            t_eval=model.t, max_step=np.inf,
        )
        for left, right in zip(legacy, explicit):
            np.testing.assert_array_equal(left, right)

    def test_vector_initial_states_are_allowed_for_em_only(self):
        model, parameters = build_model(Xi=0.0, t_steps=3)
        initial = np.array([0.2, 0.5, 0.8])
        result = model.euler_maruyama_solver(
            initial, parameters, 0.0,
            n_paths=3,
            boundary="none",
            brownian_increments=np.zeros((3, 2)),
        )
        np.testing.assert_array_equal(result[1], np.tile(initial[:, None], (1, 3)))
        with self.assertRaises(ValueError):
            bm_1_validation.stochastic_inputs(initial, parameters, 0.0)

    def test_em_output_grid_must_span_the_model_interval(self):
        model, parameters = build_model()
        for t_eval in (np.array([0.1, 1.0]), np.array([0.0, 0.9])):
            with self.subTest(t_eval=t_eval), self.assertRaisesRegex(ValueError, "span"):
                model.euler_maruyama_solver(
                    [0.5], parameters, 0.0,
                    boundary="none", t_eval=t_eval,
                )


if __name__ == "__main__":
    unittest.main()
