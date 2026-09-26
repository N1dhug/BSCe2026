"""F2-C regression tests for the pure 1D finite-volume Fokker-Planck reference."""

import unittest

import numpy as np
from scipy.special import betainc

from src.model.bm_1 import bm_1
from src.solver.finite_volume import finite_volume_1d
from src.tools.input_generator.input_builder import input_builder


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

        lower, diagonal, upper, diagnostics = self.solver.generator_tridiagonal(
            dx,
            drift_faces,
            diffusion,
        )

        generator = (
            np.diag(diagonal)
            + np.diag(lower, k=-1)
            + np.diag(upper, k=1)
        )
        np.testing.assert_allclose(
            np.sum(generator, axis=0),
            0.0,
            atol=2.0e-14,
            rtol=0.0,
        )
        self.assertGreaterEqual(np.min(lower), 0.0)
        self.assertGreaterEqual(np.min(upper), 0.0)
        self.assertLessEqual(np.max(diagonal), 0.0)
        self.assertLessEqual(diagnostics["maximum_abs_column_sum"], 2.0e-14)
        self.assertGreaterEqual(diagnostics["minimum_off_diagonal_rate"], 0.0)

    def test_zero_operator_leaves_cell_mass_unchanged(self):
        initial_mass = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        result = self.solver.solve(
            np.array([0.0, 0.3, 1.0]),
            initial_mass,
            lambda time, x: np.zeros_like(x),
            lambda time, x: np.zeros_like(x),
            max_step=0.1,
        )
        np.testing.assert_allclose(
            result["cell_mass"],
            np.tile(initial_mass, (3, 1)),
            atol=1.0e-15,
            rtol=0.0,
        )
        np.testing.assert_allclose(result["mass"], 1.0, atol=1.0e-15, rtol=0.0)

    def test_uniform_density_is_stationary_for_reflecting_constant_diffusion(self):
        x_steps = 41
        initial_mass = np.full(x_steps, 1.0 / x_steps)
        diffusion = np.sqrt(0.2)
        result = self.solver.solve(
            np.array([0.0, 0.5]),
            initial_mass,
            lambda time, x: np.zeros_like(x),
            lambda time, x: diffusion * np.ones_like(x),
            max_step=0.01,
        )
        np.testing.assert_allclose(
            result["cell_mass"][-1],
            initial_mass,
            atol=2.0e-14,
            rtol=0.0,
        )
        self.assertGreaterEqual(result["diagnostics"]["minimum_cell_mass"], 0.0)
        self.assertFalse(result["diagnostics"]["used_clipping"])
        self.assertFalse(result["diagnostics"]["used_renormalization"])

    def test_heat_equation_benchmark_improves_under_spatial_refinement(self):
        final_time = 0.1
        diffusion_coefficient = 0.05
        amplitude = 0.5
        errors = []

        for x_steps in (40, 80):
            edges, _, dx = self.solver.grid(x_steps)
            initial_mass = (
                dx
                + amplitude
                / np.pi
                * (
                    np.sin(np.pi * edges[1:])
                    - np.sin(np.pi * edges[:-1])
                )
            )
            result = self.solver.solve(
                np.array([0.0, final_time]),
                initial_mass,
                lambda time, x: np.zeros_like(x),
                lambda time, x: np.sqrt(2.0 * diffusion_coefficient)
                * np.ones_like(x),
                max_step=5.0e-4,
            )

            exact_amplitude = amplitude * np.exp(
                -diffusion_coefficient * np.pi**2 * final_time
            )
            exact_mass = (
                dx
                + exact_amplitude
                / np.pi
                * (
                    np.sin(np.pi * edges[1:])
                    - np.sin(np.pi * edges[:-1])
                )
            )
            errors.append(np.sum(np.abs(result["cell_mass"][-1] - exact_mass)))
            self.assertLess(
                result["diagnostics"]["maximum_abs_mass_error"],
                2.0e-13,
            )
            self.assertGreaterEqual(
                result["diagnostics"]["minimum_cell_mass"],
                -1.0e-13,
            )

        self.assertLess(errors[1], errors[0])

    def test_solver_rejects_unnormalized_initial_mass(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            self.solver.solve(
                np.array([0.0, 1.0]),
                np.array([0.2, 0.2, 0.2]),
                lambda time, x: np.zeros_like(x),
                lambda time, x: np.zeros_like(x),
            )


class BM1FiniteVolumeReferenceTests(unittest.TestCase):
    def build_case(self):
        T = (0.0, 1.0)
        B = input_builder.energy((np.array(T), np.array([0.3, 0.3])))
        price_edges = np.array([0.0, 0.37, 1.0])
        U = input_builder.price_intervals(
            (price_edges, np.array([0.75, 0.25]))
        )
        parameters = [U, linear_response, linear_response, 1.0, 2.0, 1.0]
        return bm_1(B, B, T, 3), parameters, price_edges[1:-1]

    def test_bm1_reference_preserves_probability_without_post_correction(self):
        model, parameters, breakpoints = self.build_case()
        x_steps = 101
        edges, _, _ = model.fokker_planc_reference.grid(x_steps)
        initial_mass = np.diff(betainc(20.0, 20.0, edges))

        result = model.fokker_planc_reference_solver(
            initial_mass,
            parameters,
            0.5,
            t_eval=np.array([0.0, 0.5, 1.0]),
            breakpoints=breakpoints,
            max_step=1.0 / 96.0,
        )

        (
            t,
            x,
            density,
            cell_mass,
            mean,
            variance,
            price,
            baseline,
            expected_demand,
            diagnostics,
        ) = result

        np.testing.assert_array_equal(t, [0.0, 0.5, 1.0])
        self.assertEqual(density.shape, (3, x_steps))
        self.assertEqual(cell_mass.shape, (3, x_steps))
        np.testing.assert_allclose(np.sum(cell_mass, axis=1), 1.0, atol=2.0e-13)
        self.assertTrue(np.all(mean >= 0.0))
        self.assertTrue(np.all(mean <= 1.0))
        self.assertTrue(np.all(variance >= 0.0))
        self.assertTrue(np.all(np.isfinite(expected_demand)))
        self.assertFalse(diagnostics["used_clipping"])
        self.assertFalse(diagnostics["used_renormalization"])
        self.assertLess(diagnostics["maximum_abs_mass_error"], 2.0e-13)
        self.assertGreaterEqual(diagnostics["minimum_cell_mass"], -1.0e-13)
        self.assertTrue(np.any(diagnostics["integration_t"] == breakpoints[0]))
        self.assertEqual(price.shape, t.shape)
        self.assertEqual(baseline.shape, t.shape)
        self.assertEqual(x.shape, (x_steps,))


if __name__ == "__main__":
    unittest.main()
