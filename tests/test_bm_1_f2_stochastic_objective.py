"""F2-H tests for the stochastic BM1 tracking objective."""

import unittest

import numpy as np

from src.exe.run_model.bm_1.stochastic_objective_reference import (
    run_objective_reference,
)
from src.method.bm_1_stochastic_objective import (
    discrete_demand_moments,
    midpoint_quadrature,
    monte_carlo_tracking_objective,
    stochastic_tracking_objective,
)


class StochasticObjectiveTheoryTests(unittest.TestCase):
    def test_midpoint_quadrature_respects_breakpoint_and_step(self):
        breakpoint = 0.37
        edges, midpoints, widths, solver_t_eval = midpoint_quadrature(
            (0.0, 1.0),
            breakpoints=np.array([breakpoint]),
            max_step=0.1,
        )
        self.assertTrue(np.any(edges == breakpoint))
        self.assertLessEqual(np.max(widths), 0.1 + 1.0e-15)
        self.assertFalse(
            np.any(
                (edges[:-1] < breakpoint)
                & (edges[1:] > breakpoint)
            )
        )
        np.testing.assert_allclose(
            midpoints,
            0.5 * (edges[:-1] + edges[1:]),
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(solver_t_eval[0], 0.0)
        self.assertEqual(solver_t_eval[-1], 1.0)
        np.testing.assert_array_equal(solver_t_eval[1:-1], midpoints)

    def test_bias_variance_decomposition_matches_known_values(self):
        widths = np.array([0.5, 0.5])
        expected = np.array([1.0, 2.0])
        variance = np.array([0.25, 0.50])
        second = np.square(expected) + variance
        target = np.array([0.0, 1.0])

        result = stochastic_tracking_objective(
            widths,
            expected,
            second,
            target,
        )
        self.assertAlmostEqual(result.bias, 1.0)
        self.assertAlmostEqual(result.variance, 0.375)
        self.assertAlmostEqual(result.total, 1.375)
        np.testing.assert_allclose(
            result.total_integrand,
            np.array([1.25, 1.50]),
            atol=1.0e-15,
        )

    def test_zero_variance_reduces_to_deterministic_squared_tracking(self):
        widths = np.array([0.2, 0.3, 0.5])
        demand = np.array([0.2, 0.4, 0.7])
        target = np.array([0.3, 0.3, 0.5])

        result = stochastic_tracking_objective(
            widths,
            demand,
            np.square(demand),
            target,
        )
        deterministic = float(
            np.dot(widths, np.square(demand - target))
        )
        self.assertAlmostEqual(result.total, deterministic, places=15)
        self.assertAlmostEqual(result.bias, deterministic, places=15)
        self.assertAlmostEqual(result.variance, 0.0, places=15)

    def test_discrete_demand_moments_match_direct_definition(self):
        demand = np.array(
            [
                [0.0, 0.5, 1.0],
                [0.2, 0.4, 0.8],
            ]
        )
        mass = np.array(
            [
                [0.2, 0.5, 0.3],
                [0.1, 0.6, 0.3],
            ]
        )
        expected, second, variance = discrete_demand_moments(
            demand,
            mass,
        )
        np.testing.assert_allclose(
            expected,
            np.sum(demand * mass, axis=1),
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            second,
            np.sum(np.square(demand) * mass, axis=1),
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            variance,
            second - np.square(expected),
            atol=1.0e-15,
        )

    def test_mc_direct_and_decomposed_objectives_are_identical(self):
        widths = np.array([0.25, 0.75])
        demand_paths = np.array(
            [
                [0.2, 0.4],
                [0.3, 0.7],
                [0.5, 0.6],
                [0.4, 0.8],
            ]
        )
        target = np.array([0.3, 0.5])

        result = monte_carlo_tracking_objective(
            widths,
            demand_paths,
            target,
        )
        direct_paths = (
            np.square(demand_paths - target[None, :]) @ widths
        )
        self.assertAlmostEqual(
            result.objective.total,
            np.mean(direct_paths),
            places=15,
        )
        self.assertAlmostEqual(
            result.direct_total,
            result.objective.total,
            places=15,
        )
        np.testing.assert_allclose(
            result.path_objectives,
            direct_paths,
            atol=1.0e-15,
        )
        self.assertLess(result.maximum_decomposition_error, 1.0e-15)
        self.assertGreater(result.standard_error, 0.0)


class StochasticObjectiveIntegrationTests(unittest.TestCase):
    def test_small_reference_evaluates_same_cost_across_methods(self):
        result = run_objective_reference(
            n_paths=2_000,
            degrees=(20,),
            x_steps=51,
            objective_max_step=1.0 / 24.0,
            solver_max_step=1.0 / 96.0,
            cfl=0.35,
            initial_seed=91,
            brownian_seed=92,
        )

        mc = result["mc"]
        fv = result["fv"]
        mixture = result["mixture"][20]

        self.assertEqual(result["em_diagnostics"]["total_violations"], 0)
        self.assertLess(mc.maximum_decomposition_error, 1.0e-12)
        self.assertLess(
            abs(mc.direct_total - mc.objective.total),
            1.0e-12,
        )
        for objective in (mc.objective, fv, mixture):
            self.assertTrue(np.isfinite(objective.total))
            self.assertGreaterEqual(objective.total, 0.0)
            self.assertGreaterEqual(objective.bias, 0.0)
            self.assertGreaterEqual(objective.variance, 0.0)
            self.assertAlmostEqual(
                objective.total,
                objective.bias + objective.variance,
                places=12,
            )

        self.assertLess(abs(mc.objective.total - fv.total), 0.05)
        self.assertLess(abs(mixture.total - fv.total), 0.05)


if __name__ == "__main__":
    unittest.main()
