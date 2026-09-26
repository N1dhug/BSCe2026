"""F2-I tests for stochastic BM1 price optimization."""

import unittest

import numpy as np

from src.exe.run_model.bm_1.stochastic_optimization_reference import (
    DynamicMixturePriceEvaluator,
    INITIAL_PRICES,
    KNOWN_DETERMINISTIC_PRICES,
    _build_fv_evaluator,
    build_deterministic_target,
    build_stochastic_model,
    evaluate_mc_candidates_common_random_numbers,
    paired_mc_improvement,
)
from src.method.bm_1_stochastic_optimization import (
    best_surface_starts,
    coarse_price_surface,
    multistart_powell,
)


class PowellOptimizationTests(unittest.TestCase):
    def test_multistart_powell_recovers_bounded_quadratic_minimum(self):
        optimum = np.array([0.2, 0.8])

        def objective(prices):
            prices = np.asarray(prices, dtype=float)
            return float(np.sum(np.square(prices - optimum)))

        result = multistart_powell(
            objective,
            np.array(
                [
                    [0.1, 0.1],
                    [0.5, 0.5],
                    [0.9, 0.9],
                ]
            ),
            options={
                "xtol": 1.0e-8,
                "ftol": 1.0e-12,
                "maxiter": 100,
            },
        )
        self.assertTrue(result.best.success)
        np.testing.assert_allclose(
            result.best.x,
            optimum,
            atol=1.0e-6,
            rtol=0.0,
        )
        self.assertLess(result.best.fun, 1.0e-12)

    def test_coarse_surface_returns_best_price_starts(self):
        class Toy:
            n_intervals = 2

            def __call__(self, prices):
                p = np.asarray(prices, dtype=float)
                return float(
                    (p[0] - 0.25) ** 2
                    + (p[1] - 0.75) ** 2
                )

        values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        surface = coarse_price_surface(Toy(), values)
        starts = best_surface_starts(values, surface, 2)
        np.testing.assert_array_equal(starts[0], [0.25, 0.75])


class StochasticPriceEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target, _, _ = build_deterministic_target()
        cls.model = build_stochastic_model(target)

    def test_fv_evaluator_is_deterministic_and_rejects_invalid_prices(self):
        evaluator = _build_fv_evaluator(
            self.model,
            x_steps=41,
            objective_max_step=1.0 / 24.0,
            solver_max_step=1.0 / 96.0,
        )
        left = evaluator.evaluate(KNOWN_DETERMINISTIC_PRICES)
        right = evaluator.evaluate(KNOWN_DETERMINISTIC_PRICES)

        self.assertAlmostEqual(
            left.objective.total,
            right.objective.total,
            places=14,
        )
        self.assertGreater(left.objective.total, 0.0)
        self.assertAlmostEqual(
            left.objective.total,
            left.objective.bias + left.objective.variance,
            places=12,
        )
        self.assertLess(
            left.diagnostics["maximum_abs_mass_error"],
            1.0e-9,
        )

        with self.assertRaises(ValueError):
            evaluator.evaluate([0.5])
        with self.assertRaises(ValueError):
            evaluator.evaluate([-0.1, 0.5])

    def test_small_fv_optimization_does_not_worsen_reference_prices(self):
        evaluator = _build_fv_evaluator(
            self.model,
            x_steps=41,
            objective_max_step=1.0 / 12.0,
            solver_max_step=1.0 / 96.0,
        )
        known = evaluator(KNOWN_DETERMINISTIC_PRICES)
        surface = coarse_price_surface(
            evaluator,
            np.array([0.25, 0.5, 0.75]),
        )
        starts = best_surface_starts(
            np.array([0.25, 0.5, 0.75]),
            surface,
            2,
        )
        starts = np.vstack(
            (
                starts,
                KNOWN_DETERMINISTIC_PRICES,
                INITIAL_PRICES,
            )
        )
        result = multistart_powell(
            evaluator,
            starts,
            options={
                "xtol": 2.0e-3,
                "ftol": 1.0e-5,
                "maxiter": 25,
            },
        )

        self.assertTrue(result.best.success)
        self.assertLessEqual(result.best.fun, known + 1.0e-8)
        self.assertTrue(np.all(result.best.x >= 0.0))
        self.assertTrue(np.all(result.best.x <= 1.0))

    def test_small_dynamic_mixture_evaluator_returns_valid_objective(self):
        support = _build_fv_evaluator(
            self.model,
            x_steps=31,
            objective_max_step=1.0 / 12.0,
            solver_max_step=1.0 / 48.0,
        )
        evaluator = DynamicMixturePriceEvaluator(
            support,
            degree=10,
            cfl=0.35,
        )
        result = evaluator.evaluate(KNOWN_DETERMINISTIC_PRICES)

        self.assertTrue(np.isfinite(result.objective.total))
        self.assertGreater(result.objective.total, 0.0)
        self.assertAlmostEqual(
            result.objective.total,
            result.objective.bias + result.objective.variance,
            places=10,
        )
        self.assertFalse(result.diagnostics["used_clipping"])
        self.assertFalse(result.diagnostics["used_renormalization"])

    def test_common_random_numbers_support_paired_objective_difference(self):
        evaluator = _build_fv_evaluator(
            self.model,
            x_steps=31,
            objective_max_step=1.0 / 12.0,
            solver_max_step=1.0 / 48.0,
        )
        candidates = {
            "reference": KNOWN_DETERMINISTIC_PRICES,
            "initial": INITIAL_PRICES,
        }
        result = evaluate_mc_candidates_common_random_numbers(
            evaluator,
            candidates,
            n_paths=500,
            initial_seed=111,
            brownian_seed=112,
        )

        self.assertEqual(
            result["reference"].diagnostics["total_violations"],
            0,
        )
        self.assertEqual(
            result["initial"].diagnostics["total_violations"],
            0,
        )
        self.assertEqual(
            result["reference"].objective.path_objectives.shape,
            result["initial"].objective.path_objectives.shape,
        )
        improvement, standard_error = paired_mc_improvement(
            result["reference"],
            result["initial"],
        )
        self.assertTrue(np.isfinite(improvement))
        self.assertTrue(np.isfinite(standard_error))
        self.assertGreater(standard_error, 0.0)


if __name__ == "__main__":
    unittest.main()
