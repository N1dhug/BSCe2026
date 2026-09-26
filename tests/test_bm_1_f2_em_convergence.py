"""Regression tests for the F2-B coupled EM convergence diagnostics."""

import unittest

import numpy as np

from src.diagnostics.stochastic import (
    loglog_convergence_order,
    rms_strong_error,
    successive_convergence_orders,
)
from src.method.euler_maruyama import euler_maruyama


class ConvergenceDiagnosticTests(unittest.TestCase):
    def test_rms_strong_error_matches_definition(self):
        approximation = np.array([1.0, 2.0, 4.0])
        reference = np.array([0.0, 2.0, 2.0])
        expected = np.sqrt((1.0 + 0.0 + 4.0) / 3.0)
        self.assertAlmostEqual(
            rms_strong_error(approximation, reference),
            expected,
        )

    def test_known_power_law_recovers_its_order(self):
        step_sizes = np.array([1.0, 0.5, 0.25, 0.125])
        errors = 3.0 * step_sizes**0.5
        np.testing.assert_allclose(
            successive_convergence_orders(errors),
            [0.5, 0.5, 0.5],
            atol=1.0e-14,
        )
        self.assertAlmostEqual(
            loglog_convergence_order(step_sizes, errors),
            0.5,
            places=14,
        )

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
        n_paths = 2_000
        step_counts = np.array([24, 48, 96, 192], dtype=int)
        finest_steps = int(step_counts[-1])
        finest_t = np.linspace(0.0, 1.0, finest_steps + 1)
        fine_dW = method.brownian_increments(
            finest_t,
            n_paths=n_paths,
            seed=12345,
        )

        exact_terminal = 1.0 / (
            1.0
            + np.exp(
                -(
                    np.log(x0 / (1.0 - x0))
                    + sigma * np.sum(fine_dW, axis=1)
                )
            )
        )

        def drift(time, state):
            del time
            state = np.asarray(state, dtype=float)
            return (
                0.5
                * sigma**2
                * state
                * (1.0 - state)
                * (1.0 - 2.0 * state)
            )

        def diffusion(time, state):
            del time
            state = np.asarray(state, dtype=float)
            return sigma * state * (1.0 - state)

        errors = []
        for n_steps in step_counts:
            factor = finest_steps // int(n_steps)
            dW = (
                fine_dW
                if factor == 1
                else method.coarsen_brownian_increments(
                    fine_dW,
                    factor=factor,
                )
            )
            t = np.linspace(0.0, 1.0, int(n_steps) + 1)
            paths, diagnostics = method.solve_paths(
                t,
                x0,
                drift,
                diffusion,
                n_paths=n_paths,
                boundary="none",
                brownian_increments=dW,
                return_diagnostics=True,
            )
            self.assertEqual(diagnostics["total_violations"], 0)
            errors.append(
                rms_strong_error(paths[:, -1], exact_terminal)
            )

        errors = np.asarray(errors)
        self.assertTrue(np.all(np.diff(errors) < 0.0))
        order = loglog_convergence_order(
            1.0 / step_counts.astype(float),
            errors,
        )
        self.assertGreater(order, 0.35)
        self.assertLess(order, 0.70)


if __name__ == "__main__":
    unittest.main()
