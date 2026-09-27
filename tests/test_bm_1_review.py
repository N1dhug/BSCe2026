"""Compatibility and safety tests for the F1/F2 consolidation."""

import ast
import importlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from src.exe.run_model.bm_1.f2_reference_case import (
    beta_initial_cell_mass, build_reference_case, sample_beta_initial_states,
)
from src.method.bm_1_beta_mixture import evolve_projected_density
from src.method.bm_1_stochastic_optimization import (
    DynamicMixturePriceEvaluator, FVStochasticPriceEvaluator, multistart_powell,
)


class BMReviewCompatibilityTests(unittest.TestCase):
    def test_executable_imports_do_not_run_simulations(self):
        code = """
from unittest.mock import patch
from importlib import import_module
import matplotlib.pyplot  # Finish pyplot setup before patching Figure.savefig.
with patch('src.model.bm_1.bm_1.euler_maruyama_solver', side_effect=AssertionError('EM at import')), \\
     patch('src.model.bm_1.bm_1.fokker_planc_reference_solver', side_effect=AssertionError('FV at import')), \\
     patch('matplotlib.figure.Figure.savefig', side_effect=AssertionError('figure at import')):
    for name in ('em_reference', 'fp_reference', 'em_convergence', 'em_price_step', 'fp_price_step'):
        import_module('src.exe.run_model.bm_1.' + name)
"""
        result = subprocess.run(
            [sys.executable, '-B', '-c', code], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1], timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')

    def test_legacy_helper_imports_remain_aliases(self):
        dynamic = importlib.import_module('src.exe.run_model.bm_1.beta_dynamic_reference')
        optimizer = importlib.import_module('src.exe.run_model.bm_1.stochastic_optimization_reference')
        self.assertIs(dynamic._evolve_projected_density, evolve_projected_density)
        self.assertIs(optimizer.DynamicMixturePriceEvaluator, DynamicMixturePriceEvaluator)

    def test_reusable_methods_do_not_import_executables(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'src/method').glob('bm_1_*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('src.exe'), str(path))
                elif isinstance(node, ast.Import):
                    self.assertTrue(all(not item.name.startswith('src.exe') for item in node.names))

    def test_sampling_rejects_fractional_path_count(self):
        with self.assertRaises(ValueError):
            sample_beta_initial_states(2.5, seed=0)
        with self.assertRaises(TypeError):
            sample_beta_initial_states(True, seed=0)

    def test_beta_grid_rejects_nonunit_domain(self):
        with self.assertRaises(ValueError):
            beta_initial_cell_mass(np.array([-1.0e-12, 0.5, 1.0]))


class BMReviewEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.model, self.parameters = build_reference_case()
        self.edges = np.array([0.0, 0.5, 1.0])
        xedges, _, _ = self.model.fokker_planc_reference.grid(31)
        self.initial = beta_initial_cell_mass(xedges)

    def evaluator(self, **kwargs):
        return FVStochasticPriceEvaluator(
            self.model, self.parameters, self.edges, 0.5, self.initial,
            objective_max_step=0.1, solver_max_step=0.02, **kwargs,
        )

    def test_evaluator_owns_input_arrays(self):
        evaluator = self.evaluator()
        saved_mass = evaluator.initial_mass.copy()
        self.edges[1] = 0.7
        self.initial[:] = 0.0
        np.testing.assert_array_equal(evaluator.interval_edges, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(evaluator.initial_mass, saved_mass)

    def test_deterministic_midpoint_cost_handles_a_known_jump(self):
        from src.model.bm_1 import bm_1
        from src.method.bm_1_stochastic_objective import deterministic_midpoint_objective
        jump = 0.37
        baseline = lambda t: np.where(np.asarray(t) < jump, 0.3, 0.8)
        target = lambda t: np.full_like(np.asarray(t, dtype=float), 0.1)
        model = bm_1(baseline, target, (0.0, 1.0), 11)
        parameters = list(self.parameters)
        parameters[5] = 0.0  # No flexibility: demand equals the known baseline.
        result = deterministic_midpoint_objective(
            model, [0.5], parameters, objective_max_step=0.1,
            breakpoints=[jump], solver_max_step=0.1,
        )
        exact = jump * (0.3 - 0.1)**2 + (1.0 - jump) * (0.8 - 0.1)**2
        self.assertAlmostEqual(result.total, exact, places=14)
        self.assertEqual(result.variance, 0.0)

    def test_extra_target_and_baseline_breakpoints_are_respected(self):
        evaluator = self.evaluator(breakpoints=[0.37])
        np.testing.assert_array_equal(evaluator.breakpoints, [0.37, 0.5])
        self.assertIn(0.37, evaluator.objective_edges)
        self.assertFalse(np.any((evaluator.objective_edges[:-1] < 0.37)
                                & (evaluator.objective_edges[1:] > 0.37)))
        with self.assertRaises(ValueError):
            self.evaluator(breakpoints=[np.nan])

    def test_projection_setup_is_reused_between_price_evaluations(self):
        evaluator = DynamicMixturePriceEvaluator(self.evaluator(), degree=10)
        with patch('src.method.bm_1_beta_mixture._projection_matrices',
                   side_effect=AssertionError('repeated fixed Hessian setup')):
            left = evaluator.evaluate([0.7, 0.3])
            right = evaluator.evaluate([0.7, 0.3])
        self.assertEqual(left.objective.total, right.objective.total)
        np.testing.assert_array_equal(left.cell_mass, right.cell_mass)

    def test_mixture_rejects_noninteger_degree(self):
        with self.assertRaises(ValueError):
            DynamicMixturePriceEvaluator(self.evaluator(), degree=10.5)

    def test_exact_price_cache_does_not_change_powell_solution(self):
        calls = [0]
        def objective(prices):
            calls[0] += 1
            return float(np.sum((prices - np.array([0.2, 0.8])) ** 2))
        starts = [[0.1, 0.1], [0.9, 0.9]]
        messages = []
        cached = multistart_powell(objective, starts, progress=messages.append)
        cached_calls = calls[0]
        calls[0] = 0
        uncached = multistart_powell(objective, starts, cache=False)
        np.testing.assert_array_equal(cached.best.x, uncached.best.x)
        self.assertEqual(cached.best.fun, uncached.best.fun)
        self.assertLess(cached_calls, calls[0])
        self.assertEqual(cached.objective_evaluations, cached_calls)
        self.assertGreater(cached.cache_hits, 0)
        self.assertTrue(messages)

    def test_dynamic_core_rejects_invalid_probability(self):
        _, x, dx = self.model.fokker_planc_reference.grid(31)
        with self.assertRaises(ValueError):
            evolve_projected_density(
                method=self.model.fokker_planc, t_eval=np.array([0.0, 1.0]),
                breakpoints=None, x=x, dx=dx, basis=np.ones((1, x.size)),
                initial_coefficients=np.array([0.9]),
                drift=lambda t, x: np.zeros_like(x),
                diffusion=lambda t, x: np.zeros_like(x),
                max_step=0.1, cfl=0.4,
            )

    def test_raw_mass_loss_is_rejected_before_projection(self):
        _, x, dx = self.model.fokker_planc_reference.grid(31)
        with patch.object(self.model.fokker_planc, '_flux_rhs',
                          return_value=-np.ones_like(x)):
            with self.assertRaisesRegex(RuntimeError, 'probability conservation'):
                evolve_projected_density(
                    method=self.model.fokker_planc, t_eval=np.array([0.0, 1.0]),
                    breakpoints=None, x=x, dx=dx, basis=np.ones((1, x.size)),
                    initial_coefficients=np.array([1.0]),
                    drift=lambda t, x: np.zeros_like(x),
                    diffusion=lambda t, x: np.zeros_like(x),
                    max_step=0.1, cfl=0.4,
                )


if __name__ == '__main__':
    unittest.main()
