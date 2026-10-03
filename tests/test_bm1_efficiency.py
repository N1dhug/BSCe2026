"""Delivery optimizations must preserve numerical outputs and failure guards."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from src.control.bm_1 import (
    FVStochasticPriceEvaluator, FastFVStochasticPriceEvaluator, beta_initial_cell_mass,
)
from src.method.bm_1_estimation import BM1FVLikelihoodResult, run_fv_filter
from src.solver.finite_volume import finite_volume_1d_cached
from src.validation.bm_1 import estimation as core
from src.validation.bm_1.cases import build_reference_case


class LikelihoodStorageTests(unittest.TestCase):
    def setUp(self):
        self.case = core.build_f4_b_case()
        sequence = core.generate_synthetic_sequence(self.case, sequence_id=15011)
        self.model, self.parameters = self.case.build_model()
        self.model.fokker_planc_reference = finite_volume_1d_cached(vectorized_inputs=True)
        self.mass = core.beta_initial_mass(self.case, self.model, 31)
        self.observations = sequence.observations

    def replay(self, **kwargs):
        return run_fv_filter(self.model, self.parameters, .5, .03, self.mass, 0.,
            self.observations, breakpoints=self.case.breakpoints, max_step=1/96,
            likelihood_quadrature_order=3, **kwargs)

    def test_scores_final_state_and_all_step_diagnostics_match_full_replay(self):
        for missing in (False, True):
            mask = self.observations.observed.copy()
            mask[::3] = not missing
            self.observations = replace(self.observations, observed=mask, price_side='left')
            full, compact = self.replay(), self.replay(keep_history=False)
            self.assertIsInstance(compact, BM1FVLikelihoodResult)
            np.testing.assert_array_equal(compact.log_evidence, full.log_evidence)
            np.testing.assert_array_equal(compact.final_step.posterior_mass, full.posterior_mass[-1])
            np.testing.assert_array_equal(compact.final_step.predicted_mass, full.predicted_mass[-1])
            self.assertEqual(compact.total_log_likelihood, full.total_log_likelihood)
            self.assertEqual(compact.maximum_abs_mass_error, full.maximum_abs_mass_error)
            self.assertEqual(compact.minimum_cell_mass, full.minimum_cell_mass)
            self.assertEqual(compact.n_steps, full.integration_t.size-1)
            self.assertEqual(compact.actual_max_step, np.max(np.diff(full.integration_t)))
            self.assertFalse(compact.log_evidence.flags.writeable)
            self.assertFalse(compact.final_step.posterior_mass.flags.writeable)
            self.assertFalse(hasattr(compact, 'posterior_mass'))

    def test_all_missing_has_zero_likelihood_and_still_propagates(self):
        self.observations = replace(self.observations, observed=np.zeros(24, dtype=bool))
        compact = self.replay(keep_history=False)
        self.assertEqual(compact.total_log_likelihood, 0.)
        self.assertTrue(np.all(np.isnan(compact.log_evidence)))
        self.assertGreater(compact.n_steps, 0)

    def test_bad_option_and_later_numerical_failure_are_not_ignored(self):
        with self.assertRaises(TypeError):
            self.replay(keep_history='false')
        solver = self.model.fokker_planc_reference
        original = solver.solve
        calls = 0
        def fail_later(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError('numerical guard failed')
            return original(*args, **kwargs)
        with patch.object(solver, 'solve', side_effect=fail_later):
            with self.assertRaisesRegex(RuntimeError, 'numerical guard failed'):
                self.replay(keep_history=False)
        self.assertEqual(calls, 3)

    def test_existing_likelihood_candidate_uses_same_scores_and_diagnostics(self):
        full = self.replay()
        candidate = core.f4_e_truth_candidate(self.case)
        result = core._evaluate_f4_e_prepared(self.case, self.observations, candidate,
            model=self.model, base_parameters=self.parameters, initial_mass=self.mass,
            x_steps=31, max_step=1/96, likelihood_quadrature_order=3)
        self.assertEqual(result['total_log_likelihood'], full.total_log_likelihood)
        np.testing.assert_array_equal(result['log_evidence'], full.log_evidence)
        self.assertEqual(result['n_steps'], full.integration_t.size-1)


class TargetReuseTests(unittest.TestCase):
    def evaluator(self, cls):
        model, parameters = build_reference_case()
        model.fokker_planc_reference = finite_volume_1d_cached(vectorized_inputs=True)
        edges, _, _ = model.fokker_planc_reference.grid(31)
        return cls(model, parameters, [0., .5, 1.], .5, beta_initial_cell_mass(edges),
            objective_max_step=1/24, solver_max_step=1/96,
            terminal_reference=.5, terminal_weight=.2, price_smoothing_weight=.3)

    def test_target_sweep_matches_independent_evaluations_with_one_propagation(self):
        prices = [.7, .3]
        for cls in (FVStochasticPriceEvaluator, FastFVStochasticPriceEvaluator):
            evaluator = self.evaluator(cls)
            targets = np.array([np.full(evaluator.midpoints.size, a) for a in (.2, .5, .8)])
            independent = []
            for target in targets:
                evaluator.target = target.copy()
                independent.append(evaluator.evaluate(prices))
            with patch.object(evaluator, 'evaluate_cost', wraps=evaluator.evaluate_cost) as cost:
                shared = evaluator.evaluate_targets(prices, targets)
            self.assertEqual(cost.call_count, 1)
            for want, got in zip(independent, shared):
                self.assertEqual(got.total, want.total)
                self.assertEqual(got.penalties, want.penalties)
                np.testing.assert_array_equal(got.objective.total_integrand, want.objective.total_integrand)
            with patch.object(evaluator, 'evaluate_cost', wraps=evaluator.evaluate_cost) as cost:
                evaluator.evaluate_targets([.4, .6], targets)
            self.assertEqual(cost.call_count, 1)  # A new price plan requires a fresh propagation.

    def test_invalid_targets_fail_before_propagation_and_fast_path_avoids_history(self):
        evaluator = self.evaluator(FastFVStochasticPriceEvaluator)
        n = evaluator.midpoints.size
        for targets in (np.ones(n), np.ones((0, n)), np.ones((2, n+1)),
                        np.full((2, n), np.nan), np.ones((2, n), complex)*1j):
            with patch.object(evaluator, 'evaluate_cost', side_effect=AssertionError('propagation')):
                with self.assertRaises(ValueError):
                    evaluator.evaluate_targets([.7, .3], targets)
        with patch.object(evaluator, 'evaluate', side_effect=AssertionError('full history')):
            evaluator.evaluate_targets([.7, .3], np.ones((2, n))*.5)
