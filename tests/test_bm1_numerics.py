"""BM1 regression tests grouped by responsibility."""

import numpy as np
import unittest
from src.control.bm_1 import BM1ControlNumerics
from src.validation.bm_1.control import build_f3_a_problem
from src.validation.bm_1.numerics import (_aggregate_brownian_to_grid, _decision_gap_refinement, _nested_em_time_grids,
    _prefix_statistics, fixed_candidates, run_coupled_mc_time_refinement, run_fv_time_refinement,
    run_quadrature_refinement, run_spatial_refinement, run_fixed_policy_validation,
    run_f3_f_high_resolution_validation)

class F3BNumericalReferenceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.problem, _ = build_f3_a_problem()

    def one_candidate(self):
        return {'deterministic_reference': np.asarray(self.problem.observed_prices, dtype=float).copy()}

    def test_fixed_candidates_are_owned_and_do_not_change_problem(self):
        candidates = fixed_candidates(self.problem)
        self.assertEqual(set(candidates), {'deterministic_reference', 'f3_a_stochastic', 'neutral'})
        candidates['deterministic_reference'][0] = 0.1
        self.assertAlmostEqual(self.problem.observed_prices[0], 0.72)

    def test_spatial_refinement_changes_only_spatial_resolution(self):
        rows = run_spatial_refinement(self.problem, self.one_candidate(), x_steps=(21, 31), objective_max_step=0.125,
            solver_max_step=0.0625)
        self.assertEqual([row['x_steps'] for row in rows], [21, 31])
        self.assertTrue(all((row['objective_max_step'] == 0.125 for row in rows)))
        self.assertTrue(all((row['solver_max_step'] == 0.0625 for row in rows)))
        self.assertEqual(rows[-1]['abs_total_difference_to_finest'], 0.0)
        self.assertTrue(all((np.isfinite(row['total']) for row in rows)))

    def test_fv_time_refinement_holds_objective_quadrature_fixed(self):
        rows = run_fv_time_refinement(self.problem, self.one_candidate(), x_steps=31, objective_max_step=0.25,
            solver_max_steps=(0.125, 0.0625))
        self.assertEqual([row['solver_max_step'] for row in rows], [0.125, 0.0625])
        self.assertTrue(all((row['objective_max_step'] == 0.25 for row in rows)))
        self.assertEqual(rows[-1]['abs_total_difference_to_finest'], 0.0)

    def test_quadrature_refinement_reuses_one_common_fv_propagation(self):
        rows = run_quadrature_refinement(self.problem, self.one_candidate(), x_steps=31,
            quadrature_max_steps=(0.25, 0.125), solver_max_step=0.0625)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row['common_output_points'] for row in rows}), 1)
        self.assertEqual(len({row['common_fv_steps'] for row in rows}), 1)
        self.assertEqual(rows[-1]['abs_total_difference_to_finest'], 0.0)
        self.assertNotEqual(rows[0]['objective_max_step'], rows[1]['objective_max_step'])

    def test_nested_em_grids_are_exact_subsets_and_respect_steps(self):
        t_eval = np.array([0.0, 0.13, 0.5, 0.77, 1.0])
        steps = (0.2, 0.1, 0.05)
        grids = _nested_em_time_grids(t_eval, np.array([0.33, 0.66]), steps)
        fine = grids[steps[-1]]
        for step in steps:
            grid = grids[step]
            self.assertLessEqual(np.max(np.diff(grid)), step + 2e-14)
            indices = np.searchsorted(fine, grid)
            np.testing.assert_array_equal(fine[indices], grid)
            output_indices = np.searchsorted(grid, t_eval)
            np.testing.assert_allclose(grid[output_indices], t_eval, atol=2e-14, rtol=0.0)

    def test_brownian_aggregation_sums_nested_fine_increments(self):
        fine_t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        fine = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, -0.5]])
        coarse = _aggregate_brownian_to_grid(fine_t, fine, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_array_equal(coarse, np.array([[3.0, 7.0], [-0.5, 1.5]]))
        with self.assertRaises(ValueError):
            _aggregate_brownian_to_grid(fine_t, fine, np.array([0.0, 0.6, 1.0]))

    def test_decision_gap_refinement_tracks_gap_not_absolute_cost(self):
        rows = [{'candidate': 'deterministic_reference', 'level': 1.0, 'total': 1.1},
            {'candidate': 'f3_a_stochastic', 'level': 1.0, 'total': 1.0},
            {'candidate': 'deterministic_reference', 'level': 2.0, 'total': 1.205},
            {'candidate': 'f3_a_stochastic', 'level': 2.0, 'total': 1.1}]
        result = _decision_gap_refinement(rows, 2.0, 1.0)
        self.assertAlmostEqual(result['previous_gap'], 0.1)
        self.assertAlmostEqual(result['finest_gap'], 0.105)
        self.assertAlmostEqual(result['last_gap_change'], 0.005)
        self.assertAlmostEqual(result['last_gap_change_over_finest_gap'], 0.005 / 0.105)

    def test_prefix_statistics_match_direct_sample_mean_and_se(self):
        values = np.arange(1.0, 9.0)
        rows = _prefix_statistics(values, (4, 8))
        self.assertAlmostEqual(rows[0]['total'], np.mean(values[:4]))
        self.assertAlmostEqual(rows[0]['standard_error'], np.std(values[:4], ddof=1) / np.sqrt(4))
        self.assertEqual(rows[-1]['difference_to_largest_prefix'], 0.0)

    def test_fixed_policy_validation_uses_supplied_reference_without_reoptimization(self):
        candidates = {
            'reference': np.array([0.73, 0.26]),
            'candidate': np.array([0.735, 0.264]),
        }
        numerics = BM1ControlNumerics(
            x_steps=31,
            objective_max_step=0.25,
            solver_max_step=0.0625,
        )
        result = run_fixed_policy_validation(
            self.problem,
            candidates,
            reference_name='reference',
            mc_paths=64,
            mc_time_paths=32,
            numerics=numerics,
            solver_max_steps=(0.125, 0.0625),
            objective_max_step=0.25,
            initial_seed=13,
            brownian_seed=14,
            mc_time_initial_seed=15,
            mc_time_brownian_seed=16,
        )
        self.assertEqual(result['reference_name'], 'reference')
        self.assertEqual(set(result['fv']), set(candidates))
        self.assertIn('candidate', result['fv_differences'])
        self.assertIn('candidate', result['paired_mc'])
        self.assertEqual(len(result['paired_mc_time']), 2)
        self.assertTrue(np.isfinite(result['paired_mc']['candidate']['reference_minus_candidate']))

    def test_f3_f_high_resolution_validation_adds_new_fv_reference_policy(self):
        candidates = {
            'fv_reference_policy': np.array([0.7306, 0.2644]),
            'mixture_policy': np.array([0.7359, 0.2636]),
        }
        numerics = BM1ControlNumerics(
            x_steps=31,
            objective_max_step=0.25,
            solver_max_step=0.0625,
        )
        result = run_f3_f_high_resolution_validation(
            self.problem,
            candidates,
            mc_paths=32,
            mc_time_paths=0,
            numerics=numerics,
            solver_max_steps=(0.125, 0.0625),
            progress=None,
        )
        self.assertIn('highres_fv_policy', result['candidates'])
        self.assertIn('highres_fv_optimization', result)
        self.assertTrue(result['highres_fv_optimization'].best.success)
        self.assertIn('mixture_policy', result['fv_differences'])
        self.assertIn('mixture_policy', result['paired_mc'])

    def test_small_coupled_mc_time_study_has_exact_zero_finest_difference(self):
        rows = run_coupled_mc_time_refinement(self.problem, self.one_candidate(), n_paths=64,
            solver_max_steps=(0.125, 0.0625), objective_max_step=0.25, initial_seed=5, brownian_seed=6)
        self.assertEqual(len(rows), 2)
        finest = rows[-1]
        self.assertEqual(finest['mean_path_cost_difference_to_finest'], 0.0)
        self.assertEqual(finest['rms_terminal_state_difference_to_finest'], 0.0)
        self.assertGreater(rows[1]['integration_steps'], rows[0]['integration_steps'])
        self.assertTrue(all((np.isfinite(row['total']) for row in rows)))


class AuditNumericalContractTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.problem, _ = build_f3_a_problem()
        cls.numerics = BM1ControlNumerics(x_steps=31, objective_max_step=.25, solver_max_step=.0625)
        cls.candidates = {'reference': np.array([.73, .264]), 'candidate': np.array([.736, .26])}

    def test_invalid_validation_counts_fail_before_price_optimization(self):
        from unittest.mock import patch
        for settings in ({'mc_paths': 1}, {'mc_paths': True}, {'mc_paths': 2.5},
                         {'mc_paths': np.nan}, {'mc_paths': 4, 'mc_time_paths': 1}):
            with self.subTest(settings=settings), \
                 patch('src.validation.bm_1.numerics.multistart_powell', side_effect=AssertionError('optimizer called')):
                with self.assertRaises((TypeError, ValueError)):
                    run_f3_f_high_resolution_validation(self.problem, self.candidates,
                        numerics=self.numerics, **settings)

    def test_reserved_reference_name_cannot_overwrite_optimizer_result(self):
        from unittest.mock import patch
        with patch('src.validation.bm_1.numerics.multistart_powell', side_effect=AssertionError('optimizer called')):
            with self.assertRaisesRegex(ValueError, 'reserved'):
                run_f3_f_high_resolution_validation(self.problem,
                    {'highres_fv_policy': [.5, .5]}, mc_paths=4, numerics=self.numerics)

    def test_nondefault_initial_law_requires_explicit_matching_sampler(self):
        from dataclasses import replace
        from src.control.bm_1 import point_initial_cell_mass
        problem = replace(self.problem, initial_mass_builder=lambda edges: point_initial_cell_mass(edges, .5))
        with self.assertRaisesRegex(ValueError, 'initial_sampler'):
            run_fixed_policy_validation(problem, self.candidates, reference_name='reference',
                mc_paths=8, numerics=self.numerics)
        result = run_fixed_policy_validation(problem, self.candidates, reference_name='reference',
            mc_paths=8, mc_time_paths=8, numerics=self.numerics,
            solver_max_steps=(.125, .0625), initial_sampler=lambda n, seed: np.full(n, .5))
        self.assertTrue(np.isfinite(result['paired_mc']['candidate']['reference_minus_candidate']))

    def test_time_refinement_includes_f3_penalties(self):
        from dataclasses import replace
        augmented = replace(self.problem, terminal_reference=.4, terminal_weight=.3, price_smoothing_weight=.2)
        kwargs = dict(n_paths=32, solver_max_steps=(.125, .0625), objective_max_step=.25,
                      initial_seed=451, brownian_seed=452, reference_name='reference')
        plain = run_coupled_mc_time_refinement(self.problem, self.candidates, **kwargs)
        full = run_coupled_mc_time_refinement(augmented, self.candidates, **kwargs)
        for old, new in zip(plain, full):
            self.assertEqual(old['tracking'], new['tracking'])
            self.assertGreater(new['penalty_total'], 0.)
            self.assertAlmostEqual(new['total'], old['total'] + new['penalty_total'], places=14)
            self.assertIn('paired_standard_error', new)
            self.assertLessEqual(new['actual_max_step'], new['solver_max_step'] + 1e-14)
        self.assertNotEqual(plain[-1]['reference_minus_candidate'], full[-1]['reference_minus_candidate'])
        self.assertEqual(full[-1]['paired_gap_change_to_finest'], 0.)
        self.assertEqual(full[-1]['se_paired_gap_change_to_finest'], 0.)

    def test_reported_actual_step_reflects_output_grid_and_nesting(self):
        rows = run_coupled_mc_time_refinement(self.problem, self.candidates, n_paths=8,
            solver_max_steps=(.125, .0625), objective_max_step=.025)
        self.assertLess(rows[0]['actual_max_step'], rows[0]['solver_max_step'])

    def test_time_refinement_rejects_raw_boundary_violations(self):
        from unittest.mock import patch
        outputs = (np.full((8, 6), .5), np.ones((8, 6)), {'total_violations': 1})
        with patch('src.validation.bm_1.numerics._em_outputs_on_integration_grid', return_value=outputs):
            with self.assertRaisesRegex(RuntimeError, 'boundary violations'):
                run_coupled_mc_time_refinement(self.problem, self.candidates, n_paths=8,
                    solver_max_steps=(.125, .0625), objective_max_step=.25)


    def test_final_mc_comparison_reports_full_penalized_costs(self):
        from dataclasses import replace
        from src.control.bm_1 import evaluate_mc_candidates_common_random_numbers
        from src.validation.bm_1.numerics import run_final_mc_comparison
        problem = replace(self.problem, terminal_reference=.4, terminal_weight=.3,
                          price_smoothing_weight=.2)
        result = run_final_mc_comparison(problem, self.candidates, n_paths=32,
            numerics=self.numerics, initial_seed=91, brownian_seed=92)
        evaluator = problem.build_evaluator(self.numerics)
        direct = evaluate_mc_candidates_common_random_numbers(evaluator, self.candidates,
            n_paths=32, initial_seed=91, brownian_seed=92)
        for row in result['rows']:
            name = row['candidate']
            self.assertAlmostEqual(row['mc_total'], direct[name].total, places=14)
            self.assertAlmostEqual(row['mc_standard_error'], direct[name].standard_error, places=14)
            self.assertAlmostEqual(row['fv_total'], evaluator.evaluate(self.candidates[name]).total, places=14)
            self.assertAlmostEqual(result['prefixes'][name][-1]['total'], direct[name].total, places=14)

    def test_spatial_cell_counts_are_not_silently_truncated(self):
        with self.assertRaises((TypeError, ValueError)):
            run_spatial_refinement(self.problem, self.candidates, x_steps=(21.5, 31))


if __name__ == '__main__':
    unittest.main()
