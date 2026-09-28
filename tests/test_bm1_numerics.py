"""BM1 regression tests grouped by responsibility."""

import numpy as np
import unittest
from src.control.bm_1 import BM1ControlNumerics
from src.validation.bm_1.control import build_f3_a_problem
from src.validation.bm_1.numerics import (_aggregate_brownian_to_grid, _decision_gap_refinement, _nested_em_time_grids,
    _prefix_statistics, fixed_candidates, run_coupled_mc_time_refinement, run_fv_time_refinement,
    run_quadrature_refinement, run_spatial_refinement, run_fixed_policy_validation)

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

    def test_small_coupled_mc_time_study_has_exact_zero_finest_difference(self):
        rows = run_coupled_mc_time_refinement(self.problem, self.one_candidate(), n_paths=64,
            solver_max_steps=(0.125, 0.0625), objective_max_step=0.25, initial_seed=5, brownian_seed=6)
        self.assertEqual(len(rows), 2)
        finest = rows[-1]
        self.assertEqual(finest['mean_path_cost_difference_to_finest'], 0.0)
        self.assertEqual(finest['rms_terminal_state_difference_to_finest'], 0.0)
        self.assertGreater(rows[1]['integration_steps'], rows[0]['integration_steps'])
        self.assertTrue(all((np.isfinite(row['total']) for row in rows)))


if __name__ == '__main__':
    unittest.main()
