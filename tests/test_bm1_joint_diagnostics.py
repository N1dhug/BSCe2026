"""Small mathematical checks for posthoc G2a grid diagnostics."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad

from src.validation.bm_1 import joint_diagnostics as diagnosis
from src.exe.run_model.bm_1 import estimate


class RectangularHatTests(unittest.TestCase):
    def test_constant_scores_preserve_uniform_prior_and_evidence(self):
        scores = np.tile(np.array([[.2], [-.3], [.7]]), (1, 81))
        law = diagnosis.rectangular_hat_law(scores, (3, 9))
        np.testing.assert_allclose(law['mass'], 1, atol=1e-14)
        np.testing.assert_allclose(law['density'], 1, atol=1e-14)
        np.testing.assert_allclose(law['mean'], .5, atol=1e-14)
        np.testing.assert_allclose(law['sd'], 1/np.sqrt(12), atol=1e-14)
        np.testing.assert_allclose(law['evidence'], [.2, -.1, .6], atol=1e-14)

    def test_rectangular_law_preserves_axis_order(self):
        u = np.linspace(0, 1, 9)
        likelihood = (1+u[:,None])*(1+2*u[None,:])
        law = diagnosis.rectangular_hat_law(np.log(likelihood).reshape(1,-1), (3,9))
        expected = (1+law['unit'][0][:,None])*(1+2*law['unit'][1][None,:])/3
        np.testing.assert_allclose(law['density'][0], expected, atol=1e-14)
        np.testing.assert_allclose(law['mean'][0], [5/9,7/12], atol=1e-14)

    def test_nested_bilinear_density_has_zero_refinement_tv(self):
        u = np.linspace(0,1,9)
        scores = np.log((1+u[:,None])*(1+u[None,:])).reshape(1,-1)
        comparison = diagnosis.compare_rectangular_hats(
            diagnosis.rectangular_hat_law(scores,(3,5)), diagnosis.rectangular_hat_law(scores,(9,9)))
        self.assertLess(comparison['maximum_errors']['parameter_tv'], 1e-14)
        self.assertTrue(comparison['parameter_only_thresholds_pass'])

    def test_tv_matches_independent_one_dimensional_integral(self):
        u = np.linspace(0,1,9)
        scores = np.tile((-20*(u-.3)**2)[:,None],(1,9)).reshape(1,-1)
        coarse = diagnosis.rectangular_hat_law(scores,(3,5))
        fine = diagnosis.rectangular_hat_law(scores,(9,9))
        def difference(x):
            return abs(np.interp(x,coarse['unit'][0],coarse['density'][0,:,0])
                       -np.interp(x,fine['unit'][0],fine['density'][0,:,0]))
        expected = .5*quad(difference,0,1,points=u[1:-1],epsabs=1e-11,limit=200)[0]
        got = diagnosis.compare_rectangular_hats(coarse,fine)['maximum_errors']['parameter_tv']
        self.assertAlmostEqual(got, expected, places=9)

    def test_invalid_score_or_grid_contract_is_rejected(self):
        for scores, counts in ((np.ones((1,80)),(3,9)), (np.ones((1,81)),(4,9)),
                               (np.full((1,81),np.nan),(3,9)), (np.ones((1,81)),(3,))):
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    diagnosis.rectangular_hat_law(scores,counts)

    def test_state_comparison_remaps_every_prefix_conservatively(self):
        left = {phase: np.array([[.5,.5],[.25,.75]]) for phase in ('predicted','posterior')}
        right = {phase: np.array([[.25,.25,.25,.25],[.125,.125,.375,.375]])
                 for phase in ('predicted','posterior')}
        result = diagnosis._state_difference(left,right)
        for values in result.values():
            np.testing.assert_allclose(values,0,atol=1e-15)


class DiagnosisCommandTests(unittest.TestCase):
    def test_small_panel_replays_all_levels_and_retains_negative_input(self):
        pair = ['rho','Xi']
        report = dict(rows=[dict(coordinates=pair, parameter_nodes_per_axis=65,
            fv_label=label, settings=dict(x_steps=cells,max_step=1/1536,likelihood_quadrature_order=3),
            conditional_log_evidence=np.zeros((168,1)).tolist())
            for label,cells in (('middle',3),('spatial_fine',5))], passes=False)
        panel = [dict(coordinates=pair,sequence_id=1358,selected_indices=[0],hypotheses=[[.1,.5]])]
        seen = []
        def replay(task):
            cells = task[2]['settings']['x_steps']
            seen.append(cells)
            return 0,0,dict(scores=np.zeros(168),predicted=np.full((168,cells),1/cells),
                posterior=np.full((168,cells),1/cells),elapsed_seconds=.1,actual_max_step=1/1536,
                maximum_abs_mass_error=0.,minimum_cell_mass=1/cells)
        with patch.object(diagnosis.grid,'_g2a_truth_record',return_value=(object(),
                type('Sequence',(),{'observations':object()})(),dict(passes=True))), \
             patch.object(diagnosis.fv,'_candidate_replay',side_effect=replay):
            result = diagnosis.run_spatial_panel(report,panel)
        self.assertEqual(seen,[3,5,9,17])
        self.assertEqual(result['conditional_runs'],4)
        self.assertFalse(report['passes'])
        self.assertFalse(result['stored_conditional_states'])
        for row in result['refinement_comparisons']:
            self.assertFalse(row['certifies_full_bank'])
            for maximum in row['maximum_errors'].values():
                self.assertLess(maximum,1e-14)

    def test_cli_requires_explicit_input_output(self):
        args = estimate.build_parser().parse_args(['joint-bank-grid-diagnose',
            '--g2a-report','negative.json.gz','--output','diagnosis.json','--blas-threads','1'])
        self.assertIs(args.handler, estimate.run_joint_grid_diagnosis)
        self.assertEqual(args.blas_threads,1)

    def test_dirty_source_is_rejected_before_reading_or_replaying(self):
        args = estimate.build_parser().parse_args(['joint-bank-grid-diagnose',
            '--g2a-report','negative.json.gz','--output','unused_diagnosis.json'])
        with patch.object(estimate,'_source_snapshot',return_value=dict(working_tree_dirty=True)), \
             patch.object(estimate,'_read_bank_dependency') as read:
            with self.assertRaisesRegex(ValueError,'clean source'):
                estimate.run_joint_grid_diagnosis(args)
            read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
