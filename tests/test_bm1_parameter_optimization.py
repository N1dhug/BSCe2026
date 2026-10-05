"""Pilot contracts: prediction timing, existing likelihood and real work budgets."""
import unittest
from dataclasses import replace
from time import perf_counter
from unittest.mock import patch

import numpy as np

from src.params.bm_1 import BM1ObservationSeries
from src.validation.bm_1 import estimation as existing
from src.validation.bm_1 import parameter_optimization as pilot


class PhysicalOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.case = existing.build_f4_b_case(observation_times=[.25,.75])
        self.candidate = existing.f4_e_truth_candidate(self.case)
        self.observations = BM1ObservationSeries([.25,.75],[.45,.55],'day')
        self.settings = dict(cells=31,max_step=1/96,quadrature_order=3)

    def test_score_matches_existing_filter_and_current_measurement_does_not_predict_itself(self):
        result = pilot.replay(self.case,self.observations,self.candidate,**self.settings)
        reference = existing.evaluate_f4_e_candidate(self.case,self.observations,self.candidate,
            x_steps=31,max_step=1/96,likelihood_quadrature_order=3)
        self.assertAlmostEqual(sum(row['log_score'] for row in result['rows']),
                               reference['total_log_likelihood'],places=11)
        changed = BM1ObservationSeries([.25,.75],[.75,.55],'day')
        other = pilot.replay(self.case,changed,self.candidate,**self.settings)
        self.assertEqual(result['rows'][0]['mean'],other['rows'][0]['mean'])
        self.assertNotAlmostEqual(result['rows'][1]['mean'],other['rows'][1]['mean'],places=6)

    def test_no_response_exact_gaussian_and_missing_mask(self):
        case = replace(self.case,Xi=0.)
        candidate = existing.f4_e_truth_candidate(case)
        observations = BM1ObservationSeries([.25,.75],[.45,np.nan],'day',observed=[True,False])
        result = pilot.pooled(case,[(1,observations)],candidate,**self.settings)
        self.assertEqual(len(result['residuals']),1)
        self.assertAlmostEqual(result['residuals'][0],-.05,places=14)
        expected = .5*(.05/.03)**2 + np.log(.03*np.sqrt(2*np.pi))
        self.assertAlmostEqual(result['negative_log_likelihood'],expected,places=12)

    def test_coordinate_mapping_preserves_fixed_parameters_and_grid_endpoints(self):
        candidate = pilot.physical_candidate([0.,1.],('rho','Xi'),self.candidate)
        self.assertEqual(candidate.rho,.5)
        self.assertEqual(candidate.Xi,1.)
        self.assertEqual(candidate.k,self.candidate.k)
        for name in ('sigma_x','sigma_y'):
            for z,value in ((0.,existing.F4_E_PARAMETER_GRIDS[name][0]),
                            (1.,existing.F4_E_PARAMETER_GRIDS[name][-1])):
                actual = pilot.physical_candidate([z],(name,),self.candidate)
                self.assertAlmostEqual(getattr(actual,name),value,places=14)

    def test_both_starts_get_reserved_budget_including_differences(self):
        def fake_pool(case,records,candidate,**kwargs):
            return dict(negative_log_likelihood=(candidate.rho-1.)**2,
                        residuals=[candidate.rho-1.])
        for method in ('mle','ls'):
            evaluator = pilot.Evaluator(self.case,[],('rho',),4,deadline=perf_counter()+10)
            with patch.object(pilot,'pooled',side_effect=fake_pool) as mock:
                result = pilot.optimize(evaluator,method)
                self.assertEqual(len(result),2)
                self.assertEqual([row['new_candidates'] for row in result],[2,2])
                self.assertEqual(mock.call_count,4)
                evaluator([.25])
                self.assertEqual(mock.call_count,4)
                with self.assertRaises(pilot.BudgetStop): evaluator([.123])


if __name__ == '__main__':
    unittest.main()
