"""Pooled fit algebra, negative control, no holdout and strict replay budget."""

from dataclasses import replace
import unittest

import numpy as np

from src.method.bm_1_response import (CurveTrainingRecord, curve_pair_from_logits,
    curve_pair_logits, pooled_curve_log_likelihood, fit_shared_curve_pair)
from src.params.bm_1 import BM1ObservationSeries
from src.params.bm_1_response import BM1PiecewiseResponse


def constant(t):
    return np.full(np.asarray(t).shape, .5)


def initial(edges):
    return np.diff(edges)  # Same physical uniform prior at every resolution.


class SharedCurveFitTests(unittest.TestCase):
    def record(self, name, values):
        observations = BM1ObservationSeries([.25, .75], values, 'day')
        linear = BM1PiecewiseResponse.linear()
        return CurveTrainingRecord(name, 'TRAIN', constant, (0., 1.),
            (constant, linear, linear, 1., 2., 0.), 0., .1, initial, observations)

    def test_pooled_sequential_gaussian_score_matches_independent_closed_form(self):
        records = (self.record('a', [.5, .6]), self.record('b', [.4, .55]))
        f, g = curve_pair_from_logits([.8, -.4], 2)
        result = pooled_curve_log_likelihood(records, f, g, cells=11, max_step=.25)
        ys = np.array([.5,.6,.4,.55])
        expected = np.sum(-.5*((ys-.5)/.1)**2 - np.log(.1) - .5*np.log(2*np.pi))
        self.assertAlmostEqual(result['log_likelihood'], expected, places=12)
        self.assertEqual([r['record_id'] for r in result['records']], ['a','b'])
        linear = BM1PiecewiseResponse.linear()
        reference = pooled_curve_log_likelihood(records, linear, linear, cells=11, max_step=.25)
        # Xi=0 is an explicit non-identification control, irrespective of optimizer success.
        self.assertEqual(result['log_likelihood'], reference['log_likelihood'])

    def test_two_to_four_piece_start_preserves_both_curves(self):
        f, g = curve_pair_from_logits([.8,-.4], 2)
        lifted = curve_pair_logits(f,g,pieces=4)
        fine_f, fine_g = curve_pair_from_logits(lifted,4)
        x = np.linspace(0.,1.,101)
        np.testing.assert_allclose(fine_f(x),f(x),atol=4e-16,rtol=0.)
        np.testing.assert_allclose(fine_g(x),g(x),atol=4e-16,rtol=0.)

    def test_budget_counts_distinct_whole_record_replays_without_state_reuse(self):
        calls = []
        def counted(edges):
            calls.append(edges.size)
            return initial(edges)
        records = tuple(replace(self.record(name,[.5,.6]),initial_mass_builder=counted) for name in ('a','b'))
        result = fit_shared_curve_pair(records,pieces=4,starts=np.zeros((1,6)),
            regularization=0.,max_evaluations=3,cells=11,max_step=.25)
        self.assertTrue(result['budget_exhausted'])
        self.assertEqual(result['distinct_evaluations'],3)
        self.assertEqual(len(calls),6)
        self.assertEqual(len(result['trace']),3)
        for row in result['trace']:
            self.assertAlmostEqual(row['log_likelihood'],sum(row['record_log_likelihoods'].values()))

    def test_regularized_negative_control_selects_nested_linear_reference(self):
        records = (self.record('a',[.5,.6]), self.record('b',[.4,.55]))
        result = fit_shared_curve_pair(records,pieces=2,starts=[[0.,0.],[.8,-.4]],
            regularization=1.,max_evaluations=100,cells=11,max_step=.25)
        self.assertFalse(result['budget_exhausted'])
        self.assertTrue(all(r['success'] for r in result['optimizer_results']))
        np.testing.assert_array_equal(result['f'].values,[1.,0.,-1.])
        np.testing.assert_array_equal(result['g'].values,[1.,0.,-1.])

    def test_holdout_duplicates_and_time_units_are_rejected(self):
        record = self.record('a',[.5,.6])
        for role in ('MODEL_SELECTION','FINAL_HOLDOUT'):
            with self.assertRaises(ValueError): replace(record,role=role)
        curve = BM1PiecewiseResponse.linear()
        for records in ((record,record),(record,replace(record,record_id='b',
                observations=BM1ObservationSeries([.25,.75],[.5,.6],'hour')))):
            with self.assertRaises(ValueError): pooled_curve_log_likelihood(records,curve,curve)

    def test_missing_observation_is_not_imputed_or_scored(self):
        observations = BM1ObservationSeries([.25,.75],[.5,np.nan],'day',observed=[True,False])
        record = replace(self.record('a',[.5,.6]),observations=observations)
        curve = BM1PiecewiseResponse.linear()
        result = pooled_curve_log_likelihood([record],curve,curve,cells=11,max_step=.25)
        expected = -np.log(.1)-.5*np.log(2*np.pi)
        self.assertAlmostEqual(result['log_likelihood'],expected,places=12)
        self.assertTrue(np.isnan(result['records'][0]['log_evidence'][1]))


if __name__ == '__main__':
    unittest.main()
