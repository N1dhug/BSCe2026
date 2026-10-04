"""Resumed optimizer reuses scores, never a different candidate's posterior."""

from copy import deepcopy
import unittest
import numpy as np

from src.method.bm_1_response import CurveTrainingRecord, fit_shared_curve_pair
from src.params.bm_1 import BM1ObservationSeries
from src.params.bm_1_response import BM1PiecewiseResponse


class CurveResumeTests(unittest.TestCase):
    def setUp(self):
        const = lambda t: np.full(np.asarray(t).shape, .5)
        curve = BM1PiecewiseResponse.linear()
        self.records = [CurveTrainingRecord('a', 'TRAIN', const, (0., 1.),
            (const, curve, curve, 1., 2., 0.), 0., .1, np.diff,
            BM1ObservationSeries([.25, .75], [.5, .6], 'day'))]
        self.settings = dict(pieces=2, starts=[[.8, -.4], [0., 0.]], regularization=1.,
                             max_evaluations=100, cells=11, max_step=.25)

    def test_pause_resume_matches_uninterrupted_trace_and_posterior(self):
        full = fit_shared_curve_pair(self.records, **self.settings)
        checkpoints = []
        part = fit_shared_curve_pair(self.records, **self.settings, max_new_evaluations=5,
                                     checkpoint=lambda trace: checkpoints.append(trace))
        self.assertTrue(part['paused'])
        self.assertFalse(part['budget_exhausted'])
        self.assertEqual(part['new_evaluations'], 5)
        resumed = fit_shared_curve_pair(self.records, **self.settings, resume_trace=checkpoints[-1])
        self.assertEqual(resumed['trace'], full['trace'])
        self.assertEqual(resumed['new_evaluations'], full['distinct_evaluations']-5)
        np.testing.assert_array_equal(resumed['replay']['records'][0]['final_step'].posterior_mass,
                                      full['replay']['records'][0]['final_step'].posterior_mass)
        self.assertFalse(resumed['paused'])
        # Returning a cached best posterior requires a fresh replay, not state reuse.
        cached = fit_shared_curve_pair(self.records, **self.settings, resume_trace=full['trace'])
        self.assertEqual(cached['new_evaluations'], 0)
        self.assertTrue(cached['restored_best_replay'])

    def test_total_budget_remains_binding_after_resume(self):
        settings = dict(self.settings, max_evaluations=8)
        part = fit_shared_curve_pair(self.records, **settings, max_new_evaluations=3)
        resumed = fit_shared_curve_pair(self.records, **settings, resume_trace=part['trace'])
        self.assertTrue(resumed['budget_exhausted'])
        self.assertFalse(resumed['paused'])
        self.assertEqual(resumed['distinct_evaluations'], 8)
        self.assertEqual(resumed['new_evaluations'], 5)
        self.assertEqual(resumed['starts_completed'], 0)

    def test_changed_scores_penalty_records_duplicates_and_bounds_rejected(self):
        trace = fit_shared_curve_pair(self.records, **self.settings, max_new_evaluations=5)['trace']
        variants = []
        for key in ('objective', 'penalty', 'log_likelihood', 'maximum_abs_mass_error'):
            bad = deepcopy(trace); bad[0][key] += 1.; variants.append(bad)
        bad = deepcopy(trace); bad[0]['logits'][0] = 5.; variants.append(bad)
        bad = deepcopy(trace); bad[0]['record_log_likelihoods'] = {'other': 1.}; variants.append(bad)
        variants.append(trace+trace[:1])
        for bad in variants:
            with self.assertRaises(ValueError):
                fit_shared_curve_pair(self.records, **self.settings, resume_trace=bad)

    def test_checkpoint_failure_is_not_swallowed_as_optimizer_failure(self):
        def fail(trace): raise OSError('checkpoint disk error')
        with self.assertRaisesRegex(OSError, 'checkpoint disk error'):
            fit_shared_curve_pair(self.records, **self.settings, checkpoint=fail)


if __name__ == '__main__': unittest.main()
