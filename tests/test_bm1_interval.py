"""Independent interval law, causality, positive mass and shared H replay."""

from dataclasses import replace
import unittest

import numpy as np

from src.method.bm_1_estimation import run_fv_filter
from src.method.bm_1_interval_likelihood import run_interval_filter
from src.method.bm_1_response import (CurveTrainingRecord, replay_curve_record,
    pooled_curve_log_likelihood, fit_shared_curve_pair)
from src.model.bm_1 import bm_1
from src.params.bm_1 import BM1ObservationSeries
from src.method.bm_1_observation import gaussian_log_likelihood, bayes_update, mass_to_log
from src.params.bm_1 import BM1IntervalObservationSeries
from src.params.bm_1_response import BM1PiecewiseResponse
from src.solver.finite_volume import finite_volume_1d_fitted
from src.solver.finite_volume_additive import propagate_additive
from src.validation.bm_1.interval import finite_state_gaussian_reference


def constant(value):
    return lambda t, x=None: np.full(np.asarray(t if x is None else x).shape, value)


class IntervalTests(unittest.TestCase):
    def model(self, *, xi=.8):
        model = bm_1(constant(.5), constant(.5), (0., 1.), 2)
        model.fokker_planc_reference = finite_volume_1d_fitted(vectorized_inputs=True)
        linear = BM1PiecewiseResponse.linear()
        params = (constant(.3), linear, linear, .2, 3., xi)
        return model, params

    def test_interval_contract_owns_values_and_excludes_partial_future_hours(self):
        ys = np.array([.4, .7])
        obs = BM1IntervalObservationSeries([0., .5], [.5, 1.], ys, 'hour')
        ys[0] = 99.
        self.assertEqual(obs.values[0], .4)
        self.assertFalse(obs.values.flags.writeable)
        np.testing.assert_array_equal(obs.prefix(.75).ends, [.5])
        with self.assertRaises(ValueError): obs.prefix(.49)
        for starts, ends in (([0., .4], [.5, 1.]), ([0.], [0.]), ([.5], [.4])):
            with self.assertRaises(ValueError):
                BM1IntervalObservationSeries(starts, ends, np.zeros(len(starts)), 'hour')
        with self.assertRaises(NotImplementedError):
            BM1ObservationSeries([1.], [.5], 'hour', observation_kind='interval_average')

    def test_state_independent_reward_has_exact_gaussian_evidence_without_deposition_noise(self):
        model, params = self.model(xi=0.)
        obs = BM1IntervalObservationSeries([0., .5], [.5, 1.], [.45, .6], 'day')
        initial = np.full(17, 1/17)
        result = run_interval_filter(model, params, .4, .1, initial, 0., obs,
                                    max_step=1/32, accumulator_spacing=1/11)
        expected = gaussian_log_likelihood(.45, [.5], .1)[0]+gaussian_log_likelihood(.6, [.5], .1)[0]
        self.assertAlmostEqual(result.total_log_likelihood, expected, places=12)
        for d in result.prediction_diagnostics:
            self.assertEqual(d['accumulator_nodes'], 1)
            self.assertEqual(d['accumulator_deposition_variance_bound'], 0.)
        reference = model.fokker_planc_reference_solver(initial, params, .4,
                                                       t_eval=[0., .5, 1.], max_step=1/32)
        np.testing.assert_allclose(result.posterior_mass, reference[3][1:], atol=2e-15, rtol=0.)

    def test_joint_likelihood_converges_to_independent_finite_state_fourier_reference(self):
        solver = finite_volume_1d_fitted()
        initial = np.array([.2, .5, .3])
        drift = constant(.03)
        diffusion = constant(.3)
        reward = lambda t, x: np.array([.15, .5, .85])
        lo, diag, up, _ = solver.generator_tridiagonal(1/3, [.03, .03], [.3]*3)
        Q = np.diag(diag)+np.diag(lo, -1)+np.diag(up, 1)
        reference = finite_state_gaussian_reference(Q, reward(0., None), initial, 1., .62, .12, order=96)
        refined = finite_state_gaussian_reference(Q, reward(0., None), initial, 1., .62, .12, order=160)
        self.assertAlmostEqual(reference['log_evidence'], refined['log_evidence'], places=12)
        errors = []
        posterior_errors = []
        for resolution in (64, 512):
            joint = propagate_additive(solver, [0., 1.], initial, drift, diffusion, reward,
                max_step=1/resolution, accumulator_spacing=1/resolution, max_accumulator_nodes=2048)
            update = bayes_update(mass_to_log(joint['joint_mass'].ravel()),
                np.broadcast_to(gaussian_log_likelihood(.62, joint['accumulator'], .12), joint['joint_mass'].shape).ravel())
            posterior = update.mass.reshape(joint['joint_mass'].shape).sum(axis=1)
            errors.append(abs(update.log_evidence-reference['log_evidence']))
            posterior_errors.append(float(np.max(np.abs(posterior-reference['posterior_mass']))))
            np.testing.assert_allclose(joint['joint_mass'].sum(axis=1),
                solver.solve([0., 1.], initial, drift, diffusion, max_step=1/resolution)['cell_mass'][-1], atol=2e-14)
        self.assertLess(errors[1], errors[0]/2.)
        self.assertLess(errors[1], .008)
        self.assertLess(posterior_errors[1], posterior_errors[0]/2.)
        self.assertLess(posterior_errors[1], .008)

    def test_breakpoint_inside_hour_changes_interval_mean_not_endpoint_value(self):
        model, params = self.model(xi=0.)
        model.B = lambda t: np.where(np.asarray(t) < .5, .2, .8)
        obs = BM1IntervalObservationSeries([0.], [1.], [.5], 'day')
        initial = np.full(11, 1/11)
        interval = run_interval_filter(model, params, 0., .05, initial, 0., obs,
                                       breakpoints=[.5], max_step=.25)
        point = run_fv_filter(model, params, 0., .05, initial, 0.,
            BM1ObservationSeries([1.], [.5], 'day'), breakpoints=[.5], max_step=.25)
        self.assertAlmostEqual(interval.prediction_diagnostics[0]['predicted_interval_mean'], .5)
        self.assertGreater(interval.total_log_likelihood-point.total_log_likelihood, 17.)

    def test_gap_missing_hour_full_and_likelihood_only_keep_same_causal_posterior(self):
        model, params = self.model()
        obs = BM1IntervalObservationSeries([.1, .2, .4], [.2, .3, .5], [.55, np.nan, .65],
                                           'day', [True, False, True])
        initial = np.full(21, 1/21)
        kw = dict(max_step=.01, accumulator_spacing=1/128)
        full = run_interval_filter(model, params, .3, .08, initial, 0., obs, **kw)
        only = run_interval_filter(model, params, .3, .08, initial, 0., obs, keep_history=False, **kw)
        np.testing.assert_array_equal(full.log_evidence, only.log_evidence)
        np.testing.assert_array_equal(full.posterior_mass[-1], only.final_step.posterior_mass)
        self.assertTrue(np.isnan(full.log_evidence[1]))
        np.testing.assert_array_equal(full.posterior_mass[1], full.predicted_mass[1])
        changed = replace(obs, values=[.55, np.nan, 100.])
        future = run_interval_filter(model, params, .3, .08, initial, 0., changed, **kw)
        np.testing.assert_array_equal(full.posterior_mass[:2], future.posterior_mass[:2])
        self.assertEqual(only.n_steps, len(full.integration_t)-1)
        self.assertLess(full.maximum_abs_mass_error, 1e-10)

    def test_node_budget_aborts_instead_of_truncating_probability(self):
        solver = finite_volume_1d_fitted()
        with self.assertRaisesRegex(ValueError, 'node budget'):
            propagate_additive(solver, [0., 1.], np.full(5, .2), constant(0.), constant(.2),
                lambda t, x: x, max_step=.5, accumulator_spacing=.01, max_accumulator_nodes=2)

    def test_shared_curve_fit_replays_all_intervals_with_one_pair_and_bounded_budget(self):
        model, params = self.model(xi=0.)
        obs = BM1IntervalObservationSeries([0., .5], [.5, 1.], [.5, .6], 'day')
        record = CurveTrainingRecord('interval', 'TRAIN', model.B, (0., 1.), params, .2, .1, np.diff, obs)
        numerics = dict(cells=11, max_step=.1, accumulator_spacing=1/32, max_accumulator_nodes=128)
        linear = BM1PiecewiseResponse.linear()
        expected = replay_curve_record(record, linear, linear, **numerics).total_log_likelihood
        fit = fit_shared_curve_pair([record], pieces=2, starts=[[0., 0.]], regularization=0.,
            max_evaluations=2, **numerics)
        self.assertEqual(fit['distinct_evaluations'], 2)
        self.assertTrue(fit['budget_exhausted'])
        self.assertAlmostEqual(fit['log_likelihood'], expected, places=12)
        self.assertEqual(fit['replay']['records'][0]['final_step'].time, 1.)
        with self.assertRaisesRegex(ValueError, 'explicit accumulator'):
            pooled_curve_log_likelihood([record], linear, linear, cells=11)


if __name__ == '__main__':
    unittest.main()
