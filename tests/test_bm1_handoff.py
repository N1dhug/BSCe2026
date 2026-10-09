"""Independent probability, chronology and F3 integration checks."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from src.control.bm_1 import BM1ControlNumerics, BM1Parameters
from src.control.bm_1_handoff import BM1ForecastOrigin, forecast_fv
from src.method.bm_1_estimation import BM1FVFilterStep, BM1FVRhoBankStep
from src.model.bm_1 import bm_1
from src.solver.finite_volume import finite_volume_1d_cached
from src.tools.input_generator.input_builder import input_builder


def response(value):
    return 1. - 2. * np.asarray(value)


def baseline(times):
    return np.full(np.asarray(times).shape, .5)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.parameters = BM1Parameters(response, response, 1., 2., 1., .5)
        self.edges = np.linspace(0., 1., 32)
        self.mass = np.arange(1., 32.)
        self.mass /= self.mass.sum()
        self.origin = BM1ForecastOrigin(1., 'day', self.parameters, .03,
                                       self.edges, self.mass)

    def step(self, mass):
        return BM1FVFilterStep(1., self.edges, (self.edges[1:] + self.edges[:-1])/2,
                              mass, mass, True, 0., {})

    def test_map_handoff_selects_own_conditional_law_and_rho_reciprocal(self):
        other = self.mass[::-1].copy()
        step = BM1FVRhoBankStep([.5, 2.], np.log([.2, .8]),
                                (self.step(self.mass), self.step(other)))
        origin = BM1ForecastOrigin.from_rho_bank_step(
            step, self.parameters, .03, time_unit='day')
        self.assertEqual(origin.selected_node, 1)
        self.assertEqual(origin.parameters.C, .5)
        self.assertEqual(origin.parameters.sigma_x, .5)
        self.assertEqual(origin.sigma_y, .03)
        self.assertIs(origin.parameters.state_response, response)
        np.testing.assert_array_equal(origin.posterior_mass, other)
        self.assertGreater(np.sum(np.abs(origin.posterior_mass - step.posterior_mass)), 0.)

    def test_map_ties_use_first_node_without_averaging_parameters(self):
        step = BM1FVRhoBankStep([.5, 2.], np.log([.5, .5]),
                                (self.step(self.mass), self.step(self.mass[::-1])))
        origin = BM1ForecastOrigin.from_rho_bank_step(
            step, self.parameters, .03, time_unit='hour')
        self.assertEqual(origin.selected_node, 0)
        self.assertEqual(origin.parameters.C, 2.)

    def test_owned_origin_and_exact_independent_overlap_handoff(self):
        self.mass[:] = 1. / self.mass.size
        self.assertFalse(self.origin.posterior_mass.flags.writeable)
        target_edges = np.linspace(0., 1., 18)
        actual = self.origin.initial_mass(target_edges)
        expected = []
        for left, right in zip(target_edges[:-1], target_edges[1:]):
            overlap = np.maximum(0., np.minimum(right, self.edges[1:])
                                 - np.maximum(left, self.edges[:-1]))
            expected.append(np.sum(overlap / np.diff(self.edges) * self.origin.posterior_mass))
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0.)
        self.assertAlmostEqual(actual.sum(), 1., places=14)
        problem = self.origin.control_problem(baseline=baseline, target=baseline,
                    horizon=(1., 2.), price_edges=[1., 1.5, 2.])
        evaluator = problem.build_evaluator(BM1ControlNumerics(x_steps=17,
                    objective_max_step=.25, solver_max_step=.125), fast=False)
        np.testing.assert_array_equal(evaluator.initial_mass, actual)

    def test_one_propagation_matches_uncached_reference_at_every_horizon(self):
        times = np.array([1.125, 1.5, 2.])
        edges, prices = [1., 1.25, 2.], [.8, .2]
        original = finite_volume_1d_cached.solve
        calls = []
        def counted(solver, *args, **kwargs):
            calls.append(args[0])
            return original(solver, *args, **kwargs)
        with patch.object(finite_volume_1d_cached, 'solve', counted):
            actual = forecast_fv(self.origin, baseline=baseline, price_edges=edges,
                                 prices=prices, times=times, max_step=1/96)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0], np.r_[1., times])
        model = bm_1(baseline, baseline, (1., 2.), 2)
        template = self.parameters.parameter_template(edges)
        template[0] = input_builder.price_intervals((np.asarray(edges), np.asarray(prices)))
        reference = model.fokker_planc_reference_solver(
            self.origin.posterior_mass, template, .5, t_eval=np.r_[1., times],
            breakpoints=[1.25], max_step=1/96)
        np.testing.assert_allclose(actual['cell_mass'], reference[3][1:], atol=1e-12, rtol=0.)
        np.testing.assert_allclose(actual['demand_mean'], reference[8][1:], atol=1e-12, rtol=0.)
        np.testing.assert_allclose(actual['observation_variance'] - actual['demand_variance'],
                                   .03**2, atol=5e-18, rtol=0.)

    def test_zero_flexibility_has_constant_demand_but_no_observation_assimilation(self):
        origin = replace(self.origin, parameters=replace(self.parameters, Xi=0., sigma_x=0.))
        with patch('src.method.bm_1_estimation.fv_filter_step', side_effect=AssertionError('assimilation')):
            actual = forecast_fv(origin, baseline=baseline, price_edges=[1., 8.],
                                 prices=[.5], times=[2., 8.], max_step=.25)
        np.testing.assert_allclose(actual['cell_mass'], np.tile(origin.posterior_mass, (2, 1)),
                                   atol=1e-15, rtol=0.)
        np.testing.assert_allclose(actual['demand_mean'], .5, atol=1e-15, rtol=0.)
        np.testing.assert_allclose(actual['demand_variance'], 0., atol=1e-28, rtol=0.)

    def test_origin_plan_and_time_contracts_fail_before_any_solve(self):
        with self.assertRaises(ValueError):
            replace(self.origin, time_unit='minute')
        with self.assertRaises(ValueError):
            self.origin.control_problem(baseline=baseline, target=baseline,
                                         horizon=(1.1, 2.), price_edges=[1.1, 2.])
        for times in ([1.], [.9, 2.], [2., 1.5], [2., 3.]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                forecast_fv(self.origin, baseline=baseline, price_edges=[1., 2.],
                             prices=[.5], times=times, max_step=.1)
        with self.assertRaises(ValueError):
            self.origin.initial_mass([.1, .5, 1.])

    def test_single_horizon_and_different_forecast_grid(self):
        result = forecast_fv(self.origin, baseline=baseline, price_edges=[1., 2.],
                             prices=[.5], times=[2.], max_step=.125, x_steps=17)
        self.assertEqual(result['cell_mass'].shape, (1, 17))
        self.assertAlmostEqual(result['cell_mass'].sum(), 1., places=12)

    def test_short_forecast_ignores_later_input_breakpoints(self):
        short = forecast_fv(self.origin, baseline=baseline, price_edges=[1., 2.],
                             prices=[.4], times=[1.5, 2.], max_step=.125)
        longer_plan = forecast_fv(self.origin, baseline=baseline,
                                  price_edges=[1., 2., 3., 8.], prices=[.4, .6, .8],
                                  times=[1.5, 2.], max_step=.125, breakpoints=[4.])
        np.testing.assert_array_equal(longer_plan['cell_mass'], short['cell_mass'])
        # State is continuous; demand at the jump uses the declared right price.
        self.assertEqual(longer_plan['demand_mean'][0], short['demand_mean'][0])
        self.assertNotEqual(longer_plan['demand_mean'][1], short['demand_mean'][1])


if __name__ == '__main__':
    unittest.main()
